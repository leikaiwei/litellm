import asyncio
import base64
import copy
import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from io import BytesIO
from time import time
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from PIL import Image
from pydantic import TypeAdapter
from pytest_socket import disable_socket, enable_socket

import litellm
from litellm import ModelResponse
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.prompt_templates.common_utils import encrypted_reasoning_signature
from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.llms.anthropic.chat import handler as chat_handler
from litellm.llms.anthropic.count_tokens import handler as count_handler
from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.anthropic.messages.stream_assembly import assemble_anthropic_sse_stream
from litellm.llms.compaction import CompactionProtocol
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.proxy.common_utils import config_sync_pubsub
from litellm.responses.litellm_completion_transformation.compaction_items import summary_only_compaction_content
from litellm.router_strategy.complexity_router.context_compaction import compaction_executor
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import CallTypes, Choices
from litellm.utils import custom_llm_setup
from local_guardrails import request_admission
from local_guardrails.request_admission import RequestAdmission

_HOOK: Final = RequestAdmission()
_MODEL: Final = "admission_test/model"


@pytest.fixture(autouse=True)
def no_network() -> Iterator[None]:
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()


def _deployment(limit: int, name: str = "candidate") -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": _MODEL},
        "model_info": {"max_input_tokens": limit, "id": name},
    }


def _messages(text: str = "Hello") -> list[AllMessageValues]:
    return [{"role": "user", "content": text}]


def _answer(response: object) -> str | None:
    assert isinstance(response, ModelResponse)
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices)
    return choice.message.content


async def _filter(
    messages: list[AllMessageValues] | None,
    limit: int = 10000,
    **data: object,
) -> list[dict[str, object]]:
    return await _HOOK.async_filter_deployments("group", [_deployment(limit)], messages, dict(data))


@pytest.mark.asyncio
async def test_context_boundary_filters_candidates_without_rewriting_request() -> None:
    messages: Final = _messages("ordinary context " * 30)
    count: Final = litellm.token_counter(  # pyright: ignore[reportUnknownMemberType]  # 上游计数器仍有未注解参数
        model=_MODEL, messages=messages
    )
    before: Final = copy.deepcopy(messages)
    larger: Final = _deployment(count + 1, "larger")
    accepted: Final = await _HOOK.async_filter_deployments("group", [_deployment(count, "equal"), larger], messages)
    assert accepted == [larger]
    assert messages == before
    with pytest.raises(litellm.ContextWindowExceededError, match="/compact"):
        await _filter(messages, limit=count)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("system", "tools"))
async def test_system_and_tools_are_included_in_limit(field: str) -> None:
    messages: Final = _messages()
    baseline: Final = litellm.token_counter(  # pyright: ignore[reportUnknownMemberType]  # 上游计数器仍有未注解参数
        model=_MODEL, messages=messages
    )
    addition: Final[object] = (
        "system instructions " * 100
        if field == "system"
        else [{"type": "function", "function": {"name": "run", "description": "tool documentation " * 100}}]
    )
    assert await _filter(messages, limit=baseline + 1)
    with pytest.raises(litellm.ContextWindowExceededError):
        await _filter(messages, limit=baseline + 1, **{field: addition})


@pytest.mark.asyncio
@pytest.mark.parametrize("anthropic", (False, True))
async def test_named_tool_choice_is_included_in_context_budget(anthropic: bool) -> None:
    messages: Final = _messages()
    baseline: Final = litellm.token_counter(  # pyright: ignore[reportUnknownMemberType]  # 上游计数器仍有未注解参数
        model=_MODEL, messages=messages
    )
    choice: Final = (
        {"type": "tool", "name": "read"} if anthropic else {"type": "function", "function": {"name": "read"}}
    )
    assert await _filter(messages, limit=baseline + 1)
    with pytest.raises(litellm.ContextWindowExceededError):
        await _filter(messages, limit=baseline + 1, tool_choice=choice)


@pytest.mark.asyncio
async def test_responses_instructions_and_function_history_are_counted() -> None:
    assert await _filter(None, input="hello")
    with pytest.raises(litellm.ContextWindowExceededError):
        await _filter(
            None,
            limit=100,
            input=[
                {"role": "user", "content": "hello"},
                {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "tool output " * 200},
            ],
            instructions="system instructions",
        )


_UNVERIFIABLE_RESPONSES: Final[tuple[dict[str, object], ...]] = (
    {"previous_response_id": "previous-id"},
    {"conversation": "conversation-id"},
    {"input": [{"type": "item_reference", "id": "message-id"}]},
    {"input": [{"type": "future_unknown", "private_value": "not-counted " * 100}]},
    {"input": [{"type": "future_unknown", "encrypted_content": "opaque"}]},
    {"input": [{"type": "reasoning", "summary": [], "encrypted_content": 42}]},
    {"input": [{"type": "compaction", "encrypted_content": ""}]},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", _UNVERIFIABLE_RESPONSES)
async def test_responses_hidden_history_is_not_silently_counted_as_zero(extra: dict[str, object]) -> None:
    with pytest.raises(litellm.BadRequestError, match=r"完整|新会话"):
        await _HOOK.async_filter_deployments("group", [_deployment(10000)], None, {"input": "hello", **extra})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("reasoning", "compaction"))
@pytest.mark.parametrize("readable_field", ("input", "summary", "instructions", "tools", "tool_output"))
async def test_encrypted_history_preserves_payload_and_checks_readable_context(kind: str, readable_field: str) -> None:
    opaque: Final[dict[str, object]] = {"type": kind, "summary": [], "encrypted_content": "opaque-history" * 10000}
    request: Final[dict[str, object]] = {"input": [opaque, {"role": "user", "content": "Continue"}]}
    before: Final = copy.deepcopy(request)
    assert await _filter(None, limit=100, **request)
    assert (
        await _HOOK.async_pre_call_deployment_hook(
            {"model": _MODEL, "model_info": {"max_input_tokens": 100}, **request}, CallTypes.aresponses
        )
        is None
    )
    assert request == before

    text: Final = "readable context " * 200
    addition: Final[dict[str, object]] = (
        {"input": [opaque, {"role": "user", "content": text}]}
        if readable_field == "input"
        else {
            "input": [
                {
                    **opaque,
                    **(
                        {"summary": [{"type": "summary_text", "text": text}]}
                        if kind == "reasoning"
                        else {"encrypted_content": summary_only_compaction_content(text)}
                    ),
                }
            ]
        }
        if readable_field == "summary"
        else {"instructions": text}
        if readable_field == "instructions"
        else {"tools": [{"type": "function", "name": "read", "description": text, "parameters": {"type": "object"}}]}
        if readable_field == "tools"
        else {"input": [opaque, {"type": "function_call_output", "call_id": "call_1", "output": text}]}
    )
    too_long: Final = {**request, **addition}
    with pytest.raises(litellm.ContextWindowExceededError):
        await _filter(None, limit=100, **too_long)
    with pytest.raises(litellm.ContextWindowExceededError):
        await _HOOK.async_pre_call_deployment_hook(
            {"model": _MODEL, "model_info": {"max_input_tokens": 100}, **too_long}, CallTypes.aresponses
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "responses"))
async def test_real_router_replays_encrypted_history_to_responses_provider(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    encrypted: Final = "opaque-upstream-history"
    item: Final[dict[str, object]] = {"type": "reasoning", "summary": [], "encrypted_content": encrypted}
    transport: Final = ResponsesHistoryTransport()
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))

    def get_client(**kwargs: object) -> AsyncHTTPHandler:
        return client

    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", get_client)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "encrypted-fixture",
                "litellm_params": {
                    "model": "openai/gpt-6.1-terra",
                    "api_key": "test-key",
                    "api_base": "https://test.invalid/v1",
                },
                "model_info": {"max_input_tokens": 100, "supported_endpoints": ["/v1/responses"]},
            }
        ],
        num_retries=0,
    )
    try:
        with patch.object(litellm, "callbacks", [RequestAdmission()]):
            if route == "responses":
                await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                    model="encrypted-fixture",
                    input=[item, {"role": "user", "content": "Continue"}],
                    max_output_tokens=10,
                )
            else:
                await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                    model="encrypted-fixture",
                    messages=[
                        {"role": "user", "content": "Hello"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "redacted_thinking", "data": encrypted_reasoning_signature(encrypted)}
                            ],
                        },
                        {"role": "user", "content": "Continue"},
                    ],
                    max_tokens=10,
                )
        assert len(transport.requests) == 1
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        assert item in TypeAdapter(list[dict[str, object]]).validate_python(sent["input"])
        assert transport.requests[0].url.path == "/v1/responses"
    finally:
        router.reset()
        await client.client.aclose()


class ResponsesHistoryTransport:
    def __init__(self) -> None:
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(
            200,
            json={
                "id": "resp_fixture",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6.1-terra",
                "output": [
                    {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Accepted", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 1, "total_tokens": 21},
            },
        )


@pytest.mark.asyncio
async def test_responses_cannot_use_extra_messages_to_hide_the_real_input() -> None:
    with pytest.raises(litellm.BadRequestError, match="input/messages"):
        await _HOOK.async_pre_call_deployment_hook(
            {
                "model": _MODEL,
                "model_info": {"max_input_tokens": 20},
                "input": "long history " * 1000,
                "messages": _messages(),
                "previous_response_id": "previous-id",
            },
            CallTypes.aresponses,
        )


@pytest.mark.asyncio
async def test_responses_readable_reasoning_and_ordinary_tool_arguments_are_preserved() -> None:
    assert await _filter(
        None,
        input=[
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Readable summary"}]},
            {
                "type": "function_call",
                "name": "inspect",
                "call_id": "call_1",
                "arguments": '{"encrypted_content":"ordinary function argument","type":"item_reference"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            {"role": "user", "content": "Continue"},
        ],
    )


def _png() -> str:
    with BytesIO() as output:
        Image.new("RGB", (2, 2), "red").save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ("chat", "messages", "responses"))
async def test_inline_image_validation_covers_three_protocols(api: str) -> None:
    def image(encoded: str) -> dict[str, object]:
        if api == "messages":
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}
        if api == "responses":
            return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

    def data(encoded: str) -> dict[str, object]:
        body: Final = [{"role": "user", "content": [image(encoded)]}]
        return {"input" if api == "responses" else "messages": body}

    valid: Final = data(_png())
    invalid: Final = data("AAAA")
    assert await _HOOK.async_filter_deployments("group", [_deployment(10000)], None, valid)
    with pytest.raises(litellm.BadRequestError, match="图片损坏") as error:
        await _HOOK.async_filter_deployments("group", [_deployment(10000)], None, invalid)
    assert "[0].content[0]" in str(error.value)


@pytest.mark.asyncio
async def test_nested_tool_result_image_is_checked_but_urls_are_never_fetched() -> None:
    invalid: Final = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
                }
            ],
        }
    ]
    with pytest.raises(litellm.BadRequestError, match=r"messages\[0\].content\[0\].content\[0\]"):
        await _filter(cast(list[AllMessageValues], invalid))  # cast-ok: tokenizer also accepts Anthropic blocks
    remote: Final = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://invalid.test/x.png"}}]}
    ]
    assert await _filter(cast(list[AllMessageValues], remote))  # cast-ok: literal uses OpenAI image content shape


@pytest.mark.asyncio
async def test_unknown_content_is_rejected_without_echoing_payload() -> None:
    messages: Final = [{"role": "user", "content": [{"type": "unknown-new-block", "secret": "private-marker"}]}]
    with pytest.raises(litellm.BadRequestError, match="无法可靠计算") as error:
        await _filter(cast(list[AllMessageValues], messages))  # cast-ok: deliberate unsupported input for regression
    assert "private-marker" not in str(error.value)


@pytest.mark.asyncio
async def test_remote_images_use_existing_upper_bound_without_fetching_or_rewriting() -> None:
    messages: Final[list[dict[str, object]]] = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://invalid.test/x.png", "detail": "high"}}],
        }
    ]
    original: Final = copy.deepcopy(messages)
    with pytest.raises(litellm.ContextWindowExceededError):
        await _filter(
            cast(list[AllMessageValues], messages), limit=high_detail_image_token_upper_bound()
        )  # cast-ok: OpenAI 图片消息
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_kind", ("bytes", "pixels"))
async def test_image_validation_respects_allocation_limits(monkeypatch: pytest.MonkeyPatch, limit_kind: str) -> None:
    encoded: Final = _png()
    if limit_kind == "bytes":
        monkeypatch.setattr(request_admission, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", 1 / 1024 / 1024)
    else:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    request: Final[dict[str, object]] = {
        "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}]}
        ]
    }
    with pytest.raises(litellm.BadRequestError, match="图片大小限制"):
        await _HOOK.async_filter_deployments("group", [_deployment(10000)], None, request)


@pytest.mark.asyncio
async def test_disabled_counter_does_not_silently_allow_limited_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_token_counter", True)
    with pytest.raises(litellm.BadRequestError, match="计数已关闭"):
        await _filter(_messages())


@pytest.mark.asyncio
async def test_unknown_context_limit_is_excluded_or_reported_as_configuration_error() -> None:
    unknown: Final[dict[str, object]] = {
        "model_name": "unknown",
        "litellm_params": {"model": "custom_openai/unmapped-admission"},
        "model_info": {"input_cost_per_token": 0, "output_cost_per_token": 0},
    }
    bounded: Final = _deployment(10000)
    assert await _HOOK.async_filter_deployments("group", [unknown, bounded], _messages()) == [bounded]
    with pytest.raises(litellm.BadRequestError, match="max_input_tokens"):
        await _HOOK.async_filter_deployments("group", [unknown], _messages())
    with pytest.raises(litellm.BadRequestError):
        await _HOOK.async_pre_call_deployment_hook(
            {"model": "custom_openai/unmapped-admission", "messages": _messages()}, CallTypes.acompletion
        )


@pytest.mark.asyncio
async def test_only_internal_fixed_response_provider_needs_no_context_budget() -> None:
    local: Final[dict[str, object]] = {"litellm_params": {"model": "claude_code_review_allow/allow"}}
    assert await _HOOK.async_filter_deployments("review", [local], _messages()) == [local]
    assert (
        await _HOOK.async_pre_call_deployment_hook(
            {"model": "claude_code_review_allow/allow", "messages": _messages()}, CallTypes.acompletion
        )
        is None
    )


@pytest.mark.asyncio
async def test_supported_server_tool_and_compaction_history_remain_usable() -> None:
    messages: Final[list[dict[str, object]]] = [
        {"role": "user", "content": "Search"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "example"}},
                {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": []},
                {"type": "compaction", "content": "Summary of previous work"},
            ],
        },
        {"role": "user", "content": "Continue"},
    ]
    assert await _filter(cast(list[AllMessageValues], messages))  # cast-ok: Anthropic history supported by tokenizer


@pytest.mark.asyncio
async def test_selected_deployment_path_checks_limits_and_skips_unrelated_calls() -> None:
    request: Final[dict[str, object]] = {
        "model": _MODEL,
        "messages": _messages("long context " * 100),
        "model_info": {"max_input_tokens": 100},
    }
    with pytest.raises(litellm.ContextWindowExceededError):
        await _HOOK.async_pre_call_deployment_hook(request, CallTypes.acompletion)
    assert await _HOOK.async_pre_call_deployment_hook(request, CallTypes.aembedding) is None
    assert await _HOOK.async_filter_deployments("group", [_deployment(100)], None, {"query": "embedding"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "mode"),
    (
        ("aembedding", "embedding"),
        ("aembedding_batch", "embedding"),
        ("arerank", "rerank"),
        ("arerank", "evaluation"),
        ("atranscription", "audio_transcription"),
        ("atranscription", "catalog"),
    ),
)
async def test_real_router_non_chat_calls_skip_chat_admission(
    operation: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_HOOK])
    backend: Final = AsyncMock(return_value={"proof": "backend reached"})
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "nonchat",
                "litellm_params": {
                    "model": "openai/whisper-1" if mode == "catalog" else "openai/noncatalog-nonchat",
                    "api_key": "mock-only",
                },
                "model_info": {} if mode == "catalog" else {"mode": mode},
            }
        ],
        num_retries=0,
        enable_pre_call_checks=True,
    )
    target: Final = "aembedding" if operation == "aembedding_batch" else operation
    try:
        with patch.object(litellm, target, backend):
            if target == "aembedding":
                await router.aembedding(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
                    model="nonchat", input=["hello", "world"] if operation == "aembedding_batch" else "hello"
                )
            elif target == "arerank":
                await router.arerank(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
                    model="nonchat", query="hello", documents=["world"]
                )
            else:
                await router.atranscription(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
                    model="nonchat", file=BytesIO(b"RIFFmock")
                )
        assert backend.await_count == 1
    finally:
        router.reset()


@pytest.mark.asyncio
async def test_client_mode_cannot_bypass_chat_admission() -> None:
    with pytest.raises(litellm.ContextWindowExceededError):
        await _filter(_messages("long history " * 100), limit=20, mode="embedding", model_info={"mode": "embedding"})
    with pytest.raises(litellm.ContextWindowExceededError):
        await _HOOK.async_pre_call_deployment_hook(
            {
                "model": _MODEL,
                "messages": _messages("long history " * 100),
                "model_info": {"mode": "embedding", "max_input_tokens": 20},
            },
            CallTypes.acompletion,
        )


class RecordingProvider(CustomLLM):
    def __init__(self) -> None:
        self.models: tuple[str, ...] = ()
        self.requests: tuple[Mapping[str, object], ...] = ()

    async def acompletion(self, model: str, *args: object, **kwargs: object) -> ModelResponse:
        self.models = (*self.models, model)
        self.requests = (*self.requests, kwargs)
        return ModelResponse(model=model, choices=[{"message": {"role": "assistant", "content": "accepted"}}])


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> RecordingProvider:
    provider: Final = RecordingProvider()
    monkeypatch.setattr(litellm, "callbacks", [_HOOK])
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "admission_test", "custom_handler": provider}])
    custom_llm_setup()
    return provider


@pytest.mark.asyncio
async def test_real_router_blocks_invalid_requests_before_provider(backend: RecordingProvider) -> None:
    router: Final = litellm.Router(model_list=[_deployment(100)], num_retries=0)
    try:
        response: Final[object] = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
            model="candidate", messages=_messages()
        )
        assert _answer(response) == "accepted"
        assert backend.models == ("model",)
        with pytest.raises(litellm.ContextWindowExceededError):
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
                model="candidate", messages=_messages("long history " * 100)
            )
        assert backend.models == ("model",)
    finally:
        router.reset()


@pytest.mark.asyncio
async def test_context_fallback_rechecks_actual_target(backend: RecordingProvider) -> None:
    router: Final = litellm.Router(
        model_list=[_deployment(20, "small"), _deployment(1000, "large")],
        context_window_fallbacks=[{"small": ["large"]}],
        num_retries=0,
    )
    try:
        response: Final[object] = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
            model="small", messages=_messages("moderate history " * 50)
        )
        assert _answer(response) == "accepted"
        assert backend.models == ("model",)
    finally:
        router.reset()


@pytest.mark.asyncio
async def test_direct_deployment_id_cannot_bypass_admission(backend: RecordingProvider) -> None:
    router: Final = litellm.Router(
        model_list=[{**_deployment(20, "group"), "model_info": {"id": "direct-id", "max_input_tokens": 20}}],
        num_retries=0,
    )
    try:
        with pytest.raises(litellm.ContextWindowExceededError):
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
                model="direct-id", messages=_messages("long history " * 100)
            )
        assert backend.models == ()
    finally:
        router.reset()


@pytest.mark.asyncio
async def test_existing_router_compaction_runs_before_final_admission(
    backend: RecordingProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # 上游价格表仍使用未参数化的 dict
        "summary-fixture",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_input_tokens": 32000,
            "max_output_tokens": 4096,
            "supports_anthropic_compaction": True,
        },
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "auto",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "small", "MEDIUM": "large", "COMPLEX": "large", "REASONING": "large"},
                        "keyword_tier_rules": [{"keywords": ["answer"], "tier": "SIMPLE"}],
                        "enable_context_window_escalation": False,
                        "max_tokens_from_tier_model": False,
                        "context_compaction": {"model": "large", "max_tokens": 512},
                    },
                },
            },
            _deployment(512, "small"),
            {
                "model_name": "large",
                "litellm_params": {"model": "anthropic/summary-fixture", "api_key": "unused"},
                "model_info": {"max_input_tokens": 32000, "max_output_tokens": 4096},
            },
        ],
        enable_pre_call_checks=True,
        num_retries=0,
    )

    async def summarize(
        protocol: CompactionProtocol, payload: Mapping[str, object], parent_model: str | None = None
    ) -> Mapping[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "provider_specific_fields": {
                            "compaction_blocks": [{"type": "compaction", "content": "Brief summary", "signature": "x"}]
                        }
                    }
                }
            ]
        }

    token: Final = compaction_executor.set(summarize)
    try:
        response: Final[object] = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 旧参数未标注
            model="auto",
            messages=[
                {"role": "user", "content": "earlier context " * 1000},
                {"role": "assistant", "content": "Noted"},
                {"role": "user", "content": "answer now"},
            ],
            max_tokens=64,
        )
        assert _answer(response) == "accepted"
        assert backend.models == ("model",)
        assert "earlier context" not in json.dumps(backend.requests[0]["messages"])
        assert "Brief summary" in json.dumps(backend.requests[0]["messages"])
    finally:
        compaction_executor.reset(token)
        router.reset()


_OPUS: Final = "claude-opus-5-5"
_CLAUDE_MODELS: Final = (_OPUS, "claude-sonnet-5-5", "claude-haiku-5-5")


def _claude_deployment(limit: int = 100, model: str = _OPUS) -> dict[str, object]:
    return {
        "model_name": "claude-fixture",
        "litellm_params": {"model": f"anthropic/{model}", "api_key": "test-key", "api_base": "https://counter.invalid"},
        "model_info": {"id": "claude-fixture", "max_input_tokens": limit},
    }


class ClaudeTransport:
    def __init__(self) -> None:
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        if request.url.path == "/v1/messages/count_tokens":
            return httpx.Response(503, json={"error": "计数服务不可用"})
        assert request.url.path == "/v1/messages"
        return httpx.Response(
            200,
            json={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": TypeAdapter(dict[str, object]).validate_json(request.content)["model"],
                "content": [{"type": "text", "text": "Accepted"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 75, "output_tokens": 1},
            },
        )


def _install_claude_transport(monkeypatch: pytest.MonkeyPatch, transport: ClaudeTransport) -> AsyncHTTPHandler:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))

    def get_client(llm_provider: object) -> AsyncHTTPHandler:
        return client

    monkeypatch.setattr(count_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(chat_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", get_client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", (CallTypes.aanthropic_messages, CallTypes.acompletion))
async def test_claude_container_remains_rejected_before_inference(
    monkeypatch: pytest.MonkeyPatch, call_type: CallTypes
) -> None:
    transport: Final = ClaudeTransport()
    client: Final = _install_claude_transport(monkeypatch, transport)
    try:
        with pytest.raises(litellm.BadRequestError, match="container.*无法本地核验"):
            await RequestAdmission().async_pre_call_deployment_hook(
                {
                    "model": _OPUS,
                    "api_key": "test-key",
                    "max_tokens": 100,
                    "messages": _messages(),
                    "extra_body": {"container": {"id": "fixture-container"}},
                    "model_info": {"max_input_tokens": 100},
                },
                call_type,
            )
        assert transport.requests == ()
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    (
        {"thinking": {"type": "adaptive", "display": "updates"}},
        {"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}},
        {"output_format": {"type": "json_schema", "schema": {"type": "object"}}},
        {"mcp_servers": [{"type": "url", "name": "fixture", "url": "https://mcp.invalid"}]},
    ),
)
async def test_claude_chat_extra_body_remains_rejected_before_inference(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, object]
) -> None:
    transport: Final = ClaudeTransport()
    client: Final = _install_claude_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment()], num_retries=0)
    try:
        with pytest.raises(litellm.BadRequestError, match="无法本地核验 extra_body"):
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=100, extra_body=extra
            )
        assert transport.requests == ()
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "chat", "responses"))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_real_router_claude_generates_without_remote_counting(
    monkeypatch: pytest.MonkeyPatch, model: str, route: str
) -> None:
    transport: Final = ClaudeTransport()
    client: Final = _install_claude_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=10000, model=model)], num_retries=0)
    try:
        if route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=10
            )
        elif route == "chat":
            response: Final = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=10
            )
            assert _answer(response) == "Accepted"
        else:
            await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", input="Hello", max_output_tokens=10
            )
        assert tuple(request.url.path for request in transport.requests) == ("/v1/messages",)
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "chat", "responses"))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_claude_local_admission_preserves_generation_request(
    monkeypatch: pytest.MonkeyPatch, model: str, route: str
) -> None:
    transport: Final = ClaudeTransport()
    client: Final = _install_claude_transport(monkeypatch, transport)
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=10000, model=model)], num_retries=0)
    schema: Final = {"type": "object", "properties": {"query": {"type": "string"}}}
    headers: Final = {"user-agent": "configured-inference/1.0", "x-app": "configured-inference"}

    async def request() -> None:
        if route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Earlier reasoning", "signature": "fixture-signature"},
                            {"type": "text", "text": "Ready"},
                        ],
                    },
                    {"role": "system", "content": [], "output_config": {"effort": "high"}},
                    {"role": "user", "content": "Continue"},
                ],
                system="System instructions",
                tools=[{"name": "lookup", "description": "Look up records", "input_schema": schema}],
                extra_headers=headers,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                max_tokens=10,
            )
        elif route == "chat":
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游路由参数未标注
                model="claude-fixture",
                messages=[{"role": "system", "content": "System instructions"}, *_messages()],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "lookup", "description": "Look up records", "parameters": schema},
                    }
                ],
                extra_headers=headers,
                max_tokens=10,
            )
        else:
            await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture",
                input="Hello",
                instructions="System instructions",
                tools=[{"type": "function", "name": "lookup", "description": "Look up records", "parameters": schema}],
                extra_headers=headers,
                max_output_tokens=10,
            )

    try:
        monkeypatch.setattr(litellm, "callbacks", [])
        await request()
        monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
        await request()
        assert tuple(request.url.path for request in transport.requests) == ("/v1/messages", "/v1/messages")
        baseline, admitted = transport.requests
        assert admitted.headers == baseline.headers
        assert admitted.content == baseline.content
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_claude_local_boundary_checks_candidates_and_selected_deployment(model: str) -> None:
    messages: Final = _messages("ordinary context " * 30)
    count: Final = litellm.token_counter(  # pyright: ignore[reportUnknownMemberType]  # 上游计数器仍有未注解参数
        model=model, messages=messages
    )
    allowed: Final = _claude_deployment(limit=count + 1, model=model)
    assert await _HOOK.async_filter_deployments("claude-fixture", [allowed], messages) == [allowed]
    with pytest.raises(litellm.ContextWindowExceededError, match="本地上下文预检"):
        await _HOOK.async_filter_deployments("claude-fixture", [_claude_deployment(limit=count, model=model)], messages)
    with pytest.raises(litellm.ContextWindowExceededError, match="本地上下文预检"):
        await _HOOK.async_pre_call_deployment_hook(
            {"model": model, "messages": messages, "model_info": {"max_input_tokens": count}}, CallTypes.acompletion
        )
    assert (
        await _HOOK.async_pre_call_deployment_hook(
            {"model": model, "messages": messages, "model_info": {"max_input_tokens": count + 1}}, CallTypes.acompletion
        )
        is None
    )


class HybridCounter:
    def __init__(self, local_tokens: int = 700_000, remote_tokens: int = 800_000, fail: bool = False) -> None:
        self.local_tokens = local_tokens
        self.remote_tokens = remote_tokens
        self.fail = fail
        self.calls: tuple[tuple[Mapping[str, object], CallTypes], ...] = ()

    def local_count(self, data: Mapping[str, object], model: str) -> int:
        return 1000 if frozenset(data) == frozenset(("messages",)) else self.local_tokens

    async def remote_count(self, data: Mapping[str, object], call_type: CallTypes) -> int:
        self.calls = (*self.calls, (copy.deepcopy(dict(data)), call_type))
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("计数服务不可用")
        return self.remote_tokens


def _hybrid_request(**overrides: object) -> dict[str, object]:
    return {
        "model": f"anthropic/{_OPUS}",
        "custom_llm_provider": "anthropic",
        "api_key": "fixture-provider-key",
        "api_base": "https://counter.invalid",
        "model_info": {"id": "hybrid-deployment", "base_model": _OPUS, "max_input_tokens": 950_000},
        "litellm_metadata": {"user_api_key_hash": "fixture-caller"},
        "messages": _messages("Original history"),
        "system": "Keep the original instructions",
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "max_tokens": 128,
        **overrides,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("local_tokens, expected_calls", ((699_999, 0), (700_000, 1), (950_000, 1)))
async def test_hybrid_remote_threshold_and_authoritative_remote_count(local_tokens: int, expected_calls: int) -> None:
    counter: Final = HybridCounter(local_tokens=local_tokens)
    hook: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=DualCache()
    )
    request: Final = _hybrid_request()
    original: Final = copy.deepcopy(request)
    assert await hook.async_pre_call_deployment_hook(request, CallTypes.aanthropic_messages) is None
    assert len(counter.calls) == expected_calls
    assert request == original


@pytest.mark.asyncio
@pytest.mark.parametrize("remote_tokens", (950_000, 950_001))
async def test_hybrid_remote_limit_rejects_locally_acceptable_input(remote_tokens: int) -> None:
    counter: Final = HybridCounter(remote_tokens=remote_tokens)
    hook: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=DualCache()
    )
    with pytest.raises(litellm.ContextWindowExceededError, match="/compact"):
        await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_hybrid_failed_remote_count_remains_retryable_without_cached_estimate() -> None:
    counter: Final = HybridCounter(fail=True)
    hook: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=DualCache()
    )
    for _ in range(2):
        with pytest.raises(litellm.BadRequestError, match="count_tokens.*请求未发送"):
            await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    assert len(counter.calls) == 2


@pytest.mark.asyncio
async def test_hybrid_identical_request_reuses_count_across_callback_instances() -> None:
    counter: Final = HybridCounter()
    cache: Final = DualCache()
    first: Final = RequestAdmission(claude_counter=counter.remote_count, local_counter=counter.local_count, cache=cache)
    second: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=cache
    )
    await first.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    await second.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_hybrid_parallel_identical_requests_share_one_count() -> None:
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=DualCache()
    )
    await asyncio.gather(
        *(hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages) for _ in range(6))
    )
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_hybrid_known_prefix_reuses_count_for_small_plain_user_increment() -> None:
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=DualCache()
    )
    await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    await hook.async_pre_call_deployment_hook(
        _hybrid_request(messages=[*_messages("Original history"), *_messages("Short new turn")]),
        CallTypes.aanthropic_messages,
    )
    assert len(counter.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    (
        {"system": "Different instructions"},
        {"tools": [{"name": "different_tool", "input_schema": {"type": "object"}}]},
        {"api_key": "different-provider-key"},
        {"litellm_metadata": {"user_api_key_hash": "different-caller"}},
        {"model_info": {"id": "different-deployment", "base_model": _OPUS, "max_input_tokens": 950_000}},
        {"extra_headers": {"user-agent": "different-client"}},
        {"metadata": {"user_id": "different-upstream-session"}},
        {"extra_headers": {"Authorization": "Bearer different-provider-token"}},
        {"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": 1}]}},
        {"messages": [*_messages("Changed history"), *_messages("Short new turn")]},
        {
            "messages": [
                *_messages("Original history"),
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "", "signature": "unobserved-signature"}],
                },
                *_messages("Short new turn"),
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": [{"type": "compaction", "content": "Compacted history"}]},
                *_messages("Short new turn"),
            ]
        },
    ),
    ids=(
        "system",
        "tools",
        "provider-key",
        "caller",
        "deployment",
        "identity",
        "metadata-identity",
        "forwarded-auth",
        "clearing",
        "history",
        "thinking",
        "compaction",
    ),
)
async def test_hybrid_changed_context_or_scope_requires_new_remote_count(changed: dict[str, object]) -> None:
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(
        claude_counter=counter.remote_count, local_counter=counter.local_count, cache=DualCache()
    )
    await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    await hook.async_pre_call_deployment_hook(
        _hybrid_request(**{"messages": [*_messages("Original history"), *_messages("Short new turn")], **changed}),
        CallTypes.aanthropic_messages,
    )
    assert len(counter.calls) == 2


class HybridAsyncStream(httpx.AsyncByteStream):
    def __init__(self, frames: tuple[bytes, ...]) -> None:
        self.frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self.frames:
            await asyncio.sleep(0)
            yield frame


class HybridTransport:
    def __init__(
        self,
        remote_tokens: int = 800_000,
        count_status: int = 200,
        signed_thinking: bool = False,
        usage: Mapping[str, int] | None = None,
        context_management: Mapping[str, object] | None = None,
        complete_stream: bool = True,
    ) -> None:
        self.remote_tokens = remote_tokens
        self.count_status = count_status
        self.signed_thinking = signed_thinking
        self.usage = dict(usage) if usage is not None else {"input_tokens": 75, "output_tokens": 1}
        self.context_management = context_management
        self.complete_stream = complete_stream
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        if request.url.path == "/v1/messages/count_tokens":
            return httpx.Response(
                self.count_status,
                json={"input_tokens": self.remote_tokens}
                if self.count_status == 200
                else {"error": {"type": "overloaded_error", "message": "计数服务不可用"}},
            )
        assert request.url.path == "/v1/messages"
        body: Final = TypeAdapter(dict[str, object]).validate_json(request.content)
        signature: Final = f"fixture-signature-{sum(item.url.path == '/v1/messages' for item in self.requests)}"
        thinking: Final = {"type": "thinking", "thinking": "A brief summary", "signature": signature}
        response: Final = {
            "id": "msg_hybrid",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [*([thinking] if self.signed_thinking else []), {"type": "text", "text": "Accepted"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": self.usage,
            **({"context_management": self.context_management} if self.context_management is not None else {}),
        }
        if body.get("stream") is not True:
            return httpx.Response(200, json=response)
        thinking_events: Final = (
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": thinking["thinking"]},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            },
            {"type": "content_block_stop", "index": 0},
        )
        text_index: Final = int(self.signed_thinking)
        events: Final = (
            {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
            *(thinking_events if self.signed_thinking else ()),
            {"type": "content_block_start", "index": text_index, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": "Accepted"}},
            {"type": "content_block_stop", "index": text_index},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": self.usage["output_tokens"]},
                **({"context_management": self.context_management} if self.context_management is not None else {}),
            },
            *(({"type": "message_stop"},) if self.complete_stream else ()),
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=HybridAsyncStream(
                tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
            ),
        )


def _install_hybrid_transport(
    monkeypatch: pytest.MonkeyPatch, transport: Callable[[httpx.Request], httpx.Response]
) -> AsyncHTTPHandler:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))

    def get_client(llm_provider: object) -> AsyncHTTPHandler:
        return client

    monkeypatch.setattr(count_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(chat_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(litellm, "module_level_aclient", client)
    return client


async def _hybrid_router_request(router: litellm.Router, route: str, stream: bool) -> object:
    schema: Final = {"type": "object", "properties": {"query": {"type": "string"}}}
    headers: Final = {"user-agent": "admission-fixture/1.0", "x-app": "admission-fixture"}
    if route == "messages":
        return cast(
            object,
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture",
                messages=_messages(),
                system="System instructions",
                tools=[{"name": "lookup", "description": "Find records", "input_schema": schema}],
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                max_tokens=128,
                extra_headers=headers,
                stream=stream,
            ),
        )
    if route == "chat":
        return await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
            model="claude-fixture",
            messages=[{"role": "system", "content": "System instructions"}, *_messages()],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "description": "Find records", "parameters": schema},
                }
            ],
            max_tokens=128,
            extra_headers=headers,
            stream=stream,
        )
    return cast(
        object,
        await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
            model="claude-fixture",
            input="Hello",
            instructions="System instructions",
            tools=[{"type": "function", "name": "lookup", "description": "Find records", "parameters": schema}],
            max_output_tokens=128,
            extra_headers=headers,
            stream=stream,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "chat", "responses"))
@pytest.mark.parametrize("stream", (False, True))
async def test_hybrid_real_router_counts_the_generation_payload_before_sending(
    monkeypatch: pytest.MonkeyPatch, route: str, stream: bool
) -> None:
    transport: Final = HybridTransport()
    client: Final = _install_hybrid_transport(monkeypatch, transport)
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(local_counter=counter.local_count, cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=950_000)], num_retries=0)
    try:
        response: Final = await _hybrid_router_request(router, route, stream)
        if stream:
            chunks: Final = tuple([chunk async for chunk in cast(AsyncIterator[object], response)])
            assert chunks
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted, generated = transport.requests
        counted_body: Final = TypeAdapter(dict[str, object]).validate_json(counted.content)
        generated_body: Final = TypeAdapter(dict[str, object]).validate_json(generated.content)
        for field in ("model", "messages", "system", "tools", "thinking", "output_config", "context_management"):
            assert counted_body.get(field) == generated_body.get(field), field
        for header in ("user-agent", "x-app"):
            assert counted.headers[header] == generated.headers[header]
        generation_beta: Final = generated.headers["anthropic-beta"] if "anthropic-beta" in generated.headers else ""
        counted_beta: Final = counted.headers["anthropic-beta"] if "anthropic-beta" in counted.headers else ""
        assert frozenset(filter(None, generation_beta.split(","))) <= frozenset(filter(None, counted_beta.split(",")))
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("remote_tokens, count_status", ((950_000, 200), (800_000, 503), (0, 200)))
@pytest.mark.parametrize("stream", (False, True))
async def test_hybrid_real_router_never_generates_after_failed_or_excessive_count(
    monkeypatch: pytest.MonkeyPatch, remote_tokens: int, count_status: int, stream: bool
) -> None:
    transport: Final = HybridTransport(remote_tokens=remote_tokens, count_status=count_status)
    client: Final = _install_hybrid_transport(monkeypatch, transport)
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(local_counter=counter.local_count, cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=950_000)], num_retries=0)
    try:
        with pytest.raises(litellm.BadRequestError):
            await _hybrid_router_request(router, "messages", stream)
        assert transport.requests
        assert all(request.url.path == "/v1/messages/count_tokens" for request in transport.requests)
    finally:
        router.reset()
        await client.client.aclose()


async def _hybrid_native_payload(response: object, stream: bool, allow_incomplete: bool = False) -> dict[str, object]:
    if not stream:
        return TypeAdapter(dict[str, object]).validate_python(response)
    chunks: Final = tuple([chunk async for chunk in cast(AsyncIterator[object], response)])
    assembled: Final = assemble_anthropic_sse_stream(
        chunks, restore_identity=True, through_upstream_error=allow_incomplete
    )
    assert assembled is not None
    return TypeAdapter(dict[str, object]).validate_python(
        LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(assembled)
    )


async def _hybrid_native_round(
    router: litellm.Router, history: tuple[dict[str, object], ...], stream: bool, allow_incomplete: bool = False
) -> tuple[dict[str, object], ...]:
    response: Final = cast(
        object,
        await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
            model="claude-fixture",
            messages=[*history, {"role": "user", "content": "A short follow-up"}],
            system="Preserve the conversation",
            tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
            thinking={"type": "adaptive"},
            context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            litellm_metadata={"user_api_key_hash": "fixture-caller"},
            max_tokens=8192,
            stream=stream,
        ),
    )
    payload: Final = await _hybrid_native_payload(response, stream, allow_incomplete)
    content: Final = TypeAdapter(list[dict[str, object]]).validate_python(payload["content"])
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "A brief summary"
    assert str(content[0]["signature"]).startswith("fixture-signature-")
    assert content[1]["text"] == "Accepted"
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    return (*history, {"role": "user", "content": "A short follow-up"}, {"role": "assistant", "content": content})


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("cache_creation, expected_counts", ((100_000, 1), (225_000, 3)))
async def test_hybrid_native_success_reuses_signed_history_and_rechecks_full_cached_usage(
    monkeypatch: pytest.MonkeyPatch, stream: bool, cache_creation: int, expected_counts: int
) -> None:
    usage: Final = {
        "input_tokens": 50_000,
        "cache_read_input_tokens": 650_000,
        "cache_creation_input_tokens": cache_creation,
        "output_tokens": 1000,
    }
    transport: Final = HybridTransport(
        remote_tokens=usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"],
        signed_thinking=True,
        usage=usage,
    )
    client: Final = _install_hybrid_transport(monkeypatch, transport)
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(local_counter=counter.local_count, cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=950_000)], num_retries=0)
    try:
        first: Final = await _hybrid_native_round(router, (), stream)
        second: Final = await _hybrid_native_round(router, first, stream)
        third: Final = await _hybrid_native_round(router, second, stream)
        assert len(third) == 6
        assert sum(request.url.path == "/v1/messages" for request in transport.requests) == 3
        assert sum(request.url.path == "/v1/messages/count_tokens" for request in transport.requests) == expected_counts
    finally:
        router.reset()
        await client.client.aclose()


class HybridClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_hybrid_expired_count_requires_remote_recheck() -> None:
    clock: Final = HybridClock()
    counter: Final = HybridCounter()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    hook: Final = RequestAdmission(claude_counter=counter.remote_count, local_counter=counter.local_count, cache=cache)
    await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    assert len(counter.calls) == 1
    clock.now = 301.0
    await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    assert len(counter.calls) == 2


@pytest.mark.asyncio
async def test_hybrid_claude_remote_check_uses_deployment_model_despite_other_base_model() -> None:
    counter: Final = HybridCounter(remote_tokens=950_000)
    hook: Final = RequestAdmission(claude_counter=counter.remote_count, local_counter=counter.local_count)
    request: Final = _hybrid_request(
        model_info={"id": "base-model-override", "base_model": _MODEL, "max_input_tokens": 950_000}
    )
    with pytest.raises(litellm.ContextWindowExceededError, match="/compact"):
        await hook.async_pre_call_deployment_hook(request, CallTypes.aanthropic_messages)
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_hybrid_large_plain_increment_requires_remote_recheck() -> None:
    counter: Final = HybridCounter()

    def local_count(data: Mapping[str, object], model: str) -> int:
        return 20_001 if frozenset(data) == frozenset(("messages",)) else 700_000

    hook: Final = RequestAdmission(claude_counter=counter.remote_count, local_counter=local_count)
    await hook.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    await hook.async_pre_call_deployment_hook(
        _hybrid_request(messages=[*_messages("Original history"), *_messages("Large new tool result")]),
        CallTypes.aanthropic_messages,
    )
    assert len(counter.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("native", (False, True))
async def test_hybrid_count_uses_the_same_explicit_auth_and_beta_sources_as_generation(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    transport: Final = HybridTransport()
    client: Final = _install_hybrid_transport(monkeypatch, transport)
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(local_counter=counter.local_count)
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=950_000)], num_retries=0)
    headers: Final = {"x-api-key": "fixture-forwarded-key", "anthropic-beta": "fixture-beta-one"}
    extra_headers: Final = {"anthropic-beta": "fixture-beta-two", "user-agent": "fixture-header-parity"}
    try:
        if native:
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # Router 原生接口仍使用动态参数
                model="claude-fixture",
                messages=_messages(),
                max_tokens=128,
                headers=headers,
                extra_headers=extra_headers,
            )
        else:
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router Chat 接口仍使用动态参数
                model="claude-fixture",
                messages=_messages(),
                max_tokens=128,
                headers=headers,
                extra_headers=extra_headers,
            )
        counted, generated = transport.requests
        for name in ("x-api-key", "authorization", "user-agent"):
            assert counted.headers.get(name) == generated.headers.get(name)
        generation_beta: Final = generated.headers["anthropic-beta"] if "anthropic-beta" in generated.headers else ""
        count_beta: Final = counted.headers["anthropic-beta"] if "anthropic-beta" in counted.headers else ""
        assert frozenset(filter(None, generation_beta.split(","))) <= frozenset(filter(None, count_beta.split(",")))
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, context_management, complete_stream, expected_counts",
    (
        (False, {"applied_edits": []}, True, 1),
        (
            False,
            {
                "applied_edits": [
                    {"type": "clear_thinking_20251015", "cleared_thinking_turns": 1, "cleared_input_tokens": 1000}
                ]
            },
            True,
            2,
        ),
        (True, {"applied_edits": []}, True, 1),
        (True, {"applied_edits": []}, False, 2),
    ),
    ids=("ordinary-no-edit", "ordinary-edited", "stream-complete", "stream-without-message-stop"),
)
async def test_hybrid_native_usage_requires_complete_unedited_response(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    context_management: dict[str, object],
    complete_stream: bool,
    expected_counts: int,
) -> None:
    transport: Final = HybridTransport(
        signed_thinking=True,
        usage={"input_tokens": 800_000, "output_tokens": 1000},
        context_management=context_management,
        complete_stream=complete_stream,
    )
    client: Final = _install_hybrid_transport(monkeypatch, transport)
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(local_counter=counter.local_count, cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=950_000)], num_retries=0)
    try:
        first: Final = await _hybrid_native_round(router, (), stream, allow_incomplete=not complete_stream)
        await _hybrid_native_round(router, first, stream, allow_incomplete=not complete_stream)
        assert sum(request.url.path == "/v1/messages" for request in transport.requests) == 2
        assert sum(request.url.path == "/v1/messages/count_tokens" for request in transport.requests) == expected_counts
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
async def test_hybrid_workers_share_coordination_redis_when_auth_cache_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared: Final = InMemoryCache()
    redis: Final = cast(
        RedisCache,
        Mock(
            spec=RedisCache,
            async_get_cache=AsyncMock(side_effect=shared.async_get_cache),  # pyright: ignore[reportUnknownMemberType]  # 上游缓存接口尚未注解
            async_set_cache=AsyncMock(side_effect=shared.async_set_cache),  # pyright: ignore[reportUnknownMemberType]  # 上游缓存接口尚未注解
        ),
    )
    monkeypatch.setattr(config_sync_pubsub, "coordination_redis_cache", lambda: redis)
    counter: Final = HybridCounter()
    first: Final = RequestAdmission(claude_counter=counter.remote_count, local_counter=counter.local_count)
    second: Final = RequestAdmission(claude_counter=counter.remote_count, local_counter=counter.local_count)
    await first.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), _hybrid_request(), "aanthropic_messages")
    await second.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), _hybrid_request(), "aanthropic_messages")
    await first.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    await second.async_pre_call_deployment_hook(_hybrid_request(), CallTypes.aanthropic_messages)
    assert len(counter.calls) == 1


@pytest.mark.asyncio
async def test_hybrid_slow_generation_starts_usage_anchor_ttl_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Final = HybridClock()
    started: Final = time()
    clock.now = started
    monkeypatch.setattr(request_admission, "time", SimpleNamespace(time=clock))
    transport: Final = HybridTransport(signed_thinking=True, usage={"input_tokens": 800_000, "output_tokens": 1000})

    def delayed_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages":
            clock.now += 301
        return transport(request)

    client: Final = _install_hybrid_transport(monkeypatch, delayed_transport)
    counter: Final = HybridCounter()
    hook: Final = RequestAdmission(local_counter=counter.local_count, cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = litellm.Router(model_list=[_claude_deployment(limit=950_000)], num_retries=0)
    try:
        first: Final = await _hybrid_native_round(router, (), stream=True)
        second: Final = await _hybrid_native_round(router, first, stream=True)
        await _hybrid_native_round(router, second, stream=True)
        assert clock.now == started + 903
        assert sum(request.url.path == "/v1/messages" for request in transport.requests) == 3
        assert sum(request.url.path == "/v1/messages/count_tokens" for request in transport.requests) == 1
    finally:
        router.reset()
        await client.client.aclose()
