import base64
import copy
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from io import BytesIO
from typing import Final, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from PIL import Image
from pydantic import TypeAdapter
from pytest_socket import disable_socket, enable_socket

import litellm
from litellm import ModelResponse
from litellm.exceptions import RequestAdmissionError
from litellm.litellm_core_utils.prompt_templates.common_utils import encrypted_reasoning_signature
from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.llms.anthropic.chat import handler as chat_handler
from litellm.llms.anthropic.count_tokens import handler as count_handler
from litellm.llms.compaction import CompactionProtocol
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.responses.litellm_completion_transformation.reasoning_items import encode_thinking_blocks
from litellm.router_strategy.complexity_router.context_compaction import compaction_executor
from litellm.types.llms.anthropic import ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, Choices, CredentialItem
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
                        else {"content": text}
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_claude_filters_defer_counting_and_selected_deployment_uses_provider_limit(model: str) -> None:
    counter: Final = AsyncMock(return_value=100)
    hook: Final = RequestAdmission(claude_counter=counter)
    deployment: Final = _claude_deployment(model=model)
    messages: Final = _messages()
    assert await hook.async_filter_deployments("claude-fixture", [deployment], messages) == [deployment]
    counter.assert_not_awaited()
    with pytest.raises(litellm.ContextWindowExceededError, match="上游计数 100"):
        await hook.async_pre_call_deployment_hook(
            {"model": model, "messages": messages, "model_info": {"max_input_tokens": 100}}, CallTypes.acompletion
        )
    assert counter.await_count == 1
    assert (
        await hook.async_pre_call_deployment_hook(
            {"model": model, "messages": messages, "model_info": {"max_input_tokens": 101}}, CallTypes.acompletion
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_claude_counter_failure_is_closed_and_other_models_keep_local_counting(model: str) -> None:
    counter: Final = AsyncMock(side_effect=TimeoutError("private-upstream-marker"))
    hook: Final = RequestAdmission(claude_counter=counter)
    with pytest.raises(litellm.BadRequestError, match="上游计数失败") as error:
        await hook.async_pre_call_deployment_hook(
            {"model": model, "messages": _messages(), "model_info": {"max_input_tokens": 100}}, CallTypes.acompletion
        )
    assert "private-upstream-marker" not in str(error.value)
    assert (
        await hook.async_pre_call_deployment_hook(
            {"model": _MODEL, "messages": _messages(), "model_info": {"max_input_tokens": 100}}, CallTypes.acompletion
        )
        is None
    )
    assert await hook.async_filter_deployments(model, [_deployment(100, model)], _messages())
    assert counter.await_count == 1


class CountTransport:
    def __init__(self, result: object, status: int = 200, allow_inference: bool = False) -> None:
        self.result = result
        self.allow_inference = allow_inference
        self.status = status
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        if self.allow_inference and request.url.path == "/v1/messages":
            body: Final = TypeAdapter(dict[str, object]).validate_json(request.content)
            if body.get("stream") is True:
                events: Final[tuple[Mapping[str, object], ...]] = (
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_fixture",
                            "type": "message",
                            "role": "assistant",
                            "model": body["model"],
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 75, "output_tokens": 0},
                        },
                    },
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Accepted"}},
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
                )
            return httpx.Response(
                200,
                json={
                    "id": "msg_fixture",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": "Accepted"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 75, "output_tokens": 1},
                },
            )
        assert request.url.path == "/v1/messages/count_tokens"
        if request.headers.get("content-type") != "application/json":
            return httpx.Response(422, json={"error": "请求必须使用单个 application/json 类型头"})
        return httpx.Response(self.status, json=self.result)


def _install_count_transport(monkeypatch: pytest.MonkeyPatch, transport: CountTransport) -> AsyncHTTPHandler:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))

    def get_client(llm_provider: object) -> AsyncHTTPHandler:
        return client

    monkeypatch.setattr(count_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(chat_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", get_client)
    monkeypatch.setattr(litellm, "module_level_aclient", client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", (CallTypes.aanthropic_messages, CallTypes.acompletion))
async def test_opus_provider_count_preserves_named_credentials_system_tools_and_options(
    monkeypatch: pytest.MonkeyPatch, call_type: CallTypes
) -> None:
    transport: Final = CountTransport({"input_tokens": 75})
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="count-fixture",
                credential_info={},
                credential_values={"api_key": "named-test-key", "api_base": "https://named.invalid"},
            )
        ],
    )
    tool: Final = {
        "name": "lookup",
        "description": "Look up records",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "$defs": {"detail": {"type": "string"}},
        },
    }
    native: Final = call_type == CallTypes.aanthropic_messages
    data: Final[dict[str, object]] = {
        "model": f"anthropic/{_OPUS}",
        "litellm_credential_name": "count-fixture",
        "max_tokens": 10,
        "messages": _messages(),
        "system": "System instructions",
        "tools": [tool]
        if native
        else [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
        ],
        "tool_choice": {"type": "auto"} if native else "auto",
    }
    original: Final = copy.deepcopy(data)
    try:
        assert (
            await RequestAdmission().async_pre_call_deployment_hook(
                {**data, "model_info": {"max_input_tokens": 76}}, call_type
            )
            is None
        )
        request: Final = transport.requests[0]
        body: Final = TypeAdapter(dict[str, object]).validate_json(request.content)
        assert request.url.host == "named.invalid"
        assert request.headers["x-api-key"] == "named-test-key"
        assert "System instructions" in json.dumps(body["system"])
        counted_tools: Final = TypeAdapter(list[dict[str, object]]).validate_python(body["tools"])
        assert counted_tools[0]["input_schema"] == tool["input_schema"]
        assert body["tool_choice"] == {"type": "auto"}
        assert body["messages"] in (_messages(), [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])
        assert data == original
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    (
        {},
        {"input_tokens": None},
        {"input_tokens": True},
        {"input_tokens": 0},
        {"input_tokens": -1},
        {"input_tokens": "75"},
    ),
)
async def test_opus_malformed_count_is_rejected_without_local_fallback(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    transport: Final = CountTransport(result)
    client: Final = _install_count_transport(monkeypatch, transport)
    try:
        with pytest.raises(litellm.BadRequestError, match="上游计数失败"):
            await RequestAdmission().async_pre_call_deployment_hook(
                {
                    "model": _OPUS,
                    "api_key": "test-key",
                    "max_tokens": 10,
                    "messages": _messages(),
                    "model_info": {"max_input_tokens": 100},
                },
                CallTypes.aanthropic_messages,
            )
        assert len(transport.requests) == 1
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("metadata", "litellm_metadata", "proxy_server_request", None))
@pytest.mark.parametrize("outbound_source", (None, "headers", "extra_headers", "provider_specific_header"))
@pytest.mark.parametrize("api_key", ("test-key", "sk-ant-oat-fixture-token"))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_claude_count_matches_generation_identity_without_forwarding_original_credentials(
    monkeypatch: pytest.MonkeyPatch, model: str, source: str | None, api_key: str, outbound_source: str | None
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    deployment: Final = _claude_deployment(model=model)
    params: Final = TypeAdapter(dict[str, object]).validate_python(deployment["litellm_params"])
    outgoing: Final = {
        "user-agent": "configured-client/1.0",
        "x-app": "configured-app",
        "X-Session-ID": "configured-session",
        "anthropic-user-profile-id": "configured-profile",
        "Anthropic-Beta": "fixture-configured-feature",
    }
    header_options: Final[dict[str, object]] = (
        {}
        if outbound_source is None
        else {
            outbound_source: (
                {"custom_llm_provider": "anthropic", "extra_headers": outgoing}
                if outbound_source == "provider_specific_header"
                else outgoing
            )
        }
    )
    router: Final = litellm.Router(
        model_list=[{**deployment, "litellm_params": {**params, "api_key": api_key, **header_options}}], num_retries=0
    )
    inbound: Final = {
        "User-Agent": "claude-cli/2.1.319 (fixture)",
        "X-App": "cli",
        "Anthropic-Beta": "fixture-original-feature",
        "Authorization": "Bearer inbound-secret",
        "X-Api-Key": "inbound-secret",
        "Cookie": "session=inbound-secret",
        "X-Private-Header": "inbound-secret",
    }
    carried: Final = {source: {"headers": inbound}} if source is not None else {}

    async def request() -> None:
        await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
            model="claude-fixture",
            messages=_messages(),
            max_tokens=100,
            thinking={"type": "adaptive", "display": "updates"},
            extra_headers={
                "anthropic-beta": "fixture-caller-feature",
                **(outgoing if outbound_source == "extra_headers" else {}),
            },
            **carried,  # pyright: ignore[reportArgumentType]  # 测试只提供原始请求上下文
        )

    try:
        monkeypatch.setattr(litellm, "callbacks", [])
        await request()
        monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
        await request()
        assert tuple(item.url.path for item in transport.requests) == (
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        baseline, counted, generated = transport.requests
        assert generated.headers == baseline.headers
        assert counted.headers.get("authorization") == baseline.headers.get("authorization")
        assert counted.headers.get("x-api-key") == baseline.headers.get("x-api-key")
        assert "inbound-secret" not in json.dumps(dict(counted.headers))
        assert "cookie" not in counted.headers
        assert "x-private-header" not in counted.headers
        assert set(baseline.headers["anthropic-beta"].split(",")) <= set(counted.headers["anthropic-beta"].split(","))
        assert counted.headers.get("user-agent") == generated.headers.get("user-agent")
        assert counted.headers.get("x-app") == generated.headers.get("x-app")
        assert counted.headers.get("x-session-id") == generated.headers.get("x-session-id")
        assert counted.headers.get("anthropic-user-profile-id") == generated.headers.get("anthropic-user-profile-id")
        assert "fixture-original-feature" not in counted.headers["anthropic-beta"].split(",")
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "chat", "responses"))
async def test_claude_count_preserves_generation_identity_header_duplicates(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    deployment: Final = _claude_deployment()
    params: Final = TypeAdapter(dict[str, object]).validate_python(deployment["litellm_params"])
    router: Final = litellm.Router(
        model_list=[
            {
                **deployment,
                "litellm_params": {
                    **params,
                    "headers": {"User-Agent": "forwarded-client", "X-App": "forwarded-app"},
                    "extra_headers": {"user-agent": "extra-client", "x-app": "extra-app"},
                    "provider_specific_header": [
                        {
                            "custom_llm_provider": "openai",
                            "extra_headers": {"USER-AGENT": "unrelated-client", "X-APP": "unrelated-app"},
                        },
                        {
                            "custom_llm_provider": "anthropic",
                            "extra_headers": {"USER-AGENT": "scoped-client", "X-APP": "scoped-app"},
                        },
                    ],
                },
            }
        ],
        num_retries=0,
    )
    try:
        if route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=100
            )
        elif route == "chat":
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=100
            )
        else:
            await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", input="Hello", max_output_tokens=100
            )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted, generated = transport.requests
        identity_names: Final = frozenset((b"user-agent", b"x-app"))
        counted_identity: Final = tuple(pair for pair in counted.headers.raw if pair[0].lower() in identity_names)
        generated_identity: Final = tuple(pair for pair in generated.headers.raw if pair[0].lower() in identity_names)
        assert counted_identity == generated_identity
        assert generated.headers.get_list("user-agent") == ["forwarded-client", "extra-client", "scoped-client"]
        assert generated.headers.get_list("x-app") == ["forwarded-app", "extra-app", "scoped-app"]
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "chat", "responses"))
@pytest.mark.parametrize("failure", (False, True))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_real_router_claude_uses_one_provider_count_and_never_sends_rejected_inference(
    monkeypatch: pytest.MonkeyPatch, model: str, route: str, failure: bool
) -> None:
    transport: Final = CountTransport(
        {"error": "count unavailable"} if failure else {"input_tokens": 100}, 503 if failure else 200
    )
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment(model=model)], num_retries=0)

    async def request() -> None:
        if route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由方法参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=10
            )
        elif route == "chat":
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游路由方法参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=10
            )
        else:
            await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由方法参数未标注
                model="claude-fixture",
                input=[{"role": "user", "content": "Hello"}],
                max_output_tokens=10,
            )

    try:
        with pytest.raises(litellm.BadRequestError if failure else litellm.ContextWindowExceededError):
            await request()
        assert len(transport.requests) == 1
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "responses", "messages"))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_real_router_claude_count_and_inference_use_the_same_transformed_body(
    monkeypatch: pytest.MonkeyPatch, model: str, route: str
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    identity: Final = {"user-agent": "configured-inference/1.0", "x-app": "configured-inference"}
    deployment: Final = _claude_deployment(model=model)
    params: Final = TypeAdapter(dict[str, object]).validate_python(deployment["litellm_params"])
    router: Final = litellm.Router(
        model_list=[{**deployment, "litellm_params": {**params, "extra_headers": identity}}], num_retries=0
    )
    schema: Final = {"type": "object", "properties": {"query": {"type": "string"}}}
    try:
        if route == "chat":
            response: Final = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游路由方法参数未标注
                model="claude-fixture",
                messages=[{"role": "system", "content": "System instructions"}, *_messages()],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "lookup", "description": "Look up records", "parameters": schema},
                    }
                ],
                max_tokens=10,
            )
            assert _answer(response) == "Accepted"
        elif route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由方法参数未标注
                model="claude-fixture",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "system", "content": [], "output_config": {"effort": "high"}},
                    {"role": "user", "content": "Continue"},
                ],
                system="System instructions",
                tools=[{"name": "lookup", "description": "Look up records", "input_schema": schema}],
                extra_headers={**identity, "anthropic-beta": "fixture-client-feature"},
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                max_tokens=10,
            )
        else:
            await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由方法参数未标注
                model="claude-fixture",
                input="Hello",
                instructions="System instructions",
                tools=[{"type": "function", "name": "lookup", "description": "Look up records", "parameters": schema}],
                max_output_tokens=10,
            )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        assert transport.requests[0].headers.get("user-agent") == transport.requests[1].headers.get("user-agent")
        assert transport.requests[1].headers.get("user-agent") == identity["user-agent"]
        assert transport.requests[0].headers.get("x-app") == transport.requests[1].headers.get("x-app")
        counted: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[1].content)
        assert {key: counted[key] for key in ("model", "system", "messages", "tools")} == {
            key: sent[key] for key in ("model", "system", "messages", "tools")
        }
        assert counted["model"] == sent["model"] == model
        if route == "messages":
            assert counted["thinking"] == sent["thinking"]
            assert counted["output_config"] == sent["output_config"]
            assert set(transport.requests[1].headers["anthropic-beta"].split(",")) <= set(
                transport.requests[0].headers["anthropic-beta"].split(",")
            )
            assert ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER in transport.requests[0].headers[
                "anthropic-beta"
            ].split(",")
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (False, True))
@pytest.mark.parametrize("route", ("chat", "messages"))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_rejected_claude_request_does_not_cool_down_the_next_request(
    monkeypatch: pytest.MonkeyPatch, model: str, failure: bool, route: str
) -> None:
    counter: Final = AsyncMock(side_effect=[TimeoutError("fixture") if failure else 100, 75])
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission(claude_counter=counter)])
    router: Final = litellm.Router(
        model_list=[_claude_deployment(model=model)], num_retries=0, allowed_fails=0, cooldown_time=60
    )

    async def request() -> None:
        if route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由
                model="claude-fixture", messages=_messages(), max_tokens=10
            )
        else:
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=10
            )

    try:
        with pytest.raises(litellm.BadRequestError if failure else litellm.ContextWindowExceededError):
            await request()
        assert transport.requests == ()
        await request()
        assert counter.await_count == 2
        assert tuple(item.url.path for item in transport.requests) == ("/v1/messages",)
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "auto_summary"),
    (
        ({"thinking": {"type": "adaptive", "display": "updates"}}, False),
        ({"thinking": {"type": "adaptive", "display": "updates"}}, True),
        ({"thinking": {"type": "enabled", "budget_tokens": 4096}}, False),
        ({"thinking": {"type": "disabled"}}, False),
        ({"reasoning_effort": "high"}, False),
        ({"output_format": {"type": "json_schema", "schema": {"type": "object"}}}, False),
        ({"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}}, False),
        ({"cache_control": {"type": "ephemeral"}}, False),
        ({"compaction": {"type": "summarize", "instructions": "Keep important details"}}, False),
        ({"thinking": {"type": "adaptive"}, "additional_drop_params": ["thinking"]}, False),
        (
            {
                "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
                "additional_drop_params": ["context_management"],
            },
            False,
        ),
    ),
)
@pytest.mark.parametrize("route", ("messages", "chat"))
async def test_opus_count_matches_final_inference_options(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, object], auto_summary: bool, route: str
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    monkeypatch.setattr(litellm, "reasoning_auto_summary", auto_summary)
    router: Final = litellm.Router(model_list=[_claude_deployment()], num_retries=0)
    try:
        if route == "messages":
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture",
                messages=_messages(),
                max_tokens=8192,
                system="Policy",
                **options,  # pyright: ignore[reportArgumentType]  # 测试仅传 Anthropic 原生可选参数
            )
        else:
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType, reportCallIssue]  # 参数化原生参数
                model="claude-fixture",
                messages=[{"role": "system", "content": "Policy"}, *_messages()],
                max_tokens=8192,
                **options,  # pyright: ignore[reportArgumentType]  # 测试只提供原生参数，不覆盖 stream
            )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[1].content)
        expected: Final = {
            key: value for key, value in sent.items() if key not in ("max_tokens", "stream", "output_format")
        }
        assert counted == (
            {**expected, "output_config": {"format": sent["output_format"]}} if "output_format" in sent else expected
        )
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_parameter", ("max_tokens", "max_completion_tokens"))
@pytest.mark.parametrize("budget", (1000, 1500))
async def test_chat_count_matches_inference_with_legacy_thinking_budget_and_dropped_tools(
    monkeypatch: pytest.MonkeyPatch, budget_parameter: str, budget: int
) -> None:
    model: Final = "claude-sonnet-admission-fixture"
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            model: {
                "max_input_tokens": 100000,
                "max_output_tokens": 8192,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_function_calling": True,
                "supports_reasoning": True,
                "supports_adaptive_thinking": False,
            }
        },
    )
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment(model=model)], num_retries=0)
    try:
        await router.acompletion(  # pyright: ignore[reportUnknownMemberType, reportCallIssue]  # 上游参数化路由预算
            model="claude-fixture",
            messages=_messages(),
            thinking={"type": "adaptive"},
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            additional_drop_params=["tools"],
            **{budget_parameter: budget},  # pyright: ignore[reportArgumentType]  # 参数化输出预算字段
        )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[1].content)
        assert counted == {key: value for key, value in sent.items() if key not in ("max_tokens", "stream")}
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", (CallTypes.aanthropic_messages, CallTypes.acompletion))
async def test_opus_container_is_rejected_before_count_or_inference(
    monkeypatch: pytest.MonkeyPatch, call_type: CallTypes
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    try:
        with pytest.raises(litellm.BadRequestError, match="不支持 container"):
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
    ),
)
async def test_native_opus_extra_body_does_not_override_the_effective_request(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, object]
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment()], num_retries=0)
    before: Final = copy.deepcopy(extra)
    try:
        await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
            model="claude-fixture",
            messages=_messages(),
            max_tokens=8192,
            thinking={"type": "adaptive", "display": "omitted"},
            extra_body=extra,
        )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[1].content)
        assert counted == {key: value for key, value in sent.items() if key not in ("max_tokens", "stream")}
        assert extra == before
    finally:
        router.reset()
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
async def test_chat_opus_extra_body_cannot_be_counted_as_flattened_native_input(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, object]
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment()], num_retries=0)
    try:
        with pytest.raises(litellm.BadRequestError, match="无法核验 extra_body"):
            await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
                model="claude-fixture", messages=_messages(), max_tokens=100, extra_body=extra
            )
        assert transport.requests == ()
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("force_chat", (False, True))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_native_responses_cannot_bypass_claude_count_but_explicit_chat_bridge_can_run(
    monkeypatch: pytest.MonkeyPatch, model: str, force_chat: bool
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    deployment: Final = {
        **_claude_deployment(model=model),
        "model_info": {"id": "claude-fixture", "max_input_tokens": 100, "supported_endpoints": ["/v1/responses"]},
    }
    router: Final = litellm.Router(model_list=[deployment], num_retries=0)
    try:
        if force_chat:
            await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由方法参数未标注
                model="claude-fixture",
                input="Hello",
                max_output_tokens=10,
                use_chat_completions_api=True,
            )
            assert tuple(request.url.path for request in transport.requests) == (
                "/v1/messages/count_tokens",
                "/v1/messages",
            )
        else:
            with pytest.raises(litellm.BadRequestError, match="原生 Responses 无法核验"):
                await router.aresponses(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由方法参数未标注
                    model="claude-fixture",
                    input="Hello",
                    max_output_tokens=10,
                )
            assert transport.requests == ()
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", ({"custom_llm_provider": "openai"}, {"model": "custom-opus-alias"}))
@pytest.mark.parametrize("model", _CLAUDE_MODELS)
async def test_claude_base_model_cannot_defer_an_unverified_responses_backend(
    model: str, overrides: dict[str, object]
) -> None:
    counter: Final = AsyncMock(return_value=75)
    hook: Final = RequestAdmission(claude_counter=counter)
    with pytest.raises(litellm.BadRequestError, match="原生 Responses 无法核验"):
        await hook.async_pre_call_deployment_hook(
            {
                "model": model,
                "input": "Hello",
                "model_info": {"base_model": model, "max_input_tokens": 100},
                **overrides,
            },
            CallTypes.aresponses,
        )
    counter.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("messages", "chat", "responses"))
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("redacted", (False, True))
async def test_claude_reasoning_history_is_counted_and_forwarded_without_rewriting(
    monkeypatch: pytest.MonkeyPatch, route: str, stream: bool, redacted: bool
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment()], num_retries=0)
    block: Final = (
        {"type": "redacted_thinking", "data": "opaque-history"}
        if redacted
        else {"type": "thinking", "thinking": "Short visible summary", "signature": "opaque-history"}
    )
    messages: Final = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": [block, {"type": "text", "text": "Earlier answer"}]},
        {"role": "user", "content": "Continue"},
    ]
    before: Final = copy.deepcopy(messages)

    async def consume(response: object) -> None:
        if stream:
            assert isinstance(response, AsyncIterator)
            chunks: Final = [
                chunk
                async for chunk in cast(AsyncIterator[object], response)  # cast-ok: 已确认响应实现异步迭代协议
            ]
            assert chunks

    try:
        if route == "messages":
            await consume(
                await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # 上游动态路由
                    model="claude-fixture", messages=messages, max_tokens=10, stream=stream
                )
            )
        elif route == "chat":
            await consume(
                await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由
                    model="claude-fixture",
                    messages=TypeAdapter(list[AllMessageValues]).validate_python(
                        [
                            messages[0],
                            {"role": "assistant", "content": "Earlier answer", "thinking_blocks": [block]},
                            messages[2],
                        ]
                    ),
                    max_tokens=10,
                    stream=stream,
                )
            )
        else:
            await consume(
                await router.aresponses(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # 上游动态路由
                    model="claude-fixture",
                    input=[
                        {"role": "user", "content": "Earlier question"},
                        {"type": "reasoning", "summary": [], "encrypted_content": encode_thinking_blocks([block])},
                        {"role": "assistant", "content": [{"type": "output_text", "text": "Earlier answer"}]},
                        {"role": "user", "content": "Continue"},
                    ],
                    max_output_tokens=10,
                    stream=stream,
                )
            )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[1].content)
        assert counted["messages"] == sent["messages"]
        counted_messages: Final = TypeAdapter(list[dict[str, object]]).validate_python(counted["messages"])
        assert block in TypeAdapter(list[dict[str, object]]).validate_python(counted_messages[1]["content"])
        assert messages == before
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    (
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "url", "url": "https://image.invalid/latest"}}],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "document", "source": {"type": "file", "file_id": "private-file"}}],
                }
            ]
        },
        {"mcp_servers": [{"type": "url", "name": "remote", "url": "https://mcp.invalid"}]},
        {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        {
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
    ),
)
async def test_claude_native_media_tools_and_options_are_counted_and_forwarded(
    monkeypatch: pytest.MonkeyPatch, extra: Mapping[str, object]
) -> None:
    transport: Final = CountTransport({"input_tokens": 75}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[_claude_deployment()], num_retries=0)
    before: Final = copy.deepcopy(extra)
    try:
        await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由参数未标注
            **{"model": "claude-fixture", "messages": _messages(), "max_tokens": 10, **extra}  # pyright: ignore[reportArgumentType]  # 参数化 Anthropic 原生请求
        )
        assert tuple(request.url.path for request in transport.requests) == (
            "/v1/messages/count_tokens",
            "/v1/messages",
        )
        counted: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[0].content)
        sent: Final = TypeAdapter(dict[str, object]).validate_json(transport.requests[1].content)
        assert counted == {key: value for key, value in sent.items() if key not in ("max_tokens", "stream")}
        assert extra == before
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
async def test_claude_tool_arguments_are_not_mistaken_for_protocol_history(monkeypatch: pytest.MonkeyPatch) -> None:
    transport: Final = CountTransport({"input_tokens": 1}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    try:
        await RequestAdmission().async_pre_call_deployment_hook(
            {
                "model": _OPUS,
                "api_key": "test-key",
                "messages": [
                    {"role": "user", "content": "Inspect this object"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "inspect",
                                "input": {"type": "thinking", "signature": "ordinary application data"},
                            }
                        ],
                    },
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Complete"}]},
                ],
                "max_tokens": 10,
                "model_info": {"max_input_tokens": 100},
            },
            CallTypes.aanthropic_messages,
        )
        assert tuple(request.url.path for request in transport.requests) == ("/v1/messages/count_tokens",)
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("composition", ("claude", "mixed", "other"))
@pytest.mark.parametrize("problem", ("image", "missing_limit"))
async def test_filter_local_failure_is_terminal_only_for_claude_candidates(composition: str, problem: str) -> None:
    limit: Final = 0 if problem == "missing_limit" else 100
    claude: Final = _claude_deployment(limit=limit, model="claude-opus-local-preflight-fixture")
    other: Final = {**_deployment(limit), "litellm_params": {"model": "local_preflight_fixture/model"}}
    deployments: Final = [claude, other] if composition == "mixed" else [claude if composition == "claude" else other]
    data: Final[Mapping[str, object]] = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,invalid"}}]
                if problem == "image"
                else "Hello",
            }
        ]
    }
    before: Final = copy.deepcopy(data)
    with pytest.raises(litellm.BadRequestError) as caught:
        await RequestAdmission().async_filter_deployments("fixture", deployments, None, dict(data))
    assert isinstance(caught.value, RequestAdmissionError) is (composition == "claude")
    if composition != "claude":
        assert isinstance(caught.value, litellm.MalformedRequestFieldError) is (problem == "image")
    assert data == before


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", (CallTypes.acompletion, CallTypes.aanthropic_messages))
@pytest.mark.parametrize(
    "overrides",
    ({"extra_body": []}, {"extra_body": "private-validation-marker"}, {"tools": "private-validation-marker"}),
)
async def test_claude_request_parsing_errors_use_terminal_boundary(
    monkeypatch: pytest.MonkeyPatch, call_type: CallTypes, overrides: Mapping[str, object]
) -> None:
    transport: Final = CountTransport({"input_tokens": 1})
    client: Final = _install_count_transport(monkeypatch, transport)
    data: Final = {
        "model": _OPUS,
        "messages": _messages(),
        "max_tokens": 10,
        "api_key": "test-key",
        "model_info": {"max_input_tokens": 100},
        **overrides,
    }
    before: Final = copy.deepcopy(data)
    try:
        with pytest.raises(RequestAdmissionError) as caught:
            await RequestAdmission().async_pre_call_deployment_hook(data, call_type)
        assert caught.value.status_code == 400
        assert "private-validation-marker" not in str(caught.value)
        assert data == before
        assert transport.requests == ()
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
async def test_non_claude_extra_body_keeps_existing_local_admission() -> None:
    counter: Final = AsyncMock(return_value=1)
    result: Final = await RequestAdmission(claude_counter=counter).async_pre_call_deployment_hook(
        {"model": _MODEL, "messages": _messages(), "model_info": {"max_input_tokens": 100}, "extra_body": []},
        CallTypes.acompletion,
    )
    assert result is None
    counter.assert_not_awaited()


@pytest.mark.asyncio
async def test_claude_existing_terminal_rejection_is_preserved_by_boundary() -> None:
    rejection: Final = RequestAdmissionError(message="already rejected", model=_OPUS, llm_provider="anthropic")
    counter: Final = AsyncMock(side_effect=rejection)
    with pytest.raises(RequestAdmissionError) as caught:
        await RequestAdmission(claude_counter=counter).async_pre_call_deployment_hook(
            {"model": _OPUS, "messages": _messages(), "model_info": {"max_input_tokens": 100}},
            CallTypes.acompletion,
        )
    assert caught.value is rejection
    assert counter.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_source", ("deployment", "catalog"))
@pytest.mark.parametrize(
    ("configured_limit", "input_tokens", "allowed"),
    (
        pytest.param(1000000, 899914, True, id="original-899914-remains-allowed"),
        (1000000, 949999, True),
        (1000000, 950000, False),
        (1000000, 950001, False),
        (200000, 199999, True),
        (200000, 200000, False),
    ),
)
async def test_claude_provider_count_respects_950000_ceiling_and_smaller_limits(
    monkeypatch: pytest.MonkeyPatch, limit_source: str, configured_limit: int, input_tokens: int, allowed: bool
) -> None:
    model: Final = "claude-opus-ceiling-fixture"
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            model: {
                "max_input_tokens": configured_limit,
                "max_output_tokens": 8192,
                "litellm_provider": "anthropic",
                "mode": "chat",
            }
        },
    )
    deployment: Final = {
        **_claude_deployment(model=model),
        "model_info": {
            "id": "claude-fixture",
            **({"max_input_tokens": configured_limit} if limit_source == "deployment" else {}),
        },
    }
    transport: Final = CountTransport({"input_tokens": input_tokens}, allow_inference=True)
    client: Final = _install_count_transport(monkeypatch, transport)
    monkeypatch.setattr(litellm, "callbacks", [RequestAdmission()])
    router: Final = litellm.Router(model_list=[deployment], num_retries=0)
    try:
        if allowed:
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由
                model="claude-fixture", messages=_messages(), max_tokens=10
            )
        else:
            with pytest.raises(litellm.ContextWindowExceededError, match=str(min(configured_limit, 950000))) as caught:
                await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # 上游动态路由
                    model="claude-fixture", messages=_messages(), max_tokens=10
                )
            assert isinstance(caught.value, RequestAdmissionError)
        assert tuple(request.url.path for request in transport.requests) == (
            ("/v1/messages/count_tokens", "/v1/messages") if allowed else ("/v1/messages/count_tokens",)
        )
    finally:
        router.reset()
        await client.client.aclose()


@pytest.mark.asyncio
async def test_non_claude_context_limit_can_still_exceed_950000() -> None:
    messages: Final = _messages("token " * 950000)
    count: Final = litellm.token_counter(  # pyright: ignore[reportUnknownMemberType]  # 上游计数器仍有未注解参数
        model=_MODEL, messages=messages
    )
    assert count > 950000
    deployment: Final = _deployment(count + 1)
    assert await RequestAdmission().async_filter_deployments("fixture", [deployment], messages) == [deployment]
    with pytest.raises(litellm.ContextWindowExceededError) as caught:
        await RequestAdmission().async_filter_deployments("fixture", [_deployment(count)], messages)
    assert not isinstance(caught.value, RequestAdmissionError)
