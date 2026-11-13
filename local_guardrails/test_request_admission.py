import base64
import copy
import json
from collections.abc import Iterator, Mapping
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
from litellm.litellm_core_utils.prompt_templates.common_utils import encrypted_reasoning_signature
from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.llms.anthropic.chat import handler as chat_handler
from litellm.llms.anthropic.count_tokens import handler as count_handler
from litellm.llms.compaction import CompactionProtocol
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.responses.litellm_completion_transformation.compaction_items import summary_only_compaction_content
from litellm.router_strategy.complexity_router.context_compaction import compaction_executor
from litellm.types.llms.openai import AllMessageValues
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
