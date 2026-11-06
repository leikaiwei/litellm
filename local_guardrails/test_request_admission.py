import base64
import copy
import json
from collections.abc import Iterator, Mapping
from io import BytesIO
from typing import Final, cast

import pytest
from PIL import Image
from pytest_socket import disable_socket, enable_socket

import litellm
from litellm import ModelResponse
from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.llms.compaction import CompactionProtocol
from litellm.llms.custom_llm import CustomLLM
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
    {"input": [{"type": "reasoning", "summary": [], "encrypted_content": "opaque"}]},
    {
        "input": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Short summary"}],
                "encrypted_content": "opaque",
            }
        ]
    },
)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", _UNVERIFIABLE_RESPONSES)
async def test_responses_hidden_history_is_not_silently_counted_as_zero(extra: dict[str, object]) -> None:
    with pytest.raises(litellm.BadRequestError, match=r"完整|新会话"):
        await _HOOK.async_filter_deployments("group", [_deployment(10000)], None, {"input": "hello", **extra})


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
