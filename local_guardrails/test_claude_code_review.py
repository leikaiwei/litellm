import json
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Final

import pytest
from pydantic import ConfigDict, TypeAdapter
from pytest_socket import disable_socket, enable_socket

import litellm
from litellm import ModelResponse
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup as LogSetup
from litellm.llms.custom_llm import CustomLLM
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.route_llm_request import (
    route_request,  # pyright: ignore[reportUnknownVariableType]  # 上游路由缺少返回类型
)
from litellm.types.utils import CallTypesLiteral, GenericStreamingChunk
from litellm.utils import custom_llm_setup
from local_guardrails.claude_code_review import ClaudeCodeReviewHook, is_review_request

MODEL_GROUP: Final = "review-test"
PROVIDER: Final = "review_test_provider"
callback: Final = ClaudeCodeReviewHook(MODEL_GROUP)


class ReviewTestProvider(CustomLLM):
    def __init__(self, decision: str) -> None:
        super().__init__()
        self.decision = decision

    async def acompletion(self, model: str, *args: object, **kwargs: object) -> ModelResponse:
        if model == "unavailable":
            raise RuntimeError("审核模型不可用")
        return ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": self.decision}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        )

    async def astreaming(  # pyright: ignore[reportIncompatibleMethodOverride]  # 上游基类把异步生成器误标成协程
        self, model: str, *args: object, **kwargs: object
    ) -> AsyncIterator[GenericStreamingChunk]:
        yield {
            "text": self.decision,
            "is_finished": True,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            "index": 0,
        }


_SYSTEM: Final = (
    "You are a security monitor for autonomous AI coding agents.\n\n"
    "## Context\nReview the proposed action.\n\n"
    "## Output Format\n\n"
    "If the action should be blocked:\n<block>yes</block><reason>reason</reason>\n\n"
    "If the action should be allowed:\n<block>no</block>"
)


@pytest.fixture(autouse=True)
def no_network() -> Iterator[None]:
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()


def _request(**overrides: object) -> dict[str, object]:
    return {
        "model": "original-model",
        "system": [{"type": "text", "text": "client metadata"}, {"type": "text", "text": _SYSTEM}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Conversation transcript"}]},
            {"role": "user", "content": [{"type": "text", "text": "Proposed action"}]},
        ],
        "max_tokens": 2112,
        "stop_sequences": ["</block>"],
        **overrides,
    }


async def _pre_call(
    data: dict[str, object], call_type: CallTypesLiteral = "anthropic_messages"
) -> dict[str, object] | None:
    return await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, call_type)


_MATCHED_CASES: Final[tuple[dict[str, object], ...]] = (
    {},
    {"max_tokens": 10240, "stop_sequences": None},
    {"max_tokens": 64, "system": _SYSTEM, "tools": []},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", _MATCHED_CASES)
async def test_recognized_review_routes_to_reviewer_without_mutating_original(overrides: dict[str, object]) -> None:
    data: Final = _request(
        litellm_metadata={"request_tag": "keep", "spend_logs_metadata": {"audit_tag": "keep"}}, **overrides
    )
    original: Final = json.dumps(data, sort_keys=True)
    result: Final = await _pre_call(data)
    assert result is not None
    assert result["model"] == MODEL_GROUP
    assert result["system"] == data["system"]
    assert result["messages"] == data["messages"]
    assert result["max_tokens"] == data["max_tokens"]
    assert result["cache"] == {"no-cache": True, "no-store": True}
    assert result["caching"] is False
    assert result["disable_fallbacks"] is True
    expected_audit: Final = {
        "audit_tag": "keep",
        "claude_code_review": "model_review",
        "claude_code_review_requested_model": "original-model",
    }
    result_metadata: Final = TypeAdapter(dict[str, object]).validate_python(result["litellm_metadata"])
    assert result_metadata == {"request_tag": "keep", "spend_logs_metadata": expected_audit}
    logged: Final = LogSetup.get_standard_logging_metadata(  # pyright: ignore[reportUnknownMemberType]  # 上游可选参数有裸 dict
        result_metadata
    )
    assert logged["spend_logs_metadata"] == expected_audit
    assert json.dumps(data, sort_keys=True) == original


_UNMATCHED_CASES: Final[tuple[dict[str, object], ...]] = (
    {"tools": [{"name": "Bash", "input_schema": {"type": "object"}}]},
    {"system": "You are a coding assistant", "messages": [{"role": "user", "content": _SYSTEM}]},
    {"system": [{"type": "text", "text": f"Quoted documentation:\n{_SYSTEM}"}]},
    {"system": _SYSTEM.replace("<block>no</block>", "<severity>0</severity>")},
    {"system": _SYSTEM.replace("## Output Format", "Example output")},
    {"stop_sequences": ["</severity>"]},
    {"messages": []},
    {"messages": [{"role": "assistant", "content": "Ordinary conversation"}]},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
    {"system": [{"type": "text", "text": _SYSTEM}, {"type": "image"}]},
    {"messages": "malformed"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", _UNMATCHED_CASES)
async def test_unrecognized_requests_remain_untouched(overrides: dict[str, object]) -> None:
    data: Final = _request(**overrides)
    assert not is_review_request(data)
    assert await _pre_call(data) is None


@pytest.mark.asyncio
async def test_chat_completion_with_same_text_is_not_intercepted() -> None:
    assert await _pre_call(_request(), "acompletion") is None


@pytest.mark.asyncio
async def test_unconfigured_hook_preserves_original_request() -> None:
    data: Final = _request()
    original: Final = json.dumps(data, sort_keys=True)
    assert (
        await ClaudeCodeReviewHook().async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "anthropic_messages")
        is None
    )
    assert json.dumps(data, sort_keys=True) == original


@pytest.fixture(params=("<block>no</block>", "<block>yes</block><reason>Unsafe action</reason>"))
def local_provider(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    decision: Final = TypeAdapter(str).validate_python(request.param)
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": PROVIDER, "custom_handler": ReviewTestProvider(decision)}]
    )
    monkeypatch.setattr(litellm, "_custom_providers", [])
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    custom_llm_setup()
    yield decision


@pytest.mark.asyncio
async def test_messages_response_uses_local_provider_with_reported_usage(local_provider: str) -> None:
    response: Final = await litellm.anthropic_messages(
        model=f"{PROVIDER}/review",
        messages=[{"role": "user", "content": "local response"}],
        max_tokens=64,
        cache={"no-cache": True, "no-store": True},
    )
    assert isinstance(response, dict)
    assert response.get("content") == [{"type": "text", "text": local_provider}]
    assert response.get("usage") == {"input_tokens": 12, "output_tokens": 8}
    assert response.get("stop_reason") == "end_turn"


@pytest.mark.asyncio
async def test_messages_stream_uses_standard_events_and_reported_usage(local_provider: str) -> None:
    response: Final = await litellm.anthropic_messages(
        model=f"{PROVIDER}/review",
        messages=[{"role": "user", "content": "local response"}],
        max_tokens=64,
        stream=True,
        cache={"no-cache": True, "no-store": True},
    )
    assert isinstance(response, AsyncIterator)
    byte_chunk: Final = TypeAdapter(bytes)
    chunks: Final = tuple([byte_chunk.validate_python(chunk) async for chunk in response])
    wire: Final = b"".join(chunks).decode()
    event_adapter: Final = TypeAdapter(dict[str, object])
    events: Final = tuple(
        event_adapter.validate_json(line[6:]) for line in wire.splitlines() if line.startswith("data: ")
    )
    assert tuple(event["type"] for event in events) == (
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    )
    assert events[2]["delta"] == {"type": "text_delta", "text": local_provider}
    assert events[4]["usage"] == {"input_tokens": 12, "output_tokens": 8}
    assert events[4]["delta"] == {"stop_reason": "end_turn"}


_EXTERNAL_MODEL: Final = {
    "model_name": MODEL_GROUP,
    "litellm_params": {"model": "openai/unreachable", "api_key": "unused", "api_base": "http://127.0.0.1:1"},
}
_ROUTE_OVERRIDES: Final[tuple[dict[str, object], ...]] = (
    {"user_config": {"model_list": [_EXTERNAL_MODEL], "num_retries": 0}},
    {"model_list": [{**_EXTERNAL_MODEL, "model_name": f"{PROVIDER}/review"}]},
    {"specific_deployment": True},
    {"custom_llm_provider": "openai", "api_base": "http://127.0.0.1:1", "api_key": "unused"},
    {"mock_response": "This must not replace the policy decision"},
    {"router_settings_override": {"fallbacks": [{"*": ["openai/unreachable"]}]}},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", _ROUTE_OVERRIDES)
async def test_route_overrides_cannot_escape_local_provider(local_provider: str, overrides: dict[str, object]) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": f"{PROVIDER}/review"},
            }
        ],
        num_retries=0,
    )
    try:
        data: Final = await _pre_call(_request(**overrides))
        assert data is not None
        pending_response: Final = TypeAdapter(
            Awaitable[object], config=ConfigDict(arbitrary_types_allowed=True)
        ).validate_python(
            await route_request(data=data, llm_router=router, user_model=None, route_type="anthropic_messages")
        )
        response: Final = TypeAdapter(dict[str, object]).validate_python(await pending_response)
        assert response["content"] == [{"type": "text", "text": local_provider}]
        assert response["usage"] == {"input_tokens": 12, "output_tokens": 8}
    finally:
        router.discard()


@pytest.mark.asyncio
async def test_reviewer_failure_does_not_become_allow(local_provider: str) -> None:
    router: Final = litellm.Router(
        model_list=[{"model_name": MODEL_GROUP, "litellm_params": {"model": f"{PROVIDER}/unavailable"}}],
        num_retries=0,
    )
    try:
        data: Final = await _pre_call(_request())
        assert data is not None
        pending: Final = TypeAdapter(
            Awaitable[object], config=ConfigDict(arbitrary_types_allowed=True)
        ).validate_python(
            await route_request(data=data, llm_router=router, user_model=None, route_type="anthropic_messages")
        )
        with pytest.raises(litellm.APIConnectionError, match="审核模型不可用"):
            await pending
    finally:
        router.discard()
