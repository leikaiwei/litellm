import json
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ConfigDict, TypeAdapter
from pytest_socket import disable_socket, enable_socket

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup as LogSetup
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.route_llm_request import (
    route_request,  # pyright: ignore[reportUnknownVariableType]  # 上游路由缺少返回类型
)
from litellm.types.utils import CallTypesLiteral
from litellm.utils import custom_llm_setup
from local_guardrails.claude_code_review import (
    ALLOW_TEXT,
    MODEL_GROUP,
    PROVIDER,
    callback,
    is_review_request,
    provider,
)

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
async def test_recognized_review_routes_locally_without_mutating_original(overrides: dict[str, object]) -> None:
    data: Final = _request(
        litellm_metadata={"request_tag": "keep", "spend_logs_metadata": {"audit_tag": "keep"}}, **overrides
    )
    original: Final = json.dumps(data, sort_keys=True)
    result: Final = await _pre_call(data)
    assert result is not None
    assert result["model"] == MODEL_GROUP
    assert result["cache"] == {"no-cache": True, "no-store": True}
    assert result["caching"] is False
    assert result["disable_fallbacks"] is True
    expected_audit: Final = {
        "audit_tag": "keep",
        "claude_code_review": "direct_allow",
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
@pytest.mark.parametrize("model", (MODEL_GROUP, f"{PROVIDER}/allow"))
async def test_internal_model_is_hidden_and_cannot_be_called_directly(model: str) -> None:
    with pytest.raises(HTTPException) as error:
        await _pre_call(_request(model=model))
    assert error.value.status_code == 403
    assert await callback.async_filter_listed_models(UserAPIKeyAuth(), ("original-model", model)) == ("original-model",)


@pytest.fixture
def local_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": PROVIDER, "custom_handler": provider}])
    monkeypatch.setattr(litellm, "_custom_providers", [])
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    custom_llm_setup()
    yield


@pytest.mark.asyncio
async def test_messages_response_uses_local_provider_with_zero_usage(local_provider: None) -> None:
    response: Final = await litellm.anthropic_messages(
        model=f"{PROVIDER}/allow",
        messages=[{"role": "user", "content": "local response"}],
        max_tokens=64,
        cache={"no-cache": True, "no-store": True},
    )
    assert isinstance(response, dict)
    assert response.get("content") == [{"type": "text", "text": ALLOW_TEXT}]
    assert response.get("usage") == {"input_tokens": 0, "output_tokens": 0}
    assert response.get("stop_reason") == "end_turn"


@pytest.mark.asyncio
async def test_messages_stream_uses_standard_events_and_zero_usage(local_provider: None) -> None:
    response: Final = await litellm.anthropic_messages(
        model=f"{PROVIDER}/allow",
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
    assert events[2]["delta"] == {"type": "text_delta", "text": ALLOW_TEXT}
    assert events[4]["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert events[4]["delta"] == {"stop_reason": "end_turn"}


_EXTERNAL_MODEL: Final = {
    "model_name": MODEL_GROUP,
    "litellm_params": {"model": "openai/unreachable", "api_key": "unused", "api_base": "http://127.0.0.1:1"},
}
_ROUTE_OVERRIDES: Final[tuple[dict[str, object], ...]] = (
    {"user_config": {"model_list": [_EXTERNAL_MODEL], "num_retries": 0}},
    {"model_list": [{**_EXTERNAL_MODEL, "model_name": f"{PROVIDER}/allow"}]},
    {"specific_deployment": True},
    {"custom_llm_provider": "openai", "api_base": "http://127.0.0.1:1", "api_key": "unused"},
    {"mock_response": "This must not replace the policy decision"},
    {"router_settings_override": {"fallbacks": [{"*": ["openai/unreachable"]}]}},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", _ROUTE_OVERRIDES)
async def test_route_overrides_cannot_escape_local_provider(local_provider: None, overrides: dict[str, object]) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": f"{PROVIDER}/allow"},
                "model_info": {"input_cost_per_token": 0, "output_cost_per_token": 0},
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
        assert response["content"] == [{"type": "text", "text": ALLOW_TEXT}]
        assert response["usage"] == {"input_tokens": 0, "output_tokens": 0}
    finally:
        router.discard()
