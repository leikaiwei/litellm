import copy
from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from pydantic import TypeAdapter
from pytest_socket import disable_socket, enable_socket

import litellm
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import proxy_exception_from_http_exception
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import CallTypesLiteral
from local_guardrails.parameter_compatibility import ParameterCompatibilityGuardrail

_MODEL: Final = "claude-sonnet-5-5"


@pytest.fixture(autouse=True)
def no_network() -> Iterator[None]:
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ("anthropic_messages", "acompletion", "aresponses"))
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("drop_params", (False, True))
async def test_known_incompatibility_is_rejected_without_changing_request(
    call_type: CallTypesLiteral, nested: bool, drop_params: bool
) -> None:
    thinking: Final = {"thinking": {"type": "adaptive", "display": "updates"}}
    data: Final[dict[str, object]] = {
        "model": _MODEL,
        "drop_params": drop_params,
        **({"extra_body": thinking} if nested else thinking),
    }
    original: Final = copy.deepcopy(data)
    guard: Final = ParameterCompatibilityGuardrail(blocked_models=[_MODEL])
    with pytest.raises(HTTPException) as error:
        await guard.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, call_type)
    assert error.value.status_code == 400
    assert TypeAdapter(dict[str, object]).validate_python(error.value.detail)["param"] == "thinking.display"
    assert proxy_exception_from_http_exception(error.value, {}).param == "thinking.display"
    assert data == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "thinking"),
    (
        (_MODEL, None),
        (_MODEL, {"display": "summarized"}),
        (_MODEL, {"display": "omitted"}),
        ("other-model", {"display": "updates"}),
    ),
)
async def test_other_requests_are_unchanged(model: str, thinking: object) -> None:
    data: Final[dict[str, object]] = {"model": model, "thinking": thinking}
    original: Final = copy.deepcopy(data)
    guard: Final = ParameterCompatibilityGuardrail(blocked_models=[_MODEL])
    assert await guard.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "anthropic_messages") is None
    assert data == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_http_pre_call_rejects_before_count_or_routing(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    guard: Final = ParameterCompatibilityGuardrail(blocked_models=[_MODEL], guardrail_name="parameter-compatibility")
    monkeypatch.setattr(litellm, "callbacks", [guard])
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    downstream: Final = AsyncMock(return_value={"accepted": True})
    app: Final = FastAPI()

    async def messages(data: dict[str, object]) -> object:
        await logging.pre_call_hook(  # pyright: ignore[reportUnknownMemberType]  # 上游 hook 的请求 dict 未标注
            UserAPIKeyAuth(), data, "anthropic_messages", guardrails_only=True
        )
        await downstream(data)
        return {"accepted": True}

    app.add_api_route("/v1/messages", messages, methods=["POST"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        response: Final = await client.post(
            "/v1/messages",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "thinking": {"type": "adaptive", "display": "updates"},
                "stream": stream,
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "unsupported_request_parameter"
        downstream.assert_not_awaited()
        allowed: Final = await client.post("/v1/messages", json={"model": _MODEL, "stream": stream})
        assert allowed.status_code == 200
        downstream.assert_awaited_once()
