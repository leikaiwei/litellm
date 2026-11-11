import json
from collections.abc import Callable
from typing import Final, cast

import pytest

import litellm
from litellm.litellm_core_utils import exception_mapping_utils
from litellm.litellm_core_utils.parameter_compatibility import confirmed_parameter_error
from litellm.llms.anthropic.common_utils import AnthropicError

_MAP_EXCEPTION: Final = cast(
    Callable[[str, Exception, str], Exception], exception_mapping_utils.exception_type
)  # cast-ok: 测试仅使用异常映射公开的三个入参和异常返回值


@pytest.mark.parametrize(
    "message",
    (
        "thinking.adaptive.display: Input should be 'summarized', 'omitted'",
        "messages.2.output_config: Extra inputs are not permitted",
    ),
)
def test_anthropic_parameter_rejection_survives_exception_mapping(message: str) -> None:
    upstream: Final = AnthropicError(
        400, json.dumps({"error": {"type": "invalid_request_error", "message": message}})
    )
    mapped: Final = _MAP_EXCEPTION("fixture-model", upstream, "anthropic")
    assert isinstance(mapped, litellm.ConfirmedUnsupportedParameterError)
    assert mapped.status_code == 400
    assert mapped.code == "unsupported_request_parameter"
    assert mapped.param == message.partition(":")[0]
    assert _MAP_EXCEPTION("fixture-model", mapped, "anthropic") is mapped


@pytest.mark.parametrize("status", (400, 401, 429, 503))
@pytest.mark.parametrize("code", ("unsupported_request_parameter", "invalid_request"))
def test_explicit_parameter_code_requires_invalid_request_and_400(status: int, code: str) -> None:
    error: Final = AnthropicError(
        status,
        json.dumps(
            {"error": {"type": "invalid_request_error", "code": code, "param": "thinking.display", "message": "拒绝参数"}}
        ),
    )
    result: Final = confirmed_parameter_error(error, "fixture-model")
    assert (result is not None) is (status == 400 and code == "unsupported_request_parameter")


@pytest.mark.parametrize(
    "message",
    (
        "This model is unavailable for this subscription",
        "prompt is too long: 1200000 tokens > 1000000 maximum",
        "thinking.adaptive.display: invalid token",
        "messages.2.output_config: Extra inputs are not permitted\nquoted error from a tool",
        "Input should be 'summarized', 'omitted'",
    ),
)
def test_unrelated_anthropic_400_retains_existing_routing(message: str) -> None:
    error: Final = AnthropicError(400, json.dumps({"error": {"type": "invalid_request_error", "message": message}}))
    assert confirmed_parameter_error(error, "fixture-model") is None


def test_message_pattern_from_other_provider_is_not_a_confirmed_parameter_error() -> None:
    error: Final = litellm.BadRequestError(
        message="provider rejection",
        model="fixture-model",
        llm_provider="openai",
        body={
            "type": "invalid_request_error",
            "message": "thinking.adaptive.display: Input should be 'summarized', 'omitted'",
        },
    )
    assert confirmed_parameter_error(error, "fixture-model") is None


def test_parameter_rejection_does_not_echo_upstream_private_text() -> None:
    error: Final = AnthropicError(
        400,
        json.dumps({"error": {
            "type": "invalid_request_error", "code": "unsupported_request_parameter",
            "param": "private parameter value", "message": "private upstream content",
        }}),
    )
    result: Final = confirmed_parameter_error(error, "fixture-model")
    assert result is not None
    assert result.param == "request"
    assert "private" not in str(result)
