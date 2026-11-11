import re
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.exceptions import ConfirmedUnsupportedParameterError, MidStreamFallbackError

_OBJECT: Final = TypeAdapter(dict[str, object])
_PARAMETER_PATH: Final = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_0-9]+)*")
_PARAMETER_ERROR: Final = re.compile(
    r"(?P<param>(?:thinking|output_config|tool_choice|tools|messages|system|context_management)"
    r"(?:\.[A-Za-z_][A-Za-z_0-9]*|\.\d+)*): "
    r"(?:Input should be '[^\n]+'|Extra inputs are not permitted)"
)


def _structured_error(error: Exception) -> Mapping[str, object] | None:
    body: Final[object] = getattr(error, "body", None)
    response: Final[object] = getattr(error, "response", None)
    message: Final[object] = getattr(error, "message", None)
    try:
        payload: Final = (
            _OBJECT.validate_python(body)
            if body is not None
            else _OBJECT.validate_json(response.content)
            if isinstance(response, httpx.Response) and response.content
            else _OBJECT.validate_json(message)
            if isinstance(message, str)
            else {}
        )
        return _OBJECT.validate_python(payload.get("error", payload))
    except ValidationError:
        return None


def confirmed_parameter_error(error: Exception, model: str) -> ConfirmedUnsupportedParameterError | None:
    if isinstance(error, ConfirmedUnsupportedParameterError):
        return error
    if isinstance(error, MidStreamFallbackError) and error.original_exception is not None:
        return confirmed_parameter_error(error.original_exception, model)
    if getattr(error, "status_code", None) != 400:
        return None
    details: Final = _structured_error(error)
    if details is None or details.get("type") != "invalid_request_error":
        return None
    message: Final = details.get("message")
    if not isinstance(message, str):
        return None
    explicit_code: Final = details.get("code") == "unsupported_request_parameter"
    provider: Final = getattr(error, "llm_provider", None)
    anthropic: Final = provider == "anthropic" or type(error).__module__ == "litellm.llms.anthropic.common_utils"
    match: Final = _PARAMETER_ERROR.fullmatch(message) if anthropic else None
    if not explicit_code and match is None:
        return None
    parameter: Final = details.get("param") if explicit_code else match.group("param") if match is not None else None
    safe_parameter: Final = (
        parameter
        if isinstance(parameter, str) and len(parameter) <= 256 and _PARAMETER_PATH.fullmatch(parameter)
        else "request"
    )
    return ConfirmedUnsupportedParameterError(
        message=f"{safe_parameter} 参数不兼容，请检查参数值和所需 Beta 支持",
        model=model,
        llm_provider=provider if isinstance(provider, str) else "anthropic" if anthropic else "",
        body={"type": "invalid_request_error", "code": "unsupported_request_parameter", "param": safe_parameter},
    )


def validate_anthropic_thinking(data: Mapping[str, object], model: str) -> None:
    thinking: Final = _OBJECT.validate_python(data["thinking"]) if isinstance(data.get("thinking"), Mapping) else {}
    display: Final = thinking.get("display")
    if display is None:
        return
    if display in ("summarized", "omitted", "updates") and thinking.get("type") not in ("disabled", "between_tools"):
        return
    raise ConfirmedUnsupportedParameterError(
        message="thinking.display 参数不兼容，请使用 summarized、omitted 或 updates，disabled/between_tools 不接受 display",
        model=model,
        llm_provider="anthropic",
        body={"type": "invalid_request_error", "code": "unsupported_request_parameter", "param": "thinking.display"},
    )
