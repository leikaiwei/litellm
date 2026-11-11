from collections.abc import Mapping, Sequence
from typing import ClassVar, Final

from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral

_OBJECT: Final = TypeAdapter(dict[str, object])
_MODELS: Final = TypeAdapter(tuple[str, ...])
_CALLS: Final = frozenset(("completion", "acompletion", "anthropic_messages", "responses", "aresponses"))


class _ParameterRejection(HTTPException):
    param: ClassVar[str] = "thinking.display"


def _uses_updates(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return _OBJECT.validate_python(value).get("display") == "updates"


class ParameterCompatibilityGuardrail(CustomGuardrail):
    def __init__(
        self,
        blocked_models: Sequence[str],
        guardrail_name: str | None = None,
        default_on: bool = True,
        **kwargs: object,
    ) -> None:
        self._blocked_models: Final = frozenset(_MODELS.validate_python(blocked_models))
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # 上游 guardrail 构造器的 kwargs 未标注
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=GuardrailEventHooks.pre_call,
            default_on=default_on,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> None:
        model: Final = data.get("model")
        if call_type not in _CALLS or not isinstance(model, str) or model not in self._blocked_models:
            return
        extra_body: Final = data.get("extra_body")
        extra: Final = _OBJECT.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}
        # 入口直接拒绝，不让计数请求进入 Router 的重试和降级链
        if _uses_updates(data.get("thinking")) or _uses_updates(extra.get("thinking")):
            raise _ParameterRejection(
                status_code=400,
                detail={
                    "type": "invalid_request_error",
                    "param": "thinking.display",
                    "code": "unsupported_request_parameter",
                    "message": "当前模型组的上游计数链路不支持 thinking.display=updates，请改用 summarized 或 omitted",
                },
            )
