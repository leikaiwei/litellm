import os
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

_IDENTITY: Final = "You are a security monitor for autonomous AI coding agents"
_BLOCKS: Final = TypeAdapter(tuple[dict[str, object], ...])
_METADATA: Final = TypeAdapter(dict[str, object])
_ROUTING_OVERRIDES: Final = frozenset(
    {
        "user_config",
        "model_list",
        "router_settings_override",
        "specific_deployment",
        "custom_llm_provider",
        "api_base",
        "base_url",
        "api_key",
        "model_info",
        "mock_response",
    }
)


def _texts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        blocks: Final = _BLOCKS.validate_python(value)
    except ValidationError:
        return ()
    texts: Final = tuple(block.get("text") for block in blocks if block.get("type") == "text")
    if len(texts) != len(blocks) or not all(isinstance(text, str) and text for text in texts):
        return ()
    return tuple(text for text in texts if isinstance(text, str))


def is_review_request(data: Mapping[str, object]) -> bool:
    if data.get("tools") not in (None, []):
        return False
    if data.get("stop_sequences") not in (None, [], ["</block>"]):
        return False
    system_texts: Final = _texts(data.get("system"))
    if not any(
        text.startswith(_IDENTITY)
        and "## Output Format" in text
        and "If the action should be allowed:\n<block>no</block>" in text
        and "If the action should be blocked:" in text
        and "<block>yes</block>" in text
        and "<severity>" not in text
        for text in system_texts
    ):
        return False
    try:
        messages: Final = _BLOCKS.validate_python(data.get("messages"))
    except ValidationError:
        return False
    return bool(messages) and all(
        message.get("role") == "user" and bool(_texts(message.get("content"))) for message in messages
    )


class ClaudeCodeReviewHook(CustomLogger):
    def __init__(self, review_model: str | None = None) -> None:
        super().__init__()
        self.review_model = review_model

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | None:
        model: Final = data.get("model")
        if not self.review_model:
            return None
        if call_type not in ("anthropic_messages", "aanthropic_messages") or not is_review_request(data):
            return None
        if not isinstance(model, str):
            return None
        try:
            metadata: Final = _METADATA.validate_python(data.get("litellm_metadata") or {})
            spend_metadata: Final = _METADATA.validate_python(metadata.get("spend_logs_metadata") or {})
        except ValidationError:
            return None
        verbose_proxy_logger.info("Claude Code 独立审查请求分流到审核模型")
        return {
            **{key: value for key, value in data.items() if key not in _ROUTING_OVERRIDES},
            "model": self.review_model,
            "caching": False,
            "cache": {"no-cache": True, "no-store": True},
            "disable_fallbacks": True,
            "litellm_metadata": {
                **metadata,
                "spend_logs_metadata": {
                    **spend_metadata,
                    "claude_code_review": "model_review",
                    "claude_code_review_requested_model": model,
                },
            },
        }


callback: Final = ClaudeCodeReviewHook(os.environ.get("CLAUDE_CODE_REVIEW_MODEL"))
