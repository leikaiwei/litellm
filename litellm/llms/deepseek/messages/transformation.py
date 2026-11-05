"""
DeepSeek Anthropic-compatible messages transformation config.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field
from litellm.llms.anthropic.common_utils import validated_object_list, validated_object_mapping
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.bedrock.common_utils import (
    normalize_json_schema_custom_types_to_object,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

DEEPSEEK_ANTHROPIC_API_BASE: Final = "https://api.deepseek.com/anthropic"
_DEEPSEEK_CUSTOM_TOOL_ALLOWED_FIELDS: Final = frozenset(
    {
        "cache_control",
        "description",
        "input_schema",
        "name",
    }
)


_PLACEHOLDER_THINKING_BLOCK: Final = MappingProxyType({"type": "thinking", "thinking": " "})
_SCHEMA_ADAPTER: Final = TypeAdapter(dict[str, object])


def _is_deepseek_thinking_block_error(error_text: str) -> bool:
    """
    检测 DeepSeek Anthropic 兼容 /v1/messages 因 assistant 历史里的 thinking 块
    而返回的 400。

    fallback 时（如 qwen/claude 回退到 deepseek）另一模型产的推理内容被原样回放，
    命中两类 400（错误串实测自 api.deepseek.com/anthropic）：
      - "unknown variant `redacted_thinking`, expected one of ..., `thinking`"
        DeepSeek 不认 Anthropic 原生 redacted_thinking 变体
      - "The `content[].thinking` in the thinking mode must be passed back to the API."
        thinking 模式下含 tool_use 的 assistant 消息必须带 thinking 块
    """
    if not error_text:
        return False
    lower: Final = error_text.lower()
    if "redacted_thinking" in lower:
        return True
    if "unknown variant" in lower and "thinking" in lower:
        return True
    return "must be passed back" in lower and "thinking" in lower


def _repair_assistant_message_for_deepseek(message: object) -> object:
    """
    修复单条 assistant 消息使 DeepSeek 接受：
      - redacted_thinking 块 -> 占位 thinking 块（DeepSeek 不认 redacted_thinking）
      - 含 tool_use 但无 thinking 块时 -> 前置占位 thinking 块
        （thinking 模式下 DeepSeek 要求这类消息回传 thinking）
    非 assistant、或 content 非 list 的消息原样返回。
    """
    fields: Final = validated_object_mapping(message)
    if fields is None or fields.get("role") != "assistant":
        return message
    if not isinstance(fields.get("content"), list):
        return message
    content: Final = validated_object_list(fields.get("content")) or ()

    converted: Final = tuple(
        dict(_PLACEHOLDER_THINKING_BLOCK) if message_field(block, "type") == "redacted_thinking" else block
        for block in content
    )
    has_thinking: Final = any(message_field(block, "type") == "thinking" for block in converted)
    has_tool_use: Final = any(message_field(block, "type") == "tool_use" for block in converted)
    new_content: Final = (
        [dict(_PLACEHOLDER_THINKING_BLOCK), *converted] if has_tool_use and not has_thinking else list(converted)
    )
    return {**fields, "content": new_content}


def _sanitize_tool_for_deepseek(tool: object) -> object:
    fields: Final = validated_object_mapping(tool)
    if fields is None or fields.get("type") not in (None, "custom") or "name" not in fields:
        return tool
    sanitized: Final = {key: value for key, value in fields.items() if key in _DEEPSEEK_CUSTOM_TOOL_ALLOWED_FIELDS}
    if not isinstance(sanitized.get("input_schema"), dict):
        return sanitized
    input_schema: Final = _SCHEMA_ADAPTER.validate_python(sanitized["input_schema"])
    normalize_json_schema_custom_types_to_object(input_schema)
    return {**sanitized, "input_schema": input_schema}


class DeepSeekAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DeepSeek exposes an Anthropic-compatible Messages API at
    https://api.deepseek.com/anthropic.

    It accepts the native Anthropic Messages conversation shape, including
    thinking blocks in assistant history, but rejects Anthropic's explicit
    custom-tool discriminator (`{"type": "custom"}`).
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "deepseek"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("DEEPSEEK_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return (
            api_base
            or get_secret_str("DEEPSEEK_ANTHROPIC_API_BASE")
            or get_secret_str("DEEPSEEK_API_BASE")
            or DEEPSEEK_ANTHROPIC_API_BASE
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: Sequence[object],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        dynamic_api_key: Final = self.get_api_key(api_key=api_key)

        if "x-api-key" not in headers and "authorization" not in headers and dynamic_api_key is not None:
            headers["x-api-key"] = dynamic_api_key

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        return headers, api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages") and "/anthropic/" in base_url:
            return base_url
        base_url = base_url.removesuffix("/v1/messages")
        base_url = base_url.removesuffix("/v1")
        base_url = base_url.removesuffix("/beta")

        if not base_url.endswith("/anthropic") and "/anthropic/" not in base_url:
            base_url = f"{base_url}/anthropic"

        return f"{base_url}/v1/messages"

    @staticmethod
    def _sanitize_tools_for_deepseek(tools: object) -> object:
        tool_items: Final = validated_object_list(tools)
        if not isinstance(tools, list):
            return tools
        return [_sanitize_tool_for_deepseek(tool) for tool in tool_items or ()]

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        anthropic_messages_request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if "tools" in anthropic_messages_request:
            anthropic_messages_request["tools"] = self._sanitize_tools_for_deepseek(anthropic_messages_request["tools"])
        return anthropic_messages_request

    def should_retry_anthropic_messages_on_http_error(
        self, e: httpx.HTTPStatusError, litellm_params: Mapping[str, object], request_body: object = None
    ) -> bool:
        """
        除父类的无效签名恢复外，额外覆盖 DeepSeek 因外来 thinking 块导致的 400。

        fallback 到 DeepSeek 时（qwen/claude -> deepseek），assistant 历史带着另一
        模型产的 thinking 块（含 DeepSeek 不认的 redacted_thinking），或含 tool_use
        却缺 thinking 块，DeepSeek 拒绝；修复 thinking 块后重试即可恢复。
        请求体透传父类，保留调用方明确要求的 thinking 绑定错误。
        """
        if super().should_retry_anthropic_messages_on_http_error(
            e=e, litellm_params=dict(litellm_params), request_body=request_body
        ):
            return True
        return e.response.status_code == 400 and _is_deepseek_thinking_block_error(e.response.text)

    def transform_anthropic_messages_request_on_http_error(
        self,
        e: httpx.HTTPStatusError,
        request_data: dict[str, object],  # mutable-ok: HTTP 重试调用方丢弃返回值，要求原地修复请求
    ) -> dict[str, object]:  # mutable-ok: 保持父类返回同一可变请求对象的接口
        """
        命中 DeepSeek thinking 块 400 时修复 assistant 历史后重试：redacted_thinking
        转为占位 thinking 块，含 tool_use 却无 thinking 块的消息前置占位 thinking 块。
        父类的签名错误场景由 super() 处理。
        """
        super().transform_anthropic_messages_request_on_http_error(e=e, request_data=request_data)
        if e.response.status_code == 400 and _is_deepseek_thinking_block_error(e.response.text):
            messages: Final = validated_object_list(request_data.get("messages"))
            if messages is not None:
                request_data["messages"] = [  # rebind-ok: HTTP 重试调用方丢弃返回值，要求原地修复请求
                    _repair_assistant_message_for_deepseek(m) for m in messages
                ]
        return request_data
