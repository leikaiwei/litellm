"""
DeepSeek Anthropic-compatible messages transformation config.
"""

from typing import Any, Final

import httpx

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.bedrock.common_utils import (
    normalize_json_schema_custom_types_to_object,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

DEEPSEEK_ANTHROPIC_API_BASE = "https://api.deepseek.com/anthropic"
_DEEPSEEK_CUSTOM_TOOL_ALLOWED_FIELDS = {
    "cache_control",
    "description",
    "input_schema",
    "name",
}


_PLACEHOLDER_THINKING_BLOCK = {"type": "thinking", "thinking": " "}


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
    lower = error_text.lower()
    if "redacted_thinking" in lower:
        return True
    if "unknown variant" in lower and "thinking" in lower:
        return True
    if "must be passed back" in lower and "thinking" in lower:
        return True
    return False


def _repair_assistant_message_for_deepseek(message: Any) -> Any:
    """
    修复单条 assistant 消息使 DeepSeek 接受：
      - redacted_thinking 块 -> 占位 thinking 块（DeepSeek 不认 redacted_thinking）
      - 含 tool_use 但无 thinking 块时 -> 前置占位 thinking 块
        （thinking 模式下 DeepSeek 要求这类消息回传 thinking）
    非 assistant、或 content 非 list 的消息原样返回。
    """
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return message
    content = message.get("content")
    if not isinstance(content, list):
        return message

    converted = tuple(
        dict(_PLACEHOLDER_THINKING_BLOCK)
        if isinstance(block, dict) and block.get("type") == "redacted_thinking"
        else block
        for block in content
    )
    has_thinking = any(isinstance(block, dict) and block.get("type") == "thinking" for block in converted)
    has_tool_use = any(isinstance(block, dict) and block.get("type") == "tool_use" for block in converted)
    new_content = (
        [dict(_PLACEHOLDER_THINKING_BLOCK), *converted] if has_tool_use and not has_thinking else list(converted)
    )
    return {**message, "content": new_content}


def _repair_thinking_blocks_for_deepseek(request_data: dict) -> None:
    """
    原地重建 request_data["messages"]，逐条修复 assistant 消息的 thinking 块。
    handler 依赖对 request_data 的原地修改（返回值被丢弃）。
    """
    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return
    request_data["messages"] = [_repair_assistant_message_for_deepseek(m) for m in messages]


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
        messages: list[Any],
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
    def _sanitize_tools_for_deepseek(tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools

        sanitized_tools: Final = []
        for tool in tools:
            if not isinstance(tool, dict):
                sanitized_tools.append(tool)
                continue

            tool_type = tool.get("type")
            if isinstance(tool_type, str) and tool_type.startswith("web_search_"):
                sanitized_tools.append(tool)
                continue

            if tool_type in (None, "custom") and "name" in tool:
                sanitized_tool = {
                    key: value for key, value in tool.items() if key in _DEEPSEEK_CUSTOM_TOOL_ALLOWED_FIELDS
                }
                input_schema = sanitized_tool.get("input_schema")
                if isinstance(input_schema, dict):
                    # DeepSeek 不支持 Anthropic custom JSON Schema 类型。
                    normalize_json_schema_custom_types_to_object(input_schema)
                sanitized_tools.append(sanitized_tool)
            else:
                sanitized_tools.append(tool)
        return sanitized_tools

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

    def should_retry_anthropic_messages_on_http_error(self, e: httpx.HTTPStatusError, litellm_params: dict) -> bool:
        """
        除父类的无效签名恢复外，额外覆盖 DeepSeek 因外来 thinking 块导致的 400。

        fallback 到 DeepSeek 时（qwen/claude -> deepseek），assistant 历史带着另一
        模型产的 thinking 块（含 DeepSeek 不认的 redacted_thinking），或含 tool_use
        却缺 thinking 块，DeepSeek 拒绝；修复 thinking 块后重试即可恢复。检测只能基于
        错误文本，因为此钩子拿不到请求体。
        """
        if super().should_retry_anthropic_messages_on_http_error(e=e, litellm_params=litellm_params):
            return True
        return e.response.status_code == 400 and _is_deepseek_thinking_block_error(e.response.text)

    def transform_anthropic_messages_request_on_http_error(self, e: httpx.HTTPStatusError, request_data: dict) -> dict:
        """
        命中 DeepSeek thinking 块 400 时修复 assistant 历史后重试：redacted_thinking
        转为占位 thinking 块，含 tool_use 却无 thinking 块的消息前置占位 thinking 块。
        父类的签名错误场景由 super() 处理。
        """
        request_data = super().transform_anthropic_messages_request_on_http_error(e=e, request_data=request_data)
        if e.response.status_code == 400 and _is_deepseek_thinking_block_error(e.response.text):
            _repair_thinking_blocks_for_deepseek(request_data)
        return request_data
