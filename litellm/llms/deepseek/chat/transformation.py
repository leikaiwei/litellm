"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.bedrock.common_utils import (
    normalize_json_schema_custom_types_to_object,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to DeepSeek params.

        Handles `thinking` and `reasoning_effort` parameters for DeepSeek reasoner models.
        DeepSeek only supports `{"type": "enabled"}` - no budget_tokens like Anthropic.

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Pop thinking/reasoning_effort from optional_params first (parent may have added them)
        # Then re-add only if valid for DeepSeek
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Handle thinking parameter - only accept {"type": "enabled"}
        if thinking_value is not None:
            if (
                isinstance(thinking_value, dict)
                and thinking_value.get("type") == "enabled"
            ):
                # DeepSeek only accepts {"type": "enabled"}, ignore budget_tokens
                optional_params["thinking"] = {"type": "enabled"}

        # Handle reasoning_effort - map to thinking enabled
        elif reasoning_effort is not None and reasoning_effort != "none":
            optional_params["thinking"] = {"type": "enabled"}

        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # DeepSeek 不支持非标准 JSON Schema 类型（如 Anthropic 的 "custom"）
        tools = optional_params.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                func = tool.get("function")
                if not isinstance(func, dict):
                    continue
                params = func.get("parameters")
                if isinstance(params, dict):
                    normalize_json_schema_custom_types_to_object(params)
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]: ...

    def _ensure_reasoning_content_on_assistant_messages(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        DeepSeek V4 thinking mode 要求每条 assistant 历史消息都携带 reasoning_content，
        缺失则返回 HTTP 400。

        Anthropic 适配器将 thinking 内容存为 thinking_blocks 而非 reasoning_content，
        需要同时检测两种格式，并将 thinking_blocks 转换为 reasoning_content。
        """
        thinking_active = any(
            msg.get("role") == "assistant"
            and ("reasoning_content" in msg or msg.get("thinking_blocks"))
            for msg in messages
        )
        if not thinking_active:
            return messages
        for message in messages:
            if message.get("role") != "assistant":
                continue
            # 将 thinking_blocks 转换为 reasoning_content
            if "reasoning_content" not in message:
                thinking_blocks = message.get("thinking_blocks") or []
                if thinking_blocks:
                    message["reasoning_content"] = " ".join(
                        block.get("thinking") or ""
                        for block in thinking_blocks
                        if isinstance(block, dict)
                        and block.get("type") == "thinking"
                    )
                else:
                    message["reasoning_content"] = ""
            # DeepSeek 不识别 thinking_blocks 字段
            message.pop("thinking_blocks", None)  # type: ignore
        return messages

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        DeepSeek does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        messages = self._ensure_reasoning_content_on_assistant_messages(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.deepseek.com/beta"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
