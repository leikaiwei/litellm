"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.bedrock.common_utils import (
    normalize_json_schema_custom_types_to_object,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_reasoning

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

    @staticmethod
    def _normalize_openai_tool_schemas(optional_params: dict) -> None:
        tools = optional_params.get("tools")
        if not isinstance(tools, list):
            return

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function")
            if not isinstance(func, dict):
                continue
            params = func.get("parameters")
            if isinstance(params, dict):
                # DeepSeek 不支持 Anthropic custom JSON Schema 类型。
                normalize_json_schema_custom_types_to_object(params)

    def _fill_reasoning_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        DeepSeek thinking mode requires `reasoning_content` to be passed back on
        every assistant message in multi-turn conversations. If it is missing,
        the API returns:
          "The reasoning_content in the thinking mode must be passed back to the API."

        For each assistant message that is missing `reasoning_content`:
          1. Promote it from `thinking_blocks` if present.
          2. Promote it from `provider_specific_fields["reasoning_content"]` if present
             (LiteLLM stores provider-specific response fields there).
          3. Otherwise inject a single space — the minimum value the API accepts.
        """
        result: List[AllMessageValues] = []
        for msg in messages:
            if msg.get("role") != "assistant":
                result.append(msg)
                continue

            patched = dict(cast(dict, msg))
            thinking_blocks = patched.pop("thinking_blocks", None)
            if not patched.get("reasoning_content"):
                reasoning_content = None
                reasoning_content_from_provider = False
                if isinstance(thinking_blocks, list):
                    reasoning_content = " ".join(
                        block.get("thinking") or ""
                        for block in thinking_blocks
                        if isinstance(block, dict) and block.get("type") == "thinking"
                    )

                provider_fields = patched.get("provider_specific_fields") or {}
                if not reasoning_content and isinstance(provider_fields, dict):
                    reasoning_content = provider_fields.get("reasoning_content")
                    reasoning_content_from_provider = bool(reasoning_content)

                if reasoning_content:
                    patched["reasoning_content"] = reasoning_content
                    if reasoning_content_from_provider:
                        cleaned = dict(provider_fields)
                        cleaned.pop("reasoning_content", None)
                        patched["provider_specific_fields"] = cleaned
                else:
                    litellm.verbose_logger.warning(
                        "DeepSeek thinking mode: assistant message is missing "
                        "`reasoning_content` and none was saved in "
                        "`provider_specific_fields`. A single-space placeholder "
                        "is being injected to satisfy API validation, but the "
                        "model will receive a blank reasoning chain for this turn, "
                        "which may silently degrade multi-turn response quality. "
                        "Preserve `reasoning_content` from the original assistant "
                        "response when building multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
            result.append(cast(AllMessageValues, patched))
        return result

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

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        DeepSeek does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _thinking_mode_active(self, model: str, optional_params: dict) -> bool:
        """
        Returns True only when thinking mode is actually active for this request:
          - model supports reasoning (capability check)
          - user explicitly passed thinking={"type": "enabled"} (opt-in check)
        """
        return (
            supports_reasoning(model=model, custom_llm_provider="deepseek")
            and (optional_params.get("thinking") or {}).get("type") == "enabled"
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Ensures `reasoning_content` is forwarded on assistant messages for
        multi-turn thinking-mode conversations (issue #28045).

        Only runs when thinking mode is actually active - guarded by both
        supports_reasoning() (model capability) and optional_params["thinking"]
        (user explicitly enabled it), preventing spurious injection on models
        like deepseek-v3.2 that support thinking as opt-in but not always-on.
        """
        if self._thinking_mode_active(model=model, optional_params=optional_params):
            messages = self._fill_reasoning_content(messages)
        self._normalize_openai_tool_schemas(optional_params)
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Async equivalent of transform_request — applies the same reasoning_content
        fix for multi-turn thinking-mode conversations.
        """
        if self._thinking_mode_active(model=model, optional_params=optional_params):
            messages = self._fill_reasoning_content(messages)
        self._normalize_openai_tool_schemas(optional_params)
        return await super().async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
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
