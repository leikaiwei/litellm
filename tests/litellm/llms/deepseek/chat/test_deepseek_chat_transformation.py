"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result


class TestDeepSeekFillReasoningContent:
    """测试 reasoning_content 回传逻辑"""

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    def test_placeholder_injected_when_absent(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = self.config.fill_reasoning_content(messages)
        assert result[1]["reasoning_content"] == " "
        # 原始消息不应被修改
        assert "reasoning_content" not in messages[1]

    def test_existing_reasoning_content_preserved(self):
        messages = [
            {"role": "assistant", "content": "hi", "reasoning_content": "I thought about it"},
        ]
        result = self.config.fill_reasoning_content(messages)
        assert result[0] is messages[0]  # 同一对象，未拷贝
        assert result[0]["reasoning_content"] == "I thought about it"

    def test_provider_specific_fields_promoted(self):
        messages = [
            {
                "role": "assistant",
                "content": "hi",
                "provider_specific_fields": {"reasoning_content": "stored reasoning"},
            },
        ]
        result = self.config.fill_reasoning_content(messages)
        assert result[0]["reasoning_content"] == "stored reasoning"
        assert "reasoning_content" not in result[0]["provider_specific_fields"]

    def test_non_assistant_messages_untouched(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "you are helpful"},
        ]
        result = self.config.fill_reasoning_content(messages)
        assert result[0] is messages[0]
        assert result[1] is messages[1]
        assert "reasoning_content" not in result[0]
        assert "reasoning_content" not in result[1]

    def test_assistant_with_tool_calls_also_patched(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        ]
        result = self.config.fill_reasoning_content(messages)
        assert result[0]["reasoning_content"] == " "


class TestDeepSeekIsThinkingEnabled:
    """测试 _is_thinking_enabled 判断逻辑"""

    def test_reasoning_model_returns_true(self):
        assert DeepSeekChatConfig._is_thinking_enabled("deepseek/deepseek-reasoner", {}) is True

    def test_dynamic_thinking_param_returns_true(self):
        assert DeepSeekChatConfig._is_thinking_enabled(
            "deepseek-chat", {"thinking": {"type": "enabled"}}
        ) is True

    def test_no_thinking_returns_false(self):
        assert DeepSeekChatConfig._is_thinking_enabled("deepseek-chat", {}) is False

    def test_thinking_disabled_returns_false(self):
        assert DeepSeekChatConfig._is_thinking_enabled(
            "deepseek-chat", {"thinking": {"type": "disabled"}}
        ) is False


class TestDeepSeekTransformRequestReasoning:
    """测试 transform_request 中 reasoning_content 填充的端到端流程"""

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    def test_reasoning_model_fills_content(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "user", "content": "continue"},
        ]
        result = self.config.transform_request(
            model="deepseek/deepseek-reasoner",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant_msg = result["messages"][1]
        assert assistant_msg["reasoning_content"] == " "

    def test_dynamic_thinking_fills_content(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "user", "content": "continue"},
        ]
        result = self.config.transform_request(
            model="deepseek-chat",
            messages=messages,
            optional_params={"thinking": {"type": "enabled"}},
            litellm_params={},
            headers={},
        )
        assistant_msg = result["messages"][1]
        assert assistant_msg["reasoning_content"] == " "

    def test_non_thinking_model_no_fill(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        result = self.config.transform_request(
            model="deepseek-chat",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant_msg = result["messages"][1]
        assert "reasoning_content" not in assistant_msg
