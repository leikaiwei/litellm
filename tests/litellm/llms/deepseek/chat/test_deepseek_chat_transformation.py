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


class TestDeepSeekReasoningContentInjection:
    """Test reasoning_content injection for multi-turn thinking mode conversations."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    def test_injects_empty_string_when_prior_turn_has_reasoning_content(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "Let me think..."},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Sure"},
        ]
        result = self.config._ensure_reasoning_content_on_assistant_messages(messages)
        assert result[3]["reasoning_content"] == ""

    def test_no_injection_when_no_prior_reasoning_content(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Sure"},
        ]
        result = self.config._ensure_reasoning_content_on_assistant_messages(messages)
        assert "reasoning_content" not in result[1]
        assert "reasoning_content" not in result[3]

    def test_preserves_existing_reasoning_content(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "Deep thought"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Sure", "reasoning_content": "More thought"},
        ]
        result = self.config._ensure_reasoning_content_on_assistant_messages(messages)
        assert result[1]["reasoning_content"] == "Deep thought"
        assert result[3]["reasoning_content"] == "More thought"

    def test_skips_non_assistant_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "thinking"},
            {"role": "user", "content": "Follow up"},
            {"role": "tool", "content": "tool result", "tool_call_id": "123"},
        ]
        result = self.config._ensure_reasoning_content_on_assistant_messages(messages)
        assert "reasoning_content" not in result[0]
        assert "reasoning_content" not in result[2]
        assert "reasoning_content" not in result[3]

    def test_no_injection_on_plain_assistant_without_prior_reasoning(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self.config._ensure_reasoning_content_on_assistant_messages(messages)
        assert "reasoning_content" not in result[1]

    def test_transform_messages_injects_reasoning_content_when_prior_turn_has_it(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "Let me think"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Sure"},
        ]
        result = self.config._transform_messages(messages, "deepseek-v4-pro", is_async=False)
        assert result[3]["reasoning_content"] == ""
        assert result[1]["reasoning_content"] == "Let me think"

    def test_transform_messages_does_not_inject_in_non_thinking_mode(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Follow up"},
        ]
        result = self.config._transform_messages(messages, "deepseek-v4-pro", is_async=False)
        assert "reasoning_content" not in result[1]
