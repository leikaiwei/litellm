import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    DEEPSEEK_ANTHROPIC_API_BASE,
    AnthropicMessagesConfig,
    DeepSeekAnthropicMessagesConfig,
)
from litellm.utils import ProviderConfigManager


def test_deepseek_anthropic_sanitizes_custom_tool_wrapper_fields():
    config = AnthropicMessagesConfig()
    optional_params = {
        "max_tokens": 1024,
        "tools": [
            {
                "type": "custom",
                "name": "Agent",
                "description": "Run a subagent",
                "input_schema": {
                    "type": "custom",
                    "properties": {"prompt": {"type": "string"}},
                },
                "defer_loading": True,
            },
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            },
        ],
    }

    result = config.transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={"api_base": DEEPSEEK_ANTHROPIC_API_BASE},
        headers={},
    )

    assert result["tools"][0] == {
        "name": "Agent",
        "description": "Run a subagent",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
        },
    }
    assert result["tools"][1] == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }


def test_deepseek_anthropic_does_not_auto_add_anthropic_beta_header():
    config = AnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"output_format": {"type": "json_schema", "schema": {}}},
        litellm_params={"api_base": DEEPSEEK_ANTHROPIC_API_BASE},
        api_key="deepseek-key",
        api_base=None,
    )

    assert headers["x-api-key"] == "deepseek-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "anthropic-beta" not in headers


def test_provider_manager_returns_deepseek_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.DEEPSEEK,
    )

    assert isinstance(config, DeepSeekAnthropicMessagesConfig)
    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/anthropic/",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
