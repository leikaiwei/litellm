import httpx

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.deepseek.messages.transformation import (
    DeepSeekAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager


def _http_status_error(status_code: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.deepseek.com/anthropic/v1/messages")
    response = httpx.Response(status_code, content=body.encode(), request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_deepseek_provider_uses_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.DEEPSEEK,
    )

    assert isinstance(config, DeepSeekAnthropicMessagesConfig)
    assert config.custom_llm_provider == "deepseek"


def test_deepseek_anthropic_messages_config_defaults():
    config = DeepSeekAnthropicMessagesConfig()

    assert config.custom_llm_provider == "deepseek"
    assert config.get_api_base() == "https://api.deepseek.com/anthropic"


def test_anthropic_provider_keeps_default_config_for_deepseek_named_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, DeepSeekAnthropicMessagesConfig)


def test_deepseek_anthropic_messages_url_defaults_to_anthropic_endpoint():
    config = DeepSeekAnthropicMessagesConfig()

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
            api_base="https://api.deepseek.com/anthropic/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/anthropic",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/v1/messages",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )


def test_deepseek_anthropic_messages_headers_use_deepseek_key():
    config = DeepSeekAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="deepseek-v4-pro",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-deepseek",
        api_base="https://example.test/anthropic",
    )

    assert api_base == "https://example.test/anthropic"
    assert headers["x-api-key"] == "sk-deepseek"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


def test_deepseek_anthropic_messages_preserves_thinking_and_sanitizes_custom_tools():
    config = DeepSeekAnthropicMessagesConfig()
    messages = [
        {
            "role": "user",
            "content": "Use the tool.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should call the tool.",
                    "signature": "sig",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Sao Paulo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Sunny",
                }
            ],
        },
    ]

    request = config.transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=messages,
        anthropic_messages_optional_request_params={
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {
                    "type": "custom",
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"},
                },
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 1,
                },
            ],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["messages"] == messages
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert request["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object"},
    }
    assert request["tools"][1]["type"] == "web_search_20260209"


def test_should_retry_on_deepseek_redacted_thinking_400():
    config = DeepSeekAnthropicMessagesConfig()
    error = _http_status_error(
        400,
        '{"error":{"message":"unknown variant `redacted_thinking`, expected one of `text`, `thinking`"}}',
    )

    assert config.should_retry_anthropic_messages_on_http_error(e=error, litellm_params={}) is True


def test_should_retry_on_deepseek_reasoning_content_must_be_passed_back_400():
    config = DeepSeekAnthropicMessagesConfig()
    error = _http_status_error(
        400,
        '{"error":{"message":"The reasoning_content in the thinking mode must be passed back to the API."}}',
    )

    assert config.should_retry_anthropic_messages_on_http_error(e=error, litellm_params={}) is True


def test_should_not_retry_on_unrelated_400():
    config = DeepSeekAnthropicMessagesConfig()
    error = _http_status_error(
        400,
        '{"error":{"message":"max_tokens is required"}}',
    )

    assert config.should_retry_anthropic_messages_on_http_error(e=error, litellm_params={}) is False


def test_should_retry_still_handles_signature_error():
    config = DeepSeekAnthropicMessagesConfig()
    error = _http_status_error(
        400,
        "messages.1.content.0: Invalid `signature` in `thinking` block",
    )

    assert config.should_retry_anthropic_messages_on_http_error(e=error, litellm_params={}) is True


def test_non_400_not_retried():
    config = DeepSeekAnthropicMessagesConfig()
    error = _http_status_error(
        500,
        "unknown variant `redacted_thinking`",
    )

    assert config.should_retry_anthropic_messages_on_http_error(e=error, litellm_params={}) is False


def test_transform_on_error_strips_thinking_and_param():
    config = DeepSeekAnthropicMessagesConfig()
    request_data = {
        "model": "deepseek-v4-pro",
        "thinking": {"type": "enabled"},
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "foreign chain", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ],
    }
    error = _http_status_error(400, "unknown variant `redacted_thinking`")

    result = config.transform_anthropic_messages_request_on_http_error(e=error, request_data=request_data)

    assert "thinking" not in result
    assistant_content = result["messages"][1]["content"]
    assert all(block.get("type") not in ("thinking", "redacted_thinking") for block in assistant_content)
    assert assistant_content == [{"type": "text", "text": "answer"}]


def test_transform_on_error_noop_for_unrelated_400():
    config = DeepSeekAnthropicMessagesConfig()
    request_data = {
        "model": "deepseek-v4-pro",
        "thinking": {"type": "enabled"},
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "keep me", "signature": "sig"}],
            },
        ],
    }
    error = _http_status_error(400, "max_tokens is required")

    result = config.transform_anthropic_messages_request_on_http_error(e=error, request_data=request_data)

    assert result["thinking"] == {"type": "enabled"}
    assert result["messages"][0]["content"][0]["type"] == "thinking"
