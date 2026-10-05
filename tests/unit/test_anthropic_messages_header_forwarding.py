from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup


def test_anthropic_messages_headers_are_forwarded_without_global_header_forwarding():
    data = {"headers": {"x-existing": "1"}}

    result = LiteLLMProxyRequestSetup.add_anthropic_messages_headers_to_llm_call(
        data=data,
        headers={
            "Anthropic-Beta": "advanced-tool-use-2025-11-20",
            "anthropic-version": "2023-06-01",
            "authorization": "Bearer litellm-key",
            "x-api-key": "client-provider-key",
        },
        request_path="/v1/messages",
    )

    assert result["headers"] == {
        "x-existing": "1",
        "anthropic-beta": "advanced-tool-use-2025-11-20",
        "anthropic-version": "2023-06-01",
    }


def test_anthropic_messages_headers_are_not_forwarded_to_other_routes():
    result = LiteLLMProxyRequestSetup.add_anthropic_messages_headers_to_llm_call(
        data={},
        headers={"anthropic-beta": "advanced-tool-use-2025-11-20"},
        request_path="/v1/chat/completions",
    )

    assert "headers" not in result
