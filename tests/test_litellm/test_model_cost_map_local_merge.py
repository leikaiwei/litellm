from unittest.mock import MagicMock, patch

from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    get_model_cost_map,
)


def test_valid_remote_model_cost_map_keeps_local_only_entries():
    """Valid remote maps should keep local fork entries that are not upstream yet."""
    backup = GetModelCostMap.load_local_model_cost_map()
    remote_map = {
        key: value
        for key, value in backup.items()
        if key != "anthropic/deepseek-v4-pro"
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = remote_map

    with patch("httpx.get", return_value=mock_response):
        result = get_model_cost_map("https://fake-url.com/model_prices.json")

    assert "anthropic/deepseek-v4-pro" in result
    assert result["anthropic/deepseek-v4-pro"]["litellm_provider"] == "anthropic"
