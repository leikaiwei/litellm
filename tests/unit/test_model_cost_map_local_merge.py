import json

import httpx
import pytest

from litellm.litellm_core_utils import get_model_cost_map as module
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    ModelCostMapReloaded,
    get_model_cost_map,
    refetch_model_cost_map,
)

_URL = "https://example.invalid/model_prices.json"
_FORK_ONLY_MODEL = "anthropic/deepseek-v4-pro"


@pytest.fixture(autouse=True)
def _isolate_cost_map_state(monkeypatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    source_info = module._cost_map_source_info
    for name in ("source", "url", "is_env_forced", "fallback_reason", "loaded_at", "source_revision", "etag"):
        monkeypatch.setattr(source_info, name, getattr(source_info, name))


def _remote_without_fork_entry(request: httpx.Request) -> httpx.Response:
    backup = GetModelCostMap.load_local_model_cost_map()
    assert _FORK_ONLY_MODEL in backup
    remote_map = {key: value for key, value in backup.items() if key != _FORK_ONLY_MODEL}
    remote_map["gpt-4o"] = {**remote_map["gpt-4o"], "input_cost_per_token": 0.123}
    return httpx.Response(200, content=json.dumps(remote_map).encode())


def _assert_local_fills_gaps_and_remote_wins(cost_map: dict) -> None:
    assert cost_map[_FORK_ONLY_MODEL]["litellm_provider"] == "anthropic"
    assert cost_map["gpt-4o"]["input_cost_per_token"] == 0.123


def test_boot_load_keeps_local_only_entries():
    """远端价格表没有 fork 自加的模型时，首拉结果仍保留本地条目，且同名条目以远端为准。"""
    client = httpx.Client(transport=httpx.MockTransport(_remote_without_fork_entry))

    _assert_local_fills_gaps_and_remote_wins(get_model_cost_map(url=_URL, client=client))


@pytest.mark.asyncio
async def test_runtime_reload_keeps_local_only_entries():
    """proxy 定时重载走另一条入口，同样不能把 fork 自加的模型丢掉。"""
    client = httpx.AsyncClient(transport=httpx.MockTransport(_remote_without_fork_entry))

    result = await refetch_model_cost_map(url=_URL, client=client)

    assert isinstance(result, ModelCostMapReloaded)
    _assert_local_fills_gaps_and_remote_wins(result.model_cost_map)
