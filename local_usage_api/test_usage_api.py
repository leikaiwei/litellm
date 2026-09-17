"""usage_api 的回归测试。

跑法：.venv/bin/python -m pytest local_usage_api/ -q
"""

import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Final
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from usage_api import CHINA_TZ, china_day_start, create_app, hash_key

from litellm.proxy._types import hash_token

RAW_KEY: Final = "sk-test-usage-api-0123456789"
KEY_HASH: Final = hash_token(RAW_KEY)
# 中国时间 2026-09-17 07:30，UTC 仍是 09-16，用来区分两种"今天"
NOW: Final = datetime(2026, 9, 16, 23, 30, tzinfo=timezone.utc)
TODAY_WINDOW_SECONDS: Final = 7 * 3600 + 1800
LAST_7_DAYS_WINDOW_SECONDS: Final = 6 * 86400 + TODAY_WINDOW_SECONDS


def _vector(value: str | None) -> httpx.Response:
    result: Final = [] if value is None else [{"metric": {}, "value": [NOW.timestamp(), value]}]
    return httpx.Response(200, json={"status": "success", "data": {"resultType": "vector", "result": result}})


def _client(
    handler: Callable[[httpx.Request], httpx.Response], now: datetime = NOW
) -> tuple[TestClient, list[httpx.Request]]:
    seen: Final[list[httpx.Request]] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    prometheus: Final = httpx.AsyncClient(
        base_url="http://prometheus:9090", transport=httpx.MockTransport(recording_handler)
    )
    return TestClient(create_app(prometheus, lambda: now)), seen


def _promql(request: httpx.Request) -> str:
    return parse_qs(urlparse(str(request.url)).query)["query"][0]


def test_hash_key_matches_litellm_hash_token():
    assert hash_key(RAW_KEY) == KEY_HASH


def test_china_day_start_uses_utc8_not_utc():
    assert china_day_start(NOW, 0) == datetime(2026, 9, 17, tzinfo=CHINA_TZ)
    assert china_day_start(NOW, 6) == datetime(2026, 9, 11, tzinfo=CHINA_TZ)


def test_usage_sums_each_metric_per_window_by_key_hash():
    values: Final = {
        f'sum(increase(litellm_proxy_total_requests_metric_total{{hashed_api_key="{KEY_HASH}"}}[{TODAY_WINDOW_SECONDS}s]))': "12.6",
        f'sum(increase(litellm_total_tokens_metric_total{{hashed_api_key="{KEY_HASH}"}}[{TODAY_WINDOW_SECONDS}s]))': "345678.6",
        f'sum(increase(litellm_spend_metric_total{{hashed_api_key="{KEY_HASH}"}}[{TODAY_WINDOW_SECONDS}s]))': "1.23456",
        f'sum(increase(litellm_proxy_total_requests_metric_total{{hashed_api_key="{KEY_HASH}"}}[{LAST_7_DAYS_WINDOW_SECONDS}s]))': "5909.6",
        f'sum(increase(litellm_total_tokens_metric_total{{hashed_api_key="{KEY_HASH}"}}[{LAST_7_DAYS_WINDOW_SECONDS}s]))': "859122990.6",
        f'sum(increase(litellm_spend_metric_total{{hashed_api_key="{KEY_HASH}"}}[{LAST_7_DAYS_WINDOW_SECONDS}s]))': "807.15",
    }
    client, seen = _client(lambda request: _vector(values[_promql(request)]))

    response: Final = client.get("/usage", headers={"Authorization": f"Bearer {RAW_KEY}"})

    assert response.status_code == 200
    body: Final = response.json()
    assert body["today"] | {"since": None} == {"since": None, "requests": 13, "tokens": 345679, "spend": 1.2346}
    assert body["last_7_days"] | {"since": None} == {
        "since": None,
        "requests": 5910,
        "tokens": 859122991,
        "spend": 807.15,
    }
    assert datetime.fromisoformat(body["today"]["since"]) == datetime(2026, 9, 17, tzinfo=CHINA_TZ)
    assert datetime.fromisoformat(body["last_7_days"]["since"]) == datetime(2026, 9, 11, tzinfo=CHINA_TZ)
    assert sorted(_promql(request) for request in seen) == sorted(values)
    assert {parse_qs(urlparse(str(request.url)).query)["time"][0] for request in seen} == {str(NOW.timestamp())}


def test_raw_key_never_leaves_the_service():
    client, seen = _client(lambda _: _vector("1"))

    client.get("/usage", headers={"Authorization": f"Bearer {RAW_KEY}"})

    assert seen
    assert all(RAW_KEY not in str(request.url) and RAW_KEY not in json.dumps(dict(request.headers)) for request in seen)


def test_key_without_series_reports_zero():
    client, _ = _client(lambda _: _vector(None))

    body: Final = client.get("/usage", headers={"Authorization": f"Bearer {RAW_KEY}"}).json()

    assert (body["today"]["requests"], body["today"]["tokens"], body["today"]["spend"]) == (0, 0, 0.0)
    assert (body["last_7_days"]["requests"], body["last_7_days"]["tokens"], body["last_7_days"]["spend"]) == (0, 0, 0.0)


def test_today_window_at_china_midnight_is_clamped_to_one_second():
    china_midnight: Final = datetime(2026, 9, 17, tzinfo=CHINA_TZ)
    client, seen = _client(lambda _: _vector(None), now=china_midnight)

    client.get("/usage", headers={"Authorization": f"Bearer {RAW_KEY}"})

    windows: Final = {_promql(request).rsplit("[", 1)[1] for request in seen}
    assert windows == {"1s]))", f"{6 * 86400}s]))"}


def test_request_without_bearer_is_rejected_before_querying_prometheus():
    client, seen = _client(lambda _: _vector("1"))

    assert client.get("/usage").status_code in (401, 403)
    assert client.get("/usage", headers={"Authorization": RAW_KEY}).status_code in (401, 403)
    assert seen == []


def test_prometheus_failure_maps_to_502():
    client, _ = _client(lambda _: httpx.Response(503, text="unavailable"))

    response: Final = client.get("/usage", headers={"Authorization": f"Bearer {RAW_KEY}"})

    assert response.status_code == 502
    assert RAW_KEY not in response.text
