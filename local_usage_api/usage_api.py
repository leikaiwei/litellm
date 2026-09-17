"""按虚拟 key 自查今天与近 7 天的请求数、token 与费用，数据取自 Prometheus。"""

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

CHINA_TZ: Final = timezone(timedelta(hours=8))
REQUESTS_METRIC: Final = "litellm_proxy_total_requests_metric_total"
TOKENS_METRIC: Final = "litellm_total_tokens_metric_total"
SPEND_METRIC: Final = "litellm_spend_metric_total"
PROMETHEUS_TIMEOUT_SECONDS: Final = 10.0


class Usage(BaseModel):
    since: datetime
    requests: int
    tokens: int
    spend: float


class UsageReport(BaseModel):
    as_of: datetime
    today: Usage
    last_7_days: Usage


class _PrometheusSample(BaseModel):
    value: tuple[float, str]


class _PrometheusData(BaseModel):
    result: tuple[_PrometheusSample, ...]


class _PrometheusResponse(BaseModel):
    data: _PrometheusData


def hash_key(raw_key: str) -> str:
    # 与 litellm 的 hash_token 同算法，结果即 Prometheus 的 hashed_api_key 标签值
    return hashlib.sha256(raw_key.encode()).hexdigest()


def china_day_start(now: datetime, days_back: int) -> datetime:
    today: Final = now.astimezone(CHINA_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=days_back)


def build_promql(metric: str, key_hash: str, window_seconds: int) -> str:
    return f'sum(increase({metric}{{hashed_api_key="{key_hash}"}}[{window_seconds}s]))'


async def query_sum(prometheus: httpx.AsyncClient, promql: str, at: datetime) -> float:
    response: Final = await prometheus.get("/api/v1/query", params={"query": promql, "time": at.timestamp()})
    response.raise_for_status()
    result: Final = _PrometheusResponse.model_validate_json(response.content).data.result
    return float(result[0].value[1]) if result else 0.0


async def query_usage(prometheus: httpx.AsyncClient, key_hash: str, since: datetime, now: datetime) -> Usage:
    # increase 至少要一个抓取点，窗口不足 1 秒时按 1 秒查，结果自然为 0
    window_seconds: Final = max(1, int((now - since).total_seconds()))
    requests, tokens, spend = await asyncio.gather(
        *(
            query_sum(prometheus, build_promql(metric, key_hash, window_seconds), now)
            for metric in (REQUESTS_METRIC, TOKENS_METRIC, SPEND_METRIC)
        )
    )
    return Usage(since=since, requests=round(requests), tokens=round(tokens), spend=round(spend, 4))


def create_app(prometheus: httpx.AsyncClient, clock: Callable[[], datetime]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await prometheus.aclose()

    app: Final = FastAPI(title="LiteLLM key usage", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    bearer: Final = HTTPBearer()

    @app.get("/usage")
    async def get_usage(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)]) -> UsageReport:
        # 只收明文 key：hash 在 Prometheus 里人人可见，收 hash 等于谁都能查别人
        key_hash: Final = hash_key(credentials.credentials)
        now: Final = clock().astimezone(CHINA_TZ)
        try:
            today, last_7_days = await asyncio.gather(
                query_usage(prometheus, key_hash, china_day_start(now, 0), now),
                query_usage(prometheus, key_hash, china_day_start(now, 6), now),
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail="Prometheus 查询失败") from e
        return UsageReport(as_of=now, today=today, last_7_days=last_7_days)

    return app


def build_app() -> FastAPI:
    prometheus: Final = httpx.AsyncClient(
        base_url=os.environ.get("PROMETHEUS_URL", "http://prometheus:9090"),
        timeout=PROMETHEUS_TIMEOUT_SECONDS,
    )
    return create_app(prometheus, lambda: datetime.now(timezone.utc))
