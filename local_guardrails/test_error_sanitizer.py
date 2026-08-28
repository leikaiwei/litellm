"""
error_sanitizer guardrail 的回归测试。

跑法：.venv/bin/python -m pytest local_guardrails/test_error_sanitizer.py -q

测试与被测文件同目录，不放 tests/test_litellm/，避免在 upstream 跟踪的路径下留改动。

最有价值的是末尾那组 e2e：把 guardrail 挂进 `litellm.callbacks`，让真实的
`ProxyRateLimitError`（限速器实际抛的那个类）走完
`ProxyBaseLLMRequestProcessing._handle_llm_api_exception`，断言下游最终拿到的
`ProxyException` 里一个内部细节都不剩。前面的单元测试全过而它挂掉，说明改写虽然发生了
却没能落到响应体上 —— 那正是这个补丁存在的意义。
"""

import json
import os
import sys
from typing import Final
from unittest.mock import patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_sanitizer import (
    SANITIZED_MESSAGES,
    ErrorSanitizerGuardrail,
    retained_headers,
    sanitized_status_code,
    stream_already_delivered,
)

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.utils import ProxyLogging

GUARDRAIL_NAME: Final = "error-sanitizer"

# 生产里下游实际看到的那一串。改写要挡掉的就是它
REAL_RATE_LIMIT_DETAIL: Final = (
    "Rate limit exceeded for model_per_key: "
    "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef0badc0de:ratelimit-test. "
    "Limit type: requests. Current limit: 3, Remaining: 0. "
    "Limit resets at: 2026-08-28 09:57:17 UTC"
)

# 429 原文里每一项都不该出现在下游响应中
LEAKED_FRAGMENTS: Final = (
    "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef0badc0de",
    "ratelimit-test",
    "model_per_key",
    "Current limit",
    "Remaining",
    "Limit resets at",
    "09:57:17",
    "litellm",
    "LiteLLM",
)


def build_guardrail() -> ErrorSanitizerGuardrail:
    return ErrorSanitizerGuardrail(
        guardrail_name=GUARDRAIL_NAME,
        event_hook="post_call",
        default_on=True,
    )


def real_rate_limit_error() -> ProxyRateLimitError:
    """限速器实际抛的那个类，构造参数照抄 parallel_request_limiter_v3._handle_rate_limit_error。"""
    return ProxyRateLimitError(
        detail=REAL_RATE_LIMIT_DETAIL,
        headers={
            "retry-after": "60",
            "rate_limit_type": "requests",
            "reset_at": "2026-08-28 09:57:17 UTC",
        },
        rate_limit_type="key",
        model="ratelimit-test",
        llm_provider="litellm_proxy",
    )


def api_key_dict() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="u-test")


async def run_hook(exception: Exception, request_data: dict | None = None) -> HTTPException | None:
    return await build_guardrail().async_post_call_failure_hook(
        request_data=request_data if request_data is not None else {},
        original_exception=exception,
        user_api_key_dict=api_key_dict(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", sorted(SANITIZED_MESSAGES))
async def test_sanitized_codes_are_replaced(status_code: int) -> None:
    result = await run_hook(HTTPException(status_code=status_code, detail="内部细节 litellm.SomeError xyz"))
    assert result is not None
    assert result.status_code == status_code
    assert result.detail == SANITIZED_MESSAGES[status_code]
    assert "xyz" not in json.dumps(result.detail, ensure_ascii=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (400, 401, 403, 404, 422, 499))
async def test_passthrough_codes_are_untouched(status_code: int) -> None:
    assert await run_hook(HTTPException(status_code=status_code, detail="原文要留着")) is None


@pytest.mark.asyncio
async def test_exception_without_status_code_is_untouched() -> None:
    assert await run_hook(ValueError("没有 status_code")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (None, True, "429", 429.0))
async def test_non_int_status_code_is_untouched(status_code: object) -> None:
    """状态码不是改写表里的整数键时一律放行。"429" 和 429.0 这两个尤其要守：
    前者是字符串不等于键，后者 `429.0 in {429: ...}` 为真，会被脱敏 —— 那是可接受的，
    因为它确实是个 429。这里锁住前三个放行、浮点那个被处理。"""

    class Weird(Exception):
        pass

    exception = Weird()
    exception.status_code = status_code  # pyright: ignore[reportAttributeAccessIssue]  # 模拟第三方异常的鸭子类型
    result = await run_hook(exception)
    if status_code == 429.0:
        assert result is not None
    else:
        assert result is None


@pytest.mark.asyncio
async def test_real_rate_limit_error_is_sanitized() -> None:
    result = await run_hook(real_rate_limit_error())
    assert result is not None
    assert result.status_code == 429
    assert result.detail == SANITIZED_MESSAGES[429]
    for fragment in LEAKED_FRAGMENTS:
        assert fragment not in str(result.detail)


@pytest.mark.asyncio
async def test_retry_after_survives_but_window_details_do_not() -> None:
    """retry-after 是客户端退避的唯一依据，必须透传；reset_at / rate_limit_type 暴露
    限速窗口边界与维度，必须丢掉。改写后的异常不会自动继承原异常的头，所以这条断言
    守的是一个真出现过的漏洞。"""
    result = await run_hook(real_rate_limit_error())
    assert result is not None
    assert result.headers is not None
    assert result.headers["retry-after"] == "60"
    assert "reset_at" not in result.headers
    assert "rate_limit_type" not in result.headers


def test_retained_headers_whitelist() -> None:
    assert retained_headers(real_rate_limit_error()) == {"retry-after": "60"}
    assert retained_headers(HTTPException(status_code=500, detail="x")) == {}
    assert retained_headers(ValueError("x")) == {}


@pytest.mark.asyncio
async def test_stream_already_delivered_keeps_original() -> None:
    """流已吐字节时返回 HTTPException 会让 async_data_generator 直接 raise，截断连接，
    比原路径的 SSE error 帧更糟，所以这条路必须放行。"""
    result = await run_hook(real_rate_limit_error(), {"combined_usage_object": {"total_tokens": 12}})
    assert result is None


@pytest.mark.asyncio
async def test_stream_not_yet_delivered_is_sanitized() -> None:
    result = await run_hook(real_rate_limit_error(), {"stream": True})
    assert result is not None
    assert result.detail == SANITIZED_MESSAGES[429]


def test_stream_already_delivered_judgement() -> None:
    assert stream_already_delivered({"combined_usage_object": {}}) is True
    assert stream_already_delivered({}) is False
    assert stream_already_delivered({"combined_usage_object": None}) is False


def test_sanitized_status_code_selection() -> None:
    assert sanitized_status_code(HTTPException(status_code=429, detail="x")) == 429
    assert sanitized_status_code(HTTPException(status_code=400, detail="x")) is None
    assert sanitized_status_code(ValueError("x")) is None


@pytest.mark.asyncio
async def test_original_exception_is_logged_in_full() -> None:
    """改写只影响响应体，内部取证能力不能丢。"""
    with patch("error_sanitizer.verbose_proxy_logger.info") as info:
        await run_hook(real_rate_limit_error())
    logged = " ".join(str(arg) for arg in info.call_args.args)
    assert "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef0badc0de" in logged
    assert "Current limit" in logged


@pytest.mark.asyncio
async def test_e2e_downstream_response_carries_no_internal_detail() -> None:
    """真实限速异常走完 _handle_llm_api_exception，断言下游拿到的字节里没有细节。

    这是唯一能守住本 guardrail 意义的测试：hook 返回值必须真的顶替掉响应体。
    """
    guardrail: Final = build_guardrail()
    original_callbacks: Final = list(litellm.callbacks)
    litellm.callbacks = [*original_callbacks, guardrail]
    try:
        processor: Final = ProxyBaseLLMRequestProcessing(data={"model": "ratelimit-test"})
        with pytest.raises(ProxyException) as exc_info:
            await processor._handle_llm_api_exception(
                e=real_rate_limit_error(),
                user_api_key_dict=api_key_dict(),
                proxy_logging_obj=ProxyLogging(user_api_key_cache=None),
            )
    finally:
        litellm.callbacks = original_callbacks

    proxy_exception: Final = exc_info.value
    assert proxy_exception.code == "429"
    assert proxy_exception.message == SANITIZED_MESSAGES[429]
    assert proxy_exception.type == "rate_limit_error"

    downstream_bytes: Final = json.dumps(proxy_exception.to_dict(), ensure_ascii=False)
    for fragment in LEAKED_FRAGMENTS:
        assert fragment not in downstream_bytes, f"下游响应泄漏了 {fragment}: {downstream_bytes}"

    # 客户端退避靠它，不能连 retry-after 一起脱掉
    assert proxy_exception.headers.get("retry-after") == "60"


@pytest.mark.asyncio
async def test_e2e_400_reaches_downstream_verbatim() -> None:
    """400 保留原文是刻意选择：客户端得知道自己请求哪里写错了。"""
    guardrail: Final = build_guardrail()
    original_callbacks: Final = list(litellm.callbacks)
    litellm.callbacks = [*original_callbacks, guardrail]
    try:
        processor: Final = ProxyBaseLLMRequestProcessing(data={"model": "m"})
        with pytest.raises(ProxyException) as exc_info:
            await processor._handle_llm_api_exception(
                e=HTTPException(status_code=400, detail="messages: role must be one of user/assistant"),
                user_api_key_dict=api_key_dict(),
                proxy_logging_obj=ProxyLogging(user_api_key_cache=None),
            )
    finally:
        litellm.callbacks = original_callbacks

    assert exc_info.value.code == "400"
    assert "role must be one of" in exc_info.value.message
