"""
把返回给下游的错误体收敠成"只给状态码和一句固定文案"，详细内容只留在内部日志。

## 为什么需要它

litellm 默认把内部细节原样透给客户端。限速就是最典型的一例，下游看到的是：

    Error Code: 429
    Message: litellm.RateLimitError: Rate limit exceeded for model_per_key:
      deadbeef0123456789abcdef0123456789abcdef0123456789abcdef0badc0de:ratelimit-test.
      Limit type: requests. Current limit: 3, Remaining: 0.
      Limit resets at: 2026-08-28 09:57:17 UTC

一行里泄漏了限速维度、key 的哈希、当前额度、剩余量和重置时刻，还带 `litellm.` 前缀
暴露了网关实现。客户端要做的只是退避重试，这些都不必知道。

## 改写规则

    429 / 500 / 502 / 503 / 504  ->  换成固定文案
    其余（含 400、401、403、404）  ->  一个字都不碰

400 保留原文是刻意的：客户端得知道自己请求哪里写错了，换成一句 "Bad request" 会让人
无从下手。401 / 403 / 404 同理，本身不含内部细节，且下游要靠原文区分是 key 无效还是
模型不存在。

改写只影响 HTTP 响应体。原始异常照旧由 litellm 自己的 failure logging 落进
`LiteLLM_ErrorLogs` 与容器日志，取证能力不受影响。

## 两个不生效的边界（都是上游结构决定的，不是本文件的缺陷）

**1. 认证阶段的错误改不了。** `user_api_key_auth` 里抛的异常（key 无效、模型不在白
名单、预算超限）走 `ProxyLogging._handle_logging_proxy_only_error`，那条路调
`post_call_failure_hook` **只为记日志**，返回值不参与响应构造。好在这类错误本身文案
就干净，泄漏面小。

**2. 流式首字节之后的错误不改。** `async_data_generator`（common_request_processing.py
的 `except Exception`）拿到 `HTTPException` 会**直接 raise 而不是 yield SSE error 帧**，
连接被截断，客户端看到的是传输层异常而非 JSON 错误体。原路径至少会给一个 error 帧，
所以中途改写只会更糟。判据是 `request_data` 里有没有 `combined_usage_object`：
`ProxyLogging.post_call_failure_hook` 会把已交付 chunk 的用量提上来，有它就说明字节
已经出门了。

## 上线方式

外部自定义 guardrail，不修改 litellm 源码。按文件挂载进容器
（`./error_sanitizer.py:/app/error_sanitizer.py:ro`），然后：

    guardrails:
      - guardrail_name: "error-sanitizer"
        litellm_params:
          guardrail: error_sanitizer.ErrorSanitizerGuardrail
          mode: "post_call"
          default_on: true

`default_on: true` 是对的，与其他 guardrail 相反：错误改写要全局生效，而
`async_post_call_failure_hook` 由 `litellm.callbacks` 无条件遍历，不看 `default_on`
也不看模型级 `guardrails` 列表，所以挂 `false` 同样会全局生效。写 `true` 只是让
配置读起来不误导人。同理，`mode` 取哪个值都不影响本 guardrail，`post_call` 是最贴近
语义的那个。

因为它绕过模型级开关，模型级 `guardrails: [...]` 列表里**不需要**写它，漏写也不会
把它关掉。
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth

SANITIZED_MESSAGES: Final[dict[int, str]] = {
    429: "Rate limit exceeded, please retry later",
    500: "Internal server error",
    502: "Upstream service error",
    503: "Service temporarily unavailable",
    504: "Upstream request timed out",
}

ERROR_TYPES: Final[dict[int, str]] = {
    429: "rate_limit_error",
    500: "internal_server_error",
    502: "upstream_error",
    503: "service_unavailable",
    504: "timeout_error",
}


def sanitized_status_code(exception: Exception) -> int | None:
    """取出该异常应被改写的状态码，不在改写范围内则返回 None。"""
    status_code = getattr(exception, "status_code", None)
    return status_code if status_code in SANITIZED_MESSAGES else None


def stream_already_delivered(request_data: Mapping[str, object]) -> bool:
    """流式是否已经吐过字节：有已交付 chunk 的用量就说明出过门。"""
    return request_data.get("combined_usage_object") is not None


def retained_headers(exception: Exception) -> dict[str, str]:
    """只留 retry-after：客户端退避必须靠它。

    同一个异常上的 reset_at / rate_limit_type 不透传 —— 前者是绝对时刻，暴露了限速
    窗口边界，后者暴露了限速维度，两者客户端都用不上。

    必须显式带上：`_handle_llm_api_exception` 拿的是被顶替后那个异常的 `headers`，
    原异常的头不会自动继承。漏了它下游就只能靠猜退避间隔。
    """
    headers: Final = getattr(exception, "headers", None)
    if not isinstance(headers, Mapping):
        return {}
    retry_after: Final[object] = cast("Mapping[str, object]", headers).get("retry-after")
    return {"retry-after": str(retry_after)} if retry_after is not None else {}


class _SanitizedError(HTTPException):
    """detail 用纯字符串：dict 会被 _serialize_http_exception_detail 拆进
    provider_specific_fields 原样回给下游，反而多泄漏一层。"""

    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.type = ERROR_TYPES[status_code]
        self.param = None
        super().__init__(
            status_code=status_code,
            detail=SANITIZED_MESSAGES[status_code],
            headers=headers or None,
        )


class ErrorSanitizerGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = True,
        **_kwargs: object,
    ) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.post_call],
            event_hook=event_hook,
            default_on=default_on,
        )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: "UserAPIKeyAuth",
        traceback_str: str | None = None,
    ) -> HTTPException | None:
        status_code: Final = sanitized_status_code(original_exception)
        if status_code is None:
            return None

        if stream_already_delivered(request_data):
            verbose_proxy_logger.debug(
                "error-sanitizer: 流已交付字节，%s 保持原样",
                status_code,
            )
            return None

        verbose_proxy_logger.info(
            "error-sanitizer: %s 已脱敏，原文 %s",
            status_code,
            original_exception,
        )
        return _SanitizedError(status_code, retained_headers(original_exception))
