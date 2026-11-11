"""
Test that the Router retry loop correctly handles non-retryable errors.

Verifies that:
1. Non-retryable errors (e.g., 400 ContextWindowExceeded) inside the retry loop
   break out immediately instead of being swallowed.
2. original_exception is updated to the latest error, not stuck on the first.
3. Retryable errors (e.g., 429 RateLimitError) still retry normally.

Regression tests for https://github.com/BerriAI/litellm/issues/21343
"""

import asyncio
import datetime
from collections.abc import Awaitable, Callable
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError, RequestAdmissionContextWindowExceededError, RequestAdmissionError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    _union_duration_ms,
    response_timing_metrics,
)
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.rules import Rules
from litellm.router_utils.cooldown_handlers import get_cooldown_deployments
from litellm.router_utils.router_callbacks.track_deployment_metrics import get_deployment_failures_for_current_minute
from litellm.types.router import RetryPolicy
from litellm.types.utils import CallTypes
from litellm.utils import function_setup


class AdmissionVerdicts(CustomLogger):
    def __init__(self, errors: tuple[Exception, ...]) -> None:
        super().__init__()
        self.errors = errors
        self.calls: tuple[str, ...] = ()

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:
        index: Final = len(self.calls)
        self.calls = (*self.calls, str(kwargs.get("model")))
        if index < len(self.errors):
            raise self.errors[index]
        return None


def _admission_router(
    model: str = "anthropic/claude-admission-fixture", ordered: bool = False, **options: object
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {
                    "model": model,
                    "api_key": "unused",
                    "mock_response": "accepted",
                    "order": order if ordered else 0,
                },
                "model_info": {"id": identifier},
            }
            for group, order, identifier in (
                ("primary", 0, "primary"),
                ("primary", 1, "secondary"),
                ("backup", 0, "backup"),
                ("last", 0, "last"),
            )
        ],
        **options,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (RequestAdmissionError, RequestAdmissionContextWindowExceededError))
@pytest.mark.parametrize("surface", ("messages", "chat", "responses"))
@pytest.mark.parametrize("stream", (False, True))
async def test_admission_rejection_stops_recovery_without_cooling_deployment(
    monkeypatch: pytest.MonkeyPatch, error_type: type[RequestAdmissionError], surface: str, stream: bool
) -> None:
    rejection: Final = error_type(message="admission rejected", model="fixture", llm_provider="anthropic")
    hook: Final = AdmissionVerdicts((rejection,))
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = _admission_router(
        num_retries=2,
        retry_policy=RetryPolicy(BadRequestErrorRetries=2),
        fallbacks=[{"primary": ["backup", "last"]}],
        context_window_fallbacks=[{"primary": ["backup", "last"]}],
        fallback_on_unsupported_params=True,
        enable_weighted_failover=True,
        allowed_fails=0,
    )
    try:
        request: Final[Awaitable[object]]
        if surface == "messages":
            request = router.anthropic_messages(
                model="primary", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=stream
            )
        elif surface == "chat":
            request = router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}], stream=stream)
        else:
            request = router.aresponses(model="primary", input="hi", stream=stream)
        with pytest.raises(RequestAdmissionError) as caught:
            await request
        assert caught.value is rejection
        assert len(hook.calls) == 1
        assert get_cooldown_deployments(router, parent_otel_span=None) == []
        assert tuple(
            get_deployment_failures_for_current_minute(router, identifier)
            for identifier in ("primary", "secondary", "backup", "last")
        ) == (0, 0, 0, 0)
        response: Final = await router.anthropic_messages(
            model="primary", messages=[{"role": "user", "content": "next request"}], max_tokens=10
        )
        assert response["content"][0]["text"] == "accepted"
    finally:
        router.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (RequestAdmissionError, RequestAdmissionContextWindowExceededError))
@pytest.mark.parametrize("recovery", ("retry", "fallback", "order", "weighted"))
async def test_admission_rejection_on_later_attempt_stops_the_remaining_chain(
    monkeypatch: pytest.MonkeyPatch, error_type: type[RequestAdmissionError], recovery: str
) -> None:
    rejection: Final = error_type(message="admission rejected", model="fixture", llm_provider="anthropic")
    hook: Final = AdmissionVerdicts((_make_rate_limit_error(), rejection))
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = _admission_router(
        ordered=recovery == "order",
        num_retries=2 if recovery == "retry" else 0,
        retry_policy=RetryPolicy(BadRequestErrorRetries=2, RateLimitErrorRetries=2) if recovery == "retry" else None,
        retry_after=0,
        fallbacks=[{"primary": ["backup", "last"]}],
        context_window_fallbacks=[{"primary": ["backup", "last"]}],
        fallback_on_unsupported_params=True,
        enable_weighted_failover=recovery == "weighted",
        disable_cooldowns=True,
    )
    try:
        with pytest.raises(RequestAdmissionError) as caught:
            await router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}])
        assert caught.value is rejection
        assert len(hook.calls) == 2
    finally:
        router.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (RequestAdmissionError, RequestAdmissionContextWindowExceededError))
async def test_stream_retry_propagates_admission_rejection_before_fallback(
    error_type: type[RequestAdmissionError],
) -> None:
    rejection: Final = error_type(message="admission rejected", model="fixture", llm_provider="anthropic")

    async def rejected_attempt(**kwargs: object) -> object:
        raise rejection

    router: Final = _admission_router(num_retries=2, retry_after=0)
    dropped_stream: Final = MidStreamFallbackError(
        message="upstream disconnected",
        model="primary",
        llm_provider="anthropic",
        original_exception=_make_rate_limit_error(),
        is_pre_first_chunk=True,
    )
    try:
        with pytest.raises(RequestAdmissionError) as caught:
            await router._astream_retry_same_group(dropped_stream, {"model": "primary"}, rejected_attempt)
        assert caught.value is rejection
    finally:
        router.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (litellm.ContextWindowExceededError, litellm.RateLimitError))
async def test_ordinary_provider_errors_still_reach_configured_fallback(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[litellm.ContextWindowExceededError] | type[litellm.RateLimitError],
) -> None:
    upstream_error: Final = error_type(message="upstream rejected", model="fixture", llm_provider="openai")
    hook: Final = AdmissionVerdicts((upstream_error,))
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router: Final = _admission_router(
        model="openai/admission-fixture",
        num_retries=0,
        fallbacks=[{"primary": ["backup"]}],
        context_window_fallbacks=[{"primary": ["backup"]}],
        disable_cooldowns=True,
    )
    try:
        response: Final = await router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "accepted"
        assert len(hook.calls) == 2
        assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "backup"
    finally:
        router.reset()


def _make_rate_limit_error(message="Rate limited"):
    """Create a RateLimitError for testing."""
    return litellm.RateLimitError(
        message=message,
        llm_provider="bedrock",
        model="anthropic.claude-v2",
    )


def _make_context_window_error(message="prompt is too long: 1205821 tokens > 200000"):
    """Create a ContextWindowExceededError for testing."""
    return litellm.ContextWindowExceededError(
        message=message,
        llm_provider="vertex_ai",
        model="claude-3-opus",
    )


def _make_bad_request_error(message="Invalid request"):
    """Create a BadRequestError for testing."""
    return litellm.BadRequestError(
        message=message,
        llm_provider="openai",
        model="gpt-4",
    )


def _make_not_found_error(message="Model not found"):
    """Create a NotFoundError for testing."""
    return litellm.NotFoundError(
        message=message,
        llm_provider="openai",
        model="gpt-99",
    )


def _create_router(num_retries=2):
    """Create a Router with two deployments for testing."""
    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key-1",
                },
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key-2",
                },
            },
        ],
        num_retries=num_retries,
    )


def _base_kwargs():
    """Return kwargs required by async_function_with_retries."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "original_function": AsyncMock(),
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_non_retryable_error_in_retry_loop_raises_immediately():
    """
    When a non-retryable error (400 ContextWindowExceeded) occurs inside the
    retry loop, the router should raise it immediately instead of swallowing it
    and raising the original error.

    Scenario: First call -> 429, Retry -> 400 (non-retryable)
    Expected: ContextWindowExceededError is raised, NOT RateLimitError
    """
    router = _create_router(num_retries=2)

    rate_limit_error = _make_rate_limit_error()
    context_window_error = _make_context_window_error()

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        else:
            raise context_window_error

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.ContextWindowExceededError):
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )


@pytest.mark.asyncio
async def test_bad_request_error_in_retry_loop_raises_immediately():
    """
    A generic 400 BadRequestError inside the retry loop should also break out
    immediately since 400 is not retryable.
    """
    router = _create_router(num_retries=2)

    rate_limit_error = _make_rate_limit_error()
    bad_request_error = _make_bad_request_error()

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        else:
            raise bad_request_error

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.BadRequestError):
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )


@pytest.mark.asyncio
async def test_original_exception_updated_to_latest_error():
    """
    When all retries are exhausted with retryable errors, the LAST error
    should be raised, not the first one.
    """
    router = _create_router(num_retries=2)

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error(f"Rate limit attempt {call_count}")

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )
        # Should be the LAST error, not the first
        assert "Rate limit attempt 3" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retryable_errors_still_retry_normally():
    """
    Retryable errors (429 RateLimitError) should still be retried the
    configured number of times before raising.
    """
    router = _create_router(num_retries=3)

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error(f"Rate limit attempt {call_count}")

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.async_function_with_retries(
                num_retries=3,
                **_base_kwargs(),
            )

        # Initial call + 3 retries = 4 total calls
        assert call_count == 4


@pytest.mark.asyncio
async def test_not_found_error_in_retry_loop_raises_immediately():
    """
    A 404 NotFoundError inside the retry loop should break out immediately.
    """
    router = _create_router(num_retries=2)

    rate_limit_error = _make_rate_limit_error()
    not_found_error = _make_not_found_error()

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        else:
            raise not_found_error

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.NotFoundError):
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )

        # Only 2 calls: initial + first retry that hits non-retryable
        assert call_count == 2


@pytest.mark.asyncio
async def test_retry_attempts_accumulate_timing_in_shared_request_metadata():
    received_at: Final = datetime.datetime.now()
    metadata: dict[str, object] = {
        "model_group": "test-model",
        "litellm_received_at": received_at,
    }
    logging_obj_raw, _ = function_setup(
        "acompletion",
        Rules(),
        datetime.datetime.now(),
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        metadata=metadata,
        litellm_call_id="retry-timing-test",
        is_async_call=True,
    )
    assert isinstance(logging_obj_raw, Logging)
    logging_obj: Final[Logging] = logging_obj_raw
    attempt_numbers: list[int] = []
    metadata_ids: list[int] = []

    @track_llm_api_timing()
    async def timed_attempt(*, logging_obj: Logging, **kwargs: object) -> str:
        del kwargs
        attempt_numbers.append(len(attempt_numbers) + 1)
        metadata_ids.append(id(logging_obj.model_call_details["litellm_params"]["metadata"]))
        await asyncio.sleep(0.01)
        if len(attempt_numbers) == 1:
            raise _make_rate_limit_error()
        return "success"

    async def invoke(original_function: Callable[..., Awaitable[str]], *args: object, **kwargs: object) -> str:
        return await original_function(*args, **kwargs)

    router = _create_router(num_retries=1)
    with (
        patch.object(router, "make_call", new=AsyncMock(side_effect=invoke)),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            new=AsyncMock(return_value=(["d1"], ["d1"])),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
    ):
        result = await router.async_function_with_retries(
            original_function=timed_attempt,
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            metadata=metadata,
            logging_obj=logging_obj,
            num_retries=1,
        )

    request_metadata: Final = logging_obj.model_call_details["litellm_params"]["metadata"]
    windows: Final = request_metadata["llm_api_timing_windows"]
    end_time: Final = datetime.datetime.fromtimestamp(max(window[1] for window in windows))
    timing_metrics: Final = response_timing_metrics(received_at, end_time, logging_obj)
    assert result == "success"
    assert attempt_numbers == [1, 2]
    assert request_metadata is metadata
    assert metadata_ids == [id(metadata), id(metadata)]
    assert len(windows) == 2
    union_duration_ms: Final = _union_duration_ms(windows, received_at.timestamp(), end_time.timestamp())
    assert union_duration_ms is not None
    total_response_time_ms: Final = (end_time.timestamp() - received_at.timestamp()) * 1000
    assert timing_metrics["litellm_overhead_time_ms"] == pytest.approx(
        round(total_response_time_ms - union_duration_ms, 4)
    )
