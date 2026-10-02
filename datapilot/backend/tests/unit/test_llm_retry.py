"""Tests for the transient-unavailability retry wrapper."""

from __future__ import annotations

import pytest

from app.services.llm_service import (
    LLMServiceError,
    retry_transient_unavailable,
    _TRANSIENT_RETRIES,
)


async def _flaky(attempts_before_success: int, *, failure_code: str = "LLM_UNAVAILABLE"):
    calls = 0

    async def impl():
        nonlocal calls
        calls += 1
        if calls <= attempts_before_success:
            raise LLMServiceError(code=failure_code, message="spike")
        return "ok"

    wrapped = retry_transient_unavailable(impl)
    return wrapped, lambda: calls


class TestRetryTransientUnavailable:
    @pytest.mark.asyncio
    async def test_succeeds_after_transient_spikes(self):
        wrapped, calls = await _flaky(attempts_before_success=1)
        assert await wrapped() == "ok"
        assert calls() == 2

    @pytest.mark.asyncio
    async def test_succeeds_without_retry_when_first_call_works(self):
        wrapped, calls = await _flaky(attempts_before_success=0)
        assert await wrapped() == "ok"
        assert calls() == 1

    @pytest.mark.asyncio
    async def test_gives_up_after_bounded_attempts(self):
        wrapped, calls = await _flaky(attempts_before_success=_TRANSIENT_RETRIES + 5)
        with pytest.raises(LLMServiceError) as exc:
            await wrapped()
        assert exc.value.code == "LLM_UNAVAILABLE"
        assert calls() == _TRANSIENT_RETRIES + 1

    @pytest.mark.asyncio
    async def test_auth_errors_are_never_retried(self):
        wrapped, calls = await _flaky(
            _TRANSIENT_RETRIES + 5, failure_code="LLM_AUTHENTICATION_ERROR"
        )
        with pytest.raises(LLMServiceError) as exc:
            await wrapped()
        assert exc.value.code == "LLM_AUTHENTICATION_ERROR"
        assert calls() == 1

    @pytest.mark.asyncio
    async def test_rate_limits_are_never_retried(self):
        wrapped, calls = await _flaky(
            _TRANSIENT_RETRIES + 5, failure_code="LLM_RATE_LIMITED"
        )
        with pytest.raises(LLMServiceError) as exc:
            await wrapped()
        assert exc.value.code == "LLM_RATE_LIMITED"
        assert calls() == 1

    @pytest.mark.asyncio
    async def test_timeouts_are_never_retried(self):
        wrapped, calls = await _flaky(
            _TRANSIENT_RETRIES + 5, failure_code="LLM_TIMEOUT"
        )
        with pytest.raises(LLMServiceError) as exc:
            await wrapped()
        assert exc.value.code == "LLM_TIMEOUT"
        assert calls() == 1