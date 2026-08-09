"""Tests for prediction-step helpers."""

from __future__ import annotations

from tabbench_llm.predictions import _is_context_window_error, _is_infrastructure_error


def test_context_window_error_detection():
    # The real proxy message the LLM cells hit (litellm wrapping a vLLM 400).
    real = (
        "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError "
        "- This model's maximum context length is 262144 tokens. However, you requested 448 "
        "output tokens and your prompt contains at least 261697 input tokens."
    )
    assert _is_context_window_error(Exception(real))
    assert _is_context_window_error(Exception("ContextWindowExceededError: prompt too long"))
    assert _is_context_window_error(ValueError("maximum context length is 4096 tokens"))

    # Unrelated failures must not be swallowed as a context skip.
    assert not _is_context_window_error(Exception("connection reset by peer"))
    assert not _is_context_window_error(ValueError("invalid api key"))


def test_infrastructure_error_detection():
    class HttpError(Exception):
        status_code = 429

    assert _is_infrastructure_error(HttpError("too many requests"))
    assert _is_infrastructure_error(Exception("503 Service Unavailable"))
    assert _is_infrastructure_error(Exception("connection reset by peer"))
    assert not _is_infrastructure_error(Exception("model returned an invalid class"))
