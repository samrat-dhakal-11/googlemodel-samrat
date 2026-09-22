"""
Error classification tests for RateLimitManager.

Verifies that mark_failed() correctly categorizes every kind of
error a real Gemini call can raise, using the right cooling policy:

  - 429 / quota      -> cool the (key, model) pair
  - 404 / not_found  -> cool the model only
  - 5xx / timeout    -> cool the key only
  - unknown          -> cool the key only (conservative)
  - daily quota      -> cool the pair until next midnight
"""
from __future__ import annotations

import pytest
from google.api_core.exceptions import (
    BadGateway,
    GatewayTimeout,
    InternalServerError,
    NotFound,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    Unauthenticated,
)

from googlemodel_samrat.core import RateLimitManager


# ─────────────────────────────────────────────────────────────────────
# 429 / quota -> pair cooldown
# ─────────────────────────────────────────────────────────────────────
def test_429_via_string_message_cools_pair():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 Too Many Requests"))
    assert mgr.failed_pairs == [("k1", "m1")]
    assert mgr.failed_keys == []
    assert mgr.failed_models == []


def test_resource_exhausted_exception_cools_pair():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", ResourceExhausted("quota exceeded"))
    assert mgr.failed_pairs == [("k1", "m1")]


def test_permission_denied_cools_pair():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", PermissionDenied("invalid api key"))
    assert mgr.failed_pairs == [("k1", "m1")]


def test_unauthenticated_cools_pair():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Unauthenticated("key rejected"))
    assert mgr.failed_pairs == [("k1", "m1")]


def test_exception_with_429_code_attr():
    class FakeError(Exception):
        code = 429
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", FakeError("rate limited"))
    assert mgr.failed_pairs == [("k1", "m1")]


def test_exception_with_400_code_attr():
    class FakeError(Exception):
        code = 400
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", FakeError("bad request"))
    assert mgr.failed_pairs == [("k1", "m1")]


def test_daily_quota_sleeps_until_midnight():
    import time
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 daily quota reached"))
    state = mgr._pair_state[("k1", "m1")]
    assert state.daily_until > time.time() + 60 * 60


# ─────────────────────────────────────────────────────────────────────
# 404 / not_found -> model only
# ─────────────────────────────────────────────────────────────────────
def test_404_string_cools_model_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("404 model not found"))
    assert mgr.failed_models == ["m1"]
    assert mgr.failed_keys == []
    assert mgr.failed_pairs == []


def test_not_found_exception_cools_model_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", NotFound("model deprecated"))
    assert mgr.failed_models == ["m1"]


def test_deprecated_string_cools_model_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("model has been deprecated"))
    assert mgr.failed_models == ["m1"]


def test_unsupported_model_cools_model_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("unsupported model"))
    assert mgr.failed_models == ["m1"]


# ─────────────────────────────────────────────────────────────────────
# 5xx / transient -> key only
# ─────────────────────────────────────────────────────────────────────
def test_500_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("500 internal server error"))
    assert mgr.failed_keys == ["k1"]
    assert mgr.failed_models == []
    assert mgr.failed_pairs == []


def test_502_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("502 Bad Gateway"))
    assert mgr.failed_keys == ["k1"]


def test_503_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("503 Service Unavailable"))
    assert mgr.failed_keys == ["k1"]


def test_504_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("504 Gateway Timeout"))
    assert mgr.failed_keys == ["k1"]


def test_timeout_string_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("request timed out"))
    assert mgr.failed_keys == ["k1"]


def test_service_unavailable_exception_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", ServiceUnavailable("retry later"))
    assert mgr.failed_keys == ["k1"]


def test_internal_server_error_exception_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", InternalServerError("boom"))
    assert mgr.failed_keys == ["k1"]


def test_bad_gateway_exception_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", BadGateway("upstream error"))
    assert mgr.failed_keys == ["k1"]


def test_gateway_timeout_exception_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", GatewayTimeout("no response"))
    assert mgr.failed_keys == ["k1"]


# ─────────────────────────────────────────────────────────────────────
# Unknown -> key only (conservative)
# ─────────────────────────────────────────────────────────────────────
def test_unknown_error_cools_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("something bizarre happened"))
    assert mgr.failed_keys == ["k1"]
    assert mgr.failed_models == []
    assert mgr.failed_pairs == []


def test_arbitrary_exception_class_cools_key_only():
    class CustomError(Exception):
        pass
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", CustomError("nope"))
    assert mgr.failed_keys == ["k1"]


# ─────────────────────────────────────────────────────────────────────
# Case-insensitivity of the string matching
# ─────────────────────────────────────────────────────────────────────
def test_uppercase_quota_still_classified_as_quota():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("RESOURCE_EXHAUSTED"))
    assert mgr.failed_pairs == [("k1", "m1")]


def test_mixed_case_404_still_classified_as_model_error():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("Model Not Found"))
    assert mgr.failed_models == ["m1"]


# ─────────────────────────────────────────────────────────────────────
# Multiple errors on different pairs accumulate
# ─────────────────────────────────────────────────────────────────────
def test_multiple_pair_failures_accumulate():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    mgr.mark_failed("k2", "m2", Exception("429 quota"))
    assert set(mgr.failed_pairs) == {("k1", "m1"), ("k2", "m2")}


def test_same_pair_failed_twice_still_one_entry():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    assert mgr.failed_pairs == [("k1", "m1")]
