"""
Deep unit tests for the RateLimitManager rotation engine.

Covers edge cases not hit by test_rotation.py:
  - Empty/whitespace/invalid pools
  - Round-robin wraparound math
  - Cooldown expiry timing
  - Independent pair vs key vs model state
  - Stats consistency
"""
from __future__ import annotations

import time

import pytest

from googlemodel_samrat.core import (
    RateLimitManager,
    _looks_like_daily_quota,
    _next_midnight_ts,
)
from googlemodel_samrat.exceptions import (
    AllResourcesExhaustedError,
    ConfigurationError,
)


# ─────────────────────────────────────────────────────────────────────
# Construction & validation
# ─────────────────────────────────────────────────────────────────────
def test_rejects_none_api_keys():
    with pytest.raises(ConfigurationError):
        RateLimitManager(api_keys=None, models=["m1"])  # type: ignore[arg-type]


def test_rejects_none_models():
    with pytest.raises(ConfigurationError):
        RateLimitManager(api_keys=["k1"], models=None)  # type: ignore[arg-type]


def test_rejects_all_whitespace_keys():
    with pytest.raises(ConfigurationError):
        RateLimitManager(api_keys=["  ", "\t", ""], models=["m1"])


def test_rejects_all_empty_models():
    with pytest.raises(ConfigurationError):
        RateLimitManager(api_keys=["k1"], models=["", None])  # type: ignore[list-item]


def test_strips_whitespace_from_keys():
    mgr = RateLimitManager(api_keys=["  k1  ", "k2 "], models=["m1"])
    assert mgr.api_keys == ["k1", "k2"]


def test_drops_empty_string_models():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1", "", "m2"])
    assert mgr.models == ["m1", "m2"]


# ─────────────────────────────────────────────────────────────────────
# Round-robin geometry
# ─────────────────────────────────────────────────────────────────────
def test_max_attempts_is_product_of_pools():
    mgr = RateLimitManager(api_keys=["k1", "k2", "k3"], models=["m1", "m2"])
    assert mgr.max_attempts == 6


def test_single_single_pool():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"])
    assert mgr.max_attempts == 1
    assert mgr.has_key_rotation is False
    assert mgr.has_model_rotation is False


def test_round_robin_visits_every_pair_exactly_once():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2", "m3"])
    seen = [tuple(mgr.get_next_config().values()) for _ in range(6)]
    assert len(set(seen)) == 6


def test_round_robin_wraps_after_full_cycle():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    a = mgr.get_next_config()
    b = mgr.get_next_config()
    c = mgr.get_next_config()
    assert a == c
    assert a != b


# ─────────────────────────────────────────────────────────────────────
# Cooldown timing
# ─────────────────────────────────────────────────────────────────────
def test_cooldown_actually_expires():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"], cooldown_seconds=0.1)
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    with pytest.raises(AllResourcesExhaustedError):
        mgr.get_next_config()
    time.sleep(0.15)
    assert mgr.get_next_config() == {"api_key": "k1", "model": "m1"}


def test_shortest_wait_returns_positive_number():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"], cooldown_seconds=30)
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    wait = mgr.shortest_wait()
    assert 0 < wait <= 30


def test_shortest_wait_zero_when_all_healthy():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"])
    assert mgr.shortest_wait() == 0.0


def test_cooldown_uses_shortest_of_multiple_pairs():
    mgr = RateLimitManager(
        api_keys=["k1", "k2"],
        models=["m1"],
        cooldown_seconds=5,
    )
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    # Shortest wait for the only cooling pair should be ≈5s
    assert 4 < mgr.shortest_wait() <= 5


# ─────────────────────────────────────────────────────────────────────
# Daily quota detection
# ─────────────────────────────────────────────────────────────────────
def test_daily_markers_variations():
    for marker in ("per day", "daily quota", "requests per day", "rpd limit"):
        assert _looks_like_daily_quota(marker.lower())


def test_per_minute_is_not_daily():
    assert not _looks_like_daily_quota("per minute rate limit")


def test_next_midnight_is_in_the_future():
    assert _next_midnight_ts() > time.time()


def test_daily_hint_sleeps_until_midnight():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"])
    mgr.mark_failed("k1", "m1", Exception("429 daily quota reached"))
    state = mgr._pair_state[("k1", "m1")]
    # Should be sleeping until ~next midnight, not just cooldown
    assert state.daily_until > time.time() + 60 * 30  # at least 30 min


def test_daily_sleep_disabled_still_cools_pair():
    mgr = RateLimitManager(
        api_keys=["k1"], models=["m1"], daily_sleep=False, cooldown_seconds=0.05
    )
    mgr.mark_failed("k1", "m1", Exception("429 daily quota reached"))
    state = mgr._pair_state[("k1", "m1")]
    assert state.daily_until == 0
    assert state.cooldown_until > time.time()


# ─────────────────────────────────────────────────────────────────────
# Pair isolation
# ─────────────────────────────────────────────────────────────────────
def test_pair_cooldown_does_not_affect_key():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    assert mgr.failed_keys == []
    assert mgr.failed_models == []
    assert mgr.failed_pairs == [("k1", "m1")]


def test_pair_cooldown_does_not_affect_other_pairs():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    # Every other combination should still be selectable
    for _ in range(3):
        cfg = mgr.get_next_config()
        assert (cfg["api_key"], cfg["model"]) != ("k1", "m1")


def test_same_key_different_model_unaffected():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    cfg = mgr.get_next_config()
    assert cfg == {"api_key": "k1", "model": "m2"}


def test_same_model_different_key_unaffected():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    cfg = mgr.get_next_config()
    assert cfg == {"api_key": "k2", "model": "m1"}


# ─────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────
def test_reset_clears_all_state():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    mgr.reset_failures()
    assert mgr.failed_keys == []
    assert mgr.failed_models == []
    assert mgr.failed_pairs == []
    assert mgr.shortest_wait() == 0


def test_reset_clears_daily_sleep():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"])
    mgr.mark_failed("k1", "m1", Exception("429 daily quota reached"))
    mgr.reset_failures()
    assert mgr.get_next_config() == {"api_key": "k1", "model": "m1"}


# ─────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────
def test_stats_keys_present():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    s = mgr.stats
    for key in (
        "total_keys", "total_models", "active_keys", "active_models",
        "cooling_keys", "cooling_models", "cooling_pairs",
        "available_combinations",
    ):
        assert key in s


def test_stats_available_combinations_decrease_on_pair_failure():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    before = mgr.stats["available_combinations"]
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    after = mgr.stats["available_combinations"]
    assert after == before - 1


def test_stats_cooling_pairs_count():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    mgr.mark_failed("k2", "m2", Exception("429 quota"))
    assert mgr.stats["cooling_pairs"] == 2
