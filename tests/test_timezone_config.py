"""v0.2.0: daily_reset_hour_utc / daily_reset_tz configuration."""
import time

import pytest

from googlemodel_samrat.core import (
    RateLimitManager,
    _next_midnight_ts,
    _next_reset_ts,
)


def test_default_manager_matches_legacy_utc_midnight():
    m = RateLimitManager(api_keys=["k"], models=["m"])
    assert m._next_reset_ts() == pytest.approx(_next_midnight_ts(), abs=2.0)


def test_hour_utc_config_moves_reset():
    m = RateLimitManager(api_keys=["k"], models=["m"], daily_reset_hour_utc=8)
    assert m._next_reset_ts() == pytest.approx(_next_reset_ts(8, "UTC"), abs=2.0)


def test_tz_config_moves_reset():
    m = RateLimitManager(
        api_keys=["k"], models=["m"], daily_reset_tz="Asia/Kathmandu"
    )
    assert m._next_reset_ts() == pytest.approx(
        _next_reset_ts(0, "Asia/Kathmandu"), abs=2.0
    )


def test_hour_and_tz_combined():
    m = RateLimitManager(
        api_keys=["k"],
        models=["m"],
        daily_reset_hour_utc=20,
        daily_reset_tz="America/Los_Angeles",
    )
    assert m._next_reset_ts() == pytest.approx(
        _next_reset_ts(20, "America/Los_Angeles"), abs=2.0
    )


def test_hour_is_clamped():
    m = RateLimitManager(api_keys=["k"], models=["m"], daily_reset_hour_utc=99)
    assert m.daily_reset_hour_utc == 23
    m2 = RateLimitManager(api_keys=["k"], models=["m"], daily_reset_hour_utc=-5)
    assert m2.daily_reset_hour_utc == 0


def test_reset_ts_always_in_future():
    now = time.time()
    for hour in range(24):
        assert _next_reset_ts(hour, "UTC") > now


def test_daily_mark_uses_configured_hour():
    m = RateLimitManager(
        api_keys=["k"], models=["m"], daily_reset_hour_utc=8, cooldown_seconds=60.0
    )
    m._mark_key("k", "429 daily quota exceeded", time.time(), daily_hint=True)
    expected = _next_reset_ts(8, "UTC") - time.time()
    assert m.shortest_wait() == pytest.approx(expected, abs=5.0)


def test_chat_passes_reset_config_to_manager():
    from googlemodel_samrat.chat import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        api_keys=["k1"],
        models=["m1"],
        daily_reset_hour_utc=8,
        daily_reset_tz="Asia/Kathmandu",
    )
    assert llm._manager.daily_reset_hour_utc == 8
    assert llm._manager.daily_reset_tz == "Asia/Kathmandu"
    
def test_embeddings_passes_reset_config_to_manager():
    from googlemodel_samrat.embeddings import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1"],
        models=["e1"],
        daily_reset_hour_utc=8,
        daily_reset_tz="Asia/Kathmandu",
    )
    assert emb._manager.daily_reset_hour_utc == 8
    assert emb._manager.daily_reset_tz == "Asia/Kathmandu"