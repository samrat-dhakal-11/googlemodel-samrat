"""
Unit tests for the googlemodel-samrat rotation engine.

No network calls — the inner langchain-google-genai client is faked.

Run:
    pip install -e ".[dev]"
    pytest tests/ -v
"""
from __future__ import annotations

import threading
import time
from typing import Any, List, Optional

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import langchain_google_genai  # noqa: F401  (ensures module is patchable)

from googlemodel_samrat.chat import ChatGoogleGenerativeAI
from googlemodel_samrat.core import (
    RateLimitManager,
    _looks_like_daily_quota,
    _next_midnight_ts,
)
from googlemodel_samrat.embeddings import GoogleGenerativeAIEmbeddings
from googlemodel_samrat.exceptions import (
    AllResourcesExhaustedError,
    ConfigurationError,
)


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def make_fake_lc(
    behavior: Optional[Any] = None,
    stream_behavior: Optional[Any] = None,
):
    class FakeLC:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def invoke(self, messages, **kw):
            if behavior is None:
                return AIMessage(content="ok")
            return behavior(self.kwargs, messages)

        def stream(self, messages, **kw):
            if stream_behavior is None:
                yield AIMessageChunk(content="chunk")
                return
            yield from stream_behavior(self.kwargs, messages)

        async def ainvoke(self, messages, **kw):
            return self.invoke(messages, **kw)

        async def astream(self, messages, **kw):
            for c in self.stream(messages, **kw):
                yield c

    return FakeLC


def make_fake_embeddings(behavior: Optional[Any] = None):
    class FakeEmb:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def embed_query(self, text: str):
            if behavior is not None:
                return behavior(self.kwargs, text)
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts: List[str]):
            return [[0.1, 0.2] for _ in texts]

    return FakeEmb


@pytest.fixture
def happy_chat(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(),
    )


@pytest.fixture
def happy_emb(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        make_fake_embeddings(),
    )


# ─────────────────────────────────────────────────────────────────────
# RateLimitManager — configuration
# ─────────────────────────────────────────────────────────────────────
def test_manager_rejects_empty_keys():
    with pytest.raises(ConfigurationError):
        RateLimitManager(api_keys=[], models=["m1"])


def test_manager_rejects_empty_models():
    with pytest.raises(ConfigurationError):
        RateLimitManager(api_keys=["k"], models=[])


def test_manager_strips_whitespace_keys():
    mgr = RateLimitManager(api_keys=["  k1  ", "", "  "], models=["m1"])
    assert mgr.api_keys == ["k1"]


# ─────────────────────────────────────────────────────────────────────
# RateLimitManager — 1-key vs multi-key semantics
# ─────────────────────────────────────────────────────────────────────
def test_single_key_no_key_rotation():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1", "m2"])
    assert mgr.has_key_rotation is False
    assert mgr.has_model_rotation is True


def test_single_model_no_model_rotation():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    assert mgr.has_key_rotation is True
    assert mgr.has_model_rotation is False


def test_multi_key_multi_model_both_rotations():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    assert mgr.has_key_rotation is True
    assert mgr.has_model_rotation is True
    assert mgr.max_attempts == 4


def test_single_key_never_cycles_key_index():
    mgr = RateLimitManager(api_keys=["only"], models=["m1", "m2", "m3"])
    for _ in range(6):
        cfg = mgr.get_next_config()
        assert cfg["api_key"] == "only"


def test_multi_key_cycles_all_combinations():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    seen = {tuple(mgr.get_next_config().items()) for _ in range(4)}
    assert len(seen) == 4


# ─────────────────────────────────────────────────────────────────────
# RateLimitManager — failure classification
# ─────────────────────────────────────────────────────────────────────
def test_quota_error_marks_key_only():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    mgr.mark_failed("k1", "m1", Exception("429 quota exceeded"))
    cfg = mgr.get_next_config()
    assert cfg["api_key"] == "k2"
    assert cfg["model"] == "m1"


def test_model_not_found_marks_model_only():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("404 model not found"))
    cfg = mgr.get_next_config()
    assert cfg["api_key"] == "k1"
    assert cfg["model"] == "m2"


def test_server_error_marks_key_only():
    """5xx is transient — cool the key, but keep the model usable via other keys."""
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("503 service unavailable"))
    cfg = mgr.get_next_config()
    assert cfg["api_key"] == "k2"
    assert cfg["model"] == "m1"


def test_daily_quota_detection():
    assert _looks_like_daily_quota("You have exceeded your quota per day")
    assert _looks_like_daily_quota("rpd limit reached")
    assert not _looks_like_daily_quota("per minute limit")


def test_daily_quota_uses_midnight_sleep():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"], daily_sleep=True)
    mgr.mark_failed("k1", "m1", Exception("429 daily quota exceeded for the day"))
    st = mgr._key_state["k1"]
    assert st.daily_until > time.time() + 60 * 60
    assert abs(st.daily_until - _next_midnight_ts()) < 5


def test_daily_sleep_can_be_disabled():
    mgr = RateLimitManager(
        api_keys=["k1", "k2"], models=["m1"], daily_sleep=False, cooldown_seconds=1
    )
    mgr.mark_failed("k1", "m1", Exception("429 daily quota exceeded for the day"))
    st = mgr._key_state["k1"]
    assert st.daily_until == 0
    assert st.cooldown_until > time.time()


# ─────────────────────────────────────────────────────────────────────
# RateLimitManager — cooldown expiry, reset, stats
# ─────────────────────────────────────────────────────────────────────
def test_cooldown_expires():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"], cooldown_seconds=0.05)
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    with pytest.raises(AllResourcesExhaustedError):
        mgr.get_next_config()
    time.sleep(0.1)
    cfg = mgr.get_next_config()
    assert cfg["api_key"] == "k1"


def test_reset_failures_clears_state():
    mgr = RateLimitManager(api_keys=["k1"], models=["m1"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    mgr.reset_failures()
    cfg = mgr.get_next_config()
    assert cfg == {"api_key": "k1", "model": "m1"}


def test_stats_reports_rotation_flags():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    s = mgr.stats
    assert s["key_rotation_enabled"] is True
    assert s["model_rotation_enabled"] is False
    assert s["available_combinations"] == 2
    assert s["cooling_keys"] == 0


def test_stats_counts_cooling_resources():
    """5xx cools only the key; model stays usable via other keys."""
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("503 server error"))
    s = mgr.stats
    assert s["cooling_keys"] == 1
    assert s["cooling_models"] == 0


# ─────────────────────────────────────────────────────────────────────
# RateLimitManager — thread safety
# ─────────────────────────────────────────────────────────────────────
def test_concurrent_get_next_config_is_safe():
    mgr = RateLimitManager(api_keys=["k1", "k2", "k3"], models=["m1", "m2"])
    results: List[tuple] = []
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            cfg = mgr.get_next_config()
            with lock:
                results.append((cfg["api_key"], cfg["model"]))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 400
    valid = {("k1", "m1"), ("k1", "m2"), ("k2", "m1"),
             ("k2", "m2"), ("k3", "m1"), ("k3", "m2")}
    assert set(results).issubset(valid)


def test_concurrent_mark_failed_is_safe():
    mgr = RateLimitManager(api_keys=["k1", "k2", "k3"], models=["m1", "m2"])

    def worker(i: int):
        for _ in range(30):
            mgr.mark_failed(f"k{(i % 3) + 1}", f"m{(i % 2) + 1}",
                            Exception("429 quota"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(k in mgr.api_keys for k in (mgr.failed_keys or ["k1"]))


# ─────────────────────────────────────────────────────────────────────
# ChatGoogleGenerativeAI — init & configuration
# ─────────────────────────────────────────────────────────────────────
def test_chat_rejects_no_keys(monkeypatch, happy_chat):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        ChatGoogleGenerativeAI(api_keys=[], models=["m1"])


def test_chat_single_key_stats(happy_chat):
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1", "m2"])
    s = llm.get_rotation_stats()
    assert s["key_rotation_enabled"] is False
    assert s["model_rotation_enabled"] is True


# ─────────────────────────────────────────────────────────────────────
# ChatGoogleGenerativeAI — invoke happy path + attribution
# ─────────────────────────────────────────────────────────────────────
def test_invoke_returns_aimessage(happy_chat):
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    msg = llm.invoke("hello")
    assert isinstance(msg, AIMessage)
    assert msg.content == "ok"


def test_invoke_populates_attribution(happy_chat):
    llm = ChatGoogleGenerativeAI(api_keys=["k1", "k2"], models=["m1", "m2"])
    llm.invoke("hello")
    assert llm.last_successful_model in {"m1", "m2"}
    assert llm.last_successful_key_index in {0, 1}


def test_invoke_works_through_lcel(happy_chat):
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chain = llm | (lambda m: m.content.upper())
    assert chain.invoke("hi") == "OK"


# ─────────────────────────────────────────────────────────────────────
# ChatGoogleGenerativeAI — fallback & rotation
# ─────────────────────────────────────────────────────────────────────
def test_invoke_rotates_key_on_429(monkeypatch):
    seen = []

    def behavior(kwargs, messages):
        seen.append((kwargs["google_api_key"], kwargs["model"]))
        if len(seen) == 1:
            raise Exception("429 quota exceeded")
        return AIMessage(content="recovered")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(behavior=behavior),
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2"],
        models=["m1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    msg = llm.invoke("hi")
    assert msg.content == "recovered"
    assert seen[0][0] == "k1" and seen[1][0] == "k2"
    assert seen[0][1] == seen[1][1] == "m1"


def test_invoke_rotates_model_on_404(monkeypatch):
    seen = []

    def behavior(kwargs, messages):
        seen.append((kwargs["google_api_key"], kwargs["model"]))
        if kwargs["model"] == "m1":
            raise Exception("404 model not found")
        return AIMessage(content="model-2")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(behavior=behavior),
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1"],
        models=["m1", "m2"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    msg = llm.invoke("hi")
    assert msg.content == "model-2"
    assert seen[-1][1] == "m2"


def test_all_failures_raise_friendly_error(monkeypatch):
    def behavior(kwargs, messages):
        raise Exception("429 quota exceeded")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(behavior=behavior),
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2"],
        models=["m1", "m2"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    with pytest.raises(AllResourcesExhaustedError) as ei:
        llm.invoke("hi")
    msg = str(ei.value)
    assert "Troubleshooting" in msg
    assert "Attempted 4" in msg


# ─────────────────────────────────────────────────────────────────────
# ChatGoogleGenerativeAI — streaming
# ─────────────────────────────────────────────────────────────────────
def test_stream_yields_chunks(monkeypatch):
    def stream_behavior(kwargs, messages):
        yield AIMessageChunk(content="Hel")
        yield AIMessageChunk(content="lo!")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(stream_behavior=stream_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    text = "".join(c.content for c in llm.stream("hi"))
    assert text == "Hello!"


def test_stream_rotates_on_first_chunk_failure(monkeypatch):
    calls = {"n": 0}

    def stream_behavior(kwargs, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("503 server unavailable")
        yield AIMessageChunk(content="ok")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(stream_behavior=stream_behavior),
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2"],
        models=["m1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    text = "".join(c.content for c in llm.stream("hi"))
    assert text == "ok"
    assert calls["n"] == 2


# ─────────────────────────────────────────────────────────────────────
# ChatGoogleGenerativeAI — async
# ─────────────────────────────────────────────────────────────────────
async def test_ainvoke_returns_aimessage(happy_chat):
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    msg = await llm.ainvoke("hello")
    assert isinstance(msg, AIMessage)
    assert msg.content == "ok"


async def test_astream_yields_chunks(monkeypatch):
    def stream_behavior(kwargs, messages):
        yield AIMessageChunk(content="a")
        yield AIMessageChunk(content="b")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        make_fake_lc(stream_behavior=stream_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chunks = [c.content async for c in llm.astream("hi")]
    assert "".join(chunks) == "ab"


# ─────────────────────────────────────────────────────────────────────
# Embeddings wrapper
# ─────────────────────────────────────────────────────────────────────
def test_embeddings_embed_query(happy_emb):
    emb = GoogleGenerativeAIEmbeddings(api_keys=["k1"], models=["e1"])
    vec = emb.embed_query("hello")
    assert vec == [0.1, 0.2, 0.3]


def test_embeddings_rejects_no_keys(monkeypatch, happy_emb):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        GoogleGenerativeAIEmbeddings(api_keys=[])


def test_embeddings_rotates_on_failure(monkeypatch):
    seen = []

    def behavior(kwargs, text):
        seen.append(kwargs["google_api_key"])
        if len(seen) == 1:
            raise Exception("429 quota exceeded")
        return [1.0, 2.0]

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        make_fake_embeddings(behavior=behavior),
    )
    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1", "k2"],
        models=["e1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    vec = emb.embed_query("hi")
    assert vec == [1.0, 2.0]
    assert seen == ["k1", "k2"]


def test_embeddings_attribution(happy_emb):
    emb = GoogleGenerativeAIEmbeddings(api_keys=["k1"], models=["e1"])
    emb.embed_query("hi")
    assert emb.last_successful_model == "e1"
    assert emb.last_successful_key_index == 0