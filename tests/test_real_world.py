"""
Real-world integration tests for googlemodel-samrat.

No network. Uses a fake langchain_google_genai.ChatGoogleGenerativeAI
so we can simulate quota errors, timeouts, streaming failures, etc.

Covers:
  1. Multi-client isolation
  2. Concurrent threads under load
  3. Streaming with fallback
  4. Async under FastAPI-style loop
  5. Real LCEL chains
  6. Midnight-sleep / daily-quota semantics
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterator, List, Optional

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from googlemodel_samrat import ChatGoogleGenerativeAI
from googlemodel_samrat.core import (
    RateLimitManager,
    _next_midnight_ts,
)
from googlemodel_samrat.exceptions import AllResourcesExhaustedError


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def _fake_lc_factory(
    invoke_behavior: Optional[Callable[[dict, list], AIMessage]] = None,
    stream_behavior: Optional[Callable[[dict, list], Iterator[AIMessageChunk]]] = None,
):
    """
    Two SEPARATE callables because a function containing `yield`
    becomes a generator function and its `return X` path yields a
    generator object, not X.

    invoke_behavior(kwargs, messages) -> AIMessage | raises
    stream_behavior(kwargs, messages) -> iterator[AIMessageChunk] | raises
    """

    class FakeLC:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def invoke(self, messages, **kw):
            if invoke_behavior is None:
                return AIMessage(content="ok")
            return invoke_behavior(self.kwargs, messages)

        def stream(self, messages, **kw):
            if stream_behavior is None:
                yield AIMessageChunk(content="ok")
                return
            yield from stream_behavior(self.kwargs, messages)

        async def ainvoke(self, messages, **kw):
            return self.invoke(messages, **kw)

        async def astream(self, messages, **kw):
            for c in self.stream(messages, **kw):
                yield c

    return FakeLC


# ─────────────────────────────────────────────────────────────────────
# 1. Multi-client isolation
# ─────────────────────────────────────────────────────────────────────
def test_two_clients_do_not_share_state(monkeypatch):
    hits = {"n": 0}

    def invoke_behavior(kwargs, messages):
        if kwargs["google_api_key"] == "key-shared":
            hits["n"] += 1
            if hits["n"] == 1:
                raise Exception("429 quota exceeded")
        return AIMessage(content="ok")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )

    a = ChatGoogleGenerativeAI(
        api_keys=["key-shared", "key-fallback"],
        models=["m1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    b = ChatGoogleGenerativeAI(
        api_keys=["key-shared", "key-fallback"],
        models=["m1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )

    a.invoke("hi")
    b.invoke("hi")

    assert a.get_rotation_stats()["cooling_pairs"] == 1
    assert b.get_rotation_stats()["cooling_pairs"] == 0


# ─────────────────────────────────────────────────────────────────────
# 2. Concurrent threads under load
# ─────────────────────────────────────────────────────────────────────
def test_concurrent_invoke_under_load(monkeypatch):
    results: List[str] = []
    lock = threading.Lock()

    def invoke_behavior(kwargs, messages):
        # Deterministically fail ~30% of keys to force rotation
        if hash(kwargs["google_api_key"]) % 10 < 3:
            raise Exception("429 quota exceeded")
        return AIMessage(content=f"ok-{kwargs['google_api_key']}")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )

    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2", "k3", "k4"],
        models=["m1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )

    def worker():
        try:
            r = llm.invoke("hi")
            with lock:
                results.append(r.content)
        except AllResourcesExhaustedError:
            with lock:
                results.append("EXHAUSTED")

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 50
    stats = llm.get_rotation_stats()
    assert stats["total_keys"] == 4
    assert stats["cooling_keys"] <= 4


# ─────────────────────────────────────────────────────────────────────
# 3. Streaming with fallback
# ─────────────────────────────────────────────────────────────────────
def test_stream_recovers_from_first_chunk_failure(monkeypatch):
    attempt = {"n": 0}

    def stream_behavior(kwargs, messages):
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise Exception("503 service unavailable")
        yield AIMessageChunk(content="recovered ")
        yield AIMessageChunk(content="stream")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(stream_behavior=stream_behavior),
    )

    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2"],
        models=["m1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    text = "".join(c.content for c in llm.stream("hi"))
    assert text == "recovered stream"
    assert attempt["n"] == 2


# ─────────────────────────────────────────────────────────────────────
# 4. Async under FastAPI-style loop
# ─────────────────────────────────────────────────────────────────────
async def test_async_in_fastapi_style_loop(monkeypatch):
    def invoke_behavior(kwargs, messages):
        return AIMessage(content=f"ok-{kwargs['google_api_key']}")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )

    llm = ChatGoogleGenerativeAI(api_keys=["k1", "k2"], models=["m1"])

    async def handler():
        return await llm.ainvoke("hi")

    results = await asyncio.gather(*[handler() for _ in range(20)])
    assert len(results) == 20
    assert all(r.content.startswith("ok-") for r in results)


async def test_async_streaming_in_fastapi_style_loop(monkeypatch):
    def stream_behavior(kwargs, messages):
        yield AIMessageChunk(content="hello ")
        yield AIMessageChunk(content="world")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(stream_behavior=stream_behavior),
    )

    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])

    async def handler():
        chunks = []
        async for c in llm.astream("hi"):
            chunks.append(c.content)
        return "".join(chunks)

    results = await asyncio.gather(*[handler() for _ in range(10)])
    assert all(r == "hello world" for r in results)


# ─────────────────────────────────────────────────────────────────────
# 5. Real LCEL chains
# ─────────────────────────────────────────────────────────────────────
def test_lcel_chain_with_prompt_template(monkeypatch):
    def invoke_behavior(kwargs, messages):
        user_text = messages[0].content if messages else ""
        return AIMessage(content=f"[{kwargs['model']}] echoes: {user_text[:30]}")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )

    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    prompt = PromptTemplate.from_template("Summarize: {text}")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": "The quick brown fox jumps over the lazy dog"})
    assert "echoes: Summarize:" in result


def test_lcel_chain_falls_back_mid_chain(monkeypatch):
    seen_models: List[str] = []

    def invoke_behavior(kwargs, messages):
        seen_models.append(kwargs["model"])
        if kwargs["model"] == "m1":
            raise Exception("429 quota exceeded")
        return AIMessage(content=f"served by {kwargs['model']}")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )

    llm = ChatGoogleGenerativeAI(
        api_keys=["k1"],
        models=["m1", "m2"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    chain = (lambda x: x) | llm | StrOutputParser()
    result = chain.invoke("hi")
    assert "served by m2" in result
    assert "m1" in seen_models and "m2" in seen_models


# ─────────────────────────────────────────────────────────────────────
# 6. Midnight-sleep / daily-quota semantics
# ─────────────────────────────────────────────────────────────────────
def test_midnight_sleep_uses_next_utc_midnight():
    mgr = RateLimitManager(
        api_keys=["k1", "k2"],
        models=["m1"],
        daily_sleep=True,
        cooldown_seconds=60,
    )
    mgr.mark_failed("k1", "m1", Exception("429 daily quota exceeded for the day"))

    state = mgr._pair_state[("k1", "m1")]
    expected_ts = _next_midnight_ts()
    assert abs(state.daily_until - expected_ts) < 5

    # _next_midnight_ts returns next 00:00 UTC — verify in UTC
    from datetime import timezone
    wake_utc = datetime.fromtimestamp(state.daily_until, tz=timezone.utc)
    assert wake_utc.hour == 0
    assert wake_utc.minute == 0
    assert wake_utc.second == 0


def test_cooldown_and_daily_sleep_are_independent():
    mgr = RateLimitManager(
        api_keys=["k1", "k2"],
        models=["m1"],
        daily_sleep=True,
        cooldown_seconds=0.05,
    )
    mgr._key_state["k1"].daily_until = _next_midnight_ts()

    for _ in range(5):
        cfg = mgr.get_next_config()
        assert cfg["api_key"] != "k1"


def test_reset_failures_clears_daily_sleep():
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    mgr._key_state["k1"].daily_until = _next_midnight_ts()
    assert "k1" in mgr.failed_keys

    mgr.reset_failures()
    assert mgr.failed_keys == []
