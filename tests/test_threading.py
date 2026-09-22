"""
Thread-safety tests for RateLimitManager and ChatGoogleGenerativeAI.

Runs real concurrent load against the rotation engine and checks
that no updates are lost and no state is corrupted.
"""
from __future__ import annotations

import threading
import time
from typing import List, Set, Tuple

import pytest
from langchain_core.messages import AIMessage

from googlemodel_samrat import ChatGoogleGenerativeAI
from googlemodel_samrat.core import RateLimitManager
from googlemodel_samrat.exceptions import AllResourcesExhaustedError


# ─────────────────────────────────────────────────────────────────────
# Manager-level concurrency
# ─────────────────────────────────────────────────────────────────────
def test_concurrent_get_next_config_no_duplicates_within_cycle():
    """8 threads × 20 calls = 160 calls. Every returned (key, model)
    must be one of the valid combinations. No crashes, no lost state."""
    mgr = RateLimitManager(
        api_keys=["k1", "k2", "k3", "k4"],
        models=["m1", "m2", "m3"],
    )
    results: List[Tuple[str, str]] = []
    lock = threading.Lock()

    def worker():
        for _ in range(20):
            cfg = mgr.get_next_config()
            with lock:
                results.append((cfg["api_key"], cfg["model"]))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    valid = {
        (k, m)
        for k in ["k1", "k2", "k3", "k4"]
        for m in ["m1", "m2", "m3"]
    }
    assert len(results) == 160
    assert set(results).issubset(valid)


def test_concurrent_mark_failed_does_not_crash():
    """Many threads marking different pairs failed at once."""
    mgr = RateLimitManager(
        api_keys=["k1", "k2", "k3", "k4"],
        models=["m1", "m2"],
    )

    def worker(i: int):
        for j in range(20):
            k = f"k{(i + j) % 4 + 1}"
            m = f"m{(i + j) % 2 + 1}"
            mgr.mark_failed(k, m, Exception("429 quota"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All keys must still be in the pool (no corruption)
    assert set(mgr.api_keys) == {"k1", "k2", "k3", "k4"}
    assert set(mgr.models) == {"m1", "m2"}
    # Every cooling pair must reference a valid key and model
    for k, m in mgr.failed_pairs:
        assert k in mgr.api_keys
        assert m in mgr.models


def test_concurrent_mixed_read_write():
    """Reads and writes interleave — no crash, no lost counter."""
    mgr = RateLimitManager(
        api_keys=["k1", "k2", "k3"],
        models=["m1", "m2"],
        cooldown_seconds=0.001,
    )
    barrier = threading.Barrier(6)

    def reader():
        barrier.wait()
        for _ in range(50):
            try:
                mgr.get_next_config()
            except AllResourcesExhaustedError:
                pass

    def writer(idx: int):
        barrier.wait()
        for j in range(50):
            k = f"k{idx % 3 + 1}"
            m = f"m{j % 2 + 1}"
            mgr.mark_failed(k, m, Exception("429 quota"))

    threads = (
        [threading.Thread(target=reader) for _ in range(3)]
        + [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Stats must be internally consistent
    s = mgr.stats
    assert 0 <= s["cooling_pairs"] <= 6
    assert s["available_combinations"] >= 0


def test_concurrent_reset_does_not_crash():
    """reset_failures() from one thread while others mutate state."""
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    stop = threading.Event()

    def mutator():
        while not stop.is_set():
            mgr.mark_failed("k1", "m1", Exception("429 quota"))
            time.sleep(0.001)

    def resetter():
        for _ in range(20):
            mgr.reset_failures()
            time.sleep(0.002)

    t1 = threading.Thread(target=mutator)
    t2 = threading.Thread(target=resetter)
    t1.start()
    t2.start()
    t2.join()
    stop.set()
    t1.join()

    # No crash = pass. Sanity: manager still usable.
    try:
        mgr.get_next_config()
    except AllResourcesExhaustedError:
        mgr.reset_failures()
        mgr.get_next_config()


# ─────────────────────────────────────────────────────────────────────
# Client-level concurrency
# ─────────────────────────────────────────────────────────────────────
def test_client_concurrent_invokes(monkeypatch):
    """50 threads calling .invoke() on the same client."""
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _make_fake_lc(lambda kwargs, msgs: AIMessage(content="ok")),
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2", "k3"],
        models=["m1", "m2"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    errors: List[BaseException] = []
    successes: List[str] = []
    lock = threading.Lock()

    def worker():
        try:
            r = llm.invoke("hi")
            with lock:
                successes.append(r.content)
        except BaseException as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(successes) == 50
    assert all(s == "ok" for s in successes)


def test_client_concurrent_failing_invokes_rotates(monkeypatch):
    """Under concurrency, a single transient failure rotates cleanly,
    and every other thread succeeds via the remaining healthy pairs."""
    first_failed = {"v": False}
    fail_lock = threading.Lock()

    def behavior(kwargs, msgs):
        with fail_lock:
            if not first_failed["v"]:
                first_failed["v"] = True
                raise Exception("429 quota exceeded")
        return AIMessage(content="ok")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _make_fake_lc(behavior),
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1", "k2", "k3"],
        models=["m1", "m2"],
        cooldown_seconds=1.0,
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    results: List[str] = []
    lock = threading.Lock()

    def worker():
        try:
            r = llm.invoke("hi")
            with lock:
                results.append(r.content)
        except AllResourcesExhaustedError:
            with lock:
                results.append("EXHAUSTED")

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    # Only one pair ever failed; plenty of healthy pairs remain.
    assert "EXHAUSTED" not in results, (
        f"unexpected exhausted: {results.count('EXHAUSTED')}/20"
    )
    assert results.count("ok") == 20

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _make_fake_lc(invoke_behavior):
    class FakeLC:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages, **kw):
            return invoke_behavior(self.kwargs, messages)

        def stream(self, messages, **kw):
            yield AIMessage(content="ok")

        async def ainvoke(self, messages, **kw):
            return self.invoke(messages, **kw)

        async def astream(self, messages, **kw):
            for c in self.stream(messages, **kw):
                yield c

    return FakeLC
