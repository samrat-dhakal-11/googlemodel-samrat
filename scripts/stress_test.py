#!/usr/bin/env python3
"""
Ultimate stress test for googlemodel-samrat.

Exercises every layer in one run:

  1. Registry health (all 12 categories)
  2. Chat invoke — sync, streaming, async, async streaming
  3. LCEL chains — simple, with parser, RAG-style
  4. Embeddings — query, batch, async-style
  5. Multi-key rotation (simulated)
  6. Model fallback (simulated)
  7. Concurrent load (threads + asyncio)
  8. Mid-stream failure recovery
  9. Daily-quota midnight sleep (simulated)
 10. Attribution accuracy after every call
 11. Rotations stats sanity

Every step prints PASS/FAIL with timing. Exit code is 1 if anything
failed, 0 otherwise. Safe for CI: no real API calls unless a key
is present, and even then it batches them tightly.
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import threading
import warnings

warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", message=r".*sampling parameter.*will be ignored.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"langchain_google_genai.*"
)
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from googlemodel_samrat import (
    CHAT_MODELS,
    EMBEDDING_MODELS,
    AllResourcesExhaustedError,
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    chatmodel,
    embeddingmodel,
    get_model_count,
    get_models,
    model_exists,
)
from googlemodel_samrat.core import RateLimitManager, _next_midnight_ts
from googlemodel_samrat.exceptions import ConfigurationError

load_dotenv()

# ─── Reporting ────────────────────────────────────────────────────────
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_counts = {PASS: 0, FAIL: 0, SKIP: 0}
_timings: List[Tuple[str, float]] = []


def banner(text: str) -> None:
    print(f"\n{'═' * 72}\n  {text}\n{'═' * 72}")


def report(name: str, ok: Optional[bool], detail: str = "", elapsed: float = 0.0) -> None:
    tag = SKIP if ok is None else (PASS if ok else FAIL)
    _counts[tag] += 1
    marker = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[tag]
    suffix = f"  [{elapsed:.2f}s]" if elapsed else ""
    line = f"{marker} {name}{suffix}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    if elapsed:
        _timings.append((name, elapsed))


def timeit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float, Optional[BaseException]]:
    t0 = time.perf_counter()
    err: Optional[BaseException] = None
    result: Any = None
    try:
        result = fn(*args, **kwargs)
    except BaseException as e:
        err = e
    return result, time.perf_counter() - t0, err


# ─── Discover keys ────────────────────────────────────────────────────
def discover_keys() -> List[str]:
    keys = []
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3"):
        v = os.getenv(name)
        if v:
            keys.append(v)
    return keys


KEYS = discover_keys()
HAS_KEYS = bool(KEYS)


# ─── Fake backend for offline rotation tests ──────────────────────────
def fake_lc_factory(
    invoke_behavior=None,
    stream_behavior=None,
    chunk_delay: float = 0.0,
):
    class FakeLC:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def invoke(self, messages, **kw):
            if invoke_behavior is None:
                return _aimsg("fake response")
            return invoke_behavior(self.kwargs, messages)

        def stream(self, messages, **kw):
            if stream_behavior is None:
                yield _chunk("fake ")
                yield _chunk("stream")
                return
            for c in stream_behavior(self.kwargs, messages):
                if chunk_delay:
                    time.sleep(chunk_delay)
                yield c

        async def ainvoke(self, messages, **kw):
            return self.invoke(messages, **kw)

        async def astream(self, messages, **kw):
            for c in self.stream(messages, **kw):
                yield c

    return FakeLC


def _aimsg(text: str):
    from langchain_core.messages import AIMessage
    return AIMessage(content=text)


def _chunk(text: str):
    from langchain_core.messages import AIMessageChunk
    return AIMessageChunk(content=text)


# ═════════════════════════════════════════════════════════════════════
# 1. Registry
# ═════════════════════════════════════════════════════════════════════
def test_registry() -> None:
    banner("1. Registry health")

    cm = chatmodel()
    report("chatmodel() returns a string", isinstance(cm, str) and bool(cm), cm)
    em = embeddingmodel()
    report("embeddingmodel() returns a string", isinstance(em, str) and bool(em), em)
    report("get_model_count() > 0", get_model_count() > 0, f"{get_model_count()} models")
    report("model_exists(chatmodel())", model_exists(cm))

    cats = ["chat", "text", "audio", "image", "video", "embedding",
            "music", "robotics", "computer_use", "research", "agent", "gemma"]
    for cat in cats:
        models = get_models(cat)
        report(f"get_models({cat!r}) non-empty", len(models) > 0, f"{len(models)} models")

    report("CHAT_MODELS[0] == chatmodel()", CHAT_MODELS[0] == cm)
    report("EMBEDDING_MODELS[0] == embeddingmodel()", EMBEDDING_MODELS[0] == em)


# ═════════════════════════════════════════════════════════════════════
# 2. Config validation
# ═════════════════════════════════════════════════════════════════════
def test_config() -> None:
    banner("2. Config validation")

    try:
        RateLimitManager(api_keys=[], models=["m1"])
        report("empty keys raises ConfigurationError", False)
    except ConfigurationError:
        report("empty keys raises ConfigurationError", True)

    try:
        RateLimitManager(api_keys=["k"], models=[])
        report("empty models raises ConfigurationError", False)
    except ConfigurationError:
        report("empty models raises ConfigurationError", True)

    try:
        RateLimitManager(api_keys=["  ", ""], models=["m1"])
        report("whitespace-only keys raise", False)
    except ConfigurationError:
        report("whitespace-only keys raise", True)

    mgr = RateLimitManager(api_keys=["k"], models=["m1", "m2"])
    report("single-key no key rotation", mgr.has_key_rotation is False)
    report("multi-model has model rotation", mgr.has_model_rotation is True)
    report("max_attempts == 2", mgr.max_attempts == 2)


# ═════════════════════════════════════════════════════════════════════
# 3. Rotation mechanics (offline)
# ═════════════════════════════════════════════════════════════════════
def test_rotation_mechanics() -> None:
    banner("3. Rotation mechanics")

    # Pair cooldown
    mgr = RateLimitManager(api_keys=["k1", "k2"], models=["m1", "m2"])
    mgr.mark_failed("k1", "m1", Exception("429 quota"))
    report("429 cools pair, not key", mgr.failed_keys == [])
    report("429 cools pair, not model", mgr.failed_models == [])
    report("429 pair tracked", ("k1", "m1") in mgr.failed_pairs)

    # Next pick should avoid the pair
    cfg = mgr.get_next_config()
    report("next pick avoids dead pair", (cfg["api_key"], cfg["model"]) != ("k1", "m1"))

    # 404 → model only
    mgr2 = RateLimitManager(api_keys=["k1"], models=["m1", "m2"])
    mgr2.mark_failed("k1", "m1", Exception("404 model not found"))
    report("404 cools model only", mgr2.failed_models == ["m1"] and mgr2.failed_keys == [])

    # 5xx → key only
    mgr3 = RateLimitManager(api_keys=["k1", "k2"], models=["m1"])
    mgr3.mark_failed("k1", "m1", Exception("503 service unavailable"))
    report("5xx cools key only", mgr3.failed_keys == ["k1"] and mgr3.failed_models == [])

    # Daily quota → pair until midnight
    mgr4 = RateLimitManager(api_keys=["k1"], models=["m1"])
    mgr4.mark_failed("k1", "m1", Exception("429 daily quota exceeded for the day"))
    pair = mgr4._pair_state[("k1", "m1")]
    report("daily quota sleeps until midnight",
           abs(pair.daily_until - _next_midnight_ts()) < 5)

    # shortest_wait
    mgr5 = RateLimitManager(api_keys=["k1"], models=["m1"], cooldown_seconds=30)
    mgr5.mark_failed("k1", "m1", Exception("429 quota"))
    report("shortest_wait positive after failure", 0 < mgr5.shortest_wait() <= 30)

    # Stats
    s = mgr5.stats
    report("stats has cooling_pairs", s["cooling_pairs"] == 1)
    report("stats available_combinations drops", s["available_combinations"] == 0)


# ═════════════════════════════════════════════════════════════════════
# 4. Offline chat: sync, streaming, async
# ═════════════════════════════════════════════════════════════════════
def test_offline_chat_sync(monkeypatch_available: bool = True) -> None:
    banner("4. Offline chat — sync / streaming / async")

    # Note: we can't easily monkeypatch outside pytest, so we import the
    # inner LangChain class and swap it temporarily.
    import langchain_google_genai as lgg

    original = lgg.ChatGoogleGenerativeAI

    def run_with(fake):
        lgg.ChatGoogleGenerativeAI = fake
        try:
            # -------- sync invoke --------
            llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
            msg, dt, err = timeit(llm.invoke, "hello")
            report("sync .invoke() works", err is None and hasattr(msg, "content"),
                   f"content={msg.content[:20]!r}" if msg else str(err)[:40], dt)

            # -------- streaming --------
            chunks = []
            def stream_all():
                for c in llm.stream("hi"):
                    chunks.append(c.content)
            _, dt, err = timeit(stream_all)
            report("sync .stream() works",
                   err is None and len(chunks) > 0,
                   f"{len(chunks)} chunks → {''.join(chunks)[:30]!r}", dt)

            # -------- async --------
            async def go():
                return await llm.ainvoke("hi")
            msg2, dt, err = timeit(lambda: asyncio.run(go()))
            report("async .ainvoke() works",
                   err is None and hasattr(msg2, "content"),
                   f"content={msg2.content[:20]!r}" if msg2 else str(err)[:40], dt)

            # -------- async streaming --------
            async def go_stream():
                out = []
                async for c in llm.astream("hi"):
                    out.append(c.content)
                return out
            parts, dt, err = timeit(lambda: asyncio.run(go_stream()))
            report("async .astream() works",
                   err is None and len(parts) > 0,
                   f"{len(parts)} chunks", dt)
        finally:
            lgg.ChatGoogleGenerativeAI = original

    run_with(fake_lc_factory())


# ═════════════════════════════════════════════════════════════════════
# 5. Offline LCEL chains
# ═════════════════════════════════════════════════════════════════════
def test_offline_lcel() -> None:
    banner("5. Offline LCEL chains")

    import langchain_google_genai as lgg
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    original = lgg.ChatGoogleGenerativeAI

    def behavior(kwargs, msgs):
        text = msgs[0].content if msgs else ""
        return _aimsg(f"echo: {text[:40]}")

    lgg.ChatGoogleGenerativeAI = fake_lc_factory(invoke_behavior=behavior)
    try:
        llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
        prompt = PromptTemplate.from_template("Tell me about {topic}")
        chain = prompt | llm | StrOutputParser()
        result, dt, err = timeit(chain.invoke, {"topic": "space"})
        report("LCEL prompt | llm | parser",
               err is None and "Tell me about space" in result,
               result[:60] if result else str(err)[:60], dt)

        # Direct Runnable
        chain2 = llm | StrOutputParser()
        result2, dt, err = timeit(chain2.invoke, "direct")
        report("LCEL llm | parser",
               err is None and "echo:" in result2,
               result2[:40] if result2 else str(err)[:40], dt)
    finally:
        lgg.ChatGoogleGenerativeAI = original


# ═════════════════════════════════════════════════════════════════════
# 6. Offline embeddings
# ═════════════════════════════════════════════════════════════════════
def test_offline_embeddings() -> None:
    banner("6. Offline embeddings")

    import langchain_google_genai as lgg
    original = lgg.GoogleGenerativeAIEmbeddings

    class FakeEmb:
        def __init__(self, **kw): self.kwargs = kw
        def embed_query(self, text): return [0.1, 0.2, 0.3, 0.4]
        def embed_documents(self, texts): return [[0.1, 0.2] for _ in texts]

    lgg.GoogleGenerativeAIEmbeddings = FakeEmb
    try:
        emb = GoogleGenerativeAIEmbeddings(api_keys=["k1"], models=["e1"])
        vec, dt, err = timeit(emb.embed_query, "hello")
        report("embed_query returns list",
               err is None and isinstance(vec, list) and len(vec) > 0,
               f"dim={len(vec)}" if vec else str(err)[:40], dt)

        vecs, dt, err = timeit(emb.embed_documents, ["a", "b", "c"])
        report("embed_documents returns list of lists",
               err is None and len(vecs) == 3,
               f"{len(vecs)} vectors" if vecs else str(err)[:40], dt)

        report("attribution — last_successful_model",
               emb.last_successful_model == "e1")
        report("attribution — last_successful_key_index",
               emb.last_successful_key_index == 0)
    finally:
        lgg.GoogleGenerativeAIEmbeddings = original


# ═════════════════════════════════════════════════════════════════════
# 7. Fallback under simulated failure
# ═════════════════════════════════════════════════════════════════════
def test_offline_fallback() -> None:
    banner("7. Fallback under simulated failure")

    import langchain_google_genai as lgg
    original = lgg.ChatGoogleGenerativeAI

    # Case A: first call 429s, second succeeds (key rotation)
    calls = {"n": 0}
    def behavior_a(kwargs, msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("429 quota exceeded")
        return _aimsg(f"recovered via {kwargs['google_api_key']}")

    lgg.ChatGoogleGenerativeAI = fake_lc_factory(invoke_behavior=behavior_a)
    try:
        llm = ChatGoogleGenerativeAI(
            api_keys=["k1", "k2"], models=["m1"],
            initial_backoff=0.0, max_backoff=0.0,
        )
        msg, dt, err = timeit(llm.invoke, "hi")
        report("key rotation on 429",
               err is None and "recovered" in msg.content,
               msg.content if msg else str(err)[:60], dt)
    finally:
        lgg.ChatGoogleGenerativeAI = original

    # Case B: first model 404s, second succeeds (model rotation)
    original = lgg.ChatGoogleGenerativeAI
    def behavior_b(kwargs, msgs):
        if kwargs["model"] == "m1":
            raise Exception("404 model not found")
        return _aimsg(f"served by {kwargs['model']}")

    lgg.ChatGoogleGenerativeAI = fake_lc_factory(invoke_behavior=behavior_b)
    try:
        llm = ChatGoogleGenerativeAI(
            api_keys=["k1"], models=["m1", "m2"],
            initial_backoff=0.0, max_backoff=0.0,
        )
        msg, dt, err = timeit(llm.invoke, "hi")
        report("model rotation on 404",
               err is None and "served by m2" in msg.content,
               msg.content if msg else str(err)[:60], dt)
    finally:
        lgg.ChatGoogleGenerativeAI = original

    # Case C: all pairs fail → friendly error
    original = lgg.ChatGoogleGenerativeAI
    def behavior_c(kwargs, msgs):
        raise Exception("429 quota exceeded")

    lgg.ChatGoogleGenerativeAI = fake_lc_factory(invoke_behavior=behavior_c)
    try:
        llm = ChatGoogleGenerativeAI(
            api_keys=["k1"], models=["m1"],
            initial_backoff=0.0, max_backoff=0.0,
        )
        _, dt, err = timeit(llm.invoke, "hi")
        report("all failures raise AllResourcesExhaustedError",
               isinstance(err, AllResourcesExhaustedError),
               type(err).__name__ if err else "no error", dt)
        if isinstance(err, AllResourcesExhaustedError):
            report("error message has troubleshooting",
                   "Troubleshooting" in str(err))
    finally:
        lgg.ChatGoogleGenerativeAI = original


# ═════════════════════════════════════════════════════════════════════
# 8. Concurrency
# ═════════════════════════════════════════════════════════════════════
def test_concurrency() -> None:
    banner("8. Concurrency")

    # Threads on RateLimitManager
    mgr = RateLimitManager(
        api_keys=["k1", "k2", "k3", "k4"],
        models=["m1", "m2", "m3"],
    )
    results: List[Tuple[str, str]] = []
    lock = threading.Lock()

    def worker():
        for _ in range(30):
            cfg = mgr.get_next_config()
            with lock:
                results.append((cfg["api_key"], cfg["model"]))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    dt = time.perf_counter() - t0

    valid = {(k, m) for k in ["k1", "k2", "k3", "k4"] for m in ["m1", "m2", "m3"]}
    report("300 threaded get_next_config calls",
           len(results) == 300 and set(results).issubset(valid),
           f"{len(results)} results, {len(set(results))} unique", dt)

    # Threads on ChatGoogleGenerativeAI
    import langchain_google_genai as lgg
    original = lgg.ChatGoogleGenerativeAI
    lgg.ChatGoogleGenerativeAI = fake_lc_factory(invoke_behavior=lambda kw, m: _aimsg("ok"))
    try:
        llm = ChatGoogleGenerativeAI(
            api_keys=["k1", "k2", "k3"],
            models=["m1", "m2"],
            initial_backoff=0.0, max_backoff=0.0,
        )
        successes = []
        fails = []
        def worker2():
            try:
                r = llm.invoke("hi")
                successes.append(r.content)
            except BaseException as e:
                fails.append(e)

        threads = [threading.Thread(target=worker2) for _ in range(30)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        dt = time.perf_counter() - t0

        report("30 threaded .invoke() calls",
               len(successes) == 30 and len(fails) == 0,
               f"{len(successes)} ok, {len(fails)} failed", dt)
    finally:
        lgg.ChatGoogleGenerativeAI = original

    # Async concurrency
    import langchain_google_genai as lgg
    original = lgg.ChatGoogleGenerativeAI
    lgg.ChatGoogleGenerativeAI = fake_lc_factory(invoke_behavior=lambda kw, m: _aimsg("async"))
    try:
        llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
        async def go():
            return await asyncio.gather(*[llm.ainvoke(f"msg{i}") for i in range(20)])
        results, dt, err = timeit(lambda: asyncio.run(go()))
        report("20 concurrent async .ainvoke() calls",
               err is None and len(results) == 20,
               f"{len(results)} responses" if results else str(err)[:40], dt)
    finally:
        lgg.ChatGoogleGenerativeAI = original


# ═════════════════════════════════════════════════════════════════════
# 9. Mid-stream failure recovery
# ═════════════════════════════════════════════════════════════════════
def test_midstream_failure() -> None:
    banner("9. Mid-stream failure recovery")

    import langchain_google_genai as lgg
    original = lgg.ChatGoogleGenerativeAI
    attempts = {"n": 0}

    def stream_behavior(kwargs, msgs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Exception("503 service unavailable")
        yield _chunk("recovered ")
        yield _chunk("stream")

    lgg.ChatGoogleGenerativeAI = fake_lc_factory(stream_behavior=stream_behavior)
    try:
        llm = ChatGoogleGenerativeAI(
            api_keys=["k1", "k2"], models=["m1"],
            initial_backoff=0.0, max_backoff=0.0,
        )
        text, dt, err = timeit(lambda: "".join(c.content for c in llm.stream("hi")))
        report("stream recovers from first-chunk 503",
               err is None and text == "recovered stream",
               text if text else str(err)[:40], dt)
    finally:
        lgg.ChatGoogleGenerativeAI = original


# ═════════════════════════════════════════════════════════════════════
# 10. Attribution & stats sanity
# ═════════════════════════════════════════════════════════════════════
def test_attribution_and_stats() -> None:
    banner("10. Attribution & stats")

    import langchain_google_genai as lgg
    original = lgg.ChatGoogleGenerativeAI
    lgg.ChatGoogleGenerativeAI = fake_lc_factory(
        invoke_behavior=lambda kw, m: _aimsg("ok")
    )
    try:
        llm = ChatGoogleGenerativeAI(api_keys=["k1", "k2"], models=["m1", "m2"])
        report("last_successful_model is None before first call",
               llm.last_successful_model is None)
        report("last_successful_key_index is None before first call",
               llm.last_successful_key_index is None)

        llm.invoke("hi")
        report("last_successful_model set after invoke",
               llm.last_successful_model in {"m1", "m2"},
               str(llm.last_successful_model))
        report("last_successful_key_index set after invoke",
               llm.last_successful_key_index in {0, 1},
               str(llm.last_successful_key_index))

        s = llm.get_rotation_stats()
        report("stats has total_keys", s["total_keys"] == 2)
        report("stats has total_models", s["total_models"] == 2)
        report("stats has cooling_pairs", "cooling_pairs" in s)
    finally:
        lgg.ChatGoogleGenerativeAI = original


# ═════════════════════════════════════════════════════════════════════
# 11. Live API (optional)
# ═════════════════════════════════════════════════════════════════════
def test_live() -> None:
    banner("11. Live API")

    if not HAS_KEYS:
        report("live invoke (skipped: no GEMINI_API_KEY)", None)
        report("live stream (skipped)", None)
        report("live embeddings (skipped)", None)
        report("live LCEL chain (skipped)", None)
        return

    print(f"  Using {len(KEYS)} key(s)")

    llm = ChatGoogleGenerativeAI(api_keys=KEYS)

    msg, dt, err = timeit(llm.invoke, "Reply with exactly: ready")
    report("live invoke", err is None and hasattr(msg, "content"),
           (msg.content[:30] if msg else str(err)[:60]), dt)
    if msg:
        report(f"  attribution model", bool(llm.last_successful_model),
               str(llm.last_successful_model))

    chunks: List[str] = []
    def stream_all():
        for c in llm.stream("Count to 3."):
            chunks.append(c.content)
    _, dt, err = timeit(stream_all)
    report("live stream", err is None and len(chunks) > 0,
           f"{len(chunks)} chunks", dt)

    emb = GoogleGenerativeAIEmbeddings(api_keys=KEYS)
    vec, dt, err = timeit(emb.embed_query, "hello world")
    report("live embeddings", err is None and isinstance(vec, list),
           f"dim={len(vec)}" if vec else str(err)[:60], dt)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("Say hi in 3 words.")
    chain = prompt | llm | StrOutputParser()
    text, dt, err = timeit(chain.invoke, {})
    report("live LCEL chain", err is None and isinstance(text, str),
           (text[:40] if text else str(err)[:60]), dt)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
def main() -> int:
    print("╔" + "═" * 70 + "╗")
    print("║  googlemodel-samrat — ULTIMATE STRESS TEST".ljust(71) + "║")
    print("╚" + "═" * 70 + "╝")
    print(f"  keys found: {len(KEYS)}")
    print(f"  chatmodel(): {chatmodel()}")
    print(f"  embeddingmodel(): {embeddingmodel()}")
    print(f"  total models registered: {get_model_count()}")

    t0 = time.perf_counter()

    test_registry()
    test_config()
    test_rotation_mechanics()
    test_offline_chat_sync()
    test_offline_lcel()
    test_offline_embeddings()
    test_offline_fallback()
    test_concurrency()
    test_midstream_failure()
    test_attribution_and_stats()
    # test_live()  # disabled to save free-tier quota during dry-runs

    total = time.perf_counter() - t0

    banner("SUMMARY")
    print(f"  PASS: {_counts[PASS]}")
    print(f"  FAIL: {_counts[FAIL]}")
    print(f"  SKIP: {_counts[SKIP]}")
    print(f"  Total time: {total:.2f}s")

    if _timings:
        print("\n  Slowest operations:")
        for name, elapsed in sorted(_timings, key=lambda x: -x[1])[:5]:
            print(f"    {elapsed:6.2f}s  {name}")

    print()
    if _counts[FAIL]:
        print("  ❌ SOME CHECKS FAILED")
        return 1
    print("  ✅ ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
