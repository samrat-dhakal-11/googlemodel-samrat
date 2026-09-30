"""v0.2.0: GMS_VERBOSE env fallback + on_rotate rotation callback."""
import asyncio

import pytest

from googlemodel_samrat.core import RateLimitManager
from googlemodel_samrat.exceptions import AllResourcesExhaustedError


def _chat(**kw):
    from googlemodel_samrat.chat import ChatGoogleGenerativeAI

    base = dict(api_keys=["k1", "k2"], models=["m1"])
    base.update(kw)
    return ChatGoogleGenerativeAI(**base)


def _emb(**kw):
    from googlemodel_samrat.embeddings import GoogleGenerativeAIEmbeddings

    base = dict(api_keys=["k1"], models=["e1"])
    base.update(kw)
    return GoogleGenerativeAIEmbeddings(**base)


def _flaky():
    """Return (func, calls): raises once, then succeeds. Records call kwargs."""
    calls = []

    def func(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    return func, calls


# ── manager-level ────────────────────────────────────────────


def test_manager_stores_on_rotate():
    cb = lambda *a: None
    m = RateLimitManager(api_keys=["k"], models=["m"], on_rotate=cb)
    assert m.on_rotate is cb
    m2 = RateLimitManager(api_keys=["k"], models=["m"])
    assert m2.on_rotate is None


def test_fire_on_rotate_swallows_callback_errors():
    seen = []

    def bad_cb(key, model, error):
        seen.append(key)
        raise ValueError("monitor down")

    m = RateLimitManager(api_keys=["k"], models=["m"], on_rotate=bad_cb)
    m._fire_on_rotate("k", "m", RuntimeError("x"))  # must not raise
    assert seen == ["k"]


# ── rotation hook across all four execution paths ───────────


def test_on_rotate_fires_on_rotation(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    seen = []
    llm = _chat(on_rotate=lambda k, m, e: seen.append((k, m, e)))
    func, calls = _flaky()

    assert llm._execute_with_rotation(func) == "ok"
    assert len(seen) == 1
    key, model, err = seen[0]
    assert (key, model) == ("k1", "m1")
    assert isinstance(err, RuntimeError)
    assert [c["api_key"] for c in calls] == ["k1", "k2"]


def test_on_rotate_not_fired_on_final_attempt(monkeypatch):
    # Don't assume the max_attempts formula — derive the expectation.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    seen = []

    def always_fails(**kw):
        raise RuntimeError("boom")

    llm = _chat(api_keys=["k1"], on_rotate=lambda k, m, e: seen.append((k, m)))
    with pytest.raises(AllResourcesExhaustedError):
        llm._execute_with_rotation(always_fails)
    assert len(seen) == llm._manager.max_attempts - 1


def test_on_rotate_callback_error_does_not_kill_request(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    seen = []

    def bad_cb(key, model, error):
        seen.append(key)
        raise ValueError("monitor down")

    llm = _chat(on_rotate=bad_cb)
    func, calls = _flaky()
    assert llm._execute_with_rotation(func) == "ok"
    assert seen == ["k1"]
    assert len(calls) == 2


def test_on_rotate_fires_sync_stream(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    seen = []
    llm = _chat(on_rotate=lambda k, m, e: seen.append((k, m)))
    calls = []

    def gen(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("boom")
        yield "c1"
        yield "c2"

    assert list(llm._execute_stream_with_rotation(gen)) == ["c1", "c2"]
    assert seen == [("k1", "m1")]


def test_on_rotate_fires_async(monkeypatch):
    async def fake_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    seen = []
    llm = _chat(on_rotate=lambda k, m, e: seen.append((k, m)))
    calls = []

    async def aflaky(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"

    assert asyncio.run(llm._aexecute_with_rotation(aflaky)) == "ok"
    assert seen == [("k1", "m1")]


def test_on_rotate_fires_async_stream(monkeypatch):
    async def fake_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    seen = []
    llm = _chat(on_rotate=lambda k, m, e: seen.append((k, m)))
    calls = []

    async def agen(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("boom")
        yield "c1"
        yield "c2"

    async def collect():
        return [c async for c in llm._aexecute_stream_with_rotation(agen)]

    assert asyncio.run(collect()) == ["c1", "c2"]
    assert seen == [("k1", "m1")]


# ── client passthrough ───────────────────────────────────────


def test_chat_passes_on_rotate_to_manager():
    cb = lambda *a: None
    llm = _chat(api_keys=["k1"], on_rotate=cb)
    assert llm._manager.on_rotate is cb


def test_embeddings_passes_on_rotate_to_manager():
    cb = lambda *a: None
    emb = _emb(on_rotate=cb)
    assert emb._manager.on_rotate is cb


# ── GMS_VERBOSE env fallback ─────────────────────────────────


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_gms_verbose_env_truthy_chat(monkeypatch, val):
    monkeypatch.setenv("GMS_VERBOSE", val)
    assert _chat()._verbose_rotation is True


def test_gms_verbose_env_falsy_chat(monkeypatch):
    monkeypatch.setenv("GMS_VERBOSE", "0")
    assert _chat()._verbose_rotation is False


def test_gms_verbose_env_unset_chat(monkeypatch):
    monkeypatch.delenv("GMS_VERBOSE", raising=False)
    assert _chat()._verbose_rotation is False


def test_explicit_verbose_beats_env_chat(monkeypatch):
    monkeypatch.setenv("GMS_VERBOSE", "1")
    assert _chat(verbose=False)._verbose_rotation is False


def test_explicit_verbose_true_without_env_chat(monkeypatch):
    monkeypatch.delenv("GMS_VERBOSE", raising=False)
    assert _chat(verbose=True)._verbose_rotation is True


def test_gms_verbose_env_emb(monkeypatch):
    monkeypatch.setenv("GMS_VERBOSE", "on")
    assert _emb()._verbose_rotation is True


def test_gms_verbose_env_unset_emb(monkeypatch):
    monkeypatch.delenv("GMS_VERBOSE", raising=False)
    assert _emb()._verbose_rotation is False


def test_explicit_verbose_beats_env_emb(monkeypatch):
    monkeypatch.setenv("GMS_VERBOSE", "1")
    assert _emb(verbose=False)._verbose_rotation is False


# ── the noisy print is now verbose-gated ─────────────────────


def test_initializing_print_suppressed_when_not_verbose(monkeypatch, capsys):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    llm = _chat()
    func, _ = _flaky()
    assert llm._execute_with_rotation(func) == "ok"
    assert "Initializing" not in capsys.readouterr().out


def test_initializing_print_shown_when_verbose(monkeypatch, capsys):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    llm = _chat(verbose=True)
    func, _ = _flaky()
    assert llm._execute_with_rotation(func) == "ok"
    assert capsys.readouterr().out.count("Initializing") == 2