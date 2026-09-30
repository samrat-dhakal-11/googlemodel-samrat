"""
v0.2.0 flagship: rotation follows the model NAME.

chatmodel()/embeddingmodel() return RotatingModelName — a str
subclass carrying its failover pool. Rotation-aware clients (ours)
adopt the pool automatically; stock clients still get a plain str
and keep working unchanged. No monkey-patching, ever.
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

import googlemodel_samrat as gms
from googlemodel_samrat import (
    CHAT_MODELS,
    EMBEDDING_MODELS,
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    RotatingModelName,
    chatmodel,
    embeddingmodel,
)

# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def _fake_lc(behavior: Any = None):
    class FakeLC:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def invoke(self, messages, **kw):
            if behavior is None:
                return AIMessage(content="ok")
            return behavior(self.kwargs, messages)

        def stream(self, messages, **kw):
            yield AIMessageChunk(content="ok")

        async def ainvoke(self, messages, **kw):
            return self.invoke(messages, **kw)

        async def astream(self, messages, **kw):
            for c in self.stream(messages, **kw):
                yield c

    return FakeLC

def _fake_emb():
    class FakeEmb:
        def __init__(self, **kw):
            self.kwargs = kw

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    return FakeEmb

# ─────────────────────────────────────────────────────────────────────
# Layer 1 — RotatingModelName is a str
# ─────────────────────────────────────────────────────────────────────
def test_rotating_model_name_is_str():
    name = chatmodel()
    assert isinstance(name, str)
    assert isinstance(name, RotatingModelName)
    assert "gemini" in name
    assert name + "-x" == f"{name}-x"
    assert len(name) > 0
    assert name.upper() == str(name).upper()

def test_rotating_model_name_carries_pool():
    assert chatmodel().rotation_pool == list(CHAT_MODELS)
    assert embeddingmodel().rotation_pool == list(EMBEDDING_MODELS)

def test_rotating_model_name_survives_str_ops():
    name = chatmodel()
    assert str(name) == CHAT_MODELS[0]
    assert name in CHAT_MODELS
    assert [name][0] == CHAT_MODELS[0]
    assert name == CHAT_MODELS[0]

# ─────────────────────────────────────────────────────────────────────
# Layer 2 — rotation-aware clients adopt the pool
# ─────────────────────────────────────────────────────────────────────
def test_client_adopts_pool_from_singular_rotating_name(monkeypatch):
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _fake_lc())
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], model=chatmodel())
    stats = llm.get_rotation_stats()
    assert stats["total_models"] == len(CHAT_MODELS)
    assert stats["model_rotation_enabled"] is True

def test_plain_string_model_stays_single(monkeypatch):
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _fake_lc())
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], model="gemini-2.5-flash")
    assert llm.get_rotation_stats()["total_models"] == 1

def test_explicit_models_list_not_expanded(monkeypatch):
    """An explicit models=[...] is a deliberate choice — never rewritten."""
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _fake_lc())
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=[chatmodel(), "m2"])
    assert llm.get_rotation_stats()["total_models"] == 2

def test_singular_api_key_alias_works(monkeypatch):
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _fake_lc())
    llm = ChatGoogleGenerativeAI(api_key="k1", model="m1")
    assert llm.get_rotation_stats()["total_keys"] == 1
    llm.invoke("hi")  # aliases must not leak into the inner client kwargs

def test_embeddings_adopt_pool(monkeypatch):
    monkeypatch.setattr("langchain_google_genai.GoogleGenerativeAIEmbeddings", _fake_emb())
    emb = GoogleGenerativeAIEmbeddings(api_keys=["k1"], model=embeddingmodel())
    assert emb.get_rotation_stats()["total_models"] == len(EMBEDDING_MODELS)

def test_manager_bound_after_adoption(monkeypatch):
    """A manager can't exist before adoption (RateLimitManager requires
    API keys), so the name's _manager is bound by the adopting client."""
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _fake_lc())
    name = chatmodel()
    assert name._manager is None          # pure getter, no side effects
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], model=name)
    assert name._manager is not None
    assert name._manager is llm._manager

# ─────────────────────────────────────────────────────────────────────
# Layer 3 — the langchain shim
# ─────────────────────────────────────────────────────────────────────
def test_langchain_shim_import_path():
    from googlemodel_samrat.langchain import (
        ChatGoogleGenerativeAI as ShimChat,
        GoogleGenerativeAIEmbeddings as ShimEmb,
    )
    assert ShimChat is gms.ChatGoogleGenerativeAI
    assert ShimEmb is gms.GoogleGenerativeAIEmbeddings

def test_shim_class_is_base_chat_model():
    from googlemodel_samrat.langchain import ChatGoogleGenerativeAI as ShimChat
    assert issubclass(ShimChat, BaseChatModel)

def test_shim_rotates_on_429(monkeypatch):
    seen: List[str] = []

    def behavior(kwargs, msgs):
        seen.append(kwargs["google_api_key"])
        if len(seen) == 1:
            raise Exception("429 quota exceeded")
        return AIMessage(content="recovered")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI", _fake_lc(behavior)
    )
    from googlemodel_samrat.langchain import ChatGoogleGenerativeAI as ShimChat

    llm = ShimChat(
        api_keys=["k1", "k2"], models=["m1"],
        initial_backoff=0.0, max_backoff=0.0,
    )
    assert llm.invoke("hi").content == "recovered"
    assert seen == ["k1", "k2"]

def test_no_monkey_patch_by_default():
    """Importing googlemodel_samrat must NEVER silently replace the
    stock langchain_google_genai classes."""
    import langchain_google_genai
    import googlemodel_samrat  # noqa: F401

    assert langchain_google_genai.ChatGoogleGenerativeAI is not gms.ChatGoogleGenerativeAI
    assert langchain_google_genai.GoogleGenerativeAIEmbeddings is not gms.GoogleGenerativeAIEmbeddings