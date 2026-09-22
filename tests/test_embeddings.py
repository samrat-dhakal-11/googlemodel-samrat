"""
GoogleGenerativeAIEmbeddings tests.

Covers:
  - Construction with singular/plural args
  - embed_query / embed_documents
  - Rotation on failure
  - Attribution metadata
  - LangChain Embeddings interface compliance
  - Empty & edge-case inputs
"""
from __future__ import annotations

from typing import Any, List

import pytest
from langchain_core.embeddings import Embeddings

from googlemodel_samrat import GoogleGenerativeAIEmbeddings
from googlemodel_samrat.exceptions import (
    AllResourcesExhaustedError,
    ConfigurationError,
)


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def _fake_emb_factory(query_behavior=None, docs_behavior=None):
    class FakeEmb:
        def __init__(self, **kw):
            self.kwargs = kw

        def embed_query(self, text: str):
            if query_behavior is None:
                return [0.1, 0.2, 0.3]
            return query_behavior(self.kwargs, text)

        def embed_documents(self, texts: List[str]):
            if docs_behavior is None:
                return [[0.1, 0.2] for _ in texts]
            return docs_behavior(self.kwargs, texts)

    return FakeEmb


# ─────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────
def test_singular_api_key(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_key="k1")
    assert emb._manager.api_keys == ["k1"]


def test_singular_model(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_key="k1", model="e1")
    assert emb._manager.models == ["e1"]


def test_plural_api_keys(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_keys=["k1", "k2"])
    assert emb._manager.api_keys == ["k1", "k2"]


def test_plural_models(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1"],
        models=["e1", "e2"],
    )
    assert emb._manager.models == ["e1", "e2"]


def test_no_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    with pytest.raises(ConfigurationError):
        GoogleGenerativeAIEmbeddings(api_keys=[])


def test_isinstance_langchain_embeddings(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_key="k1")
    assert isinstance(emb, Embeddings)


# ─────────────────────────────────────────────────────────────────────
# embed_query
# ─────────────────────────────────────────────────────────────────────
def test_embed_query_returns_floats(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_key="k1")
    vec = emb.embed_query("hello")
    assert vec == [0.1, 0.2, 0.3]


def test_embed_query_rotates_on_failure(monkeypatch):
    seen: List[str] = []

    def query_behavior(kwargs, text):
        key = kwargs["google_api_key"]
        seen.append(key)
        if key == "k1":
            raise Exception("429 quota exceeded")
        return [1.0, 2.0]

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(query_behavior=query_behavior),
    )
    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1", "k2"],
        models=["e1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    vec = emb.embed_query("hi")
    assert vec == [1.0, 2.0]
    assert "k1" in seen and "k2" in seen


def test_embed_query_rotates_model_on_404(monkeypatch):
    seen: List[str] = []

    def query_behavior(kwargs, text):
        m = kwargs["model"]
        seen.append(m)
        if m == "e1":
            raise Exception("404 model not found")
        return [3.0]

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(query_behavior=query_behavior),
    )
    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1"],
        models=["e1", "e2"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    vec = emb.embed_query("hi")
    assert vec == [3.0]
    assert "e2" in seen


def test_embed_query_all_failures_raise(monkeypatch):
    def query_behavior(kwargs, text):
        raise Exception("429 quota exceeded")

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(query_behavior=query_behavior),
    )
    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1"],
        models=["e1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    with pytest.raises(AllResourcesExhaustedError):
        emb.embed_query("hi")


# ─────────────────────────────────────────────────────────────────────
# embed_documents
# ─────────────────────────────────────────────────────────────────────
def test_embed_documents_returns_list_of_lists(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_key="k1")
    vecs = emb.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(isinstance(v, list) for v in vecs)


def test_embed_documents_empty_list(monkeypatch):
    def docs_behavior(kwargs, texts):
        return []

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(docs_behavior=docs_behavior),
    )
    emb = GoogleGenerativeAIEmbeddings(api_key="k1")
    vecs = emb.embed_documents([])
    assert vecs == []


def test_embed_documents_rotates_on_failure(monkeypatch):
    seen: List[str] = []

    def docs_behavior(kwargs, texts):
        key = kwargs["google_api_key"]
        seen.append(key)
        if key == "k1":
            raise Exception("429 quota exceeded")
        return [[1.0], [2.0]]

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(docs_behavior=docs_behavior),
    )
    emb = GoogleGenerativeAIEmbeddings(
        api_keys=["k1", "k2"],
        models=["e1"],
        initial_backoff=0.0,
        max_backoff=0.0,
    )
    vecs = emb.embed_documents(["a", "b"])
    assert vecs == [[1.0], [2.0]]


# ─────────────────────────────────────────────────────────────────────
# Attribution
# ─────────────────────────────────────────────────────────────────────
def test_attribution_populated(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_keys=["k1"], models=["e1"])
    assert emb.last_successful_model is None
    assert emb.last_successful_key_index is None
    emb.embed_query("hi")
    assert emb.last_successful_model == "e1"
    assert emb.last_successful_key_index == 0


def test_get_rotation_stats(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        _fake_emb_factory(),
    )
    emb = GoogleGenerativeAIEmbeddings(api_keys=["k1", "k2"], models=["e1"])
    stats = emb.get_rotation_stats()
    assert stats["total_keys"] == 2
    assert stats["total_models"] == 1
