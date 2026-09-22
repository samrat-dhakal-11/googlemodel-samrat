"""
ChatGoogleGenerativeAI integration tests.

Covers the LangChain-facing behaviour:
  - LCEL compatibility
  - Message-object inputs (not just strings)
  - Multi-turn conversations
  - Streaming + async streaming
  - Async invoke
  - Attribution metadata
  - Config passthrough (temperature, top_p, safety, extras)
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterator, List

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from googlemodel_samrat import ChatGoogleGenerativeAI
from googlemodel_samrat.exceptions import AllResourcesExhaustedError


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def _fake_lc_factory(
    invoke_behavior=None,
    stream_behavior=None,
):
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
# Basic invoke
# ─────────────────────────────────────────────────────────────────────
def test_invoke_returns_aimessage(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    msg = llm.invoke("hello")
    assert isinstance(msg, AIMessage)
    assert msg.content == "ok"


def test_invoke_accepts_message_list(monkeypatch):
    received: List[List[Any]] = []

    def invoke_behavior(kwargs, messages):
        received.append(messages)
        return AIMessage(content="got it")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    llm.invoke(msgs)
    assert received[0] == msgs


def test_invoke_accepts_prompt_template_chain(monkeypatch):
    def invoke_behavior(kwargs, messages):
        text = messages[0].content
        return AIMessage(content=f"echo: {text}")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    prompt = PromptTemplate.from_template("Ask about {topic}")
    chain = prompt | llm
    result = chain.invoke({"topic": "space"})
    assert isinstance(result, AIMessage)
    assert "Ask about space" in result.content


def test_invoke_accepts_chat_prompt_template_chain(monkeypatch):
    def invoke_behavior(kwargs, messages):
        return AIMessage(content=f"{len(messages)} messages")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=invoke_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful"),
        ("human", "{q}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"q": "hi"})
    assert result.content == "2 messages"


def test_lcel_with_str_output_parser(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=lambda kw, m: AIMessage(content="plain text")),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chain = llm | StrOutputParser()
    assert chain.invoke("hi") == "plain text"


# ─────────────────────────────────────────────────────────────────────
# Attribution metadata
# ─────────────────────────────────────────────────────────────────────
def test_last_successful_model_populated(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=lambda kw, m: AIMessage(content="x")),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1", "m2"])
    assert llm.last_successful_model is None
    llm.invoke("hi")
    assert llm.last_successful_model in {"m1", "m2"}


def test_last_successful_key_index_populated(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=lambda kw, m: AIMessage(content="x")),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1", "k2"], models=["m1"])
    assert llm.last_successful_key_index is None
    llm.invoke("hi")
    assert llm.last_successful_key_index in {0, 1}


def test_get_rotation_stats_returns_dict(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1", "k2"], models=["m1", "m2"])
    stats = llm.get_rotation_stats()
    assert isinstance(stats, dict)
    assert stats["total_keys"] == 2
    assert stats["total_models"] == 2


# ─────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────
def test_stream_yields_aimessagechunks(monkeypatch):
    def stream_behavior(kwargs, messages):
        yield AIMessageChunk(content="Hello ")
        yield AIMessageChunk(content="world!")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(stream_behavior=stream_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chunks = list(llm.stream("hi"))
    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(c.content for c in chunks) == "Hello world!"


def test_stream_via_lcel_chain(monkeypatch):
    def stream_behavior(kwargs, messages):
        yield AIMessageChunk(content="part1 ")
        yield AIMessageChunk(content="part2")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(stream_behavior=stream_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chain = llm | StrOutputParser()
    pieces = list(chain.stream("hi"))
    assert "".join(pieces) == "part1 part2"


# ─────────────────────────────────────────────────────────────────────
# Async
# ─────────────────────────────────────────────────────────────────────
async def test_ainvoke(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=lambda kw, m: AIMessage(content="async-ok")),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    msg = await llm.ainvoke("hi")
    assert msg.content == "async-ok"


async def test_astream(monkeypatch):
    def stream_behavior(kwargs, messages):
        yield AIMessageChunk(content="a")
        yield AIMessageChunk(content="b")
        yield AIMessageChunk(content="c")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(stream_behavior=stream_behavior),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chunks = [c.content async for c in llm.astream("hi")]
    assert "".join(chunks) == "abc"


async def test_lcel_async_chain(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(invoke_behavior=lambda kw, m: AIMessage(content="hey")),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    chain = llm | StrOutputParser()
    result = await chain.ainvoke("hi")
    assert result == "hey"


# ─────────────────────────────────────────────────────────────────────
# Config passthrough
# ─────────────────────────────────────────────────────────────────────
def test_temperature_passed_to_inner_client(monkeypatch):
    seen: List[dict] = []

    def make(**kw):
        seen.append(kw)
        class F:
            def __init__(self, **kwargs): self.kwargs = kwargs
            def invoke(self, messages, **kw): return AIMessage(content="ok")
            def stream(self, messages, **kw): yield AIMessageChunk(content="ok")
        return F(**kw)

    class FakeLC:
        def __init__(self, **kw):
            seen.append(kw)
            self.kwargs = kw
        def invoke(self, messages, **kw):
            return AIMessage(content="ok")
        def stream(self, messages, **kw):
            yield AIMessageChunk(content="ok")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        FakeLC,
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1"],
        models=["m1"],
        temperature=0.42,
        max_output_tokens=123,
    )
    llm.invoke("hi")
    assert seen[0]["temperature"] == 0.42
    assert seen[0]["max_output_tokens"] == 123


def test_top_p_top_k_passed(monkeypatch):
    seen: List[dict] = []

    class FakeLC:
        def __init__(self, **kw):
            seen.append(kw)
        def invoke(self, messages, **kw):
            return AIMessage(content="ok")
        def stream(self, messages, **kw):
            yield AIMessageChunk(content="ok")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        FakeLC,
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1"],
        models=["m1"],
        top_p=0.9,
        top_k=40,
    )
    llm.invoke("hi")
    assert seen[0]["top_p"] == 0.9
    assert seen[0]["top_k"] == 40


def test_extra_kwargs_passed_through(monkeypatch):
    seen: List[dict] = []

    class FakeLC:
        def __init__(self, **kw):
            seen.append(kw)
        def invoke(self, messages, **kw):
            return AIMessage(content="ok")
        def stream(self, messages, **kw):
            yield AIMessageChunk(content="ok")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        FakeLC,
    )
    llm = ChatGoogleGenerativeAI(
        api_keys=["k1"],
        models=["m1"],
        safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"},
        stop_sequences=["END"],
    )
    llm.invoke("hi")
    assert "safety_settings" in seen[0]
    assert seen[0]["stop_sequences"] == ["END"]


# ─────────────────────────────────────────────────────────────────────
# llm_type
# ─────────────────────────────────────────────────────────────────────
def test_llm_type_is_distinct(monkeypatch):
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        _fake_lc_factory(),
    )
    llm = ChatGoogleGenerativeAI(api_keys=["k1"], models=["m1"])
    assert llm._llm_type == "googlemodel-samrat-chat"
