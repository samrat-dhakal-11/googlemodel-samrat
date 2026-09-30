"""
LangChain-native chat client with automatic key & model rotation.

Fully LCEL-compatible: inherits from BaseChatModel, supports
`|` pipelines, `.invoke()`, `.stream()`, `.ainvoke()`, `.astream()`.

v0.2.0:
  - ONE LangSmith trace per call (inner client driven via its
    protected _generate/_stream — no second root run, no
    double-counted tokens/costs/latency).
  - Singular api_key=/model= aliases work as documented.
  - model=chatmodel() adopts the model's whole failover pool.
  - Public failed_pairs / reset_failures() / __repr__.
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from .core import (
    RateLimitManager,
    RotationExecutionMixin,
    _silence_sdk_warnings,
    quiet_sdk_warnings,
)
from .exceptions import AllResourcesExhaustedError, ConfigurationError
from .registry import CHAT_MODELS, RotatingModelName


class ChatGoogleGenerativeAI(BaseChatModel, RotationExecutionMixin):
    """
    Drop-in, LCEL-compatible replacement for
    `langchain_google_genai.ChatGoogleGenerativeAI` with automatic
    API key rotation and model fallback.

    Example:
        >>> from googlemodel_samrat import ChatGoogleGenerativeAI, chatmodel
        >>> llm = ChatGoogleGenerativeAI(
        ...     api_keys=["key1", "key2"],
        ...     model=chatmodel(),          # adopts the FULL chat pool
        ...     temperature=0.7,
        ... )
        >>> llm.invoke("Hello!")
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # ── public config ────────────────────────────────────────────────
    api_keys: Optional[List[str]] = Field(default=None)
    models: Optional[List[str]] = Field(default=None)
    temperature: float = Field(default=0.7)
    max_output_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)

    # rotation tuning
    cooldown_seconds: float = Field(default=60.0)
    daily_sleep: bool = Field(default=True)
    daily_reset_hour_utc: int = Field(default=0)
    daily_reset_tz: str = Field(default="UTC")
    on_rotate: Optional[Callable[[str, str, BaseException], Any]] = None
    initial_backoff: float = Field(default=1.0)
    max_backoff: float = Field(default=60.0)

    # behavior
    verbose: Optional[bool] = Field(default=None)
    suppress_warnings: bool = Field(default=True)

    # ── private runtime state ────────────────────────────────────────
    _manager: Any = PrivateAttr(default=None)
    _last_successful_model: Optional[str] = PrivateAttr(default=None)
    _last_successful_key: Optional[str] = PrivateAttr(default=None)

    # ── init ─────────────────────────────────────────────────────────
    def __init__(self, **data: Any) -> None:
        # v0.2.0 FIX: singular aliases. These were documented in the
        # README since 0.1.5 but silently landed in model_extra (and
        # leaked into the inner client kwargs). Pop them before pydantic
        # sees them so they behave as real aliases.
        data = dict(data)
        singular_key = data.pop("api_key", None)
        singular_model = data.pop("model", None)

        if singular_key is not None:
            if isinstance(singular_key, (list, tuple)):
                data.setdefault("api_keys", [k for k in singular_key if k])
            else:
                data.setdefault("api_keys", [singular_key])

        if singular_model is not None and "models" not in data:
            if isinstance(singular_model, (list, tuple)):
                data["models"] = [m for m in singular_model if m]
            elif isinstance(singular_model, RotatingModelName):
                # v0.2.0 flagship: rotation travels with the name.
                # A RotatingModelName carries its whole failover pool;
                # one line gives you full rotation.
                data["models"] = singular_model.rotation_pool
            else:
                data["models"] = [singular_model]

        super().__init__(**data)

        keys = self.api_keys or [os.getenv("GEMINI_API_KEY", "")]
        keys = [k for k in keys if k]
        if not keys:
            raise ConfigurationError(
                "No API key found. Looked in api_key=, api_keys=, and the "
                "GEMINI_API_KEY env var (not set). Set one of them."
            )

        active_models = self.models or list(CHAT_MODELS)

        self._manager = RateLimitManager(
            api_keys=keys,
            models=active_models,
            cooldown_seconds=self.cooldown_seconds,
            daily_sleep=self.daily_sleep,
            daily_reset_hour_utc=self.daily_reset_hour_utc,
            daily_reset_tz=self.daily_reset_tz,
            on_rotate=self.on_rotate,
        )
        self._initial_backoff = self.initial_backoff
        self._max_backoff = self.max_backoff
        if self.verbose is None:
            self._verbose_rotation = os.getenv(
                "GMS_VERBOSE", ""
            ).strip().lower() in ("1", "true", "yes", "on")
        else:
            self._verbose_rotation = self.verbose
        self._suppress_warnings = self.suppress_warnings

        # Bind the manager back onto the rotating name so the value
        # itself knows its rotation context (introspection / reuse).
        if isinstance(singular_model, RotatingModelName):
            singular_model._manager = self._manager

        if self.suppress_warnings:
            _silence_sdk_warnings()

    # ── LangChain metadata ───────────────────────────────────────────
    @property
    def _llm_type(self) -> str:
        return "googlemodel-samrat-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "models": self.models or list(CHAT_MODELS),
            "temperature": self.temperature,
        }

    # ── v0.2.0: notebook-friendly repr ───────────────────────────────
    def __repr__(self) -> str:
        n_keys = len(self.api_keys) if self.api_keys else 0
        n_models = len(self.models) if self.models else len(CHAT_MODELS)
        return (
            f"ChatGoogleGenerativeAI(keys={n_keys}, models={n_models}, "
            f"last_model={self._last_successful_model!r})"
        )

    # ── attribution metadata ─────────────────────────────────────────
    @property
    def last_successful_model(self) -> Optional[str]:
        return self._last_successful_model

    @property
    def last_successful_key_index(self) -> Optional[int]:
        if self._last_successful_key is None or self._manager is None:
            return None
        try:
            return self._manager.api_keys.index(self._last_successful_key)
        except ValueError:
            return None

    def get_rotation_stats(self) -> Dict[str, Any]:
        return self._manager.stats if self._manager else {}

    # ── internal: build inner LangChain client ───────────────────────
    def _build_lc_client(self, model: str, api_key: str) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI as _LC

        kwargs: Dict[str, Any] = {
            "model": model,
            "google_api_key": api_key,
            "temperature": self.temperature,
        }
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k

        # merge pydantic extras (safety_settings, stop_sequences, ...),
        # but never leak the singular aliases into the inner client
        extras = dict(getattr(self, "model_extra", None) or {})
        extras.pop("api_key", None)
        extras.pop("model", None)
        kwargs.update(extras)

        if self.suppress_warnings:
            with quiet_sdk_warnings():
                return _LC(**kwargs)
        return _LC(**kwargs)

    # ── v0.2.0: inner-call adapters (single-trace telemetry) ─────────
    #
    # The inner client's PUBLIC .invoke()/.stream() start their own
    # traced run, so one llm.invoke() showed up as TWO root traces in
    # LangSmith (double-counted tokens, costs, latency). Calling the
    # protected _generate/_stream directly bypasses the inner Runnable
    # lifecycle — our BaseChatModel run is the one and only trace, and
    # passing run_manager through keeps token accounting on that run.
    # Non-LangChain inner clients (the test-suite fakes) fall back to
    # the public API so every existing test keeps working unchanged.

    def _invoke_inner(
        self,
        client: Any,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> AIMessage:
        if hasattr(client, "_generate"):
            result = client._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return result.generations[0].message
        return client.invoke(messages, stop=stop, **kwargs)

    def _stream_inner(
        self,
        client: Any,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        if hasattr(client, "_stream"):
            for gen in client._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield gen.message
        else:
            for c in client.stream(messages, stop=stop, **kwargs):
                yield c

    async def _ainvoke_inner(
        self,
        client: Any,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> AIMessage:
        if hasattr(client, "_agenerate"):
            result = await client._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return result.generations[0].message
        return await client.ainvoke(messages, stop=stop, **kwargs)

    def _astream_inner(
        self,
        client: Any,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        if hasattr(client, "_astream"):
            async def _gen() -> AsyncIterator[AIMessageChunk]:
                async for gen in client._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield gen.message
            return _gen()
        return client.astream(messages, stop=stop, **kwargs)

    # ── LangChain hooks ──────────────────────────────────────────────
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        def _call(*, api_key: str, model: str) -> AIMessage:
            client = self._build_lc_client(model, api_key)
            return self._invoke_inner(client, messages, stop, run_manager, **kwargs)

        response = self._execute_with_rotation(_call)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        def _stream_call(*, api_key: str, model: str) -> Iterator[AIMessageChunk]:
            client = self._build_lc_client(model, api_key)
            return self._stream_inner(client, messages, stop, run_manager, **kwargs)

        for chunk in self._execute_stream_with_rotation(_stream_call):
            yield ChatGenerationChunk(message=chunk)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        async def _acall(*, api_key: str, model: str) -> AIMessage:
            client = self._build_lc_client(model, api_key)
            return await self._ainvoke_inner(client, messages, stop, run_manager, **kwargs)

        response = await self._aexecute_with_rotation(_acall)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        def _astream_call(*, api_key: str, model: str) -> AsyncIterator[AIMessageChunk]:
            client = self._build_lc_client(model, api_key)
            return self._astream_inner(client, messages, stop, run_manager, **kwargs)

        async for chunk in self._aexecute_stream_with_rotation(_astream_call):
            yield ChatGenerationChunk(message=chunk)

    # ── convenience (kept for backwards compat) ──────────────────────
    def generate_messages(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """Legacy alias for `.invoke(messages)`. Prefer `.invoke()`."""
        response = self.invoke(messages, **kwargs)
        return response.content if hasattr(response, "content") else str(response)