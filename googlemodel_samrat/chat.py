"""
LangChain-native chat client with automatic key & model rotation.

Fully LCEL-compatible: inherits from BaseChatModel, supports
`|` pipelines, `.invoke()`, `.stream()`, `.ainvoke()`, `.astream()`.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

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
)
from .exceptions import AllResourcesExhaustedError, ConfigurationError
from .registry import CHAT_MODELS


class ChatGoogleGenerativeAI(BaseChatModel, RotationExecutionMixin):
    """
    Drop-in, LCEL-compatible replacement for `langchain_google_genai.ChatGoogleGenerativeAI`
    with automatic API key rotation and model fallback.

    Example:
        >>> from googlemodel_samrat import ChatGoogleGenerativeAI, chatmodel
        >>> llm = ChatGoogleGenerativeAI(
        ...     api_keys=["key1", "key2"],
        ...     models=[chatmodel(), "gemini-2.5-flash"],
        ...     temperature=0.7,
        ... )
        >>> llm.invoke("Hello!")
    """

    # pydantic v2 config — allow extra kwargs to pass through to inner client
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
    initial_backoff: float = Field(default=1.0)
    max_backoff: float = Field(default=60.0)

    # behavior
    verbose: bool = Field(default=False)
    suppress_warnings: bool = Field(default=True)

    # ── private runtime state ────────────────────────────────────────
    _manager: Any = PrivateAttr(default=None)
    _last_successful_model: Optional[str] = PrivateAttr(default=None)
    _last_successful_key: Optional[str] = PrivateAttr(default=None)

    # ── init ─────────────────────────────────────────────────────────
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        keys = self.api_keys or [os.getenv("GEMINI_API_KEY", "")]
        keys = [k for k in keys if k]
        if not keys:
            raise ConfigurationError(
                "No API keys provided. Pass api_keys=[...] or set GEMINI_API_KEY."
            )

        active_models = self.models or list(CHAT_MODELS)

        self._manager = RateLimitManager(
            api_keys=keys,
            models=active_models,
            cooldown_seconds=self.cooldown_seconds,
            daily_sleep=self.daily_sleep,
        )
        # expose to mixin
        self._initial_backoff = self.initial_backoff
        self._max_backoff = self.max_backoff
        self._verbose_rotation = self.verbose

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

        # merge any pydantic extras (safety_settings, stop_sequences, etc.)
        extras = getattr(self, "model_extra", None) or {}
        kwargs.update(extras)

        # Silence known-harmless langchain_google_genai UserWarnings
        # (fixed-sampling notices, AFC reminders, etc.) that fire when
        # certain models are constructed.
        if self.suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*")
                warnings.filterwarnings("ignore", message=r".*sampling parameter.*will be ignored.*")
                warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain_google_genai.*")
                return _LC(**kwargs)
        return _LC(**kwargs)

    # ── non-streaming (sync) ─────────────────────────────────────────
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        def _call(*, api_key: str, model: str) -> AIMessage:
            client = self._build_lc_client(model, api_key)
            return client.invoke(messages, stop=stop, **kwargs)  # type: ignore[return-value]

        response = self._execute_with_rotation(_call)
        return ChatResult(generations=[ChatGeneration(message=response)])

    # ── streaming (sync) ─────────────────────────────────────────────
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        def _stream_call(*, api_key: str, model: str) -> Iterator[AIMessageChunk]:
            client = self._build_lc_client(model, api_key)
            return client.stream(messages, stop=stop, **kwargs)

        for chunk in self._execute_stream_with_rotation(_stream_call):
            yield ChatGenerationChunk(message=chunk)

    # ── async non-streaming ──────────────────────────────────────────
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        async def _acall(*, api_key: str, model: str) -> AIMessage:
            client = self._build_lc_client(model, api_key)
            return await client.ainvoke(messages, stop=stop, **kwargs)

        response = await self._aexecute_with_rotation(_acall)
        return ChatResult(generations=[ChatGeneration(message=response)])

    # ── async streaming ──────────────────────────────────────────────
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        def _astream_call(*, api_key: str, model: str) -> AsyncIterator[AIMessageChunk]:
            client = self._build_lc_client(model, api_key)
            return client.astream(messages, stop=stop, **kwargs)

        async for chunk in self._aexecute_stream_with_rotation(_astream_call):
            yield ChatGenerationChunk(message=chunk)

    # ── convenience (kept for backwards compat) ──────────────────────
    def generate_messages(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """Legacy alias for `.invoke(messages)`. Prefer `.invoke()`."""
        response = self.invoke(messages, **kwargs)
        return response.content if hasattr(response, "content") else str(response)