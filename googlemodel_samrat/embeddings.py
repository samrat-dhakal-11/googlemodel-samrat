"""
Rotation-capable Gemini embeddings wrapper.

Mirrors `langchain_google_genai.GoogleGenerativeAIEmbeddings` with
multi-key rotation and model fallback.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from .core import RateLimitManager, RotationExecutionMixin, _silence_sdk_warnings
from .exceptions import ConfigurationError
from .registry import EMBEDDING_MODELS


class GoogleGenerativeAIEmbeddings(Embeddings, RotationExecutionMixin):
    """
    Gemini embeddings with automatic key & model rotation.

    Example:
        >>> from googlemodel_samrat import GoogleGenerativeAIEmbeddings
        >>> emb = GoogleGenerativeAIEmbeddings(
        ...     api_keys=["key1", "key2"],
        ... )
        >>> emb.embed_query("Hello, world")
    """

    def __init__(
        self,
        *,
        api_keys: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
        model: Optional[str] = None,
        cooldown_seconds: float = 60.0,
        daily_sleep: bool = True,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        verbose: bool = False,
        suppress_warnings: bool = True,
        task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        keys = api_keys or ([api_key] if api_key else None) or [os.getenv("GEMINI_API_KEY", "")]
        keys = [k for k in keys if k]
        if not keys:
            raise ConfigurationError(
                "No API keys provided. Pass api_keys=[...] or set GEMINI_API_KEY."
            )

        active_models = models or ([model] if model else None) or list(EMBEDDING_MODELS)

        self._manager = RateLimitManager(
            api_keys=keys,
            models=active_models,
            cooldown_seconds=cooldown_seconds,
            daily_sleep=daily_sleep,
        )
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._verbose_rotation = verbose
        self.task_type = task_type
        self._extra = kwargs

        self._last_successful_model: Optional[str] = None
        self._last_successful_key: Optional[str] = None

        if suppress_warnings:
            _silence_sdk_warnings()

    # ── attribution ──────────────────────────────────────────────────
    @property
    def last_successful_model(self) -> Optional[str]:
        return self._last_successful_model

    @property
    def last_successful_key_index(self) -> Optional[int]:
        if self._last_successful_key is None:
            return None
        try:
            return self._manager.api_keys.index(self._last_successful_key)
        except ValueError:
            return None

    def get_rotation_stats(self) -> Dict[str, Any]:
        return self._manager.stats

    # ── inner client builder ─────────────────────────────────────────
    def _build_lc_client(self, model: str, api_key: str) -> Any:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings as _LC

        kwargs: Dict[str, Any] = {
            "model": model,
            "google_api_key": api_key,
        }
        if self.task_type is not None:
            kwargs["task_type"] = self.task_type
        kwargs.update(self._extra)
        return _LC(**kwargs)

    # ── Embeddings interface ─────────────────────────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        def _call(*, api_key: str, model: str) -> List[List[float]]:
            client = self._build_lc_client(model, api_key)
            return client.embed_documents(texts)

        return self._execute_with_rotation(_call)

    def embed_query(self, text: str) -> List[float]:
        def _call(*, api_key: str, model: str) -> List[float]:
            client = self._build_lc_client(model, api_key)
            return client.embed_query(text)

        return self._execute_with_rotation(_call)