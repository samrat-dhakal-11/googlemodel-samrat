"""
Core rotation engine for googlemodel-samrat.

Thread-safe, stateful, timestamped key & model rotation with:
  - Per-resource cooldown tracking
  - "Midnight sleep" for hard daily quotas
  - Jittered exponential backoff
  - Friendly error wrapping
"""
from __future__ import annotations

import logging
import random
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from google.api_core.exceptions import (
    BadGateway,
    GatewayTimeout,
    InternalServerError,
    NotFound,
    PermissionDenied,
    ResourceExhausted,
    RetryError,
    ServiceUnavailable,
    Unauthenticated,
)

from .exceptions import AllResourcesExhaustedError, ConfigurationError

logger = logging.getLogger(__name__)
T = TypeVar("T")


_DAILY_MARKERS = (
    "per day", "per-day", "daily", "requests per day", "rpd",
    "day limit", "quota limit reached for the day", "per_day",
)


def _looks_like_daily_quota(text: str) -> bool:
    return any(m in text for m in _DAILY_MARKERS)


def _next_midnight_ts() -> float:
    """Next 00:00 UTC. ..."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    return tomorrow.timestamp()


def _silence_sdk_warnings() -> None:
    for name in ("google.generativeai", "google_genai", "google.ai.generativelanguage"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r".*AFC.*")
    warnings.filterwarnings("ignore", message=r".*[Aa]utomatic [Ff]unction [Cc]alling.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.generativeai.*")


@dataclass
class _State:
    cooldown_until: float = 0.0
    daily_until: float = 0.0
    failure_count: int = 0
    last_error: str = ""

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until and now >= self.daily_until

    def wait(self, now: float) -> float:
        return max(0.0, self.cooldown_until - now, self.daily_until - now)


class RateLimitManager:
    """Stateful, thread-safe rotation manager."""

    def __init__(
        self,
        api_keys: List[str],
        models: List[str],
        *,
        cooldown_seconds: float = 60.0,
        daily_sleep: bool = True,
    ) -> None:
        clean_keys = [k.strip() for k in (api_keys or []) if k and k.strip()]
        if not clean_keys:
            raise ConfigurationError(
                "No API keys provided. Pass api_keys=[...] or set GEMINI_API_KEY."
            )
        clean_models = [m for m in (models or []) if m]
        if not clean_models:
            raise ConfigurationError(
                "No models provided. Pass models=[...] or rely on the default registry."
            )

        self.api_keys: List[str] = clean_keys
        self.models: List[str] = clean_models
        self.cooldown_seconds = float(cooldown_seconds)
        self.daily_sleep = bool(daily_sleep)

        self._lock = threading.RLock()
        self._key_state: Dict[str, _State] = {k: _State() for k in self.api_keys}
        self._model_state: Dict[str, _State] = {m: _State() for m in self.models}
        # Per (key, model) cooldowns for pair-scoped 429 quota errors.
        self._pair_state: Dict[Tuple[str, str], _State] = {}
        self._key_index = 0
        self._model_index = 0

    @property
    def max_attempts(self) -> int:
        return len(self.api_keys) * len(self.models)

    @property
    def has_key_rotation(self) -> bool:
        return len(self.api_keys) > 1

    @property
    def has_model_rotation(self) -> bool:
        return len(self.models) > 1

    @property
    def failed_keys(self) -> List[str]:
        with self._lock:
            now = time.time()
            return [k for k, s in self._key_state.items() if not s.available(now)]

    @property
    def failed_models(self) -> List[str]:
        with self._lock:
            now = time.time()
            return [m for m, s in self._model_state.items() if not s.available(now)]

    @property
    def failed_pairs(self) -> List[Tuple[str, str]]:
        with self._lock:
            now = time.time()
            return [p for p, s in self._pair_state.items() if not s.available(now)]

    def _advance_locked(self) -> None:
        self._model_index = (self._model_index + 1) % len(self.models)
        if self._model_index == 0:
            self._key_index = (self._key_index + 1) % len(self.api_keys)

    def get_next_config(self) -> Dict[str, str]:
        with self._lock:
            now = time.time()
            for _ in range(self.max_attempts):
                key = self.api_keys[self._key_index]
                model = self.models[self._model_index]
                pair = self._pair_state.get((key, model))
                pair_ok = pair is None or pair.available(now)
                if (
                    self._key_state[key].available(now)
                    and self._model_state[model].available(now)
                    and pair_ok
                ):
                    config = {"api_key": key, "model": model}
                    self._advance_locked()
                    return config
                self._advance_locked()
            raise self._build_exhausted_error()

    def shortest_wait(self) -> float:
        with self._lock:
            now = time.time()
            waits = (
                [self._key_state[k].wait(now) for k in self.api_keys]
                + [self._model_state[m].wait(now) for m in self.models]
                + [s.wait(now) for s in self._pair_state.values()]
            )
            positive = [w for w in waits if w > 0]
            return min(positive) if positive else 0.0

    def mark_failed(self, api_key: str, model: str, error: BaseException) -> None:
        text = str(error).lower()
        code = getattr(error, "code", None)
        now = time.time()

        with self._lock:
            quota_terms = (
                "quota", "429", "resource_exhausted", "rate_limit", "rate limit",
                "too many requests", "invalid api key", "invalid_api_key",
                "permission_denied", "unauthenticated", "invalid_argument",
            )
            if (
                any(t in text for t in quota_terms)
                or isinstance(error, (ResourceExhausted, PermissionDenied, Unauthenticated))
                or code in (400, 401, 403, 429)
            ):
                # Quota is scoped to the (key, model) pair.
                daily = _looks_like_daily_quota(text)
                self._mark_pair(api_key, model, text, now, daily_hint=daily)
                return

            model_terms = (
                "not_found", "deprecated", "404", "invalid model",
                "model not found", "unsupported model", "not supported",
                "no such model",
            )
            if (
                any(t in text for t in model_terms)
                or isinstance(error, NotFound)
                or code == 404
            ):
                self._mark_model(model, text, now)
                return

            server_terms = (
                "502", "500", "503", "504", "bad gateway", "internal server",
                "service unavailable", "gateway timeout", "timeout",
                "connection reset", "server error", "deadline exceeded",
                "unavailable",
            )
            if (
                any(t in text for t in server_terms)
                or isinstance(
                    error,
                    (ServiceUnavailable, InternalServerError, BadGateway,
                     GatewayTimeout, RetryError),
                )
            ):
                self._mark_key(api_key, text, now, daily_hint=False)
                return

            self._mark_key(api_key, text, now, daily_hint=False)

    def _mark_key(self, key: str, text: str, now: float, *, daily_hint: bool) -> None:
        st = self._key_state.get(key)
        if st is None:
            return
        st.failure_count += 1
        st.last_error = text[:200]
        if daily_hint and self.daily_sleep:
            st.daily_until = _next_midnight_ts()
            wake = datetime.fromtimestamp(st.daily_until).strftime("%Y-%m-%d %H:%M")
            print(f"[quota] key ...{key[-4:]} sleeping until {wake} (daily limit)", flush=True)
        else:
            st.cooldown_until = now + self.cooldown_seconds
            print(f"[quota] key ...{key[-4:]} sleeping for {self.cooldown_seconds:.0f}s", flush=True)

    def _mark_model(self, model: str, text: str, now: float) -> None:
        st = self._model_state.get(model)
        if st is None:
            return
        st.failure_count += 1
        st.last_error = text[:200]
        st.cooldown_until = now + self.cooldown_seconds
        print(f"[quota] {model} sleeping for {self.cooldown_seconds:.0f}s", flush=True)

    def _mark_pair(self, key: str, model: str, text: str, now: float,
                   *, daily_hint: bool) -> None:
        """Cool down a single (key, model) pair. Key and model stay
        globally usable for other combinations."""
        st = self._pair_state.setdefault((key, model), _State())
        st.failure_count += 1
        st.last_error = text[:200]
        if daily_hint and self.daily_sleep:
            st.daily_until = _next_midnight_ts()
            wake = datetime.fromtimestamp(st.daily_until, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(
                f"[quota] {model} x key ...{key[-4:]} sleeping until {wake} (daily limit)",
                flush=True,
            )
        else:
            st.cooldown_until = now + self.cooldown_seconds
            print(
                f"[quota] {model} x key ...{key[-4:]} sleeping for {self.cooldown_seconds:.0f}s",
                flush=True,
            )

    def reset_failures(self) -> None:
        with self._lock:
            for st in self._key_state.values():
                st.cooldown_until = st.daily_until = 0.0
                st.failure_count = 0
                st.last_error = ""
            for st in self._model_state.values():
                st.cooldown_until = st.daily_until = 0.0
                st.failure_count = 0
                st.last_error = ""
            self._pair_state.clear()
        logger.info("Rotation failures reset.")

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            active_keys = sum(1 for k in self.api_keys if self._key_state[k].available(now))
            active_models = sum(1 for m in self.models if self._model_state[m].available(now))
            return {
                "total_keys": len(self.api_keys),
                "total_models": len(self.models),
                "active_keys": active_keys,
                "active_models": active_models,
                "cooling_keys": len(self.api_keys) - active_keys,
                "cooling_models": len(self.models) - active_models,
                "cooling_pairs": sum(
                    1 for s in self._pair_state.values() if not s.available(now)
                ),
                "available_combinations": max(
                    0,
                    active_keys * active_models
                    - sum(1 for s in self._pair_state.values() if not s.available(now)),
                ),
                "current_key_index": self._key_index,
                "current_model_index": self._model_index,
                "key_rotation_enabled": self.has_key_rotation,
                "model_rotation_enabled": self.has_model_rotation,
            }

    def _build_exhausted_error(self) -> AllResourcesExhaustedError:
        now = time.time()
        failed_keys = [k for k, s in self._key_state.items() if not s.available(now)]
        failed_models = [m for m, s in self._model_state.items() if not s.available(now)]
        return AllResourcesExhaustedError(
            message="No (api_key, model) combinations are currently available.",
            last_error=None,
            attempted_combinations=self.max_attempts,
            failed_keys=failed_keys,
            failed_models=failed_models,
            next_available_in=self.shortest_wait(),
        )


class RotationExecutionMixin:
    """Mixin providing `_execute_with_rotation` for any client."""

    _manager: RateLimitManager
    _initial_backoff: float = 1.0
    _max_backoff: float = 60.0
    _verbose_rotation: bool = False

    def _record_success(self, config: Dict[str, str]) -> None:
        self._last_successful_model = config["model"]
        self._last_successful_key = config["api_key"]

    def _friendly_exhausted(self, err: AllResourcesExhaustedError) -> AllResourcesExhaustedError:
        return err

    def _backoff_wait(self, attempt: int) -> float:
        base = min(self._initial_backoff * (2 ** attempt), self._max_backoff)
        return base * (0.5 + random.random())

    def _log_success(self, config: Dict[str, str]) -> None:
        if self._verbose_rotation:
            logger.info("Response from model: %s", config["model"])

    def _execute_with_rotation(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        last_error: Optional[BaseException] = None
        attempted = 0

        for attempt in range(self._manager.max_attempts):
            try:
                config = self._manager.get_next_config()
            except AllResourcesExhaustedError as e:
                raise self._friendly_exhausted(e) from e

            attempted += 1
            print(f"Initializing {config['model']} ......", flush=True)
            try:
                result = func(api_key=config["api_key"], model=config["model"], **kwargs)
                self._record_success(config)
                self._log_success(config)
                return result
            except Exception as e:
                last_error = e
                self._manager.mark_failed(config["api_key"], config["model"], e)
                if attempt + 1 < self._manager.max_attempts:
                    time.sleep(self._backoff_wait(attempt))

        raise self._friendly_exhausted(
            AllResourcesExhaustedError(
                message=f"All {self._manager.max_attempts} (key, model) combinations failed.",
                last_error=last_error,
                attempted_combinations=attempted,
                next_available_in=self._manager.shortest_wait(),
            )
        )

    def _execute_stream_with_rotation(self, func: Callable[..., Any], *args: Any, **kwargs: Any):
        last_error: Optional[BaseException] = None
        attempted = 0

        for attempt in range(self._manager.max_attempts):
            try:
                config = self._manager.get_next_config()
            except AllResourcesExhaustedError as e:
                raise self._friendly_exhausted(e) from e

            attempted += 1
            print(f"Initializing {config['model']} ......", flush=True)
            try:
                stream = func(api_key=config["api_key"], model=config["model"], **kwargs)
                first = next(stream)
                self._record_success(config)
                self._log_success(config)
                yield first
                for chunk in stream:
                    yield chunk
                return
            except StopIteration:
                self._record_success(config)
                return
            except Exception as e:
                last_error = e
                self._manager.mark_failed(config["api_key"], config["model"], e)
                if attempt + 1 < self._manager.max_attempts:
                    time.sleep(self._backoff_wait(attempt))

        raise self._friendly_exhausted(
            AllResourcesExhaustedError(
                message=f"All {self._manager.max_attempts} (key, model) combinations failed during streaming.",
                last_error=last_error,
                attempted_combinations=attempted,
                next_available_in=self._manager.shortest_wait(),
            )
        )

    async def _aexecute_with_rotation(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        import asyncio
        last_error: Optional[BaseException] = None
        attempted = 0

        for attempt in range(self._manager.max_attempts):
            try:
                config = self._manager.get_next_config()
            except AllResourcesExhaustedError as e:
                raise self._friendly_exhausted(e) from e

            attempted += 1
            print(f"Initializing {config['model']} ......", flush=True)
            try:
                result = await func(api_key=config["api_key"], model=config["model"], **kwargs)
                self._record_success(config)
                self._log_success(config)
                return result
            except Exception as e:
                last_error = e
                self._manager.mark_failed(config["api_key"], config["model"], e)
                if attempt + 1 < self._manager.max_attempts:
                    await asyncio.sleep(self._backoff_wait(attempt))

        raise self._friendly_exhausted(
            AllResourcesExhaustedError(
                message=f"All {self._manager.max_attempts} (key, model) combinations failed.",
                last_error=last_error,
                attempted_combinations=attempted,
                next_available_in=self._manager.shortest_wait(),
            )
        )

    async def _aexecute_stream_with_rotation(self, func: Callable[..., Any], *args: Any, **kwargs: Any):
        import asyncio
        last_error: Optional[BaseException] = None
        attempted = 0

        for attempt in range(self._manager.max_attempts):
            try:
                config = self._manager.get_next_config()
            except AllResourcesExhaustedError as e:
                raise self._friendly_exhausted(e) from e

            attempted += 1
            print(f"Initializing {config['model']} ......", flush=True)
            try:
                stream = func(api_key=config["api_key"], model=config["model"], **kwargs)
                try:
                    first = await stream.__anext__()
                except StopAsyncIteration:
                    self._record_success(config)
                    return
                self._record_success(config)
                self._log_success(config)
                yield first
                async for chunk in stream:
                    yield chunk
                return
            except Exception as e:
                last_error = e
                self._manager.mark_failed(config["api_key"], config["model"], e)
                if attempt + 1 < self._manager.max_attempts:
                    await asyncio.sleep(self._backoff_wait(attempt))

        raise self._friendly_exhausted(
            AllResourcesExhaustedError(
                message=f"All {self._manager.max_attempts} (key, model) combinations failed during streaming.",
                last_error=last_error,
                attempted_combinations=attempted,
                next_available_in=self._manager.shortest_wait(),
            )
        )
