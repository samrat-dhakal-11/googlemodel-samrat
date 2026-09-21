"""
Core Rotation Engine for Gemini Rotator
========================================

This module contains the heart of the library: the RateLimitManager
and BaseGeminiClient classes that implement intelligent, automatic
failover across multiple API keys and model identifiers.

Key Features:
    - Round-robin rotation with failure tracking
    - Intelligent error classification (quota vs model vs server)
    - Exponential backoff on transient failures
    - Support for 429, 500, 502, 503, timeouts, and more
    - Thread-safe operation for concurrent usage

Architecture:
    RateLimitManager  →  Manages key/model pools and failure state
    BaseGeminiClient  →  Wraps API calls with automatic retry logic
"""

import os
import time
import logging
from typing import List, Optional, Callable, Any, Dict, Set

# Google API Core exceptions cover most standard HTTP/gRPC errors
from google.api_core.exceptions import (
    ResourceExhausted,      # 429 Too Many Requests / Quota Exceeded
    NotFound,               # 404 Model Not Found / Deprecated
    ServiceUnavailable,     # 503 Service Unavailable
    InternalServerError,    # 500 Internal Server Error
    BadGateway,             # 502 Bad Gateway
    GatewayTimeout,         # 504 Gateway Timeout
    RetryError,             # Exhausted retries in underlying SDK
)

from .exceptions import AllResourcesExhaustedError, ConfigurationError

# Configure module logger
logger = logging.getLogger(__name__)


class RateLimitManager:
    """
    Manages a pool of API Keys and Model IDs with intelligent rotation.
    
    Implements a round-robin strategy with per-resource failure tracking.
    When a request fails, the manager classifies the error and marks
    the appropriate resource(s) as temporarily unavailable, preventing
    wasteful retries against known-bad resources.
    
    Error Classification Logic:
        ┌─────────────────────┬──────────────────┬─────────────────────┐
        │ Error Type          │ Marked Failed    │ Reason              │
        ├─────────────────────┼──────────────────┼─────────────────────┤
        │ 429 / Quota         │ API Key only     │ Key exhausted,      │
        │                     │                  │ model still valid   │
        ├─────────────────────┼───────────────────────────────────────┤
        │ 404 / Not Found     │ Model only       │ Model deprecated,   │
        │                     │                  │ key still valid     │
        ├─────────────────────┼──────────────────┼─────────────────────
        │ 500/502/503/Timeout │ Both Key & Model │ Server-side issue,  │
        │                     │                  │ path may be broken  │
        └─────────────────────┴──────────────────┴─────────────────────┘
    
    Attributes:
        api_keys (List[str]): Pool of API keys to rotate through.
        models (List[str]): Pool of model IDs to rotate through.
        key_index (int): Current position in the API key pool.
        model_index (int): Current position in the model pool.
        failed_keys (Set[str]): Set of API keys marked as failed.
        failed_models (Set[str]): Set of models marked as failed.
        
    Example:
        >>> manager = RateLimitManager(
        ...     api_keys=["key1", "key2"],
        ...     models=["gemini-3.8-flash", "gemini-2.5-pro"]
        ... )
        >>> config = manager.get_next_config()
        >>> print(config)
        {'api_key': 'key1', 'model': 'gemini-3.8-flash'}
    """
    
    def __init__(self, api_keys: List[str], models: List[str]):
        """
        Initialize the RateLimitManager with API keys and models.
        
        Args:
            api_keys: List of valid Gemini API key strings.
                      Must contain at least one non-empty key.
            models: List of valid Gemini model ID strings.
                    Must contain at least one model.
                    
        Raises:
            ConfigurationError: If api_keys or models is empty/invalid.
        """
        if not api_keys:
            raise ConfigurationError(
                "At least one API key is required. "
                "Pass api_keys list or set GEMINI_API_KEY env var."
            )
        if not models:
            raise ConfigurationError(
                "At least one model ID is required. "
                "Pass models list or use default CHAT_MODELS."
            )
            
        # Validate no empty strings
        valid_keys = [k for k in api_keys if k and k.strip()]
        if not valid_keys:
            raise ConfigurationError(
                "All provided API keys are empty strings."
            )
            
        self.api_keys = valid_keys
        self.models = models
        
        # Rotation state
        self.key_index: int = 0
        self.model_index: int = 0
        
        # Failure tracking sets
        self.failed_keys: Set[str] = set()
        self.failed_models: Set[str] = set()
        
        logger.info(
            f"RateLimitManager initialized: "
            f"{len(self.api_keys)} keys, {len(self.models)} models"
        )

    def get_next_config(self) -> Dict[str, str]:
        """
        Get the next available (api_key, model) configuration pair.
        
        Skips any keys or models that have been marked as failed.
        Uses round-robin selection across the Cartesian product of
        keys × models.
        
        Returns:
            dict: Dictionary with 'api_key' and 'model' keys.
            
        Raises:
            AllResourcesExhaustedError: If all combinations have been
                                        tried and marked as failed.
        """
        max_attempts = len(self.api_keys) * len(self.models)
        
        if max_attempts == 0:
            raise AllResourcesExhaustedError("No resources configured.")
        
        attempts = 0
        
        while attempts < max_attempts:
            current_key = self.api_keys[self.key_index]
            current_model = self.models[self.model_index]
            
            # Check if this combination is currently blocked
            key_failed = current_key in self.failed_keys
            model_failed = current_model in self.failed_models
            
            if not key_failed and not model_failed:
                config = {
                    "api_key": current_key,
                    "model": current_model
                }
                
                # Advance indices for NEXT call (not this one)
                self._advance_indices()
                
                logger.debug(
                    f"Selected config: model={current_model}, "
                    f"key=...{current_key[-4:]}"
                )
                return config
            
            # Skip this combination and advance
            self._advance_indices()
            attempts += 1
            
        raise AllResourcesExhaustedError(
            f"All {max_attempts} (key, model) combinations have been "
            f"exhausted or marked as failed. "
            f"Failed keys: {len(self.failed_keys)}, "
            f"Failed models: {len(self.failed_models)}"
        )

    def _advance_indices(self) -> None:
        """
        Advance the round-robin indices to the next combination.
        
        Iterates through models first, then wraps to next key.
        This ensures we try all models with each key before moving on.
        """
        self.model_index = (self.model_index + 1) % len(self.models)
        
        # Wrap model index means we've tried all models for current key
        if self.model_index == 0:
            self.key_index = (self.key_index + 1) % len(self.api_keys)

    def mark_failed(self, api_key: str, model: str, error: Exception) -> None:
        """
        Intelligently mark resources as failed based on error analysis.
        
        Analyzes the exception type and message to determine whether
        to mark the API key, the model, or both as temporarily failed.
        
        Args:
            api_key: The API key that was used in the failed request.
            model: The model ID that was used in the failed request.
            error: The exception that was raised during the request.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        error_code = getattr(error, 'code', None)
        
        # ── CATEGORY 1: QUOTA / RATE LIMIT (429) ───
        # The KEY is exhausted but the MODEL is fine.
        # Only mark the key; other keys can still use this model.
        quota_indicators = [
            "quota", "429", "resource_exhausted", "rate_limit",
            "rate limit", "too many requests", "invalid_argument", "400", "invalid api key"
        ]
        if any(term in error_str for term in quota_indicators):
            self.failed_keys.add(api_key)
            logger.warning(
                f"️  QUOTA HIT | Key ending in ...{api_key[-4:]} "
                f"marked as cooling down. "
                f"Model '{model}' remains available for other keys."
            )
            return
        
        # ─── CATEGORY 2: MODEL NOT FOUND / DEPRECATED (404) ───
        # The MODEL is invalid but the KEY is fine.
        # Only mark the model; other models can still use this key.
        model_indicators = [
            "not_found", "deprecated", "404", "invalid model",
            "model not found", "unsupported model"
        ]
        if any(term in error_str for term in model_indicators) or error_code == 404:
            self.failed_models.add(model)
            logger.warning(
                f"⚠️  MODEL UNAVAILABLE | '{model}' marked as failed. "
                f"Key ...{api_key[-4:]} remains available for other models."
            )
            return
        
        # ─── CATEGORY 3: TRANSIENT SERVER ERRORS (5xx) ───
        # Could be key-specific endpoint issue OR model-specific server issue.
        # Mark BOTH to find a healthy path fastest.
        server_indicators = [
            "502", "500", "503", "504", 
            "bad gateway", "internal server", "service unavailable",
            "gateway timeout", "timeout", "connection reset",
            "server error"
        ]
        server_exception_types = (
            ServiceUnavailable, InternalServerError, 
            BadGateway, GatewayTimeout, RetryError
        )
        
        if (any(term in error_str for term in server_indicators) or 
            isinstance(error, server_exception_types)):
            
            self.failed_keys.add(api_key)
            self.failed_models.add(model)
            logger.warning(
                f"⚠️  SERVER ERROR ({error_code or error_type}) | "
                f"Both Key ...{api_key[-4:]} AND Model '{model}' "
                f"marked as failed. Rotating to find healthy path."
            )
            return
        
        # ─── DEFAULT: UNKNOWN ERROR ───
        # Be conservative: mark both to avoid repeated failures
        logger.warning(
            f"⚠️  UNKNOWN ERROR ({error_type}) | "
            f"Marking both Key ...{api_key[-4:]} and Model '{model}' "
            f"as failed as precaution."
        )
        self.failed_keys.add(api_key)
        self.failed_models.add(model)

    def reset_failures(self) -> None:
        """
        Reset all failure tracking.
        
        Call this after a significant wait period (e.g., 1 hour)
        to allow previously failed keys/models to recover.
        Useful for long-running batch processes.
        
        Example:
            >>> manager.reset_failures()
            >>> # Now all keys and models will be retried
        """
        prev_key_count = len(self.failed_keys)
        prev_model_count = len(self.failed_models)
        
        self.failed_keys.clear()
        self.failed_models.clear()
        
        logger.info(
            f"🔄 Failure tracking reset. "
            f"Cleared {prev_key_count} failed keys and "
            f"{prev_model_count} failed models."
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get current rotation statistics.
        
        Returns:
            dict: Dictionary with rotation state information.
        """
        return {
            "total_keys": len(self.api_keys),
            "total_models": len(self.models),
            "failed_keys": len(self.failed_keys),
            "failed_models": len(self.failed_models),
            "available_combinations": (
                len(self.api_keys) - len(self.failed_keys)
            ) * (
                len(self.models) - len(self.failed_models)
            ),
            "current_key_index": self.key_index,
            "current_model_index": self.model_index,
        }


class BaseGeminiClient:
    """
    Base class for all Gemini Rotator clients.
    
    Provides the core `_execute_with_fallback` method that wraps
    any API call with automatic retry and rotation logic. Subclasses
    implement modality-specific request construction.
    
    Attributes:
        api_keys (List[str]): Configured API keys.
        models (List[str]): Configured model IDs.
        manager (RateLimitManager): The rotation manager instance.
        
    Example:
        >>> client = BaseGeminiClient(
        ...     api_keys=["key1", "key2"],
        ...     models=["gemini-3.8-flash"]
        ... )
        >>> result = client._execute_with_fallback(
        ...     my_api_function, arg1, arg2
        ... )
    """
    
    def __init__(
        self, 
        api_keys: Optional[List[str]] = None, 
        models: Optional[List[str]] = None
    ):
        """
        Initialize the base client with API keys and models.
        
        Args:
            api_keys: List of API keys. Falls back to GEMINI_API_KEY
                      environment variable if not provided.
            models: List of model IDs. Should be set by subclasses
                    based on their modality (chat, image, etc.).
                    
        Raises:
            ConfigurationError: If no valid API keys are available.
        """
        # Load from env if not explicitly provided
        self.api_keys = api_keys or [os.getenv("GEMINI_API_KEY")]
        self.models = models or []
        
        # Validate
        if not self.api_keys or not self.api_keys[0]:
            raise ConfigurationError(
                "No API keys provided. Either pass api_keys parameter "
                "or set the GEMINI_API_KEY environment variable."
            )
        
        if not self.models:
            raise ConfigurationError(
                "No models provided. Subclasses must specify models "
                "appropriate for their modality."
            )
        
        # Initialize rotation manager
        self.manager = RateLimitManager(
            api_keys=self.api_keys, 
            models=self.models
        )
        
        logger.info(
            f"BaseGeminiClient initialized with "
            f"{len(self.api_keys)} key(s) and {len(self.models)} model(s)"
        )

    def _execute_with_fallback(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute a function with automatic retry and resource rotation.
        
        This is the core resilience mechanism. It attempts the function
        call with different (key, model) combinations until one succeeds
        or all combinations are exhausted.
        
        Retry Behavior:
            - Catches 429, 404, 500, 502, 503, 504, timeouts
            - Classifies errors to mark appropriate resources as failed
            - Applies exponential backoff: 2^attempt seconds
            - Maximum attempts = len(keys) × len(models)
        
        Args:
            func: Callable to execute. Must accept google_api_key
                  and model as keyword arguments.
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func. Will be
                      augmented with google_api_key and model.
                      
        Returns:
            Any: The return value of func on success.
            
        Raises:
            AllResourcesExhaustedError: If all combinations fail.
        """
        last_error: Optional[Exception] = None
        max_attempts = len(self.api_keys) * len(self.models)
        
        logger.info(
            f"🚀 Starting fallback execution: "
            f"max {max_attempts} attempts"
        )
        
        for attempt in range(max_attempts):
            try:
                # Get next available configuration
                config = self.manager.get_next_config()
                
                # Inject rotation config into kwargs
                kwargs['google_api_key'] = config['api_key']
                kwargs['model'] = config['model']
                
                # Log the attempt
                logger.info(
                    f"🔄 Attempt {attempt + 1}/{max_attempts} | "
                    f"Model: {config['model']} | "
                    f"Key: ...{config['api_key'][-4:]}"
                )
                
                # Execute the actual API call
                result = func(*args, **kwargs)
                
                # Success! Log and return
                logger.info(
                    f"✅ SUCCESS on attempt {attempt + 1} | "
                    f"Model: {config['model']} | "
                    f"Key: ...{config['api_key'][-4:]}"
                )
                return result
                
            except Exception as e:
                last_error = e
                current_key = kwargs.get('google_api_key', 'unknown')
                current_model = kwargs.get('model', 'unknown')
                
                # Classify and mark failures
                self.manager.mark_failed(current_key, current_model, e)
                if isinstance(e, AllResourcesExhaustedError):
                    raise
                
                # Calculate exponential backoff
                # attempt 0 → 1s, attempt 1 → 2s, attempt 2 → 4s, etc.
                wait_time = 2 ** attempt
                max_wait = 60  # Cap at 60 seconds
                
                actual_wait = min(wait_time, max_wait)
                
                logger.warning(
                    f" Attempt {attempt + 1} FAILED | "
                    f"Model: {current_model} | "
                    f"Error: {type(e).__name__}: {str(e)[:100]} | "
                    f"Waiting {actual_wait}s before retry..."
                )
                
                # Sleep with backoff
                time.sleep(actual_wait)
        
        # All attempts exhausted
        error_msg = (
            f"All {max_attempts} (key, model) combinations failed. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        logger.error(f" {error_msg}")
        
        raise AllResourcesExhaustedError(
            message=error_msg,
            last_error=last_error,
            attempted_combinations=max_attempts
        )
