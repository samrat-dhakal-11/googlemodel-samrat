"""
Custom exceptions for googlemodel-samrat.

Friendly, actionable errors with troubleshooting tips.
"""
from __future__ import annotations

from typing import List, Optional


class GeminiRotatorError(Exception):
    """Base exception for every error raised by this library."""


class ConfigurationError(GeminiRotatorError):
    """Raised when the library is misconfigured."""


class AllResourcesExhaustedError(GeminiRotatorError):
    """
    Raised when every (api_key, model) combination failed.

    Provides a friendly explanation and troubleshooting tips.
    """

    def __init__(
        self,
        message: str,
        last_error: Optional[BaseException] = None,
        attempted_combinations: int = 0,
        failed_keys: Optional[List[str]] = None,
        failed_models: Optional[List[str]] = None,
        next_available_in: Optional[float] = None,
    ):
        super().__init__(message)
        self.last_error = last_error
        self.attempted_combinations = attempted_combinations
        self.failed_keys = failed_keys or []
        self.failed_models = failed_models or []
        self.next_available_in = next_available_in

    def __str__(self) -> str:
        lines = [super().__str__()]
        if self.attempted_combinations:
            lines.append(f"Attempted {self.attempted_combinations} (key, model) combinations.")
        if self.failed_keys:
            lines.append(f"Failed keys: {len(self.failed_keys)}")
        if self.failed_models:
            lines.append(f"Failed models: {len(self.failed_models)}")
        if self.last_error is not None:
            lines.append(f"Last error: {type(self.last_error).__name__}: {self.last_error}")
        if self.next_available_in:
            lines.append(f"Earliest resource may recover in ~{self.next_available_in:.0f}s.")

        lines.append("")
        lines.append("Troubleshooting tips:")
        lines.append("  1. Verify your API keys are valid and still have quota.")
        lines.append("  2. Add more keys via api_keys=[...] to raise total quota.")
        lines.append("  3. Increase cooldown_seconds, or wait for cooldowns to expire.")
        lines.append("  4. Check Google AI status: https://status.cloud.google.com/")
        return "\n".join(lines)