"""
Custom exceptions for the Gemini Rotator library.

This module defines the exception hierarchy used throughout the library
to provide clear, actionable error messages when rotation fails or
configuration is invalid.
"""


class GeminiRotatorError(Exception):
    """
    Base exception class for all Gemini Rotator errors.
    
    All custom exceptions in this library inherit from this class,
    allowing users to catch any library-specific error with a single
    except block:
    
        try:
            response = llm.invoke(prompt)
        except GeminiRotatorError as e:
            print(f"Library error: {e}")
    """
    pass


class AllResourcesExhaustedError(GeminiRotatorError):
    """
    Raised when all API keys and models have been attempted and failed.
    
    This exception indicates that the rotation system has cycled through
    every available (key, model) combination and none were successful.
    This typically means:
        - All API keys have hit their quota limits
        - All models are temporarily unavailable or deprecated
        - There is a network connectivity issue
        
    Attributes:
        last_error (Exception): The final exception that caused the failure.
        attempted_combinations (int): Number of (key, model) pairs tried.
    """
    
    def __init__(self, message: str, last_error: Exception = None, 
                 attempted_combinations: int = 0):
        super().__init__(message)
        self.last_error = last_error
        self.attempted_combinations = attempted_combinations
        
    def __str__(self):
        base = super().__str__()
        if self.last_error:
            base += f"\nLast error: {type(self.last_error).__name__}: {self.last_error}"
        if self.attempted_combinations > 0:
            base += f"\nAttempted {self.attempted_combinations} combinations."
        return base


class ConfigurationError(GeminiRotatorError):
    """
    Raised when the library is misconfigured.
    
    Common causes:
        - No API keys provided
        - Empty model list
        - Invalid model category name
        - Missing required environment variables
    """
    pass
