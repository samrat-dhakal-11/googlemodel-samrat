import logging
from typing import List, Optional, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage

from .core import BaseGeminiClient
from .registry import CHAT_MODELS

logger = logging.getLogger(__name__)

"""
Chat Client for Gemini Rotator
===============================

Provides a drop-in replacement for langchain_google_generativeai's
ChatGoogleGenerativeAI with automatic key/model rotation on failures.

This class mimics the standard LangChain interface while adding
resilience through the BaseGeminiClient's fallback mechanism.

Usage:
    >>> from gemini_rotator import ChatGoogleGenerativeAI
    >>> llm = ChatGoogleGenerativeAI(
    ...     api_keys=["key1", "key2"],
    ...     temperature=0.7
    ... )
    >>> response = llm.invoke("Explain quantum physics")
    >>> print(response)
"""


class ChatGoogleGenerativeAI(BaseGeminiClient):
    """
    Gemini Chat LLM with automatic key/model rotation.
    
    A drop-in replacement for langchain_google_generativeai's
    ChatGoogleGenerativeAI that adds intelligent failover when
    rate limits (429), server errors (502/500), or model
    unavailability occurs.
    
    Features:
        - Automatic rotation across multiple API keys
        - Automatic fallback to alternative chat models (latest to oldest)
        - Exponential backoff on transient failures
        - Full LangChain message interface support
        - Temperature and generation parameter control
        
    Attributes:
        temperature (float): Sampling temperature (0.0-1.0).
        kwargs (dict): Additional parameters passed to underlying LLM.
        
    Example:
        Simple string prompt:
            >>> llm = ChatGoogleGenerativeAI(api_keys=["key1"])
            >>> response = llm.invoke("Hello!")
            >>> print(response)
            'Hello! How can I help you today?'
            
        With multiple keys and custom temperature:
            >>> llm = ChatGoogleGenerativeAI(
            ...     api_keys=["key1", "key2", "key3"],
            ...     temperature=0.9,
            ...     top_p=0.95
            ... )
            >>> response = llm.invoke("Write a poem about AI")
    """
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the rotating chat client.
        
        Args:
            api_keys: List of Gemini API keys. Defaults to 
                      GEMINI_API_KEY env var if not provided.
            models: List of chat-compatible model IDs. 
                    Defaults to CHAT_MODELS registry if not provided.
            temperature: Sampling temperature for generation.
                         0.0 = deterministic, 1.0 = creative.
            **kwargs: Additional parameters passed to the underlying
                      langchain_google_generativeai.ChatGoogleGenerativeAI.
                      Common options: top_p, top_k, max_output_tokens,
                      safety_settings, stop_sequences.
                      
        Raises:
            ConfigurationError: If no valid API keys are available.
        """
        # Default to registry's chat models if none specified
        active_models = models or CHAT_MODELS
        
        # Initialize parent with chat-compatible models only
        super().__init__(api_keys=api_keys, models=active_models)
        
        # Store generation parameters
        self.temperature = temperature
        self.kwargs = kwargs
        
        logger.info(
            f"ChatGoogleGenerativeAI initialized | "
            f"temp={temperature} | "
            f"{len(active_models)} models | "
            f"{len(self.api_keys)} keys"
        )

    def invoke(self, prompt: Union[str, List[BaseMessage]], **kwargs) -> str:
        """
        Generate a chat response with automatic fallback.
        
        Accepts either a simple string prompt or a list of LangChain
        message objects for multi-turn conversations.
        
        Args:
            prompt: Either a string prompt or list of BaseMessage objects.
            **kwargs: Override parameters for this specific invocation.
                      Takes precedence over constructor kwargs.
                      
        Returns:
            str: The generated response content as a plain string.
            
        Raises:
            AllResourcesExhaustedError: If all key/model combos fail.
            
        Example:
            String prompt:
                >>> response = llm.invoke("What is Python?")
                
            Message history:
                >>> from langchain_core.messages import HumanMessage, AIMessage
                >>> messages = [
                ...     HumanMessage(content="Hi"),
                ...     AIMessage(content="Hello!"),
                ...     HumanMessage(content="How are you?")
                ... ]
                >>> response = llm.invoke(messages)
        """
        def _make_request(google_api_key: str, model: str, **local_kwargs) -> Any:
            """
            Inner function that creates the actual LangChain client
            and executes the request. Called by _execute_with_fallback.
            """
            # Import here to avoid circular dependencies
            from langchain_google_genai import (
                ChatGoogleGenerativeAI as _LC_Client
            )
            
            # Merge constructor kwargs with invocation kwargs
            merged_kwargs = {**self.kwargs, **local_kwargs}
            
            # Create the underlying LangChain client
            lc_client = _LC_Client(
                model=model,
                google_api_key=google_api_key,
                temperature=self.temperature,
                **merged_kwargs
            )
            
            # Handle both string and message list inputs
            if isinstance(prompt, str):
                messages = [HumanMessage(content=prompt)]
            else:
                messages = prompt
            
            # Execute and return the raw LangChain response
            return lc_client.invoke(messages)

        # Execute with automatic fallback/rotation
        response = self._execute_with_fallback(_make_request, **kwargs)
        
        # Extract content from LangChain message object
        if hasattr(response, 'content'):
            return response.content
        
        # Fallback: convert to string
        return str(response)

    def generate_messages(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """
        Convenience method for multi-turn conversation generation.
        
        Args:
            messages: List of LangChain BaseMessage objects representing
                      the conversation history.
            **kwargs: Override parameters for this invocation.
            
        Returns:
            str: The assistant's response content.
            
        Example:
            >>> from langchain_core.messages import HumanMessage, AIMessage
            >>> history = [
            ...     HumanMessage("What's the capital of France?"),
            ...     AIMessage("Paris"),
            ...     HumanMessage("What about Spain?")
            ... ]
            >>> response = llm.generate_messages(history)
            >>> print(response)
            'The capital of Spain is Madrid.'
        """
        return self.invoke(messages, **kwargs)

    def get_rotation_stats(self) -> dict:
        """
        Get current rotation statistics for monitoring/debugging.
        
        Returns:
            dict: Dictionary containing rotation state information
                  including failed keys, failed models, and available
                  combinations.
                  
        Example:
            >>> stats = llm.get_rotation_stats()
            >>> print(f"Available combos: {stats['available_combinations']}")
        """
        return self.manager.stats
