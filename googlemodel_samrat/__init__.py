"""
Gemini Rotator - Intelligent API Key & Model Rotation Library
=============================================================

An open-source Python library designed to help developers leverage 
the free tier of the Gemini API without hitting disruptive rate limits.

Features:
    - Automatic failover across multiple API keys
    - Smart model rotation on 429, 502, 500, and timeout errors
    - Clean SDK-mimicking interface (ChatGoogleGenerativeAI, etc.)
    - Comprehensive model registry for all active Gemini endpoints

Usage:
    >>> from gemini_rotator import ChatGoogleGenerativeAI, chatmodel
    >>> llm = ChatGoogleGenerativeAI(api_keys=["key1", "key2"], model=chatmodel())
    >>> response = llm.invoke("Hello!")

Author: Gemini Rotator Contributors
Version: 0.1.0
License: MIT
"""

from .chat import ChatGoogleGenerativeAI
from .registry import (
    # Registry Management Functions
    print_all_models, 
    get_models, 
    get_model_count,
    model_exists,
    get_model_category,
    
    # Top-Priority Convenience Getters
    chatmodel,
    audiomodel,
    imagemodel,
    videomodel,
    embeddingmodel,
    musicmodel,
    roboticsmodel,
    computerusemodel,
    researchmodel,
    agentmodel,
    gemmamodel,
    
    # Model Lists (for advanced users)
    CHAT_MODELS,
    TEXT_MODELS,
    AUDIO_MODELS,
    IMAGE_MODELS,
    VIDEO_MODELS,
    EMBEDDING_MODELS,
    MUSIC_MODELS,
    ROBOTICS_MODELS,
    COMPUTER_USE_MODELS,
    RESEARCH_MODELS,
    AGENT_MODELS,
    GEMMA_MODELS,
    ALL_CURRENT_MODELS
)

__version__ = "0.1.0"
__author__ = "Gemini Rotator Contributors"
__license__ = "MIT"

__all__ = [
    # Core Clients
    "ChatGoogleGenerativeAI",
    
    # Registry Functions
    "print_all_models",
    "get_models",
    "get_model_count",
    "model_exists",
    "get_model_category",
    
    # Convenience Getters
    "chatmodel",
    "audiomodel",
    "imagemodel",
    "videomodel",
    "embeddingmodel",
    "musicmodel",
    "roboticsmodel",
    "computerusemodel",
    "researchmodel",
    "agentmodel",
    "gemmamodel",
    
    # Model Lists (for advanced users)
    "CHAT_MODELS",
    "TEXT_MODELS",
    "AUDIO_MODELS",
    "IMAGE_MODELS",
    "VIDEO_MODELS",
    "EMBEDDING_MODELS",
    "MUSIC_MODELS",
    "ROBOTICS_MODELS",
    "COMPUTER_USE_MODELS",
    "RESEARCH_MODELS",
    "AGENT_MODELS",
    "GEMMA_MODELS",
    "ALL_CURRENT_MODELS",
]
