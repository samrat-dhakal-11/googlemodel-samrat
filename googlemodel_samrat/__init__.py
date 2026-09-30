"""
googlemodel-samrat
==================
Intelligent Gemini API key & model rotation for resilient usage.

v0.2.0 highlights:
    - RotatingModelName: chatmodel()/embeddingmodel() return a str
      subclass carrying its failover pool — pass it to OUR clients
      as model= and the whole pool rotates.
    - googlemodel_samrat.langchain: one-line import shim
        from googlemodel_samrat.langchain import ChatGoogleGenerativeAI
    - ONE LangSmith trace per call (no duplicate root runs)
    - Google retryDelay hints honored on 429s
    - AFC / fixed-sampling warnings silenced at call sites
"""
import logging as _logging

# Ship a sane default handler so rotation messages are visible
# out of the box. Users can override with logging.basicConfig(...).
if not _logging.getLogger("googlemodel_samrat").handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("%(message)s"))
    _logger = _logging.getLogger("googlemodel_samrat")
    _logger.addHandler(_h)
    _logger.setLevel(_logging.INFO)

from .chat import ChatGoogleGenerativeAI
from .embeddings import GoogleGenerativeAIEmbeddings
from .exceptions import (
    AllResourcesExhaustedError,
    ConfigurationError,
    GeminiRotatorError,
)
from .core import RateLimitManager, RotationExecutionMixin
from .registry import (
    RotatingModelName,
    # helpers
    print_all_models,
    get_models,
    get_model_count,
    model_exists,
    get_model_category,
    # convenience getters
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
    # registries
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
    ALL_CURRENT_MODELS,
)

__version__ = "0.2.0"
__author__ = "Samrat Dhakal"
__license__ = "MIT"

__all__ = [
    # clients
    "ChatGoogleGenerativeAI",
    "GoogleGenerativeAIEmbeddings",
    # engine
    "RateLimitManager",
    "RotationExecutionMixin",
    # v0.2.0 flagship
    "RotatingModelName",
    # errors
    "GeminiRotatorError",
    "AllResourcesExhaustedError",
    "ConfigurationError",
    # registry functions
    "print_all_models",
    "get_models",
    "get_model_count",
    "model_exists",
    "get_model_category",
    # getters
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
    # model lists
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