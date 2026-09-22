"""
Public API surface tests.

Verifies:
  - Everything in __all__ is importable
  - No unexpected names leak into the public namespace
  - __version__ is a sane string
  - Registry getters work and return non-empty strings
  - Legacy/compat exports remain stable
"""
from __future__ import annotations

import re

import googlemodel_samrat as gms


# ─────────────────────────────────────────────────────────────────────
# __all__ integrity
# ─────────────────────────────────────────────────────────────────────
def test_all_is_defined():
    assert hasattr(gms, "__all__")
    assert isinstance(gms.__all__, list)
    assert len(gms.__all__) > 20


def test_all_names_actually_exist():
    missing = [name for name in gms.__all__ if not hasattr(gms, name)]
    assert missing == [], f"missing from module: {missing}"


def test_no_duplicate_names_in_all():
    assert len(gms.__all__) == len(set(gms.__all__))


# ─────────────────────────────────────────────────────────────────────
# Version
# ─────────────────────────────────────────────────────────────────────
def test_version_is_string():
    assert isinstance(gms.__version__, str)


def test_version_matches_semver():
    assert re.match(r"^\d+\.\d+\.\d+", gms.__version__), gms.__version__


def test_author_metadata():
    assert isinstance(gms.__author__, str)
    assert isinstance(gms.__license__, str)


# ─────────────────────────────────────────────────────────────────────
# Public classes
# ─────────────────────────────────────────────────────────────────────
def test_chat_client_exported():
    assert hasattr(gms, "ChatGoogleGenerativeAI")


def test_embeddings_exported():
    assert hasattr(gms, "GoogleGenerativeAIEmbeddings")


def test_rate_limit_manager_exported():
    assert hasattr(gms, "RateLimitManager")


def test_rotation_mixin_exported():
    assert hasattr(gms, "RotationExecutionMixin")


def test_exceptions_exported():
    assert hasattr(gms, "GeminiRotatorError")
    assert hasattr(gms, "AllResourcesExhaustedError")
    assert hasattr(gms, "ConfigurationError")


def test_exception_hierarchy():
    assert issubclass(gms.AllResourcesExhaustedError, gms.GeminiRotatorError)
    assert issubclass(gms.ConfigurationError, gms.GeminiRotatorError)
    assert issubclass(gms.GeminiRotatorError, Exception)


# ─────────────────────────────────────────────────────────────────────
# Registry getters
# ─────────────────────────────────────────────────────────────────────
def test_chatmodel_returns_string():
    m = gms.chatmodel()
    assert isinstance(m, str) and m


def test_embeddingmodel_returns_string():
    m = gms.embeddingmodel()
    assert isinstance(m, str) and m


def test_all_category_getters_return_strings():
    getters = [
        gms.chatmodel,
        gms.audiomodel,
        gms.imagemodel,
        gms.videomodel,
        gms.embeddingmodel,
        gms.musicmodel,
        gms.roboticsmodel,
        gms.computerusemodel,
        gms.researchmodel,
        gms.agentmodel,
        gms.gemmamodel,
    ]
    for g in getters:
        v = g()
        assert isinstance(v, str) and v, f"{g.__name__} returned {v!r}"


# ─────────────────────────────────────────────────────────────────────
# Registry lists
# ─────────────────────────────────────────────────────────────────────
def test_model_lists_exported():
    for name in (
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
    ):
        assert hasattr(gms, name), name
        obj = getattr(gms, name)
        assert isinstance(obj, list) or isinstance(obj, dict), name


def test_all_models_dict_has_expected_categories():
    expected = {
        "chat", "text", "audio", "image", "video", "embedding",
        "music", "robotics", "computer_use", "research", "agent", "gemma",
    }
    assert set(gms.ALL_CURRENT_MODELS.keys()) == expected


def test_chatmodel_is_first_of_chat_list():
    assert gms.chatmodel() == gms.CHAT_MODELS[0]


# ─────────────────────────────────────────────────────────────────────
# Registry utilities
# ─────────────────────────────────────────────────────────────────────
def test_get_models_by_category():
    chat = gms.get_models("chat")
    assert chat is gms.CHAT_MODELS


def test_get_models_unknown_category_raises():
    import pytest
    with pytest.raises(ValueError):
        gms.get_models("nonexistent")


def test_get_model_count_positive():
    assert gms.get_model_count() > 0


def test_model_exists_true_for_known():
    assert gms.model_exists(gms.chatmodel())


def test_model_exists_false_for_unknown():
    assert not gms.model_exists("not-a-real-model-xyz")


def test_get_model_category_for_chat_model():
    cats = gms.get_model_category(gms.chatmodel())
    assert "chat" in cats


# ─────────────────────────────────────────────────────────────────────
# No private leakage
# ─────────────────────────────────────────────────────────────────────
def test_no_private_names_in_all():
    for name in gms.__all__:
        assert not name.startswith("_"), f"{name} should not be public"
