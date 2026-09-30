"""
Google Gemini API - Current Model Registry
==========================================

IMPORTANT NOTES:
    - Contains currently listed Gemini/Google AI model IDs across all 12 categories.
    - Ordered strictly from LATEST to OLDEST to support automatic failover rotation.
    - Includes helper getter functions for all categories.
    - v0.2.0: getters return RotatingModelName — a str subclass that
      carries its failover pool, so rotation travels with the name.
"""

from __future__ import annotations

from typing import Any, List, Optional


# ============================================================
# 0. ROTATING MODEL NAME (v0.2.0 flagship)
# ============================================================
class RotatingModelName(str):
    """A ``str`` that carries its rotation context.

    Behaves EXACTLY like ``str`` for every existing use::

        len(name), name.upper(), f"{name}", name in CHAT_MODELS,
        client(model=name), name == "gemini-3.5-flash-lite"

    ...but it also carries the full failover pool it came from, so
    rotation-aware clients (``googlemodel_samrat.ChatGoogleGenerativeAI``
    and ``googlemodel_samrat.GoogleGenerativeAIEmbeddings``) adopt the
    whole pool automatically when you pass it as ``model=``::

        llm = ChatGoogleGenerativeAI(api_key="...", model=chatmodel())
        # -> models pool = ALL of CHAT_MODELS, not just one model

    Stock (non-rotation-aware) clients still receive a perfectly
    ordinary string and keep working unchanged — they just don't
    rotate, which is exactly why ``googlemodel_samrat.langchain``
    exists as a one-line migration path.
    """

    def __new__(
        cls,
        name: str,
        *,
        pool: Optional[List[str]] = None,
        manager: Any = None,
    ) -> "RotatingModelName":
        obj = super().__new__(cls, name)
        obj._pool = list(pool) if pool else [str(name)]
        obj._manager = manager
        return obj

    @property
    def rotation_pool(self) -> List[str]:
        """The full failover pool (category order) this name came from."""
        return list(self._pool)


# ============================================================
# 1. CHAT / TEXT / REASONING MODELS (failover priority)
# ============================================================
# v0.2.0 order: the two flash-lites first for maximum free-tier RPD
# headroom, then the alternating fast/heavy flow.
CHAT_MODELS = [
    "gemini-3.5-flash-lite",       # 1.  fast  — first choice
    "gemini-3.1-flash-lite",       # 2.  fast  — second lite
    "gemini-3.8-flash",            # 3.  heavy
    "gemini-3.1-pro-preview",      # 4.  heavy
    "gemini-2.5-flash-lite",       # 5.  fast
    "gemini-3-flash-preview",      # 6.  heavy
    "gemini-2.5-flash",            # 7.  fast
    "gemini-3.7-flash",            # 8.  heavy
    "gemini-3.5-flash",            # 9.  fast
    "gemini-3.6-flash",            # 10. heavy
    "gemini-2.5-pro",              # 11. heavy
    "gemini-1.5-flash-latest",     # 12. last resort
]

TEXT_MODELS = CHAT_MODELS

# ============================================================
# 2. AUDIO / LIVE / SPEECH MODELS
# ============================================================
AUDIO_MODELS = [
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3.5-live-translate-preview",
    "gemini-3.1-flash-live-preview",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.5-transcribe",
    "gemini-3.5-transcribe-live",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

# ============================================================
# 3. IMAGE GENERATION / EDITING MODELS
# ============================================================
IMAGE_MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]

# ============================================================
# 4. VIDEO / MULTIMODAL GENERATION MODELS
# ============================================================
VIDEO_MODELS = [
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
    "gemini-omni-1.1-flash",
]

# ============================================================
# 5. EMBEDDING MODELS
# ============================================================
# v0.2.0 order: GA model FIRST — live testing hit back-to-back 429s
# because the preview models sat on top of the list. Preview models
# have lower per-project quotas than GA models. Same philosophy as
# CHAT_MODELS: abundant-quota resource first.
EMBEDDING_MODELS = [
    "gemini-embedding-001",        # GA — 100 RPM, most free-tier headroom
    "gemini-embedding-2",          # newer
    "gemini-embedding-2-preview",  # preview — lowest quota tier
]

# ============================================================
# 6. MUSIC GENERATION MODELS
# ============================================================
MUSIC_MODELS = [
    "lyria-3.5",
    "lyria-3-clip-preview",
    "lyria-3-pro-preview",
    "lyria-realtime-exp",
]

# ============================================================
# 7. ROBOTICS MODELS
# ============================================================
ROBOTICS_MODELS = [
    "gemini-robotics-er-2-preview",
    "gemini-robotics-er-2-streaming-preview",
]

# ============================================================
# 8. COMPUTER USE MODELS
# ============================================================
COMPUTER_USE_MODELS = [
    "gemini-2.5-computer-use-preview-10-2025",
]

# ============================================================
# 9. DEEP RESEARCH MODELS
# ============================================================
RESEARCH_MODELS = [
    "deep-research-max-preview-04-2026",
    "deep-research-preview-04-2026",
]

# ============================================================
# 10. MANAGED AGENT MODELS
# ============================================================
AGENT_MODELS = [
    "antigravity-preview-09-2026",
]

# ============================================================
# 11. GEMMA OPEN-WEIGHT MODELS
# ============================================================
GEMMA_MODELS = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]

# ============================================================
# 12. COMPLETE MODEL REGISTRY DICTIONARY
# ============================================================
ALL_CURRENT_MODELS = {
    "chat":         CHAT_MODELS,
    "text":         TEXT_MODELS,
    "audio":        AUDIO_MODELS,
    "image":        IMAGE_MODELS,
    "video":        VIDEO_MODELS,
    "embedding":    EMBEDDING_MODELS,
    "music":        MUSIC_MODELS,
    "robotics":     ROBOTICS_MODELS,
    "computer_use": COMPUTER_USE_MODELS,
    "research":     RESEARCH_MODELS,
    "agent":        AGENT_MODELS,
    "gemma":        GEMMA_MODELS,
}

# ============================================================
# 13. CONVENIENCE HELPERS FOR ALL 12 CATEGORIES
# ============================================================
# v0.2.0: every getter returns a RotatingModelName — a plain str for
# all existing code, but carrying its full failover pool so rotation
# travels with the name into any rotation-aware client:
#
#     llm = ChatGoogleGenerativeAI(api_key="...", model=chatmodel())
#     # -> the WHOLE CHAT_MODELS pool rotates, not just one model.

def chatmodel() -> RotatingModelName:
    """Newest top-priority chat model (carries the CHAT_MODELS pool)."""
    if not CHAT_MODELS: raise ValueError("No chat models registered.")
    return RotatingModelName(CHAT_MODELS[0], pool=CHAT_MODELS)

def audiomodel() -> RotatingModelName:
    """Newest top-priority audio model (carries the AUDIO_MODELS pool)."""
    if not AUDIO_MODELS: raise ValueError("No audio models registered.")
    return RotatingModelName(AUDIO_MODELS[0], pool=AUDIO_MODELS)

def imagemodel() -> RotatingModelName:
    """Newest top-priority image model (carries the IMAGE_MODELS pool)."""
    if not IMAGE_MODELS: raise ValueError("No image models registered.")
    return RotatingModelName(IMAGE_MODELS[0], pool=IMAGE_MODELS)

def videomodel() -> RotatingModelName:
    """Newest top-priority video model (carries the VIDEO_MODELS pool)."""
    if not VIDEO_MODELS: raise ValueError("No video models registered.")
    return RotatingModelName(VIDEO_MODELS[0], pool=VIDEO_MODELS)

def embeddingmodel() -> RotatingModelName:
    """Newest top-priority embedding model (carries the EMBEDDING_MODELS pool).

    v0.2.0: returns the GA model `gemini-embedding-001` first —
    live testing showed the preview models 429 on a normal day.
    """
    if not EMBEDDING_MODELS: raise ValueError("No embedding models registered.")
    return RotatingModelName(EMBEDDING_MODELS[0], pool=EMBEDDING_MODELS)

def musicmodel() -> RotatingModelName:
    """Newest top-priority music model (carries the MUSIC_MODELS pool)."""
    if not MUSIC_MODELS: raise ValueError("No music models registered.")
    return RotatingModelName(MUSIC_MODELS[0], pool=MUSIC_MODELS)

def roboticsmodel() -> RotatingModelName:
    """Newest top-priority robotics model (carries the ROBOTICS_MODELS pool)."""
    if not ROBOTICS_MODELS: raise ValueError("No robotics models registered.")
    return RotatingModelName(ROBOTICS_MODELS[0], pool=ROBOTICS_MODELS)

def computerusemodel() -> RotatingModelName:
    """Newest top-priority computer use model (carries the pool)."""
    if not COMPUTER_USE_MODELS: raise ValueError("No computer use models registered.")
    return RotatingModelName(COMPUTER_USE_MODELS[0], pool=COMPUTER_USE_MODELS)

def researchmodel() -> RotatingModelName:
    """Newest top-priority deep research model (carries the pool)."""
    if not RESEARCH_MODELS: raise ValueError("No research models registered.")
    return RotatingModelName(RESEARCH_MODELS[0], pool=RESEARCH_MODELS)

def agentmodel() -> RotatingModelName:
    """Newest top-priority agent model (carries the AGENT_MODELS pool)."""
    if not AGENT_MODELS: raise ValueError("No agent models registered.")
    return RotatingModelName(AGENT_MODELS[0], pool=AGENT_MODELS)

def gemmamodel() -> RotatingModelName:
    """Newest top-priority Gemma model (carries the GEMMA_MODELS pool)."""
    if not GEMMA_MODELS: raise ValueError("No Gemma models registered.")
    return RotatingModelName(GEMMA_MODELS[0], pool=GEMMA_MODELS)

# ============================================================
# 14. GENERAL REGISTRY UTILITIES
# ============================================================

def get_all_models() -> dict:
    return ALL_CURRENT_MODELS

def get_models(category: str) -> list:
    category = category.lower().strip()
    if category not in ALL_CURRENT_MODELS:
        raise ValueError(f"Unknown category '{category}'. Valid: {list(ALL_CURRENT_MODELS.keys())}")
    return ALL_CURRENT_MODELS[category]

def get_model_count() -> int:
    unique_models = set()
    for models in ALL_CURRENT_MODELS.values():
        unique_models.update(models)
    return len(unique_models)

def print_all_models() -> None:
    print("\n" + "=" * 70)
    print("GOOGLE GEMINI MODEL REGISTRY (Latest to Oldest Priority)")
    print("=" * 70)
    for category, models in ALL_CURRENT_MODELS.items():
        print(f"\n[{category.upper()} - {len(models)} models]")
        for index, model in enumerate(models, start=1):
            print(f"  {index:>2}. {model}")
    print("=" * 70)

def model_exists(model_name: str) -> bool:
    for models in ALL_CURRENT_MODELS.values():
        if model_name in models: return True
    return False

def get_model_category(model_name: str) -> list:
    return [cat for cat, models in ALL_CURRENT_MODELS.items() if model_name in models]