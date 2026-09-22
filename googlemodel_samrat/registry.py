"""
Google Gemini API - Current Model Registry
==========================================

IMPORTANT NOTES:
    - Contains currently listed Gemini/Google AI model IDs across all 12 categories.
    - Ordered strictly from LATEST to OLDEST to support automatic failover rotation.
    - Includes helper getter functions for all categories.
"""

# ============================================================
# 1. CHAT / TEXT / REASONING MODELS (Latest to Oldest)
# ============================================================
CHAT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
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
EMBEDDING_MODELS = [
    "gemini-embedding-2-preview",
    "gemini-embedding-2",
    "gemini-embedding-001",
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

def chatmodel() -> str:
    """Returns the newest top-priority chat model."""
    if not CHAT_MODELS: raise ValueError("No chat models registered.")
    return CHAT_MODELS[0]

def audiomodel() -> str:
    """Returns the newest top-priority audio model."""
    if not AUDIO_MODELS: raise ValueError("No audio models registered.")
    return AUDIO_MODELS[0]

def imagemodel() -> str:
    """Returns the newest top-priority image model."""
    if not IMAGE_MODELS: raise ValueError("No image models registered.")
    return IMAGE_MODELS[0]

def videomodel() -> str:
    """Returns the newest top-priority video model."""
    if not VIDEO_MODELS: raise ValueError("No video models registered.")
    return VIDEO_MODELS[0]

def embeddingmodel() -> str:
    """Returns the newest top-priority embedding model."""
    if not EMBEDDING_MODELS: raise ValueError("No embedding models registered.")
    return EMBEDDING_MODELS[0]

def musicmodel() -> str:
    """Returns the newest top-priority music model."""
    if not MUSIC_MODELS: raise ValueError("No music models registered.")
    return MUSIC_MODELS[0]

def roboticsmodel() -> str:
    """Returns the newest top-priority robotics model."""
    if not ROBOTICS_MODELS: raise ValueError("No robotics models registered.")
    return ROBOTICS_MODELS[0]

def computerusemodel() -> str:
    """Returns the newest top-priority computer use model."""
    if not COMPUTER_USE_MODELS: raise ValueError("No computer use models registered.")
    return COMPUTER_USE_MODELS[0]

def researchmodel() -> str:
    """Returns the newest top-priority deep research model."""
    if not RESEARCH_MODELS: raise ValueError("No research models registered.")
    return RESEARCH_MODELS[0]

def agentmodel() -> str:
    """Returns the newest top-priority agent model."""
    if not AGENT_MODELS: raise ValueError("No agent models registered.")
    return AGENT_MODELS[0]

def gemmamodel() -> str:
    """Returns the newest top-priority Gemma model."""
    if not GEMMA_MODELS: raise ValueError("No Gemma models registered.")
    return GEMMA_MODELS[0]


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