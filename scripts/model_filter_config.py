"""
Model Filter Configuration

Configures which models to include/exclude when syncing from provider APIs.
"""
import re
from typing import Dict, List, Optional

# Models to always include regardless of API response
ALWAYS_INCLUDE: List[str] = [
    # OpenAI
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
    # Anthropic
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

# Patterns to exclude (regex)
EXCLUDE_PATTERNS: List[str] = [
    r".*-instruct$",       # Exclude instruct variants for some providers
    r".*-vision-preview$", # Exclude preview versions
    r"ft:.*",              # Exclude fine-tuned models
    r".*:ft:.*",           # Exclude fine-tuned models (alt format)
    r".*embedding.*",      # Exclude embedding models
    r".*whisper.*",        # Exclude whisper models
    r".*tts.*",            # Exclude text-to-speech models
    r".*dall-e.*",         # Exclude DALL-E models
    r".*-search-.*",       # Exclude search models
    r".*babbage.*",        # Exclude old models
    r".*davinci.*",        # Exclude old davinci models
    r".*curie.*",          # Exclude old curie models
    r".*ada.*",            # Exclude old ada models (but not embedding-ada)
    r".*text-.*",          # Exclude legacy text completions
    r".*code-.*",          # Exclude legacy code models
    r".*moderation.*",     # Exclude moderation models
]

# Chat model prefixes to include
CHAT_MODEL_PREFIXES: List[str] = [
    "gpt-4",
    "gpt-3.5",
    "o1",
    "o3",
    "claude",
    "grok",
    "llama",
    "mixtral",
    "qwen",
]

# Model metadata overrides - used to enhance API-provided data
MODEL_METADATA: Dict[str, Dict] = {
    # OpenAI models
    "gpt-4": {
        "name": "GPT-4",
        "recommended": False,
        "context_window": 8192,
        "max_output_tokens": 4096,
        "supports_vision": False,
        "supports_function_calling": True,
        "family": "gpt-4",
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "recommended": True,
        "context_window": 128000,
        "max_output_tokens": 4096,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "gpt-4",
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "recommended": True,
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "gpt-4o",
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "recommended": True,
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "gpt-4o",
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "recommended": False,
        "context_window": 16385,
        "max_output_tokens": 4096,
        "supports_vision": False,
        "supports_function_calling": True,
        "family": "gpt-3.5",
    },
    "o1": {
        "name": "O1",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 100000,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "o1",
    },
    "o1-mini": {
        "name": "O1 Mini",
        "recommended": True,
        "context_window": 128000,
        "max_output_tokens": 65536,
        "supports_vision": False,
        "supports_function_calling": True,
        "family": "o1",
    },
    "o3-mini": {
        "name": "O3 Mini",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 100000,
        "supports_vision": False,
        "supports_function_calling": True,
        "family": "o3",
    },
    # Anthropic models
    "claude-3-opus-20240229": {
        "name": "Claude 3 Opus",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 4096,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "claude-3",
    },
    "claude-3-sonnet-20240229": {
        "name": "Claude 3 Sonnet",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 4096,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "claude-3",
    },
    "claude-3-haiku-20240307": {
        "name": "Claude 3 Haiku",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 4096,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "claude-3",
    },
    "claude-3-5-sonnet-20241022": {
        "name": "Claude 3.5 Sonnet",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 8192,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "claude-3.5",
    },
    "claude-3-5-haiku-20241022": {
        "name": "Claude 3.5 Haiku",
        "recommended": True,
        "context_window": 200000,
        "max_output_tokens": 8192,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "claude-3.5",
    },
    # XAI models
    "grok-2": {
        "name": "Grok 2",
        "recommended": True,
        "context_window": 131072,
        "max_output_tokens": 32768,
        "supports_vision": False,
        "supports_function_calling": True,
        "family": "grok",
    },
    "grok-2-vision": {
        "name": "Grok 2 Vision",
        "recommended": True,
        "context_window": 32768,
        "max_output_tokens": 8192,
        "supports_vision": True,
        "supports_function_calling": True,
        "family": "grok",
    },
    "grok-3": {
        "name": "Grok 3",
        "recommended": True,
        "context_window": 131072,
        "max_output_tokens": 32768,
        "supports_vision": False,
        "supports_function_calling": True,
        "family": "grok",
    },
}

# HuggingFace model families to search for
HUGGINGFACE_MODEL_FAMILIES: List[str] = [
    "meta-llama",
    "Qwen",
    "mistralai",
    "microsoft",
    "google",
    "deepseek-ai",
    "01-ai",
    "THUDM",  # GLM models
    "moonshot",  # Kimi models
]

# Specific HuggingFace models to include
HUGGINGFACE_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-V2.5",
    "THUDM/glm-4-9b-chat",
]


def is_chat_model(model_id: str) -> bool:
    """
    Filter for chat-capable models.

    Args:
        model_id: The model identifier

    Returns:
        True if the model is a chat model
    """
    model_id_lower = model_id.lower()

    # Check against exclude patterns first
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, model_id_lower):
            return False

    # Check if it matches any chat model prefix
    return any(model_id_lower.startswith(prefix) for prefix in CHAT_MODEL_PREFIXES)


def should_include_model(model_id: str) -> bool:
    """
    Determine if a model should be included in the registry.

    Args:
        model_id: The model identifier

    Returns:
        True if the model should be included
    """
    # Always include models in the ALWAYS_INCLUDE list
    if model_id in ALWAYS_INCLUDE:
        return True

    # Check if it's a chat model
    return is_chat_model(model_id)


def get_model_metadata(model_id: str) -> Optional[Dict]:
    """
    Get metadata overrides for a model.

    Args:
        model_id: The model identifier

    Returns:
        Metadata dictionary if available, None otherwise
    """
    return MODEL_METADATA.get(model_id)


def generate_display_name(model_id: str) -> str:
    """
    Generate a human-readable display name from a model ID.

    Args:
        model_id: The model identifier

    Returns:
        Human-readable display name
    """
    # Check if we have a custom name
    metadata = get_model_metadata(model_id)
    if metadata and "name" in metadata:
        return metadata["name"]

    # Generate from ID
    name = model_id.replace("-", " ").replace("_", " ")
    # Capitalize words, handling special cases
    words = name.split()
    capitalized = []
    for word in words:
        if word.lower() in ("gpt", "llm", "ai", "api"):
            capitalized.append(word.upper())
        elif word.lower() == "mini":
            capitalized.append("Mini")
        else:
            capitalized.append(word.capitalize())

    return " ".join(capitalized)
