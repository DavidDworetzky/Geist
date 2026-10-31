"""Canonical identifiers for local language models."""

META_LLAMA_31_8B_INSTRUCT_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
META_LLAMA_31_8B_BASE_ID = "meta-llama/Meta-Llama-3.1-8B"
META_LLAMA_3_8B_INSTRUCT_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

DEFAULT_LOCAL_MODEL_ID = META_LLAMA_31_8B_INSTRUCT_ID

_LEGACY_LOCAL_MODEL_IDS = {
    "Meta-Llama-3.1-8B-Instruct": META_LLAMA_31_8B_INSTRUCT_ID,
    "Meta-Llama-3.1-8B": META_LLAMA_31_8B_BASE_ID,
    "Meta-Llama-3-8B-Instruct": META_LLAMA_3_8B_INSTRUCT_ID,
}


def canonicalize_local_model_id(model_id: str) -> str:
    """Return the canonical repository-qualified ID for a local model."""
    return _LEGACY_LOCAL_MODEL_IDS.get(model_id, model_id)
