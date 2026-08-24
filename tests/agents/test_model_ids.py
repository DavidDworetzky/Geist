from agents.model_ids import (
    META_LLAMA_3_8B_INSTRUCT_ID,
    META_LLAMA_31_8B_BASE_ID,
    META_LLAMA_31_8B_INSTRUCT_ID,
    canonicalize_local_model_id,
)


def test_canonicalizes_legacy_llama_model_ids():
    assert canonicalize_local_model_id("Meta-Llama-3.1-8B-Instruct") == META_LLAMA_31_8B_INSTRUCT_ID
    assert canonicalize_local_model_id("Meta-Llama-3.1-8B") == META_LLAMA_31_8B_BASE_ID
    assert canonicalize_local_model_id("Meta-Llama-3-8B-Instruct") == META_LLAMA_3_8B_INSTRUCT_ID


def test_preserves_already_canonical_and_qwen_model_ids():
    assert canonicalize_local_model_id(META_LLAMA_31_8B_INSTRUCT_ID) == META_LLAMA_31_8B_INSTRUCT_ID
    assert canonicalize_local_model_id("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_offline_catalog_exposes_each_llama_model_under_its_canonical_id():
    from agents.architectures.registry import STATIC_MODELS, OnlineModelProviders

    model_ids = [model.id for model in STATIC_MODELS[OnlineModelProviders.OFFLINE]]

    assert "Meta-Llama-3.1-8B-Instruct" not in model_ids
    assert model_ids.count(META_LLAMA_31_8B_INSTRUCT_ID) == 1
