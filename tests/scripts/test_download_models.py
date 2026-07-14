"""Tests for model-specific weight download locations."""
from scripts.download_models import DEFAULT_MODEL_ID, default_weights_dir


def test_default_llama_download_preserves_legacy_mlx_directory():
    assert default_weights_dir(DEFAULT_MODEL_ID) == "app/model_weights/llama_3_1"


def test_other_models_use_isolated_directories():
    assert default_weights_dir("Qwen/Qwen3-4B") == "app/model_weights/Qwen_Qwen3-4B"
