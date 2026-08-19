"""Tests for model-specific weight download locations."""
import os

import pytest

from scripts.copy_weights import model_dir_name as copy_model_dir_name
from scripts.download_models import DEFAULT_MODEL_ID, default_weights_dir


def test_default_llama_download_preserves_legacy_mlx_directory():
    assert default_weights_dir(DEFAULT_MODEL_ID) == "app/model_weights/llama_3_1"


@pytest.mark.parametrize("size", ["4B", "8B"])
def test_qwen3_models_use_isolated_directories(size):
    assert default_weights_dir(f"Qwen/Qwen3-{size}") == f"app/model_weights/Qwen_Qwen3-{size}"


def test_model_directories_cannot_escape_on_windows_or_posix():
    hostile_model_id = r"..\..\private/model"
    download_path = default_weights_dir(hostile_model_id)

    assert os.path.dirname(download_path) == "app/model_weights"
    assert copy_model_dir_name(hostile_model_id) == "_.._private_model"
