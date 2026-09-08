"""Tests for model-specific weight download locations."""

import os

import pytest

from scripts.copy_weights import model_dir_name as copy_model_dir_name
from scripts.download_models import (
    DEFAULT_MODEL_ID,
    VOICE_MODELS,
    default_weights_dir,
    download_voice_model_weights,
)


def test_default_llama_download_preserves_legacy_mlx_directory():
    assert default_weights_dir(DEFAULT_MODEL_ID) == "app/model_weights/llama_3_1"


def test_other_models_use_isolated_directories():
    assert default_weights_dir("Qwen/Qwen3-4B") == "app/model_weights/Qwen_Qwen3-4B"


def test_model_directories_cannot_escape_on_windows_or_posix():
    hostile_model_id = r"..\..\private/model"
    download_path = default_weights_dir(hostile_model_id)

    assert os.path.dirname(download_path) == "app/model_weights"
    assert copy_model_dir_name(hostile_model_id) == "_.._private_model"


def test_unknown_voice_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown voice model"):
        download_voice_model_weights("not-a-voice-model")


def test_voice_models_download_into_hugging_face_cache(monkeypatch):
    """Voice runtimes load via from_pretrained/hf_hub_download, so voice
    downloads must go to the HF cache (no local_dir) to be found offline."""
    snapshot_calls = []
    file_calls = []

    monkeypatch.setattr(
        "scripts.download_models.snapshot_download",
        lambda **kwargs: snapshot_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "scripts.download_models.hf_hub_download",
        lambda **kwargs: file_calls.append(kwargs),
    )

    for name in VOICE_MODELS:
        download_voice_model_weights(name)

    assert all("local_dir" not in call for call in snapshot_calls)
    fetched_repos = {call["repo_id"] for call in snapshot_calls + file_calls}
    assert "sesame/csm-1b" in fetched_repos
    assert "facebook/mms-1b-all" in fetched_repos
    # Sesame needs the Mimi codec weights alongside the CSM checkpoint
    assert any(call["repo_id"] == "kyutai/moshiko-pytorch-bf16" for call in file_calls)


def test_voice_registry_covers_local_tts_provider_models():
    """Every locally-run TTS model published to the frontend should be
    downloadable through the voice model registry."""
    tts = pytest.importorskip("app.services.tts")

    registry_repos = {
        download["repo_id"] for spec in VOICE_MODELS.values() for download in spec["downloads"]
    }
    local_models = pytest.importorskip("app.services.local_models")
    managed_artifact_ids = {artifact.id for artifact in local_models.CURATED_LOCAL_ARTIFACTS}

    for provider in tts.SUPPORTED_TTS_PROVIDERS:
        if provider["type"] != "local":
            continue
        for model in provider["models"]:
            if model.get("artifact_id"):
                assert model["artifact_id"] in managed_artifact_ids
                continue
            assert model["id"] in registry_repos, (
                f"Local TTS model {model['id']} is not downloadable via "
                "scripts/download_models.py VOICE_MODELS"
            )
