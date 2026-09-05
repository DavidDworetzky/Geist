import argparse
import os
import re
import subprocess  # nosec B404 - CLI helper uses an argument list, never a shell

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Voice models load through the Hugging Face cache (`from_pretrained` /
# `hf_hub_download`) rather than app/model_weights, so they are pre-fetched
# into the cache instead of a local_dir. Each entry lists the repos to fetch;
# "allow_patterns" limits a fetch to the files the runtime actually loads.
VOICE_MODELS: dict[str, dict] = {
    "sesame-csm-1b": {
        "description": "Sesame CSM 1B TTS (+ Mimi audio codec and Llama tokenizer)",
        "downloads": [
            {"repo_id": "sesame/csm-1b"},
            # Mimi codec weights used by agents/architectures/sesame/generator.py
            # (see MIMI_NAME in agents/architectures/moshi/models/loaders.py).
            {
                "repo_id": "kyutai/moshiko-pytorch-bf16",
                "allow_patterns": ["tokenizer-e351c8d8-checkpoint125.safetensors"],
            },
            # Gated repo: only the tokenizer files are needed for CSM text input.
            {
                "repo_id": "meta-llama/Llama-3.2-1B",
                "allow_patterns": ["tokenizer*", "special_tokens_map.json"],
            },
        ],
    },
    "qwen3-tts-0.6b": {
        "description": "Qwen3 TTS 0.6B Custom Voice",
        "downloads": [{"repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}],
    },
    "qwen3-tts-1.7b": {
        "description": "Qwen3 TTS 1.7B Custom Voice",
        "downloads": [{"repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}],
    },
    "mms-1b-all": {
        "description": "Meta MMS 1B speech-to-text (adapters/mms_adapter.py)",
        "downloads": [
            {
                "repo_id": "facebook/mms-1b-all",
                "allow_patterns": [
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                ],
            }
        ],
    },
}


def model_dir_name(model_id):
    """Convert repository IDs to one safe local directory component."""
    directory_name = re.sub(r"[\\/]+", "_", model_id.strip()).strip(".")
    if not directory_name:
        raise ValueError("Model ID must contain a directory-safe name")
    return directory_name


def default_weights_dir(model_id):
    """Preserve the existing MLX Llama directory; isolate all other models."""
    if model_id == DEFAULT_MODEL_ID:
        return os.path.join("app", "model_weights", "llama_3_1")
    return os.path.join("app", "model_weights", model_dir_name(model_id))


def download_model_weights(model_id, weights_dir=None, use_cli=False, revision=None):
    """Download files without instantiating the model or doubling memory use."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    weights_dir = weights_dir or default_weights_dir(model_id)

    if use_cli:
        print(f"Downloading model to {weights_dir} using huggingface-cli")
        command = ["huggingface-cli", "download", model_id, "--local-dir", weights_dir]
        if revision:
            command.extend(["--revision", revision])
        subprocess.run(command, check=True)  # nosec B603 - model IDs are separate argv values
    else:
        print(f"Downloading model files to {weights_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=weights_dir,
            token=token,
            revision=revision,
        )

    print("Download complete!")


def download_voice_model_weights(voice_model):
    """Pre-fetch a voice (STT/TTS) model into the Hugging Face cache."""
    if voice_model not in VOICE_MODELS:
        raise ValueError(
            f"Unknown voice model '{voice_model}'. " f"Available: {', '.join(sorted(VOICE_MODELS))}"
        )

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    spec = VOICE_MODELS[voice_model]
    print(f"Downloading {voice_model}: {spec['description']}")

    for download in spec["downloads"]:
        repo_id = download["repo_id"]
        allow_patterns = download.get("allow_patterns")
        if allow_patterns and len(allow_patterns) == 1 and "*" not in allow_patterns[0]:
            print(f"Fetching {allow_patterns[0]} from {repo_id}")
            hf_hub_download(repo_id=repo_id, filename=allow_patterns[0], token=token)
        else:
            print(f"Fetching {repo_id} into the Hugging Face cache")
            snapshot_download(repo_id=repo_id, token=token, allow_patterns=allow_patterns)

    print("Download complete!")


# Backward-compatible import for existing scripts.
download_llama_weights = download_model_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face model weights")
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Directory to store weights (defaults to a model-specific directory)",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Optional immutable model revision"
    )
    parser.add_argument(
        "--use_cli",
        action="store_true",
        help="Use huggingface-cli for downloading instead of transformers",
    )
    parser.add_argument(
        "--voice_model",
        type=str,
        default=None,
        choices=[*sorted(VOICE_MODELS), "all"],
        help="Pre-fetch a voice (STT/TTS) model into the Hugging Face "
        "cache instead of downloading an LLM. Use 'all' for every "
        "voice model. Gated repos need HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--list_voice_models", action="store_true", help="List downloadable voice models and exit"
    )

    args = parser.parse_args()

    if args.list_voice_models:
        for name in sorted(VOICE_MODELS):
            print(f"{name}: {VOICE_MODELS[name]['description']}")
    elif args.voice_model:
        names = sorted(VOICE_MODELS) if args.voice_model == "all" else [args.voice_model]
        for name in names:
            download_voice_model_weights(name)
    else:
        download_model_weights(args.model_id, args.weights_dir, args.use_cli, args.revision)
