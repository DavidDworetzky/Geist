import argparse
import os
import subprocess

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def default_weights_dir(model_id):
    """Preserve the existing MLX Llama directory; isolate all other models."""
    if model_id == DEFAULT_MODEL_ID:
        return os.path.join("app", "model_weights", "llama_3_1")
    return os.path.join("app", "model_weights", model_id.replace("/", "_"))


def download_model_weights(model_id, weights_dir=None, use_cli=False, revision=None):
    """Download files without instantiating the model or doubling memory use."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    weights_dir = weights_dir or default_weights_dir(model_id)

    if use_cli:
        print(f"Downloading model to {weights_dir} using huggingface-cli")
        command = ["huggingface-cli", "download", model_id, "--local-dir", weights_dir]
        if revision:
            command.extend(["--revision", revision])
        subprocess.run(command, check=True)
    else:
        print(f"Downloading model files to {weights_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=weights_dir,
            token=token,
            revision=revision,
        )

    print("Download complete!")


# Backward-compatible import for existing scripts.
download_llama_weights = download_model_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face model weights")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Hugging Face model ID")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Directory to store weights (defaults to a model-specific directory)")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional immutable model revision")
    parser.add_argument("--use_cli", action="store_true",
                        help="Use huggingface-cli for downloading instead of transformers")

    args = parser.parse_args()

    download_model_weights(args.model_id, args.weights_dir, args.use_cli, args.revision)
