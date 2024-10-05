import os
import argparse
import subprocess
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_llama_weights(model_id, weights_dir, use_cli=False):
    # Login to Hugging Face
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    login(token=token)

    if use_cli:
        # Download using huggingface-cli
        print(f"Downloading model to {weights_dir} using huggingface-cli")
        command = f"huggingface-cli download {model_id} --include \"original/*\" --local-dir {weights_dir}"
        subprocess.run(command, shell=True, check=True)
    else:
        # Download model and tokenizer using transformers
        print(f"Downloading model to {weights_dir} using transformers")
        
        # Add custom configuration to handle rope_scaling mismatch
        config_kwargs = {
            "rope_scaling": {
                "type": "linear",
                "factor": 8.0
            },
        }
        
        AutoModelForCausalLM.from_pretrained(model_id, cache_dir=weights_dir, token=token, **config_kwargs)
        AutoTokenizer.from_pretrained(model_id, cache_dir=weights_dir, token=token)

    print("Download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama model weights")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Hugging Face model ID")
    parser.add_argument("--weights_dir", type=str, default="app/model_weights/llama_3_1",
                        help="Directory to store the model weights")
    parser.add_argument("--use_cli", action="store_true",
                        help="Use huggingface-cli for downloading instead of transformers")
    
    args = parser.parse_args()
    
    download_llama_weights(args.model_id, args.weights_dir, args.use_cli)