import os
import argparse
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_llama_weights(model_id, weights_dir):
    # Login to Hugging Face
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    login(token=token)

    # Download model and tokenizer
    print(f"Downloading model to {weights_dir}")
    
    # Add custom configuration to handle rope_scaling mismatch
    config_kwargs = {
        "rope_scaling": {
            "type": "linear",
            "factor": 8.0
        }
    }
    
    AutoModelForCausalLM.from_pretrained(model_id, cache_dir=weights_dir, **config_kwargs)
    AutoTokenizer.from_pretrained(model_id, cache_dir=weights_dir)
    print("Download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama model weights")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Hugging Face model ID")
    parser.add_argument("--weights_dir", type=str, default="app/model_weights/llama_3_1",
                        help="Directory to store the model weights")
    
    args = parser.parse_args()
    
    download_llama_weights(args.model_id, args.weights_dir)