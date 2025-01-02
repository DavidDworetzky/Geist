import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import numpy as np
import json
import os
import logging
from typing import List, Dict
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from mlx.utils import tree_unflatten
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    
class LlamaMLX:
    def __init__(self, max_new_tokens: int, temperature: float = 0.7, top_p: float = 0.95):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.max_new_tokens = max_new_tokens
        self.weights_dir = "app/model_weights/llama_3_1"
        self.temperature = temperature
        self.top_p = top_p
        self.config = ModelConfig()
        
        logger.info("Loading model and tokenizer")
        self.load_model_and_tokenizer()
        logger.info("Model and tokenizer loaded successfully")

    def load_model_and_tokenizer(self):
        # Load tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_dir)
        
        # Load model weights and config
        with open(os.path.join(self.weights_dir, "config.json"), "r") as f:
            config = json.load(f)
            self.config = ModelConfig(**config)
        
        # Load weights into MLX format
        weights = np.load(os.path.join(self.weights_dir, "weights.npz"))
        self.weights = {k: mx.array(v) for k, v in weights.items()}
        
    def download_model(self):
        """Download model weights and tokenizer from HuggingFace"""
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
            
        # Download model files
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "weights.npz"]
        for file in files:
            hf_hub_download(
                repo_id=self.model_id,
                filename=file,
                token=token,
                cache_dir=self.weights_dir
            )
        logger.info(f"Downloaded model files to {self.weights_dir}")

    def generate_text(self, prompt: str) -> mx.array:
        """Generate text using MLX"""
        # Tokenize input
        tokens = mx.array(self.tokenizer.encode(prompt))
        
        # Initialize generation
        for _ in range(self.max_new_tokens):
            # Get predictions for next token
            logits = self.model(tokens)
            
            # Apply temperature scaling
            if self.temperature > 0:
                logits = logits / self.temperature
                
            # Apply top-p sampling
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = mx.sort(logits, descending=True)
                cumulative_probs = mx.cumsum(mx.softmax(sorted_logits), axis=-1)
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                logits = mx.scatter(sorted_logits, sorted_indices)
            
            # Sample next token
            next_token = mx.argmax(logits[-1])
            tokens = mx.concatenate([tokens, mx.array([next_token])])
            
            # Check for end of generation
            if next_token == self.tokenizer.eos_token_id:
                break
                
        return tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Complete the prompt with system and user messages"""
        # Check if model exists, if not, download it
        if not os.path.exists(os.path.join(self.weights_dir, "config.json")):
            self.download_model()

        # Format prompt
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""

        logger.info(f"Starting text generation with parameters - temperature {self.temperature}, top_p {self.top_p}, max_new_tokens {self.max_new_tokens}")
        
        try:
            # Generate text
            output_tokens = self.generate_text(prompt)
            
            # Decode output
            output_text = self.tokenizer.decode(output_tokens.tolist())
            
            logger.info("Text generation completed successfully")
            logger.info(f"Output: {output_text}")
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise