import os
import time
import json
import glob
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# Tokenizer imports
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import safetensors
import torch
import safetensors.torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """
    Merge of 'ModelArgs' from the first code and default fields from the second.
    """
    # Matches the naming of the first script for correctness, with some defaults:
    dim: int = 4096          # was "n_embd" in second script
    n_layers: int = 32       # was "n_layer"
    n_heads: int = 32        # was "n_head"
    # Additional fields from the first code:
    head_dim: int = 128      # will often be dim // n_heads
    n_kv_heads: int = 32
    hidden_dim: int = 11008
    norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 10000
    rope_traditional: bool = True

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        # How many times each key/value head is repeated to match the query heads
        self.repeats = self.n_heads // self.n_kv_heads

        # Per the LLama approach, scale = head_dim^-0.5
        self.scale = args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        # RoPE for positional embeddings
        self.rope = nn.RoPE(args.head_dim, traditional=args.rope_traditional, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries = self.wq(x)
        keys    = self.wk(x)
        values  = self.wv(x)

        # Reshape into (B, n_heads, L, head_dim) for queries,
        #            (B, n_kv_heads, L, head_dim) for keys/values
        queries = queries.reshape(B, L, self.n_heads,  self.args.head_dim).transpose(0, 2, 1, 3)
        keys    = keys   .reshape(B, L, self.n_kv_heads, self.args.head_dim).transpose(0, 2, 1, 3)
        values  = values .reshape(B, L, self.n_kv_heads, self.args.head_dim).transpose(0, 2, 1, 3)

        # More efficient implementation of key/value head repetition
        if self.repeats > 1:
            # Use reshape and repeat operations more efficiently
            # This avoids creating multiple copies in memory
            keys_expanded = mx.repeat(
                mx.expand_dims(keys, axis=1),  # Shape: [B, 1, n_kv_heads, L, head_dim]
                repeats=self.repeats,
                axis=1
            )  # Shape: [B, repeats, n_kv_heads, L, head_dim]
            keys = keys_expanded.reshape(B, self.n_heads, L, self.args.head_dim)
            
            values_expanded = mx.repeat(
                mx.expand_dims(values, axis=1),  # Shape: [B, 1, n_kv_heads, L, head_dim]
                repeats=self.repeats,
                axis=1
            )  # Shape: [B, repeats, n_kv_heads, L, head_dim]
            values = values_expanded.reshape(B, self.n_heads, L, self.args.head_dim)
        
        # Handle caching
        if cache is not None:
            key_cache, value_cache = cache
            # offset = length so far for rope
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            keys    = self.rope(keys,    offset=offset)
            keys    = mx.concatenate([key_cache, keys], axis=2)
            values  = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys    = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # LLaMA uses SwiGLU, here approximated with silu * linear
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Attention
        r, cache = self.attention(self.attention_norm(x), mask=mask, cache=cache)
        h = x + r
        # Feedforward
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Llama(nn.Module):
    """
    Core LLaMA model definition from the first code sample.
    """
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for full sequence inference (just like in the first script).
        """
        # Create a causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        x = self.tok_embeddings(x)
        for layer in self.layers:
            x, _ = layer(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, x: mx.array, temp=1.0, top_p=1.0, max_new_tokens=100, eos_token_id=None):
        """
        Token-by-token generator with optional temperature and top-p sampling.
        Yields one token at a time until EOS token or max_new_tokens is reached.
        
        Parameters:
            x (mx.array): Input token ids with shape (batch_size, seq_len)
            temp (float): Temperature for sampling (1.0 = no change, <1.0 = more deterministic)
            top_p (float): Nucleus sampling probability threshold
            max_new_tokens (int): Maximum number of new tokens to generate
            eos_token_id (int, optional): Token ID that signals the end of sequence
            
        Yields:
            int: Generated token IDs one at a time
        """
        def sample_logits(logits: mx.array) -> int:
            """
            Sample a token from the logits with temperature scaling and optional nucleus sampling.
            
            Parameters:
                logits (mx.array): Logits from the model.
            
            Returns:
                int: The sampled token.
            """
            # Optionally apply temperature scaling:
            if temp != 0.0:
                logits = logits / temp

            # Convert to a NumPy array for sampling logic in Python
            np_logits = np.array(logits)

            # Ensure logits has a 1D shape
            if np_logits.ndim != 1:
                logger.error(f"Unexpected logits shape: {np_logits.shape}. Expected a 1D tensor.")
                raise ValueError("Logits tensor has an unexpected shape.")

            # Check for invalid values in the logits tensor
            if np.any(np.isnan(np_logits)) or np.any(np.isinf(np_logits)):
                logger.error("Logits tensor contains NaN or Inf values.")
                raise ValueError("Logits tensor contains invalid values (NaN or Inf).")

            # If top_p < 1.0, perform nucleus (top-p) sampling:
            if top_p < 1.0:
                # Compute probabilities once
                probs = np.exp(np_logits - np.max(np_logits))  # Subtract max for numerical stability
                probs = probs / np.sum(probs)
                
                # Sort probabilities in descending order
                sorted_indices = np.argsort(-probs)
                sorted_probs = probs[sorted_indices]
                cumulative_probs = np.cumsum(sorted_probs)
                
                # Find cutoff index for top-p
                cutoff_idx = np.searchsorted(cumulative_probs, top_p)
                
                # Create a mask for the top-p tokens
                top_p_mask = np.zeros_like(probs)
                top_p_mask[sorted_indices[:cutoff_idx+1]] = 1
                
                # Apply the mask and renormalize
                masked_probs = probs * top_p_mask
                masked_probs = masked_probs / np.sum(masked_probs)
                
                # Sample from the masked distribution
                next_id = np.random.choice(len(masked_probs), p=masked_probs)
                return int(next_id)
            else:
                return int(np_logits.argmax())

        # Caches to store K/V from each layer
        cache_per_layer = []

        # Process the initial prompt:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        h = self.tok_embeddings(x)
        for layer in self.layers:
            h, c = layer(h, mask)
            cache_per_layer.append(c)
        h = self.norm(h)

        # Get 2D logits with shape (1, vocab_size)
        logits = self.output(h[:, -1])
        
        # Squeeze the batch dimension to get a 1D tensor of shape (vocab_size,)
        logits = logits[0]
        
        new_token = sample_logits(logits)
        yield new_token
        
        # Check if we should stop after the first token
        if eos_token_id is not None and new_token == eos_token_id:
            logger.info("Generation stopped after first token (EOS token generated)")
            return

        # Generate up to max_new_tokens:
        tokens_generated = 1
        # Add a timeout mechanism to prevent infinite loops
        start_time = time.time()
        max_generation_time = 60  # 60 seconds timeout
        
        while tokens_generated < max_new_tokens:
            # Check for timeout
            if time.time() - start_time > max_generation_time:
                logger.warning(f"Generation timed out after {max_generation_time} seconds")
                break
            
            tokens_generated += 1
            x = mx.array([[new_token]])  # shape (batch=1, seq=1)
            # Use cached keys/values
            h = self.tok_embeddings(x)
            for i, layer in enumerate(self.layers):
                h, updated_cache = layer(h, mask=None, cache=cache_per_layer[i])
                cache_per_layer[i] = updated_cache
            h = self.norm(h)
            logits = self.output(h[:, -1])
            logits = logits[0]
            new_token = sample_logits(logits)
            yield new_token
            
            # Stop if we hit the EOS token
            if eos_token_id is not None and new_token == eos_token_id:
                logger.info(f"Generation stopped at token {tokens_generated}: EOS token generated")
                break

class LlamaMLX:
    """
    - __init__ with max_new_tokens, temperature, top_p
    - load_model_and_tokenizer
    - download_model
    - generate_text
    - complete
    """

    def __init__(self, max_new_tokens: int, temperature: float = 0.7, top_p: float = 1.0, cache_converted_safetensors: bool = False):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Local directory to store model & tokenizer
        self.weights_dir = "app/model_weights/llama_3_1"
        self.cache_converted_safetensors = cache_converted_safetensors
        
        # Our merged config
        self.config = ModelConfig()

        logger.info("Initializing LlamaMLX.")
        self.load_model_and_tokenizer()
        logger.info("LlamaMLX initialized successfully.")

    def download_model(self):
        """
        Download model files from huggingface_hub if they are not present locally.
        """
        logger.info("Checking if model files exist locally...")
        config_path = os.path.join(self.weights_dir, "config.json")
        weights_path = os.path.join(self.weights_dir, "weights.npz")
        tokenizer_path = os.path.join(self.weights_dir, "tokenizer.json")

        # If any key file doesn't exist, pull them
        if not (os.path.exists(config_path) and
                os.path.exists(weights_path) and
                os.path.exists(tokenizer_path)):
            logger.info("Downloading model files from HuggingFace...")
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                raise ValueError(
                    "HUGGING_FACE_HUB_TOKEN not found in environment variables. "
                    "Set your token to download model from Hugging Face."
                )

            files = ["config.json", "tokenizer.json", "tokenizer_config.json", "weights.npz"]
            for file in files:
                hf_hub_download(
                    repo_id=self.model_id,
                    filename=file,
                    token=token,
                    cache_dir=self.weights_dir
                )
            logger.info(f"Downloaded model files to {self.weights_dir}")
        else:
            logger.info("All model files found locally. No download needed.")

    def load_model_and_tokenizer(self):
        """
        Load the tokenizer (Hugging Face) and the LLaMA model (MLX) from local disk.
        """

        # Check if model exists, if not, download it
        if not os.path.exists(os.path.join(self.weights_dir, "config.json")):
            self.download_model()

        # Step 2: Load the tokenizer
        logger.info("Loading tokenizer from local weights...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_dir)
        if not hasattr(self.tokenizer, "eos_token_id") or self.tokenizer.eos_token_id is None:
            # If the HF tokenizer is missing special tokens, set some default
            self.tokenizer.eos_token_id = 2

        # Step 3: Load the config
        config_path = os.path.join(self.weights_dir, "config.json")
        with open(config_path, "r") as f:
            config_json = json.load(f)

        # Merge what we read into self.config
        # We allow the config.json to override defaults in ModelConfig
        for field in vars(self.config):
            if field in config_json:
                setattr(self.config, field, config_json[field])

        # Step 4: Load the weights
        logger.info(f"Loading model weights from {self.weights_dir} ...")
        weights_dict = {}
        safetensors_files = glob.glob(os.path.join(self.weights_dir, "model-*.safetensors"))
        
        if not safetensors_files:
            raise FileNotFoundError(f"No model weights found in {self.weights_dir}. Expected weights.npz or model-*.safetensors files.")
        
        for file_path in safetensors_files:
            logger.info(f"Loading safetensors file: {file_path}")
            # Load the safetensors file using PyTorch
            tensors = safetensors.torch.load_file(file_path, device="cpu")
            for key, tensor in tensors.items():
                # Convert to float32 PyTorch tensor
                tensor = tensor.to(torch.float32)
                weights_dict[key] = tensor

        # Step 5: Instantiate the LLaMA model using our config and the converted weights
        logger.info("Instantiating LLaMA model.")
        self.model = Llama(self.config)
        # Load the converted weights
        for key, tensor in weights_dict.items():
            self.model.update({key: mx.array(tensor.numpy())})
        logger.info("Model loaded into MLX successfully.")

        if self.cache_converted_safetensors:
            # Save the converted weights to a new safetensors file
            converted_weights_path = os.path.join(self.weights_dir, "converted_weights.safetensors")
            metadata = {"format": "pt"}  # Metadata should be a dictionary
            safetensors.torch.save(weights_dict, converted_weights_path, metadata=metadata)

    def generate_text(self, prompt: str) -> mx.array:
        """
        Generate text using the LLaMA model's `generate` method (token-by-token).
        Returns the final token array (including prompt + generation).
        
        Parameters:
            prompt (str): The input prompt to generate from
            
        Returns:
            mx.array: Array of token IDs including prompt and generated tokens
        """
        logger.info(f"Generating text with prompt length: {len(prompt)}")
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        x = mx.array([prompt_ids], dtype=mx.int64)  # shape (1, seq_length)

        # We'll build up the sequence in a Python list as tokens come out
        generated_tokens = prompt_ids[:]  # start with the prompt

        # The model's generator yields tokens one by one
        token_iter = self.model.generate(
            x,
            temp=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Set a timeout for the entire generation process
        start_time = time.time()
        max_generation_time = 120  # 2 minutes timeout
        
        try:
            for new_tok in token_iter:
                generated_tokens.append(new_tok)
                
                # Check if we've been generating for too long
                if time.time() - start_time > max_generation_time:
                    logger.warning(f"Generation timed out after {max_generation_time} seconds")
                    break
                
                # If we hit EOS, we can stop
                if new_tok == self.tokenizer.eos_token_id:
                    logger.info("Generation complete: EOS token generated")
                    break
                
                # Print progress every 10 tokens
                if len(generated_tokens) % 10 == 0:
                    logger.debug(f"Generated {len(generated_tokens) - len(prompt_ids)} tokens so far")
        except Exception as e:
            logger.error(f"Error during token generation: {str(e)}")
            # Return what we have so far rather than failing completely
            logger.info(f"Returning partial generation of {len(generated_tokens)} tokens")
        
        logger.info(f"Generated tokens: {len(generated_tokens) - len(prompt_ids)} new, {len(generated_tokens)} total")
        return mx.array(generated_tokens)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Provide a standard "instruct" format completion. 
        Combines system_prompt and user_prompt into a single prompt, then calls `generate_text`.
        
        Parameters:
            system_prompt (str): The system instructions for the model
            user_prompt (str): The user's query or instruction
            
        Returns:
            str: The generated completion text
        """
        logger.info("Beginning completion call...")

        # Format the prompt according to Llama 3.1 chat template
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]
"""

        logger.info(
            f"Starting text generation with temperature={self.temperature}, "
            f"top_p={self.top_p}, max_new_tokens={self.max_new_tokens}"
        )

        try:
            # Generate
            output_tokens = self.generate_text(prompt)
            # Decode the output
            output_list = output_tokens.tolist()
            output_text = self.tokenizer.decode(output_list)
            
            # Extract just the assistant's response (remove the prompt)
            response_start = output_text.find("[/INST]") + len("[/INST]")
            if response_start >= 0:
                output_text = output_text[response_start:].strip()

            logger.info("Text generation completed successfully.")
            logger.info(f"Output: {output_text}")
            return output_text

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise
