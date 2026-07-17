import glob
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

# MLX imports
import mlx.core as mx
import mlx.nn as nn

# Tokenizer imports
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from agents.models.llama_completion import strings_to_message_dict


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KVCache:
    """Grow K/V storage in chunks and update it in place during decoding."""

    def __init__(self, step: int = 256):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        previous = self.offset
        required = previous + keys.shape[2]
        if self.keys is None or required > self.keys.shape[2]:
            batch, n_kv_heads, _, key_dim = keys.shape
            value_dim = values.shape[-1]
            growth = ((keys.shape[2] + self.step - 1) // self.step) * self.step
            key_shape = (batch, n_kv_heads, growth, key_dim)
            value_shape = (batch, n_kv_heads, growth, value_dim)
            new_keys = mx.zeros(key_shape, dtype=keys.dtype)
            new_values = mx.zeros(value_shape, dtype=values.dtype)
            if self.keys is None:
                self.keys, self.values = new_keys, new_values
            else:
                if previous % self.step:
                    self.keys = self.keys[..., :previous, :]
                    self.values = self.values[..., :previous, :]
                self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                self.values = mx.concatenate([self.values, new_values], axis=2)

        self.offset = required
        self.keys[..., previous : self.offset, :] = keys
        self.values[..., previous : self.offset, :] = values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )


def sample_logits(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample entirely on the MLX device, avoiding a per-token NumPy copy."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    if top_p < 1:
        probabilities = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(probabilities, axis=-1)
        sorted_probabilities = mx.take_along_axis(probabilities, sorted_indices, axis=-1)
        cumulative_probabilities = mx.cumsum(sorted_probabilities, axis=-1)
        kept_probabilities = mx.where(
            cumulative_probabilities > 1 - top_p,
            sorted_probabilities,
            0,
        )
        inverse_indices = mx.argsort(sorted_indices, axis=-1)
        probabilities = mx.take_along_axis(kept_probabilities, inverse_indices, axis=-1)
        logits = mx.log(probabilities)

    return mx.random.categorical(logits / temperature)


@dataclass
class ModelConfig:
    """
    ModelConfig.
    """

    # Matches the naming of the first script for correctness, with some defaults:
    dim: int = 4096  # was "n_embd" in second script
    n_layers: int = 32  # was "n_layer"
    n_heads: int = 32  # was "n_head"
    # Additional fields from the first code:
    head_dim: int = 128  # will often be dim // n_heads
    n_kv_heads: int = 32
    hidden_dim: int = 11008
    norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 10000
    rope_traditional: bool = True
    max_position_embeddings: int = 2048
    rope_scaling: dict | None = field(default=None)


class Llama3RoPE(nn.Module):
    """
    Llama 3 style RoPE scaling with low/high frequency factors.
    """

    def __init__(self, dim: int, base: float, rope_scaling: dict, max_position_embeddings: int):
        super().__init__()
        self.dim = dim
        self.base = base
        self.rope_scaling = rope_scaling or {}
        self.max_position_embeddings = max_position_embeddings
        factor = float(self.rope_scaling.get("factor", 1.0))
        low_freq_factor = float(self.rope_scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(self.rope_scaling.get("high_freq_factor", 1.0))
        orig_max_pos = float(
            self.rope_scaling.get("original_max_position_embeddings", self.max_position_embeddings)
        )

        frequencies = self.base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        wavelen = 2.0 * mx.pi * frequencies
        low_freq_wavelen = orig_max_pos / low_freq_factor
        high_freq_wavelen = orig_max_pos / high_freq_factor

        frequencies = mx.where(
            wavelen > low_freq_wavelen,
            frequencies * factor,
            frequencies,
        )
        medium = (wavelen > high_freq_wavelen) & (wavelen < low_freq_wavelen)
        smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_frequencies = frequencies / ((1.0 - smooth) / factor + smooth)
        self._frequencies = mx.where(medium, smooth_frequencies, frequencies)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._frequencies,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        # Per the LLama approach, scale = head_dim^-0.5
        self.scale = args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        # RoPE for positional embeddings
        if args.rope_scaling and args.rope_scaling.get("rope_type") == "llama3":
            self.rope = Llama3RoPE(
                args.head_dim,
                base=args.rope_theta,
                rope_scaling=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
        else:
            self.rope = nn.RoPE(
                args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
        batch, sequence_length, model_dim = x.shape

        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        # Reshape into (B, n_heads, L, head_dim) for queries,
        #            (B, n_kv_heads, L, head_dim) for keys/values
        queries = queries.reshape(
            batch, sequence_length, self.n_heads, self.args.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, sequence_length, self.n_kv_heads, self.args.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            batch, sequence_length, self.n_kv_heads, self.args.head_dim
        ).transpose(0, 2, 1, 3)

        # Handle caching
        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, sequence_length, model_dim)
        return self.wo(output), cache


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
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, KVCache | None]:
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
        """Yield generated token IDs using an in-place K/V cache."""
        if max_new_tokens <= 0:
            return

        if eos_token_id is None:
            eos_ids = None
        elif isinstance(eos_token_id, list | tuple | set):
            eos_ids = {int(token_id) for token_id in eos_token_id}
        else:
            eos_ids = {int(eos_token_id)}

        prompt_token_count = int(x.shape[1])
        cache_per_layer = [KVCache() for _ in self.layers]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        prompt_started = time.perf_counter()
        h = self.tok_embeddings(x)
        for index, layer in enumerate(self.layers):
            h, _ = layer(h, mask, cache=cache_per_layer[index])
        h = self.norm(h)
        logits = self.output(h[:, -1])[0]
        mx.eval(logits)
        prompt_elapsed = time.perf_counter() - prompt_started

        start_time = time.time()
        generation_started = time.perf_counter()
        max_generation_time = 120
        generated_count = 0
        for tokens_generated in range(1, max_new_tokens + 1):
            token = sample_logits(logits, temp, top_p)
            mx.eval(token)
            token_id = int(token.item())
            generated_count = tokens_generated
            generation_elapsed = time.perf_counter() - generation_started
            self.last_stats = {
                "prompt_tokens": prompt_token_count,
                "prompt_tps": prompt_token_count / prompt_elapsed,
                "generation_tokens": generated_count,
                "generation_tps": generated_count / generation_elapsed,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
            yield token_id

            if eos_ids is not None and token_id in eos_ids:
                break
            if time.time() - start_time > max_generation_time:
                logger.warning("Generation timed out after %s seconds", max_generation_time)
                break
            if tokens_generated == max_new_tokens:
                break

            h = self.tok_embeddings(mx.array([[token_id]], dtype=mx.int64))
            for index, layer in enumerate(self.layers):
                h, _ = layer(h, cache=cache_per_layer[index])
            h = self.norm(h)
            logits = self.output(h[:, -1])[0]


class LlamaMLX:
    """
    - __init__ with max_new_tokens, temperature, top_p
    - load_model_and_tokenizer
    - download_model
    - generate_text
    - complete
    """

    def __init__(
        self,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 1.0,
        cache_converted_safetensors: bool = False,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        weights_dir: str | None = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Local directory to store model & tokenizer
        self.weights_dir = weights_dir or "app/model_weights/llama_3_1"
        self.cache_converted_safetensors = cache_converted_safetensors

        # Our merged config
        self.config = ModelConfig()
        self.eos_token_ids: list[int] | None = None
        self._weights_dtype = mx.float16

        logger.info("Initializing LlamaMLX.")
        self.load_model_and_tokenizer()
        logger.info("LlamaMLX initialized successfully.")

    def _apply_hf_config(self, config_json: dict):
        """
        Apply HF config.json fields to our ModelConfig, with Llama 3.1-specific keys.
        """
        # Map HF config keys to ModelConfig fields.
        hf_to_internal = {
            "hidden_size": "dim",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "n_heads",
            "num_key_value_heads": "n_kv_heads",
            "intermediate_size": "hidden_dim",
            "rms_norm_eps": "norm_eps",
            "vocab_size": "vocab_size",
            "rope_theta": "rope_theta",
            "max_position_embeddings": "max_position_embeddings",
        }
        for hf_key, internal_key in hf_to_internal.items():
            if hf_key in config_json:
                setattr(self.config, internal_key, config_json[hf_key])

        # Ensure head_dim stays consistent with dim/n_heads.
        if self.config.n_heads:
            self.config.head_dim = self.config.dim // self.config.n_heads

        # Some Llama 3 configs omit num_key_value_heads; default to n_heads.
        if not self.config.n_kv_heads:
            self.config.n_kv_heads = self.config.n_heads

        # Llama 3 uses non-traditional RoPE. We don't fully implement rope_scaling,
        # but we can at least switch off "traditional" RoPE when rope_type is llama3.
        rope_scaling = config_json.get("rope_scaling") or {}
        if rope_scaling:
            self.config.rope_scaling = rope_scaling
            if rope_scaling.get("rope_type") == "llama3":
                self.config.rope_traditional = False

        # Track EOS token IDs from config for stopping conditions.
        eos_ids = config_json.get("eos_token_id")
        if isinstance(eos_ids, list):
            self.eos_token_ids = [int(e) for e in eos_ids]
        elif eos_ids is not None:
            self.eos_token_ids = [int(eos_ids)]

        # Choose a sensible default weights dtype based on config.
        torch_dtype = str(config_json.get("torch_dtype", "")).lower()
        if "bfloat16" in torch_dtype and hasattr(mx, "bfloat16"):
            self._weights_dtype = mx.bfloat16
        else:
            self._weights_dtype = mx.float16

    def _map_hf_to_mlx_key(self, key: str) -> str | None:
        """
        Map HuggingFace Llama parameter names to MLX module names.
        """
        if key == "model.embed_tokens.weight":
            return "tok_embeddings.weight"
        if key == "model.norm.weight":
            return "norm.weight"
        if key == "lm_head.weight":
            return "output.weight"

        layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", key)
        if not layer_match:
            return None

        layer_idx, rest = layer_match.groups()
        prefix = f"layers.{layer_idx}."

        if rest == "input_layernorm.weight":
            return prefix + "attention_norm.weight"
        if rest == "post_attention_layernorm.weight":
            return prefix + "ffn_norm.weight"
        if rest == "self_attn.q_proj.weight":
            return prefix + "attention.wq.weight"
        if rest == "self_attn.k_proj.weight":
            return prefix + "attention.wk.weight"
        if rest == "self_attn.v_proj.weight":
            return prefix + "attention.wv.weight"
        if rest == "self_attn.o_proj.weight":
            return prefix + "attention.wo.weight"
        if rest == "mlp.gate_proj.weight":
            return prefix + "feed_forward.w1.weight"
        if rest == "mlp.down_proj.weight":
            return prefix + "feed_forward.w2.weight"
        if rest == "mlp.up_proj.weight":
            return prefix + "feed_forward.w3.weight"

        return None

    def _build_instruct_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Build a Llama 3.1 instruct/chat prompt string.
        """
        # Llama 3.1 chat template with explicit headers and <|eot_id|> delimiters.
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _build_messages_prompt(self, messages: list[dict[str, str | None]]) -> str:
        normalized = [
            {"role": message["role"], "content": message.get("content") or ""}
            for message in messages
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt = "<|begin_of_text|>"
        for message in normalized:
            prompt += (
                f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                f"{message['content']}<|eot_id|>"
            )
        return f"{prompt}<|start_header_id|>assistant<|end_header_id|>\n\n"

    def download_model(self):
        """
        Download model files from huggingface_hub if they are not present locally.
        """
        logger.info("Checking if model files exist locally...")
        config_path = os.path.join(self.weights_dir, "config.json")
        tokenizer_path = os.path.join(self.weights_dir, "tokenizer.json")
        weights = glob.glob(os.path.join(self.weights_dir, "model*.safetensors"))

        # If any key file doesn't exist, pull them
        if not (os.path.exists(config_path) and weights and os.path.exists(tokenizer_path)):
            logger.info("Downloading model files from HuggingFace...")
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                raise ValueError(
                    "HUGGING_FACE_HUB_TOKEN not found in environment variables. "
                    "Set your token to download model from Hugging Face."
                )

            snapshot_download(
                repo_id=self.model_id,
                token=token,
                local_dir=self.weights_dir,
                allow_patterns=["*.json", "*.model", "*.safetensors"],
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
        with open(config_path) as f:
            config_json = json.load(f)

        # Merge HF config into our ModelConfig with Llama-specific keys.
        self._apply_hf_config(config_json)
        if self.eos_token_ids:
            # Tokenizer expects a single eos_token_id; use the primary one for decode.
            self.tokenizer.eos_token_id = self.eos_token_ids[0]
        elif self.tokenizer.eos_token_id is not None:
            # Fall back to tokenizer eos if config didn't provide a list.
            self.eos_token_ids = [int(self.tokenizer.eos_token_id)]

        # Step 4: Load the weights
        logger.info(f"Loading model weights from {self.weights_dir} ...")
        safetensors_files = sorted(glob.glob(os.path.join(self.weights_dir, "model*.safetensors")))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No model weights found in {self.weights_dir}. "
                "Expected model*.safetensors files."
            )

        unmapped_keys = 0
        loaded_keys = set()
        cached_weights: dict[str, mx.array] = {}
        logger.info("Instantiating LLaMA model.")
        self.model = Llama(self.config)
        for file_path in safetensors_files:
            logger.info(f"Loading safetensors file: {file_path}")
            tensors = mx.load(file_path)
            mapped_weights: dict[str, mx.array] = {}
            for key, tensor in tensors.items():
                # Convert HF key names to MLX parameter names.
                mlx_key = self._map_hf_to_mlx_key(key)
                if not mlx_key:
                    unmapped_keys += 1
                    continue

                mapped_weights[mlx_key] = tensor.astype(self._weights_dtype)

            self.model.load_weights(list(mapped_weights.items()), strict=False)
            loaded_keys.update(mapped_weights)
            if self.cache_converted_safetensors:
                cached_weights.update(mapped_weights)

        expected_keys = {
            key
            for key, _ in tree_flatten(self.model.parameters())
            if not key.endswith(".rope._frequencies")
        }
        missing_keys = expected_keys - loaded_keys
        if missing_keys:
            missing = ", ".join(sorted(missing_keys)[:10])
            raise ValueError(f"Missing {len(missing_keys)} model weights: {missing}")
        mx.eval(self.model.parameters())
        self.model.eval()
        if unmapped_keys:
            logger.warning(f"Unmapped HF weights encountered: {unmapped_keys}")
        logger.info("Model loaded into MLX successfully.")

        if self.cache_converted_safetensors:
            # Save the converted weights (MLX-named) to a new safetensors file for reuse.
            converted_weights_path = os.path.join(self.weights_dir, "converted_weights.safetensors")
            mx.save_safetensors(
                converted_weights_path,
                cached_weights,
                metadata={"format": "mlx"},
            )

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
        self._last_prompt_token_count = len(prompt_ids)
        x = mx.array([prompt_ids], dtype=mx.int64)  # shape (1, seq_length)

        # We'll build up the sequence in a Python list as tokens come out
        generated_tokens = prompt_ids[:]  # start with the prompt

        # The model's generator yields tokens one by one
        token_iter = self.model.generate(
            x,
            temp=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_ids or self.tokenizer.eos_token_id,
        )

        for new_tok in token_iter:
            generated_tokens.append(new_tok)

            eos_ids = self.eos_token_ids or [self.tokenizer.eos_token_id]
            if new_tok in eos_ids:
                logger.info("Generation complete: EOS token generated")
                break

            if len(generated_tokens) % 10 == 0:
                logger.debug(
                    "Generated %s tokens so far",
                    len(generated_tokens) - len(prompt_ids),
                )

        logger.info(
            f"Generated tokens: {len(generated_tokens) - len(prompt_ids)} new, {len(generated_tokens)} total"
        )
        return mx.array(generated_tokens)

    def complete(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        """
        Provide a standard "instruct" format completion.
        Combines system_prompt and user_prompt into a single prompt, then calls `generate_text`.

        Parameters:
            system_prompt (str): The system instructions for the model
            user_prompt (str): The user's query or instruction

        Returns:
            str: The generated completion text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.complete_messages(messages)

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
    ) -> list[dict[str, str]]:
        """Complete a structured conversation using native chat-template roles."""
        logger.info("Beginning completion call...")
        prompt = self._build_messages_prompt(messages)
        user_prompt = next(
            (
                message.get("content") or ""
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

        logger.info(
            f"Starting text generation with temperature={self.temperature}, "
            f"top_p={self.top_p}, max_new_tokens={self.max_new_tokens}"
        )

        try:
            # Generate
            output_tokens = self.generate_text(prompt)
            # Decode only newly generated IDs so prompt text cannot leak into the response.
            output_list = output_tokens.tolist()[self._last_prompt_token_count :]
            output_text = self.tokenizer.decode(
                output_list,
                skip_special_tokens=True,
            ).strip()
            self.last_stats = dict(getattr(self.model, "last_stats", {}))
            self.last_stats.update(
                {
                    "prompt_tokens": self._last_prompt_token_count,
                    "generation_tokens": len(output_list),
                }
            )

            logger.info("Text generation completed successfully.")
            logger.info(f"Output: {output_text}")
            messages = strings_to_message_dict(user_prompt, output_text)
            return messages

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise
