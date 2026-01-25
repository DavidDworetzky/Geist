"""Unit tests for MLX Llama shapes and helper behaviors."""
import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx

from agents.architectures.llama.llama_mlx import Attention, Llama, LlamaMLX, ModelConfig


def _make_llama_mlx_stub():
    # Avoid __init__ to skip loading real weights.
    obj = LlamaMLX.__new__(LlamaMLX)
    obj.config = ModelConfig()
    obj.eos_token_ids = None
    obj._weights_dtype = mx.float16
    return obj


def test_attention_cache_shapes():
    cfg = ModelConfig(
        dim=16,
        n_layers=1,
        n_heads=4,
        head_dim=4,
        n_kv_heads=2,
        hidden_dim=32,
        vocab_size=64,
    )
    attn = Attention(cfg)
    x = mx.zeros((1, 3, 16))

    out, cache = attn(x, mask=None, cache=None)
    assert out.shape == (1, 3, 16)

    k, v = cache
    assert k.shape == (1, 2, 3, 4)
    assert v.shape == (1, 2, 3, 4)

    x2 = mx.zeros((1, 1, 16))
    out2, cache2 = attn(x2, mask=None, cache=cache)
    assert out2.shape == (1, 1, 16)

    k2, v2 = cache2
    assert k2.shape == (1, 2, 4, 4)
    assert v2.shape == (1, 2, 4, 4)


def test_apply_hf_config_mapping():
    obj = _make_llama_mlx_stub()
    config_json = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "rope_theta": 500000.0,
        "rope_scaling": {"rope_type": "llama3"},
        "eos_token_id": [1, 2, 3],
        "torch_dtype": "bfloat16",
    }

    obj._apply_hf_config(config_json)

    assert obj.config.dim == 64
    assert obj.config.n_layers == 2
    assert obj.config.n_heads == 4
    assert obj.config.n_kv_heads == 2
    assert obj.config.hidden_dim == 128
    assert obj.config.norm_eps == 1e-6
    assert obj.config.vocab_size == 256
    assert obj.config.rope_theta == 500000.0
    assert obj.config.head_dim == 16
    assert obj.config.rope_traditional is False
    assert obj.eos_token_ids == [1, 2, 3]


def test_map_hf_to_mlx_key():
    obj = _make_llama_mlx_stub()
    assert obj._map_hf_to_mlx_key("model.embed_tokens.weight") == "tok_embeddings.weight"
    assert obj._map_hf_to_mlx_key("model.norm.weight") == "norm.weight"
    assert obj._map_hf_to_mlx_key("lm_head.weight") == "output.weight"
    assert obj._map_hf_to_mlx_key("model.layers.0.self_attn.q_proj.weight") == "layers.0.attention.wq.weight"
    assert obj._map_hf_to_mlx_key("model.layers.1.mlp.down_proj.weight") == "layers.1.feed_forward.w2.weight"
    assert obj._map_hf_to_mlx_key("model.layers.3.post_attention_layernorm.weight") == "layers.3.ffn_norm.weight"
    assert obj._map_hf_to_mlx_key("model.layers.3.input_layernorm.weight") == "layers.3.attention_norm.weight"
    assert obj._map_hf_to_mlx_key("model.layers.3.unknown.weight") is None


def test_build_instruct_prompt():
    obj = _make_llama_mlx_stub()
    prompt = obj._build_instruct_prompt("sys", "user")

    assert prompt.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
    assert "sys" in prompt
    assert "user" in prompt


def test_generate_stops_on_eos():
    cfg = ModelConfig(
        dim=8,
        n_layers=1,
        n_heads=2,
        head_dim=4,
        n_kv_heads=1,
        hidden_dim=16,
        vocab_size=16,
    )
    model = Llama(cfg)

    # Force logits to all zeros so argmax returns 0 (EOS).
    model.tok_embeddings.weight = mx.zeros((cfg.vocab_size, cfg.dim))
    model.output.weight = mx.zeros((cfg.vocab_size, cfg.dim))

    x = mx.array([[1, 2, 3]], dtype=mx.int64)
    tokens = list(model.generate(x, temp=0.0, top_p=1.0, max_new_tokens=5, eos_token_id=0))

    assert tokens == [0]
