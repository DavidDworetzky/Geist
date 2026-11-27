# Inkling 2 Mini MLX Local Inference and Serving Speedups

## Goal

Add text-only local MLX inference for `thinkingmachines/Inkling-2-Mini` on Apple Silicon using Geist's existing pinned `mlx==0.32.2` and `mlx-lm==0.31.3` dependencies, validate live inference against the official Hugging Face checkpoint, and benchmark/implement concrete local serving speedups.

## What Already Exists

- `agents/model_catalog.py` and `agents/architectures/registry.py` expose local models and route Apple Silicon models to `mlx_llama`.
- `app/services/local_models.py` manages immutable pinned Hugging Face snapshots (`LocalModelArtifact`) with progress tracking and atomic install.
- `agents/architectures/mlx_llama_runner.py` and `agents/architectures/llama/mlx_lm_backend.py` load and stream non-Llama MLX models through `mlx-lm`.
- `mlx==0.32.2` natively supports `nvfp4`, `mxfp8`, and `affine` `mx.gather_qmm`, `mx.fast.metal_kernel`, and `mx.fast.scaled_dot_product_attention(..., sinks=...)`.

## What Needs to Be Added

1. **In-repo MLX Inkling 2 Mini architecture (`agents/architectures/inkling2/`)**:
   - `config.py`: `TextArgs`, `VisionArgs`, and `ModelArgs` for `inkling2_loop` / `inkling2`.
   - `moe.py`: `SwitchLinear` and `QuantizedSwitchLinear` supporting NVFP4 per-expert `global_scale`, MXFP8, and 4-bit affine experts on `mlx==0.32.2`.
   - `kernels.py`: Fused Metal kernels for `depth_attention`, `relative_attention`, and `expert_routing`, with pure-MLX fallbacks.
   - `layers.py`: Looped decoder layers, sliding/full attention with relative position logits and learned sinks, static/dynamic short convolutions, softcapped SwiGLU MoE, block attention residuals, and cross-pass message gates.
   - `checkpoint.py`: Weight renaming and NVFP4/MXFP8 quantization wiring for both `thinkingmachines/Inkling-2-Mini` and `mlx-community/Inkling-2-Mini-4bit`, skipping unused vision weights in text-only mode.
   - `model.py`: `TextModel`, `Inkling2Model`, and `Model` implementing `make_cache()` for 48 logical layer visits.
2. **Catalog and Curated Artifacts**:
   - Register `thinkingmachines/Inkling-2-Mini` in `agents/model_catalog.py` (1M context, 26.4B total / 3.4B active, function calling, reasoning, `mlx_llama` backend).
   - Pin immutable artifacts in `app/services/local_models.py` for `thinkingmachines/Inkling-2-Mini` (`ed40e7da625f594fc9c4990f5f66be24873c22d7`, native NVFP4/MXFP8 + `mtp.safetensors`) and `mlx-community/Inkling-2-Mini-4bit` (`9856dc53edead71405f8348930937c3638ff21a2`).
3. **MLX Backend and Response Parsing**:
   - Hook `agents/architectures/llama/mlx_lm_backend.py` into `mlx_lm.utils.load_model(..., get_model_classes=...)` for `inkling2_loop` / `inkling2` checkpoints.
   - Default `reasoning_effort="none"` for Inkling 2 chat templates (overridable via `chat_template_kwargs`), strip Inkling message/thinking control markup cleanly, and parse `<|content_invoke_tool_json|>` tool calls in `agents/architectures/chat_template_tools.py`.
4. **Local Serving Speedups**:
   - Benchmark and optimize prefill/decode latency on Apple Silicon: fused Metal kernels, compiled/fast single-token decode paths, multi-turn prompt/KV prefix caching in `MLXLMBackend`, and DSpark (`mtp.safetensors`) speculative decoding exploration.
5. **Tests and Validation**:
   - Add focused unit tests in `tests/agents/test_inkling2.py` covering catalog metadata, artifacts, routing, checkpoint sanitization, tiny synthetic forward pass + cache, and response/tool parsing.
   - Run live MLX inference and speedup benchmarks on `thinkingmachines/Inkling-2-Mini`.

## Non-goals

- Multimodal image input support (Geist's local chat path remains text-only).
- Upgrading pinned `mlx` or `mlx-lm` packages in `pyproject.toml`.
- Changing Geist's default local model.
