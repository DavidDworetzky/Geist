# Plan: MLX Llama unit tests

## Goals
- Add unit tests covering MLX Llama tensor shapes and helper behaviors.
- Avoid loading large HF weights; keep tests fast and deterministic.

## Steps
1. Add tests for Attention output and KV cache shapes using a tiny ModelConfig.
2. Add tests for config mapping, prompt formatting, and HF→MLX key mapping.
3. Add a lightweight generation test verifying EOS stop behavior and output shape.

## Notes
- Skip tests if `mlx` is unavailable in the environment.
- Avoid calling LlamaMLX.__init__ to prevent loading real weights.
