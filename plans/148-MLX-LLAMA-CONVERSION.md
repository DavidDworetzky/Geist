# Plan: MLX Llama 3.1 weight conversion + instruct behavior

## Goals
- Convert HF Llama 3.1 safetensors to MLX parameter names/shapes and load reliably.
- Align config fields with HF config.json.
- Match Llama 3.1 instruct/chat template behavior.
- Reduce obvious MLX performance pitfalls in sampling/caching.

## Steps
1. Inspect current MLX Llama loader, config mapping, and generation path; identify mismatches vs HF naming.
2. Implement HF->MLX weight mapping + minimal reshaping/transposing where required.
3. Fix config merge to read HF keys and update ModelConfig accordingly.
4. Update prompt formatting to Llama 3.1 instruct template and EOS handling.
5. Reduce per-token CPU sync in sampling and limit KV cache expansion.
6. Smoke-check loader path and generation for obvious errors.

## Notes
- Keep changes localized to `agents/architectures/llama/llama_mlx.py`.
- Add succinct comments for conversion steps and template formatting.
