# Qwen 3.8 27B Linux GGUF Support

## Goal

Make Qwen 3.8 27B downloadable and selectable on Linux through Geist's existing managed GGUF and private `llama-server` path, while preserving the current MLX artifact and Apple-silicon default.

## Implementation

1. Add a pinned, checksummed Q4_K_M GGUF artifact for `Qwen/Qwen3.8-27B` using an immutable upstream revision.
2. Make model-ID artifact lookup prefer an installed artifact supported by the current platform, then any supported artifact, so MLX and GGUF variants can coexist safely.
3. Keep artifact-ID selection authoritative so settings-derived agent creation resolves `llama_server` on Linux and `mlx_llama` on Apple silicon without a new runner.
4. Add focused catalogue, platform-resolution, settings, and runner-routing tests without downloading model weights.

## Validation

- Run focused local-model manager, factory, settings, and llama-server tests in Docker.
- Confirm the models API exposes the Linux GGUF as supported and the MLX artifact as unavailable on Linux.
- Run frontend tests/build and browser smoke for the Local Models inventory.
- Do not claim model loading or token generation unless the 19 GB artifact and a compatible staged `llama-server` are available.
