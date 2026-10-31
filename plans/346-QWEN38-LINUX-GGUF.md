# Qwen 3.8 27B Linux GGUF Support

## Goal

Make Qwen 3.8 27B downloadable and selectable on Linux through Geist's existing managed GGUF and private `llama-server` path, while preserving the current MLX artifact and Apple-silicon default.

Ship the Linux x86_64 CPU inference runtime with the Docker distribution so a
supported, installed GGUF is actually runnable without operator configuration.

## Implementation

1. Add a pinned, checksummed Q4_K_M GGUF artifact for `Qwen/Qwen3.8-27B` using an immutable upstream revision.
2. Make model-ID artifact lookup prefer an installed artifact supported by the current platform, then any supported artifact, so MLX and GGUF variants can coexist safely.
3. Keep artifact-ID selection authoritative so settings-derived agent creation resolves `llama_server` on Linux and `mlx_llama` on Apple silicon without a new runner.
4. Add focused catalogue, platform-resolution, settings, and runner-routing tests without downloading model weights.
5. Pin the official llama.cpp Linux x64 CPU release archive and SHA-256 in the
   Docker build. Preserve the complete release directory because `llama-server`
   dynamically loads the bundled ggml and llama libraries.
6. Resolve a packaged runtime root by default while keeping
   `GEIST_LLAMA_SERVER_PATH` and `GEIST_LLAMA_RUNTIME_ROOT` as explicit operator
   overrides.
7. Report managed GGUF artifacts as supported only when both the host platform
   and a runnable llama.cpp executable are available.
8. Let an artifact's concrete backend select the runner automatically; retain
   `GEIST_LOCAL_RUNNER` as the fallback for models without a selected artifact.

## Validation

- Run focused local-model manager, factory, settings, and llama-server tests in Docker.
- Confirm the models API exposes the Linux GGUF as supported and the MLX artifact as unavailable on Linux.
- Run frontend tests/build and browser smoke for the Local Models inventory.
- Build the production Docker image and verify the packaged `llama-server`
  version and runtime status.
- Start the installed Qwen3 4B GGUF through the public runtime API, wait for
  readiness, and request a real short completion through Geist's chat API.
- Do not claim Qwen 3.8 27B token generation unless its 19 GB artifact is fully
  installed; Qwen3 4B is the required real-inference smoke model for the runtime
  contract.
