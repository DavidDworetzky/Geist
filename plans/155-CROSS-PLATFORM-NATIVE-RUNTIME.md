# Cross-Platform Native Runtime

## Goal

Run Geist natively on Windows x64, macOS ARM64, and Linux x64 without requiring
Docker or WSL. Geist supports a managed native sidecar contract for desktop
hosts, while Docker remains an optional deployment path.

## Supported inference backends

- Windows x64: bundled `llama-server`, with CPU fallback and Vulkan acceleration.
- macOS ARM64: MLX remains the primary and default local backend.
- Linux x64: bundled `llama-server`, with CPU fallback and Vulkan acceleration.

Geist owns model acquisition on every platform. MLX snapshots and GGUF files use
the same managed model catalog, download jobs, progress state, verification,
storage, selection, and removal flow. `llama-server` receives only a verified
local GGUF path and never downloads model weights itself.

## Work streams

1. Split the portable server dependencies from MLX, Transformers, and voice-only
   dependencies. Resolve the locked environment for Windows, macOS, and Linux,
   and lazily import optional features.
2. Add an installable `geist` CLI with loopback-only managed serving, dynamic
   ports, machine-readable readiness, stdin-EOF shutdown, health endpoints,
   migrations, deterministic application-data paths, and compiled SPA serving.
3. Extract the current MLX first-load download into a shared model manager. Add
   curated GGUF artifacts, resumable verified downloads, local GGUF import, API
   installation state, and Models-page controls.
4. Add one application-scoped `llama-server` manager and a runner that reuses
   Geist's OpenAI-compatible streaming and tool-call contracts. Keep one active
   model process initially and supervise it with retained process handles.
5. Replace platform-specific backend/frontend launchers with one native Geist
   sidecar contract. Package target-native Geist and inference runtimes as
   external resources, but keep downloaded models in per-user data so upgrades
   preserve them.
6. Add Windows, macOS, and Linux CI coverage, real CPU inference smoke tests,
   hardware Vulkan release gates, macOS MLX download/inference tests, and clean
   installer tests with no development prerequisites installed.

## Compatibility

- Existing macOS MLX selections remain MLX selections and keep working.
- Existing settings are not silently mapped to a different GGUF model.
- Windows users with an unavailable MLX selection receive setup guidance to
  download or import a GGUF model.
- Attach mode and optional Docker deployment remain supported.
- Host-provided legacy database/model locations are imported explicitly and
  never deleted.

## Completion gates

- Native source startup succeeds on all three target platforms without Docker
  or WSL.
- A clean Windows x64 Geist package launches without Python, Node, uv, Git,
  Docker, WSL, or compilers.
- The installed app can download or import a GGUF, run CPU inference, use Vulkan
  on supported hardware, restart without re-downloading, and exit without
  leaving Geist or `llama-server` processes behind.
- The macOS ARM64 package downloads and runs an MLX model through the same model
  management UI.
