# Qwen3-8B full MLX artifact

## Goal

Make the existing full `Qwen/Qwen3-8B` Hugging Face snapshot installable and
selectable through Geist's managed MLX model path on Apple Silicon.

## Plan

1. Add a curated MLX snapshot artifact pinned to the immutable full-model revision.
2. Preserve the canonical `Qwen/Qwen3-8B` catalog identity and existing runner routing.
3. Copy the already-cached snapshot into Geist's managed artifact directory with a
   matching completion manifest.
4. Validate through the live PR #327 MLX streaming run rather than a separate test pass.

## Non-goals

- Change the default local model.
- Add or download another quantization.
- Change the shared streaming contract or runner architecture.

## Acceptance criteria

- The Models API exposes a supported `qwen3-8b-full-mlx` artifact on macOS ARM64.
- The existing cached snapshot is recognized as installed without another download.
- Settings-driven Qwen3-8B chat uses the managed MLX artifact and streams in PR #327.
