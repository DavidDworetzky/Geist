# Qwen3.8-27B local weights

## Goal

Add text-only local inference for the official `Qwen/Qwen3.8-27B` model on Apple Silicon, using the pinned `mlx-community/Qwen3.8-27B-4bit` snapshot, and keep the workflow documented as a reusable repo skill.

## Plan

1. Add accurate Qwen3.8-27B catalog metadata and a curated, immutable 4-bit MLX artifact.
2. Upgrade the pinned Transformers, Hugging Face Hub, safetensors, and mlx-lm versions required for the `qwen3_5` architecture.
3. Route Qwen3.8-27B through `mlx_lm` by default while preserving explicit implementation overrides.
4. Keep online-only settings out of local-agent construction and retain the thread-safe MLX generation stream.
5. Add focused checks for catalog identity, artifact lookup, settings routing, dependency compatibility, and local generation.
6. Keep the repo skill aligned with curated Hugging Face/MLX artifacts and the full validation loop.

## Non-goals

- Add multimodal image or video inputs; Geist's current local chat contract remains text-only.
- Load the roughly 56 GB official BF16 checkpoint when the 4-bit MLX snapshot is the practical local artifact.
- Change Geist's existing local-model defaults.

## Acceptance criteria

- `Qwen/Qwen3.8-27B` is selectable with 262,144-token context metadata.
- The pinned runtime recognizes the `qwen3_5` architecture.
- The Models page exposes the pinned `mlx-community/Qwen3.8-27B-4bit` snapshot for Apple Silicon.
- Settings-driven native MLX inference loads the managed snapshot and generates text.
- The repo skill validates successfully.
