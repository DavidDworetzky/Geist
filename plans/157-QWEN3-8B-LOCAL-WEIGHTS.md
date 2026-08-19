# Qwen3-8B local weights

## Goal

Confirm that Geist's generic Hugging Face local-model path supports `Qwen/Qwen3-8B`, and capture the workflow as a reusable repo skill.

## Plan

1. Make the model's minimum Transformers version explicit in the catalog.
2. Keep online-only settings out of local-agent construction and allow `mlx_lm` to fall back to Hugging Face when no managed artifact exists.
3. Use an MLX generation stream that remains valid when streaming responses resume on another worker thread.
4. Add focused checks for Qwen3 architecture recognition, cross-platform routing, and the model-specific weight directory.
5. Add a repo skill covering catalog, runner, weight, test, and runtime validation work for future Hugging Face models.

## Non-goals

- Add a model-specific runner when the generic runner already supports Qwen3.
- Download multi-gigabyte weights as part of the code change.
- Change Geist's existing local-model defaults.

## Acceptance criteria

- `Qwen/Qwen3-8B` remains selectable as a local Transformers model.
- The pinned Transformers version recognizes the `qwen3` architecture.
- Downloads resolve to `app/model_weights/Qwen_Qwen3-8B`.
- Settings-driven native MLX inference can fall back to Hugging Face and stream a completion.
- The repo skill validates successfully.
