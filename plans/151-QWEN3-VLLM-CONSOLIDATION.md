# 151 - Consolidate Qwen3 runner with vLLM runner

## Goal
Unify Qwen3 and vLLM runner behavior so we maintain one inference implementation path while preserving existing `qwen3` runner compatibility.

## Plan
1. Replace the placeholder `VLLMRunner` implementation with the current transformer-based logic used by `Qwen3Runner`.
2. Convert `Qwen3Runner` into a thin compatibility subclass of `VLLMRunner` so existing runner registrations and imports continue to work.
3. Add/adjust tests to verify that `Qwen3Runner` now delegates to the shared `VLLMRunner` implementation.
4. Run targeted backend tests for the runner consolidation.
