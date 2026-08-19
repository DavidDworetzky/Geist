---
name: add-huggingface-local-model
description: This skill should be used when adding or updating a Hugging Face causal language model for Geist local inference, including catalog exposure, runner routing, weight download or lookup, settings selection, dependency compatibility, focused tests, and runtime validation.
---

# Add a Hugging Face Local Model

Add the smallest complete integration that makes a Hugging Face model selectable and runnable through Geist's local-agent path. Reuse an existing runner whenever the model architecture is already supported.

## 1. Inspect the Current Path

Read `AGENTS.md` and create the required feature plan under `plans/` before implementation.

Trace the model through these existing surfaces:

- `agents/model_catalog.py` and `agents/architectures/registry.py` for model metadata and runner registration.
- `agents/factory.py` for model-to-runner selection.
- `agents/architectures/` for the compatible runner and its default weight path.
- `app/main.py` and `app/models/user_settings.py` for settings-driven local-agent construction.
- `app/api/v1/endpoints/models.py` and `client/geist/src/Hooks/useAvailableModels.tsx` for backend discovery and the frontend fallback catalog.
- `scripts/download_models.py` and `scripts/copy_weights.py` for explicit weight placement.

Inspect the live diff first and preserve unrelated worktree changes, especially when any of these files are already modified.

## 2. Verify the Hugging Face Contract

Use the official model card and config to confirm:

- Exact repository ID and instruct/base variant.
- Architecture or `model_type`.
- Native context length and optional scaling behavior.
- Minimum `transformers`, `huggingface-hub`, tokenizer, or runtime version.
- Whether custom code, authentication, or license acceptance is required.
- Weight format and expected memory footprint.

Do not infer compatibility from the model name alone. Check that the pinned environment recognizes the model architecture without downloading weights, for example with `AutoConfig.for_model(<model_type>)` when supported.

Follow the repository dependency policy. Request approval before package changes, use `uv add PACKAGE==VERSION` with exact pins, update `pyproject.toml` and `uv.lock` together, and run a Python dependency audit.

## 3. Choose the Smallest Runner Change

Reuse a registered runner when it already supports the architecture's `AutoModelForCausalLM`, tokenizer, chat template, and generation contract.

Add only the model ID, metadata, and model-to-runner mapping when reuse is sufficient. Add a new runner only when the model requires materially different loading, prompt rendering, multimodal inputs, or generation behavior.

Keep one source of truth for runner inference. Ensure settings-derived and application local-agent construction both reach the same factory inference path. Preserve explicit `runner_type` overrides.

## 4. Align Weight Placement

Match explicit downloads or copies to the runner's lookup convention. The Hugging Face download path derives:

```text
app/model_weights/<hugging-face-id-with-slashes-replaced-by-underscores>
```

For example, `Qwen/Qwen3-8B` maps to `app/model_weights/Qwen_Qwen3-8B`.

Prefer loading from an existing local pretrained directory when present. Allow the runner's established Hugging Face fallback when local files are absent. Never download multi-gigabyte weights merely to validate catalog or routing changes unless the user explicitly requests a live inference run.

## 5. Complete Discovery and Settings Wiring

Add the model to the offline catalog with accurate capabilities. Update the frontend static fallback only when the model must remain selectable while the model API is unavailable.

Verify that saving the local model, previewing the agent config, and creating the cached local agent all choose the intended runner. Keep existing defaults unchanged unless the request explicitly changes them.

## 6. Test in Layers

Add focused regression tests for:

- Catalog identity and metadata.
- Model-to-runner inference, including case normalization when applicable.
- Settings-derived runner selection.
- Application local-agent construction.
- Local-directory and Hugging Face loading paths without downloading weights.
- Dependency-level architecture recognition.

Run the focused inference and service tests in Docker when available. Then follow `.agents/skills/geist-test-loop/SKILL.md` for Docker startup, logs, `curl`, browser smoke checks, and native `make run MLX_BACKEND=1` validation when the change touches native/local model behavior.

Report separately whether validation proved configuration/routing, model loading, and actual token generation. State the exact blocker when weights, memory, credentials, Docker, or MLX prevent a live generation test.
