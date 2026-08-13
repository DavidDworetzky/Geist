# Switchable llama.cpp Compute Backend

## Goal

Expose CPU/GPU selection for managed llama.cpp runtimes on Windows and Linux,
including explicit selection of one or more Vulkan devices. Keep Apple-silicon
MLX behavior unchanged.

## Behavior

- New settings start with no resolved backend. The first successful llama.cpp
  model load inventories Vulkan devices, prefers the suitable device with the
  most free memory, and persists GPU or CPU.
- Integrated GPUs are not selected automatically, but remain available as
  warned manual choices.
- Automatic GPU startup falls back to CPU. An explicit GPU selection fails
  visibly if any selected device is unavailable or llama-server cannot start.
- Multiple explicit GPUs are passed to llama.cpp as a comma-separated
  `--device` list, using llama.cpp's default layer split.
- The Settings control is hidden when managed llama.cpp is unavailable, notably
  on Apple-silicon MLX installations. Environment and explicit-binary overrides
  remain authoritative and lock the user-facing control.
- A full settings reset clears the detected backend and device selection so the
  next llama.cpp use detects again.
- Settings presents the current hardware recommendation directly as CPU or GPU;
  the internal unresolved state is not exposed as pending or unverified UI.

## Persistence and API

- `user_settings.llama_backend`: nullable `cpu` or `gpu`; null means detection
  is pending.
- `user_settings.llama_gpu_device_ids`: JSON list of opaque stable device IDs.
- `GET /api/v1/models/local/runtime/devices` publishes managed-runtime support,
  override state, device inventory, and the automatic recommendation.
- Runtime status includes the effective Vulkan device IDs.

## Validation

Cover device parsing and classification, automatic fallback, explicit
single/multi-GPU startup, settings migration and first-use persistence, Settings
UI behavior, native Windows CPU/GPU smoke tests, Docker startup, and a macOS MLX
regression smoke when a suitable host is available.
