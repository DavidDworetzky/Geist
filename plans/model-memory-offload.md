# Model memory recovery

The Windows Vulkan runtime currently receives `--n-gpu-layers 999`. A 17.7 GiB
Qwen GGUF fails on a 16 GiB RTX 3080 Laptop GPU with `ErrorOutOfDeviceMemory`.
The same installed runtime and model load successfully with GPU layers set to
`auto`, retaining the existing context size.

## Behavior

- Persist `llama_allow_system_ram`, default false, beside the compute settings.
- When enabled, allow llama.cpp to fit model layers across GPU and system RAM.
  Explain that responses may be slower. Keep explicit operator layer overrides.
- Include the preference in runner configuration and model reuse identity so a
  changed preference takes effect on the next load.
- Classify confirmed GPU/host allocation failures from bounded startup output;
  retain technical output in logs and return a safe structured failure code.
- Show a specific memory error in a dismissible Geist modal with links to
  compute settings or a smaller model. Keep ordinary load errors distinct and
  retain retry. Show it once per failed attempt; a small "See more…" badge
  beside the existing "Model unavailable" composer text reopens it.
- Migrate existing and unversioned supported databases without changing user
  model selections. Reset to Defaults disables the new preference.

## MLX and recovery actions

- Translate catchable MLX/Metal allocation failures during loading and generation
  into a unified-memory error, preserving other failures and cancellation.
- Carry `can_offload_to_system_ram` with memory failures through the load-status
  API. Enable it only for llama.cpp GPU failures with offloading disabled.
- Include memory failures in chat error events using the existing `model_load`
  shape. Show the same modal and composer badge for generation failures, scoped
  to the affected model and chat, without adding an inline error panel.
- Keep Retry available and clear the failed attempt when retried. Never offer
  offloading for MLX. OS/native process termination cannot be translated by an
  in-process Python exception handler.

## Validation

Cover migration defaults/persistence, setting propagation and cache invalidation,
startup argument policy and allocation-error classification, and the Settings
and Chat recovery flows. Build the frontend and run native Windows startup with
the installed model for both policies. Do not modify model files or dependencies.
