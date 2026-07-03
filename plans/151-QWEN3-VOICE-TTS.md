# Qwen3 Voice TTS Backend Support

## Goal

Add backend support for Qwen3 as a selectable text-to-speech voice model and expose supported voice model metadata so the frontend can discover provider, model, voice, language, and instruction options.

## Scope

- Add a Qwen3 TTS provider behind the existing `TTSProvider` abstraction.
- Keep Qwen model loading lazy so app startup and tests do not download weights.
- Thread voice model parameters through the WebSocket and upload endpoints.
- Add a metadata endpoint that returns supported TTS providers and model defaults.
- Add frontend hook parameters for voice model selection.
- Add focused unit tests that validate provider selection and metadata without running model inference.

## Non-Goals

- Do not install or pin Qwen dependencies in this change.
- Do not implement a full voice settings UI.
- Do not require real model downloads in tests.
- Do not rewrite the existing voice session protocol.

## Implementation Notes

- Qwen3 support will use the `qwen_tts` package and raise a clear runtime error when dependencies are missing.
- The first implementation can generate full audio and chunk it through the existing streaming interface. If the local Qwen runtime exposes true incremental audio chunks, the provider should consume that path later without changing the API surface.
- Provider metadata should include whether a provider is local/API backed, whether it supports instruction control, and the default sample rate.

## Validation

- Run focused tests for TTS provider metadata and voice session wiring.
- Run a Python import/compile check for edited backend modules.
