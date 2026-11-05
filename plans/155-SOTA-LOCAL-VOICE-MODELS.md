# 155 — SOTA Local Voice Models

## Goal

Close the gap between Geist's voice pipeline and the current state of the art
in local speech models:

1. **Local STT** — the only accurate STT option was the OpenAI Whisper API;
   the local option (Meta MMS 1B) is well behind the Open ASR Leaderboard.
   Add NVIDIA Parakeet TDT (leaderboard-class WER at very high throughput)
   and local Whisper via faster-whisper (CTranslate2).
2. **Lightweight TTS** — Sesame CSM-1B needs a real GPU. Add Kokoro-82M,
   which runs on CPU or ~2-3 GB VRAM with strong quality.
3. **Neural VAD** — replace the hand-rolled RMS threshold with Silero VAD
   (with RMS retained as fallback).
4. **Streaming TTS** — add CosyVoice2-0.5B, which supports native
   low-latency incremental synthesis (plan 145 listed true streaming TTS as
   known future work).

## What was built

### STT

- `adapters/faster_whisper_adapter.py` — `FasterWhisperAdapter` over
  `faster_whisper.WhisperModel`. Default model `large-v3-turbo`; allowlisted
  sizes down to `tiny`.
- `adapters/parakeet_adapter.py` — `ParakeetAdapter` over NeMo
  `ASRModel.from_pretrained`. Default `nvidia/parakeet-tdt-0.6b-v3`.
- `app/services/stt.py` — new STT catalog + factory mirroring the TTS
  pattern: `SUPPORTED_STT_PROVIDERS` metadata, `_validate_stt_model`
  allowlist (the model id is client-controlled and reaches model loaders),
  and `create_stt_adapter()`. `VoiceSessionService` now uses the factory
  instead of a hardcoded if/elif. The factory checks the optional runtime
  and loads the model before the WebSocket reports `ready`, so a missing
  package is a setup error rather than a silent per-chunk failure.
- `adapters/shared_model_cache.py` — process-wide cache so each new voice
  connection reuses loaded faster-whisper, Parakeet, Kokoro and CosyVoice2
  weights. Each entry carries a lock; TTS engines are locked per segment
  so a slow client does not stall other sessions. Silero VAD is stateful
  and tiny, so it is loaded per session instead.
- Partial transcripts re-run STT over the whole utterance buffer, so they
  are throttled to once per `partial_interval_ms` (1 s) of new audio.
  Incremental/streaming decoding is not attempted.

### VAD

- `app/services/vad.py` — `VADProvider` ABC with `SileroVAD` (512-sample
  windows at 16 kHz, stateful, `reset()` between utterances) and `RMSVAD`
  (original behavior). `create_vad("auto")` prefers Silero when the
  `silero-vad` package is installed and falls back to RMS otherwise, so
  default deployments are unchanged until the package is added. Any
  inference-time exception (including torch `RuntimeError`) degrades
  permanently to RMS rather than breaking a live session; an explicit
  `silero` selection without the package fails at setup.

### TTS

- `KokoroPyTorchTTSProvider` (provider id `kokoro_pytorch`) in
  `app/services/tts.py` — optional `kokoro` package on CPU/CUDA,
  per-segment streaming (audio starts before the full text is synthesized),
  multi-language via Kokoro lang codes. Voice, language and speed are
  validated against the catalog: `KPipeline` would otherwise `torch.load`
  any voice string ending in `.pt` and blend comma-separated voices.
- Coexistence with the managed MLX Kokoro provider (PR #355): that PR owns
  the bare `kokoro` id (`kokoro-82m-bf16-mlx` artifact, `af_heart`, Apple
  Silicon). This provider is the cross-platform backend for the same
  `hexgrad/Kokoro-82M` model under a distinct id, sharing the
  `DEFAULT_KOKORO_TTS_MODEL` / `KOKORO_DEFAULT_VOICE` constant names. Its
  catalog entry lives in `app/services/voice_catalog.py` with the other
  runtime-free metadata. Once both land, `kokoro` could dispatch to MLX on
  Apple Silicon and to this provider elsewhere.
- `CosyVoice2TTSProvider` in `app/services/tts.py` — experimental, optional
  `cosyvoice` package (FunAudioLLM/CosyVoice repo). Native `stream=True`
  incremental synthesis. CosyVoice2-0.5B is zero-shot, so the reference
  voice comes from server-side env config (`COSYVOICE2_PROMPT_AUDIO` +
  `COSYVOICE2_PROMPT_TEXT`), never from client input; falls back to
  pretrained speaker ids when the loaded model ships any.

### API

- `GET /api/v1/voice/models` now also returns `stt_providers` metadata.
- `/voice/stream` and `/voice/upload` accept `stt_model` and the new
  provider names (`stt_provider=faster_whisper|parakeet`,
  `tts_provider=kokoro_pytorch|cosyvoice2`).

## Dependencies (intentionally undeclared)

Following the qwen_tts precedent from plan 151, the new model packages are
not declared in `pyproject.toml`; each provider fails at session setup
with install instructions:

| Feature | Package |
| --- | --- |
| faster_whisper STT | `pip install faster-whisper==1.2.1` |
| parakeet STT | `pip install "nemo-toolkit[asr]"` |
| silero VAD | `pip install silero-vad==6.2.3` |
| kokoro_pytorch TTS | `pip install kokoro==0.9.4` |
| cosyvoice2 TTS | CosyVoice repo install (FunAudioLLM/CosyVoice) |

Rationale: NeMo and CosyVoice have heavy dependency trees that conflict
easily with the pinned inference stack, and regenerating `uv.lock` needs a
Python 3.11 toolchain that was not available in the implementation
environment. Promoting the well-behaved ones to a declared extra is a good
follow-up:

```toml
voice-sota = [
    "faster-whisper==1.2.1",
    "silero-vad==6.2.3",
    "kokoro==0.9.4",
    # misaki would otherwise pip-install its spaCy model at first use.
    "en-core-web-sm==3.8.0",
]
```

with the same `en-core-web-sm` wheel URL under `[tool.uv.sources]` that
PR #355 uses for MLX Kokoro.

## Non-goals

- Wiring the vendored Moshi speech-to-speech stack into the app (tracked
  separately; larger project).
- Voice cloning from client-supplied audio (security: file paths and audio
  must not come from the client).
- Frontend selector UI for the new providers (the `/models` endpoint now
  exposes everything needed).

## Testing

- `tests/services/test_stt.py` — catalog metadata, setup-time loading and
  cross-session model sharing, model allowlist rejection, missing-package
  errors at setup (including the WebSocket sending `error` instead of
  `ready`), transcription result coercion with fake engines.
- `tests/services/test_vad.py` — RMS behavior, auto-fallback, Silero
  windowing/threshold with a fake model, fallback on torch `RuntimeError`,
  reset propagation.
- `tests/services/test_tts.py` — Kokoro/CosyVoice2 metadata, factory,
  streaming with fake pipelines/engines, model and voice allowlist
  rejection (paths, blends), shared Kokoro pipelines.
- `tests/services/test_voice_session.py` — updated to mock the STT factory;
  VAD wiring covered via provider override.
