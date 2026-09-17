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

- `adapters/faster_whisper_adapter.py` — `FasterWhisperAdapter`, lazy-loads
  `faster_whisper.WhisperModel`. Default model `large-v3-turbo`; allowlisted
  sizes down to `tiny`.
- `adapters/parakeet_adapter.py` — `ParakeetAdapter`, lazy-loads NeMo
  `ASRModel.from_pretrained`. Default `nvidia/parakeet-tdt-0.6b-v3`.
- `app/services/stt.py` — new STT catalog + factory mirroring the TTS
  pattern: `SUPPORTED_STT_PROVIDERS` metadata, `_validate_stt_model`
  allowlist (the model id is client-controlled and reaches model loaders),
  and `create_stt_adapter()`. `VoiceSessionService` now uses the factory
  instead of a hardcoded if/elif.

### VAD

- `app/services/vad.py` — `VADProvider` ABC with `SileroVAD` (512-sample
  windows at 16 kHz, stateful, `reset()` between utterances) and `RMSVAD`
  (original behavior). `create_vad("auto")` prefers Silero when the
  `silero-vad` package is installed and falls back to RMS otherwise, so
  default deployments are unchanged until the package is added. Runtime
  inference failures degrade permanently to RMS rather than breaking a live
  session.

### TTS

- `KokoroTTSProvider` in `app/services/tts.py` — optional `kokoro` package,
  per-segment streaming (audio starts before the full text is synthesized),
  multi-language via Kokoro lang codes, curated voice list in metadata.
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
  `tts_provider=kokoro|cosyvoice2`).

## Dependencies (intentionally undeclared)

Following the qwen_tts precedent from plan 151, the new model packages are
not declared in `pyproject.toml`; each provider fails loudly at first use
with install instructions:

| Feature | Package |
| --- | --- |
| faster_whisper STT | `pip install faster-whisper==1.2.1` |
| parakeet STT | `pip install "nemo-toolkit[asr]"` |
| silero VAD | `pip install silero-vad==6.2.1` |
| kokoro TTS | `pip install kokoro==0.9.4` |
| cosyvoice2 TTS | CosyVoice repo install (FunAudioLLM/CosyVoice) |

Rationale: NeMo and CosyVoice have heavy dependency trees that conflict
easily with the pinned inference stack, and re-locking against the pytorch
CPU index was not possible in the implementation environment. Promoting the
well-behaved ones (faster-whisper, silero-vad, kokoro) to declared extras
with a regenerated `uv.lock` is a good follow-up.

## Non-goals

- Wiring the vendored Moshi speech-to-speech stack into the app (tracked
  separately; larger project).
- Voice cloning from client-supplied audio (security: file paths and audio
  must not come from the client).
- Frontend selector UI for the new providers (the `/models` endpoint now
  exposes everything needed).

## Testing

- `tests/services/test_stt.py` — catalog metadata, factory lazy-construction,
  model allowlist rejection, missing-package errors, transcription result
  coercion with fake engines.
- `tests/services/test_vad.py` — RMS behavior, auto-fallback, Silero
  windowing/threshold with a fake model, reset propagation.
- `tests/services/test_tts.py` — Kokoro/CosyVoice2 metadata, factory,
  streaming with fake pipelines/engines, allowlist rejection.
- `tests/services/test_voice_session.py` — updated to mock the STT factory;
  VAD wiring covered via provider override.
