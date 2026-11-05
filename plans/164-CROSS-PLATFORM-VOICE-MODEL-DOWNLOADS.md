# Cross-Platform Voice Model Downloads and Runtime Plan

## Status

Approved as a thin vertical slice after desk research and Apple Silicon voice evaluation on
2026-09-04. Kokoro 82M BF16 with `af_heart` replaces Qwen as the preferred Mac voice because it
was the strongest subjective sample and delivered a 71.2 ms warm median for a 2.35 second test
utterance. The one-time 15.9 second Misaki/spaCy language-model download is not an acceptable
runtime preload: English G2P assets must be installed from the lockfile and synthesis must never
install packages or fetch model files. This plan supersedes the runtime, dependency, and download
assumptions in `plans/151-QWEN3-VOICE-TTS.md`.

## Thin-Slice Implementation Checkpoint

Implemented in the working tree:

- platform-filtered Qwen MLX and Magpie NeMo-Speech model artifacts, with immutable revisions,
  file validation, composite Magpie bundle staging, and Models-tab download/remove controls;
- voice catalog readiness metadata, one fixed voice and English locale per runtime, and disabled
  Voice Settings options until both model and runtime are ready;
- Qwen's direct MLX Audio `generate_custom_voice` streaming path and an isolated framed Magpie
  helper using NeMo-Speech.cpp's stable PCM callback ABI;
- sentence-level LLM-to-TTS handoff, PCM format metadata, dynamic browser sample rates, ordered
  Web Audio scheduling, and provider cleanup; and
- focused artifact, provider, worker-protocol, voice-session, and frontend tests.

Evidence-based Mac pivot implemented and validated:

- add pinned `mlx-community/Kokoro-82M-bf16` weights and only the `af_heart` voice to the managed
  Models-tab catalog;
- lock Misaki's `en_core_web_sm` dependency into the macOS voice installation and fail closed if
  it is missing, before Misaki can attempt a network download; and
- make Kokoro/`af_heart` (displayed as Vera) the preferred installed Mac voice while retaining the
  Qwen adapter as an experimental compatibility lane.

The managed Kokoro provider was exercised with Hugging Face offline and Misaki's downloader
replaced by a failing test sentinel. On the test Apple Silicon host, a clean process produced the
first 2.35-second utterance in 2.318 seconds (including Metal graph compilation) and repeated it in
98 ms without a network or package-install attempt.

Still gated on Linux runtime availability:

- install the NeMo-Speech.cpp v0.1.0 Linux x86_64 CUDA runtime, then run a real Magpie utterance
  on an NVIDIA host; and
- record real TTFA, RTF, cancellation, memory, and speech-to-speech QA evidence. Until those pass,
  the UI reports the local provider as unavailable rather than claiming runtime support.

## Goal

Make local text-to-speech a supported Geist capability on the two primary native targets:

- Apple Silicon: Kokoro 82M BF16 through MLX Audio, using `af_heart` as Vera.
- Linux x86_64 with an NVIDIA GPU: NVIDIA MagpieTTS Multilingual 357M through a native,
  isolated runtime.
- Initial scope: exactly one curated local TTS model per native target, one built-in voice per
  model, and streamed voice response. Voice cloning, voice design, and alternate quantizations
  are deferred.

Expose the required voice artifacts in the Models tab with the same managed download,
resume, cancellation, verification, removal, and platform filtering behavior used by local
language models.

## Recommendation

Use one supportable runtime per platform instead of forcing one model family through every
backend:

| Target | Initial artifact | Runtime | Role |
| --- | --- | --- | --- |
| macOS arm64 | `mlx-community/Kokoro-82M-bf16` at `a71e4d38b236d968966a2002c4c895dbd12b1c3c` | `mlx-audio` sentence-buffered API | Initial Mac TTS (Vera / `af_heart`) |
| Linux x86_64 + NVIDIA | `nvidia/magpie_tts_multilingual_357m` v2602 plus NanoCodec and tokenizer assets | `NeMo-Speech.cpp` stable C ABI in an isolated helper process | Initial Linux TTS |

Use Kokoro voice `af_heart` (display name Vera) and Magpie speaker `John` for the first slice.
Expose speaker metadata in the catalog, but do not add a speaker picker until the two runtime
paths are stable.

Qwen's official Python package and vLLM-Omni remain research/reference lanes on Linux. The
official `qwen-tts` package is batch-oriented and strictly pins `transformers==4.57.3`, while
Geist's current environments use different Transformers/Torch profiles. vLLM-Omni now supports
real PCM and WebSocket streaming, but it is a materially heavier deployment than Magpie's
roughly 527 MB of GGUF weights and the small NeMo-Speech.cpp runtime.

Do not advertise an upstream latency number as Geist's latency. Qwen reports a 97 ms first
packet in its own environment, while MLX streaming behavior has varied by release and NVIDIA's
published Magpie numbers are for hosted/A100-class serving. Release readiness must be based on
measurements on supported Geist hardware.

## Current-State Findings

1. `app/services/tts.py` imports Torch and Torchaudio at module import time. This makes the
   entire voice route unavailable when the optional voice extra is not installed.
2. The Qwen provider imports `QwenTTS` and probes `synthesize`, `generate`, and `infer`.
   Qwen's official package exposes `Qwen3TTSModel` and task-specific methods such as
   `generate_custom_voice`, `generate_voice_clone`, and `generate_voice_design`.
3. The current `synthesize_streaming` fallback generates the whole utterance before slicing
   PCM, and `VoiceSessionService` waits for the complete LLM response before invoking TTS.
   Neither behavior provides end-to-end low latency.
4. `scripts/download_models.py` can prefetch several voice repositories into the Hugging Face
   cache, but those downloads are not represented in Geist's managed artifact index and do
   not appear in the Models tab.
5. `LocalModelArtifact` and `LocalModelManager` already implement most of the required state
   machine, but assume a single LLM GGUF or a Transformers/MLX snapshot. Snapshot validation
   also assumes LLM files such as root `config.json` and tokenizer weights.
6. `client/geist/src/Models.tsx` assumes an installed local artifact can become
   `default_local_artifact_id`. Voice artifacts therefore need modality-aware actions rather
   than the current universal **Use** button.
7. The Linux lock currently resolves Torch and Torchaudio from the CPU-only PyTorch index.
   CUDA support must be an explicit, conflicting install/build profile rather than silently
   changing the existing Docker image.
8. The native PyInstaller sidecar excludes Torch voice modules. A shippable voice feature
   therefore needs either a voice-enabled distribution or a separate runtime process.
9. The repository has no committed speech fixture suitable for a repeatable real QA loop.
   Existing voice tests use mocks and synthetic numeric buffers.

## Architecture

### 1. Separate logical voice models from downloadable artifacts

Add a typed voice catalog, separate from the LLM-only `agents/model_catalog.py`. A logical
voice model describes the user-facing capability; one or more artifacts describe how it runs
on a target platform.

Suggested contracts:

```text
VoiceModelSpec
  id, provider, display_name, modality
  languages, speakers, sample_rate
  supports_streaming, supports_instruction_control, supports_voice_cloning
  artifact_ids

LocalModelArtifact
  existing fields
  modality: llm | tts | stt
  runtime: mlx_audio | qwen_tts_cuda | nemo_speech
  validation_profile
  components[] (optional composite bundle)
```

Keep one artifact manager for all local model files so there is one index, storage root, and
download state machine. Do not add voice models to the LLM catalog or route them through an
LLM runner.

### 2. Support composite, verified voice bundles

Extend managed artifacts from a single file/snapshot to a curated bundle of components. Each
component must declare an allowlisted repository, immutable revision, filename or pattern,
expected size when known, and SHA-256 when available.

The Magpie bundle contains:

- the Magpie token-generator GGUF;
- the matching ready-to-use tokenizer directory from the same immutable Hugging Face commit;
- the NeMo NanoCodec decoder GGUF;
- optional text-normalization grammar assets, kept out of the first English-only slice unless
  the size and licensing are acceptable.

Stage every component under `.downloads`, verify the complete manifest, then atomically promote
the bundle. The selected v2602 revision publishes tokenizer files directly, so the thin slice
does not need to download or extract the 1.21 GB `.nemo` archive. Cancellation or failure must
leave no runnable half-install. Store a versioned installation manifest so the runtime can
reject mixed Magpie/tokenizer revisions.

Initial curated entries:

- `qwen3-tts-0.6b-customvoice-mlx-6bit` for Darwin arm64;
- `magpie-tts-multilingual-357m-nemo-speech` for Linux x86_64, initially using the known-good
  preconverted v2602 stack.

Use the researched immutable revisions and file hashes recorded below. Treat any upstream
revision change as a catalog update with a new smoke test, never as an implicit floating update.

### 3. Model platform support and runtime readiness separately

`supported` currently means only that Geist has a backend for the OS/architecture. Add a
distinct readiness result so the API can explain conditions such as:

- supported on Darwin arm64, but `mlx-audio` is missing;
- supported on Linux x86_64, but no compatible NVIDIA GPU/CUDA runtime is available;
- artifact is installed, but the required `nemo-speech` binary is absent;
- license acceptance or Hugging Face authentication is required.

The Models tab should allow downloads only for platform-compatible artifacts and show a clear
runtime prerequisite instead of failing at synthesis time.

### 4. Isolate heavyweight runtime dependencies

Keep the FastAPI process and voice protocol independent of MLX, CUDA, and NeMo implementation
details. Define a small TTS runtime contract that accepts text, model, speaker, language, and
instruction and emits typed PCM chunks with cancellation.

Preferred implementation:

- Mac thin slice: a lazy, serialized in-process `mlx-audio` adapter loaded from the managed
  snapshot. Move it to a subprocess only if dependency resolution, cancellation, or crash
  isolation proves necessary after hardware validation.
- Linux Magpie: a small managed helper process that loads NeMo-Speech.cpp's stable TTS C ABI.
  The ABI emits PCM through callbacks and cancels when the callback returns false. Model the
  lifecycle after `agents/architectures/llama_server_process.py`, but use a framed local IPC
  protocol rather than the NeMo HTTP endpoint because the HTTP speech endpoint buffers the
  complete utterance.

The parent process should own lifecycle and address selection, bind only to loopback, pass
explicit managed artifact paths, and never let client-controlled model IDs become repository
IDs or filesystem paths.

This process boundary avoids forcing mutually incompatible MLX, CPU Torch, CUDA Torch, and
native speech packages into one Python environment, and isolates native runtime crashes from
FastAPI. Keep the official `qwen-tts` and vLLM-Omni CUDA paths out of the first slice.

### 5. Correct and slim the TTS service layer

Split `app/services/tts.py` into:

- typed catalog and request/result models with no Torch import;
- provider factory and common PCM chunk contract;
- lazy provider adapters for OpenAI, Sesame, Qwen, and Magpie;
- runtime lifecycle modules for MLX/CUDA/NeMo processes.

Replace the current generic Qwen method probing with the actual selected backend APIs. Keep
Torch/MLX imports inside their provider modules. An installation problem should make one
provider unavailable, not prevent `/api/v1/voice/models` or unrelated API routes from loading.

### 6. Make the voice wire protocol format-aware

Stop assuming all output is 24 kHz. Send an audio-start metadata event before binary chunks:

```json
{
  "type": "audio_start",
  "encoding": "pcm_s16le",
  "sample_rate": 24000,
  "channels": 1,
  "model_id": "qwen3-tts-0.6b-customvoice"
}
```

Qwen currently emits 24 kHz audio; Magpie emits 22.05 kHz PCM. The client can construct the
correct audio buffer from metadata, or the provider boundary can resample all output to one
documented session rate after latency and quality measurement. The protocol must support
cancel/barge-in without playing queued audio from an obsolete response.

### 7. Start TTS before the LLM finishes

Add a sentence/phrase accumulator to `VoiceSessionService`. As LLM text arrives, enqueue stable
segments at punctuation boundaries or a bounded character/time threshold. Preserve ordering,
apply backpressure, and flush the tail at completion. This is required for low perceived
latency even when the selected model can stream internally.

Track at least:

- user-speech-end to first LLM token;
- first LLM token to first TTS request;
- TTS request to first playable audio;
- real-time factor and underrun count;
- cancellation-to-silence time.

### 8. Add Voice Models to the Models tab

Keep the top-level **Local Models** tab, but split its inventory into **Language** and **Voice**
sections or filters. Voice rows should show:

- TTS badge and logical family;
- platform/runtime (`MLX`, `CUDA`, or `NeMo-Speech.cpp`);
- quantization, download size, languages, sample rate, and license;
- download, cancel, resume, remove, and repair actions;
- installed versus runtime-ready status;
- a **Configure voice** action instead of the LLM **Use** action.

Group alternate artifacts under one logical model so users see “Qwen3 TTS 0.6B” once, with the
compatible local build selected automatically. Do not show NVIDIA/CUDA downloads as runnable
on a Mac. An optional “Other platforms” disclosure may explain availability without offering
an invalid action.

Update Voice Settings so an uninstalled local provider has a **Download in Models** call to
action and cannot start a session. Voice selection must not modify
`default_local_artifact_id`; add separate persisted TTS model/artifact/speaker/language fields
only after confirming whether voice choice should be global or per chat.

### 9. API changes

Extend rather than replace the existing local-artifact endpoints:

- `GET /api/v1/models/local/artifacts?modality=tts`
- existing download, cancel, and remove endpoints for all curated artifact modalities;
- `GET /api/v1/voice/models` returns typed catalog entries merged with artifact installation
  and runtime readiness;
- `GET /api/v1/voice/runtime/status` exposes selected provider lifecycle and load errors;
- optional `POST /api/v1/voice/runtime/load` and `/unload` if explicit prewarming proves useful.

Return structured Pydantic response models rather than untyped dictionaries. Preserve the
existing endpoints during migration so current clients continue to work.

## Research Results (2026-09-04)

### Qwen on Apple Silicon

- The official model is Apache-2.0, provides nine built-in voices and ten languages, emits
  24 kHz audio, and uses `Qwen3TTSModel.generate_custom_voice`. The existing Geist adapter's
  `QwenTTS` import and generic method probing are not compatible.
- The official `qwen-tts==0.1.1` package strictly requires `transformers==4.57.3` and recommends
  an isolated environment. That is unsuitable for Geist's shared Mac environment.
- `mlx-audio==0.5.1` is the current release observed during research. Its current metadata
  requires Python 3.10+, `mlx>=0.31.1`, `transformers>=5.14.0`, and `sounddevice>=0.5.3`.
  Geist's Mac MLX/Transformers pins are compatible in principle, but the existing voice extra
  pins an older SoundDevice. Add or change no package until an approved `uv add` resolves and
  audits the exact set.
- MLX Audio documents direct incremental generation with `stream=True` and a Qwen-specific
  `streaming_interval` around 0.32 seconds. An open issue reports broken incremental/final
  events in the separate continuous-batch API on 0.4.3. Use the direct per-request generator
  first and serialize requests in one worker.
- Hugging Face's maintained speech-to-speech reference defaults Apple Silicon Qwen to 6-bit.
  That is the best initial balance at 1.83 GB. Keep 4-bit and 8-bit out of the UI until an
  on-device A/B test justifies exposing them.

### Qwen on Linux/NVIDIA

- Qwen's vendor number of 97 ms is an upstream best case, not a result for Geist hardware.
- The official Python API currently labels its `non_streaming_mode=False` behavior as simulated
  streaming rather than true streaming generation. Do not use it for the live path.
- vLLM-Omni now provides real raw-PCM streaming and an incremental WebSocket endpoint for
  Qwen3-TTS, so it is the best future CUDA serving candidate. It is intentionally deferred
  because it adds a much larger server/runtime surface than the first Linux slice.

### NVIDIA Magpie

- The current Magpie v2607 model has 364M parameters, five fixed English speaker voices and
  twelve advertised languages. The locally runnable preconverted NeMo-Speech.cpp artifact is
  v2602, so the first slice pins v2602 and advertises English only until each additional
  language frontend is tested.
- NeMo-Speech.cpp v0.1.0 is the first versioned release. It offers CPU, CUDA, Vulkan, and Metal
  binaries and calls its API early/evolving. Pin a release asset and checksum instead of an
  install script or `latest` URL.
- Its HTTP `/v1/audio/speech` endpoint buffers complete audio. Its documented stable C ABI,
  however, emits PCM16 chunks through a callback, cancels when that callback returns false,
  and reports TTFA, real-time factor, and inter-chunk latency. Use the C ABI behind an isolated
  Geist helper process.
- The selected v2602 Hugging Face commit publishes the F16 Magpie GGUF and ready-to-use
  tokenizer files directly. Combined with NanoCodec, the two weight files are about 527.4 MB;
  the 1.21 GB `.nemo` archive is not required.
- The NVIDIA Open Model License permits commercial use, derivatives, and distribution, but it
  is not Apache/MIT. The Models tab must link the license and show its name. If Geist ever
  redistributes weights rather than downloading them directly from Hugging Face, the bundle
  must include the license and NVIDIA's required NOTICE attribution.
- The pinned Linux x86_64 CUDA runtime archive is
  `nemo-speech-0.1.0-linux-x86_64-cuda.tar.gz` (107,310,946 bytes), published with SHA-256
  `e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147`. Runtime installation is
  intentionally separate from the Hugging Face model bundle and requires explicit approval.

### Immutable artifact candidates

| Artifact | Revision | Primary weight size | Primary weight SHA-256 |
| --- | --- | ---: | --- |
| Qwen MLX 6-bit | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit@7dc92af14613355896fcab13b268c19ede233139` | 1,146,758,090 bytes; 1.83 GB snapshot | `77f20155cf00cc7cbafeb6f51863e27bda9051603557d815f0f24e95a5a79513` |
| Magpie v2602 F16 | `nvidia/magpie_tts_multilingual_357m@61a0a4494a40e85adecebaa02b4aa6c70b5fef0d` | 448,604,832 bytes | `901d299a8b1df016cf81cae0089a7a7c15627b9633d033357e15a47d9a219a75` |
| NanoCodec decoder F16 | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps@fc00890b604aa2de298d2641ffc6c5f6caf8c4d7` | 78,823,104 bytes | `cc86d36d821a27cdc1d4ef600a3e2b0dabe76e88fcc2a8652d9543134c07ef2d` |

The Qwen snapshot was converted with MLX Audio 0.3.0 but will run against a newer runtime, so
the implementation must smoke-test config loading and one real utterance before treating the
pin as supported.

### End-to-end QA corpus and gates

Do not commit an arbitrary third-party speaker clip. Add one short project-owned 16 kHz mono
WAV recorded with explicit contributor consent and an exact transcript. This is simpler than
carrying dataset access terms and gives Geist a stable microphone-to-response fixture. The
TTS-only round trip needs no human fixture: synthesize fixed text, feed the result to the
installed STT provider, and compare its transcript.

Initial fixed prompts cover a short pangram, punctuation, a date/decimal/units sentence, a
250-character response, and cancellation midway through a long response. Generated audio is
test evidence, not a byte-for-byte golden file.

Candidate release gates on each supported baseline machine:

- warm TTS request-to-first-playable-audio p95 at or below 500 ms;
- real-time factor at or below 0.8, with p95 inter-chunk time below the duration of buffered
  playable audio so playback does not underrun;
- cancellation request to silence at or below 250 ms;
- English TTS-to-STT round-trip word error rate at or below 10% on the fixed prompts;
- clipped-sample ratio below 0.1%, leading silence below 500 ms, correct mono PCM metadata,
  and no audible gaps at chunk boundaries.

Cold load time and peak memory are reported but not hard gates for the first slice. If a path
misses the live gates, keep downloads/configuration available but label it non-realtime rather
than silently buffering the full answer.

## Implementation Sequence

### Phase 0: Resolve runtime and licensing decisions

1. Desk research, model/runtime selection, immutable model pins, and license review are complete.
2. Request approval before adding packages or downloading weights, then use exact pins and
   audit the resulting lock.
3. Run one real Qwen MLX and one real Magpie CUDA synthesis on baseline hardware before marking
   either runtime ready.
4. Record exact supported hardware, runtime binary hashes, measured latency, and memory.

### Phase 1: Catalog and managed downloads

1. Add typed voice catalog models and modality-aware artifact metadata.
2. Add verified composite-bundle support and validation profiles to `LocalModelManager`.
3. Add the curated Qwen and Magpie artifacts.
4. Merge artifact state into `/api/v1/voice/models`.
5. Remove or redirect duplicated voice download definitions in `scripts/download_models.py` so
   the managed catalog is the source of truth.

### Phase 2: Models and Voice Settings UI

1. Add Language/Voice grouping and modality-aware actions to `Models.tsx`.
2. Add download progress, prerequisites, license, and runtime-readiness messaging.
3. Link installed voice artifacts to Voice Settings without changing LLM defaults.
4. Add accessible empty, unavailable, failure, cancellation, and repair states.

### Phase 3: Platform runtimes

1. Introduce the lightweight TTS provider contract and lazy imports.
2. Implement and package the Mac MLX Qwen runtime.
3. Implement the Linux Magpie runtime and process lifecycle.
4. Add prewarm, unload, crash recovery, resource limits, and shutdown handling.

### Phase 4: Low-latency session behavior

1. Add audio metadata events and dynamic sample-rate handling.
2. Segment streaming LLM output into ordered TTS work.
3. Add bounded queues, backpressure, cancellation, and barge-in.
4. Instrument stage latency and real-time factor without logging user audio or transcript text.

### Phase 5: QA and rollout

1. Add the licensed QA fixture and repeatable benchmark command.
2. Run focused unit, contract, API, and frontend tests.
3. Run real synthesis and round-trip QA on Apple Silicon and Linux/NVIDIA.
4. Run the Geist pre-push test loop for Docker, browser, chat/settings, and native MLX behavior.
5. Roll out behind explicit local-provider selection before changing any default.

## Test Plan

- Catalog tests: identity, modality, platform filtering, capabilities, licenses, and immutable
  revisions.
- Artifact tests: bundle resume/cancel, hash failure, revision mismatch, atomic promotion,
  repair, and removal.
- API tests: modality filtering, installation/readiness merge, invalid artifact IDs, concurrent
  operations, and actionable error payloads.
- Frontend tests: grouped voice rows, progress polling, unavailable targets, Configure voice,
  license/auth messages, and protection against changing the active LLM.
- Provider tests: correct Qwen method/argument mapping, Magpie request mapping, PCM metadata,
  lazy optional imports, cancellation, and child-process cleanup.
- Session tests: phrase segmentation, output ordering, backpressure, tail flush, barge-in, and
  sample-rate changes.
- Runtime smoke tests: real first audio from Qwen MLX and Magpie CUDA, plus Qwen CUDA if shipped.
- QA loop: input speech -> STT transcript -> Geist response -> TTS audio -> optional STT
  round-trip, with timing, clipping, silence, and intelligibility results captured separately.

## Acceptance Criteria

- A supported Mac can download, verify, configure, and produce audible Qwen speech without a
  runtime download outside Geist's model store.
- A supported Linux/NVIDIA host can do the same with the selected Magpie bundle.
- Voice artifacts never appear as selectable LLMs and never overwrite the active LLM artifact.
- Unsupported hardware and missing runtime dependencies are explained before download or use.
- Cancelled and failed downloads do not leave a runnable partial bundle.
- The API communicates encoding, channels, and sample rate; the client does not hardcode 24 kHz.
- TTS begins from streamed response segments instead of waiting for the full LLM answer.
- Runtime imports are lazy, so the model catalog and non-voice Geist features work without the
  optional voice distribution.
- Exact runtime/model revisions and dependency audits are recorded in the implementation PR.
- Real hardware evidence reports cold load, warm first-audio latency, real-time factor, memory,
  cancellation latency, and round-trip intelligibility for each supported target.

## Voice Review Follow-up (September 4, 2026)

- Keep capture and WebSocket control on the event loop; run STT, LLM iteration,
  and TTS on a single serial worker per session. Hand off one output event at a
  time. Stop/reset cancels delivery immediately and closes the generator on its
  worker after the current inference step returns. GPU kernels are not forcibly
  interrupted; no subsequent work runs concurrently on that session's models.
- Use half-duplex capture: ignore microphone input during server processing and
  scheduled browser playback. Track every audio source, preserve the timeline
  across ordinary turns, and invalidate queued callbacks on stop/reconnect.
- Discard workers after incomplete framed Magpie exchanges, including generator
  cancellation, timeout, protocol errors, and transport failures.
- Resolve the installed catalog default before enabling Record, and persist only
  explicit user voice choices. Validate saved choices against the current host.
- Wait for whitespace or end-of-response before splitting trailing punctuation,
  preserving decimal numbers split across LLM tokens.
- Encode normalized samples as a 16 kHz mono PCM WAV for Whisper API uploads.
- Measure VAD silence in samples, use 64 ms browser capture frames, ignore idle
  silence, and cap an utterance at 30 seconds. Partial STT is background work with
  at most one outstanding request, at one-second audio intervals.
- Construct MMS lazily and reuse a locked process-wide English model. Recording
  paths use only cached assets; an explicit prefetch command downloads only the
  six English checkpoint/processor files, not all language adapters. First use
  still pays a cold model load; later sessions reuse it. This is not zero-cold-start
  or full-duplex/barge-in support.
- Pin SentencePiece 0.2.2 on Apple Silicon across compatible extras. The previous
  0.2.0 build reproduced a C++ flag-destructor SIGBUS when MMS and Kokoro ran in
  the same process; the updated build passes the offline round-trip and exits
  cleanly. Linux/Windows pins remain unchanged.

## Deferred Work

- Qwen VoiceDesign and arbitrary voice cloning; first ship curated built-in voices with a clear
  consent and abuse policy.
- Meta Omnilingual ASR. It is a speech-to-text model family, not a TTS response model, and its
  current Fairseq2-oriented serving path should be evaluated in a separate STT plan.
- Windows voice packaging.
- Automatic default changes based only on vendor benchmark claims.
