## 145 - VOICE CHAT (STREAMING) PLAN

### Goal
- Add push-to-talk voice chat initiated from a mic/voice icon in the main Geist chat window.
- Stream audio from the browser to the backend; stream responses (audio) back to the browser.
- While audio plays back, show the assistant transcript streaming in the chat UI.
- Integrate with existing agents: `OnlineAgent` or `LocalAgent` as the LLM source.
- Make as few architectural changes as possible; only extend core abstractions if necessary.

### What exists today
- `BaseAgent` defines the following relevant methods:
  - `complete_text(...)` (implemented by `OnlineAgent`, `LocalAgent`)
  - `complete_audio(...)` (mixed coverage: `OnlineAgent` uses Whisper transcription endpoint then calls `complete_text`; `LocalAgent` not implemented; `LlamaAgent` has an experimental path using Sesame CSM + MMS)
  - `connect_realtime_audio()` is defined but not implemented across agents (raises `NotImplementedError`).
- STT adapters exist:
  - `adapters/mms_adapter.py` (Meta MMS local STT; expects 16k samples, full-clip, not streaming by default)
  - `adapters/whisper_adapter.py` (OpenAI Whisper; currently non-streaming, full-clip)
- API endpoints in `app/main.py` support:
  - `/agent/complete_text` and variants with session
  - `/adapter/speech_to_text` for file upload → STT (non-streaming)
- A streaming voice reference exists under `agents/architectures/moshi/` with a WebSocket server that exchanges:
  - binary audio frames (Opus), and
  - streamed text tokens and audio frames back.
  This provides a good model for real-time duplex, but it’s not yet integrated with agents/workflows.

### Constraints and design preferences
- Minimal core changes: prefer adding a new API surface that composes existing `complete_text` over refactoring agent internals.
- Avoid broad refactors to `BaseAgent` unless strictly necessary.
- Reuse existing adapters where possible; prefer local STT (`MMSAdapter`) for on-device flows and Whisper API for online.
- Frontend should be simple to wire: mic button toggles a streaming session; transcript renders progressively; audio plays as it arrives.

### Proposed architecture
Phase 1 (MVP streaming, minimal changes):
- Introduce a new WebSocket endpoint `ws /agent/voice/stream` to handle real-time audio input and streamed outputs.
- Server will:
  1) Accept a chat `session_id` and `agent_type` during handshake (query params or first JSON control frame).
  2) Receive audio chunks (prefer raw 16k PCM float/int16 frames; if Opus is desired, reuse the Moshi Opus helpers to decode/encode).
  3) Buffer audio and periodically run STT on accumulated windows to produce partial transcripts; stream transcript deltas to the client.
  4) Detect phrase boundaries (VAD or time-based heuristic) → upon boundary, call `agent.complete_text(prompt=phrase_text, chat_id=session_id)`.
  5) Convert final assistant text to audio and stream audio frames back. For MVP, TTS can be non-streaming generation with chunked send; transcript streams immediately.
  6) Add token-level LLM streaming and start TTS synthesis concurrently so both text and audio start almost immediately. 
  7) For online providers that support stream, implement an OnlineAgent.stream_complete_text(...) path; for local runners, enable streaming in runners where available.

### Backend changes
1) WebSocket endpoint
   - File: `app/api/v1/endpoints/voice.py` (new)
   - Route: `ws /agent/voice/stream`
   - Protocol (binary/text frames):
     - Control handshake (JSON; includes `session_id`, `agent_type`, and optional `stt_provider`, `tts_provider`).
     - Audio in: binary frames of audio; strongly prefer 16k mono PCM float32/int16 for minimal deps. If Opus is desired, leverage existing `agents/architectures/moshi` Opus helpers.
     - Outgoing messages:
       - `transcript_partial` (JSON): partial STT updates
       - `transcript_final` (JSON): phrase-level final STT
       - `assistant_text` (JSON): final assistant response text (or incremental later)
       - `audio_chunk` (binary): stream of audio frames (PCM or Opus)
       - `done` (JSON): signals end of a response turn

2) Voice session service
   - File: `app/services/voice_session.py` (new)
   - Responsibilities:
     - Audio buffering, resampling to 16k, and segmentation (simple VAD heuristic or duration-based, e.g., 0.8–1.2s windows).
     - STT adapter selection and invocation: `MMSAdapter` (local) or `WhisperAdapter` (online). For MMS, convert chunks to the expected tensor format.
     - Maintain aggregated transcript across chunks; emit deltas upstream.
     - On phrase boundary, call `get_active_agent(agent_type).complete_text(...)` to get assistant reply (preserves chat history via `chat_id`).
     - TTS provider invocation; returns audio frames stream.
     - Flow control and backpressure to the WebSocket.

3) TTS provider
   - Preferred path (local): use Sesame CSM generator under `agents/architectures/sesame/` for text → audio; return PCM frames and optionally wrap with Opus.
   - Online alternative: leave interface open for cloud TTS vendors (can be added later without changing the endpoint contract).
   - Abstraction: `app/services/tts.py` (new) with a provider interface and a Sesame-backed implementation.

4) Configuration
   - `app/environment.py` additions for `VOICE_STT_PROVIDER` (mms|whisper) and `VOICE_TTS_PROVIDER` (sesame|...)
   - Defaults: local MMS + local Sesame when available; fall back to Whisper (for STT) if configured, and to client-side TTS if server TTS unavailable.

5) Optional HTTP fallback endpoint
   - `POST /agent/voice/upload` to accept a short recorded clip, transcribe fully, call `complete_text`, return `{transcript, text, audio_url}`. Useful for environments without WebSocket or for quick testing.

### Frontend changes
1) UI entrypoint
   - Add a mic/voice icon button next to the chat input in `client/geist/src/Chat.tsx` (or the input component used there).
   - Button toggles “Recording…” state and manages the lifetime of the WebSocket connection.

2) Audio capture
   - Use `getUserMedia` to capture microphone; use `AudioWorklet` (preferred) or `MediaRecorder` to produce steady PCM frames.
   - Send frames to the WebSocket as binary messages.
   - Add a small ring buffer to ensure predictable frame sizes (e.g., 20–40ms per frame).

3) Streaming UX
   - While recording: show a ghost “User (voice)” message with partial transcript updates.
   - When a `transcript_final` arrives: finalize the user message in chat history.
   - When `assistant_text` starts: render a streaming assistant message; once complete, commit to history.
   - Audio playback: use WebAudio to play incoming `audio_chunk` frames seamlessly.

4) Error states
   - Handle permission errors, dropped WS connections, and provider errors with inline toasts.

### Agent integration details
- We will reuse `complete_text(...)` for Phase 1. This ensures chat sessions and history are maintained without changes to agent implementations. `complete_audio(...)` is not required for this path.
- For `OnlineAgent`, STT can be Whisper (online) or MMS (local) per configuration; for `LocalAgent`, prefer MMS.
- The voice session service calls `complete_text` at phrase boundaries and handles TTS output streaming.

### Potential BaseAgent extensions (not required for MVP)
- Add a non-abstract optional method for token streaming without breaking subclasses:

```python
class BaseAgent(ABC):
    # existing methods...

    def stream_complete_text(self, prompt: str, **kwargs) -> "Iterator[str]":
        """Optional: yield tokens or text deltas. Default: not implemented."""
        raise NotImplementedError("Token streaming not supported for this agent.")
```

- Rationale: keeps MVP minimal (use existing `complete_text`) while providing a clear extension point for Phase 2.

### Data formats over WebSocket
- Handshake (JSON): `{ "session_id": number, "agent_type": "online"|"local", "stt_provider"?: "mms"|"whisper", "tts_provider"?: "sesame" }`
- Client → Server audio frames: binary payloads (PCM or Opus), constant frame size, mono, 16kHz.
- Server → Client messages:
  - `{ type: "transcript_partial", text: string }`
  - `{ type: "transcript_final", text: string }`
  - `{ type: "assistant_text", status: "start"|"chunk"|"end", text?: string }`
  - `audio_chunk` frames as binary payloads for playback
  - `{ type: "done" }`

### Dependencies and performance
- Prefer PCM to avoid new binary dependencies; if Opus is needed, reuse Moshi’s `sphn` helpers already vendored under `agents/architectures/moshi`.
- For MMS STT, maintain 16k sample rate; downsample on server if needed.
- Batch STT windows to ~1s to balance latency and accuracy; simple VAD (RMS threshold) is acceptable for MVP.

### Telemetry and logging
- Add a dedicated logger (use `__name__`) in the voice session service.
- Log session start/stop, VAD events, STT durations, LLM latency, TTS latency.

### Testing strategy
- Unit tests for the voice session service: segmentation, STT calls, `complete_text` invocation, and TTS stub.
- WebSocket integration test (loopback): feed a short PCM clip → expect transcript + assistant text + non-empty audio chunks.
- Manual E2E: `docker compose up -d`, open the client, click the mic, speak, verify streaming transcript and audible response; `curl` fallback for `/agent/voice/upload`.

### Rollout plan
1) Land backend WS endpoint + service with PCM-only MVP and Sesame TTS (or stub returning silence) to validate streaming contract.
2) Integrate frontend mic button and basic streaming UX.
3) Add configuration toggles and sensible defaults.
4) Optimize STT windows/VAD; consider token streaming in Phase 2.

### Out of scope (for now)
- Token-level LLM streaming across all agents.
- Advanced VAD/ASR diarization; multi-speaker handling.
- Mobile-specific audio worklets and echo cancellation tuning beyond browser defaults.

### Open questions
- Should we support client-side TTS fallback (Web Speech API) if server TTS is unavailable? If yes, add a setting to return `assistant_text` only and let the client synthesize.
- Do we want to reuse Moshi’s Opus pipeline end-to-end for lower bandwidth, or keep PCM for simplicity?


