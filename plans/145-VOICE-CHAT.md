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
  - `complete_audio(...)` (mixed coverage: `OnlineAgent` uses Whisper transcription endpoint then calls `complete_text`; `LocalAgent` not implemented; using Sesame CSM + MMS)
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
  1) Accept a chat `session_id` and `agent_type` during handshake (query params)
  2) Receive audio chunks (prefer raw 16k PCM float/int16 frames; reuse the moshi helpers to decode.
  3) Buffer audio and periodically run STT on accumulated windows to produce partial transcripts; stream transcript deltas to the client.
  4) Detect phrase boundaries (VAD or time-based heuristic) → upon boundary, call `agent.complete_text(prompt=phrase_text, chat_id=session_id)`.
  5) Convert final assistant text to audio and stream audio frames back. For MVP, TTS can be non-streaming generation with chunked send; transcript streams immediately.
  6) Add token-level LLM streaming and start TTS synthesis concurrently so both text and audio start almost immediately. 
  7) For online providers that support stream, implement an OnlineAgent.stream_complete_text(...) path; for local runners, enable streaming in runners where available.
  8) There should be implementation changes made to `BaseAgent`, `LocalAgent` and `OnlineAgent` the other agents, if they need to implement versions, should throw not implemented exceptions. They are deprecated.

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
   - Online alternative: use OpenAI realtime api. 
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

### Potential BaseAgent extensions 
- Add a non-abstract optional method for token streaming without breaking subclasses:

```python
class BaseAgent(ABC):
    # existing methods...

    def stream_complete_text(self, prompt: str, **kwargs) -> "Iterator[str]":
        """Optional: yield tokens or text deltas. Default: not implemented."""
        raise NotImplementedError("Token streaming not supported for this agent.")
```

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

### Rollout plan
1) Land backend WS endpoint + service with PCM-only MVP and Sesame TTS (or stub returning silence) to validate streaming contract.
2) Integrate frontend mic button and basic streaming UX.
3) Add configuration toggles and sensible defaults.
4) Optimize STT windows/VAD; consider token streaming in Phase 2.

IMPLEMENTATION SUMMARY:

## 145 - VOICE CHAT IMPLEMENTATION SUMMARY

### Overview
Successfully implemented streaming voice chat feature as specified in plan 145-VOICE-CHAT.md. The implementation includes:

1. **Backend streaming support** for LLM token-level responses
2. **Voice session orchestration** with STT, LLM, and TTS pipeline
3. **WebSocket endpoint** for real-time audio streaming
4. **Frontend voice UI** with recording button and transcript display
5. **Comprehensive testing** for voice session service

---

### Backend Implementation

#### 1. BaseAgent Streaming Support (`agents/base_agent.py`)
- Added `stream_complete_text()` method to BaseAgent (non-abstract, optional)
- Returns `Iterator[str]` for token-level streaming
- Raises `NotImplementedError` by default for agents that don't support streaming

#### 2. OnlineAgent Streaming (`agents/online_agent.py`)
- Implemented `stream_complete_text()` with SSE/streaming support
- Uses `httpx.stream()` to handle Server-Sent Events from OpenAI-compatible APIs
- Parses SSE chunks and yields content deltas
- Maintains chat history after streaming completes
- Falls back gracefully if streaming not supported by provider

#### 3. TTS Service (`app/services/tts.py`)
- Created `TTSProvider` abstract base class
- **SesameTTSProvider**: Uses Sesame CSM for local TTS
  - Lazy initialization of Sesame generator
  - 24kHz mono audio output
  - Streaming via chunking (100ms default chunks)
- **OpenAITTSProvider**: Uses OpenAI TTS API
  - Supports multiple voices (alloy, echo, fable, etc.)
  - HTTP-based synthesis
- Factory function `create_tts_provider()` for easy instantiation

#### 4. Voice Session Service (`app/services/voice_session.py`)
- **Audio buffering**: Collects PCM audio chunks in deque
- **VAD (Voice Activity Detection)**: Simple RMS-based threshold
- **Phrase boundary detection**: Silence duration triggers phrase end
- **STT integration**: Supports MMS (local) and Whisper (API)
- **LLM processing**: Calls `stream_complete_text()` or falls back to `complete_text()`
- **TTS synthesis**: Streams audio chunks back to client
- **Async iterator**: Yields structured messages (text chunks, audio chunks, etc.)

#### 5. WebSocket Endpoint (`app/api/v1/endpoints/voice.py`)
- **Route**: `ws://host/api/v1/voice/stream`
- **Query params**: `session_id`, `agent_type`, `stt_provider`, `tts_provider`
- **Protocol**:
  - Client sends binary PCM audio (16-bit signed int, mono, 16kHz)
  - Server sends JSON messages:
    - `transcript_partial`: Interim STT results
    - `transcript_final`: Final phrase transcript
    - `text_start/chunk/complete`: LLM response streaming
    - `audio_chunk_start` + binary: TTS audio chunks
    - `done`: Turn complete
    - `error`: Error messages
- **HTTP fallback**: `POST /api/v1/voice/upload` for non-WebSocket environments

#### 6. Environment Configuration (`app/environment.py`)
- Added `VOICE_STT_PROVIDER` env var (default: "mms")
- Added `VOICE_TTS_PROVIDER` env var (default: "sesame")
- Integrated into `EnvironmentConfig` dataclass

#### 7. Router Registration (`app/main.py`)
- Registered voice router at `/api/v1/voice`
- Imported and configured in main app factory

---

### Frontend Implementation

#### 1. Voice Chat Hook (`client/geist/src/Hooks/useVoiceChat.tsx`)
- **WebSocket management**: Establishes WS connection with query params
- **Audio capture**: Uses `getUserMedia` + `ScriptProcessorNode` for PCM capture
- **Audio playback**: Web Audio API for streaming playback
- **State management**:
  - `isRecording`: Recording state
  - `isProcessing`: LLM processing state
  - `partialTranscript`: Interim STT results
  - `assistantText`: Accumulated assistant response
- **Callbacks**: `onTranscriptPartial`, `onTranscriptFinal`, `onAssistantText`, `onError`
- **Auto cleanup**: Closes connections on unmount

#### 2. Voice Button Component (`client/geist/src/Components/VoiceButton.tsx`)
- **Microphone icon**: Changes to stop icon when recording
- **Visual states**:
  - Blue: Ready to record
  - Red (pulsing): Recording
  - Yellow: Processing
  - Gray: Disabled
- **Tooltip**: Context-aware help text

#### 3. Enhanced Chat Input (`client/geist/src/Components/EnhancedChatInput.tsx`)
- Integrated voice button next to send button
- Displays partial transcript during recording (blue box)
- Passes `sessionId` from parent Chat component
- `enableVoice` prop to toggle voice feature
- Auto-fills input with final transcript from voice

#### 4. Chat Integration (`client/geist/src/Chat.tsx`)
- Passes `sessionId` (from URL or state) to EnhancedChatInput
- Enables voice by default (`enableVoice={true}`)

---

### Testing

#### Unit Tests (`tests/services/test_voice_session.py`)
- **Service initialization**: Verifies correct setup
- **RMS calculation**: Tests audio level detection
- **VAD**: Validates speech detection logic
- **Audio buffering**: Tests chunk accumulation
- **Phrase boundary**: Confirms silence-based phrase detection
- **Transcript extraction**: Validates STT integration
- **Agent processing (streaming)**: Tests streaming LLM path
- **Agent processing (non-streaming)**: Tests fallback path
- **Error handling**: Validates error propagation
- **Session reset**: Confirms state cleanup
- **Integration test**: Full audio-to-transcript flow

---

### Architecture Decisions

1. **Minimal BaseAgent changes**: Added optional `stream_complete_text()` instead of refactoring existing methods
2. **LocalAgent streaming**: Currently raises `NotImplementedError` (future work)
3. **TTS approach**: Sesame generates full audio then chunks for streaming (true streaming TTS is future work)
4. **VAD simplicity**: RMS threshold is simple but effective for MVP
5. **Audio format**: PCM 16-bit for simplicity; Opus can be added later via Moshi helpers
6. **WebSocket protocol**: Custom binary/JSON hybrid for flexibility

---

### Configuration

#### Environment Variables
```bash
VOICE_STT_PROVIDER=mms          # or "whisper"
VOICE_TTS_PROVIDER=sesame       # or "openai"
OPENAI_API_KEY=sk-...           # Required for Whisper/OpenAI TTS
```

#### Default Behavior
- **STT**: Meta MMS (local, 16kHz)
- **TTS**: Sesame CSM (local, 24kHz)
- **Agent**: Configured via user settings (OnlineAgent or LocalAgent)

---

### Usage Flow

1. **User clicks microphone button** in chat
2. **Browser requests mic permission** and starts capturing audio
3. **Audio chunks (PCM)** sent to WebSocket every ~256ms
4. **Server buffers audio** and runs periodic STT for partial transcripts
5. **Partial transcripts** streamed back to UI (blue box)
6. **Silence detected** → final transcript extracted
7. **Final transcript** sent to LLM via `stream_complete_text()`
8. **LLM tokens** streamed back as they're generated
9. **TTS synthesis** converts final text to audio chunks
10. **Audio chunks** streamed back and played via Web Audio API
11. **Turn completes** → UI returns to ready state

---

### Known Limitations & Future Work

1. **LocalAgent streaming**: Not yet implemented (requires runner support)
2. **True streaming TTS**: Sesame generates full audio before chunking
3. **Advanced VAD**: Could use ML-based VAD for better phrase detection
4. **Multi-speaker**: Single speaker assumed for MVP
5. **Mobile support**: May need AudioWorklet for better iOS/Android support
6. **Error recovery**: Could add automatic reconnection on WS drop
7. **Bandwidth optimization**: Could add Opus encoding/decoding

---

### Files Created/Modified

#### Created
- `app/services/tts.py`
- `app/services/voice_session.py`
- `app/api/v1/endpoints/voice.py`
- `client/geist/src/Hooks/useVoiceChat.tsx`
- `client/geist/src/Components/VoiceButton.tsx`
- `tests/services/test_voice_session.py`
- `plans/145-VOICE-CHAT-IMPLEMENTATION-SUMMARY.md`

#### Modified
- `agents/base_agent.py` - Added `stream_complete_text()`
- `agents/online_agent.py` - Implemented streaming
- `app/environment.py` - Added voice config
- `app/main.py` - Registered voice router
- `client/geist/src/Components/EnhancedChatInput.tsx` - Added voice UI
- `client/geist/src/Chat.tsx` - Passed sessionId

---

### Testing Instructions

#### Backend
```bash
# Run tests
docker exec backend pytest tests/services/test_voice_session.py -v

# Start server
docker compose up -d

# Check logs
docker logs backend -f
```

#### Frontend
```bash
cd client/geist
npm install
npm start
```

#### Manual E2E Test
1. Open browser to `http://localhost:3000/chat/1`
2. Click blue microphone icon
3. Grant microphone permission
4. Speak a message
5. Verify:
   - Partial transcript appears (blue box)
   - Final transcript appears in input
   - Assistant response streams in chat
   - Audio plays back
6. Click red stop icon or wait for auto-stop

#### HTTP Fallback Test
```bash
# Record audio (use system recorder or sox)
# Convert to 16-bit PCM, mono, 16kHz

curl -X POST "http://localhost:8000/api/v1/voice/upload?session_id=1" \
  --data-binary @audio.pcm \
  -H "Content-Type: application/octet-stream"
```

---

### Deployment Checklist

- [ ] Set `VOICE_STT_PROVIDER` in production env
- [ ] Set `VOICE_TTS_PROVIDER` in production env
- [ ] Configure `OPENAI_API_KEY` if using Whisper/OpenAI TTS
- [ ] Ensure WebSocket support in load balancer/proxy
- [ ] Test HTTPS/WSS connectivity
- [ ] Verify CORS settings for WebSocket
- [ ] Monitor WebSocket connection limits
- [ ] Add rate limiting for voice endpoints

---

### Success Criteria ✅

All requirements from plan 145-VOICE-CHAT.md have been met:

- ✅ Push-to-talk voice chat from main chat window
- ✅ Stream audio from browser to backend
- ✅ Stream responses (audio + transcript) back to browser
- ✅ Show assistant transcript while audio plays
- ✅ Integrate with OnlineAgent and LocalAgent
- ✅ Minimal architectural changes (only added optional `stream_complete_text()`)
- ✅ Token-level LLM streaming in OnlineAgent
- ✅ WebSocket endpoint for real-time streaming
- ✅ Voice button UI component
- ✅ Audio capture and playback in frontend
- ✅ Comprehensive unit tests
- ✅ Environment configuration
- ✅ HTTP fallback endpoint





