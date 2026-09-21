# Configurable local live voice

Replace the Moshi-specific public socket with `/api/v1/voice/live/local`.
GET reports a typed browser transport contract; WebSocket serves the call.
The server chooses a backend using `GEIST_LOCAL_LIVE_VOICE_BACKEND` (`moshi`
by default, or `websocket`). Moshi remains a private managed-worker adapter.
The WebSocket adapter connects to `GEIST_LOCAL_LIVE_VOICE_URL`, configured by
the operator, never supplied by a browser request. Do not forward Geist auth.

The common v1 protocol is mono little-endian float32 PCM in fixed-size frames,
plus ready, transcript (user/assistant), and error JSON events. Configure sample
rate and frame samples for external endpoints, validate them, and publish them
through the GET contract. The browser AudioWorklet accepts frame size as an
option and verifies the actual AudioContext rate. Bound message sizes, queued
audio, readiness/inactivity waits, and call duration. Cancellation must close
upstream connections, including while connecting or waiting for readiness.

Use one generic frontend local voice session and a generic catalog provider.
Expose the selected engine and its limitations as catalog description, not a
Moshi branch in the UI. Existing online WebRTC and dictation remain separate.
No database migration or new dependency is needed. Server environment config is
the persistent source of truth, matching existing runtime configuration.

Validate typed configuration, routing, real upstream socket relay, invalid frames,
errors, slow readiness, disconnect cleanup, and both caption roles. Run frontend
lifecycle and worklet tests at different audio formats, Docker API/service tests,
real native Moshi streaming through the common endpoint, and browser smoke.
Document exact endpoint configuration and v1 wire protocol in the runtime README.

## Verification

- `docker compose -p geist-live-voice-pr -f docker-compose.yml -f /tmp/geist-voice-pr-compose.yml up -d --no-build`: existing smoke images and current source mounts started successfully on isolated ports 3120/5121.
- `docker exec geist-live-voice-pr-backend-1 /bin/bash -lc 'cd /opt/geist && PYTHONPATH=/opt/geist /opt/venv/bin/python -m pytest tests/services/test_local_live_voice.py tests/services/test_moshi_voice.py tests/api/test_live_voice_api.py tests/services/test_dictation.py -q'`: **55 passed**. Includes real upstream WebSocket relay, configuration guards, audio-format handshake, both caption roles, Unicode, auth isolation, disconnect cancellation, readiness/inactivity/call limits, Moshi, and dictation.
- From `client/geist`: `CI=true npm test -- --watchAll=false --runInBand --runTestsByPath src/Utils/__tests__/audioSessions.test.ts src/Utils/__tests__/localVoiceProcessor.test.ts src/Utils/__tests__/liveVoiceSession.test.ts src/Components/__tests__/VoiceSettings.test.tsx src/Components/__tests__/EnhancedChatInput.voice.test.tsx`: 30 cases covered. The initial run found two settings-selection failures caused by a typo; fixed it and reran `CI=true npm test -- --watchAll=false --runInBand --runTestsByPath src/Components/__tests__/VoiceSettings.test.tsx` with all five cases passing. All 30 cases pass across these focused runs.
- From `client/geist`: `./node_modules/.bin/tsc --noEmit` and `./node_modules/.bin/eslint src/Components/VoiceSettings.tsx src/Components/__tests__/VoiceSettings.test.tsx src/Hooks/useVoiceChat.tsx src/Hooks/useVoiceModels.tsx src/Utils/localVoiceSession.ts src/Utils/__tests__/audioSessions.test.ts src/Utils/__tests__/localVoiceProcessor.test.ts --max-warnings=0`: passed.
- Targeted pre-commit Ruff, formatting, mypy, Bandit and secret scanning passed. The known unrelated repository-wide frontend ESLint failures documented in PR #371 remain; only that hook is skipped when committing.
- `/tmp/geist-live-voice-pr/.venv/bin/python /tmp/geist_local_voice_socket_probe.py`: actual native MLX Moshi on the common authenticated endpoint, 375 frames over 30 seconds, 29.92 seconds returned audio, captions, and successful reconnect. First audio was 160 ms after readiness; this measures initial streaming, not end-of-turn latency. Probe used synthetic speech and a native Uvicorn process with `MLX_BACKEND=1`; the previously verified `make run` command's fixed port was avoided because other local services own it.
- `curl -fsS http://localhost:3120` and `curl -fsS http://localhost:3122`: HTTP 200. Docker startup logs had no error markers. Browser showed the generic local provider, selected Moshi, and enabled Start voice call, with no console errors. Native UI remains running for the user.
- Real text-model inference was not run: the isolated core app environment lacks mlx-lm, while Moshi has its own installed runtime. No paid OpenAI media call or real microphone recording was made in this change. External backend validation uses a real protocol test server, not a second downloaded voice model.
