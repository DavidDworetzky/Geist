# Dictation and GPT-Live voice calls

Separate voice use into two explicit modes. Dictation captures mono 16 kHz audio
on an AudioWorklet, transcribes on the server thread pool, and inserts the result
into the normal composer without running a language model or submitting a turn.
Keep MMS and Whisper transcription options. Bound recordings to two minutes.

Live chat offers only conversational voice models, starting with OpenAI GPT-Live
(`gpt-live-1`). Use direct browser-to-OpenAI WebRTC media and a data channel for
transcript events and session commands. A bounded backend endpoint exchanges the
SDP offer for a session with the server API key. Live calls do not invoke Geist's
text agents, separate STT, TTS, or tool pipelines. Show user and assistant captions
independently and pulse the existing brand mark from native audio analyser levels.

Keep existing chained voice endpoints for compatibility but remove them from the
composer's two-mode flow. No root dependency or database schema changes are needed.

Stop with session.close, wait for session.closed with a bounded timeout, and
release tracks, timers and transports on errors or unmount. Dictation cancellation
must not insert late results. Preserve unsent typed drafts when switching modes.

Validation: focused backend API and service tests, frontend lifecycle and UI tests,
type/lint checks, Docker startup/log/curl and browser smoke. Real GPT-Live audio
requires an API project with access plus microphone permission. Moshi also requires
native MLX startup and a real audio-generation check after runtime setup.

References:
- https://developers.openai.com/api/docs/guides/voice-webrtc?api=live
- https://developers.openai.com/api/docs/guides/live-conversations

## Moshi local voice

Add Moshiko 7B 4-bit MLX as a live voice provider, outside the text-only runner
registry. A dedicated Python process hosts Moshi and Mimi, reads 80 ms float32
PCM frames, and returns audio plus the model's spoken text. The FastAPI WebSocket
only relays frames; no model inference runs on its event loop. The browser uses
an AudioWorklet for capture and playback. Bound queues, sessions and concurrent
calls; terminate the worker on disconnect to release GPU memory.

Pin the official model revision and provide download/copy paths. Moshi requires
an isolated dependency set, so use a separately locked worker project rather
than changing the root dependencies. Override the upstream MLX and SentencePiece
pins with tested patched releases. Validate native loading and audio generation.
Moshi only emits assistant captions; no additional STT runs in its audio path.

## Verification on the PR branch

Based on current `origin/main`, isolated from the original workspace's unrelated
Tools/MCP and Pitchblend work. The voice port preserves main's operator-authenticated
WebSocket proxy, current brand image, and chat loading/error presentation.

- Installed `moshi-mlx==0.3.0`, `mlx==0.31.2`, and `sentencepiece==0.2.1` in the separate worker environment with `uv sync --locked --project scripts/runtimes/moshi --python 3.11`. Downloaded all three pinned Moshi/Mimi artifacts with `.venv/bin/python scripts/download_models.py --voice_model moshi-mlx-q4`.
- Audited all 20 worker registry packages with OSV querybatch: no reported advisories. The main application dependency files are unchanged.
- Native worker test: synthetic spoken question, 375 frames at 80 ms, all processed; 29.92 s of generated audio, first audio 173 ms after input started, maximum processing lag 313 ms. Mimi playback decoding runs concurrently with the next generation step. This is a streaming measurement on the development Mac, not end-of-turn response latency.
- Full native authenticated `/api/v1/voice/moshi` socket: 375 input frames, 29.92 s audio output, 159 ms to first output after readiness, assistant captions, and successful reconnect after disconnect. Used synthetic speech without accessing the microphone. No text agent or TTS request was made.
- Native `make run MLX_BACKEND=1` started successfully with `UV_NO_SYNC=1`, isolated SQLite data and operator credentials. The authenticated socket test also used an isolated Uvicorn process on port 5122. Native smoke servers were stopped after validation.
- Docker: `docker compose -p geist-live-voice-pr -f docker-compose.yml -f /tmp/geist-voice-pr-compose.yml up -d --no-build` succeeded with existing smoke images, current source mounts and ports 5121/3120. Backend startup and frontend compilation succeeded. Existing app on port 3000 was preserved.
- Docker: `/opt/venv/bin/python -m pytest -q tests/api/test_live_voice_api.py tests/services/test_dictation.py tests/services/test_moshi_voice.py tests/scripts/test_download_models.py`: **29 passed, 1 skipped**. Skip is the legacy TTS catalog check because the image lacks torchaudio; the prior native TTS/download run passed 13 tests.
- Frontend, from `client/geist`: `CI=true npm test -- --watchAll=false --runInBand --runTestsByPath src/Utils/__tests__/audioSessions.test.ts src/Utils/__tests__/liveVoiceSession.test.ts src/Components/__tests__/EnhancedChatInput.voice.test.tsx src/Components/__tests__/VoiceSettings.test.tsx src/__tests__/Chat.test.tsx src/Components/__tests__/ChatTextArea.test.tsx`: **42 passed**. Existing CRA/Babel warning remains.
- `./node_modules/.bin/tsc --noEmit`, targeted Ruff, and `git diff --check` passed. `.venv/bin/mypy --config-file=pyproject.toml app/models/live_voice.py app/services/live_voice.py app/services/dictation.py app/services/moshi_voice.py app/services/voice_catalog.py app/api/v1/endpoints/voice.py` passed.
- Browser: verified both modes, GPT-Live/Moshi selection, current branding, desktop and narrow-screen layout, no console errors, and live voice enabled when the separate text model is not installed. Text chat reached its missing-model state; actual text inference was not run.
- GPT-Live has mocked API/media lifecycle coverage; no real paid OpenAI media call was made. Real MMS dictation and the old chained voice service were not exercised because their optional inference dependencies are absent from the smoke image.

Commit checks: formatting, mypy, Bandit and staged secret scanning pass. The full
frontend ESLint hook reports 83 pre-existing errors in nine unchanged test files;
JSON lint results confirm zero errors in changed files. Only that full-repository
frontend hook is skipped for the commit; its failure is disclosed in the PR.
