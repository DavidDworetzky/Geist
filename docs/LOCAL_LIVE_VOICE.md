# Local live voice

Select **Live chat → Voice settings → Local live voice**. The browser always
connects through Geist's authenticated `/api/v1/voice/live/local` endpoint.
GET on that path reports the selected model and audio format; a WebSocket on the
same path carries the conversation. No text agent, separate speech recognizer,
or TTS chain is invoked.

## Bundled Moshi (default)

No endpoint configuration is necessary. Geist runs the isolated Moshi MLX worker
on Apple Silicon, with 24 kHz mono audio and 1,920-sample (80 ms) frames.
See [Moshi setup](../scripts/runtimes/moshi/README.md) for installation and weights.
`GEIST_MOSHI_PYTHON` still overrides the worker interpreter.

## Connect another local voice server

Set these environment variables on the **backend process**, then restart it:

```sh
export GEIST_LOCAL_LIVE_VOICE_BACKEND=websocket
export GEIST_LOCAL_LIVE_VOICE_URL=ws://127.0.0.1:8998/voice
export GEIST_LOCAL_LIVE_VOICE_MODEL=my-local-voice-model
export GEIST_LOCAL_LIVE_VOICE_SAMPLE_RATE=24000
export GEIST_LOCAL_LIVE_VOICE_FRAME_SAMPLES=1920
```

The model name is a catalog label. Configure the actual model, voice and generation
settings in the external server. `URL` is required; other values above are the
default audio format. Supported rates are 16,000, 24,000 and 48,000 Hz, with frames
between 10 and 100 ms. Example: 16,000 Hz and 320 samples gives 20 ms frames.
Browsers must support the configured AudioContext rate; Geist rejects mismatches.

Use `ws://` or `wss://`. URLs containing user credentials or fragments are rejected.
In Docker, the address is resolved from the backend container: use
`host.docker.internal` for a voice server running on the Mac host. Supply these
variables to the container explicitly if using Compose; exporting them alone does
not add variables to an existing container. Geist never forwards its operator
credentials or browser-supplied target URLs to the voice server. The upstream URL
is kept server-side and ambient HTTP proxy settings are not used for this connection.

To return to Moshi, set `GEIST_LOCAL_LIVE_VOICE_BACKEND=moshi` (or unset it) and
restart the backend. The external URL/audio settings are ignored in Moshi mode.

## External server wire protocol: `geist-pcm-v1`

The upstream must implement this protocol, either natively or through an adapter.
A generic WebSocket URL alone does not make an Opus-based server, the stock Moshi
web demo, or another vendor's protocol compatible.

1. Accept a WebSocket and initialize the voice model. Send this JSON text message
   before any audio (values must match the configured audio format):

   ```json
   {"type":"ready","protocol":"geist-pcm-v1","sample_rate":24000,"frame_samples":1920}
   ```

2. Receive and send **binary** messages containing one fixed-size frame of mono,
   little-endian float32 PCM (four bytes per sample, normalized to -1 through 1).
   At the defaults each message is exactly 7,680 bytes. There is no WAV header,
   base64 encoding or per-frame JSON header. Both directions use the same format.

3. Optionally send text deltas in JSON text messages:

   ```json
   {"type":"transcript","role":"assistant","text":" Hello"}
   {"type":"transcript","role":"user","text":"Hi there"}
   ```

   Roles are `assistant` or `user`; omitted role defaults to `assistant`. Deltas
   append to the corresponding captions. Text is limited to 4,000 characters per
   message (with a 24 KiB WebSocket message limit). The relay does not run an extra
   transcription model.

4. Close the socket on failure, or send `{"type":"error"}`. Geist returns a generic
   client error and logs relay exceptions. Close/release model resources when
   Geist disconnects. No extra stop message is required.

Geist allows ten seconds to connect, 120 seconds for readiness, 20 seconds between
upstream events, and a five-minute call. Queues/message sizes and audio writes are
bounded; stale audio ends the call instead of accumulating. Disconnecting the
browser also cancels a pending upstream connection. Moshi has its own bounded
worker queue and one-call-at-a-time GPU guard behind this common endpoint.

The OpenAI GPT-Live WebRTC option and dictation mode use their existing transports.
