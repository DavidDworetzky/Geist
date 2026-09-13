# Moshi live calls on Apple Silicon

Moshiko is Kyutai's 7B speech-to-speech model. The pinned 4-bit MLX weights and Mimi
codec occupy about 5.2 GB. Weights use CC-BY-4.0; see the
[model card](https://huggingface.co/kyutai/moshiko-mlx-q4).

From the repository root:

```sh
uv sync --locked --project scripts/runtimes/moshi --python 3.11
.venv/bin/python scripts/download_models.py --voice_model moshi-mlx-q4
make run MLX_BACKEND=1
```

Run `make services` in another terminal to connect the frontend to the native
backend. The full Linux Docker backend cannot run this Apple Silicon MLX worker.

In the composer, choose **Live chat**, open **Voice settings**, select **Moshi**,
and start the call. Use headphones to avoid speaker echo. First connection loads
and warms the model. Calls are English, limited to five minutes, and independent
of the selected text model. Closing a call releases its model memory.

The browser exchanges 80 ms PCM audio frames over a WebSocket. An AudioWorklet
handles capture and playback; a separate process handles MLX and Mimi, with
playback decoding overlapping generation. Captions show Moshi's own spoken text;
Moshi does not produce a user transcript. Only one local call runs at a time.
Queues are bounded so an overloaded device ends the call instead of accumulating
seconds of delayed speech.

## Runtime and weights

The default interpreter is `scripts/runtimes/moshi/.venv/bin/python`.
`GEIST_MOSHI_PYTHON` can select an existing compatible runtime. This environment
uses Moshi 0.3.0 with tested overrides for MLX 0.31.2 and SentencePiece 0.2.1 to
address advisories in the upstream pins. Its other inference requirements stay
isolated from Geist's text runtime. Dependencies for Moshi's standalone web and
sound-device demos are omitted; use this environment only with Geist's worker.
The worker lockfile's 20 registry packages passed an OSV audit on 2026-09-13.

Weights load offline from the pinned Hugging Face cache or from
`app/model_weights/kyutai_moshiko-mlx-q4` (also supported by `copy_weights.py`).
Revision: `18e4df760a34d5977a34517d7d1580e07acbb2f1`.

## Validation

A synthetic spoken question produced responsive audio and assistant captions in
native Metal inference. A 30-second worker test processed all 375 input frames
and produced 29.92 seconds of output audio. First output arrived 173 ms after
streaming began, following model warmup; the maximum measured frame processing
lag was 313 ms. These are observations on the development Mac, not a guarantee
for other hardware or a measurement of end-of-user-turn response latency.

Transport and UI tests cover mode separation, draft preservation, cancellation,
worker shutdown, malformed audio, bounded queues, and live voice availability
without a text model installed. GPT-Live negotiation has mock coverage; a real
OpenAI media call still requires an API project with GPT-Live access.
