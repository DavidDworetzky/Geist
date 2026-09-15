# Open-weight model support

Geist separates model identity from inference backend. Standard Hugging Face
causal language models use the generic `transformers` runner; heavyweight
models use any OpenAI-compatible hosted or self-hosted endpoint. The supported
catalog lives in `agents/model_catalog.py`.

## Baseline families

Local reference checkpoints cover Llama, Qwen 2.5/3, Mistral, Phi, SmolLM,
Gemma text, Granite, OLMo, GLM 4 9B Chat HF, gpt-oss, and DeepSeek distillations.
Gemini 3.8 Flash is available only as a hosted API model. Kimi K2.5, GLM 4.7
Flash/5.2, full DeepSeek R1, Llama 70B, Qwen 72B, Mixtral 8x7B, gpt-oss 120B,
and OpenRouter's GLM 5.3 Flash, Grok 4.6, Qwen 3.8 Flash, DeepSeek V4.1 Flash,
and Sakana Fugu Ultra v2 routes are also intentionally server-backed. For models with published
weights, their total resident weights make an in-process laptop load impractical
even when their mixture-of-experts active-parameter count is much smaller.
The retired anonymous `stealth/ox-alpha` preview has been replaced by its
stable `z-ai/glm-5.3-flash` release.
Muse Spark 1.2 Contributor is likewise hosted-only, but Meta does not disclose
its parameter count or a fixed maximum output limit, so the catalog leaves
both fields unset.

## Adding a model or provider

1. Add a `ModelSpec` to `agents/model_catalog.py`.
2. For a new OpenAI-compatible provider, add one `ProviderSpec` with its base
   URL and API-key environment variable.
3. Add runner code only when the model is not a standard causal LM (for
   example, native multimodal processor input).

Unknown instruction-tuned Hugging Face causal models default to the generic
runner. Explicit `runner_type` settings continue to override catalog routing.
As a memory-safety default, conventional IDs indicating 32B or more total
parameters—including MoE names such as `8x22B`—route to the self-hosted
provider. An explicit local `runner_type` override opts capable machines back
into in-process loading.
Catalog providers use string IDs, so adding one does not require extending the
legacy `OnlineModelProviders` enum or changing the model API routes.

## Google Gemini 3.8 Flash

Set `GEMINI_API_KEY` and select provider `google` with model
`gemini-3.8-flash`. The model has a 1,048,576-token context window and
65,536-token output limit, and supports multimodal input, reasoning, streaming,
function calling, and structured outputs. Geist currently sends text-only chat
and function-call payloads through the compatibility endpoint.
The existing `GEMINI_BASE_URL` override applies only to image generation;
Gemini chat continues to use the cataloged compatibility endpoint.

Geist routes it through Google's OpenAI-compatible endpoint. That compatibility
API is still beta and covers Geist's chat and function-calling path, but
Gemini-native server tools such as Google Search grounding require a future
direct Gemini API integration. Geist follows Google's
[Gemini 3.8 migration checklist](https://ai.google.dev/gemini-api/docs/generate-content/latest-model)
by omitting `n`, `temperature`, and `top_p` for this model.
Sibling IDs such as `gemini-3.8-flash-lite` route to Google but keep their own
sampling parameters. Add new pointer forms for the same model, such as dated
snapshots, to the catalog's `aliases` tuple when they share this request contract.

Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms) distinguish
unpaid and paid usage. Unpaid-service prompts and responses may be used to
improve Google products and may be processed by human reviewers; do not send
confidential data through unpaid quota. Paid-service prompts and responses are
not used for product improvement, though limited safety logging still applies.

## OpenRouter-hosted Grok 4.6

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`x-ai/grok-4.6`. The model uses OpenRouter's OpenAI-compatible chat endpoint,
retains native function calling, and cannot fall through to a local runner.
Grok 4.6 always reasons; Geist sends its documented default `high` effort and
omits unsupported frequency penalty, presence penalty, and stop parameters.

OpenRouter does not retain prompt or response content unless logging is
explicitly enabled, but downstream provider policy still applies. Enable
OpenRouter's Zero Data Retention routing for confidential workloads so the
request can use only eligible provider endpoints.

## Meta-hosted Muse Spark

Set `MODEL_API_KEY` and select provider `meta`. Geist offers the direct Meta
Model API IDs `muse-spark-1.1`, `muse-spark-1.2`, and the recommended
`muse-spark-1.3`.
Meta Model API exposes an OpenAI-compatible Chat Completions endpoint at
`https://api.meta.ai/v1` with a 1,048,576-token context window, multimodal
input, streaming, reasoning-capable responses, and native function calling.
Geist currently uses Meta's default reasoning effort. Muse Spark is hosted-only;
Meta describes open weights as future work.

The Contributor tier remains under provider `openrouter`. Although Meta lists
the tier, direct Chat Completions availability has not been reliable enough to
make it a first-party provider option in Geist.

## OpenRouter-hosted Qwen3.8 Flash

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`qwen/qwen3.8-flash`. OpenRouter added the stable route on August 26, 2026. It
accepts text, image, and video input, has a 1,000,000-token context window and
131,072-token output limit, and supports optional reasoning, streaming, native
function calling, and JSON-schema structured outputs. Geist omits the
unsupported `n` parameter. OpenRouter lists the current price as $0.15 per
million input tokens, $0.47 per million output tokens, and $0.016 per million
cached input tokens.

The route currently has one Alibaba endpoint, so there is no provider fallback
diversity. Alibaba states that Model Studio API data is not used for model
training, while OpenRouter does not retain prompt or response content unless
logging is explicitly enabled. However, this exact endpoint is absent from
OpenRouter's ZDR endpoint list as of August 27, 2026. Do not send confidential
workloads to this model. The hosted production model is based on the open-weight
Qwen3.8-Flash-Next release, but Geist does not assume that their parameter
metadata is identical.

## OpenRouter-hosted Tencent Hy4 Preview

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`tencent/hy4-preview`. Tencent and OpenRouter released this preview route on
August 28, 2026. It accepts text input, has a 1,048,576-token context window
and 64,000-token output limit, and supports optional reasoning, streaming,
native function calling, and structured outputs. Geist omits the unsupported
`n`, `top_p`, frequency-penalty, and presence-penalty parameters. OpenRouter
lists the current price as $0.834 per million input tokens, $2.501 per million
output tokens, and $0.042 per million cached input tokens.

Tencent discloses 770B total and 49B activated parameters and publishes the
weights under Apache-2.0. The OpenRouter route currently has one Tencent FP8
endpoint, so there is no provider fallback diversity and preview behavior may
change. As of August 28, 2026, OpenRouter identifies the Tencent endpoint as
zero retention and not used for training. OpenRouter itself does not retain
prompt or response content unless logging is explicitly enabled. Enforce ZDR
routing and re-check the endpoint policy before sending confidential workloads.

## OpenRouter-hosted DeepSeek V4.1 Flash

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`deepseek/deepseek-v4.1-flash`. OpenRouter added this stable route on September
10, 2026. It accepts text and image input, has a 1,048,576-token context window
and advertised 384,000-token maximum output, and supports optional reasoning,
streaming, native function calling, and response-format JSON. JSON-schema
enforcement varies by upstream. Geist omits the unsupported `n` parameter and
otherwise preserves the route's supported sampling and tool parameters.

OpenRouter listed four upstreams at review time. Its routed three-day
availability was 99.92%, with a 134-token/second best-provider median and a
1.40-second best-provider median latency. The displayed off-peak prices were
$0.15 per million input tokens, $0.60 per million output tokens, and $0.003 per
million cached input tokens; weekday peak windows double those rates.

DeepSeek's launch results include 90.6 on Terminal-Bench 2.1, 74.2 on DeepSWE
v1.1, and 54.8 on AutomationBench at maximum reasoning effort. Those are
provider-run results, even though the model card publishes harness settings and
reproduction instructions. Independent launch-day benchmarking was not yet
available, and early hands-on reports were mixed, so qualify the model on your
own workloads before making it a default.

OpenRouter itself does not retain prompt or response content unless logging is
explicitly enabled. The first-party DeepSeek endpoint, however, retains prompts
and may train on them. DeepInfra, Novita, and Venice instances of this exact
route appeared in OpenRouter's ZDR inventory on September 10, 2026. Enforce
OpenRouter ZDR routing before sending confidential workloads; default routing
does not make that guarantee.

## OpenRouter-hosted GLM 5.3 Flash

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`z-ai/glm-5.3-flash`. The stable route supports a 1,048,576-token context,
131,072-token maximum output, image and video input, streaming, response-format
JSON, and native function calling. Reasoning cannot be disabled; Geist sends
Z.ai's recommended `max` effort.

OpenRouter may route this model across providers with different context limits,
supported parameters, and data policies. Enable OpenRouter Zero Data Retention
routing for confidential workloads and retain normal retry handling for
provider availability changes.

## OpenRouter-hosted Sakana Fugu Ultra v2

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`sakana/fugu-ultra-v2`. OpenRouter added the stable route on September 11,
2026. It accepts text, image, and file inputs, has a 1,000,000-token context
window and a 128,000-token maximum output, and supports streaming, native
function calling, automatic tool choice, structured outputs, and built-in web
search. Fugu Ultra v2 always reasons; Geist sends OpenRouter's documented
default `xhigh` effort. Because the route does not advertise Geist's standard
generation controls, Geist omits `max_tokens`, `n`, temperature, `top_p`,
frequency penalty, presence penalty, and stop before dispatch while preserving
native tool definitions.

OpenRouter lists prices of $5 per million input tokens, $30 per million output
tokens, and $0.50 per million cached input tokens. Prompts above 272,000 tokens
use higher $10/$45/$1 rates. At review time the single Sakana endpoint showed
100% routed uptime but about 90% successful availability, a 69.6-second median
latency, and 68 output tokens per second.

Sakana's launch results report leading scores on agentic, coding, document,
and visual-reasoning evaluations, but independent v2 reproductions were not yet
available. Treat those results as provider claims and qualify the route on
representative workloads before making it a default. OpenRouter itself does
not retain prompt or response content unless logging is enabled, and its
provider table marks Sakana as not training on OpenRouter requests. Sakana
nonetheless retains prompts for an unspecified period, the worker pool is
fixed and opaque, and this endpoint is not ZDR. Do not use it for confidential
or regulated workloads.

## OpenRouter-hosted Muse Spark 1.2 Contributor

Set `OPENROUTER_API_KEY` and select provider `openrouter` with model
`meta/muse-spark-1.2-contributor`. The stable OpenRouter route supports a
1,048,576-token context, multimodal input, streaming, structured output, and
native function calling. Muse Spark always reasons; Geist supplies its
documented default `medium` effort and omits unsupported frequency penalty,
presence penalty, and stop parameters.

OpenRouter does not retain prompt or response content unless logging is
explicitly enabled. The sole upstream Meta endpoint, however, has a documented
30-day retention policy and is not on OpenRouter's ZDR endpoint list. Do not
send confidential workloads to this model.

Any self-hosted llama.cpp, vLLM, SGLang, or Transformers server can be used
without a provider entry by creating an online agent with its base endpoint and
served model name:

```python
AgentFactory.create_agent(
    "online",
    agent_context,
    model="zai-org/GLM-4.7-Flash",
    endpoint="http://inference-host:8000/v1",
)
```

The online agent appends `/chat/completions` when the supplied endpoint is a
base `/v1` URL.
For persisted self-hosted settings, set `OPENAI_COMPATIBLE_BASE_URL` to that
base URL and optionally set `API_KEY`.

## Local inference performance

The generic runner applies the following automatically:

- inference mode and `model.eval()` to disable gradient work;
- direct `model.generate()` rather than allocating a text-generation pipeline;
- BF16 on supported CUDA, FP16 on MPS/CUDA, and FP32 on CPU;
- automatic multi-GPU dispatch and low-memory loading through the pinned
  Accelerate runtime;
- one tokenizer/model instance per agent and generated-suffix-only decoding;
- no second full-model `.to(device)` copy after Accelerate dispatches weights.

Optional `device_config` keys are `device`, `dtype`, `device_map`,
`attn_implementation`, `quantization_config`, `load_in_4bit`, `load_in_8bit`,
`max_memory`, `offload_folder`, `offload_state_dict`, `use_safetensors`,
`compile`, `allow_tf32`, `revision`, `cache_dir`, `local_files_only`,
`subfolder`, `weights_dir`, `trust_remote_code`, and the advanced
`allow_server_backed` safety override.
Remote repository code is disabled by default. CUDA TF32 is enabled by default;
`compile` is opt-in because its startup cost only pays off for longer sessions.
The existing Meta Llama 3.1 ID continues to use Geist's MLX-specialized path;
an explicit `runner_type: transformers` override selects HF-native loading.

For tolerable laptop inference:

- Prefer 1B-4B models for unquantized execution.
- Prefer 4-bit MLX or GGUF weights for 7B and larger models.
- Treat active parameters as compute cost, not memory footprint: all MoE
  experts still need storage/residency.
- Run GLM 4.7 Flash, gpt-oss 20B, and larger models behind a local
  OpenAI-compatible llama.cpp, vLLM, or SGLang server when practical.
- Use `zai-org/glm-4-9b-chat-hf` for the Transformers-native local GLM
  baseline; it avoids remote repository code and supports the generic runner.
- Use non-thinking/instant modes when deep reasoning is unnecessary; this can
  reduce generated tokens and time-to-final-answer substantially.

Catalog `performance_note` values are returned by the model API so clients can
show hardware guidance before starting a large download.
The API also returns `optional_dependencies`; for example, gpt-oss advertises
the CUDA-only `kernels` accelerator without forcing that dependency into CPU or
Apple Silicon environments.
