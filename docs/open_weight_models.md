# Open-weight model support

Geist separates model identity from inference backend. Standard Hugging Face
causal language models use the generic `transformers` runner; heavyweight
models use any OpenAI-compatible hosted or self-hosted endpoint. The supported
catalog lives in `agents/model_catalog.py`.

## Baseline families

Local reference checkpoints cover Llama, Qwen 2.5/3, Mistral, Phi, SmolLM,
Gemma text, Granite, OLMo, GLM 4 9B Chat HF, gpt-oss, and DeepSeek distillations.
Kimi K2.5, GLM 4.7 Flash/5.2, full DeepSeek R1, Llama 70B, Qwen 72B, Mixtral
8x7B, and gpt-oss 120B are intentionally server-backed.
Their total resident weights make an in-process laptop load impractical even
when their mixture-of-experts active-parameter count is much smaller.

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
