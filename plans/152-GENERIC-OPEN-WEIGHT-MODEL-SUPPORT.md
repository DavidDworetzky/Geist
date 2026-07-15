# Generic Open-Weight Model Support

## Goal

Add extensible model support without creating one agent or runner per model brand. Separate model capabilities from inference backends so local Hugging Face models and OpenAI-compatible hosted/self-hosted models share one registry.

## Supported families

- Local causal language models: Llama, Qwen 2.5/3, Mistral, Phi, SmolLM, Gemma text, Granite, OLMo, GLM 4 9B, and compatible DeepSeek distillations.
- Specialized local models: gpt-oss through the same Transformers runner plus format/capability metadata.
- Server-backed heavyweight models: Kimi K2.5, GLM 5.2, full DeepSeek, and other OpenAI-compatible deployments.
- Multimodal models are advertised with capability metadata but are not falsely routed through the text-only local runner.

## Architecture

1. Introduce a model catalog containing stable metadata: family, provider, backend, context limits, vision/tools/reasoning support, gating, remote-code requirements, minimum Transformers version, and performance notes.
2. Add a generic `TransformersRunner` based on `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, and tokenizer chat templates.
3. Select local runners from catalog/backend metadata and conservative family rules. Preserve explicit runner overrides and the legacy MLX Llama runner.
4. Keep heavyweight models provider-neutral by routing them through the existing OpenAI-compatible `OnlineAgent`; publish default provider endpoints separately from model architecture metadata.
   Catalog provider IDs must not require adding another fixed enum member.
5. Make runner registration lazy so optional MLX, Torch, and model-specific dependencies are imported only when selected.
6. Extend model API responses and frontend types with backend/capability/performance metadata while remaining backward compatible.

## Performance strategy

- Prefer `device_map="auto"` on CUDA and avoid copying a fully loaded model with a second `.to(...)` pass.
- Use automatic dtype selection: BF16 where supported, FP16 on MPS/compatible CUDA, FP32 on CPU unless quantized.
- Enable `low_cpu_mem_usage`, inference mode, model evaluation mode, tokenizer/model reuse, and generated-suffix decoding without a high-level pipeline allocation.
- Support optional attention implementation and quantization settings through `device_config`, without adding mandatory dependencies.
- Recommend quantized GGUF/MLX or OpenAI-compatible local servers for models whose full weights are not laptop practical.
- Fail early with actionable messages for gated repositories, incompatible Transformers versions, unsafe remote-code requirements, or models too large for the selected backend.

## Implementation steps

1. Add the catalog and generic Transformers runner.
2. Refactor runner registration and factory routing to be lazy and capability-based.
3. Populate backend and API registries for all target families and heavyweight provider endpoints.
4. Update model synchronization, weight-copy helpers, and frontend fallback/type handling.
5. Add unit tests for catalog resolution, runner loading/generation, device/dtype behavior, routing, API serialization, and backward compatibility.
6. Run focused host tests, then the required Docker startup/log/curl validation.

## Acceptance criteria

- A new standard Hugging Face causal LM can be added primarily by one catalog entry.
- Qwen 2.5/3, Mistral, Phi, SmolLM, Gemma text, Granite, OLMo, and the Transformers-native GLM 4 9B checkpoint resolve to the generic Transformers runner.
- Existing Llama MLX and explicit runner selection remain functional.
- Kimi K2.5, GLM 4.7 Flash/5.2, and full DeepSeek models are exposed as OpenAI-compatible server models, not misrepresented as laptop-local.
- Local generation uses the model's own chat template and performs no pipeline recreation or duplicate device transfer.
- Tests pass and the app starts under Docker with clean backend logs and a successful request to `localhost:3000`.
