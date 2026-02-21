# Plan: Fix Llama 3.1 MLX RoPE scaling + chat templating

1) Inspect current MLX Llama implementation and Llama 3.1 config/tokenizer to identify missing rope scaling and chat template usage.
2) Implement Llama 3.1 rope scaling in the MLX rotary embedding path and wire config fields.
3) Use tokenizer chat template for prompt building with a fallback to the existing manual template.
4) Add/update tests to cover rope scaling config + rope selection.
