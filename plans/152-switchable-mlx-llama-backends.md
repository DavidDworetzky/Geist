# Plan: Switchable MLX Llama backends

## Goal

Keep Geist's manual Llama 3.1 implementation while making the official
`mlx-lm` runtime selectable behind the existing `mlx_llama` runner. Optimize
the manual path, preserve the existing completion response contract, and
benchmark both implementations on the same Apple Silicon host.

## Configuration

- Default implementation: `manual`.
- Environment switch: `GEIST_MLX_IMPLEMENTATION=manual|mlx_lm`.
- Programmatic override: `device_config["implementation"]`.
- Optional local model path: `device_config["weights_dir"]` or
  `LOCAL_WEIGHTS_DIR`.
- The runner must reject unknown implementation names with a useful error.

## Implementation steps

1. Add a small backend protocol and keep `MLXLlamaRunner` responsible for
   implementation selection and generation configuration.
2. Optimize the manual Llama path:
   - use MLX fused grouped-query attention;
   - sample on device without converting the vocabulary logits to NumPy;
   - use a grow-in-chunks KV cache instead of concatenating on every token;
   - load safetensors directly with MLX and realize model weights at load time;
   - expose generated-token iteration so the runner can stream later without
     another model implementation change.
3. Add an `mlx-lm` adapter using its public `load`, `generate`, sampler, and
   prompt-cache APIs while returning Geist's existing message dictionary.
4. Pin compatible `mlx`, `mlx-lm`, and Transformers dependencies and freeze the
   environment according to repository policy.
5. Add unit tests for backend selection, optional dependency errors, model/path
   propagation, fused attention, device sampling, cache growth, and response
   compatibility.
6. Add a benchmark script that reports load time, prompt tokens/second,
   generation tokens/second, and generated-token count for either backend.
7. Run the focused tests, broader architecture tests, Python dependency audit,
   and both backends against Llama 3.1.
8. Run the repository-required Docker startup, container log check, and
   `curl http://localhost:3000` smoke test.

## Data and API impact

No database model, service, or route changes are required. The switch is a
runtime concern behind the existing local-agent/runner boundary, so the public
completion API remains unchanged.

## Acceptance criteria

- Existing callers can continue using `runner_type="mlx_llama"` unchanged.
- `manual` and `mlx_lm` are both selectable and use the requested model/path.
- The manual backend no longer performs per-token NumPy sampling or per-token
  full KV-cache concatenation.
- Both backends return the same Geist `messages` envelope.
- Benchmarks for both implementations are captured with the model and hardware
  configuration stated explicitly.
- Dependency and runtime validation complete without hiding unrelated existing
  worktree changes.

## Benchmark results

Measured July 13, 2026 on an Apple M2 Max with 96 GB unified memory, using the
same local BF16 Meta Llama 3.1 8B Instruct shards, a 53-token formatted prompt,
greedy decoding, and 16 generated tokens. Each implementation ran in a fresh
process after the model files were warm in the OS file cache.

| Implementation | Load | Prompt tok/s | Generation tok/s | Completion wall | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `manual` | 1.240 s | 253.16 | 21.64 | 0.959 s | 16.128 GB |
| `mlx_lm` | 1.304 s | 127.12 | 22.90 | 1.196 s | 16.126 GB |

Both implementations produced the same greedy token sequence. The optimized
manual decoder reaches 94.5% of `mlx-lm`'s decode throughput and is about 21.6x
faster than the pre-change manual baseline of 1.002 generated tokens/second.

Commands:

```bash
conda run --no-capture-output -n geist-mac python scripts/benchmark_mlx_llama.py \
  --implementation manual --weights-dir /Users/daviddworetzky/Desktop/Local_Weights/llama_3_1 \
  --max-tokens 16 --temperature 0 --top-p 1

conda run --no-capture-output -n geist-mac python scripts/benchmark_mlx_llama.py \
  --implementation mlx_lm --weights-dir /Users/daviddworetzky/Desktop/Local_Weights/llama_3_1 \
  --max-tokens 16 --temperature 0 --top-p 1
```
