# Tool-routing evaluations

The default eval suite validates the intent dataset and scores recorded tool
calls. It does not call a model, execute a tool, read the repository `.env`, or
use the network. Model-backed behavior is covered separately by opt-in live
model testing and the native runtime smoke pass.

Run the suite with the repository's locked test environment:

```bash
uv run pytest -q evals
```

The checks use exact matching over recorded tool names and arguments, so no
model, network call, secret discovery, telemetry, or additional evaluation
dependency is needed.
