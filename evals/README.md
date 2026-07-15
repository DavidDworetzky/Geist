# Tool-routing evaluations

These tests score recorded tool calls only. They do not call a model, execute a
tool, read the repository `.env`, or use the network.

Run the suite with the repository's locked test environment:

```bash
uv run pytest -q evals
```

The checks use exact matching over recorded tool names and arguments, so no
model, network call, secret discovery, telemetry, or additional evaluation
dependency is needed.
