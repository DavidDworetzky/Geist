# Tool-routing evaluations

These tests score recorded tool calls only. They do not call a model, execute a
tool, read the repository `.env`, or use the network.

Create the isolated environment and run the suite with:

```bash
conda env create -f eval_environment.yml
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DEEPEVAL_DISABLE_DOTENV=1 \
  DEEPEVAL_DISABLE_LEGACY_KEYFILE=1 DEEPEVAL_TELEMETRY_OPT_OUT=1 \
  conda run -n geist-eval pytest --confcutdir=evals -q evals
```

`--confcutdir=evals` keeps this minimal environment independent from the
application fixtures in the repository root `conftest.py`. Plugin autoloading is
disabled because the suite calls the deterministic metric directly and does not
need DeepEval's pytest plugin or its auxiliary plugins. Without DeepEval, normal
test collection reports this suite as skipped with installation guidance.
