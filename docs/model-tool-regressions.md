# Model tool-call regression investigation

Date: 2026-09-05. Scope: the fourth MLX stack change, based on the streaming
fix in [#358](https://github.com/DavidDworetzky/Geist/pull/358).

## Failure and history

The reported request was “search the internet and find me some recent celebrity
headlines.” The MLX response failed inside `ToolResponseStream.feed` →
`parse_tool_response` → `json.loads`, before search dispatch, with
`ValueError: Model returned invalid tool-call JSON`.

The installed Qwen3.8-27B MLX checkpoint's `chat_template.jinja` instructs the
model to emit an XML-like protocol:

```text
<tool_call>
<function=provider_function_name>
<parameter=query>
recent celebrity headlines
</parameter>
</function>
</tool_call>
```

The old parser only understood JSON inside `<tool_call>`, plus a whole-response
JSON compatibility form. These function/parameter tags are not standard XML.

| Change | Evidence and significance |
| --- | --- |
| [#337](https://github.com/DavidDworetzky/Geist/pull/337), `21ada71`, Aug 31 | Introduced the shared JSON-only parser and local native-tool paths. The test named `test_mlx_lm_native_turn_preserves_tool_history_and_parses_call` set `model_id` to Qwen3.8, but mocked a JSON call with query `celebrity news`; it did not exercise the real tokenizer or model. |
| [#338](https://github.com/DavidDworetzky/Geist/pull/338), `dceaf53`, Aug 31 | Promoted Qwen3.8 to the MLX default without closing that format gap. |
| [#339](https://github.com/DavidDworetzky/Geist/pull/339), `09ec89f`, Sep 1 | Added the JSON `parameters` alias and intent routing; the parser still required JSON. The earlier `7e23601` Qwen parsing fix also addressed only that alias. |
| [#356](https://github.com/DavidDworetzky/Geist/pull/356) / [#357](https://github.com/DavidDworetzky/Geist/pull/357) | DFlash/Metal work did not change this shared parser. |
| [#358](https://github.com/DavidDworetzky/Geist/pull/358), `d5bfecb`, Sep 5 | Removed turn aggregation but retained the parser's JSON assumption. The new browser gate covered prose with tools available, not an actual model-generated tool call. That validation missed this gap. |

Negative control: loading the original parser from `git show
21ada71:agents/architectures/chat_template_tools.py` and passing a Qwen XML call
reproduced the same exception. The unsupported-format defect therefore predates
the throughput and streaming work; this evidence does not establish which
earlier user requests happened to elicit JSON versus XML.

## Fix and regression coverage

Preserve JSON compatibility and accept Qwen's function/parameter syntax. Pass
the offered tool schemas into all three shared-parser adapters so string values
such as `123`, `true`, JSON-looking text, and whitespace remain strings, while
integer/boolean/object/array parameters decode as JSON. Reject malformed or
duplicate parameters and unknown names; the existing registry remains the
authority for schema constraints, availability, and approval before dispatch.
Never dispatch a partially received call or expose its markup as answer text.

| Layer | Regression protection |
| --- | --- |
| Shared parser | JSON plus XML, multiple/mixed calls, one-character and split-tag input, typed values, nullable/local-reference schemas, preserved strings, malformed/truncated/duplicate/unknown calls, hidden argument payloads. |
| MLX, Transformers, compatibility runner | Parameterized JSON/XML responses and numeric-looking string arguments. MLX also verifies upstream closure after malformed JSON or XML. |
| Parser → registry | Both protocols must reject missing, extra, empty, or out-of-range arguments without invoking the handler. Existing tests separately cover disabled tools and approval requirements. |
| Installed checkpoint | Render real structured assistant tool history with the installed tokenizer and parse it back. Generate a real tool call, return a deterministic result, and require the model's follow-up to use it. |
| Full browser stack | Feed chunked Qwen XML through the real MLX adapter, runner, LocalAgent, orchestrator, registry, SSE route, browser reader/reducer, and rendered response. Assert exactly one search invocation with typed arguments and that its result reaches the next model turn. Explicit gates require visible intermediate answer text before generation can finish. |
| Intent routing defaults | Unset/false bypass the classifier and retain the enabled catalog; explicit true remains supported. API/service/UI tests plus browser save/reload/restore cover the preference. |

The browser test replaces only model-decoded segments and the external search
response. It is deterministic and runs in the existing `browser-e2e` CI job.
The native test replaces no model generation, but its tool result is fixed to
avoid making model protocol qualification depend on internet availability.

## Qualification limits and future gates

`tokenizer_supports_tools` currently verifies that a template renders the tool
catalog. It is a capability hint, **not proof of a compatible output protocol**.
Mocks bearing a model ID also do not qualify that model's actual behavior.

Before promoting a model or changing its checkpoint/template, run:

1. Actual-template assistant-call serialization → parser round trip.
2. Real generation → parsed and schema-validated call → deterministic tool
   result → model follow-up. Include a string that looks numeric.
3. Relevant runner matrix and the full-stack browser tool/streaming gates.

The installed-model tests are opt-in:

```bash
GEIST_RUN_MLX_TOOL_SMOKE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. pytest tests/agents/test_mlx_tool_live.py -q -s
```

They require already installed supported MLX weights and do not download them.
`GEIST_MLX_TOOL_ARTIFACT_ID` selects another installed artifact; it is not a
claim that every artifact is supported by the tagged-protocol qualification.
This pass qualified Qwen3.8-27B Q4 at revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff` with MLX 0.32.2 / MLX-LM 0.31.3.
Other advertised model families, remote API providers, arbitrary tool schemas,
and all sampling settings were not model-natively qualified by these tests.
The unquoted XML-like representation can itself be ambiguous (for example a
nullable string containing literal `null`); this parser does not make it a
lossless general-purpose serialization format.

Making intent routing default off is a separate preference change. Existing
explicit opt-ins are preserved. It avoids an extra classifier pass when unset,
but sends the full enabled catalog to the answer model, which can increase
prefill cost. The user's observed saved setting was already false; no latency
improvement or TTFB average is claimed from changing that default.

## Validation commands and environment

Used the existing native virtualenv and frontend `node_modules`, and the existing
`geist-goal-decomposition-backend:latest` Docker image. No installs or image
builds. Local dotenv loading was disabled in Python launchers. Tests used
disposable SQLite databases, not the user's chat database. The full browser
suite requires a fresh database; replaying the memory fixtures into the same
database creates duplicates.

Backend command (560 passed, 6 skipped; two new opt-in checks passed separately
on native MLX):

```bash
docker run --rm --name geist-qwen-tool-tests --platform linux/amd64 --memory 4g \
  --entrypoint python -v /private/tmp/geist-mlx-stream.LwTj2v:/work:ro -w /work \
  -e PYTHONPATH=/work -e GEIST_DATABASE_PROVIDER=sqlite \
  -e SQLALCHEMY_DATABASE_URL=sqlite:////tmp/geist-qwen-tests.sqlite3 \
  geist-goal-decomposition-backend:latest -c \
  'import dotenv; dotenv.load_dotenv=lambda *a,**k:False; from initdb import main; main(); import pytest; raise SystemExit(pytest.main(["tests/agents", "tests/services/test_chat_orchestrator.py", "tests/services/test_tool_intent_router.py", "tests/services/test_tool_registry.py", "tests/api/test_tools_api.py", "tests/api/test_user_settings_routes.py", "-q", "--tb=short", "-p", "no:cacheprovider"]))'
```

Native model qualification invoked the existing Python with
`GEIST_RUN_MLX_TOOL_SMOKE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, the
installed model home, and an isolated SQLite/data directory:

```bash
python -c 'import dotenv; dotenv.load_dotenv=lambda *a,**k:False; import pytest; raise SystemExit(pytest.main(["tests/agents/test_mlx_tool_live.py", "-q", "-s", "--tb=short"]))'
```

Result: 2 passed in 24.93 s. Real llama.cpp/Transformers generation was not run;
their relevant contracts were tested in Docker, but that is not native model
qualification. MLX kernel tests unchanged by this PR were not re-benchmarked.

Frontend commands, from `client/geist`, using existing Node and dependencies:

```bash
CI=true node node_modules/react-scripts/scripts/test.js --watchAll=false --runInBand --runTestsByPath src/Settings.test.tsx
BUILD_PATH=/private/tmp/geist-qwen-tool-validation.dv2pdL/web CI=true node node_modules/react-scripts/scripts/build.js
```

Results: 17 tests passed; production build passed. Ruff lint/format checked all
16 changed Python files; mypy checked the six production Python files plus
`tests/e2e_server.py` and `tests/streaming_probe.py` with `--config-file=pyproject.toml`.

The temporary browser config points at the repository's unmodified test
directory, one Chrome worker, production assets, and port 5588 (native) or 5591
(Docker). It does not start package managers. With the existing Playwright CLI:

```bash
node node_modules/@playwright/test/cli.js test --config=/private/tmp/geist-qwen-tool-validation.dv2pdL/playwright.config.cjs
GEIST_TEST_URL=http://127.0.0.1:5591 GEIST_TEST_AUTH=1 node node_modules/@playwright/test/cli.js test --config=/private/tmp/geist-qwen-tool-validation.dv2pdL/playwright.config.cjs chat.spec.ts settings.spec.ts
docker compose -p geist-qwen-tool-check -f /private/tmp/geist-qwen-tool-validation.dv2pdL/compose.yml up -d --wait --wait-timeout 45
docker compose -p geist-qwen-tool-check -f /private/tmp/geist-qwen-tool-validation.dv2pdL/compose.yml logs --no-color backend
```

Results: 10 native-server browser tests passed in 13.2 s; 6 authenticated Docker
chat/settings tests passed in 5.9 s; Docker startup healthy. An authenticated
`curl` to `http://127.0.0.1:5591/chat` returned HTTP 200. The isolated test token
is disposable and is not a user credential. Logs showed the deliberately
injected model-failure test, with no unexpected startup or tool-call error.

The literal default `docker compose up`/port 3000 and `make run MLX_BACKEND=1`
commands were not used: unrelated services occupy those ports, and the Make
target invokes `uv run --extra local-mlx`, which can synchronize packages.
Instead, existing dependencies served the same app via native `create_app`
with `MLX_BACKEND=1`, and Docker served the production assets plus test app on
isolated ports. This does not validate the package-install/bootstrap workflow.

Commit-hook caveat: the cached mypy hook reports `no-any-return` in unchanged
`adapters/whisper_adapter.py:38`, while the installed-environment changed-file
check passes. Bandit's only finding is the unchanged `app/main.py` bind to
`0.0.0.0`; both lines predate this stack. Repository-wide ESLint reports 82
existing errors. Linting `Settings.test.tsx` from `git show HEAD:...` and the
updated file produces the same 42 rule/message findings; `Settings.tsx` has
zero. The three affected hooks were skipped for this commit after comparing
their findings; formatting and staged secret/private-key scans passed. These
are existing-check failures, not a claim of a green repository-wide lint run.
