# QA Result Schema

Store the independent subagent's result as UTF-8 JSON. Use only `PASS`, `FAIL`, or `BLOCKED` for the overall verdict and runtime/test results. Use `NOT_APPLICABLE` only for individual runtime lanes excluded by the risk map.

```json
{
  "verdict": "PASS",
  "summary": "Highest-risk chat recovery pathway passed across applicable lanes.",
  "pr": {
    "number": 123,
    "url": "https://github.com/owner/repo/pull/123",
    "base_sha": "full-base-sha",
    "head_sha": "full-head-sha",
    "branch": "feature-branch"
  },
  "traceability": [
    {
      "rank": 1,
      "code_path": "route -> service -> response contract -> UI",
      "behavior": "The changed user-visible outcome",
      "risk": "Why this pathway was ranked first",
      "runtime_lanes": ["docker", "mlx"],
      "test": "Focused test or browser action",
      "result": "PASS",
      "evidence": "Exact visible state, response, or log evidence"
    }
  ],
  "runtime_lanes": [
    {
      "name": "docker",
      "result": "PASS",
      "runner": "transformers",
      "model": "Qwen/Qwen3-4B",
      "commands": ["docker compose up -d --build"],
      "evidence": "Health, prompt response, browser, and log evidence",
      "reason": ""
    }
  ],
  "focused_tests": [
    {
      "command": "pytest -q path/to/test.py",
      "result": "PASS",
      "evidence": "1 passed"
    }
  ],
  "failures": [],
  "blockers": [],
  "artifacts": {
    "video_path": "/private/tmp/geist-pr-123-deadbee-qa.mp4",
    "video_url": "",
    "report_url": ""
  }
}
```

Never put tokens, environment values, secret file contents, or unrelated user data in this JSON.
