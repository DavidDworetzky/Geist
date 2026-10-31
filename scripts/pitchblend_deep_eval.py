#!/usr/bin/env python3
"""Run a model-backed eval for the frontend approval boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).parents[1]
EVALUATOR_PATH = REPOSITORY_ROOT / ".github/scripts/pitchblend_review.py"
EXPECTED_PASS_RULE = "frontend-feature-low-or-medium"

PULL_REQUEST = {
    "title": "Add installed model selector to the application header",
    "body": (
        "Replace the model label with a selector backed by existing settings and "
        "installed-model APIs. Roll back the selection when saving fails and add "
        "focused frontend tests."
    ),
}

FILES = [
    {
        "filename": "client/src/AppHeader.tsx",
        "status": "modified",
        "additions": 10,
        "deletions": 1,
        "patch": """@@
-<span>{activeModel}</span>
+<select value={selectedModelId} onChange={selectModel}>
+  {installedModels.map(model => (
+    <option key={model.id} value={model.id}>{model.name}</option>
+  ))}
+</select>
""",
    },
    {
        "filename": "client/src/hooks/useInstalledModels.ts",
        "status": "added",
        "additions": 20,
        "deletions": 0,
        "patch": """@@
+export function useInstalledModels() {
+  const [selectedModelId, setSelectedModelId] = useState(settings.modelId);
+  const [installedModels, setInstalledModels] = useState<Model[]>([]);
+  useEffect(() => {
+    fetch('/api/models/installed')
+      .then(response => response.json())
+      .then(setInstalledModels);
+  }, []);
+  const selectModel = async (event: ChangeEvent<HTMLSelectElement>) => {
+    const previousModelId = selectedModelId;
+    const nextModelId = event.target.value;
+    setSelectedModelId(nextModelId);
+    try {
+      await saveSettings({ modelId: nextModelId });
+    } catch {
+      setSelectedModelId(previousModelId);
+    }
+  };
+  return { installedModels, selectedModelId, selectModel };
+}
""",
    },
    {
        "filename": "client/src/AppHeader.test.tsx",
        "status": "modified",
        "additions": 15,
        "deletions": 0,
        "patch": """@@
+it('persists an installed model selection', async () => {
+  render(<AppHeader />);
+  await user.selectOptions(screen.getByRole('combobox'), 'model-b');
+  expect(saveSettings).toHaveBeenCalledWith({ modelId: 'model-b' });
+});
+it('rolls back a failed save', async () => {
+  saveSettings.mockRejectedValueOnce(new Error('save failed'));
+  render(<AppHeader />);
+  await user.selectOptions(screen.getByRole('combobox'), 'model-b');
+  expect(screen.getByRole('combobox')).toHaveValue('model-a');
+});
""",
    },
]


def load_evaluator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pitchblend_deep_eval_evaluator", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Pitchblend evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("PITCHBLEND_EVAL_OPENAI_API_KEY is not configured")

    evaluator = load_evaluator()
    model = os.environ.get("PITCHBLEND_MODEL", "gpt-5.6-luna")
    reasoning_effort = os.environ.get("PITCHBLEND_REASONING_EFFORT", "high")
    classification = evaluator.classify_pull_request(
        api_key, model, reasoning_effort, PULL_REQUEST, FILES
    )
    matched_rule = evaluator.classification_pass_rule(classification)
    print(
        json.dumps(
            {
                "fixture": "frontend-installed-model-selector",
                "model": model,
                "reasoning_effort": reasoning_effort,
                "classification": classification,
                "objective_matrix": {
                    "approved": matched_rule is not None,
                    "matched_pass_rule": matched_rule,
                    "expected_pass_rule": EXPECTED_PASS_RULE,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if matched_rule != EXPECTED_PASS_RULE:
        raise RuntimeError(
            f"Deep eval expected {EXPECTED_PASS_RULE}, got {matched_rule or 'no pass rule'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
