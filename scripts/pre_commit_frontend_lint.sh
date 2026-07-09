#!/usr/bin/env bash
set -euo pipefail

cd client/geist

if [ ! -x node_modules/.bin/eslint ]; then
  echo "client/geist/node_modules is missing eslint. Run 'cd client/geist && npm ci' before committing frontend source changes." >&2
  exit 1
fi

npm run lint -- --max-warnings=0
