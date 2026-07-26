#!/bin/bash
set -euo pipefail

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Please ensure uv is installed and in PATH."
    exit 1
fi

cd /opt/geist

echo "Installing Python dependencies using uv..."

# Sync dependencies from lock file
# --frozen: Use the lock file as-is, don't update it
# PostgreSQL remains available through the optional postgres extra; the default
# container uses SQLite. Linux containers deliberately retain the existing
# Transformers inference path while native Windows/Linux packages use llama.cpp.
if uv sync --frozen --extra local-transformers; then
    echo "Successfully installed dependencies"
else
    echo "Error: Failed to sync dependencies"
    exit 1
fi

echo "Python environment setup complete!"
