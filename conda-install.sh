#!/bin/bash --login

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please ensure conda is installed and in PATH."
    exit 1
fi

# Source conda initialization if available
CONDA_INIT="/root/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_INIT" ]; then
    source "$CONDA_INIT"
else
    echo "Warning: conda initialization script not found at $CONDA_INIT"
    echo "Attempting to initialize conda..."
    conda init bash
fi

cd /opt/geist

# Check if environment file exists
if [ ! -f "linux_environment.yml" ]; then
    echo "Error: linux_environment.yml not found in /opt/geist"
    exit 1
fi

# Temporarily disable strict mode and activate conda:
set +euo pipefail

# Clean up all existing environments
conda clean --all --yes

# Remove existing environment (ignore errors if it doesn't exist)
conda env remove --name geist-linux-docker --yes 2>/dev/null || true

# Create environment
echo "Creating conda environment from linux_environment.yml..."
if conda env create -f linux_environment.yml; then
    echo "Successfully created geist-linux-docker environment"
else
    echo "Error: Failed to create conda environment"
    exit 1
fi

# Re-enable strict mode:
set -euo pipefail

