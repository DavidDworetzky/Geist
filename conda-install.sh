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

# Determine which environment file to use based on architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    ENV_FILE="linux_environment.yml"
elif [ "$ARCH" = "x86_64" ]; then
    ENV_FILE="linux_environment_x86_x64.yml"
else
    echo "Error: Unsupported architecture $ARCH"
    exit 1
fi

echo "Using environment file: $ENV_FILE for architecture: $ARCH"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found in /opt/geist"
    exit 1
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Temporarily disable strict mode and activate conda:
set +euo pipefail

# Clean up all existing environments
conda clean --all --yes

# Remove existing environment (ignore errors if it doesn't exist)
conda env remove --name geist-linux-docker --yes 2>/dev/null || true

# Create environment
echo "Creating conda environment from $ENV_FILE..."
if conda env create -f "$ENV_FILE"; then
    echo "Successfully created geist-linux-docker environment"
else
    echo "Error: Failed to create conda environment"
    exit 1
fi

# Re-enable strict mode:
set -euo pipefail

