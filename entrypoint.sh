#!/bin/bash --login
set -e

cd /opt/geist

# Temporarily disable strict mode and activate conda:
set +euo pipefail

#activate the environment
conda activate geist-linux-docker
mkdir -p output

# Re-enable strict mode:
set -euo pipefail

#start our geist server
python bootstrap.py