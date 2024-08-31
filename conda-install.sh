#!/bin/bash --login

cd /opt/geist

# Temporarily disable strict mode and activate conda:
set +euo pipefail

#clean up all existing environments
conda clean --all --yes
conda env remove --name geist-linux-docker
conda env create -f linux_environment.yml

#init the environment
conda init

# Re-enable strict mode:
set -euo pipefail

