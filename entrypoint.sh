#!/bin/sh
set -e

cd /opt/geist

conda clean --all --yes
conda env remove --name geist-linux-docker
# conda env create -f linux_environment.yml
conda create --name geist-linux-docker python=3.10
#conda init
#conda activate geist-linux-docker
mkdir -p output

#python bootstrap.py
tail -f /dev/null