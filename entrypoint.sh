#!/bin/sh
set -e
conda clean --all --yes
conda env remove --name geist-mac-docker
conda env create -f mac_arm_environment.yml

conda activate geist-mac-docker
mkdir -p output

python bootstrap.py
