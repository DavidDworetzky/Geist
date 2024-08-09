#!/bin/sh
set -e
conda clean --all --yes
conda env remove --name geist-linux
conda env create -f environment.yml

conda activate geist-linux
mkdir -p output

python bootstrap.py
