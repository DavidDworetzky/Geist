#!/bin/sh
set -e
conda clean --all --yes
conda env remove --name geist
conda env create -f environment.yml

conda activate geist
mkdir -p output

uvicorn main:app --reload
