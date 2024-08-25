#!/bin/sh
set -e

cd /opt/geist

#clean up all existing environments
conda clean --all --yes
conda env remove --name geist-linux-docker
conda env create -f linux_environment.yml

#init and activate the environment
conda init
conda activate geist-linux-docker
mkdir -p output

#start our geist server
python bootstrap.py