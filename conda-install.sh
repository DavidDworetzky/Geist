#!/bin/sh
set -e

cd /opt/geist

#clean up all existing environments
conda clean --all --yes
conda env remove --name geist-linux-docker
conda env create -f linux_environment.yml

#init the environment
conda init
