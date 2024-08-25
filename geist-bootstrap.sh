#!/bin/sh
set -e

cd /opt/geist

#activate the environment
conda activate geist-linux-docker
mkdir -p output

#start our geist server
python bootstrap.py