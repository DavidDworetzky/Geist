#!/bin/bash
set -e

cd /opt/geist

# Create output directory
mkdir -p output

# Initialize database
python initdb.py

# Start Geist server
python bootstrap.py
