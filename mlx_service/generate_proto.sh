#!/bin/bash
# Generate Python gRPC code from proto files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Generate Python code
python -m grpc_tools.protoc \
    --proto_path="$PROJECT_ROOT/proto" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$PROJECT_ROOT/proto/inference.proto"

echo "Generated Python gRPC code in $SCRIPT_DIR"
