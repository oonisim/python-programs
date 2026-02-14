#!/usr/bin/env bash
# Test runner script that sets up PYTHONPATH correctly
# Usage: ./run_test.sh [pytest arguments]

set -e

# Get the directory where this script is located
DIR=$(realpath $(dirname "${0}"))
cd ${DIR}

# Set PYTHONPATH to include project root and test directory
export PYTHONPATH="$(realpath ..):${DIR}"

# Run pytest with any provided arguments
pytest . "$@"
