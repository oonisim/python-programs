#!/usr/bin/env bash
# Test runner script that sets up PYTHONPATH correctly
# Usage: ./run_test.sh [pytest arguments]

set -e

# Get the directory where this script is located (run/)
DIR=$(realpath $(dirname "${0}"))

# Project root is one level up from run/
PROJECT_ROOT=$(realpath "${DIR}/..")

# Source directory containing model, training, and test modules
SRC_DIR="${PROJECT_ROOT}/src"

# Set PYTHONPATH to src directory so modules can be imported as 'from model', 'from training', etc.
export PYTHONPATH="${SRC_DIR}"

# Run pytest on test directory
python3 -m pytest "${SRC_DIR}/test" "$@"
