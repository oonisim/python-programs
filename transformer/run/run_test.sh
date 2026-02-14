#!/usr/bin/env bash
# Test runner script that sets up PYTHONPATH correctly
# Usage: ./run_test.sh [pytest arguments]

set -e

# Get the directory where this script is located (run/)
DIR=$(realpath $(dirname "${0}"))

# Project root is one level up from run/
PROJECT_ROOT=$(realpath "${DIR}/..")

# Test directory
TEST_DIR="${PROJECT_ROOT}/test"

# Set PYTHONPATH to include project root and test directory
export PYTHONPATH="${PROJECT_ROOT}:${TEST_DIR}"

# Run pytest on test directory
pytest "${TEST_DIR}" "$@"
