#!/usr/bin/env bash
# Objective: Run pytest
# example: ./run_test.sh lib/test/util_nlp
DIR=$(realpath $(dirname $0))
export PYTHONPATH="${PYTHONPATH}:${DIR}"

# pytest --capture=no -o log_cli=true ${DIR}/test/$1
pytest ${DIR}/$1 -v
