#!/usr/bin/env bash
DIR=$(realpath $(dirname $0))
export PYTHONPATH="${PYTHONPATH}:${DIR}/lib"

# pytest --capture=no -o log_cli=true ${DIR}/test/$1
pytest ${DIR}/test/$1 -v
