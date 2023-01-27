#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# Pytest Runner
# [Note]
# Make sure to run tests after the modelling pipeline as well to test the generated artifacts.
#----------------------------------------------------------------------------------------
set -ue
DIR=$(realpath $(dirname $0))

export PYTHONPATH="${DIR}/../../../lib:${DIR}/src:${DIR}/config"

pytest -s -o log_cli=true --log-cli-level=INFO
