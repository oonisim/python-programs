#!/usr/bin/env bash
DIR=$(realpath $(dirname $0))

pytest --capture=no -o log_cli=true -s ${DIR}/test/$1