#!/usr/bin/env bash
DIR=$(realpath $(dirname $0))
cd ${DIR}

python3 -m package.main -m 10 -n 15 -f ${DIR}/package/data/commands.txt
