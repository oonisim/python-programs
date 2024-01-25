#!/usr/bin/env bash
set -eu
DIR=$(realpath $(dirname $0))
cd ${DIR}

conda install --yes --file requirements.txt
pip install gensim ray
