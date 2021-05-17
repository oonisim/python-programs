#!/usr/bin/env bash
# https://stackoverflow.com/questions/66406811

#--------------------------------------------------------------------------------
# Bash crates a new shell process in which the Python virtual environment has not been activated.
# Hence will fail without setting it up.
#--------------------------------------------------------------------------------
DIR=$(realpath $(dirname $0))
PARENT=$(realpath "${DIR}/..")
cd ${DIR}

# export PYTHONPATH=$(realpath "${DIR}/.."):${DIR}
export PYTHONPATH=${PARENT}:${DIR}
echo "PYTHONPATH=$PYTHONPATH"

# python3 -m memory_profiler test_050_word2vec.py
python3 test_050_word2vec.py
