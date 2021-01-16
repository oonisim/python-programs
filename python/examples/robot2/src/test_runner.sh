#!/usr/bin/env bash
DIR=$(realpath $(dirname $0))
cd ${DIR}

export PYTHONPATH=${DIR}/src:${PYTHONPATH}
clear

#--------------------------------------------------------------------------------
# Pylint
#--------------------------------------------------------------------------------
set -eu

rm -rf test/__pycache__/
rm -f test/*.pyc

echo "--------------------------------------------------------------------------------"
echo "Running pylint in src (run in the directory in case of xxx not found in module)..."
for test in $(ls src/*.py)
do
    if [[ "${test}" != "python/six.py" ]]
     then
        python3 -m pylint -E ${test}
    fi
done

echo "--------------------------------------------------------------------------------"
echo "Running pylint in test"
for test in $(ls test/test_*.py)
do
    python3 -m pylint -E ${test}
done

#--------------------------------------------------------------------------------
# PyTest
#--------------------------------------------------------------------------------
echo "--------------------------------------------------------------------------------"
echo "Running PyTest..."
pytest --verbose --cache-clear -x ${DIR}/test/


