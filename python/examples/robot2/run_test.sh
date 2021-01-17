#!/usr/bin/env bash
# !!!!!!!!!!
# Bash crates a new shell process in which the Python virtual environment has not been activated.
# Hence will fail without setting it up.
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
pushd src
for f in $(ls *.py)
do
    if [[ "${f}" != "six.py" ]]
     then
        pylint -E ${f}
    fi
done
popd

echo "--------------------------------------------------------------------------------"
echo "Running pylint in test"
pushd test
for t in $(ls test_*.py)
do
    pylint -E ${t}
done
popd

#--------------------------------------------------------------------------------
# PyTest
#--------------------------------------------------------------------------------
echo "--------------------------------------------------------------------------------"
echo "Running PyTest..."
pytest --verbose --cache-clear -x ${DIR}/test/
