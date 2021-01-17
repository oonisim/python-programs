#!/usr/bin/env bash
#--------------------------------------------------------------------------------
# Bash crates a new shell process in which the Python virtual environment has not been activated.
# Hence will fail without setting it up.
#--------------------------------------------------------------------------------
DIR=$(realpath $(dirname $0))
cd ${DIR}

export PYTHONPATH=${DIR}/package:${PYTHONPATH}
clear

#--------------------------------------------------------------------------------
# Pylint
#--------------------------------------------------------------------------------
set -eu

rm -rf test/__pycache__/
rm -f test/*.pyc

echo "--------------------------------------------------------------------------------"
echo "Running pylint in package (run in the directory in case of xxx not found in module)..."
pushd package
for f in $(ls package/*.py)
do
    if [[ "${f}" != "six.py" ]]
     then
        pylint -E ${f}
    fi
done
popd

echo "--------------------------------------------------------------------------------"
echo "Running pylint in test"
for t in $(ls test/test_*.py)
do
    pylint -E ${t}
done

#--------------------------------------------------------------------------------
# PyTest
#--------------------------------------------------------------------------------
echo "--------------------------------------------------------------------------------"
echo "Running PyTest..."
pytest --verbose --cache-clear -x ${DIR}/test/
