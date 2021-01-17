# Bash crea
set +e
DIR=$(realpath $(dirname $0))
cd ${DIR}

export PYTHONPATH=${DIR}:${PYTHONPATH}
clear

#--------------------------------------------------------------------------------
# Pylint
#--------------------------------------------------------------------------------
set -eu

rm -rf test/__pycache__/
rm -f test/*.pyc

echo "--------------------------------------------------------------------------------"
echo "Running pylint in src (run in the directory in case of xxx not found in module)..."
for test in $(ls *.py)
do
    if [[ "${test}" != "six.py" ]]
     then
        pylint -E ${test}
    fi
done

echo "--------------------------------------------------------------------------------"
echo "Running pylint in test"
for test in $(ls test_*.py)
do
    pylint -E ${test}
done

#--------------------------------------------------------------------------------
# PyTest
#--------------------------------------------------------------------------------
echo "--------------------------------------------------------------------------------"
echo "Running PyTest..."
pytest --verbose --cache-clear -x ${DIR}


