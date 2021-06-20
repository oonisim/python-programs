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

#--------------------------------------------------------------------------------
# Pylint
#--------------------------------------------------------------------------------
set -eu

rm -rf __pycache__/
#rm -f *.pyc

echo "--------------------------------------------------------------------------------"
echo "Running pylint in package (run in the directory in case of xxx not found in module)..."
# Find "! expr":  True if expr is false.
for f in $(find . -type f -name '*.py' ! -name '__init__.py')
do
    echo ${f}
    # pylint -E ${f}
done


#--------------------------------------------------------------------------------
# PyTest
#   --pdb \ to invoke pdb at failure
#   -k 'test_softmax_classifier' to specify a test
#--------------------------------------------------------------------------------
echo "--------------------------------------------------------------------------------"
echo "Running PyTest..."
rm -rf __pycache__/
rm -rf .pytest_cache/

# pytest --log-level=DEBUG -o log_cli=True -o log_cli_level=DEBUG --verbose --cache-clear -x -capture=tee-sys ${DIR} | tee pytest.log

# To disable assert
# PYTHONOPTIMIZE=TRUE

#--------------------------------------------------------------------------------
# Parallel pytest requires pytest-xdist
# https://stackoverflow.com/questions/28908319
# conda install pytest-xdist -y
# Then use with -n option
#
# [NOTE]
# Cannot use live cli log with pytest-xdist <--- !!!
#  -o log_cli=False -o log_cli_level=WARNING \
#
# [Selector]
# https://docs.pytest.org/en/latest/example/markers.html#using-k-expr-to-select-tests-based-on-their-name
# use -k option to specify which tests to run or NOT to run.
# $ pytest -k "http or quick"
# $ pytest -k "not send_http"
#--------------------------------------------------------------------------------
NUM_CPU=6
#python3 -m cProfile -o profile -m pytest \
#python3 -m memory_profiler -m pytest \
pytest \
  -n $NUM_CPU \
  --rootdir=${DIR} \
  -vv \
  --capture=tee-sys \
  --log-level=ERROR \
  --log-auto-indent=on \
  --cache-clear -x \
  --color=yes --code-highlight=yes \
  --full-trace \
  --tb=long \
  --showlocals \
  --durations=5 \
  $@ \
${DIR}
