#!/usr/bin/env bash
# https://stackoverflow.com/questions/66406811

#--------------------------------------------------------------------------------
# Bash crates a new shell process in which the Python virtual environment has not been activated.
# Hence will fail without setting it up.
#--------------------------------------------------------------------------------
DIR=$(realpath $(dirname $0))
cd ${DIR}

export PYTHONPATH=${DIR}
echo "PYTHONPATH=$PYTHONPATH"
#clear

#--------------------------------------------------------------------------------
# Pylint
#--------------------------------------------------------------------------------
set -eu

rm -rf __pycache__/
#rm -f *.pyc

echo "--------------------------------------------------------------------------------"
echo "Running pylint in package (run in the directory in case of xxx not found in module)..."
for f in $(ls *.py)
do
    if [[ "${f}" != "six.py" ]]
     then
#        pylint -E ${f}
      echo
    fi
done

echo "--------------------------------------------------------------------------------"
echo "Running pylint in test"
for t in $(ls test_*.py)
do
    echo
    # pylint -E ${t}
done

#--------------------------------------------------------------------------------
# PyTest
#   --pdb \ to invoke pdb at failure
#   -k 'test_softmax_classifier' to specify a test
#--------------------------------------------------------------------------------
echo "--------------------------------------------------------------------------------"
echo "Running PyTest..."
# pytest --log-level=DEBUG -o log_cli=True -o log_cli_level=DEBUG --verbose --cache-clear -x -capture=tee-sys ${DIR} | tee pytest.log

# To disable assert
# PYTHONOPTIMIZE=TRUE

#pytest \
python3 -m cProfile -o profile -m pytest \
  --rootdir=${DIR} \
  -vv \
  -capture=tee-sys  \
  --log-level=ERROR \
  --log-auto-indent=on \
  --cache-clear -x \
  --color=yes --code-highlight=yes \
  --full-trace \
  --tb=long \
  --showlocals \
  --durations=5 \
${DIR}

python3 run_cprofile_analysis.py