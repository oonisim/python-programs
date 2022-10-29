#!/usr/bin/env bash
#--------------------------------------------------------------------------------
# Run the model re-training
#--------------------------------------------------------------------------------
set -eu
# Cloud build causes "git not found"
# BASE_DIR=$(git rev-parse --show-toplevel)
BASE_DIR=$(realpath ../..)
PATH_TO_REQUIREMENTS="${BASE_DIR}/requirements.txt"

echo "--------------------------------------------------------------------------------"
echo "Installing python requirements ${PATH_TO_REQUIREMENTS}"
echo "--------------------------------------------------------------------------------"
if [ ! -f "$PATH_TO_REQUIREMENTS" ]; then
    echo "$PATH_TO_REQUIREMENTS does not exist."
    exit -1
else
  pip install -r ${PATH_TO_REQUIREMENTS}
fi

echo "--------------------------------------------------------------------------------"
echo "Retraining model"
echo "--------------------------------------------------------------------------------"
rm -f ../../model/model.npy
if [ ! -f "simple_linear_regr.py" ]; then
    echo "simple_linear_regr.py does not exist."
    exit -1
else
    if python simple_linear_regr.py; then
        echo "Retraining done"
    else
        echo "Retraining failed"
        exit -1
    fi
fi
