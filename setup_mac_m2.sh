#!/usr/bin/env bash
# --------------------------------------------------------------------------------
# [Objective]
# Setup Python virtual environment for Mac Apple Silicon.
# 1. Replace MacOS BLAS with Open BLAS that handles Apple Silicon on chip vector unit.
#
# [History]
# 2023MAR06: Initial
#
# [Issue]
# tfx-bsl (the basis for TFX) does not work on Apple Silicon as of 2023 MAR as in
# https://github.com/tensorflow/tfx-bsl/issues/48. Hence TFX modules does not work.
# --------------------------------------------------------------------------------
DIR=$(realpath .)
cd $DIR

# --------------------------------------------------------------------------------
# Verify running in a virtual environment.
# --------------------------------------------------------------------------------
if [[ "${VIRTUAL_ENV}x" == "x" ]]; then
  echo "Activate Virtual Environment First"
  exit -1
fi
export VIRTUAL_ENV="${VIRTUAL_ENV:?'Activate Virtual Environment First'}"

# --------------------------------------------------------------------------------
# Replace MacOS BLAS with Open BLAS for M2 to avoid Altivec issue.
# --------------------------------------------------------------------------------
brew install openblas
export OPENBLAS="$(brew --prefix openblas)"
export MACOSX_DEPLOYMENT_TARGET="13.0.1"

# --------------------------------------------------------------------------------
# Requirement for tfx-bsl (requirement for tensorflow transform)
# --------------------------------------------------------------------------------
brew install bazel

# --------------------------------------------------------------------------------
# Latest pip
# --------------------------------------------------------------------------------
python3 -m pip install --upgrade pip

# --------------------------------------------------------------------------------
# Python setup for Mac M2
# tensorflor_transform requires future.
# --------------------------------------------------------------------------------
# https://stackoverflow.com/questions/75611977
pip install --no-cache-dir \
  setuptools \
  wheel \
  Cython \
  pyarrow==6.0.0 \
  numpy

pip install -r requirements.txt

# --------------------------------------------------------------------------------
# as a note
# --------------------------------------------------------------------------------
exit 0
pip install --no-cache-dir \
  tensorflow-macos \
  tensorflow-metal \
  tensorflow-transform \
  tensorflow-datasets \
  tensorflow_decision_forests \
  tensorboard \
  transformers

# Does not work on Apple Silicon
# pip install --no-cache-dir \
#  tfx \
#  tensorflow-transform \
#  future

pip install \
  urllib3 \
  python-dateuti \
  dateutils \
  datefinder \
  holidays \
  pytz \
  PyYAML \
  line_profiler \
  memory_profiler \
  beautifulsoup4 \
  lxml

pip install \
  pylint \
  pytest \
  pytest-xdist

pip install \
  matplotlib \
  seaborn \


pip install \
  pandas \
  numexpr \
  wurlitzer \
  pydeequ \
  scikit-learn \
  imblearn \
  jupyter \
  notebook \
  xgboost \
  nltk \
  clean-text \
  spacy \
  python-Levenshtein \
  gensim \
  h5py \
  ray \
  networkx \
  graphviz \
  tqdm \
  Jinja2 \
  Markdown \
  MarkupSafe \
  sqlfluff


