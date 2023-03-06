#!/usr/bin/env bash
DIR=$(realpath .)
cd $DIR

# --------------------------------------------------------------------------------
# [Objective]
# Setup Python virtual environment for Mac Apple Silicon
#
# [History]
# 2023MAR06: Initial
# --------------------------------------------------------------------------------
if [[ "${VIRTUAL_ENV}x" == "x" ]]; then
  echo "Activate Virtual Environment First"
  exit -1
fi
export VIRTUAL_ENV="${VIRTUAL_ENV:?'Activate Virtual Environment First'}"

# --------------------------------------------------------------------------------
# To replace MacOS BLAS with Open BLAS for M2 to avoid Altivec issue
# --------------------------------------------------------------------------------
brew install openblas
export OPENBLAS="$(brew --prefix openblas)"
# export MACOSX_DEPLOYMENT_TARGET=13.0.1

# --------------------------------------------------------------------------------
# Latest pip
# --------------------------------------------------------------------------------
python3 -m pip install --upgrade pip

# --------------------------------------------------------------------------------
# Python setup for Mac M2
# --------------------------------------------------------------------------------
# https://stackoverflow.com/questions/75611977
pip install --no-cache-dir \
  setuptools \
  wheel \
  Cython \
  pyarrow==6.0.0 \
  numpy \
  tensorflow-transform

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
  tfx \
  tensorboard \
  transformers

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


