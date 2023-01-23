#!/usr/bin/env bash
#--------------------------------------------------------------------------------
# Run serving pipeline (image search)
#--------------------------------------------------------------------------------
set -ue
DIR=$(realpath $(dirname $0))

#--------------------------------------------------------------------------------
# Run the prediction (image search) pipeline
#--------------------------------------------------------------------------------
. ${DIR}/_common.sh

#--------------------------------------------------------------------------------
# Image Search
#--------------------------------------------------------------------------------
# python -m memory_profiler serve.py \
# mprof run python serve.py \
python serve.py \
  --source-directory=${DATA_DIR_MODEL} \
  --source-filename=${NPY_IMAGE_VECTORS} \
  --target-directory=${DATA_DIR_MODEL} \
  --vectorizer-model-file=${TF_VECTORIZER_MODEL} \
  --image-data-dir=${DATA_DIR_LANDING} \
  --image-data-file=${NPY_RESIZED_RGB} \
  --image-name-file="image_names.npy" \
  --log-level=10
