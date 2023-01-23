#!/usr/bin/env bash
set -ue

DIR=$(realpath $(dirname $0))
DATA_DIR_MASTER="${DIR}/../data/original"
DATA_DIR_LANDING="${DIR}/../data/landing"
DATA_DIR_FEATURE="${DIR}/../data/feature_store"
DATA_DIR_MODEL="${DIR}/../data/model"

NPY_RESIZED_RGB="resized_rgb.npy"
NPY_FEATURE_ENGINEERED="feature_engineered.npy"
NPY_IMAGE_VECTORS="embedded_image_vectors.npy"

VECTORIZER_MODEL="vectorizer_model"  # Keras model saved by Model.save()

#--------------------------------------------------------------------------------
# ETL: Master -> Landing
#--------------------------------------------------------------------------------
#python etl.py \
#  --source-directory=${DATA_DIR_MASTER} \
#  --target-directory=${DATA_DIR_LANDING} \
#  --target-filename=${NPY_RESIZED_RGB} \
#  --log-level=10
#
##--------------------------------------------------------------------------------
## Feature Engineering: Landing -> Feature Store
##--------------------------------------------------------------------------------
#python feature_engineering.py \
#  --source-directory=${DATA_DIR_LANDING} \
#  --source-filename=${NPY_RESIZED_RGB} \
#  --target-directory=${DATA_DIR_FEATURE} \
#  --target-filename=${NPY_FEATURE_ENGINEERED} \
#  --log-level=10
#
#--------------------------------------------------------------------------------
# Training: Feature Store -> Model
#--------------------------------------------------------------------------------
python train.py \
  --source-directory=${DATA_DIR_FEATURE} \
  --source-filename=${NPY_FEATURE_ENGINEERED} \
  --target-directory=${DATA_DIR_MODEL} \
  --target-filename=${NPY_IMAGE_VECTORS} \
  --vectorizer-model-file=${VECTORIZER_MODEL} \
  --log-level=10
