#!/usr/bin/env bash
#--------------------------------------------------------------------------------
# Run the (ETL, Feature Engineering, Modelling) pipeline.
# 1. (Master->ETL->Landing)
# 2. (Landing->FE->FeatureStore)
# 3. (FeatureStore->Modelling->Model)
#
# [Terminologies]
# FE=Feature Engineering
# npy=numpy serialised file
#--------------------------------------------------------------------------------
set -ue
DIR=$(realpath $(dirname $0))

#--------------------------------------------------------------------------------
# Directory structure
#--------------------------------------------------------------------------------
. ${DIR}/_common.sh


#--------------------------------------------------------------------------------
# ETL: Master -> Landing
#--------------------------------------------------------------------------------
python etl.py \
  --source-directory=${DATA_DIR_MASTER} \
  --target-directory=${DATA_DIR_LANDING} \
  --target-filename=${NPY_RESIZED_RGB} \
  --log-level=10

#--------------------------------------------------------------------------------
# Feature Engineering: Landing -> Feature Store
#--------------------------------------------------------------------------------
python feature_engineering.py \
  --source-directory=${DATA_DIR_LANDING} \
  --source-filename=${NPY_RESIZED_RGB} \
  --target-directory=${DATA_DIR_FEATURE} \
  --target-filename=${NPY_FEATURE_ENGINEERED} \
  --log-level=10

#--------------------------------------------------------------------------------
# Training: Feature Store -> Model
#--------------------------------------------------------------------------------
python train.py \
  --source-directory=${DATA_DIR_FEATURE} \
  --source-filename=${NPY_FEATURE_ENGINEERED} \
  --target-directory=${DATA_DIR_MODEL} \
  --target-filename=${NPY_IMAGE_VECTORS} \
  --vectorizer-model-file=${TF_VECTORIZER_MODEL} \
  --log-level=40  # INFO

