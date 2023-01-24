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

#--------------------------------------------------------------------------------
# PYTHONPATH: Set path to the library directory
#--------------------------------------------------------------------------------
export PYTHONPATH="${DIR}/../../../../lib"

#--------------------------------------------------------------------------------
# Directory structure
#--------------------------------------------------------------------------------
# Master
DATA_DIR_MASTER="${DIR}/../data/master"           # Original image data

# Lansing
DATA_DIR_LANDING="${DIR}/../data/landing"         # ETL output
NPY_RESIZED_RGB="resized_rgb.npy"                 # Resized RGB order images
NPY_IMAGE_NAMES="image_names.npy"                 # Image names in npy

# Feature Engineering
DATA_DIR_FEATURE="${DIR}/../data/feature_store"   # Feature engineering output
NPY_FEATURE_ENGINEERED="feature_engineered.npy"   # ResNet preprocessed images

# Modelling
DATA_DIR_MODEL="${DIR}/../data/model"             # Modelling output
NPY_IMAGE_VECTORS="embedded_image_vectors.npy"    # Embedded image vectors
TF_VECTORIZER_MODEL="vectorizer_model"            # Vectorizer Keras Model saved
