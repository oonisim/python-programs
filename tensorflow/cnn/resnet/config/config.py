import os
import pathlib
from util_file import (
    is_dir,
)


DIR_THIS: str = os.path.dirname(pathlib.Path(__file__).resolve())
assert is_dir(DIR_THIS)

DIR_DATA: str = os.sep.join([DIR_THIS, "..", "data"])
assert is_dir(DIR_DATA)

# Master
DIR_DATA_MASTER = os.sep.join([DIR_DATA, "master"])
assert is_dir(DIR_DATA_MASTER)

# Lansing
DIR_DATA_LANDING = os.sep.join([DIR_DATA, "landing"])
FILENAME_NPY_RESIZED_RGB = "resized_rgb.npy"
FILENAME_NPY_IMAGE_NAMES = "image_names.npy"
assert is_dir(DIR_DATA_LANDING)

# Feature Engineering
DIR_DATA_FEATURE_STORE = os.sep.join([DIR_DATA, "feature_store"])
FILENAME_NPY_FEATURE_ENGINEERED = "feature_engineered.npy"
assert is_dir(DIR_DATA_FEATURE_STORE)

# Modelling
DIR_DATA_MODEL = os.sep.join([DIR_DATA, "model"])
FILENAME_NPY_IMAGE_VECTORS = "embedded_image_vectors.npy"
TF_VECTORIZER_MODEL = "vectorizer_model"
assert is_dir(DIR_DATA_MODEL)
