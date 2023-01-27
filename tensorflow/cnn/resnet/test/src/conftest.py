from config import (
    # Master
    DIR_DATA_MASTER,

    # Lansing
    DIR_DATA_LANDING,
    FILENAME_NPY_RESIZED_RGB,
    FILENAME_NPY_IMAGE_NAMES,

    # Feature Engineering
    DIR_DATA_FEATURE_STORE,
    FILENAME_NPY_FEATURE_ENGINEERED,

    # Modelling
    DIR_DATA_MODEL,
    FILENAME_NPY_IMAGE_VECTORS,
    TF_VECTORIZER_MODEL,
)


def pytest_addoption(parser):
    parser.addoption(
        "--data-directory-master",
        action="store",
        default=DIR_DATA_MASTER,
        help="path to master data directory"
    )

    # Feature engineered resized images
    parser.addoption(
        "--data-directory-landing",
        action="store",
        default=DIR_DATA_LANDING,
        help="path to landing data directory"
    )
    parser.addoption(
        "--filename-resized-rgb",
        action="store",
        default=FILENAME_NPY_RESIZED_RGB,
        help="filename of resized rgb images npy"
    )
    parser.addoption(
        "--filename-image-names",
        action="store",
        default=FILENAME_NPY_IMAGE_NAMES,
        help="filename of images names npy"
    )

    # Feature engineered resized images
    parser.addoption(
        "--data-directory-feature-store",
        action="store",
        default=DIR_DATA_FEATURE_STORE,
        help="path to feature store data directory"
    )
    parser.addoption(
        "--filename-feature-engineered",
        action="store",
        default=FILENAME_NPY_FEATURE_ENGINEERED,
        help="filename of feature engineered npy"
    )

    # Image vectors
    parser.addoption(
        "--data-directory-model",
        action="store",
        default=DIR_DATA_MODEL,
        help="path to model data directory"
    )
    parser.addoption(
        "--filename-image-vectors",
        action="store",
        default=FILENAME_NPY_IMAGE_VECTORS,
        help="filename of image vector npy"
    )
    parser.addoption(
        "--filename-vectorizer-model",
        action="store",
        default=TF_VECTORIZER_MODEL,
        help="path to vectorizer tf/keras model"
    )
