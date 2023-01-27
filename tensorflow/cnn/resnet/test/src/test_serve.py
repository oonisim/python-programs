"""
Artifact test module
"""
import logging
import os
from typing import (
    List,
    Tuple,
)

import numpy as np

from config import (
    # Master
    DIR_DATA_MASTER,

    # Lansing
    DIR_DATA_LANDING,
    FILENAME_NPY_RESIZED_RGB,
    FILENAME_NPY_IMAGE_NAMES,

    # Feature Engineering
    # Modelling
    DIR_DATA_MODEL,
    FILENAME_NPY_IMAGE_VECTORS,
    TF_VECTORIZER_MODEL,
)
from serve import (
    ImageSearchEngine
)
from util_constant import (
    TYPE_FLOAT,
)
from util_file import (
    is_file,
    is_dir,
    list_files_in_directory,
)
from util_logging import (
    get_logger
)
from util_numpy import (
    is_all_close,
)
from util_opencv.image import (
    get_image,
    resize,
)
from util_tf.resnet50 import (
    RESNET50_IMAGE_HEIGHT,
    RESNET50_IMAGE_WIDTH,
    RESNET50_IMAGE_CHANNELS,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)

# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
NUM_IMAGES_TO_SEARCH: int = 5   # Number of images to search for the query image
NUM_TEST_IMAGES: int = 100
PRECISION: TYPE_FLOAT = TYPE_FLOAT(1e-8)


# --------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------
def does_execute_tests() -> bool:
    """Check if the artifacts are available for tests
    Tests for Image Search Engine requires artifacts to be generated during the modelling pipeline.

    Returns: bool
    """
    return (
            is_dir(DIR_DATA_MASTER) and
            is_file(path_to_file=os.sep.join([DIR_DATA_MASTER, "04000.jpg"])) and
            is_file(path_to_file=os.sep.join([DIR_DATA_LANDING, FILENAME_NPY_RESIZED_RGB])) and
            is_file(path_to_file=os.sep.join([DIR_DATA_LANDING, FILENAME_NPY_IMAGE_NAMES])) and
            is_file(path_to_file=os.sep.join([DIR_DATA_MODEL, FILENAME_NPY_IMAGE_VECTORS]))
    )


# Get image names array
if does_execute_tests():
    filenames: List[str] = list_files_in_directory(path=DIR_DATA_MASTER, pattern="*.jpg")
    num_files: int = len(filenames)
    resized_rgb_images: np.ndarray = np.load(file=os.sep.join([DIR_DATA_LANDING, FILENAME_NPY_RESIZED_RGB]))
    image_names: np.ndarray = np.load(file=os.sep.join([DIR_DATA_LANDING, FILENAME_NPY_IMAGE_NAMES]))
    image_vectors: np.ndarray = np.load(file=os.sep.join([DIR_DATA_MODEL, FILENAME_NPY_IMAGE_VECTORS]))


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------


def get_random_images(
        num_images: int = NUM_TEST_IMAGES,
        extra: List[str] = None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Get N random images
    Returns: (selected images, image names, indices)
    """
    assert 1 < num_images < len(image_names)
    indices: np.ndarray = np.random.choice(
        a=range(num_images), size=NUM_TEST_IMAGES, replace=False
    )
    if len(extra) > 0:
        indices_to_extra: np.ndarray = np.nonzero(np.in1d(ar1=image_names, ar2=extra))[0]
        _logger.debug("extra indices = %s", indices_to_extra)

        indices = np.r_[
            indices_to_extra,
            indices
        ]

    names: np.ndarray = image_names[indices]
    selected: List[np.ndarray] = []
    for name in names:
        try:
            image: np.ndarray = get_image(
                path=os.sep.join([DIR_DATA_MASTER, name])
            )
            selected.append(image)
        except ValueError as exception:
            assert f"invalid image:[{name}] exception:{exception}"

    return selected, names, indices


# --------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------
def test_resize():
    """
    Objective:
        Verify the util_opencv.image.resize function generates a resized RGB image
        that is the same with the one in NPY_RESIZED_RGB.

    Select a filename from the files in DIR_DATA_MASTER.
    Load the image of the filename and resize it -> resized_rgb_new

    Get the index of the filename in the NYP_IMAGE_NAMES -> index

    Get the already resized RGB image at the index in NPY_RESIZED_RGB -> resized_rgb_new

    Test Conditions.
    1. resized_rgb_new == resized_rgb_old (nearly equal due to floating calc)
    2. resized_rgb_new.shape == resized_rgb_old.shape

    """
    if not does_execute_tests():
        _logger.error("Run the test after the modelling pipeline has been executed.")
        return

    for _ in range(100):
        try:
            # --------------------------------------------------------------------------------
            # Random select an image and resize it
            # --------------------------------------------------------------------------------
            # Random pick a filename
            filename: str = filenames[np.random.randint(0, num_files-1)]
            _logger.debug("test_resize(): resizing [%s]", filename)

            # Load the image of the filename
            image: np.ndarray = get_image(
                path=os.sep.join([DIR_DATA_MASTER, filename])
            )
            # resize image
            result: dict = resize(
                img=image, height=RESNET50_IMAGE_HEIGHT, width=RESNET50_IMAGE_WIDTH, bgr_to_rgb=True
            )
            if result['image'] is None:
                _logger.error("failed to resize image [%s]", filename)
                return

            resized_rgb_new: np.ndarray = result['image']

            # --------------------------------------------------------------------------------
            # Load the image for the same filename from NPY_RGB_RESIZED
            # --------------------------------------------------------------------------------
            # index to the image name
            index: int = np.nonzero(np.in1d(image_names, [filename]))[0][0]
            _logger.debug("test_resize(): index to the file in NPY_IMAGE_NAMES [%s]", index)

            assert image_names[index] == filename
            resized_rgb_old: np.ndarray = resized_rgb_images[index]

            # --------------------------------------------------------------------------------
            # Test condition #1: resized and the one in NPY_RGB_RESIZED are nearly equal
            # --------------------------------------------------------------------------------
            assert is_all_close(x=resized_rgb_new, y=resized_rgb_old, atol=TYPE_FLOAT(1e-8)), \
                f"expected nearly equal but max diff is [{np.max((resized_rgb_new-resized_rgb_old).ravel())}]."

            # --------------------------------------------------------------------------------
            # Test condition #2: Shape of them are equal
            # --------------------------------------------------------------------------------
            assert (
                    np.squeeze(resized_rgb_new).shape ==
                    np.squeeze(resized_rgb_old).shape ==
                    (RESNET50_IMAGE_HEIGHT, RESNET50_IMAGE_WIDTH, RESNET50_IMAGE_CHANNELS)
            ), f"file[{filename}]: resized does not match with that in NPY_RESIZED_RGB"

            del resized_rgb_new, resized_rgb_old

        except ValueError as exception:
            _logger.warning("exception [%s]", exception)


def test_serve_transform():
    """Test ImageSearchEngine.transform() function

    Objective:
        Verify the transform generates an image vector that matches the
        corresponding vector in NPY_IMAGE_VECTORS.
        Verify the cosine_similarity generates 1.0 for the same image.

    Test Conditions:
    1. Image vectors generated by transform matches NPY_IMAGE_VECTOR
    2. Cosine similarity is 1.0 for the same image
    """
    if not does_execute_tests():
        _logger.error("Run the test after the modelling pipeline has been executed.")
        return

    images: List[np.ndarray]
    names: np.ndarray
    indices: np.ndarray
    images, names, indices = get_random_images(extra=["04000.jpg"])

    engine: ImageSearchEngine = ImageSearchEngine(
        image_vectors=image_vectors,
        path_to_vectorizer_model=os.sep.join([DIR_DATA_MODEL, TF_VECTORIZER_MODEL]),
        images=resized_rgb_images,
        image_names=image_names
    )

    failed_transform: List[Tuple[str, int]] = []
    failed_similarity: List[Tuple[str, int, float]] = []
    for image, name, index in zip(images, names, indices):
        # Currently ImageSearchEngine.transform() accepts single image only.
        _logger.debug("generating image vector for [%s] for index[%s]", name, index)
        image_vector: np.ndarray = engine.transform(query=image)

        # Test condition #1. Image vectors generated by transform matches NPY_IMAGE_VECTOR
        if not is_all_close(x=image_vector, y=image_vectors[index], atol=PRECISION):
            _logger.error("image vector of [%s] does not match image_vectors[%s]", name, index)
            failed_transform.append((name, index))

        # Test condition #2. Cosine similarity is 1.0 for the same image
        similarity: np.ndarray = engine.cosine_similarity(
            query_image_vector=image_vector,
            image_vectors=image_vector
        )
        similarity = np.squeeze(similarity)
        if not is_all_close(x=similarity, y=TYPE_FLOAT(1.0)):
            _logger.error(
                "image [%s] index[%s]: expected cosine similarity 1.0, got [%s].",
                name, index, similarity
            )
            failed_similarity.append((name, index, float(similarity)))

    assert len(failed_transform) == 0 and len(failed_similarity) == 0, \
        f"image vector mismatches: {failed_transform}\n" \
        f"cosine similarity mismatch: {failed_similarity}"
