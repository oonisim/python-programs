"""
Training module

[Objective]
Generate the model (image embedding vectors) from feature-engineered images
"""
import os
import logging
from typing import (
    Dict,
    Any,
    Sequence,
    Optional,
)

import numpy as np
from keras.models import (
    Model,
)
from tensorflow import keras

from util_file import (
    mv_file,
)
from function import (
    ARG_LOG_LEVEL,
    ARG_SOURCE_DIR,
    ARG_SOURCE_FILE,
    ARG_TARGET_DIR,
    ARG_TARGET_FILE,
    ARG_IMG2VEC_MODEL_FILE,
    parse_commandline_arguments,
    process_commandline_arguments,
)
from util_logging import (
    get_logger
)
from util_numpy import (
    load,
    save,
)
from util_tf.resnet50 import (
    RESNET50_IMAGE_VECTOR_SIZE,
    ResNet50Helper,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------
class Vectorizer:
    def __init__(self):
        self._resnet: Optional[ResNet50Helper] = None
        self._image2vec: Optional[Model] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, images: Sequence[np.ndarray]):
        """Fit the instance to the data to run feature engineering on
        As TF/ResNet provides the utility instance for feature engineering (preprocess_input),
        nothing to do here.
        """
        self._resnet: ResNet50Helper = ResNet50Helper()
        self._image2vec: Model = self._resnet.feature_extractor
        self._resnet.unload_model()
        self._is_fitted: bool = True

    def transform(self, images: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        """Transform list of images into numpy vectors of image features.
        Images should be preprocessed first (padding, resize, normalize,..).

        The results are embedded vectors each of which represents an image
        in a multidimensional space where proximity represents the similarity
        of the images.

        Args:
            images: The sequence of images to transform

        Returns:
            Vectorized images as numpy array of (N, D) shape where
            N is the number of images, and D is feature vector
            size after running it through the vectorizer.

        Raises: RuntimeError when not fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("not yet fitted")

        embedding: np.ndarray = self._image2vec.predict(images)
        assert embedding.shape == (len(images), RESNET50_IMAGE_VECTOR_SIZE), \
            f"expected shape [{(len(images), RESNET50_IMAGE_VECTOR_SIZE)}] got [{embedding.shape}]."

        return embedding

    def save(self, path_to_save: str):
        self._image2vec.save(path_to_save)

    def load(self, path_to_load):
        self._image2vec: Model = keras.models.load_model(path_to_load)
        self._is_fitted: bool = True


# ================================================================================
# Main
# ================================================================================
def main():
    """Run the Feature Engineering process
    1. Load features ready for ResNet input layer from saved npy file.
    2. Generate image embedding vectors with the ResNet feature extractor.
    3. Save the embedding vectors as npy to use in search.
    """
    args: Dict[str, Any] = process_commandline_arguments(parse_commandline_arguments())
    if args[ARG_LOG_LEVEL] is not None:
        _logger.setLevel(level=args[ARG_LOG_LEVEL])
        logging.basicConfig(level=args[ARG_LOG_LEVEL])
    if args[ARG_TARGET_FILE] is None:
        raise RuntimeError(f"need [{ARG_TARGET_FILE}] option")
    if args[ARG_SOURCE_FILE] is None:
        raise RuntimeError(f"need [{ARG_SOURCE_FILE}] option")
    if args[ARG_IMG2VEC_MODEL_FILE] is None:
        raise RuntimeError(f"need [{ARG_IMG2VEC_MODEL_FILE}] option")

    # --------------------------------------------------------------------------------
    # 1. Load features ready for ResNet input layer from saved npy file.
    # --------------------------------------------------------------------------------
    source_file_path: str = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_SOURCE_FILE]])
    _logger.info("vectorizer is loading the features from [%s]...", source_file_path)
    source: np.ndarray = load(path_to_file=source_file_path)

    # --------------------------------------------------------------------------------
    # 2. Feature engineering for ResNet to generate data for ResNet input layer.
    # --------------------------------------------------------------------------------
    vectorizer: Vectorizer = Vectorizer()
    vectorizer.fit(images=source)
    embeddings: np.ndarray = vectorizer.transform(images=source)
    vectorizer.save(path_to_save=f"{args[ARG_TARGET_DIR]}{os.sep}{args[ARG_IMG2VEC_MODEL_FILE]}")

    # --------------------------------------------------------------------------------
    # 3. Save the features as numpy npy to disk.
    # --------------------------------------------------------------------------------
    target_file_path: str = os.sep.join([args[ARG_TARGET_DIR], args[ARG_TARGET_FILE]])
    _logger.info("vectorizer is saving the embedding to [%s]...", target_file_path)
    save(array=embeddings, path_to_file=target_file_path)


if __name__ == "__main__":
    main()
