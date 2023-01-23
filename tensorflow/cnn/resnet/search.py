import os
import logging
from typing import (
    List,
    Dict,
    Tuple,
    Any,
)
import numpy as np

from util_logging import (
    get_logger
)
from util_numpy import (
    load,
    save,
    get_cosine_similarity,
)
from util_opencv.image import (
    get_image,
    resize,
)
from util_tf.resnet50 import (
    RESNET50_IMAGE_HEIGHT,
    RESNET50_IMAGE_WIDTH,
)
from function import (
    ARG_LOG_LEVEL,
    ARG_SOURCE_DIR,
    ARG_SOURCE_FILE,
    ARG_TARGET_DIR,
    ARG_TARGET_FILE,
    ARG_IMAGE_DATA_DIR,
    ARG_IMAGE_DATA_FILE,
    ARG_IMAGE_NAME_FILE,
    ARG_IMG2VEC_MODEL_FILE,
    parse_commandline_arguments,
    process_commandline_arguments,
)
from feature_engineering import (
    FeatureEngineering,
)
from train import (
    Vectorizer
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


class ImageSearchEngine:
    @staticmethod
    def cosine_similarity(query_image_vector: np.ndarray, image_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between image vectors.
        D is feature vector dimensionality (e.g. 1024)
        N is the number of images in the batch.
        Args:
            query_image_vector: Vectorized image query of (1, D) shape.
            image_vectors: Vectorized images batch of (N, D) shape.

        Returns:
            The vector of (1, N) shape with values in range [-1, 1] where
            1 is max similarity i.e. two vectors are the same.
        """
        M: int = query_image_vector.shape[0]
        D: int = query_image_vector.shape[1]
        N: int = image_vectors.shape[0]
        assert M == 1 and D == image_vectors.shape[1], \
            f"expected query_image_vector shape (1, {image_vectors.shape[1]} got {query_image_vector.shape}"

        similarity: np.ndarray = get_cosine_similarity(x=query_image_vector, y=image_vectors)
        assert -1 <= similarity <= 1
        assert similarity.shape == (1, N), f"expected cosine similarity shape {(1, N)} got {similarity.shape}"

        return similarity

    def __init__(
            self,
            image_vectors: np.ndarray,
            path_to_vectorizer_model: str,
            images: np.ndarray,
            image_names: np.ndarray
    ):
        # --------------------------------------------------------------------------------
        # Re-instantiate the artifacts used in training to appoly the same transformations.
        # --------------------------------------------------------------------------------
        self._images: np.ndarray = images
        self._image_names: np.ndarray = image_names

        # Feature Engineering instance
        self._feature_engineer: FeatureEngineering = FeatureEngineering()
        self._feature_engineer.load()

        # Vectorizer (image2vec model)
        self._vectorizer: Vectorizer = Vectorizer()
        self._vectorizer.load(path_to_load=path_to_vectorizer_model)

        # Embedded image vectors
        self._image_vectors = image_vectors


    def most_similar(
            self,
            query: np.ndarray,
            n: int = 5
    ) -> List[Tuple[float, str]]:
        """
        Return top n most similar images from corpus.
        Input image should be cleaned and vectorized with fitted Vectorizer to get query image vector. After that, use
        the cosine_similarity function to get the top n most similar images from the data set.

        Args:
            query: The raw query image input from the user
            n: The number of similar image names returned from the corpus
        Returns:
        """
        name: str = "most_similar()"

        # --------------------------------------------------------------------------------
        # Apply the same processing at training pipeline
        # 1. Resize and convert to RGB.
        # 2. Run Feature engineering.
        # --------------------------------------------------------------------------------
        result: dict = resize(
            img=query, height=RESNET50_IMAGE_HEIGHT, width=RESNET50_IMAGE_WIDTH, bgr_to_rgb=True
        )
        if result['image'] is None:
            _logger.error("%s: failed resizing image due to [%s]", name, result['reason'])
            return list()

        self.cosine_similarity(query_image_vector=)


# ================================================================================
# Main
# ================================================================================
def main():
    """Run the image search
    1. Load image vectors from the npy file.
    2. Start ImageSearchEngine
    """
    args: Dict[str, Any] = process_commandline_arguments(parse_commandline_arguments())
    if args[ARG_LOG_LEVEL] is not None:
        _logger.setLevel(level=args[ARG_LOG_LEVEL])
        logging.basicConfig(level=args[ARG_LOG_LEVEL])
    if args[ARG_TARGET_FILE] is None:
        raise RuntimeError(f"need [{ARG_TARGET_FILE}] option")
    if args[ARG_SOURCE_FILE] is None:
        raise RuntimeError(f"need [{ARG_SOURCE_FILE}] option")

    if args[ARG_IMAGE_DATA_DIR] is None:
        raise RuntimeError(f"need [{ARG_IMAGE_DATA_DIR}] option")
    if args[ARG_IMAGE_DATA_FILE] is None:
        raise RuntimeError(f"need [{ARG_IMAGE_DATA_FILE}] option")
    if args[ARG_IMAGE_NAME_FILE] is None:
        raise RuntimeError(f"need [{ARG_IMAGE_NAME_FILE}] option")

    if args[ARG_IMG2VEC_MODEL_FILE] is None:
        raise RuntimeError(f"need [{ARG_IMG2VEC_MODEL_FILE}] option")

    # --------------------------------------------------------------------------------
    # 1. Load image vectors
    # --------------------------------------------------------------------------------
    source_file_path: str = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_SOURCE_FILE]])
    _logger.info("search engine is loading the image vectors from [%s]...", source_file_path)
    image_vectors: np.ndarray = load(path_to_file=source_file_path)

    image_name_file_path: str = os.sep.join([args[ARG_IMAGE_DATA_DIR], args[ARG_IMAGE_NAME_FILE]])
    _logger.info("search engine is loading the image names from [%s]...", image_name_file_path),
    image_names: np.ndarray = load(path_to_file=image_name_file_path)

    image_data_file_path: str = os.sep.join([args[ARG_IMAGE_DATA_DIR], args[ARG_IMAGE_DATA_FILE]])
    _logger.info("search engine is loading the image data from [%s]...", image_data_file_path),
    image_data: np.ndarray = load(path_to_file=image_data_file_path)

    img2vec_model_path = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_IMG2VEC_MODEL_FILE]])

    search: ImageSearchEngine = ImageSearchEngine(
        image_vectors=image_vectors,
        path_to_vectorizer_model=img2vec_model_path,
        images=image_data,
        image_names=image_names
    )


if __name__ == "__main__":
    main()

