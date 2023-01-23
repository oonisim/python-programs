import os
import sys
import logging
from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Union,
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
    RESNET50_IMAGE_VECTOR_SIZE,
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


# --------------------------------------------------------------------------------
# Search Engine
# --------------------------------------------------------------------------------
class ImageSearchEngine:
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------------
        # single query image vector is numpy array of size RESNET50_IMAGE_VECTOR_SIZE.
        # --------------------------------------------------------------------------------
        assert isinstance(query_image_vector, np.ndarray), f"got type [{type(query_image_vector)}"
        assert query_image_vector.size == RESNET50_IMAGE_VECTOR_SIZE, \
            f"expected query image vector as of size {RESNET50_IMAGE_VECTOR_SIZE} " \
            f"got [{query_image_vector.size}]"
        # --------------------------------------------------------------------------------
        # image vectors is numpy array of shape (N, RESNET50_IMAGE_VECTOR_SIZE)
        # --------------------------------------------------------------------------------
        assert isinstance(image_vectors, np.ndarray), f"got type [{type(image_vectors)}"
        assert image_vectors.ndim == 2, f"expected image vector is 2D got [{image_vectors.ndim}]."

        # --------------------------------------------------------------------------------
        # Reshape the single query image vector as (1, RESNET50_IMAGE_VECTOR_SIZE)
        # --------------------------------------------------------------------------------
        query_image_vector = query_image_vector.reshape(1, -1)

        # --------------------------------------------------------------------------------
        # For matmul/dot operation, the shape match (1, D)-(D, N) is required to get (1, N)
        # --------------------------------------------------------------------------------
        M: int = query_image_vector.shape[0]
        D: int = query_image_vector.shape[1]
        N: int = image_vectors.shape[0]
        assert M == 1 and D == image_vectors.shape[1], \
            f"expected query_image_vector shape (1, {image_vectors.shape[1]} got {query_image_vector.shape}"

        # --------------------------------------------------------------------------------
        # Cosine similarity as dot product of unit vectors
        # --------------------------------------------------------------------------------
        similarity: np.ndarray = get_cosine_similarity(x=query_image_vector, y=image_vectors)
        assert np.all(-1 <= similarity) and np.all(similarity <= 1)
        assert similarity.shape == (1, N), f"expected cosine similarity shape {(1, N)} got {similarity.shape}"

        return similarity

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
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

    def find_image_indices(self, names: List[str]) -> np.ndarray:
        """Find the indices in the image_names array for the image names
        e.g. index is 0 for image name 00000.jpg
        Args:
            names: name of the images
        Returns: array of indices
        """
        return np.in1d(self._image_names, names).nonzero()[0]

    def get_images_at_indices(self, indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        """Get images at indices
        Args:
            indices: indices to image locations
        Returns: array of images
        """
        assert len(indices) > 0 and np.max(indices) < len(self._images)
        return self._images[indices]

    def get_images_for_names(self, names: List[str]) -> np.ndarray:
        return self.get_images_at_indices(self.find_image_indices(names=names))

    def transform(self, query: np.ndarray):
        # --------------------------------------------------------------------------------
        # Apply the same processing at training pipeline
        # 1. Resize and convert to RGB.
        # 2. Run Feature engineering.
        # --------------------------------------------------------------------------------
        name: str = "transform()"
        assert query.ndim == 3, "image data should be single and of shape (H, W, C)"

        # --------------------------------------------------------------------------------
        # resize and to RGB. The image requires shape (H, W, C) for OpenCV to handle.
        # --------------------------------------------------------------------------------
        result: dict = resize(
            img=query, height=RESNET50_IMAGE_HEIGHT, width=RESNET50_IMAGE_WIDTH, bgr_to_rgb=True
        )
        if result['image'] is None:
            _logger.error("%s: failed resizing image due to [%s]", name, result['reason'])
            return list()
        resized: np.ndarray = result['image']

        # --------------------------------------------------------------------------------
        # Feature engineering
        # --------------------------------------------------------------------------------
        feature_engineered: np.array = self._feature_engineer.transform(data=resized, bgr_to_rgb=False)

        # --------------------------------------------------------------------------------
        # Vectorize
        # --------------------------------------------------------------------------------
        vectorized: np.ndarray = self._vectorizer.transform(images=feature_engineered)
        _expected_shape = (1, RESNET50_IMAGE_VECTOR_SIZE)
        assert vectorized.shape == _expected_shape, \
            f"expected vectorized single image vector shape {_expected_shape}, got {vectorized.shape}"

        return vectorized

    def most_similar(
            self,
            query: np.ndarray,
            n: int = 5
    ) -> List[Tuple[float, str]]:
        """
        Return top n most similar images from corpus.
        Input image should be cleaned and vectorized with fitted Vectorizer to get query image vector.
        Then, use the cosine_similarity function to get the top n most similar images from the data set.

        Args:
            query: The raw query image input from the user in BGR order in memory as with OpenCV
            n: The number of similar image names returned from the corpus
        Returns:
        """
        assert n > 0

        name: str = "most_similar()"
        # --------------------------------------------------------------------------------
        # Apply the same processing at training pipeline
        # --------------------------------------------------------------------------------
        vectorized: np.ndarray = self.transform(query=query)

        # --------------------------------------------------------------------------------
        # Take 5 most similar
        # --------------------------------------------------------------------------------
        similarities: np.ndarray = np.squeeze(self.cosine_similarity(
            query_image_vector=vectorized, image_vectors=self._image_vectors
        ))   # similarities is of shape (1, D). Get (D, )
        # np.argsort is ascending only.
        print(similarities[:5])
        indices = np.argsort(similarities, axis=-1)[::-1][:np.minimum(len(similarities), n)]
        print(f"indices {indices} similarties[0] {similarities[:5]}")
        scores: List[float] = similarities[indices].tolist()
        imagee_names: List[str] = self._image_names[indices]
        print(f"scores: {scores}")

        result = [(score, name) for (score, name) in zip(scores, imagee_names)]
        _logger.info("%s: similar images %s", name, result)
        return result


# ================================================================================
# Main
# ================================================================================
def interactive_image_search(engine: ImageSearchEngine):
    while True:
        try:
            print("input path to an image to search similarity.")
            path_to_image: str = input()
            query: np.ndarray = get_image(path=path_to_image)

            scores: List[float] = list()
            names: List[str] = list()
            for score, name in engine.most_similar(query=query):
                scores.append(score)
                names.append(name)

            images: np.ndarray = engine.get_images_for_names(names=names)

        except (ValueError, RuntimeError, OSError) as e:
            print(f"error {e}")
            continue
        except (KeyboardInterrupt, EOFError):
            break

        finally:
            del engine

def main():
    """Run the image search
    """
    args: Dict[str, Any] = process_commandline_arguments(parse_commandline_arguments())
    if args[ARG_LOG_LEVEL] is not None:
        _logger.setLevel(level=args[ARG_LOG_LEVEL])
        logging.basicConfig(level=args[ARG_LOG_LEVEL])
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

    # --------------------------------------------------------------------------------
    # 2. Load image names and images (resized/RGB)
    # --------------------------------------------------------------------------------
    image_name_file_path: str = os.sep.join([args[ARG_IMAGE_DATA_DIR], args[ARG_IMAGE_NAME_FILE]])
    _logger.info("search engine is loading the image names from [%s]...", image_name_file_path),
    image_names: np.ndarray = load(path_to_file=image_name_file_path)

    image_data_file_path: str = os.sep.join([args[ARG_IMAGE_DATA_DIR], args[ARG_IMAGE_DATA_FILE]])
    _logger.info("search engine is loading the image data from [%s]...", image_data_file_path),
    image_data: np.ndarray = load(path_to_file=image_data_file_path)

    # --------------------------------------------------------------------------------
    # 3. Location of the vectorizer model (img2vec) saved at train.
    # --------------------------------------------------------------------------------
    img2vec_model_path = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_IMG2VEC_MODEL_FILE]])

    # --------------------------------------------------------------------------------
    # Start the search engine
    # --------------------------------------------------------------------------------
    engine: ImageSearchEngine = ImageSearchEngine(
        image_vectors=image_vectors,
        path_to_vectorizer_model=img2vec_model_path,
        images=image_data,
        image_names=image_names
    )

    interactive_image_search(engine=engine)


if __name__ == "__main__":
    main()

