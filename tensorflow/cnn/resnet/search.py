import logging
from typing import (
    List,
    Tuple,
)
import numpy as np

from util_logging import (
    get_logger
)
from util_numpy import (
    get_cosine_similarity,
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
        raise NotImplemented()