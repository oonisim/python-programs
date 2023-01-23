"""
Model serving (image search engine) module

[Objective]
Find the images based on the embedded image vector cosine similarities.

[Note]
To prevent training/serving skew (drifts), need to use th same artifacts
fitted to data for transformations (e.g. scaling, mean-centering, PCA).

[Artifacts]
1. NPY_IMAGE_VECTORS
   Embedded image vectors, each of which represents a resized RGB image
   in the multidimensional latent space. The dimension size depends on
   the vectorizer model, e.g. ResNet50 avg_pool layer output has 2048.
   Serialized with numpy.save() method.

2. NPY_RESIZED_RGB
   Images resized and transformed to have RGB channel order in memory.
   Serialized with numpy.save() method. Each row in the array matches
   with the image name in NPY_IMAGE_NAMES.

3. NPY_IMAGE_NAMES
   Names of the resized RGB images. Each row in the array matches
   with the image in NPY_RESIZED_RGB. Serialized with numpy.save().

4. TF_VECTORIZER_MODEL
   Vectorizer Keras Model instance used at modelling to vectorize the images
   into embedded image vectors. Serialized with the Keras Model.save() method
   with the default options.

"""
import logging
import os
from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
# from memory_profiler import (
#    profile as mprofile
# )
from feature_engineering import (
    FeatureEngineering,
)
from function import (
    ARG_LOG_LEVEL,
    ARG_SOURCE_DIR,
    ARG_SOURCE_FILE,
    ARG_IMAGE_DATA_DIR,
    ARG_IMAGE_DATA_FILE,
    ARG_IMAGE_NAME_FILE,
    ARG_IMG2VEC_MODEL_FILE,
    parse_commandline_arguments,
    process_commandline_arguments,
)
from model import (
    Vectorizer
)
from util_file import (
    is_file,
)
from util_logging import (
    get_logger
)
from util_numpy import (
    load,
    get_cosine_similarity,
)
from util_opencv.image import (
    get_image,
    resize,
    convert_bgr_to_rgb,
)
from util_tf.resnet50 import (
    RESNET50_IMAGE_HEIGHT,
    RESNET50_IMAGE_WIDTH,
    RESNET50_IMAGE_VECTOR_SIZE,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)

# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
NUM_IMAGES_TO_SEARCH: int = 5   # Number of images to search for the query image


# --------------------------------------------------------------------------------
# Search Engine
# --------------------------------------------------------------------------------
class ImageSearchEngine:
    """Image search implementation class"""
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
        M: int = query_image_vector.shape[0]    # pylint: disable=invalid-name
        D: int = query_image_vector.shape[1]    # pylint: disable=invalid-name
        N: int = image_vectors.shape[0]         # pylint: disable=invalid-name
        assert M == 1 and D == image_vectors.shape[1], \
            f"expected query_image_vector shape (1, {image_vectors.shape[1]} got {query_image_vector.shape}"

        # --------------------------------------------------------------------------------
        # Cosine similarity as dot product of unit vectors
        # --------------------------------------------------------------------------------
        similarity: np.ndarray = get_cosine_similarity(x=query_image_vector, y=image_vectors)
        assert len(similarity) > 0
        # assert np.any(-1 > similarity) or np.any(similarity > 1), f"{np.where(similarity > 1.0)}"

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
        """Get images from NPY_RESIZED_RGB for the matching nase
        Args:
            names: image names to find the matching image
        Returns: matched images
        """
        return self.get_images_at_indices(self.find_image_indices(names=names))

    def transform(self, query: np.ndarray) -> np.ndarray:
        """Apply the same processing at modelling pipeline
        1. Resize and convert to RGB.
        2. Run Feature engineering.
        3. Vectorize the image

        Args:
            query: single image to transform
        Returns: Embedded image vector
        """
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
            raise RuntimeError(f"failed resizing image due to {result['reason']}")

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

        assert not np.any(np.isnan(vectorized))
        return vectorized

    # @mprofile
    def most_similar(
            self,
            query: np.ndarray,
            n: int = 5              # pylint: disable=invalid-name
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
        # Take N most similar
        # --------------------------------------------------------------------------------
        similarities: np.ndarray = np.squeeze(self.cosine_similarity(
            query_image_vector=vectorized, image_vectors=self._image_vectors
        ))   # similarities is of shape (1, D). Get (D, )
        # np.argsort is ascending only.
        indices = np.argsort(similarities, axis=-1)[::-1][:np.minimum(len(similarities), n)]
        assert len(indices) == np.minimum(len(similarities), n)

        scores: List[float] = similarities[indices].tolist()
        imagee_names: List[str] = self._image_names[indices]
        result = list(zip(scores, imagee_names))

        _logger.info("%s: similar images and scores %s", name, result)
        return result


# ================================================================================
# Main
# ================================================================================
def display_images(
        query: np.ndarray,
        images: np.ndarray,
        names: List[str],
        scores: List[float]
):
    """Display the matched similar images with name and similarity score
    Args:
        query: image for which the similar images are found
        images: similar images found
        names: names of the images found
        scores: similarity scores for the images found
    """
    plt.rc('font', size=7)
    _, axes = plt.subplots(1, 1+len(images), figsize=(10, 6))

    # --------------------------------------------------------------------------------
    # Query image at the first column
    # --------------------------------------------------------------------------------
    axes[0].axis('off')
    axes[0].imshow(convert_bgr_to_rgb(query))
    axes[0].title.set_text("query")

    # --------------------------------------------------------------------------------
    # Similar images with the name and score (rounded to 4 decimals)
    # --------------------------------------------------------------------------------
    for index in range(1, len(images)+1):
        title: str = f"{names[index-1]}\nscore {round(scores[index-1], 4)}"
        axes[index].axis('off')
        axes[index].imshow(images[index-1])
        axes[index].title.set_text(title)

    plt.tight_layout()
    plt.show()


def interactive_image_search(engine: ImageSearchEngine):
    """Command line interactive search UI
    Args:
        engine: image search engine to use
    """
    while True:
        try:
            # --------------------------------------------------------------------------------
            # path to query image
            # --------------------------------------------------------------------------------
            print("input path to an image to search similarity.")
            path_to_image: str = input()
            if not is_file(path_to_file=path_to_image):
                continue

            # --------------------------------------------------------------------------------
            # query image array in BGR
            # --------------------------------------------------------------------------------
            query: np.ndarray = get_image(path=path_to_image)

            # --------------------------------------------------------------------------------
            # Get similar image names and scores
            # --------------------------------------------------------------------------------
            scores: List[float] = []
            names: List[str] = []
            for score, name in engine.most_similar(query=query, n=NUM_IMAGES_TO_SEARCH):
                scores.append(float(score))
                names.append(str(name))

            assert len(names) == NUM_IMAGES_TO_SEARCH

            # --------------------------------------------------------------------------------
            # Display similar images
            # --------------------------------------------------------------------------------
            images: np.ndarray = engine.get_images_for_names(names=names)
            assert len(images) == NUM_IMAGES_TO_SEARCH
            display_images(
                query=query,
                images=images,
                names=names,
                scores=scores
            )

        except (ValueError, RuntimeError, OSError, AssertionError) as exception:
            print(f"error due to {exception}")
            continue
        except (KeyboardInterrupt, EOFError):   # user CTRL-C or D to stop
            break


# @mprofile
def main():
    """Run the image search
    """
    args: Dict[str, Any] = process_commandline_arguments(parse_commandline_arguments())
    print(f"log leavel is {args[ARG_LOG_LEVEL]}")

    if args[ARG_LOG_LEVEL] is not None:
        logging.basicConfig(level=args[ARG_LOG_LEVEL])
        _logger.setLevel(level=args[ARG_LOG_LEVEL])
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
    # 1. Load image vectors (NPY_IMAGE_VECTORS)
    # --------------------------------------------------------------------------------
    source_file_path: str = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_SOURCE_FILE]])
    _logger.info("search engine is loading the image vectors from [%s]...", source_file_path)
    image_vectors: np.ndarray = load(path_to_file=source_file_path)

    # --------------------------------------------------------------------------------
    # 2. Load image names and images (NPY_RESIZED_RGB)
    # Note the channel order is RGB in memory and saved with the order in disk.
    # Need RGB to BGR conversion when using with Open CV.
    # --------------------------------------------------------------------------------
    image_data_file_path: str = os.sep.join([args[ARG_IMAGE_DATA_DIR], args[ARG_IMAGE_DATA_FILE]])
    _logger.info("search engine is loading the image data from [%s]...", image_data_file_path)
    image_data: np.ndarray = load(path_to_file=image_data_file_path)

    # --------------------------------------------------------------------------------
    # 3. Load image names (NPY_IMAGE_NAMES)
    # --------------------------------------------------------------------------------
    image_name_file_path: str = os.sep.join([args[ARG_IMAGE_DATA_DIR], args[ARG_IMAGE_NAME_FILE]])
    _logger.info("search engine is loading the image names from [%s]...", image_name_file_path)
    image_names: np.ndarray = load(path_to_file=image_name_file_path)

    # --------------------------------------------------------------------------------
    # 4. Location of the vectorizer model (TF_VECTORIZER_MODEL)
    # Vectorizer loads the model with load() method.
    # --------------------------------------------------------------------------------
    img2vec_model_path = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_IMG2VEC_MODEL_FILE]])

    # --------------------------------------------------------------------------------
    # Start the search engine
    # --------------------------------------------------------------------------------
    _logger.info("starting image search engine...")
    engine: ImageSearchEngine = ImageSearchEngine(
        image_vectors=image_vectors,
        path_to_vectorizer_model=img2vec_model_path,
        images=image_data,
        image_names=image_names
    )

    _logger.info("starting interactive search...")
    interactive_image_search(engine=engine)


if __name__ == "__main__":
    main()
