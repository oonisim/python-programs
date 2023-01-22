"""
TF/Keras ResNet50 helper class

[Note]
Input Shape (https://keras.io/api/applications/resnet)
    input_shape: the input shape has to be (224, 224, 3) (with 'channels_last' data format) or
    (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
"""
import logging
from typing import (
    List,
    Tuple,
    Sequence,
    Union,
    Optional,
    Callable,
)
import numpy as np
from keras.models import (
    Model,
)
import tensorflow as tf
from keras.applications import (
    ResNet50,
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input,
    decode_predictions
)

from util_logging import (
    get_logger
)
from util_numpy import (
    get_cosine_similarity,
)
from util_opencv.image import (
    validate_image,
    get_image_dimensions,
    convert_bgr_to_rgb,
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
RESNET50_IMAGE_VECTOR_SIZE: int = 2048


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def validate_resnet_input_image(image: np.ndarray) -> bool:
    """Check if the image is valid as ResNet input
    1. Shape is (W, D, 3) where W, D is at minimum 32
    2. The image height and image should be 32 or larger.
    """
    name: str = "validate_resent_input_image()"
    try:
        _h, _w, _d = get_image_dimensions(image=image)

        if _d is None:
            _logger.warning("%s: grey scale image to skip.", name)
            return False

        if _d != 3:
            _logger.warning("%s: incorrect channel size [%s].", name, _d)
            return False

        if _h < 32:
            _logger.warning("%s: height minimum 32 bot [%s].", name, _h)
            return False

        if _w < 32:
            _logger.warning("%s: width minimum 32 bot [%s].", name, _w)
            return False

    except RuntimeError as e:
        _logger.error("%s: encountered an error [%s]", name, e)
        return False

    return True


def preprocess_rgb_image_for_resnet(image: np.ndarray):
    """Transform np array of an image to match the input format of Keras ResNet.
    The color in the image MUST be RGB, then ResNet preprocess_input convert it to BGR
    which is then fed into the ResNet model.

    https://keras.io/api/applications/resnet/
    For ResNet, call tf.keras.applications.resnet.preprocess_input on your inputs before
    passing them to the model. preprocess_input will convert the input images from RGB
    to BGR, then will zero-center each color channel with respect to the ImageNet dataset,
    without scaling.

    Args:
        image: array with shape (N, 224, 224, 3) in RGB color channel order
    Returns:
    """
    # TF/Keras is batch based. Add the dimension for batch for an image of shape (244, 244, 3)
    assert isinstance(image, np.ndarray) and (image.ndim == 3 or image.ndim == 4)

    x: np.ndarray
    if image.ndim == 3:
        x = np.expand_dims(image, axis=0)
    else:
        x = image

    num, height, width, depth = x.shape
    assert num > 0 and height == 224 and width == 224 and depth == 3, \
        f"expected shape (N, 223, 224, 3) got {(num, height, width, depth)}"

    return preprocess_input(x)


# --------------------------------------------------------------------------------
# Class
# --------------------------------------------------------------------------------
class ResNet50Helper:
    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(self):
        self._model: Union[Model, None] = None
        self._feature_extractor: Union[Model, None] = None

    def load_model(self):
        """Load the ResNet50 model"""
        if self._model is None:
            self._model = ResNet50(weights='imagenet')

    def load_feature_extractor(self):
        """Load the model to generate embedding vector for images.
        Use the output of the avg_pool layer in ResNet50 to generate embedding vector for images
        """
        if self._feature_extractor is None:
            self._feature_extractor = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("avg_pool").output
            )

    def unload(self):
        if self._model:
            del self._model
            self._model = None
        if self._feature_extractor:
            del self._feature_extractor
            self._feature_extractor = None

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------
    @property
    def model(self) -> Model:
        """Get model"""
        if not self._model:
            self.load_model()

        return self._model

    @property
    def feature_extractor(self) -> Model:
        """Get feature extractor"""
        if not self._feature_extractor:
            self.load_feature_extractor()

        return self._feature_extractor

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def convert_BGR_to_RGB(self, images: np.ndarray) -> np.ndarray:
        vectorized_convert_bgr_to_rgb: Callable = np.vectorize(convert_bgr_to_rgb, signature="(h,w,d)->(h,w,d)")
        converted: np.ndarray
        if images.ndim == 3:
            validate_image(images)
            converted = convert_bgr_to_rgb(image=images)
        else:
            assert images.ndim == 4
            converted = vectorized_convert_bgr_to_rgb(images)

        return converted

    def decode(self, predictions, top: int = 3) -> List[Tuple[str, str, float]]:
        """Decodes the prediction of an ImageNet model
        https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/decode_predictions

        Args:
            predictions: prediction from ImageNet model
        Returns: prediction tuples (class_name, class_description, score).
        """
        return decode_predictions(predictions, top=top)

    def predict(
            self,
            images: np.ndarray,
            top: int = 3,
            bgr_to_rgb: bool = False
    ) -> List[Tuple[str, str, float]]:
        """classify the images
        Args:
            images: image in shape (224, 224, 3) or (N, 224, 224, 3)
            top: number of labels to return per prediction
            bgr_to_rgb: if BGR to RGB conversion is required
        Returns: prediction tuples (class_name, class_description, score).
        """
        images_to_predict: np.ndarray
        if bgr_to_rgb:
            images_to_predict = preprocess_rgb_image_for_resnet(self.convert_BGR_to_RGB(images))
        else:
            images_to_predict = preprocess_rgb_image_for_resnet(images)

        predictions: np.ndarray = self.model.predict(images_to_predict)
        del images_to_predict

        return self.decode(predictions, top=top)

    def embed(
            self,
            images: np.ndarray,
            bgr_to_rgb: bool = False
    ) -> np.ndarray:
        """Generate embedding vectors for the images
        Use the layers in ResNet50 before the classification and generate feature embedding vector

        Args:
            images:
                images to generate the embedding vectors. Data is in RGB by default.
                Set bgr_to_rgb to True of the data is in BGR
            bgr_to_rgb: if BGR to RGB conversion is required
        Returns: vectors of size 2048 (output of the avg_pool layer of ResNet50 model)
        """
        images_to_predict: np.ndarray
        if bgr_to_rgb:
            images_to_predict = preprocess_rgb_image_for_resnet(self.convert_BGR_to_RGB(images))
        else:
            images_to_predict = preprocess_rgb_image_for_resnet(images)

        return self.feature_extractor.predict(images_to_predict)


class Vectorizer:
    def __init__(self):
        self._resnet: Optional[ResNet50Helper] = None

    def fit(self):
        self._resnet = ResNet50Helper()

    def transform(self, images: Sequence[np.ndarray], bgr_to_rgb: bool = False) -> Optional[np.ndarray]:
        """Transform list of images into numpy vectors of image features.
        Images should be preprocessed first (padding, resize, normalize,..)

        Args:
            images:
                The sequence of raw images. By default, data in RGB order is expected.
                if it is BGR, then set bgr_to_rgb to True
            bgr_to_rgb:
                execute BGR to RGB transformation e.g. when the image is in BGR (default by OpenCV).
                ResNet requires RGB format to go into preprocess_input utility which converts RGB to BGR.

        Returns:
            Vectorized images as numpy array of (N, D) shape where
            N is the number of images, and D is feature vector
            size after running it through the vectorizer.
        """
        if self._resnet is None:
            print("run fit() method first.")
            return None

        name: str = "transform()"
        result: Union[np.ndarray, None] = None

        package: List[np.ndarray] = list()
        for img in images:
            if validate_resnet_input_image(image=img):
                if bgr_to_rgb:
                    # Make sure to convert BGR to RGB
                    package.append(convert_bgr_to_rgb(image=img))
                else:
                    package.append(img)

            else:
                _logger.warning("%s: skip an image...")
                continue
        # END for loop

        result = self._resnet.embed(preprocess_rgb_image_for_resnet(np.array(package)))
        assert result.shape == (len(package), RESNET50_IMAGE_VECTOR_SIZE), \
            f"expected shape [{(len(package), RESNET50_IMAGE_VECTOR_SIZE)}] got [{result.shape}]."
        del package

        return result


class ImageSearch:
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
