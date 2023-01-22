import logging
from typing import (
    List,
    Tuple,
    Union,
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
from util_opencv.image import (
    validate_image,
    convert_bgr_to_rgb,
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


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

    def decode(self, predictions, top=3) -> List[Tuple[str, str, float]]:
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
            images: images to generate the embedding vectors
            bgr_to_rgb: if BGR to RGB conversion is required
        Returns: vectors of size 2048 (output of the avg_pool layer of ResNet50 model)
        """
        images_to_predict: np.ndarray
        if bgr_to_rgb:
            images_to_predict = preprocess_rgb_image_for_resnet(self.convert_BGR_to_RGB(images))
        else:
            images_to_predict = preprocess_rgb_image_for_resnet(images)

        return self.feature_extractor.predict(images_to_predict)
