"""
Feature engineering module

[Objective]
1. Load data processed at ETL.
2. Run feature engineering on the resized RGB data to transform into the features
   compatible with the ResNet input format.
3. Return and/or save the features generated.
"""
import os
import logging
from typing import (
    List,
    Dict,
    Sequence,
    Any,
    Union,
    Optional,
    Callable,
)
import numpy as np

from util_logging import (
    get_logger
)

from util_numpy import (
    load,
    save,
)
from util_opencv.image import (
    convert_bgr_to_rgb,
)
from util_tf.resnet50 import (
    validate_resnet_input_image,
    preprocess_rgb_image_for_resnet
)
from function import (
    ARG_LOG_LEVEL,
    ARG_SOURCE_DIR,
    ARG_SOURCE_FILE,
    ARG_TARGET_DIR,
    ARG_TARGET_FILE,
    parse_commandline_arguments,
    process_commandline_arguments,
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# FE
# --------------------------------------------------------------------------------
class FeatureEngineering:
    def __init__(self):
        self._is_fitted: bool = False
        self._feature_transformer: Optional[Callable] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data: np.ndarray):
        """Fit the instance to the data to run feature engineering on
        As TF/ResNet provides the utility instance for feature engineering (preprocess_input),
        nothing to do here.
        """
        self._is_fitted = True
        self._feature_transformer = preprocess_rgb_image_for_resnet

    def transform(
            self,
            data: np.ndarray,
            bgr_to_rgb: bool = False
    ) -> np.ndarray:
        """Transform landing data into features
        Args:
            data: sequence of resized RGB images.
            bgr_to_rgb:
                boolean flag if run the BGR to RGB transformation. If the image data is in BGR
                e.g. OpenCV default, then transform to RGB.

        Returns: list of image features in RGB to run the training on
        """
        name: str = "transform()"
        if not self.is_fitted:
            raise RuntimeError("not yet fitted")

        package: List[np.ndarray] = list()
        for index, img in enumerate(data):
            if validate_resnet_input_image(image=img):
                if bgr_to_rgb:
                    # Make sure to convert BGR to RGB
                    package.append(convert_bgr_to_rgb(image=img))
                else:
                    package.append(img)

            else:
                _logger.warning("%s: skip an invalid image at index [%s]...", name, index)
                continue
        # END for loop

        features: np.ndarray = self._feature_transformer(np.array(package))
        del package
        return features


# ================================================================================
# Main
# ================================================================================
def main():
    """Run the Feature Engineering process
    1. Load resized RGB data into memory as resized RGB images.
    2. Run the feature engineering for ResNet.
    3. Save the features as numpy npy file to disk to use in the model training.
    """
    args: Dict[str, Any] = process_commandline_arguments(parse_commandline_arguments())
    if args[ARG_LOG_LEVEL] is not None:
        _logger.setLevel(level=args[ARG_LOG_LEVEL])
        logging.basicConfig(level=args[ARG_LOG_LEVEL])
    if args[ARG_TARGET_FILE] is None:
        raise RuntimeError(f"need [{ARG_TARGET_FILE}] option")
    if args[ARG_SOURCE_FILE] is None:
        raise RuntimeError(f"need [{ARG_SOURCE_FILE}] option")

    # --------------------------------------------------------------------------------
    # 1. Load resized RGB data into memory as resized RGB images.
    # --------------------------------------------------------------------------------
    source_file_path: str = os.sep.join([args[ARG_SOURCE_DIR], args[ARG_SOURCE_FILE]])
    _logger.info("feature engineering is loading the resized RGB data from [%s]...", source_file_path)
    source: np.ndarray = load(path_to_file=source_file_path)

    # --------------------------------------------------------------------------------
    # 2. Feature engineering for ResNet to generate data for ResNet input layer.
    # --------------------------------------------------------------------------------
    feature_engineer: FeatureEngineering = FeatureEngineering()
    feature_engineer.fit(data=source)
    features: np.ndarray = feature_engineer.transform(data=source)

    # --------------------------------------------------------------------------------
    # 3. Save the features as numpy npy to disk.
    # --------------------------------------------------------------------------------
    target_file_path: str = os.sep.join([args[ARG_TARGET_DIR], args[ARG_TARGET_FILE]])
    _logger.info("feature engineering is saving the features to [%s]...", target_file_path)
    save(array=features, path_to_file=target_file_path)


if __name__ == "__main__":
    main()

