"""
ETL module.

[Objective]
1. Load images or omit the invalid images (See read_and_process_images).
2. Resize the image
"""
import logging
import os
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
    save,
)
from util_opencv.image import (
    read_and_process_images,
)
from util_tf.resnet50 import (
    RESNET50_IMAGE_HEIGHT,
    RESNET50_IMAGE_WIDTH,
)
from function import (
    ARG_LOG_LEVEL,
    ARG_SOURCE_DIR,
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
# ETL
# --------------------------------------------------------------------------------
class ETL:
    """ETL to extract original images, transform (resize/BGR2RGB), and save"""
    def __init__(
            self,
            path_to_source: str,
            path_to_destination: str,
            height_to_resize: int = 224,
            width_to_resize: int = 224,
            file_filter_pattern: str = r"*.jpg"
    ):
        """
        Args:
            path_to_source: Path to the file to load source data
            path_to_destination: Path to the file to save transformed data
            height_to_resize: height to resize to.
            width_to_resize: width to resize to
            file_filter_pattern: glob pattern to filter files by their names
        """
        self._path_to_source: str = path_to_source
        self._path_to_destination: str = path_to_destination
        self._height_to_resize: int = height_to_resize
        self._width_to_resize: int = width_to_resize
        self._file_filter_pattern: str = file_filter_pattern

    def extract(self):
        """Extract data from the data sources
        """
        pass

    def transform(self) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Load the image, resize, and transform to RGB"""
        resized_rgb_images: List[np.ndarray]
        processed_files: List[str]
        skipped_files: List[str]

        resized_rgb_images, processed_files, skipped_files = read_and_process_images(
            path_to_source=self._path_to_source,
            pattern=self._file_filter_pattern,
            do_resize=True,
            height=self._height_to_resize,
            width=self._width_to_resize,
            do_save=True,
            path_to_destination=self._path_to_destination,
            bgr_to_rgb=True,
            skip_image_error=True
        )
        return resized_rgb_images, processed_files, skipped_files


# ================================================================================
# Main
# ================================================================================
def main():
    """Run the ETL process
    1. Resize image and save them just to verify the transformation results.
       The individual resized image file will not be used at the feature engineering.
    2. Save the resized RGB data in memory as a numpy npy file, and use it at the
       feature engineering. Make sure feature engineering input is in RGB, not BGR.
    """
    args: Dict[str, Any] = process_commandline_arguments(parse_commandline_arguments())
    if args[ARG_LOG_LEVEL] is not None:
        _logger.setLevel(level=args[ARG_LOG_LEVEL])
        logging.basicConfig(level=args[ARG_LOG_LEVEL])
    if args[ARG_TARGET_FILE] is None:
        raise RuntimeError(f"need [{ARG_TARGET_FILE}] option")

    # --------------------------------------------------------------------------------
    # 1. Extract or omit images and resize them, and save images to TARGET_DIR as RGB.
    # --------------------------------------------------------------------------------
    _logger.info("ETL is loading and transforming the images from [%s]...", args[ARG_SOURCE_DIR])
    etl: ETL = ETL(
        path_to_source=args[ARG_SOURCE_DIR],
        path_to_destination=args[ARG_TARGET_DIR],
        height_to_resize=RESNET50_IMAGE_HEIGHT,
        width_to_resize=RESNET50_IMAGE_WIDTH,
        file_filter_pattern=r"*.jpg"
    )
    resized_rgb_images, processed_files, skipped_files = etl.transform()

    # --------------------------------------------------------------------------------
    # Save the resized RGB in-memory image data to disk to late be used at FE.
    # --------------------------------------------------------------------------------
    destination_file: str = os.sep.join([args[ARG_TARGET_DIR], args[ARG_TARGET_FILE]])
    _logger.info("ETL is saving the resized RGB into the npy file [%s]...", destination_file)
    save(array=np.array(resized_rgb_images), path_to_file=destination_file)

    name_file: str = os.sep.join([args[ARG_TARGET_DIR], "image_names.npy"])
    _logger.info("ETL is saving the resized image names into the npy file [%s]...", name_file)
    save(array=np.array(processed_files), path_to_file=name_file)


if __name__ == "__main__":
    main()
