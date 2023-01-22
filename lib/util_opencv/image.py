"""
OpenCV Image Utilities
Note:
    OpenCV assumes the image to be BGR or BGRA (BGR is the default OpenCV colour format).
    Other tools e.g. skimage.io.imread loads image as RGB (or RGBA).
    Make sure if the image in file is stored as RGB or BGR.

References:
    * https://stackoverflow.com/a/42406781/4281353
"""
import os
import logging
from typing import (
    Set,
    Tuple,
    Union,
    Callable
)

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from util_logging import (
    get_logger,
)
from util_file import (
    is_file,
    is_dir,
    mkdir,
    list_files_in_directory,
)


_logger: logging.Logger = get_logger(name=__name__)


def validate_image(image: np.ndarray) -> np.ndarray:
    """
    Check if the image is numpy of shape (height, width, depth)
    Args:
        image: image to check
    Raises: ValueError is image is not valid
    """
    assert isinstance(image, np.ndarray), f"invalid image data type [{type(image)}]"
    name: str = "validate_image()"

    if image.ndim == 3:
        return image
    else:
        _logger.error("%s: unexpected image shape [%s]", name, image.ndim)
        raise ValueError("image has no depth")


def can_handle_image(path: str) -> bool:
    """Check if OpenCV support the image
    Args:
        path: path to the imgge
    Returns: bool
    """
    return cv.haveImageReader(path)


def get_image(path: str, flags: Union[int, None] = None) -> np.ndarray:
    """Get image from the path
    Args:
        path: path to the image
        flags:
    """
    name: str = "get_image()"
    if can_handle_image(path=path):
        img = cv.imread(filename=path, flags=flags)
        if img is None:
            _logger.error("%s: failed to load the image [%s].", name, path)
            raise ValueError(f"OpenCV failed to read [{path}]")
    else:
        _logger.error("%s: OpenCV has no image reader for the image [%s].", name, path)
        raise ValueError(f"OpenCV has no image reader for the image [{path}].")
    return img


def convert_grey_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert Gray to RGB. If it is RGB, do nothing and return the image.
    Args:
        image: gray scale  image to convert
    Returns: RGB image
    """
    assert isinstance(image, np.ndarray)

    name: str = "convert_grey_to_rgb()"
    if image.ndim == 3:
        return image
    elif image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        _logger.error("%s: unexpected image shape [%s]", name, image.ndim)
        raise RuntimeError(f"image image shape [{image.shape}]")


def save_image(path: str, image: np.ndarray, flags: Union[int, None] = None):
    """Save the image to the path as BGR
    OpenCV assumes the image to be BGR or BGRA (BGR is the default OpenCV colour format)
    """
    # cv.imwrite(filename=path, img=cv.cvtColor(image, cv.COLOR_RGB2BGR))
    cv.imwrite(filename=path, img=image)


def get_dimensions(image: np.ndarray) -> Tuple[int, int, Union[int, None]]:
    """Get dimension (width, hegith, depth) of the image
    Args:
        image: image data loaded e.g. by cv.imread as type np.ndarray
    Returns:
        (height, width, depth)
    """
    if image.ndim == 3:         # Color RGB
        height, width, depth = image.shape
    elif image.ndim == 2:       # Gray
        height, width = image.shape
        depth = None
    else:
        raise RuntimeError(f"unexpected image shape [{image.shape}]")

    return height, width, depth


get_image_dimensions: Callable = get_dimensions


def show_image(image: np.ndarray, name: str = "", fontsize: int = None, show_axes=False) -> None:
    """Show image using matplotlib
    Convert to RGB for pyplot as OpenCV reads images with BGR format.
    See https://stackoverflow.com/a/70019644/4281353
    """
    validate_image(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.title(name, fontsize=fontsize)
    if not show_axes:
        plt.axis('off')
    plt.imshow(image)


def show_image_opencv(image: np.ndarray, name: str = "") -> None:
    """Show image using opencv.imshow(win) and close the window with 'q' key

    opencv.imshow may freeze or leave the python process unresponsive
    (may be specific to jupyter notebook)
    even with cv.destroyWindows() call followed by another waitKey().
    Use matplotlib instead if it happens.

    * https://stackoverflow.com/a/37041435/4281353
    * https://www.youtube.com/watch?v=guwebD9ENBY
    * https://github.com/opencv/opencv/issues/7343
    * https://forum.opencv.org/t/cv2-imshow-freeze/9264

    """
    validate_image(image)
    try:
        cv.startWindowThread()
        cv.namedWindow("preview")
        cv.imshow(winname=name, mat=image)
        while True:
            key = cv.waitKey(200) & 0xff
            if key in [ord("q"), ord("Q")]:
                break
    finally:
        cv.destroyAllWindows()
        cv.waitKey(1)


def resize_image(
        image: np.ndarray,
        width: int = 0,
        height: int = 0,
        interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    """Resize image to (width, height)
    """
    validate_image(image)
    return cv.resize(
        src=image,
        dsize=(width, height),
        fx=0,
        fy=0,
        interpolation=interpolation
    )


def resize_and_save_images(
        path_to_source: str,
        pattern: Union[str, None],
        height: int,
        width: int,
        path_to_destination: str,
        skip_image_error: bool = True
):
    """Resize the images in the source directory and save them to the destination.
    Args:
        path_to_source: source directory
        pattern: Pathlib glob pattern to filter files to process or None
        height: height to resize to
        width: width to resize to
        path_to_destination: destination directory
        skip_image_error: continue upon image processing errors
    Raises: RuntimeError for any filesystem errors
    """
    name = "resize_and_save_images()"
    assert is_dir(path_to_source), f"invalid source [{path_to_source}]."
    assert not is_file(path_to_destination), \
        f"destination [{path_to_destination}] is an existing file."
    assert height > 0 and width > 0, f"invalid height [{height}] or width [{width}]"

    try:
        # --------------------------------------------------------------------------------
        # Setup: Create destination directory if not exist.
        # --------------------------------------------------------------------------------
        if not is_dir(path_to_destination):
            mkdir(path=path_to_destination)

        # --------------------------------------------------------------------------------
        # List and process files
        # --------------------------------------------------------------------------------
        filenames: Set[str] = list_files_in_directory(
            path=path_to_source, pattern=pattern
        )
        if len(filenames) == 0:
            _logger.warning("%s: no file to resize in the directory [%s]", name, path_to_source)
        else:
            for _filename in filenames:
                source: str = os.sep.join([path_to_source, _filename])
                destination: str = os.sep.join([path_to_destination, _filename])
                resized: np.ndarray

                # --------------------------------------------------------------------------------
                # Load image to resize
                # --------------------------------------------------------------------------------
                try:
                    img: np.ndarray = get_image(path=source)
                    _h, _w, _d = get_image_dimensions(image=img)
                    if _d is None:  # Grey scale image
                        _logger.info("%s: skipping grey scale image [%s]...", name, _filename)
                        continue

                except ValueError as e:
                    _logger.info("%s: cannot handle and skip [%s]...", name, source)
                    continue

                if _h == height and _w == width:
                    # no change, do nothing
                    resized = img
                else:
                    # --------------------------------------------------------------------------------
                    # Interpolation option depending on if the operation is shrink or enlarge
                    # --------------------------------------------------------------------------------
                    if _h * _w > width * height:
                        interpolation: int = cv.INTER_AREA
                    else:
                        interpolation: int = cv.INTER_LINEAR

                    logging.debug("%s: resizing file [%s] to %s ...", name, source, (width, height))
                    resized = resize_image(
                        image=img, width=width, height=height, interpolation=interpolation
                    )
                del img

                _logger.debug("%s: saving image to [%s]", name, destination)
                save_image(path=destination, image=resized)

    except (OSError, RuntimeError) as e:
        _logger.error("%s: failed due to [%s]", name, e)


def rotate(img):
    # This programs calculates the orientation of an object.
    # The input is an image, and the output is an annotated image
    # with the angle of otientation for each object (0 to 180 degrees)

    from math import atan2, cos, sin, sqrt, pi

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)

    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    result = img.copy()
    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 5000 or 100000 < area:
            continue

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        if np.abs(angle % 90) > 0.1:
            # label = "  Rotation Angle: " + str(angle) + " degrees"
            # print(label)
            # textbox = cv.rectangle(
            #     result,
            #     (center[0]-35, center[1]-25),
            #     (center[0] + 295, center[1] + 10),
            #     (255,255,255),
            #     -1
            # )
            # cv.putText(result, label, (center[0]-50, center[1]),
            # cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv.LINE_AA)
            cv.drawContours(result,[box],0,(0,0,255),2)
            show_image(name=f"angle: {angle} degree", image=result)
            # Save the output image to the current directory
            # cv.imwrite(f"{i}.jpg", result)

