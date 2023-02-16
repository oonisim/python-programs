"""ML image utility module"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------
# CIFAR10
# https://www.cs.toronto.edu/~kriz/cifar.html
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
# with 6000 images per class. There are 50000 training images and 10000 test images.
# class_names = [
#   'airplane',
#   'automobile',
#   bird',
#   'cat',
#   'deer',
#   'dog',
#   'frog',
#   'horse',
#   'ship',
#   'truck'
# ]
# --------------------------------------------------------------------------------
def plot_image(image, width, height, channels, figsize=(6, 6)):
    reshaped = image.reshape(width, height, channels)
    plt.figure(figsize=figsize),
    plt.imshow(reshaped, cmap=plt.cm.binary)
    plt.axis("off")


def plot_images(
        images: np.ndarray,
        width: int,
        height: int,
        channels: int,
        images_per_row: int = 5,
        figsize=(6, 6),
        **options
):
    """
    Display images in a table format where each cell is a resized image of (height, width)
    Args:
        images: images to display
        height: image height to resize
        width: image width to resize
        channels: number of image color channels e.g. 3 for RGB
        images_per_row: number of images per row
        figsize: matplotlib figure size
        options:
    """
    plt.figure(figsize=figsize)

    images_per_row = min(len(images), images_per_row)
    images = [img.reshape(width, height, channels) for img in images]
    n_rows = (len(images) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(images)
    images.append(np.zeros((width, width * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")
