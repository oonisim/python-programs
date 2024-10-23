from torch.utils.data import (
    DataLoader
)


def get_image_data_mean_std(
        loader: DataLoader
):
    """Compute the mean and standard deviation of all pixels in the dataset.
    https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/

    Args:
        loader: data loader

    Returns: (mean, std) where mean.shape=(3,) and std.shape=(3,)
    """
    count: int = 0
    mean: float = 0.0
    std: float = 0.0

    for images, _ in loader:
        assert images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        # --------------------------------------------------------------------------------
        # Compute the mean and standard deviation for each channel separately
        # (e.g., one value for each of the RGB channels) by specifying axis=(0, 2, 3),
        # as the mean and standard deviation are computed across the batch, height,
        # and width dimensions, but not across the channel dimension.
        # --------------------------------------------------------------------------------
        count += 1
        mean += images.mean(axis=(0, 2, 3))
        std += images.std(axis=(0, 2, 3))

    mean /= count
    std /= count

    return mean, std
