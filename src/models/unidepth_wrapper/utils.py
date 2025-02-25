import numpy as np
from PIL import Image
import torch
from unidepth.utils import colorize as unidepth_colorize


def process_image(image_path):
    """Read and preprocess image.

    Args:
        image_path (str): Path to input image

    Returns:
        numpy.ndarray: Preprocessed image in RGB format
    """
    # Read image
    img = np.array(Image.open(image_path))

    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r"):
    """Colorize depth map using UniDepth's colorization.

    Args:
        depth (numpy.ndarray): Depth map
        vmin (float): Minimum value for normalization
        vmax (float): Maximum value for normalization
        cmap (str): Colormap name

    Returns:
        numpy.ndarray: Colored depth map
    """
    return unidepth_colorize(depth, vmin=vmin, vmax=vmax, cmap=cmap)
