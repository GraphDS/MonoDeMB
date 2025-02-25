import numpy as np
from PIL import Image


def process_image(image_path):
    """Read and preprocess image.

    Args:
        image_path (str): Path to input image

    Returns:
        np.ndarray: Preprocessed image in RGB format and range [0,1]
    """
    # Read image
    image = Image.open(image_path)
    image = np.array(image)

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0

    return image
