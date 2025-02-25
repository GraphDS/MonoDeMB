import cv2
import numpy as np


def process_image(image_path):
    """Read and preprocess image.

    Args:
        image_path (str): Path to input image

    Returns:
        np.ndarray: Preprocessed image in RGB format and range [0,1]
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img
