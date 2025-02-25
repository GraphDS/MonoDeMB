import numpy as np
from PIL import Image
import cv2
import torch

def process_image(image_path):
    """Read and preprocess image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        np.ndarray: Preprocessed image in RGB format
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

def colorize(depth, cmap="magma"):
    """Colorize depth map.
    
    Args:
        depth: Depth map tensor or array
        cmap: Colormap name
        
    Returns:
        np.ndarray: Colored depth map
    """
    if torch.is_tensor(depth):
        depth = depth.cpu().numpy()
        
    depth = (depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth, getattr(cv2, f"COLORMAP_{cmap.upper()}"))
    
    return colored
