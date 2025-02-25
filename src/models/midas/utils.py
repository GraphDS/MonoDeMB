import os
import cv2
import numpy as np

def process_image(image_path):
    """Read and preprocess image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        np.ndarray: Image in RGB format
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0

    return img

def colorize_depth(depth):
    """Convert depth map to color visualization.
    
    Args:
        depth: Depth map
        
    Returns:
        np.ndarray: Colored depth map
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)

    # Convert to uint8 and apply colormap
    colored = (normalized_depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(colored, cv2.COLORMAP_INFERNO)

    return colored

def save_outputs(depth, input_path, output_dir, grayscale=False):
    """Save depth prediction outputs.
    
    Args:
        depth: Predicted depth map
        input_path: Path to input image
        output_dir: Output directory
        grayscale: Use grayscale colormap
    """
    os.makedirs(output_dir, exist_ok=True)

    output_base = os.path.join(
        output_dir, os.path.splitext(os.path.basename(input_path))[0]
    )

    # Save raw depth
    np.save(f"{output_base}_depth.npy", depth)

    if grayscale:
        # Grayscale depth visualization
        depth_min = depth.min()
        depth_max = depth.max()
        depth_norm = (depth - depth_min) * 255.0 / (depth_max - depth_min)
        colored_depth = np.repeat(np.expand_dims(depth_norm, 2), 3, axis=2) / 3
        colored_depth = colored_depth.astype(np.uint8)
    else:
        # Colored depth visualization
        colored_depth = colorize_depth(depth)

    # Read input image
    input_img = cv2.imread(input_path)
    
    # Ensure colored_depth matches input size
    if colored_depth.shape[:2] != input_img.shape[:2]:
        colored_depth = cv2.resize(
            colored_depth,
            (input_img.shape[1], input_img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    cv2.imwrite(f"{output_base}_depth_colored.png", colored_depth)

    # Save side-by-side visualization
    side_by_side = np.concatenate([input_img, colored_depth], axis=1)
    cv2.imwrite(f"{output_base}_side_by_side.png", side_by_side)
