import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataset_tests')

def denormalize_image(img_tensor):
    """Denormalize image tensor to numpy array."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
        if len(img.shape) == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
    else:
        img = img_tensor
        
    # Handle different value ranges
    if img.max() > 1.0 or img.min() < 0.0:
        # Simple range normalization
        img_range = img.max() - img.min()
        if img_range > 0:
            img = (img - img.min()) / img_range
        else:
            img = np.zeros_like(img)
    
    # Ensure RGB image has 3 channels
    if len(img.shape) == 2:  # Add color channel if grayscale
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 1:  # Expand if single channel
        img = np.concatenate([img, img, img], axis=2)
    
    return img

def visualize_sample(sample, save_dir, name, dataset_name, colormap='plasma', depth_range=None):
    """Visualize a single sample with rgb, depth and mask."""
    plt.figure(figsize=(15, 5))
    
    try:
        # RGB
        plt.subplot(131)
        if 'rgb' in sample and sample['rgb'] is not None:
            img = denormalize_image(sample['rgb'])
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, 'RGB data not available', 
                    horizontalalignment='center', verticalalignment='center')
        plt.title('RGB Image')
        plt.axis('off')
        
        # Depth
        plt.subplot(132)
        if 'depth' in sample and sample['depth'] is not None:
            depth = sample['depth']
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            
            # Handle different tensor dimensions
            if len(depth.shape) == 3:
                depth = depth.squeeze()  # Remove channel dimension if present
            
            # Set display range
            if depth_range is not None:
                plt.imshow(depth, cmap=colormap, vmin=depth_range[0], vmax=depth_range[1])
            else:
                # Compute sensible range if not provided
                valid_depth = depth[depth > 0] if np.any(depth > 0) else depth
                vmin = np.min(valid_depth) if len(valid_depth) > 0 else 0
                vmax = np.max(valid_depth) if len(valid_depth) > 0 else 1
                plt.imshow(depth, cmap=colormap, vmin=vmin, vmax=vmax)
            
            plt.colorbar(label='Depth')
        else:
            plt.text(0.5, 0.5, 'Depth data not available', 
                    horizontalalignment='center', verticalalignment='center')
        plt.title('Depth Map')
        plt.axis('off')
        
        # Mask
        plt.subplot(133)
        if 'mask' in sample and sample['mask'] is not None:
            mask = sample['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            
            # Handle different tensor dimensions
            if len(mask.shape) == 3:
                mask = mask.squeeze()  # Remove channel dimension if present
            
            plt.imshow(mask, cmap='gray')
            valid_ratio = np.mean(mask > 0) if mask.size > 0 else 0
            plt.title(f'Valid Mask ({valid_ratio:.1%} valid)')
        else:
            plt.text(0.5, 0.5, 'Mask data not available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Valid Mask')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save path
        save_path = os.path.join(save_dir, f'{dataset_name}_sample_{name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        logger.info(f"Saved visualization to {save_path}")
        
    except Exception as e:
        plt.close()
        logger.error(f"Error during visualization: {str(e)}")
        # Create a simple error image
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f'Visualization error: {str(e)}', 
                horizontalalignment='center', verticalalignment='center')
        save_path = os.path.join(save_dir, f'{dataset_name}_sample_{name}_error.png')
        plt.savefig(save_path)
        plt.close()

def calculate_dataset_statistics(dataset, max_samples=10):
    """Calculate dataset statistics on a limited number of samples."""
    all_depths = []
    valid_pixel_ratios = []
    
    # Limit number of samples for testing
    num_samples = min(len(dataset), max_samples)
    
    for i in range(num_samples):
        try:
            sample = dataset[i]
            depth = sample['depth']
            
            if isinstance(depth, torch.Tensor):
                depth_numpy = depth.detach().cpu().numpy()
            else:
                depth_numpy = depth
            
            # Handle different dimensions
            if len(depth_numpy.shape) == 3:
                depth_numpy = depth_numpy.squeeze()
                
            # Find valid depth values
            valid_depth = depth_numpy[depth_numpy > 0]
            
            if len(valid_depth) > 0:
                all_depths.extend(valid_depth.flatten())
                valid_ratio = np.mean(depth_numpy > 0)
                valid_pixel_ratios.append(valid_ratio)
                
        except Exception as e:
            logger.warning(f"Error calculating statistics for sample {i}: {str(e)}")
    
    # Handle empty depths case
    if not all_depths:
        return {
            "depth_range": [0, 0],
            "mean_depth": 0,
            "valid_pixels": 0
        }
    
    all_depths = np.array(all_depths)
    
    stats = {
        "depth_range": [float(np.min(all_depths)), float(np.max(all_depths))],
        "mean_depth": float(np.mean(all_depths)),
        "valid_pixels": float(np.mean(valid_pixel_ratios)) if valid_pixel_ratios else 0
    }
    
    return stats

def print_sample_info(sample):
    """Print information about sample tensors."""
    info = []
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            info.append(f"{key}:")
            info.append(f"  Shape: {value.shape}")
            try:
                info.append(f"  Range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                info.append(f"  Mean: {value.mean().item():.3f}")
                info.append(f"  Std: {value.std().item():.3f}")
                if key == 'depth':
                    valid_depth = value[value > 0]
                    if len(valid_depth) > 0:
                        info.append(f"  Valid depth range: [{valid_depth.min().item():.3f}, {valid_depth.max().item():.3f}]")
                        info.append(f"  Valid depth mean: {valid_depth.mean().item():.3f}")
            except Exception as e:
                info.append(f"  Error computing stats: {str(e)}")
        elif isinstance(value, str):
            info.append(f"{key}: {Path(value).name}")
        info.append("")
    logger.info("\n".join(info))