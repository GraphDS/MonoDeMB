import numpy as np
import os
from PIL import Image
import torch
from pathlib import Path
from typing import Dict, List
from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

@register_dataset('synth2')
class Synth2Dataset(BaseDataset):
    """Synth2 depth dataset."""
    
    def __init__(self, config_path: str, split: str = 'test', batch_size: int = 1):
        """Initialize Synth2 dataset.
        
        Args:
            config_path: Path to dataset config file
            split: Dataset split (not used as all data is in one directory)
            batch_size: Batch size for loading data
        """
        super().__init__(config_path, split, batch_size)
        self.min_depth = 0
        self.max_depth = 255.0
            
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Find all matching RGB-D pairs in the dataset.
        
        Structure:
        root_dir/
            scene_id (UUID)/
                rgb/
                    1.jpeg
                    2.jpeg
                    ...
                gt/
                    1.png
                    2.png
                    ...
                inferno/  # Alternative rendering (ignored)
                    ...
        """
        data_pairs = []
        
        # Get all scene directories (UUID directories in root)
        try:
            root_path = self.root_dir
            # Get list of directory names as strings
            scene_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) 
                         if os.path.isdir(os.path.join(root_path, d))]
            print(f"Found {len(scene_dirs)} scenes")            
            
            for scene_dir in scene_dirs:
                scene_id = os.path.basename(scene_dir)
                rgb_dir = os.path.join(scene_dir, 'rgb')
                depth_dir = os.path.join(scene_dir, 'gt')
                
                # Check if both directories exist
                if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
                    print(f"Skipping scene {scene_id}: missing rgb or gt directory")
                    continue
                
                # Get all RGB images in this scene
                rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpeg')])
                
                for rgb_filename in rgb_files:
                    # Extract the numeric part of the filename
                    filename_stem = os.path.splitext(rgb_filename)[0]  # This gets "1" from "1.jpeg"
                    
                    # Construct corresponding depth path
                    rgb_path = os.path.join(rgb_dir, rgb_filename)
                    depth_path = os.path.join(depth_dir, f"{filename_stem}.png")
                    
                    if os.path.exists(depth_path):
                        data_pairs.append({
                            'rgb': rgb_path,
                            'depth': depth_path
                        })
                    else:
                        print(f"Missing depth for RGB: {rgb_filename} (expected {depth_path})")
            
            print(f"Found {len(data_pairs)} RGB-D pairs")
            if data_pairs:
                print("Example pair:")
                print(f"RGB: {data_pairs[0]['rgb']}")
                print(f"Depth: {data_pairs[0]['depth']}")
            
            return data_pairs
            
        except Exception as e:
            print(f"Error traversing directory: {str(e)}")
            return []

    def _load_depth(self, path: str) -> np.ndarray:
        """Load Synth2 depth map.
        
        Args:
            path: Path to depth map PNG file
            
        Returns:
            Depth map as numpy array [H, W]
        """
        try:
            # Load depth image (grayscale)
            depth = np.asarray(Image.open(path)).squeeze()  # [H, W]
            depth = torch.from_numpy(depth).float().unsqueeze(0) / self.max_depth
            return depth
        except Exception as e:
            print(f"Error loading depth from {path}: {str(e)}")
            # Return a placeholder depth map
            return torch.zeros((1, 512, 512), dtype=torch.float32)
            
    def _load_rgb_image(self, path: str) -> torch.Tensor:
        """Load RGB image.
        
        Args:
            path: Path to RGB image file
            
        Returns:
            RGB image as tensor [C, H, W]
        """
        try:
            rgb = np.array(Image.open(path))
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            return rgb_torch
        except Exception as e:
            print(f"Error loading RGB from {path}: {str(e)}")
            # Return a placeholder RGB image
            return torch.zeros((3, 512, 512), dtype=torch.float32)
    
    def _get_valid_mask(self, depth: torch.Tensor) -> torch.Tensor:
        """Get valid mask from depth tensor.
        
        Args:
            depth: Depth tensor
            
        Returns:
            Valid mask tensor
        """
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask
    
    def _download(self):
        """Download and extract Synth2 dataset if not already present."""
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("Synth2 dataset already exists.")
            return

        url = "https://ue-benchmark-dp.obs.ru-moscow-1.hc.sbercloud.ru/synth2.tar.gz"
        print("Downloading Synth2 dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir)
        )