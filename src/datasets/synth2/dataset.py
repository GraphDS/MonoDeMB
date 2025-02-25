import numpy as np
from PIL import Image
import torch
from pathlib import Path
from typing import Dict, List
from ..base_dataset import BaseDataset, register_dataset

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
            synth2_rgb/
                [scene_id]/
                    images/
                        Scene_1_80_MyPositive1187_796842.jpeg
            synth2_depth/
                [scene_id]/
                    depth_maps/
                        Scene_1_80_MyPositive1187_796842.png
        """
        data_pairs = []
        
        # Get all scene directories
        try:
            rgb_root = Path(self.root_dir) / 'synth2_rgb'
            scene_dirs = sorted([d for d in rgb_root.iterdir() if d.is_dir()])
            print(f"Found {len(scene_dirs)} scenes")
            
            for scene_dir in scene_dirs:
                scene_id = scene_dir.name
                rgb_image_dir = scene_dir / 'images'
                depth_root = Path(self.root_dir) / 'synth2_depth'
                depth_image_dir = depth_root / scene_id / 'depth_maps'
                
                if not rgb_image_dir.exists() or not depth_image_dir.exists():
                    print(f"Skipping scene {scene_id}: missing directories")
                    continue
                
                # Get all RGB images in this scene
                rgb_files = sorted(list(rgb_image_dir.glob('*.jpeg')))
                
                for rgb_path in rgb_files:
                    # Construct depth path with .png extension
                    depth_path = depth_image_dir / (rgb_path.stem + '.png')
                    
                    if depth_path.exists():
                        data_pairs.append({
                            'rgb': str(rgb_path),
                            'depth': str(depth_path)
                        })
                    else:
                        print(f"Missing depth for RGB: {rgb_path.name}")
            
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
        # try:
            # Load depth image (grayscale)
        depth = np.asarray(Image.open(path)).squeeze()  # [H, W, rgb]
        depth = torch.from_numpy(depth).float().unsqueeze(0) / self.max_depth

        return depth
            
        
    def _load_rgb_image(self, path:str) -> torch.Tensor:
        rgb = np.array(Image.open(path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)#[:3,:,:]
        return rgb_torch
    
    def _download(self):
        pass # TODO - add link