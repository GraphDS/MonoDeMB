import os
import numpy as np
import torch
from PIL import Image
from typing import Dict, List

from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

@register_dataset('nyu')
class NYUDataset(BaseDataset):
    """NYU Depth V2 dataset."""
    min_depth = 1e-3
    max_depth = 10.0
    
    
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Traverse NYU dataset directory structure."""
        data_pairs = []
        split_dir = os.path.join(self.root_dir, self.split)

        if not os.path.exists(split_dir):
            print(f"Split directory does not exist: {split_dir}")
            return data_pairs

        for scene in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            rgb_files = [
                f for f in os.listdir(scene_dir)
                if f.startswith("rgb_") and f.endswith(".png")
            ]

            for rgb_file in rgb_files:
                img_id = rgb_file.replace("rgb_", "").replace(".png", "")
                depth_file = f"depth_{img_id}.png"

                rgb_path = os.path.join(scene_dir, rgb_file)
                depth_path = os.path.join(scene_dir, depth_file)

                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    data_pairs.append({"rgb": rgb_path, "depth": depth_path})

        return sorted(data_pairs, key=lambda x: x["rgb"])

        
    def _load_depth(self, path: str) -> np.ndarray:
        """Load NYU depth map.
        
        Args:
            path: Path to depth map file
            
        Returns:
            Depth map as numpy array [H, W]
        """
        # Load depth (NYU depths are in millimeters)
        depth = super()._load_depth(path)
        depth = depth.astype(np.float32) / 1000.0
        # Normalize
        
        return depth
    
    def _load_rgb_image(self, path:str) -> torch.Tensor:
        rgb = np.array(Image.open(path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)#[:3,:,:]
        return rgb_torch
    
    def _get_valid_mask(self, depth: torch.Tensor) -> torch.Tensor:
        valid_mask = super()._get_valid_mask(depth)
        eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
        eval_mask[45:471, 41:601] = 1
        eval_mask.reshape(valid_mask.shape)
        valid_mask = torch.logical_and(valid_mask, eval_mask)
        
        return valid_mask
    
    def _download(self):
        """Download and extract NYU dataset if not already present."""
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("NYU dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_nyu_genpercept.tar.gz?download=true"
        print("Downloading NYU dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir)
        )
        