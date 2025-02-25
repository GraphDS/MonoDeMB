import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List
from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

@register_dataset('diode')
class DIODEDataset(BaseDataset):
    """DIODE depth dataset."""
    
    def __init__(self, config_path: str, split: str = 'val', batch_size: int = 1):
        """Initialize DIODE dataset."""
        super().__init__(config_path, split, batch_size)
        self.min_depth=0.6
        self.max_depth=350.0

    
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Find all matching RGB-D pairs in the dataset."""
        data_pairs = []

        for env_type in ["indoors", "outdoors"]:
            env_path = os.path.join(self.root_dir, env_type)
            if not os.path.exists(env_path):
                continue

            for scene_dir in sorted(os.listdir(env_path)):
                scene_path = os.path.join(env_path, scene_dir)
                if not os.path.isdir(scene_path):
                    continue

                for scan_dir in sorted(os.listdir(scene_path)):
                    scan_path = os.path.join(scene_path, scan_dir)
                    if not os.path.isdir(scan_path):
                        continue

                    for filename in sorted(os.listdir(scan_path)):
                        if not filename.endswith(".png") or "_depth_mask" in filename:
                            continue

                        base_name = filename[:-4]
                        rgb_path = os.path.join(scan_path, f"{base_name}.png")
                        depth_path = os.path.join(scan_path, f"{base_name}_depth.npy")
                        mask_path = os.path.join(
                            scan_path, f"{base_name}_depth_mask.npy"
                        )

                        if (
                            os.path.exists(rgb_path)
                            and os.path.exists(depth_path)
                            and os.path.exists(mask_path)
                        ):
                            data_pairs.append(
                                {
                                    "rgb": rgb_path,
                                    "depth": depth_path,
                                    "mask": mask_path,
                                }
                            )

        return sorted(data_pairs, key=lambda x: x["rgb"])

    
    def _load_depth(self, path: str) -> np.ndarray:
        """Load DIODE depth map from .npy file."""
        try:
            # Load depth map
            depth = np.load(path).squeeze()[np.newaxis, :, :].squeeze()
            depth = torch.from_numpy(depth).float().unsqueeze(0) 
                
            return depth
            
        except Exception as e:
            print(f"Error loading depth from {path}: {str(e)}")
            return np.zeros((768, 1024), dtype=np.float32)
        
    def _get_valid_mask(self, path: str):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        mask_path = path.replace('_depth.npy', '_depth_mask.npy')
        valid_mask = np.load(mask_path).squeeze()[np.newaxis, :, :].squeeze()
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0).bool()
        
        return valid_mask
            
    def _resize_depth_and_mask(self, depth: np.ndarray, target_size: tuple) -> tuple:
        """Resize depth map and create corresponding mask using proper interpolation."""
        # Convert to torch tensor and add batch and channel dimensions
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        
        # Resize depth map
        resized_depth = F.interpolate(
            depth_tensor,
            size=target_size,
            mode='bicubic',
            align_corners=True
        ).squeeze()
        
        # Create mask from resized depth
        resized_mask = (resized_depth > 0).float()
        
        return resized_depth.numpy(), resized_mask.numpy()
    
    def _load_rgb_image(self, path:str) -> torch.Tensor:
        rgb = np.array(Image.open(path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)#[:3,:,:]
        return rgb_torch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.data_pairs[idx]
        
        # Load RGB image
        # rgb = Image.open(sample['rgb']).convert('RGB')
        rgb = self._load_rgb_image(sample['rgb'])
        # Load depth map
        depth = self._load_depth(sample['depth'])
        if not torch.is_tensor(depth):
            depth = torch.from_numpy(depth)
        
        # Create valid mask
        mask = self._get_valid_mask(sample['depth'])
                
        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'rgb_path': sample['rgb'],
            'depth_path': sample['depth']
        }
        
    def _download(self):
        """Download and extract DIODE dataset if not already present."""
        # Check if data already exists
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("DIODE dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_diode_genpercept.tar.gz?download=true"
        print("Downloading DIODE dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir),
        )