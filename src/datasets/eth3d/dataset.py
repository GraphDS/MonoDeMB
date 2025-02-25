import os
import numpy as np
from PIL import Image
import torch
from typing import Dict, List
from pathlib import Path
from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

@register_dataset('eth3d')
class ETH3DDataset(BaseDataset):
    """ETH3D depth dataset."""
    
    # Fixed image dimensions for ETH3D dataset
    HEIGHT = 4032
    WIDTH = 6048
    min_depth=1e-5
    max_depth=torch.inf
    
    def __init__(self, config_path: str, split: str = 'test', batch_size: int = 1):
        """Initialize ETH3D dataset.
        
        Args:
            config_path: Path to dataset config file
            split: Dataset split (not used in ETH3D as all data is in one directory)
            batch_size: Batch size for loading data
        """
        # Currently we have only 'courtyard', will expand as we download more
        self.scenes = ['courtyard']
        super().__init__(config_path, split, batch_size)
        
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Find all matching RGB-D pairs in the dataset.
        
        Structure:
        root_dir/
            scene_name/
                rgb/
                    scene_name/
                        images/
                            dslr_images_undistorted/
                                DSC_*.JPG
                depth/
                    scene_name/
                        ground_truth_depth/
                            dslr_images/
                                DSC_*.JPG
        """

        data_pairs = []
        root_dir = Path(self.root_dir)
        
        print(f"Traversing directory: {root_dir}")
        
        for scene in os.listdir(os.path.join(root_dir, 'rgb')):
            for images in os.listdir(os.path.join(root_dir, 'rgb', scene, 'images/dslr_images')):
                rgb_img_path = os.path.join(root_dir, 'rgb', scene, 'images/dslr_images', images)
                gt_depthmap_path = os.path.join(
                    root_dir,
                    'depth',
                    f'{scene}_dslr_depth',
                    scene,
                    'ground_truth_depth/dslr_images',
                    images
                )
                if os.path.exists(rgb_img_path) and os.path.exists(gt_depthmap_path):
                    data_pairs.append({
                        'rgb': rgb_img_path,
                        'depth': gt_depthmap_path
                    })
    
        return data_pairs
    
    def _load_rgb_image(self, path:str) -> torch.Tensor:
        rgb = np.array(Image.open(path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)#[:3,:,:]
        return rgb_torch
    
    def _load_depth(self, path: str) -> np.ndarray:
        """Load ETH3D depth map.
        
        ETH3D depth maps are 4-byte float binary dumps in row-major order.
        Invalid depth values are set to infinity.
        
        Returns:
            Depth map as numpy array [H, W] with shape (4032, 6048)
        """
        depth_path = os.path.join(path)
        with open(depth_path, "rb") as file:
            binary_data = file.read()
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()
        depth_decoded[depth_decoded == torch.inf] = 0.0
        depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH)).squeeze()
        return depth_decoded
    
    def _download(self):
        """Download and extract ETH3D dataset if not already present."""
        # Check if data already exists
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("ETH3D dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_eth3d_genpercept.tar.gz?download=true"
        print("Downloading ETH3D dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir),
        )