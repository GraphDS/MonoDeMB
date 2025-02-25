from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import yaml
import os
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import torchvision.transforms as transforms

class BaseDataset(Dataset, ABC):
    """Base class for all datasets."""
    
    def __init__(self, config_path: str, split: str = 'test', batch_size: int = 1):
        """Initialize dataset.
        
        Args:
            config_path: Path to dataset config file
            split: Dataset split (train/val/test)
            batch_size: Batch size for loading data
        """
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set paths from config
        self.root_dir = self.config['paths']['root_dir']
        
        self._check_downloaded()
        
        self.split_dir = os.path.join(self.root_dir, self.config['paths'][split])
        
        # Set transforms
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['preprocessing']['rgb_mean'],
                std=self.config['preprocessing']['rgb_std']
            )
        ])
        
        # Load dataset structure
        self.data_pairs = self._traverse_directory()
        
    @abstractmethod
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Traverse directory and get pairs of RGB and depth paths.
        
        Returns:
            List of dicts containing paths for image and depth pairs
        """
        pass
        
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Dict containing:
                - rgb: RGB image tensor [3, H, W]
                - depth: Depth map tensor [1, H, W]  
                - mask: Valid depth mask [1, H, W]
        """
        sample = self.data_pairs[idx]
        
        # Load RGB image
        # rgb = Image.open(sample['rgb']).convert('RGB')
        rgb = self._load_rgb_image(sample['rgb'])
        # Load depth map
        depth = self._load_depth(sample['depth'])
        if not torch.is_tensor(depth):
            depth = torch.from_numpy(depth)
        
        # Create valid mask
        mask = self._get_valid_mask(depth)
        
        # Apply transformations
        # rgb = self.rgb_transform(rgb)
        # mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'rgb_path': sample['rgb'],
            'depth_path': sample['depth']
        }
    
    @abstractmethod
    def _load_depth(self, path: str) -> np.ndarray:
        """Load depth map from file.
        
        Args:
            path: Path to depth map file
            
        Returns:
            Depth map as numpy array [H, W]
        """
        depth = Image.open(path)  # [H, W, rgb]
        depth = np.asarray(depth)
        return depth
    
    def _get_valid_mask(self, depth: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask
    
    @abstractmethod
    def _download(self):
        """Download dataset files. Must be implemented by child classes."""
        pass
    
    def _check_downloaded(self):
        """Ensures dataset is downloaded. Called before directory traversal."""
        if not os.path.exists(self.root_dir) or len(os.listdir(self.root_dir)) == 0:
            print(f"Dataset not found in {self.root_dir}, downloading...")
            self._download()
            
        if not os.path.exists(self.root_dir) or len(os.listdir(self.root_dir)) == 0:
            raise RuntimeError(f"Failed to download dataset to {self.root_dir}")
    
    def get_batch(self, start_idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Get a batch of samples.
        
        Args:
            start_idx: Starting index
            
        Returns:
            Batch of samples and next start index
        """
        batch_size = min(self.batch_size, len(self) - start_idx)
        batch = {
            'rgb': [],
            'depth': [],
            'mask': [],
            'rgb_path': [],
            'depth_path': []
        }
        
        for i in range(batch_size):
            sample = self[start_idx + i]
            for key in batch:
                batch[key].append(sample[key])
                
        # Stack tensors
        for key in ['rgb', 'depth', 'mask']:
            batch[key] = torch.stack(batch[key])
            
        return batch, start_idx + batch_size

# Dictionary to register dataset classes
DATASET_REGISTRY = {}

def register_dataset(name: str):
    """Decorator to register a new dataset class."""
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper