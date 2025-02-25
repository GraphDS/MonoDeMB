import os
import numpy as np
from PIL import Image
from typing import Dict, List
import cv2
from ..base_dataset import BaseDataset, register_dataset

@register_dataset('mock')
class MockDataset(BaseDataset):
    """Mock dataset for testing."""
    
    def __init__(self, config_path: str, split: str = 'test', batch_size: int = 1):
        """Initialize and potentially generate mock dataset."""
        super().__init__(config_path, split, batch_size)
        
        # Generate mock data if it doesn't exist
        if not os.path.exists(self.split_dir):
            self._generate_mock_data()
            
        self.min_depth = 0
        self.max_depth = 1
            
    def _generate_mock_data(self):
        """Generate synthetic images and depth maps."""
        # Create directories
        os.makedirs(os.path.join(self.split_dir, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(self.split_dir, 'depth_images'), exist_ok=True)
        
        # Get target size from config
        height, width = self.config['preprocessing']['target_size']
        
        # Generate different test patterns
        patterns = {
            'white': np.ones((height, width, 3), dtype=np.uint8) * 255,
            'black': np.zeros((height, width, 3), dtype=np.uint8),
            'circle': self._generate_circle_image(height, width),
            'chessboard': self._generate_chessboard(height, width)
        }
        
        # Generate corresponding depth maps
        for name, img in patterns.items():
            # Save RGB image
            rgb_path = os.path.join(self.split_dir, 'imgs', f'{name}.jpg')
            Image.fromarray(img).save(rgb_path)
            
            # Generate and save depth map
            depth_map = self._generate_depth_map(name, height, width)
            depth_path = os.path.join(self.split_dir, 'depth_images', f'{name}_depth.png')
            depth_map_uint16 = (depth_map * 65535).astype(np.uint16)  # Convert to 16-bit
            Image.fromarray(depth_map_uint16).save(depth_path)
            
    def _generate_circle_image(self, height: int, width: int) -> np.ndarray:
        """Generate white image with black circle in center."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        center = (width // 2, height // 2)
        radius = min(height, width) // 4
        cv2.circle(img, center, radius, (0, 0, 0), -1)
        return img
        
    def _generate_chessboard(self, height: int, width: int, square_size: int = 50) -> np.ndarray:
        """Generate chessboard pattern."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i + j) // square_size) % 2 == 0:
                    img[i:i+square_size, j:j+square_size] = 255
        return img
        
    def _generate_depth_map(self, pattern_name: str, height: int, width: int) -> np.ndarray:
        """Generate synthetic depth map based on pattern type."""
        if pattern_name == 'circle':
            # Radial gradient from center
            y, x = np.ogrid[-height//2:height//2, -width//2:width//2]
            depth = np.sqrt(x*x + y*y)
            depth = 1 - (depth / depth.max())  # Invert so circle is closer
            
        elif pattern_name == 'chessboard':
            # Sine wave pattern
            x = np.linspace(0, 4*np.pi, width)
            y = np.linspace(0, 4*np.pi, height)
            X, Y = np.meshgrid(x, y)
            depth = np.sin(X) * np.cos(Y)
            depth = (depth + 1) / 2  # Normalize to [0,1]
            
        else:
            # Simple gradient for white/black images
            depth = np.linspace(0, 1, width)
            depth = np.tile(depth, (height, 1))
            
        return depth.astype(np.float32)
        
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Get paths to all image pairs."""
        data_pairs = []
        
        imgs_dir = os.path.join(self.split_dir, 'imgs')
        depth_dir = os.path.join(self.split_dir, 'depth_images')
        
        if not os.path.exists(imgs_dir) or not os.path.exists(depth_dir):
            self._generate_mock_data()
            
        for img_name in os.listdir(imgs_dir):
            if img_name.endswith('.jpg'):
                name = img_name.split('.')[0]
                rgb_path = os.path.join(imgs_dir, img_name)
                depth_path = os.path.join(depth_dir, f'{name}_depth.png')
                
                if os.path.exists(depth_path):
                    data_pairs.append({
                        'rgb': rgb_path,
                        'depth': depth_path
                    })
                    
        return data_pairs
        
    def _load_depth(self, path: str) -> np.ndarray:
        """Load depth map."""
        depth = np.array(Image.open(path)).astype(np.float32) / 65535.0  # Convert from 16-bit
        return depth
    
    def _download(self):
        pass
    
    def _load_rgb_image(self, path:str):
        pass