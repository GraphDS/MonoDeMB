import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms.functional as F
import os

from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import load_ckpt
from ..common_utils import download_with_progress

MODEL_URLS = {
    'resnet50': 'https://huggingface.co/ffranchina/LeReS/resolve/main/res50.pth?download=true',
    'resnext101': 'https://huggingface.co/ffranchina/LeReS/resolve/main/res101.pth?download=true'
}

class LeReSWrapper(nn.Module):
    """Wrapper for LeReS model."""

    def __init__(self, model_type="resnext101"):
        """Initialize LeReS model.

        Args:
            model_type (str): Type of backbone to use:
                - "resnext101": ResNext101 backbone
                - "resnet50": ResNet50 backbone
        """
        super().__init__()

        self.model = RelDepthModel(backbone=model_type)

        if model_type not in MODEL_URLS:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_URLS.keys())}")

        # Create weights directory if not exists
        weights_dir = os.path.join("weights", "leres")
        os.makedirs(weights_dir, exist_ok=True)

        # Set checkpoint path and download if needed
        ckpt_filename = "res50.pth" if model_type == "resnet50" else "res101.pth"
        ckpt_path = os.path.join(weights_dir, ckpt_filename)
        
        if not os.path.exists(ckpt_path):
            print(f"Downloading {model_type} weights...")
            download_with_progress(MODEL_URLS[model_type], ckpt_path)

        # Initialize model and load weights
        self.model = RelDepthModel(backbone=model_type)
        load_ckpt(ckpt_path, self.model, None, None)
                
        # Store normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def forward(self, img):
        """Run inference on image.

        Args:
            img: RGB image tensor in range [0, 1] and format BCHW where B is batch size

        Returns:
            torch.Tensor: Predicted depth map, batched
        """
        device = img.device
        batch_size = img.shape[0]
        depth_batch = []
        
        for i in range(batch_size):
            # Process each image in the batch individually
            single_img = img[i]  # Shape: [C, H, W]
            
            # Convert tensor to numpy for preprocessing
            img_np = single_img.cpu().numpy()  # Shape: [C, H, W]
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            
            # Get original size
            original_size = img_np.shape[:2]
            
            # Resize to model's expected size
            A_resize = cv2.resize(img_np.copy(), (448, 448))
            
            # Convert back to tensor and normalize
            img_torch = torch.from_numpy(A_resize).float().permute(2, 0, 1) / 255.0  # HWC -> CHW
            img_torch = img_torch.to(device)
            
            # Normalize
            img_torch = (img_torch - self.mean) / self.std
            img_torch = img_torch.unsqueeze(0)  # Add batch dimension for the model
            
            # Run inference
            with torch.no_grad():
                depth = self.model.inference(img_torch)
                
                # Convert to numpy, resize using cv2 and convert back to torch
                depth_np = depth.cpu().numpy().squeeze()
                depth_resized = cv2.resize(depth_np, (original_size[1], original_size[0]))
                depth = torch.from_numpy(depth_resized).to(device)
                
                # Normalize depth for visualization
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                
            depth_batch.append(depth)
        
        # Stack all processed images back into a batch
        return torch.stack(depth_batch)