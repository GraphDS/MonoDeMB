import torch
import numpy as np
from torchvision.transforms import Compose
import cv2

from ..base_model import BaseModel
from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class DepthAnythingWrapper(BaseModel):
    """Wrapper for DepthAnything model."""

    def __init__(self, encoder_type="vitl"):
        """Initialize DepthAnything model.
        
        Args:
            encoder_type: Type of vision transformer encoder ('vitl', 'vitb', 'vits')
        """
        super().__init__(model_name=f"depth_anything_{encoder_type}")
        
        # Load and init model
        self.model = DepthAnything.from_pretrained(
            f"LiheYoung/depth_anything_{encoder_type}14",
            cache_dir='weights/depth_anything'
        )
        self.model.eval()

        # Input transform
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # Original dimensions will be set during forward pass
        self.original_width = None
        self.original_height = None

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def forward(self, img):
        """Run inference on image.

        Args:
            img: RGB input in range [0,1] and format BCHW

        Returns:
            torch.Tensor: Predicted depth map, batched
        """
        device = next(self.model.parameters()).device
        batch_size = img.shape[0]
        depth_batch = []
        
        for i in range(batch_size):
            # Process each image in the batch individually
            single_img = img[i]  # Shape: [C, H, W]
            
            # Convert tensor to numpy for preprocessing
            img_np = single_img.cpu().numpy()  # Shape: [C, H, W]
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            
            # Store original dimensions before transform
            original_height, original_width = img_np.shape[:2]
            
            # Apply input transforms
            transformed_img = self.transform({"image": img_np / 255})["image"]
            transformed_img = torch.from_numpy(transformed_img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                depth = self.model(transformed_img).squeeze(0)
            
            # Convert to numpy for reverse transform
            depth_np = depth.cpu().numpy()
            
            # Create adaptive reverse transform based on original dimensions
            transform_reverse = Compose([
                Resize(width=original_width, height=original_height)
            ])
            
            # Apply reverse transform to match original image size
            resized_depth = transform_reverse({"image": depth_np})["image"]
            
            # Convert back to tensor and normalize
            depth_tensor = torch.from_numpy(resized_depth).float().to(device)
            normalized_depth = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
            
            depth_batch.append(normalized_depth)
        
        # Stack all processed images back into a batch
        return torch.stack(depth_batch)