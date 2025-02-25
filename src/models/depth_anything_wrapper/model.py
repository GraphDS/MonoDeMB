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

        # Output transform for fixed size
        self.transform_reverse = Compose([
            Resize(width=1242, height=375)
        ])
        self.transform_reverse = Compose([
            Resize(width=640, height=480)
        ])
        # self.transform_reverse = Compose([
        #     Resize(width=6048, height=4032)
        # ])
        # self.transform_reverse = Compose([
        #     Resize(width=1024, height=768)
        # ])
        # self.transform_reverse = Compose([
        #     Resize(width=1024, height=1024),
        # ])

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def forward(self, img):
        """Run inference on image.

        Args:
            img: RGB input in range [0,1] and format CHW

        Returns:
            torch.Tensor: Predicted depth map
        """
        # Convert tensor to numpy and correct format
        if torch.is_tensor(img):
            img = img.cpu().numpy()
            if len(img.shape) == 4:
                img = img.squeeze(0)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        
        # Apply input transforms
        img = self.transform({"image": img / 255})["image"]
        img = torch.from_numpy(img).unsqueeze(0)

        # Move to correct device
        device = next(self.model.parameters()).device
        img = img.to(device)
        # Inference
        with torch.no_grad():
            depth = self.model(img).squeeze(0)

        # Convert to numpy for reverse transform
        depth = depth.cpu().numpy()
        
        # Apply reverse transform
        depth = self.transform_reverse({"image": depth})["image"]
        
        # Convert back to tensor and normalize
        depth = torch.from_numpy(depth).float().to(device)
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        return depth