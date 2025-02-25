import torch
import numpy as np
from unidepth.models import UniDepthV1, UniDepthV2

from ..base_model import BaseModel

class UniDepthModelWrapper(BaseModel):
    """Wrapper for UniDepth model."""

    def __init__(self, model_name: str = "unidepth-v2-vitl14"):
        """Initialize UniDepth model.

        Args:
            model_name: Model variant name, e.g. 'unidepth-v2-vitl14'
        """
        super().__init__(model_name=model_name)

        # Select model version based on name
        if model_name.startswith("unidepth-v2"):
            self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
        elif model_name.startswith("unidepth-v1"):
            self.model = UniDepthV1.from_pretrained(f"lpiccinelli/{model_name}")
        else:
            raise ValueError(f"Unknown model variant: {model_name}")

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def preprocess(self, rgb):
        """Preprocess input for model.
        
        Args:
            rgb: RGB input in any format (tensor CHW or HWC, numpy HWC)
            
        Returns:
            torch.Tensor: Preprocessed input in model format
        """
        # Convert numpy to tensor if needed
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
            
        # Convert HWC to CHW if needed    
        if len(rgb.shape) == 3 and rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1)

        # Add batch dimension if needed
        if len(rgb.shape) == 3:
            rgb = rgb.unsqueeze(0)
        return rgb

    def forward(self, img):
        """Run inference on image.

        Args:
            img: RGB input in any format (tensor CHW/HWC or numpy HWC)

        Returns:
            torch.Tensor: Predicted depth map
        """
        # Preprocess input
        img = self.preprocess(img)

        # Create dummy intrinsics
        intrinsics = torch.tensor([
            [5.1885790117450188e02, 0, 3.2558244941119034e02],
            [0, 5.1946961112127485e02, 2.5373616633400465e02],
            [0, 0, 1],
        ]).to(img.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model.infer(img, intrinsics)
            depth = predictions["depth"].squeeze()

            # Normalize depth to [0,1] range
            depth = (depth - depth.min()) / (depth.max() - depth.min())

        return depth