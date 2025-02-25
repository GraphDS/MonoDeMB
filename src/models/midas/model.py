import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.hub.set_dir("weights/midas")

class MidasModel(nn.Module):
    """MiDaS depth estimation model."""

    def __init__(self, model_type="DPT_Large"):
        """Initialize MiDaS model.
        
        Args:
            model_type: Model variant (DPT_Large, DPT_Hybrid, or MiDaS_small)
        """
        super().__init__()

        # Load model and transforms from torch hub
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = self.transforms.dpt_transform
        else:
            self.transform = self.transforms.small_transform
            
        # Store model type
        self.model_type = model_type
        
        # Convert model to float32
        self.model = self.model.float()
        self.model.eval()

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def forward(self, img):
        """Run inference on image.
        
        Args:
            img: RGB image tensor in [0,1] range, format CHW
            
        Returns:
            torch.Tensor: Predicted depth map
        """
        device = next(self.model.parameters()).device
        
        # Convert tensor input to numpy for transforms
        if torch.is_tensor(img):
            # Ensure input is on CPU and correct format
            # img = img.cpu().float().numpy()
            # img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            
            # # Scale to 0-255 range
            # img = (img * 255.0).astype(np.uint8)
            
            # Record original size
            img = img.squeeze().permute(1, 2, 0).cpu().numpy()
            original_size = img.shape[:2]
            
        else:
            original_size = img.shape[:2]
            img = (img * 255.0).astype(np.uint8)

        # Apply MiDaS transforms
        input_batch = self.transform(img)
        
        # Move to device
        input_batch = input_batch.to(device)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Interpolate back to original size
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=original_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            # Normalize prediction
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

        return prediction
