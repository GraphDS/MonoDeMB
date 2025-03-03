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
            img: RGB image tensor in [0,1] range, format BCHW where B is batch size
            
        Returns:
            torch.Tensor: Predicted depth maps, batched
        """
        device = next(self.model.parameters()).device
        batch_size = img.shape[0]
        depth_batch = []
        
        for i in range(batch_size):
            # Process each image in the batch individually
            single_img = img[i]  # Shape: [C, H, W]
            
            # Convert tensor to numpy for MiDaS transforms
            img_np = single_img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            original_size = img_np.shape[:2]
            
            # Ensure correct range for MiDaS transform
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
                
            # Apply MiDaS transforms
            input_tensor = self.transform(img_np)
            
            # Move to device
            input_tensor = input_tensor.to(device)

            # Inference
            with torch.no_grad():
                prediction = self.model(input_tensor)
                
                # Interpolate back to original size
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=original_size,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                # Normalize prediction
                prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
                
            depth_batch.append(prediction)
        
        # Stack all processed images back into a batch
        return torch.stack(depth_batch)