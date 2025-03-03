import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class Metric3DV2Wrapper(nn.Module):
    """Wrapper for Metric3D model."""
    
    def __init__(self, model_variant="vit_small"):
        """Initialize Metric3D model.
        
        Args:
            model_variant (str): Type of backbone:
                - "vit_small": ViT Small backbone
                - "vit_base": ViT Base backbone  
                - "vit_large": ViT Large backbone
        """
        super().__init__()
        
        if model_variant not in ("vit_small", "vit_base", "vit_large"):
            raise ValueError(f"Unknown model variant: {model_variant}. Available variants: vit_small, vit_base, vit_large")
            
        # Load model from torch hub
        self.model = torch.hub.load('yvanyin/metric3d', f"metric3d_{model_variant}", pretrain=True)
        
        # Cache preprocessing parameters
        self.input_size = (616, 1064)  # (H, W)
        self.padding = [123.675, 116.28, 103.53]
        self.register_buffer('mean', torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None])
        self.register_buffer('std', torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None])
        
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.model = self.model.to(device)
        return self

    def forward(self, img):
        """Run inference on image.

        Args:
            img (torch.Tensor): Input image tensor in [B,C,H,W] format, range [0,255]

        Returns:
            torch.Tensor: Predicted depth map, batched
        """
        device = img.device
        batch_size = img.shape[0]
        depth_batch = []
        
        for i in range(batch_size):
            # Process each image in the batch individually
            single_img = img[i]  # [C,H,W]
            
            # Get original dimensions
            orig_H, orig_W = single_img.shape[1:]
            rgb_origin = single_img.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            
            # Calculate resize scale to maintain aspect ratio
            scale = min(self.input_size[0] / orig_H, self.input_size[1] / orig_W)
            new_h = int(orig_H * scale)
            new_w = int(orig_W * scale)
            
            # Resize keeping aspect ratio
            rgb_resized = cv2.resize(rgb_origin, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Calculate padding
            pad_h = self.input_size[0] - new_h
            pad_w = self.input_size[1] - new_w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            
            # Apply padding
            rgb_padded = cv2.copyMakeBorder(
                rgb_resized, 
                pad_h_half, pad_h - pad_h_half,
                pad_w_half, pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.padding
            )
            
            # Store padding info for later cropping
            pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
            
            # Convert to tensor and normalize
            rgb = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().to(device)  # [C,H,W]
            rgb = torch.div((rgb - self.mean), self.std)
            rgb = rgb.unsqueeze(0)  # [1,C,H,W]

            # Run inference
            with torch.no_grad():
                pred_depth, confidence, output_dict = self.model.inference({'input': rgb})
                
            # Post-process depth
            pred_depth = pred_depth.squeeze()  # Remove batch dimension
            
            # Remove padding
            pred_depth = pred_depth[
                pad_info[0] : pred_depth.shape[0] - pad_info[1],
                pad_info[2] : pred_depth.shape[1] - pad_info[3]
            ]
            
            # Resize back to original resolution
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,H,W]
                (orig_H, orig_W),
                mode='bilinear',
                align_corners=True
            ).squeeze()  # Remove extra dimensions
            
            depth_batch.append(pred_depth)
        
        # Stack all processed depth maps
        return torch.stack(depth_batch)