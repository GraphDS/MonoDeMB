import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from .marigold import MarigoldPipeline

class MarigoldWrapper(nn.Module):
    """Wrapper for Marigold model."""
    
    def __init__(self, model_type="LCM"):
        """Initialize Marigold model.
        
        Args:
            model_type (str): Type of model to use:
                - "Original": Higher quality (prs-eth/marigold-v1-0)
                - "LCM": Faster (prs-eth/marigold-lcm-v1-0)
        """
        super().__init__()
        
        ckpt_dic = {
            "Original": "prs-eth/marigold-v1-0",
            "LCM": "prs-eth/marigold-lcm-v1-0",
        }
        
        if model_type not in ckpt_dic:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.pipeline = MarigoldPipeline.from_pretrained(ckpt_dic[model_type])
        
        # Default settings optimized for speed/quality
        self.settings = {
            "Original": {
                "denoising_steps": 20,
                "ensemble_size": 10,
            },
            "LCM": {
                "denoising_steps": 4,
                "ensemble_size": 5,
            },
        }[model_type]
    
    def to(self, device):
        """Move model to device."""
        self.pipeline = self.pipeline.to(device)
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
            
            # Convert to PIL Image 
            img_np = single_img.cpu().numpy()  # Shape: [C, H, W]
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            
            # Convert to PIL image - ensure correct range
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Run prediction
            output = self.pipeline(
                pil_img,
                denoising_steps=self.settings["denoising_steps"],
                ensemble_size=self.settings["ensemble_size"],
                processing_res=768,
                match_input_res=True,
                batch_size=0,  # Process one image from the PIL image
                color_map="Spectral",
                show_progress_bar=False,
            )
            
            # Get depth map and normalize
            depth = output.depth_np
            depth = (depth - np.mean(depth)) / np.std(depth)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            # Convert to tensor and move to device
            depth_tensor = torch.from_numpy(depth).float().to(device)
            depth_batch.append(depth_tensor)
        
        # Stack all processed images back into a batch
        return torch.stack(depth_batch)