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
            img: RGB image in range [0, 1] and shape [H, W, C]

        Returns:
            torch.Tensor: Predicted depth map
        """
        # Convert to PIL Image if needed
        if isinstance(img, np.ndarray):
            # img = Image.fromarray((img * 255).astype(np.uint8))
            img = img.squeeze().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            img = Image.fromarray(img)


        # Run prediction
        output = self.pipeline(
            img,
            denoising_steps=self.settings["denoising_steps"],
            ensemble_size=self.settings["ensemble_size"],
            processing_res=768,
            match_input_res=True,
            batch_size=0,
            color_map="Spectral",
            show_progress_bar=False,
        )

        # Get depth map and normalize
        depth = output.depth_np
        depth = (depth - np.mean(depth)) / np.std(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        return torch.from_numpy(depth)
