import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTokenizer
)
from .geowizard_models.unet_2d_condition import UNet2DConditionModel
from .geowizard_models.geowizard_pipeline import DepthNormalEstimationPipeline as PipelineV1
from .geowizard_models.geowizard_v2_pipeline import DepthNormalEstimationPipeline as PipelineV2
from .geowizard_utils.seed_all import seed_all

from ..base_model import BaseModel

class GeoWizardWrapper(BaseModel):
    """Wrapper for GeoWizard model."""
    
    def __init__(self, model_type="v1", checkpoint_path="stabilityai/stable-diffusion-2", domain="outdoor"):
        """Initialize GeoWizard model.
        
        Args:
            model_type: Type of model ("v1" or "v2")
            checkpoint_path: Path to model checkpoint or HF model name
            domain: Domain prediction type ("indoor" or "outdoor")
        """
        super().__init__(model_name=f"geowizard_{model_type}")
        
        self.model_type = model_type
        self.domain = domain
        checkpoint_path = 'GonzaloMG/geowizard-e2e-ft'
        
        # Common components
        self.vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder='vae')
        self.scheduler = DDIMScheduler.from_pretrained(checkpoint_path, subfolder='scheduler')
        self.unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
        
        # Version specific components
        if model_type == "v1":
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                checkpoint_path, subfolder="image_encoder"
            )
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                checkpoint_path, subfolder="feature_extractor"
            )
            
            self.pipe = PipelineV1(
                vae=self.vae,
                image_encoder=self.image_encoder,
                feature_extractor=self.feature_extractor,
                unet=self.unet,
                scheduler=self.scheduler
            )
            
        elif model_type == "v2":
            self.text_encoder = CLIPTextModel.from_pretrained(
                checkpoint_path, subfolder="text_encoder"
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                checkpoint_path, subfolder="tokenizer"
            )
            
            self.pipe = PipelineV2(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

    def to(self, device):
        """Move model to device."""
        self.pipe = self.pipe.to(device)
        return self

    def forward(self, img, steps=10, ensemble_size=3):
        """Run inference on image.
        
        Args:
            img: Input image in range [0,1] and format CHW
            steps: Number of denoising steps
            ensemble_size: Number of predictions to ensemble
            
        Returns:
            tuple: (depth_map, normal_map) tensors
        """
        # Convert tensor to PIL Image for pipeline
        if torch.is_tensor(img):
            # img = img.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            # img = (img * 255).astype(np.uint8)
            # img = Image.fromarray(img)
            img = img.squeeze().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            img = Image.fromarray(img)

        # Run inference
        pipe_out = self.pipe(
            img,
            denoising_steps=steps,
            ensemble_size=ensemble_size,
            processing_res=768,  # Default from paper
            match_input_res=True,
            domain=self.domain,
            color_map="Spectral",
            show_progress_bar=False
        )

        # Get predictions
        depth = torch.from_numpy(pipe_out.depth_np).float()
        normals = torch.from_numpy(pipe_out.normal_np).float()
        
        # Ensure predictions are on same device as input
        if torch.is_tensor(img):
            depth = depth.to(img.device)
            normals = normals.to(img.device)
            
        return depth
