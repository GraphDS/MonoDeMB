import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Conv2d
import numpy as np
from PIL import Image
import os
import os.path as osp
from safetensors.torch import load_file, load_model
from peft import LoraConfig

from diffusers import UNet2DConditionModel, AutoencoderKL
from .genpercept import GenPerceptPipeline
from .genpercept_src.customized_modules.ddim import DDIMSchedulerCustomized
from .genpercept.models.custom_unet import CustomUNet2DConditionModel
from .genpercept.models.dpt_head import DPTNeckHeadForUnetAfterUpsample, DPTNeckHeadForUnetAfterUpsampleIdentity
from transformers import DPTConfig

from ..base_model import BaseModel

import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _replace_unet_conv_in(unet):
    """Replace UNet input conv layer to accept 8 channels."""
    logger.info("Replacing UNet input conv layer...")
    
    # Get current weights and bias
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    logger.debug(f"Original conv weight shape: {_weight.shape}")
    
    # Expand channels
    _weight = _weight.repeat((1, 2, 1, 1))
    _weight *= 0.5  # Scale activations
    logger.debug(f"Modified conv weight shape: {_weight.shape}")
    
    # Create new conv layer
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        8, _n_convin_out_channel,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1)
    )
    
    # Set weights and bias
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    
    # Update config
    unet.config["in_channels"] = 8
    logger.info("UNet input layer successfully replaced (4->8 channels)")
    
    return unet


class GenPerceptWrapper(BaseModel):
    """Wrapper for GenPercept model."""
    
    def __init__(self, model_type="sd21", 
                 checkpoint="stabilityai/stable-diffusion-2-1"):
        """Initialize GenPercept model."""
        logger.info(f"Initializing GenPercept wrapper (type={model_type})")
        super().__init__(model_name=f"genpercept_{model_type}")
        
        self.archs = 'genpercept'#'marigold'
        self.half_precision = False
        self.unet = '/weights/genpercept-models/unet_depth_v2'
        self.checkpoint = checkpoint
        self.lora_rank=0
        self.scheduler = "src/models/genpercept_wrapper/genpercept_hf_configs/scheduler_beta_1.0_1.0"
        self.ensemble_size = 1
        self.output_processing_res = False
        self.processing_res = 768
        self.resample_method = 'bilinear'
        
        pre_loaded_dict = {}
        ####
        self.load_decoder_ckpt = self.unet
     
            
        checkpoint_path = self.checkpoint
        ensemble_size = self.ensemble_size
        if ensemble_size > 15 and self.archs != 'genpercept':
            logging.warning("Running with large ensemble size will be slow.")
        match_input_res = not self.output_processing_res
        if 0 == self.processing_res and match_input_res is False:
            logging.warning(
                "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
            )
        #####
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if self.half_precision:
            dtype = torch.float16
            variant = "fp16"
            logging.warning(
                f"Running with half precision ({dtype}), might lead to suboptimal result."
            )
        else:
            dtype = torch.float32
            variant = None
        # NOTE: deal with guangkaixu/genpercept-models. It cannot detect whether customized head is used or not.
        if 'genpercept-models' in self.unet:
            unet_model_subfolder = ""
            if 'unet_disparity_dpt_head_v2' in self.unet:
                self.load_decoder_ckpt = osp.dirname(self.unet)
            else:
                self.load_decoder_ckpt = None
        else:
            unet_model_subfolder = 'unet'
            self.load_decoder_ckpt = self.unet

        pre_loaded_dict = dict()
        if self.load_decoder_ckpt: # NOTE: path to the checkpoint folder does not contain 'vae' or 'customized_head'
            if 'dpt_head_identity' in os.listdir(self.load_decoder_ckpt):
                sub_dir = "dpt_head_identity" 
                dpt_config = DPTConfig.from_pretrained("src/models/genpercept_wrapper/genpercept_hf_configs/dpt-sd2.1-unet-after-upsample-general")
                loaded_model = DPTNeckHeadForUnetAfterUpsampleIdentity(config=dpt_config)
                load_model(loaded_model, osp.join(self.load_decoder_ckpt, sub_dir, 'model.safetensors'))
                pre_loaded_dict['customized_head'] = loaded_model.to(dtype=dtype).to(device=device)
            elif 'dpt_head' in os.listdir(self.load_decoder_ckpt):
                sub_dir = "dpt_head" 
                dpt_config = DPTConfig.from_pretrained("src/models/genpercept_wrapper/genpercept_hf_configs/dpt-sd2.1-unet-after-upsample-general")
                loaded_model = DPTNeckHeadForUnetAfterUpsample(config=dpt_config)
                load_model(loaded_model, osp.join(self.load_decoder_ckpt, sub_dir, 'model.safetensors'))
                pre_loaded_dict['customized_head'] = loaded_model.to(dtype=dtype).to(device=device)
            elif 'vae_decoder' in os.listdir(self.load_decoder_ckpt) and 'vae_post_quant_conv' in os.listdir(args.load_decoder_ckpt):
                vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder='vae')
                load_model(vae.decoder, osp.join(self.load_decoder_ckpt, 'vae_decoder', 'model.safetensors'))
                load_model(vae.post_quant_conv, osp.join(self.load_decoder_ckpt, 'vae_post_quant_conv', 'model.safetensors'))
                pre_loaded_dict['vae'] = vae.to(dtype=dtype).to(device=device)
        
        if self.unet:
            if 'customized_head' in pre_loaded_dict.keys():
                unet = CustomUNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
                del unet.conv_out
                del unet.conv_norm_out
            else:
                unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
            
            if (8 != unet.config["in_channels"]) and (self.archs == 'marigold'):
                unet = _replace_unet_conv_in(unet)

            if osp.exists(osp.join(self.unet, unet_model_subfolder, 'diffusion_pytorch_model.bin')):
                unet_ckpt_path = osp.join(self.unet, unet_model_subfolder, 'diffusion_pytorch_model.bin')
            elif osp.exists(osp.join(self.unet, unet_model_subfolder, 'diffusion_pytorch_model.safetensors')):
                unet_ckpt_path = osp.join(self.unet, unet_model_subfolder, 'diffusion_pytorch_model.safetensors')
            else:
                unet_ckpt_path = osp.join(checkpoint_path, 'unet', 'diffusion_pytorch_model.safetensors')

            ckpt = load_file(unet_ckpt_path)
            if 'customized_head' in pre_loaded_dict.keys():
                ckpt_new = {}
                for key in ckpt:
                    if 'conv_out' in key:
                        continue
                    if 'conv_norm_out' in key:
                        continue
                    ckpt_new[key] = ckpt[key]
            else:
                ckpt_new = ckpt
            
            if self.lora_rank > 0:
                unet_lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_rank,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                # Add adapter and make sure the trainable params are in float32.
                unet.add_adapter(unet_lora_config)
                unet.requires_grad_(False)

            unet.load_state_dict(ckpt_new)
            pre_loaded_dict['unet'] = unet.to(dtype=dtype).to(device=device)
        else:
            unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet')
        
        
                
        if self.archs == 'marigold' or self.archs == 'rgb_blending':
            if self.scheduler is not None:
                pre_loaded_dict['scheduler'] = DDIMSchedulerCustomized.from_pretrained(
                    self.scheduler, 
                    subfolder='scheduler'
                )

            genpercept_pipeline = False
            pipe: GenPerceptPipeline = GenPerceptPipeline.from_pretrained(
                checkpoint_path, variant=variant, torch_dtype=dtype, rgb_blending=(self.archs != 'marigold'), genpercept_pipeline=genpercept_pipeline, **pre_loaded_dict
            )

        elif self.archs == 'genpercept':
            pre_loaded_dict['scheduler'] = DDIMSchedulerCustomized.from_pretrained(self.scheduler)

            genpercept_pipeline = True
            pipe: GenPerceptPipeline = GenPerceptPipeline.from_pretrained(
                checkpoint_path, variant=variant, torch_dtype=dtype, genpercept_pipeline=genpercept_pipeline, **pre_loaded_dict
            )
        else:
            raise NotImplementedError

        del pre_loaded_dict

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            logging.debug("run without xformers")

        self.pipe = pipe.to(device)

    def to(self, device):
        """Move model to device."""
        logger.info(f"Moving model to device: {device}")
        try:
            self.pipe = self.pipe.to(device)
            logger.info("Model successfully moved to device")
            return self
        except Exception as e:
            logger.error(f"Error moving model to device: {str(e)}")
            raise

    def forward(self, img):
        """Run inference on image batch.
        
        Args:
            img: RGB input in BCHW format where B is batch size
            
        Returns:
            torch.Tensor: Predicted depth maps, batched
        """
        device = img.device if hasattr(img, 'device') else None
        batch_size = img.shape[0]
        depth_batch = []
        
        logger.info(f"Processing batch of {batch_size} images")
        
        for i in range(batch_size):
            try:
                # Extract single image from batch
                single_img = img[i]  # [C, H, W]
                
                # Convert tensor to PIL Image
                logger.info(f"Converting image {i} to PIL format")
                img_np = single_img.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                pil_img = Image.fromarray((img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8))
                
                # Run inference
                logger.info(f"Running inference on image {i}")
                out = self.pipe(
                    pil_img,
                    denoising_steps=1,
                    ensemble_size=1,
                    processing_res=self.processing_res,
                    match_input_res=True,
                    batch_size=1,
                    color_map=None,
                    show_progress_bar=False,
                    resample_method=self.resample_method,
                    mode="depth",
                    fix_timesteps=None,
                    prompt=""
                )

                # Process output
                depth = torch.from_numpy(out.pred_np).float()
                logger.info(f"Image {i} depth range: [{depth.min():.3f}, {depth.max():.3f}]")
                
                # Move to correct device if needed
                if device is not None:
                    depth = depth.to(device)
                
                depth_batch.append(depth)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                # Create a placeholder depth map in case of error
                if i > 0 and depth_batch:
                    # Use the same shape as a previous successful image
                    depth_placeholder = torch.zeros_like(depth_batch[0])
                else:
                    # Create a small placeholder if we don't have any successful predictions yet
                    depth_placeholder = torch.zeros((384, 384), device=device)
                
                depth_batch.append(depth_placeholder)
                logger.warning(f"Using placeholder depth map for image {i}")
        
        # Stack all depth maps
        logger.info(f"Stacking {len(depth_batch)} depth maps")
        return torch.stack(depth_batch)