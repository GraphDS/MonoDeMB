# --------------------------------------------------------
# What Matters When Repurposing Diffusion Models for General Dense Perception Tasks? (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024, Advanced Intelligent Machines (AIM)
# Licensed under The BSD 2-Clause License [see LICENSE for details]
# Author: Guangkai Xu (https://github.com/guangkaixu/)
# --------------------------------------------------------------------------
# This code is based on Marigold and diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/aim-uofa/GenPercept#%EF%B8%8F-citation
# More information about the method can be found at https://github.com/aim-uofa/GenPercept
# --------------------------------------------------------------------------

import torch

from .base_dataset import BaseDataset, PerceptionFileNameMode


class NYUDataset(BaseDataset):
    def __init__(
        self,
        eigen_valid_mask: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            # NYUv2 dataset parameter
            min_depth=1e-3,
            max_depth=10.0,
            has_filled_depth=True,
            name_mode=PerceptionFileNameMode.rgb_id,
            **kwargs,
        )

        self.eigen_valid_mask = eigen_valid_mask

    def _read_depth_file(self, rel_path):
        print('NYUDataset._read_depth_file', rel_path)
        depth_in = self._read_image(rel_path)
        # Decode NYU depth
        print('depth_in', depth_in.shape, depth_in)
        if not self.is_exr_data:
            depth_decoded = depth_in / 1000.0
            print('depth_decoded', depth_decoded.min(), depth_decoded.max(), depth_decoded.shape)
        else:
            depth_decoded = depth_in
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = super()._get_valid_mask(depth) # убирает там где больше макс и меньше мин значений min_depth / max_depth

        # Eigen crop for evaluation
        if self.eigen_valid_mask:
            print(f'self.eigen_valid_mask {self.eigen_valid_mask}')
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            eval_mask[45:471, 41:601] = 1
            print('eval_mask[45:471, 41:601]', eval_mask.shape, eval_mask.sum())
            eval_mask.reshape(valid_mask.shape)
            print('eval_mask', eval_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)

        return valid_mask
