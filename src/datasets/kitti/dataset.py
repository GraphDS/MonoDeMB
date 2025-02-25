import os
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import skimage
import time
import scipy
import torch
from ..base_dataset import BaseDataset, register_dataset
from ..common_utils import download_and_extract

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    """Interpolate sparse depth map using colorization method."""
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    new_vals = scipy.sparse.linalg.spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    denoisedDepthImg = new_vals * maxImgAbsDepth
    
    output = denoisedDepthImg.reshape((H, W)).astype('float32')
    output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
    return output

def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216
    height, width = input_img.shape[-2:]
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape):
        out = input_img[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    return out


@register_dataset('kitti')
class KITTIDataset(BaseDataset):
    """KITTI depth dataset."""
    min_depth=1e-5
    max_depth=80.0
    
    
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Find all matching RGB-D pairs in the dataset.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing paired 'rgb' and 'depth' file paths
        """
        data_pairs = []

        # Iterate through drive dates
        for drive_date in os.listdir(self.root_dir):
            if drive_date.endswith("sync"):
                continue

            date_path = os.path.join(self.root_dir, drive_date)

            # Iterate through drive numbers
            for drive_num in os.listdir(date_path):
                rgb_dir = os.path.join(date_path, drive_num, "image_02", "data")
                depth_dir = os.path.join(
                    self.root_dir, drive_num, "proj_depth/groundtruth/image_02"
                )

                # Skip if RGB directory doesn't exist
                if not os.path.exists(rgb_dir):
                    continue

                # Get all RGB images
                for img_name in os.listdir(rgb_dir):
                    if not img_name.endswith(".png"):
                        continue

                    rgb_path = os.path.join(rgb_dir, img_name)
                    depth_path = os.path.join(depth_dir, img_name)

                    # Verify both files exist
                    try:
                        # Quick verification that depth file can be opened
                        Image.open(depth_path)

                        # Add valid pair to dataset
                        data_pairs.append({"rgb": rgb_path, "depth": depth_path})
                    except (FileNotFoundError, IOError):
                        continue

        return sorted(data_pairs, key=lambda x: x["rgb"])

    
    
    def _load_depth(self, path: str) -> np.ndarray:
        """Load KITTI depth map.
        
        KITTI depth maps are uint16 PNGs with depth values
        encoded in the actual pixel values.
        """
        depth = np.asarray(Image.open(path)).squeeze()
        depth = depth / 256.0
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        depth_crop = kitti_benchmark_crop(depth)
                    
        return depth_crop
    
    def _get_valid_mask(self, depth: torch.Tensor, valid_mask_crop="eigen"):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = super()._get_valid_mask(depth)
        if valid_mask_crop is not None:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            print('eval_mask.shape', eval_mask.shape)
            gt_height, gt_width = eval_mask.shape

            if "garg" == valid_mask_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == valid_mask_crop:
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1

            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)
        return valid_mask
    
    def _load_rgb_image(self, path:str) -> torch.Tensor:
        rgb = np.array(Image.open(path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)#[:3,:,:]
        return kitti_benchmark_crop(rgb_torch)
    
    def _download(self):
        """Download and extract KITTI dataset if not already present."""
        # Check if data already exists
        if os.path.exists(self.root_dir) and len(os.listdir(self.root_dir)) > 0:
            print("KITTI dataset already exists.")
            return

        url = "https://huggingface.co/datasets/guangkaixu/genpercept_datasets_eval/resolve/main/eval_kitti_genpercept.tar.gz?download=true"
        print("Downloading KITTI dataset...")
        download_and_extract(
            url=url,
            download_dir=os.path.dirname(self.root_dir),
            extract_dir=os.path.dirname(self.root_dir),
        )
    