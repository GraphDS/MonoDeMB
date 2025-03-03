import numpy as np
import torch


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    """Align predicted depth to ground truth using least squares.
    Now supports batched inputs.
    """
    # Check if input is batched (more than 3 dimensions)
    batch_size = gt_arr.shape[0]

    aligned_preds = []
    scales = []
    shifts = []
    
    for i in range(batch_size):
        # Process each item in batch
        gt_single = gt_arr[i].squeeze()
        pred_single = pred_arr[i].squeeze()
        mask_single = valid_mask_arr[i].squeeze()
        
        # Downsample if needed
        if max_resolution is not None:
            scale_factor = np.min(max_resolution / np.array(gt_single.shape))
            if scale_factor < 1:
                downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
                gt_single = downscaler(torch.as_tensor(gt_single).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
                pred_single = downscaler(torch.as_tensor(pred_single).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
                mask_single = downscaler(torch.as_tensor(mask_single).unsqueeze(0).unsqueeze(0).float()).squeeze().bool().numpy()
        
        # Extract valid pixels
        gt_masked = gt_single[mask_single].reshape((-1, 1))
        pred_masked = pred_single[mask_single].reshape((-1, 1))
        
        # Skip if no valid pixels
        if gt_masked.size == 0:
            # If no valid pixels, just return original
            aligned_preds.append(pred_arr[i])
            scales.append(1.0)
            shifts.append(0.0)
            continue
            
        # Compute alignment using least squares
        _ones = np.ones_like(pred_masked)
        A = np.concatenate([pred_masked, _ones], axis=-1)
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X
        
        # Apply alignment
        aligned_pred = pred_arr[i] * scale + shift
        
        aligned_preds.append(aligned_pred)
        scales.append(scale)
        shifts.append(shift)
        
    # Stack results
    aligned_pred_arr = np.stack(aligned_preds)
    scale_arr = np.array(scales)
    shift_arr = np.array(shifts)
    
    if return_scale_shift:
        return aligned_pred_arr, scale_arr, shift_arr
    else:
        return aligned_pred_arr