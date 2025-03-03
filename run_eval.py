import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from src.datasets import build_dataset
from src.datasets.kitti.dataset import kitti_benchmark_crop
from src.metrics import MetricsManager
from src.models import import_model_and_processor, MODEL_VARIANTS
from src.metrics.alignment import align_depth_least_square
import json
from pathlib import Path
import logging
import cv2
from datetime import datetime

def setup_logging(output_dir: str) -> None:
    """Setup logging configuration."""
    log_file = os.path.join(
        output_dir, f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark depth estimation models')
    
    # Dataset arguments
    parser.add_argument('--dataset-config', type=str, default='src/datasets/synth2/synth2.yaml',
                       help='Path to dataset config')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for testing')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to test on')
    
    # Model arguments
    parser.add_argument('--models', nargs='+', default=['metric3dv2'],
                       choices=list(MODEL_VARIANTS.keys()),
                       help='Models to test')
    parser.add_argument('--variants', nargs='+', 
                       default=['vit_large'],
                       help='Model variants to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
                       
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save sample visualizations')
    parser.add_argument('--save-depth-maps', action='store_true',
                       help='Save all raw depth maps paired with input images')
    parser.add_argument('--num-vis-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--alignment_max_res', type=int, default=None,
                       help='Max operating resolution used for LS alignment')
                       
    return parser.parse_args()

def save_visualization(rgb, depth_gt, depth_pred, mask, save_path, input_path=None):
    """Save visualization of results with enhanced colormap and metadata."""
    plt.switch_backend('agg')
    plt.clf()
    plt.close('all')
    
    fig = plt.figure(figsize=(20, 5), constrained_layout=True)
    
    # RGB
    ax1 = plt.subplot(141)
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.permute(1,2,0).cpu().numpy()
        rgb_np = rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    else:
        rgb_np = rgb
    rgb_np = np.clip(rgb_np, 0, 1)
    ax1.imshow(rgb_np)
    ax1.set_title('RGB Input')
    if input_path:
        ax1.set_xlabel(Path(input_path).name, fontsize=8)
    ax1.axis('off')
    
    # Ground Truth
    ax2 = plt.subplot(142)
    depth_vis = depth_gt.squeeze().cpu().numpy()
    im2 = ax2.imshow(depth_vis, cmap='magma', interpolation='nearest')
    ax2.set_title('Ground Truth Depth')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # Prediction
    ax3 = plt.subplot(143)
    if isinstance(depth_pred, torch.Tensor):
        pred_vis = depth_pred.squeeze().cpu().numpy()
    else:
        pred_vis = depth_pred
    im3 = ax3.imshow(pred_vis, cmap='magma', interpolation='nearest')
    ax3.set_title('Predicted Depth')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')
    
    # Error Map
    ax4 = plt.subplot(144)
    error = np.abs(depth_vis - pred_vis)
    error = error * mask.squeeze().cpu().numpy()
    im4 = ax4.imshow(error, cmap='hot', interpolation='nearest')
    ax4.set_title('Absolute Error')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)
    plt.close('all')

def save_depth_pair(rgb, depth, save_dir: str, index: int, basename: str):
    """Save input image and depth map pair."""
    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Process RGB
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        rgb_np = rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
    else:
        rgb_np = (rgb * 255).astype(np.uint8)

    # Process depth
    depth_np = depth.squeeze().cpu().numpy() if isinstance(depth, torch.Tensor) else depth
    depth_vis = (depth_np * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # Save images
    cv2.imwrite(
        os.path.join(rgb_dir, f"{basename}_{index:04d}.png"),
        cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(depth_dir, f"{basename}_{index:04d}.png"), depth_colored)

def main():
    args = parse_args()
    
    # Setup
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info(f"Starting benchmark with config: {args}")
    
    results = {}
    
    # Setup metrics
    metrics = ['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3']
    metrics_manager = MetricsManager(metrics)
    
    # Load dataset
    dataset_name = args.dataset_config.split('/')[-2]
    dataset = build_dataset(
        name=dataset_name,
        config_path=args.dataset_config,
        split=args.split,
        batch_size=args.batch_size
    )
    
    # Test each model
    for model_name, model_variant in zip(args.models, args.variants):
        logging.info(f"\nTesting {model_name} ({model_variant})...")
        results[f"{model_name}_{model_variant}"] = {}
        
        try:
            # Build model
            model, _ = import_model_and_processor(model_name)
            model = model(model_variant).to(args.device)
            model.eval()
            
            # Reset metrics
            metrics_manager.reset()
            
            # Process batches
            vis_count = 0
            start_idx = 0
            pbar = tqdm(total=len(dataset), desc="Processing")
            
            while start_idx < len(dataset):
                try:
                    # Get batch
                    batch, next_idx = dataset.get_batch(start_idx)
                    
                    # Move to device
                    rgb = batch['rgb'].to(args.device)
                    depth_gt = batch['depth'].to(args.device)
                    mask = batch['mask'].to(args.device)
                    
                    # Process through model
                    with torch.no_grad():
                        # Run inference              
                        depth_pred = model(rgb)
                        
                        batch_size = rgb.shape[0]
                        all_metrics = []
                        
                        for b in range(batch_size):
                
                            # Get single sample from batch
                            single_depth_pred = depth_pred[b] if batch_size > 1 else depth_pred
                            single_depth_gt = depth_gt[b]
                            single_mask = mask[b]
                            

                            depth_raw_ts = single_depth_gt
                                
                            valid_mask_ts = single_mask
                                
                            depth_raw = depth_raw_ts.cpu().numpy()
                            valid_mask = valid_mask_ts.cpu().numpy()
                            
                            # Handle Kitti dataset if needed
                            if dataset_name == "kitti" and len(single_depth_pred.shape) > 2:
                                single_depth_pred = kitti_benchmark_crop(single_depth_pred)
                                
                            # Convert to numpy for alignment
                            depth_pred_np = single_depth_pred.cpu().numpy()
                            if len(depth_pred_np.shape) == 3 and depth_pred_np.shape[0] == 1:
                                depth_pred_np = depth_pred_np.squeeze(0)  # Remove channel dim if present

                            # Align and clip depth
                            aligned_depth, scale, shift = align_depth_least_square(
                                gt_arr=depth_raw,
                                pred_arr=depth_pred_np,
                                valid_mask_arr=valid_mask,
                                return_scale_shift=True,
                                max_resolution=args.alignment_max_res,
                            )
                            
                            # Clip to dataset depth range
                            aligned_depth = np.clip(aligned_depth, a_min=dataset.min_depth, a_max=dataset.max_depth)
                            aligned_depth_ts = torch.from_numpy(aligned_depth).to(args.device)
                            
                            
                            # Add batch dimension for metrics update
                            aligned_depth_ts = aligned_depth_ts.unsqueeze(0).unsqueeze(0)         
                            depth_raw_ts = depth_raw_ts.unsqueeze(0).unsqueeze(0)    
                            valid_mask_ts = valid_mask_ts.unsqueeze(0).unsqueeze(0)
                            
                            
                            # Update metrics for this sample
                            metrics_manager.update(aligned_depth_ts, depth_raw_ts.to(args.device), valid_mask_ts.to(args.device))
                            
                            # For visualization and saving, use only the first sample in the batch
                            if b == 0:
                                if args.save_visualizations and vis_count < args.num_vis_samples:
                                    vis_dir = os.path.join(args.output_dir, f"{model_name}_{model_variant}", "visualizations")
                                    os.makedirs(vis_dir, exist_ok=True)
                                    save_visualization(
                                        rgb[0],
                                        depth_gt[0],
                                        aligned_depth_ts.squeeze(),  # Remove batch & channel dims for visualization
                                        mask[0],
                                        os.path.join(vis_dir, f"sample_{vis_count}.png"),
                                        batch['rgb_path'][0]
                                    )
                                    vis_count += 1

                                # Save depth pairs if needed
                                if args.save_depth_maps:
                                    depth_save_dir = os.path.join(args.output_dir, f"{model_name}_{model_variant}", "depth_pairs")
                                    save_depth_pair(
                                        rgb[0],
                                        aligned_depth_ts.squeeze(),  # Remove batch & channel dims for saving
                                        depth_save_dir,
                                        start_idx,
                                        Path(batch['rgb_path'][0]).stem
                                    )
                    
                    
                    # Log current metrics
                    if start_idx % 50 == 0:
                        current_metrics = metrics_manager.get_metrics()
                        logging.info(f"Current metrics at idx {start_idx}: {current_metrics}")
                    
                    # Update progress
                    pbar.update(next_idx - start_idx)
                    start_idx = next_idx
                    
                except Exception as e:
                    logging.error(f"Error processing batch at {start_idx}: {str(e)}")
                    start_idx += args.batch_size
                    continue
                
            pbar.close()
            
            # Get final metrics
            final_metrics = metrics_manager.get_metrics()
            results[f"{model_name}_{model_variant}"] = {
                k: float(v) for k, v in final_metrics.items()
            }
            
            # Print metrics
            logging.info(f"\nResults for {model_name} {model_variant}")
            logging.info("-" * 40)
            for metric_name, value in final_metrics.items():
                logging.info(f"{metric_name:>10}: {value:.4f}")
            
        except Exception as e:
            logging.error(f"Error processing model {model_name}: {str(e)}")
            continue
            
    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"\nResults saved to {results_file}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")

if __name__ == '__main__':
    main()