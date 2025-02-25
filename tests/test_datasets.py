import pytest
import torch
from src.datasets import build_dataset
from pathlib import Path
from .dataset_test_utils import visualize_sample, calculate_dataset_statistics

# Updated configurations - all use 'test' split for consistency
DATASET_CONFIGS = {
    'nyu': {
        'config': 'src/datasets/nyu/nyu.yaml',
        'split': 'test',
        'depth_range': [0, 10],  # meters
        'colormap': 'plasma'
    },
    'eth3d': {
        'config': 'src/datasets/eth3d/eth3d.yaml',
        'split': 'test',
        'depth_range': None,
        'colormap': 'plasma'
    },
    'kitti': {
        'config': 'src/datasets/kitti/kitti.yaml',
        'split': 'test',
        'depth_range': [0, 80],  # meters
        'colormap': 'plasma'
    },
    'diode': {
        'config': 'src/datasets/diode/diode.yaml',
        'split': 'test',  # Changed from 'val' to 'test'
        'depth_range': None,
        'colormap': 'plasma'
    }
}

# Function to check basic sample structure without strict format requirements
def check_sample_basic(sample):
    """Verify that a sample has the required keys and contains data."""
    print("\nChecking sample basic structure:")
    
    # Check if required keys exist
    for key in ['rgb', 'depth', 'mask']:
        assert key in sample, f"Sample missing {key} data"
        print(f"{key} is present")
        
        # Check that the values contain data
        assert sample[key] is not None, f"{key} is None"
        
        # Print shape for information
        if hasattr(sample[key], 'shape'):
            print(f"{key} shape: {sample[key].shape}")
    
    # Print some basic statistics for tensors
    if torch.is_tensor(sample['rgb']):
        print(f"RGB min/max: {sample['rgb'].min().item():.2f}/{sample['rgb'].max().item():.2f}")
    if torch.is_tensor(sample['depth']):
        print(f"Depth min/max: {sample['depth'].min().item():.2f}/{sample['depth'].max().item():.2f}")
    if torch.is_tensor(sample['mask']):
        print(f"Mask sum: {sample['mask'].sum().item()}")

@pytest.mark.parametrize("dataset_name", DATASET_CONFIGS.keys())
def test_dataset_loading(dataset_name):
    """Test that each dataset can be loaded successfully and has 10 samples."""
    config = DATASET_CONFIGS[dataset_name]
    
    # Try to build dataset
    try:
        dataset = build_dataset(
            name=dataset_name,
            config_path=config['config'],
            split=config['split']
        )
        
        # Assert dataset is not empty
        assert len(dataset) > 0, f"Dataset {dataset_name} is empty"
        
        # Limit to 10 samples for testing
        dataset.data_pairs = dataset.data_pairs[:min(10, len(dataset.data_pairs))]
        assert len(dataset) <= 10, f"Dataset {dataset_name} should have at most 10 samples for testing"
        
        print(f"\nDataset {dataset_name} loaded with {len(dataset)} samples")
        
        # Log first sample path
        first_sample = dataset[0]
        print(f"First sample RGB path: {first_sample.get('rgb_path', 'Not available')}")
        
    except Exception as e:
        pytest.fail(f"Failed to load dataset {dataset_name}: {str(e)}")

@pytest.mark.parametrize("dataset_name", DATASET_CONFIGS.keys())
def test_dataset_batch_loading(dataset_name):
    """Test that we can load batches from each dataset."""
    config = DATASET_CONFIGS[dataset_name]
    
    try:
        # Build dataset with batch size of 2
        dataset = build_dataset(
            name=dataset_name,
            config_path=config['config'],
            split=config['split'],
            batch_size=2
        )
        
        # Limit to 10 samples for testing
        dataset.data_pairs = dataset.data_pairs[:min(10, len(dataset.data_pairs))]
        
        # Try to load a batch
        batch, next_idx = dataset.get_batch(0)
        
        # Basic checks
        assert isinstance(batch, dict), "Batch should be a dictionary"
        assert 'rgb' in batch, "Batch missing RGB data"
        assert 'depth' in batch, "Batch missing depth data"
        assert 'mask' in batch, "Batch missing mask data"
        
        # Check if we got the correct number of samples
        expected_batch_size = min(2, len(dataset))
        assert len(batch['rgb']) == expected_batch_size, f"Expected batch size {expected_batch_size}, got {len(batch['rgb'])}"
        
        # Check next_idx
        assert next_idx == expected_batch_size, f"Next index should be {expected_batch_size}, got {next_idx}"
        
        print(f"\nSuccessfully loaded batch from {dataset_name} dataset")
        print(f"Batch contains {len(batch['rgb'])} samples")
        
    except Exception as e:
        pytest.fail(f"Failed to load batch from {dataset_name}: {str(e)}")

@pytest.mark.parametrize("dataset_name", DATASET_CONFIGS.keys())
def test_dataset_sample_content(dataset_name):
    """Test that samples from each dataset contain valid content."""
    config = DATASET_CONFIGS[dataset_name]
    
    try:
        dataset = build_dataset(
            name=dataset_name,
            config_path=config['config'],
            split=config['split']
        )
        
        # Limit to 10 samples for testing
        dataset.data_pairs = dataset.data_pairs[:min(10, len(dataset.data_pairs))]
        
        # Check first sample
        sample = dataset[0]
        check_sample_basic(sample)
        
        # Basic validation checks
        if torch.is_tensor(sample['rgb']):
            assert sample['rgb'].numel() > 0, "RGB tensor should not be empty"
        
        if torch.is_tensor(sample['depth']):
            assert sample['depth'].numel() > 0, "Depth tensor should not be empty"
        
        if torch.is_tensor(sample['mask']):
            assert sample['mask'].numel() > 0, "Mask tensor should not be empty"
            
    except Exception as e:
        pytest.fail(f"Sample content check failed for {dataset_name}: {str(e)}")

@pytest.mark.parametrize("dataset_name", DATASET_CONFIGS.keys())
def test_dataset_visualization(dataset_name, tmp_path):
    """Test that we can visualize samples from each dataset."""
    config = DATASET_CONFIGS[dataset_name]
    
    try:
        dataset = build_dataset(
            name=dataset_name,
            config_path=config['config'],
            split=config['split']
        )
        
        # Limit to 10 samples for testing
        dataset.data_pairs = dataset.data_pairs[:min(10, len(dataset.data_pairs))]
        
        # Get first sample
        sample = dataset[0]
        output_dir = tmp_path / "vis"
        output_dir.mkdir(exist_ok=True)
        
        # Ensure tensors have the right dimensions for visualization
        depth = sample['depth']
        if torch.is_tensor(depth):
            if len(depth.shape) == 2:
                depth = depth.unsqueeze(0)  # Add channel dimension
            sample['depth'] = depth
            
        mask = sample['mask']
        if torch.is_tensor(mask):
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension
            sample['mask'] = mask
        
        # Call visualization function
        visualize_sample(
            sample, 
            str(output_dir), 
            "test", 
            dataset_name,
            colormap=config['colormap'],
            depth_range=config['depth_range']
        )
        
        # Check that visualization was saved
        vis_files = list(output_dir.glob("*.png"))
        assert len(vis_files) > 0, "No visualization was saved"
        print(f"\nVisualization saved for {dataset_name} dataset")
        
    except Exception as e:
        pytest.fail(f"Visualization failed for {dataset_name}: {str(e)}")