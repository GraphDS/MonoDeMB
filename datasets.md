# Dataset Structures Overview

```
KITTI/
├── 2011_09_26/                      # Date directory
│   ├── 2011_09_26_drive_0002_sync/  # Drive directory
│   │   └── image_02/                # Left camera images
│   │       └── data/
│   │           ├── 0000000000.png
│   │           ├── 0000000003.png
│   │           └── ...
│   └── ...
└── 2011_09_26_drive_0002_sync/      # Drive directories in root
    └── proj_depth/                  # Depth data
        └── groundtruth/
            └── image_02/
                ├── 0000000000.png
                ├── 0000000003.png
                └── ...

File formats:
- RGB: PNG (cropped to 352×1216 pixels using kitti_benchmark_crop)
- Depth: PNG (16-bit, converted to meters by dividing by 256.0)
```

## 2. DIODE Dataset
```
DIODE/
├── indoors/                         # Indoor scenes
│   ├── scene_00001/                 # Scene directory
│   │   └── scan_00001/              # Scan directory
│   │       ├── 00001_00001_indoors_xxx_yyy.png         # RGB
│   │       ├── 00001_00001_indoors_xxx_yyy_depth.npy   # Depth (numpy file)
│   │       └── 00001_00001_indoors_xxx_yyy_depth_mask.npy  # Mask (numpy file)
│   └── ...
└── outdoors/                        # Outdoor scenes (similar structure)

File formats:
- RGB: PNG (possibly 768×1024 pixels)
- Depth: NPY (float32 numpy arrays)
- Mask: NPY (boolean numpy arrays)
```

## 3. NYU Depth V2 Dataset
```
NYU_DEPTH_V2/
├── test/                            # Test split
│   ├── bedroom/                     # Scene category
│   │   ├── rgb_0001.png             # RGB image
│   │   ├── depth_0001.png           # Corresponding depth
│   │   ├── rgb_0002.png
│   │   └── ...
│   ├── kitchen/
│   └── ...
├── train/                           # Train split (similar structure)
└── val/                             # Validation split (similar structure)

File formats:
- RGB: PNG (640×480 pixels)
- Depth: PNG (16-bit, converted from millimeters to meters by dividing by 1000.0)
```

## 4. ETH3D
```
ETH3D/
├── rgb/
│   ├── scene_name/                    # Example: 'courtyard'
│   │   └── images/
│   │       └── dslr_images/           # Contains RGB images
│   │           ├── DSC_0001.JPG       # RGB image (high-res DSLR)
│   │           ├── DSC_0002.JPG
│   │           └── ...
│   └── ...
└── depth/
    ├── scene_name_dslr_depth/         # Example: 'courtyard_dslr_depth'
    │   └── scene_name/
    │       └── ground_truth_depth/
    │           └── dslr_images/       # Contains depth files
    │               ├── DSC_0001.JPG   # Binary depth file (not actual JPG)
    │               ├── DSC_0002.JPG
    │               └── ...
    └── ...

File formats:
- RGB: JPG (4032×6048 pixels)
- Depth: Binary files (4-byte float, 4032×6048 pixels)
  * Stored as raw binary data in row-major order
  * Invalid depth values are set to infinity
  * Fixed dimensions of 4032×6048 pixels
```
## 5. Synth2

Check out [separate description](synth2_description.md)

## Key Differences (Updated):

1. Directory Structure:
- KITTI: Complex hierarchical (date/drive/camera)
- DIODE: Three-level (environment/scene/scan)
- NYU: Scene-based hierarchy with official splits

2. File Formats:
- RGB Images:
  * KITTI: PNG (1242×375)
  * DIODE: PNG (768×1024)
  * NYU: JPG (640×480)

- Depth Maps:
  * KITTI: 16-bit PNG (needs scaling by 256.0)
  * DIODE: NPY arrays with masks
  * NYU: 16-bit PNG (needs scaling by 1000.0)

# Adding New Datasets: Testing Guide

1. Dataset Configuration
```python
# Add to DATASET_CONFIGS in test_datasets.py
DATASET_CONFIGS = {
    'your_dataset': {
        'config': 'src/datasets/your_dataset/config.yaml',
        'split': 'test',  # or appropriate default split
        'depth_range': [min_depth, max_depth],  # or None if variable
        'colormap': 'plasma'  # or appropriate colormap
    },
    ...
}
```

2. Required Tests
All datasets must pass these basic tests:

```python
def test_required_methods(dataset):
    """Verify that dataset implements all required methods."""
    assert hasattr(dataset, '_traverse_directory'), "Missing _traverse_directory method"
    assert hasattr(dataset, '_load_depth'), "Missing _load_depth method"
    
    # Verify data format
    sample = dataset[0]
    assert {'rgb', 'depth', 'mask'}.issubset(sample.keys()), "Missing required keys"
    assert sample['rgb'].shape[0] == 3, "RGB should be CHW format"
    assert sample['depth'].shape[0] == 1, "Depth should be 1-channel"
```

3. Testing Structure
Each new dataset should be tested for:
- Dataset Loading
- Batch Loading
- Sample Format
- Visualization
- Statistics Calculation

4. Common Test Requirements:
```yaml
# Example config.yaml structure
paths:
  root_dir: "/path/to/dataset"
  train: "train"
  test: "test"
  val: "val"

preprocessing:
  rgb_mean: [0.485, 0.456, 0.406]  # ImageNet normalization
  rgb_std: [0.229, 0.224, 0.225]   # ImageNet normalization
  target_size: [H, W]              # Standard size
```

5. Adding Custom Tests
```python
@pytest.mark.parametrize("dataset_name", DATASET_CONFIGS.keys())
def test_dataset_specific_feature(dataset_name):
    """Test dataset-specific functionality."""
    with DatasetTestProgress() as progress:
        task_id = progress.add_task(
            f"[cyan]Testing {dataset_name} specific feature", 
            total=steps_count
        )
        
        try:
            # Your test code here
            progress.update(task_id, advance=1)
            logger.info(f"✓ Test passed for {dataset_name}")
        except Exception as e:
            logger.error(f"✗ Test failed for {dataset_name}: {str(e)}")
            pytest.fail(str(e))
```
