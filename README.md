# MonoDeMB

This repository provides source code for "MonoDeMB: Comprehensive Monocular DepthMap Benchmark" paper

![scheme](figs/pipeline_scheme.png)

## Project Structure

```
.
├── src/
│   ├── datasets/                    # Dataset implementations
│   │   ├── base_dataset.py          # Base dataset class
│   │   ├── nyu/                     # NYU Depth V2 dataset
│   │   ├── kitti/                   # KITTI dataset
│   │   ├── diode/                   # DIODE dataset
│   │   └── eth3d/                   # ETH3D dataset
│   │
│   ├── models/                      # Model implementations
│   │   ├── base_model.py           # Base model class
│   │   ├── midas/                  # MiDaS model wrapper
│   │   ├── leres_wrapper/          # LeReS model wrapper
│   │   ├── depth_anything_wrapper/ # DepthAnything wrapper
│   │   ├── genpercept_wrapper/     # GenPercept wrapper
│   │   ├── metric3dv2_wrapper/     # Metric3Dv2 wrapper
│   │   ├── unidepth_wrapper/       # UniDepth wrapper
│   │   ├── marigold_wrapper/       # Marigold wrapper
│   │   └── geowizard_wrapper/      # GeoWizard wrapper
│   │
│   └── metrics/                     # Evaluation metrics
│       ├── base_metric.py          # Base metric class
│       ├── abs_rel.py              # Absolute relative error
│       ├── rmse.py                 # Root mean square error
│       ├── silog.py                # Scale invariant logarithmic error
│       └── threshold_metrics.py    # Delta threshold metrics
│
├── tests/                          # Unit tests
├── run_eval.py                     # Main evaluation script
├── DATASETS.md                     # Detailed dataset documentation
└── README.md                       # This file
```

## Quick Links
- For detailed guidance on extending the framework, see the sections below
- For detailed dataset information, look at [DATASETS.md](DATASETS.md)

# Extending the Framework

This guide explains how to extend the framework with custom models, datasets, and metrics.

## Adding a Custom Dataset

1. Create a new directory in `src/datasets/your_dataset/`:
```
src/datasets/your_dataset/
├── __init__.py           # Export your dataset class
├── dataset.py            # Dataset implementation
└── your_dataset.yaml     # Dataset configuration
```

2. Create dataset class:
```python
# dataset.py
from ..base_dataset import BaseDataset, register_dataset

@register_dataset('your_dataset_name')
class YourDataset(BaseDataset):
    """Your dataset implementation."""
    
    def __init__(self, config_path: str, split: str = 'test', batch_size: int = 1):
        """Initialize dataset.
        
        Args:
            config_path: Path to dataset config file
            split: Dataset split (train/val/test)
            batch_size: Batch size for loading data
        """
        super().__init__(config_path, split, batch_size)
    
    def _traverse_directory(self) -> List[Dict[str, str]]:
        """Get paths to RGB-depth pairs.
        
        Returns:
            List of dicts containing paths for image and depth pairs
        """
        data_pairs = []
        # Implement your directory traversal logic:
        # 1. Get all RGB and depth paths
        # 2. Match pairs based on naming/structure
        # 3. Validate pairs exist
        return data_pairs
        
    def _load_depth(self, path: str) -> np.ndarray:
        """Load depth map from file.
        
        Args:
            path: Path to depth map file
            
        Returns:
            Depth map as numpy array [H, W]
        """
        # Implement depth loading with proper normalization
        return depth
```

3. Create configuration file:
```yaml
# your_dataset.yaml
name: 'Your Dataset'
description: 'Description of your dataset'

paths:
  root_dir: '/path/to/data'
  train: 'train'
  val: 'val'  
  test: 'test'

preprocessing:
  rgb_mean: [0.485, 0.456, 0.406]
  rgb_std: [0.229, 0.224, 0.225]
  target_size: [480, 640]  # [height, width]
  depth_scale: 1000.0  # If depths are in mm
  max_depth: 10.0     # Maximum depth in meters
```

4. Update `src/datasets/__init__.py`:
```python
from .your_dataset import YourDataset

__all__ = [
    # existing datasets
    'YourDataset',
]
```

There is an example of simple custom dataset - please, check out the [mock source code](src/datasets/mock_dataset).

## Adding a Custom Model

1. Create a new directory in `src/models/your_model_wrapper/`:
```
src/models/your_model_wrapper/
├── __init__.py          # Export your model and utilities
├── model.py             # Model wrapper implementation
└── utils.py             # Model-specific utilities
```

2. Create model wrapper:
```python
# model.py
import torch
from ...base_model import BaseModel

class YourModelWrapper(BaseModel):
    """Wrapper for your model."""
    
    def __init__(self, model_type: str = "default"):
        """Initialize model.
        
        Args:
            model_type: Type/variant of the model
        """
        super().__init__(model_name=f"your_model_{model_type}")
        self.model = # Initialize your model
        
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
        
    def preprocess(self, rgb):
        """Preprocess input for model.
        
        Args:
            rgb: RGB input in any format (tensor CHW or HWC, numpy HWC)
            
        Returns:
            torch.Tensor: Preprocessed input in model format
        """
        # Handle different input formats
        # Normalize if needed
        return processed_rgb
        
    def forward(self, img):
        """Run inference.
        
        Args:
            img: RGB image in any format (tensor/numpy, CHW/HWC)
            
        Returns:
            torch.Tensor: Predicted depth map normalized to [0,1]
        """
        # 1. Preprocess input
        img = self.preprocess(img)
        
        # 2. Run inference
        depth = self.model(img)
        
        # 3. Normalize output to [0,1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
```

3. Create utilities:
```python
# utils.py
import numpy as np
import cv2
from PIL import Image

def process_image(image_path: str) -> np.ndarray:
    """Read and preprocess image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        np.ndarray: Preprocessed image in RGB format and range [0,1]
    """
    # 1. Load image
    # 2. Convert to RGB if needed
    # 3. Normalize to [0,1]
    return img

def colorize(depth: np.ndarray, **kwargs) -> np.ndarray:
    """Colorize depth map.
    
    Args:
        depth: Depth map in range [0,1]
        
    Returns:
        np.ndarray: Colored depth map
    """
    return colored_depth
```

4. Update model registry in `src/models/__init__.py`:
```python
MODEL_REGISTRY = {
    # ... existing models ...
    'your_model': {
        'module': 'src.models.your_model_wrapper',
        'class': 'YourModelWrapper',
        'variants': ['variant1', 'variant2']  # Your model variants
    }
}
```

## Adding a Custom Metric

1. Create a new metric in `src/metrics/your_metric.py`:
```python
from .base_metric import BaseMetric, register_metric
import torch

@register_metric('your_metric')
class YourMetric(BaseMetric):
    """Your custom metric."""
    
    def compute(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor) -> float:
        """Compute metric value.
        
        Args:
            pred: Predicted depth [B, 1, H, W]
            target: Ground truth depth [B, 1, H, W] 
            mask: Valid pixels mask [B, 1, H, W]
            
        Returns:
            float: Computed metric value
        """
        # 1. Apply mask
        # 2. Compute metric
        # 3. Handle edge cases
        return value
```

2. Update `src/metrics/__init__.py`:
```python
from .your_metric import YourMetric

__all__ = [
    # existing metrics
    'YourMetric'
]
```

## Testing Your Datasets

To examine dataset and check its statistics

```bash
# Test loading and statistics
python examine_dataset.py --dataset your_dataset --make-report
```

To test datasets building and functionality:

```bash
# Test integration
pytest -v tests/test_datasets.py
```

## Model Running Inference

Models are run using a wrapper script that handles virtual environment activation. Each model or group of models requires its specific environment from the `virt_envs/` directory.

### Basic Usage
```bash
# Using default environment
bash run_benchmark.sh

# Specify environment and model
bash run_benchmark.sh your_model --models metric3dv2 --variants vit_large

# Different environment with additional parameters
bash run_benchmark.sh leres --models leres --variants resnext101 --save-depth-maps
```

### Virtual Environments
Models are expected to have their environments in the `virt_envs/` directory:
```
virt_envs/
├── leres/                # For LeReS models
├── depth_anything/       # For Depth Anything models
└── ...
```

### Command Structure
```bash
bash run_benchmark.sh [venv_name] [script_arguments...]
```
- `venv_name`: Virtual environment name from virt_envs
- `script_arguments`: Any additional arguments passed to run_eval.py

### Examples
```bash
# Run with visualization output
bash run_benchmark.sh your_env --save-visualizations --num-vis-samples 10

# Use CPU device
bash run_benchmark.sh leres --device cpu --batch-size 4

# Help information
bash run_benchmark.sh --help
```

### Virtual Environment Organization

For each model virtual environment can be created via:

```bash
python -m venv env_name
```

So env is stored in `virt_envs/` dir.

## Software Requirements Examples

> **IMPORTANT**: The codebase has been tested with the following configuration:
- CUDA Version: 11.5

So for your personal setup the other verions of libraries like PyTorch may be required. 

Following requirements were created from original repositories instructions


### UniDepth + (LeRes, MiDas, DepthAnything, GenPercept)

```bash
python -m venv virt_envs/torch_models
source virt_envs/torch_models/bin/activate

git clone https://github.com/lpiccinelli-eth/UniDepth
pip install torch torchvision torchaudio
pip install -e .
pip install xformers==0.0.24
pip install imutils==0.5.4

pip install timm==0.9.5

pip install omegaconf==2.3.0
pip install diffusers==0.32.1
pip install transformers==4.47.1
pip install peft==0.14.0
```

This env can be reused for `leres`, `genpercept`, `midas`, `depth_anything` models.

## Command Line Arguments

### Dataset Arguments
- `--dataset-config` (str, default: 'src/datasets/nyu/nyu.yaml')
  - Path to dataset configuration file
  - Example: `--dataset-config src/datasets/nyu/nyu.yaml`

- `--batch-size` (int, default: 1)
  - Batch size for testing
  - Example: `--batch-size 4`

- `--split` (str, default: 'test')
  - Dataset split to test on
  - Values: 'train', 'val', 'test'
  - Example: `--split test`

### Model Arguments
- `--models` (list of str, default: ['leres'])
  - List of models to test
  - Example: `--models depth_anything leres`

- `--variants` (list of str)
  - Model variants for each specified model
  - Must match number of models
  - Variants per model:
    - leres: ['resnet50', 'resnext101']
    - depth_anything: ['vitb', 'vitl', 'vits']
  - Example: `--variants vitl resnet50`

- `--device` (str, default: 'cuda')
  - Device to run evaluation on
  - Values: 'cuda', 'cpu'
  - Example: `--device cuda`

### Output Arguments
- `--output-dir` (str, default: 'benchmark_results')
  - Directory to save results
  - Example: `--output-dir results/nyu_test`

- `--save-visualizations` (flag)
  - Save sample visualizations with error maps
  - Example: `--save-visualizations`

- `--save-depth-maps` (flag)
  - Save all raw depth maps paired with input images
  - Creates separate directories for RGB and depth maps
  - Example: `--save-depth-maps`

- `--num-vis-samples` (int, default: 5)
  - Number of visualization samples to save
  - Only used if --save-visualizations is set
  - Example: `--num-vis-samples 10`

## Output Structure
```
benchmark_results/
├── model_name_variant/
│   ├── depth_pairs/
│   │   ├── rgb/
│   │   │   ├── image_0001.png
│   │   │   └── ...
│   │   └── depth/
│   │       ├── image_0001.png
│   │       └── ...
│   └── visualizations/
│       ├── sample_0.png
│       └── ...
├── results.json
└── benchmark_YYYYMMDD_HHMMSS.log
```

# Environment Setup

## Isolated Environments

Currently, it's recommended to use isolated virtual environments for specific models or model groups, especially when some models share code bases or have similar requirements.

### Why Isolated Environments?

1. Different models may require different versions of the same dependencies
2. Some models share code bases and require specific framework versions
3. Prevents dependency conflicts between different models
4. Makes troubleshooting easier

## Model-Specific Requirements

For detailed requirements, refer to the original model repositories:

- **LeReS**: [Original Repository](https://github.com/aim-uofa/AdelaiDepth)
- **Depth Anything**: [Original Repository](https://github.com/LiheYoung/Depth-Anything)
- **MiDaS**: [Original Repository](https://github.com/isl-org/MiDaS)
- **Metric3D**: [Original Repository](https://github.com/yvanyin/metric3d)
- **UniDepth**: [Original Repository](https://github.com/lpiccinelli-eth/UniDepth)
- **Marigold**: [Original Repository](https://github.com/prs-eth/Marigold)
- **GenPercept**: [Original Repository](https://github.com/aim-uofa/GenPercept)
- **GeoWizard**: [Original Repository](https://github.com/fuxiao0719/GeoWizard)