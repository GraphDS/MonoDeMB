from .base_dataset import BaseDataset, DATASET_REGISTRY
from .nyu import NYUDataset
from .mock_dataset import MockDataset
from .kitti import KITTIDataset
from .diode import DIODEDataset
from .eth3d import ETH3DDataset
from .synth2 import Synth2Dataset


def build_dataset(name: str, config_path: str, **kwargs):
    """Build a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
        
    return DATASET_REGISTRY[name](config_path, **kwargs)

__all__ = [
    'BaseDataset',
    'NYUDataset',
    'MockDataset',
    'build_dataset',
    'DATASET_REGISTRY',
    'KITTIDataset',
    'DIODEDataset',
    'ETH3DDataset',
    'Synth2Dataset'
]