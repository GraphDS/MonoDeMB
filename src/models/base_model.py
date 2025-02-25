from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Abstract base class for all depth estimation models."""
    
    def __init__(self, model_name: str):
        """Initialize model.
        
        Args:
            model_name: Name of the specific model instance
        """
        super().__init__()
        self._model_name = model_name
        
    @property
    def name(self) -> str:
        """Get model name."""
        return self._model_name
        
    @abstractmethod
    def to(self, device):
        """Move model to device.
        
        Args:
            device: Device to move model to
            
        Returns:
            Model instance on specified device
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Run inference on input.
        
        Args:
            x: Input tensor
            
        Returns:
            Model prediction
        """
        pass