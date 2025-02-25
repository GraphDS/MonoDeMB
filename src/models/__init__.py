"""Models package."""
import importlib
from typing import Tuple, Callable, Any
from .model_registry import MODEL_REGISTRY, MODEL_VARIANTS

def import_model_and_processor(model_name: str) -> Tuple[Any, Callable]:
    """Dynamically import model and its processor.
    
    Args:
        model_name: Name of the model to import
        
    Returns:
        tuple: (model_class, process_image_function)
        
    Raises:
        ValueError: If model_name is unknown
        ImportError: If module import fails
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_name}")
        
    model_info = MODEL_REGISTRY[model_name]
    
    try:
        module = importlib.import_module(model_info["module"])
        model_class = getattr(module, model_info["class"])
        process_fn = getattr(module, "process_image")
        return model_class, process_fn
        
    except ImportError as e:
        raise ImportError(f"Failed to import {model_name} model: {str(e)}")

