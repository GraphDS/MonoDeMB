import torch
import numpy as np

METRIC_REGISTRY = {}


def register_metric(name):
    """Decorator to register a new metric class."""

    def decorator(cls):
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator


class BaseMetric:
    """Base class for all metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.val = np.longdouble(0.0)
        self.avg = np.longdouble(0.0)
        self.sum = np.longdouble(0.0)
        self.count = np.longdouble(0.0)

    def update(self, val, n=1):
        """Update metric with new value."""
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / (self.count + 1e-6)

    def compute(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Compute metric value for current batch."""
        raise NotImplementedError

    def get_value(self):
        """Get current metric value."""
        return self.avg


class MetricsManager:
    """Manager class to handle multiple metrics."""

    def __init__(self, metrics_list):
        """Initialize with list of metric names."""
        self.metrics = {}
        for metric_name in metrics_list:
            metric_class = METRIC_REGISTRY.get(metric_name)
            if metric_class is None:
                raise ValueError(f"Unknown metric: {metric_name}")
            self.metrics[metric_name] = metric_class()

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Update all metrics with new predictions."""
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute(pred, target, mask)
            metric.update(value)
            results[name] = value

        return results

    def get_metrics(self):
        """Get all current metric values with proper float conversion."""
        return {name: float(metric.get_value()) for name, metric in self.metrics.items()}



# Export the registry and manager
__all__ = ["BaseMetric", "MetricsManager", "METRIC_REGISTRY", "register_metric"]
