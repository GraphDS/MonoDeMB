from .base_metric import BaseMetric, register_metric
import torch


@register_metric("abs_rel")
class AbsRelError(BaseMetric):
    """Absolute Relative Error metric."""

    def compute(self, pred, target, mask):
        """
        Computes absolute relative error.
        pred, target, mask should be in shape [B, C, H, W]
        """
        mask = mask.to(torch.float)
        t_m = target * mask
        p_m = pred * mask

        rel = torch.abs(t_m - p_m) / (t_m + 1e-10)
        abs_rel_sum = torch.sum(rel.reshape(rel.shape[0], rel.shape[1], -1), dim=2)
        num = torch.sum(mask.reshape(mask.shape[0], mask.shape[1], -1), dim=2)

        abs_err = abs_rel_sum / (num + 1e-10)
        valid_pics = torch.sum(num > 0)
        return torch.sum(abs_err).item()
