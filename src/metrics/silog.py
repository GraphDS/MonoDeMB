from .base_metric import BaseMetric, register_metric
import torch


@register_metric("silog")
class ScaleInvariantLogError(BaseMetric):
    """Scale Invariant Logarithmic Error metric."""

    def compute(self, pred, target, mask):
        """
        Computes scale-invariant logarithmic error.
        pred, target, mask should be in shape [B, C, H, W]
        """
        mask = mask.to(torch.float)
        t_m = target * mask
        p_m = pred * mask
        diff_log = (torch.log10(p_m + 1e-10) - torch.log10(t_m + 1e-10)) * mask
        diff_log_sum = torch.sum(
            diff_log.reshape(diff_log.shape[0], diff_log.shape[1], -1), dim=2
        )
        diff_log_square = diff_log**2
        diff_log_square_sum = torch.sum(
            diff_log_square.reshape(
                diff_log_square.shape[0], diff_log_square.shape[1], -1
            ),
            dim=2,
        )
        num = torch.sum(mask.reshape(mask.shape[0], mask.shape[1], -1), dim=2)
        silog = torch.sqrt(
            diff_log_square_sum / (num + 1e-10) - (diff_log_sum / (num + 1e-10)) ** 2
        )
        valid_pics = torch.sum(num > 0)
        return torch.sum(silog).item()
