from .base_metric import BaseMetric, register_metric
import torch


@register_metric("rmse")
class RMSEError(BaseMetric):
    """Root Mean Square Error metric."""

    def compute(self, pred, target, mask):
        """
        Computes RMSE.
        pred, target, mask should be in shape [B, C, H, W]
        """
        mask = mask.to(torch.float)
        t_m = target * mask
        p_m = pred * mask

        square = (t_m - p_m) ** 2
        rmse_sum = torch.sum(
            square.reshape(square.shape[0], square.shape[1], -1), dim=2
        )
        num = torch.sum(mask.reshape(mask.shape[0], mask.shape[1], -1), dim=2)

        rmse = torch.sqrt(rmse_sum / (num + 1e-10))
        valid_pics = torch.sum(num > 0)
        return torch.sum(rmse).item()
