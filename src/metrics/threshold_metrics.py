from .base_metric import BaseMetric, register_metric
import torch


class ThresholdMetric(BaseMetric):
    """Base class for threshold-based metrics (delta1, delta2, delta3)."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute(self, pred, target, mask):
        """
        Computes threshold metric.
        pred, target, mask should be in shape [B, C, H, W]
        """
        mask = mask.to(torch.float)
        t_m = target * mask
        p_m = pred

        gt_pred = t_m / (p_m + 1e-10)
        pred_gt = p_m / (t_m + 1e-10)
        gt_pred = gt_pred.reshape(gt_pred.shape[0], gt_pred.shape[1], -1)
        pred_gt = pred_gt.reshape(pred_gt.shape[0], pred_gt.shape[1], -1)

        ratio_max = torch.max(torch.cat((gt_pred, pred_gt), dim=1), dim=1)[0]
        delta_sum = torch.sum(ratio_max < self.threshold, dim=1)
        num = torch.sum(mask.reshape(mask.shape[0], -1), dim=1)

        delta = delta_sum / (num + 1e-10)
        valid_pics = torch.sum(num > 0)
        return torch.sum(delta).item()


@register_metric("delta1")
class Delta1(ThresholdMetric):
    """Delta1 metric (threshold = 1.25)."""

    def __init__(self):
        super().__init__(1.25)


@register_metric("delta2")
class Delta2(ThresholdMetric):
    """Delta2 metric (threshold = 1.25²)."""

    def __init__(self):
        super().__init__(1.25**2)


@register_metric("delta3")
class Delta3(ThresholdMetric):
    """Delta3 metric (threshold = 1.25³)."""

    def __init__(self):
        super().__init__(1.25**3)
