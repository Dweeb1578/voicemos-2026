import torch
import torch.nn as nn
import torch.nn.functional as F


class MOSLoss(nn.Module):
    """ACR MSE loss + CCR pairwise ranking loss with NaN masking.

    Args:
        ccr_lambda: weight for CCR term (0 during pretraining, ramps to 1 during finetuning)
    """

    def __init__(self, ccr_lambda: float = 0.0):
        super().__init__()
        self.ccr_lambda = ccr_lambda

    def forward(
        self,
        acr_pred: torch.Tensor,
        ccr_pred: torch.Tensor,
        acr_target: torch.Tensor,
        ccr_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            acr_pred:   (B,)
            ccr_pred:   (B,)
            acr_target: (B,) may contain NaN
            ccr_target: (B,) may contain NaN
        """
        acr_mask = ~torch.isnan(acr_target)
        if acr_mask.any():
            acr_loss = F.mse_loss(acr_pred[acr_mask], acr_target[acr_mask])
        else:
            acr_loss = acr_pred.sum() * 0.0  # differentiable zero

        if self.ccr_lambda == 0.0:
            return acr_loss

        ccr_mask = ~torch.isnan(ccr_target)
        cp, ct = ccr_pred[ccr_mask], ccr_target[ccr_mask]

        if cp.size(0) < 2:
            return acr_loss

        i, j = torch.triu_indices(cp.size(0), cp.size(0), offset=1)
        target_sign = torch.sign(ct[i] - ct[j])
        nonzero = target_sign != 0

        if nonzero.any():
            ccr_loss = F.margin_ranking_loss(
                cp[i][nonzero], cp[j][nonzero], target_sign[nonzero], margin=0.0,
            )
        else:
            ccr_loss = cp.sum() * 0.0

        return acr_loss + self.ccr_lambda * ccr_loss
