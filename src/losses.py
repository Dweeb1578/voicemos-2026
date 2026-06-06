import torch
import torch.nn as nn
import torch.nn.functional as F


class MOSLoss(nn.Module):
    """ACR MSE (+ optional within-source rank term) + CCR pairwise ranking, NaN-masked.

    Args:
        ccr_lambda:     weight for the CCR pairwise-ranking term.
        acr_rank_alpha: weight for the ACR pairwise-ranking term (0 = pure MSE).
    """

    def __init__(self, ccr_lambda: float = 0.0, acr_rank_alpha: float = 0.0):
        super().__init__()
        self.ccr_lambda = ccr_lambda
        self.acr_rank_alpha = acr_rank_alpha

    @staticmethod
    def _pairwise_rank(pred, target, source_ids=None):
        """Mean margin-ranking loss over all pairs; masked to same-source pairs."""
        n = pred.size(0)
        if n < 2:
            return pred.sum() * 0.0
        i, j = torch.triu_indices(n, n, offset=1)
        keep = torch.sign(target[i] - target[j]) != 0
        if source_ids is not None:
            keep = keep & (source_ids[i] == source_ids[j])
        if not keep.any():
            return pred.sum() * 0.0
        target_sign = torch.sign(target[i] - target[j])[keep]
        return F.margin_ranking_loss(pred[i][keep], pred[j][keep], target_sign, margin=0.0)

    def forward(self, acr_pred, ccr_pred, acr_target, ccr_target, source_ids=None):
        acr_mask = ~torch.isnan(acr_target)
        if acr_mask.any():
            acr_loss = F.mse_loss(acr_pred[acr_mask], acr_target[acr_mask])
            if self.acr_rank_alpha > 0.0:
                src = source_ids[acr_mask] if source_ids is not None else None
                acr_loss = acr_loss + self.acr_rank_alpha * self._pairwise_rank(
                    acr_pred[acr_mask], acr_target[acr_mask], src
                )
        else:
            acr_loss = acr_pred.sum() * 0.0

        if self.ccr_lambda == 0.0:
            return acr_loss

        ccr_mask = ~torch.isnan(ccr_target)
        cp, ct = ccr_pred[ccr_mask], ccr_target[ccr_mask]
        ccr_loss = self._pairwise_rank(cp, ct)
        return acr_loss + self.ccr_lambda * ccr_loss
