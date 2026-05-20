from typing import List

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(predictions: List[float], targets: List[float]) -> dict:
    """Compute UTT-SRCC, UTT-LCC, and MSE. NaN pairs are dropped before computation.

    Returns:
        dict with keys: srcc, lcc, mse
    """
    p = np.array(predictions, dtype=np.float64)
    t = np.array(targets, dtype=np.float64)

    valid = ~(np.isnan(p) | np.isnan(t))
    p, t = p[valid], t[valid]

    if len(p) < 2:
        return {"srcc": float("nan"), "lcc": float("nan"), "mse": float("nan")}

    srcc, _ = spearmanr(p, t)
    lcc, _ = pearsonr(p, t)
    mse = float(np.mean((p - t) ** 2))

    return {"srcc": float(srcc), "lcc": float(lcc), "mse": mse}
