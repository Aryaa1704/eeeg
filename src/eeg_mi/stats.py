from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, binomtest

def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    b_only = int(np.sum(~a_correct & b_correct)); a_only = int(np.sum(a_correct & ~b_correct))
    n = a_only + b_only
    p = float(binomtest(min(a_only, b_only), n, 0.5).pvalue) if n else 1.0
    return {"test": "exact McNemar", "statistic": min(a_only, b_only), "p_value": p, "discordant_a_only": a_only, "discordant_b_only": b_only}

def group_wilcoxon(deltas: np.ndarray) -> dict:
    x = np.asarray(deltas, float)
    if len(x) < 2 or np.allclose(x, 0): return {"test": "Wilcoxon signed-rank", "statistic": 0., "p_value": 1., "n": len(x), "effect_size_r": 0.}
    r = wilcoxon(x, alternative="two-sided", method="auto")
    return {"test": "Wilcoxon signed-rank", "statistic": float(r.statistic), "p_value": float(r.pvalue), "n": len(x), "effect_size_r": None}

def holm(p_values: list[float]) -> list[float]:
    n=len(p_values); order=np.argsort(p_values); out=np.empty(n); running=0.
    for rank, idx in enumerate(order):
        running=max(running, (n-rank)*p_values[idx]); out[idx]=min(1., running)
    return out.tolist()
