
"""
util.py - shared utilities for probe-based PCA+ridge steering.

This module implements:
- Probability clamping
- Mixing function (blend between model q and probe p_hat)
- Temperature solver via logit matching
- Fusion of multiple calibrated probe probabilities
- Debug helpers for logging
"""
import math
from typing import List, Dict

def clamp_prob(p: float, eps: float = 1e-4) -> float:
    """Clamp probability to [eps, 1-eps] for numerical stability."""
    return max(min(p, 1.0 - eps), eps)

def mix_prob(q: float, p_hat: float, lam: float = 0.3) -> float:
    """Convex combination of model prob q and probe prob p_hat."""
    return (1.0 - lam) * q + lam * p_hat

def safe_logit(p: float) -> float:
    """Logit with safety clamp inside this function."""
    p = clamp_prob(p)
    return math.log(p / (1.0 - p))

def temperature_for_target(q: float, q_star: float, alpha_min: float = 0.5, alpha_max: float = 2.0) -> float:
    """
    Compute temperature alpha to map top-1 probability q -> q_star
    using a one-vs-rest approximation:
        logit(q) = (1/alpha) * logit(q_star)
        alpha = logit(q) / logit(q_star)
    """
    num = safe_logit(q)
    den = safe_logit(q_star)
    # Avoid divide-by-zero if q_star ~ 0.5 (logit ~ 0). Clamp den.
    if abs(den) < 1e-6:
        return 1.0
    alpha = num / den
    # Clip to guardrails
    return max(min(alpha, alpha_max), alpha_min)

def ema(prev: float, new: float, beta: float = 0.8) -> float:
    """Exponential moving average."""
    return beta * prev + (1.0 - beta) * new

def fuse_logits_weighted_logit(probs: List[float], weights: List[float]) -> float:
    """
    Fuse calibrated probabilities via weighted average in logit space.
    weights should sum to 1.
    """
    assert len(probs) == len(weights), "probs and weights length mismatch"
    z = 0.0
    for p, w in zip(probs, weights):
        z += w * safe_logit(p)
    # back to prob
    return 1.0 / (1.0 + math.exp(-z))

def default_weights_from_scores(scores: List[float]) -> List[float]:
    """Normalize nonnegative scores into weights that sum to 1."""
    s = sum(max(0.0, x) for x in scores)
    if s <= 0:
        # fall back to uniform
        return [1.0 / len(scores)] * len(scores)
    return [max(0.0, x) / s for x in scores]

def pretty_pct(x: float) -> str:
    return f"{100.0*x:.1f}%"

def debug_bar(val: float, width: int = 20) -> str:
    """ASCII bar for a probability value [0,1]."""
    n = int(round(width * max(0.0, min(1.0, val))))
    return "[" + "#" * n + "-" * (width - n) + "]"

def explain_step(step:int, q:float, p_list:List[float], w_list:List[float], p_fused:float, q_star:float, alpha:float, alpha_ema:float) -> str:
    lines = []
    lines.append(f"[step {step:03d}] q(model)={q:.4f} {debug_bar(q)}")
    for i,(p,w) in enumerate(zip(p_list,w_list)):
        lines.append(f"  • layer_probe[{i}] p̂={p:.4f} w={w:.2f} {debug_bar(p)}")
    lines.append(f"  ⇒ p̂_fused={p_fused:.4f} {debug_bar(p_fused)}")
    lines.append(f"  ⇒ q* (target)={q_star:.4f}  α(raw)={alpha:.3f}  α(EMA)={alpha_ema:.3f}")
    return "\\n".join(lines)
