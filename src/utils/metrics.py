"""
metrics.py — Evaluation metrics for the Spatial Mechanistic PINN.

Provides three categories of metrics:
  1. Data fit:    Gene-level R² between predicted and empirical RNA moments
  2. Physics fit: CME residual statistics (how well the learned dynamics satisfy the ODEs)
  3. Fate fit:    Classification accuracy and calibration for terminal fate predictions
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
#  1. Data Fit Metrics
# ---------------------------------------------------------------------------

def gene_r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Per-gene R² (coefficient of determination).

    Args:
        pred:   (N, G) predicted moment values
        target: (N, G) empirical moment values

    Returns:
        (G,) R² scores. Values close to 1.0 = good fit, negative = worse than mean.
    """
    ss_res = np.sum((target - pred) ** 2, axis=0)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def moment_mse(pred_moments: Tuple[np.ndarray, ...],
               target_moments: Tuple[np.ndarray, ...]) -> Dict[str, float]:
    """
    Per-moment MSE averaged across all genes.

    Args:
        pred_moments:   tuple of 5 arrays (nascent_mean, mature_mean, nascent_var, mature_var, cov_nm)
        target_moments: same format, empirical values

    Returns:
        Dict with keys: nascent_mean_mse, mature_mean_mse, etc.
    """
    names = ["nascent_mean", "mature_mean", "nascent_var", "mature_var", "cov_nm"]
    results = {}
    for name, pred, target in zip(names, pred_moments, target_moments):
        results[f"{name}_mse"] = float(np.mean((pred - target) ** 2))
    results["total_mse"] = float(np.mean([results[k] for k in results]))
    return results


def moment_r2_summary(pred_moments: Tuple[np.ndarray, ...],
                      target_moments: Tuple[np.ndarray, ...]) -> Dict[str, float]:
    """
    Median R² across genes for each of the 5 moment types.
    """
    names = ["nascent_mean", "mature_mean", "nascent_var", "mature_var", "cov_nm"]
    results = {}
    for name, pred, target in zip(names, pred_moments, target_moments):
        r2 = gene_r2(pred, target)
        results[f"{name}_median_r2"] = float(np.median(r2))
        results[f"{name}_mean_r2"] = float(np.mean(r2))
        results[f"{name}_frac_positive"] = float(np.mean(r2 > 0))
    return results


# ---------------------------------------------------------------------------
#  2. Physics Fit Metrics
# ---------------------------------------------------------------------------

def physics_residual_stats(residuals: torch.Tensor) -> Dict[str, float]:
    """
    Summary statistics for CME residuals at collocation points.

    Args:
        residuals: scalar physics loss or (C,) per-collocation-point residuals

    Returns:
        Dict with mean, std, max, and percentiles of residual magnitudes.
    """
    if residuals.dim() == 0:
        return {"physics_loss": float(residuals.item())}

    r = residuals.abs().detach().cpu().numpy()
    return {
        "physics_mean": float(np.mean(r)),
        "physics_std": float(np.std(r)),
        "physics_max": float(np.max(r)),
        "physics_p50": float(np.percentile(r, 50)),
        "physics_p95": float(np.percentile(r, 95)),
        "physics_p99": float(np.percentile(r, 99)),
    }


def kinetic_rate_summary(beta: torch.Tensor, gamma: torch.Tensor) -> Dict[str, float]:
    """
    Summary of learned kinetic rate parameters.

    Args:
        beta:  (G,) splicing rates
        gamma: (G,) degradation rates

    Returns:
        Dict with mean, std, min, max for both beta and gamma,
        plus the mean beta/gamma ratio (RNA half-life proxy).
    """
    b = beta.detach().cpu().numpy()
    g = gamma.detach().cpu().numpy()
    return {
        "beta_mean": float(np.mean(b)),
        "beta_std": float(np.std(b)),
        "beta_min": float(np.min(b)),
        "beta_max": float(np.max(b)),
        "gamma_mean": float(np.mean(g)),
        "gamma_std": float(np.std(g)),
        "gamma_min": float(np.min(g)),
        "gamma_max": float(np.max(g)),
        "beta_gamma_ratio_mean": float(np.mean(b / (g + 1e-8))),
    }


# ---------------------------------------------------------------------------
#  3. Fate Prediction Metrics
# ---------------------------------------------------------------------------

def fate_accuracy(pred_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Classification metrics for fate prediction.

    Args:
        pred_logits: (N, F) raw logits
        targets:     (N, F) soft probability targets from Palantir

    Returns:
        Dict with hard accuracy, soft cross-entropy, and KL divergence.
    """
    pred_probs = F.softmax(pred_logits, dim=-1)
    log_probs = F.log_softmax(pred_logits, dim=-1)

    # Hard accuracy: does argmax match?
    pred_class = pred_probs.argmax(dim=-1)
    target_class = targets.argmax(dim=-1)
    hard_acc = float((pred_class == target_class).float().mean().item())

    # Soft cross-entropy: -sum(target * log_pred)
    soft_ce = float(-(targets * log_probs).sum(dim=-1).mean().item())

    # KL divergence: sum(target * log(target/pred))
    kl_div = float(F.kl_div(log_probs, targets, reduction="batchmean").item())

    # Calibration: how close are predicted probabilities to target probabilities?
    prob_mse = float(F.mse_loss(pred_probs, targets).item())

    return {
        "fate_hard_accuracy": hard_acc,
        "fate_soft_cross_entropy": soft_ce,
        "fate_kl_divergence": kl_div,
        "fate_probability_mse": prob_mse,
    }


# ---------------------------------------------------------------------------
#  Combined Evaluation
# ---------------------------------------------------------------------------

def full_evaluation(
    pred_moments: Tuple[np.ndarray, ...],
    target_moments: Tuple[np.ndarray, ...],
    physics_loss: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    fate_logits: torch.Tensor,
    fate_targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Run all evaluation metrics and return a combined dictionary.
    Useful for logging and checkpoint comparison.
    """
    results = {}
    results.update(moment_mse(pred_moments, target_moments))
    results.update(moment_r2_summary(pred_moments, target_moments))
    results.update(physics_residual_stats(physics_loss))
    results.update(kinetic_rate_summary(beta, gamma))
    results.update(fate_accuracy(fate_logits, fate_targets))
    return results
