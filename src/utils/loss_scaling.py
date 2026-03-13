"""
loss_scaling.py — Dynamic loss normalization for the Spatial Mechanistic PINN.

Implements Initial Loss Normalization (ILN): records the unweighted value of
each loss term at the very first forward pass, then divides all subsequent
losses by their respective initial values before applying the λ hyperparameters.

This maps all three losses to ~1.0 at step 1, so the λ values in
loss_weights.yaml become true relative weights rather than scale-dependent
tuning parameters.

Usage in 03_train_model.py (once confirmed):

    balancer = LossBalancer()

    # Inside train_one_epoch, replace:
    loss = lambda_data * l_data + lambda_phys * l_phys + lambda_fate * l_fate

    # With:
    loss = balancer.scale(
        losses={"data": l_data, "phys": l_phys, "fate": l_fate},
        lambdas={"data": lambda_data, "phys": lambda_phys, "fate": lambda_fate},
    )
"""

import torch
from typing import Dict


class LossBalancer:
    """
    Initial Loss Normalization for multi-objective PINN training.

    Records each loss's magnitude on the first forward pass and uses those
    as static normalisation constants on all subsequent passes.  The initial
    values are stored as plain Python floats — NOT tensors — so they carry
    no gradient and do not appear in the computation graph.

    Thread-safety: When used with DataParallel, all loss values arrive at
    this class *after* DataParallel gather, so there is no multi-GPU
    synchronisation concern.

    Attributes:
        _initial (Dict[str, float]): Initial loss magnitudes (set once, static thereafter).
        _initialized (bool): True after the first call to `scale()`.
        _step (int): Total number of calls to `scale()` for logging.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Minimum denominator to prevent division by zero if a loss
                 happens to be exactly 0 on the first step.
        """
        self._initial: Dict[str, float] = {}
        self._initialized: bool = False
        self._step: int = 0
        self._eps = eps

    def scale(
        self,
        losses: Dict[str, torch.Tensor],
        lambdas: Dict[str, float],
    ) -> torch.Tensor:
        """
        Compute the normalized, weighted total loss.

        On the first call: records each loss value as a static float constant.
        On all subsequent calls: divides each loss by its recorded initial value,
        then applies the user-defined λ weights.

        L_total = Σ_k  λ_k * (L_k / L_k^{(0)})

        Args:
            losses:  Dict mapping loss name → scalar loss tensor (with grad).
                     Keys must be consistent across calls.
            lambdas: Dict mapping loss name → float weight (λ).
                     Must have the same keys as `losses`.

        Returns:
            Scalar total loss tensor, ready for .backward().
        """
        if not self._initialized:
            # Detach and convert to float — permanently escapes the computation graph.
            for k, v in losses.items():
                initial_val = v.detach().item()
                self._initial[k] = max(abs(initial_val), self._eps)
            self._initialized = True

        total = sum(
            lambdas[k] * (v / self._initial[k])
            for k, v in losses.items()
        )
        self._step += 1
        return total

    def report(self) -> Dict[str, float]:
        """Return the recorded initial loss values for logging."""
        return dict(self._initial)

    def effective_lambda(self, lambdas: Dict[str, float]) -> Dict[str, float]:
        """
        Return the effective per-loss weight after normalization.

        Useful for logging: effective_lambda[k] = lambda_k / L_k^{(0)}
        This tells you the actual gradient scale applied to each loss.
        """
        if not self._initialized:
            return {k: v for k, v in lambdas.items()}
        return {k: lambdas[k] / self._initial[k] for k in lambdas}

    def reset(self):
        """Reset balancer — useful if resuming from a checkpoint and rebalancing."""
        self._initial = {}
        self._initialized = False
        self._step = 0
