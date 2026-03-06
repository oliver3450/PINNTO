import torch


def compute_time_derivatives(
    t: torch.Tensor,
    moment_prediction: torch.Tensor
) -> torch.Tensor:
    """
    Computes d(moment[b, c, g]) / d(t[b, c, 0]) for all beads, collocation points, and genes.

    t must be a (B, C, 1) leaf tensor created via .clone().requires_grad_(True) so that
    each (b, c) pair has an independent gradient node. This preserves the full 3D
    spatiotemporal geometry without collapsing the batch dimension.

    Diagonal Jacobian property: moment[b, c, g] depends only on t[b, c, 0], not on
    t[b', c', 0] for (b', c') != (b, c). So grad(moment[..., g].sum(), t)[b, c, 0]
    = d(moment[b, c, g]) / d(t[b, c, 0]) exactly.

    Args:
        t:                  (B, C, 1) — leaf tensor with requires_grad=True
        moment_prediction:  (B, C, G) — MomentMLP output per bead

    Returns:
        (B, C, G) — per-bead, per-collocation, per-gene time derivatives
    """
    G = moment_prediction.shape[-1]
    derivatives = []

    for g in range(G):
        grad_g = torch.autograd.grad(
            outputs=moment_prediction[..., g].sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, C, 1)
        derivatives.append(grad_g)

    return torch.cat(derivatives, dim=-1)  # (B, C, G)
