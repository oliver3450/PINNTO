import torch


def compute_time_derivatives(
    t: torch.Tensor,
    moment_prediction: torch.Tensor
) -> torch.Tensor:
    """
    Computes d(moment[c, g]) / d(t[c]) for all collocation points c and genes g.

    Exploits the diagonal Jacobian structure: moment[c, g] depends only on t[c],
    not on t[c'] for c' != c (MomentMLP processes each collocation point independently).

    By computing grad(moment[:, g].sum(), t), only t[c] contributes to moment[c, g],
    so the result[c, 0] = d(moment[c, g]) / d(t[c, 0]) exactly — no cross-point leakage.

    Args:
        t:                  (C, 1) — collocation times, must have requires_grad=True
        moment_prediction:  (C, G) — batch-mean MomentMLP output

    Returns:
        (C, G) — exact per-collocation, per-gene time derivatives
    """
    C, G = moment_prediction.shape
    derivatives = []

    for g in range(G):
        # grad of sum_c moment[c, g] w.r.t. t:
        #   result[c, 0] = d(moment[c, g]) / d(t[c, 0])   [diagonal Jacobian property]
        grad_g = torch.autograd.grad(
            outputs=moment_prediction[:, g].sum(),
            inputs=t,
            create_graph=True,   # must stay True: physics loss must flow gradients back
            retain_graph=True,
        )[0]  # (C, 1)
        derivatives.append(grad_g)

    return torch.cat(derivatives, dim=1)  # (C, G)
