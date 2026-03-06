import torch


def compute_time_derivatives(
    t: torch.Tensor,
    moment_prediction: torch.Tensor
) -> torch.Tensor:
    """
    Computes d(moment[b, c, g]) / d(t[b, c, 0]) for all beads, collocation points, and genes
    in a single vectorized autograd call via is_grads_batched=True.

    A diagonal eye mask of shape (G, B, C, G) is used as batched grad_outputs:
    each of the G "batches" is a one-hot selector that isolates gene g, so PyTorch
    computes G independent VJPs simultaneously rather than sequentially.

    Args:
        t:                  (B, C, 1) — leaf tensor with requires_grad=True
        moment_prediction:  (B, C, G) — MomentMLP output per bead

    Returns:
        (B, C, G) — per-bead, per-collocation, per-gene time derivatives
    """
    B, C, G = moment_prediction.shape

    # Diagonal mask: grad_outputs[g] selects only gene g across all (B, C)
    grad_outputs = torch.eye(G, device=t.device).view(G, 1, 1, G).expand(G, B, C, G)

    grads = torch.autograd.grad(
        outputs=moment_prediction,
        inputs=t,
        grad_outputs=grad_outputs,
        is_grads_batched=True,
        create_graph=True,
        retain_graph=True,
    )[0]  # (G, B, C, 1)

    return grads.squeeze(-1).permute(1, 2, 0)  # (B, C, G)
