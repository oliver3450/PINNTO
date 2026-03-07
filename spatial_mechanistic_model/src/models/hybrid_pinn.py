import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn_core import PhysicallyConstrainedRNN
from .moment_mlp import MomentMLP


def interpolate_to_collocation(discrete_values: torch.Tensor,
                               bin_times: torch.Tensor,
                               collocation_t: torch.Tensor) -> torch.Tensor:
    """
    Differentiable linear interpolation from discrete RNN bin outputs
    to arbitrary continuous collocation points.

    Args:
        discrete_values: (Batch, Seq_Len, D) — any per-bin quantity (a_t, b_t, or h_seq)
        bin_times:       (Seq_Len,) — the midpoint pseudotime of each bin
        collocation_t:   (Num_Collocation, 1) — random continuous time points

    Returns:
        (Batch, Num_Collocation, D) — interpolated values at each t
    """
    t_flat = collocation_t.squeeze(-1)                    # (C,)
    t_bins = bin_times                                     # (S,)

    t_clamped = t_flat.clamp(t_bins[0], t_bins[-1])

    idx_right = torch.searchsorted(t_bins, t_clamped).clamp(1, len(t_bins) - 1)
    idx_left = idx_right - 1

    t_left = t_bins[idx_left]                              # (C,)
    t_right = t_bins[idx_right]                            # (C,)

    alpha = ((t_clamped - t_left) / (t_right - t_left + 1e-8)).unsqueeze(0).unsqueeze(-1)  # (1, C, 1)

    val_left = discrete_values[:, idx_left, :]
    val_right = discrete_values[:, idx_right, :]

    return val_left + alpha * (val_right - val_left)


class SpatialMechanisticModel(nn.Module):
    def __init__(self,
                 input_spatial_dim: int,
                 num_tfs: int,
                 num_target_genes: int,
                 num_terminal_fates: int,
                 frozen_grn_matrix: torch.Tensor,
                 dt: float = 0.02,
                 moment_hidden_dim: int = 128,
                 moment_num_layers: int = 4):
        super().__init__()

        self.num_target_genes = num_target_genes
        self.num_tfs = num_tfs
        self.dt = dt

        # 1. The Upstream Memory Engine (RNN)
        self.rnn = PhysicallyConstrainedRNN(input_spatial_dim, num_tfs)

        # 2. The Frozen Biological Knowledge (W_TFTG for burst frequency)
        self.register_buffer('frozen_grn', frozen_grn_matrix)

        # 3. Separate trainable projection for burst size (W_Size)
        self.W_size = nn.Linear(num_tfs, num_target_genes, bias=False)

        # 4. The Continuous PINN Branch (MomentMLP)
        #    Conditioned on h_context so each bead gets its own RNA trajectory.
        self.moment_mlp = MomentMLP(num_target_genes, rnn_hidden_dim=num_tfs,
                                    hidden_dim=moment_hidden_dim,
                                    num_layers=moment_num_layers)

        # 5. Kinetic rate parameters: beta (splicing) and gamma (degradation)
        self.raw_beta = nn.Parameter(torch.randn(num_target_genes) * 0.1)
        self.raw_gamma = nn.Parameter(torch.randn(num_target_genes) * 0.1)

        # 6. The Topological Fate Head (Classification)
        self.fate_head = nn.Linear(num_tfs, num_terminal_fates)

    @property
    def beta(self):
        return F.softplus(self.raw_beta)

    @property
    def gamma(self):
        return F.softplus(self.raw_gamma)

    def forward(self, u_seq: torch.Tensor, collocation_t: torch.Tensor = None):
        """
        Full forward pass for all three branches.

        Args:
            u_seq:          (Batch, Seq_Len, Input_Spatial_Dim) — spatial forcing
            collocation_t:  (Num_Collocation, 1) — random times for physics loss
                            If None, only returns RNN outputs (no physics branch)

        Returns dict with:
            burst_freq:      (Batch, Seq_Len, Num_Genes)
            burst_size:      (Batch, Seq_Len, Num_Genes)
            fate_logits:     (Batch, Seq_Len, Num_Fates)
            hidden_tfs:      (Batch, Seq_Len, Num_TFs)
            moments:         tuple of 5 tensors (Batch, C, Num_Genes) if collocation_t given
            burst_freq_cont: (Batch, C, Num_Genes) if collocation_t given
            burst_size_cont: (Batch, C, Num_Genes) if collocation_t given
        """
        batch_size, seq_len, _ = u_seq.size()

        # --- PHASE 1: Recurrent Biological Memory ---
        h_seq = self.rnn(u_seq)  # (Batch, Seq_Len, Num_TFs)

        # --- PHASE 2: Branch A — Burst Parameters ---
        a_t = F.softplus(torch.matmul(h_seq, self.frozen_grn))   # (B, S, G)
        b_t = F.softplus(self.W_size(h_seq))                      # (B, S, G)

        # --- PHASE 3: Branch B — Fate Classification ---
        fate_logits = self.fate_head(h_seq)  # (B, S, Num_Fates)

        result = {
            "burst_freq": a_t,
            "burst_size": b_t,
            "fate_logits": fate_logits,
            "hidden_tfs": h_seq,
        }

        # --- PHASE 4: Branch C — Continuous PINN ---
        if collocation_t is not None:
            bin_times = torch.linspace(
                self.dt / 2, 1.0 - self.dt / 2, seq_len,
                device=u_seq.device
            )

            # Interpolate discrete burst params to continuous collocation points
            a_cont = interpolate_to_collocation(a_t, bin_times, collocation_t)    # (B, C, G)
            b_cont = interpolate_to_collocation(b_t, bin_times, collocation_t)    # (B, C, G)

            # Interpolate RNN hidden state to collocation points — this is the
            # local regulatory context that conditions each bead's RNA trajectory.
            h_cont = interpolate_to_collocation(h_seq, bin_times, collocation_t)  # (B, C, num_tfs)

            # h_cont is a regular tensor — safe to gather across DataParallel replicas.
            # t_expanded and moments are NOT stored here: DataParallel's gather breaks
            # the computation graph between per-replica leaf t_expanded and the gathered
            # moment tensors. Instead, compute_physics_loss rebuilds t_expanded and
            # re-runs moment_mlp after gathering, keeping the graph intact.
            result["burst_freq_cont"] = a_cont
            result["burst_size_cont"] = b_cont
            result["h_cont"] = h_cont

        return result
