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
        discrete_values: (Batch, Seq_Len, Num_Genes) — a_t or b_t from RNN
        bin_times:       (Seq_Len,) — the midpoint pseudotime of each bin
        collocation_t:   (Num_Collocation, 1) — random continuous time points

    Returns:
        (Batch, Num_Collocation, Num_Genes) — interpolated values at each t
    """
    t_flat = collocation_t.squeeze(-1)                    # (C,)
    t_bins = bin_times                                     # (S,)

    # Clamp collocation times to the bin range
    t_clamped = t_flat.clamp(t_bins[0], t_bins[-1])

    # Find the left bin index for each collocation point
    # searchsorted gives the insertion point; subtract 1 for left neighbor
    idx_right = torch.searchsorted(t_bins, t_clamped).clamp(1, len(t_bins) - 1)
    idx_left = idx_right - 1

    t_left = t_bins[idx_left]                              # (C,)
    t_right = t_bins[idx_right]                            # (C,)

    # Interpolation weight: how far between left and right
    alpha = ((t_clamped - t_left) / (t_right - t_left + 1e-8)).unsqueeze(0).unsqueeze(-1)  # (1, C, 1)

    # Gather left and right values: (Batch, C, Genes)
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
        self.dt = dt

        # 1. The Upstream Memory Engine (RNN)
        self.rnn = PhysicallyConstrainedRNN(input_spatial_dim, num_tfs)

        # 2. The Frozen Biological Knowledge (W_TFTG for burst frequency)
        self.register_buffer('frozen_grn', frozen_grn_matrix)

        # 3. Separate trainable projection for burst size (W_Size)
        #    Independent from W_TFTG so a_t and b_t can vary independently
        self.W_size = nn.Linear(num_tfs, num_target_genes, bias=False)

        # 4. The Continuous PINN Branch (MomentMLP)
        self.moment_mlp = MomentMLP(num_target_genes, moment_hidden_dim, moment_num_layers)

        # 5. Kinetic rate parameters: beta (splicing) and gamma (degradation)
        #    Per-gene, stored as raw values and passed through Softplus to ensure > 0
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
            burst_freq:     (Batch, Seq_Len, Num_Genes) — discrete a_t
            burst_size:     (Batch, Seq_Len, Num_Genes) — discrete b_t
            fate_logits:    (Batch, Seq_Len, Num_Fates) — raw logits (no softmax!)
            hidden_tfs:     (Batch, Seq_Len, Num_TFs)
            moments:        tuple of 5 tensors if collocation_t provided
            burst_freq_cont: (Batch, Num_Collocation, Num_Genes) if collocation_t provided
            burst_size_cont: (Batch, Num_Collocation, Num_Genes) if collocation_t provided
        """
        batch_size, seq_len, _ = u_seq.size()

        # --- PHASE 1: Recurrent Biological Memory ---
        h_seq = self.rnn(u_seq)  # (Batch, Seq_Len, Num_TFs)

        # --- PHASE 2: Branch A — Burst Parameters (two independent projections) ---
        a_t = F.softplus(torch.matmul(h_seq, self.frozen_grn))   # (B, S, G)
        b_t = F.softplus(self.W_size(h_seq))                      # (B, S, G)

        # --- PHASE 3: Branch B — Fate Classification (raw logits, NO softmax) ---
        fate_logits = self.fate_head(h_seq)  # (B, S, Num_Fates)

        result = {
            "burst_freq": a_t,
            "burst_size": b_t,
            "fate_logits": fate_logits,
            "hidden_tfs": h_seq,
        }

        # --- PHASE 4: Branch C — Continuous PINN (only if collocation points given) ---
        if collocation_t is not None:
            # Bin midpoint times for interpolation
            bin_times = torch.linspace(
                self.dt / 2, 1.0 - self.dt / 2, seq_len,
                device=u_seq.device
            )

            # Interpolate discrete burst params to continuous collocation points
            a_cont = interpolate_to_collocation(a_t, bin_times, collocation_t)
            b_cont = interpolate_to_collocation(b_t, bin_times, collocation_t)

            # MomentMLP predicts 5 RNA moments at each collocation time
            moments = self.moment_mlp(collocation_t)

            result["burst_freq_cont"] = a_cont
            result["burst_size_cont"] = b_cont
            result["moments"] = moments  # (nascent_mean, mature_mean, nascent_var, mature_var, cov_nm)

        return result
