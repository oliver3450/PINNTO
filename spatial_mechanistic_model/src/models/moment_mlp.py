import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentMLP(nn.Module):
    def __init__(self, num_target_genes: int, rnn_hidden_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()

        # Input: Time (1) + local spatial regulatory state (rnn_hidden_dim)
        # Without h_context, the network is forced to predict a single blurry
        # average trajectory for the entire tissue slice.
        self.input_dim = 1 + rnn_hidden_dim
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.SiLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        self.hidden_network = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, num_target_genes * 5)
        self.num_target_genes = num_target_genes

    def forward(self, t: torch.Tensor, h_context: torch.Tensor):
        """
        Args:
            t:         (..., 1)              — continuous pseudotime
            h_context: (..., rnn_hidden_dim) — local regulatory state from RNN

        Returns:
            Five moment tensors each with shape (..., num_target_genes)
        """
        x = torch.cat([t, h_context], dim=-1)

        hidden = self.hidden_network(x)
        raw_outputs = self.output_head(hidden)

        reshaped = raw_outputs.view(*raw_outputs.shape[:-1], 5, self.num_target_genes)

        nascent_mean = F.softplus(reshaped[..., 0, :])
        mature_mean  = F.softplus(reshaped[..., 1, :])
        nascent_var  = F.softplus(reshaped[..., 2, :])
        mature_var   = F.softplus(reshaped[..., 3, :])
        cov_nm       = reshaped[..., 4, :]  # unconstrained — can be negative

        return nascent_mean, mature_mean, nascent_var, mature_var, cov_nm
