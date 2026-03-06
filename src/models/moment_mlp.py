import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentMLP(nn.Module):
    def __init__(self, num_target_genes: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        
        # The input is always a single continuous scalar: Time (t)
        layers = [nn.Linear(1, hidden_dim), nn.SiLU()]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        self.hidden_network = nn.Sequential(*layers)
        
        # The output maps to the 5 biological moments for every target gene
        self.output_head = nn.Linear(hidden_dim, num_target_genes * 5)
        self.num_target_genes = num_target_genes

    def forward(self, t: torch.Tensor):
        """
        t shape: (Batch_Collocation_Points, 1)
        """
        hidden = self.hidden_network(t)
        raw_outputs = self.output_head(hidden)
        
        # Reshape to explicitly separate the 5 moments
        # Shape: (Batch, 5, Num_Target_Genes)
        reshaped = raw_outputs.view(-1, 5, self.num_target_genes)

        # Means and variances MUST be strictly positive (RNA counts >= 0).
        # Covariance Cov(N,M) can be negative in oscillatory or feedback systems,
        # so it is left unconstrained.
        nascent_mean = F.softplus(reshaped[:, 0, :])
        mature_mean  = F.softplus(reshaped[:, 1, :])
        nascent_var  = F.softplus(reshaped[:, 2, :])
        mature_var   = F.softplus(reshaped[:, 3, :])
        cov_nm       = reshaped[:, 4, :]  # unconstrained — can be negative

        return nascent_mean, mature_mean, nascent_var, mature_var, cov_nm
