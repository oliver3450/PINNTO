import torch
import torch.nn as nn

class PhysicallyConstrainedRNN(nn.Module):
    def __init__(self, input_spatial_dim: int, num_tfs: int):
        super().__init__()
        self.num_tfs = num_tfs
        
        # W_i2: Maps the spatial forcing signals (U_t) to the TF sensitivities
        self.input_layer = nn.Linear(input_spatial_dim, num_tfs, bias=True)
        
        # W_h2: The biological inertia (diagonal constraint). 
        # Initialized as a trainable 1D parameter vector, not a 2D matrix.
        self.raw_inertia_weights = nn.Parameter(torch.randn(num_tfs) * 0.1)
        
        self.relu = nn.ReLU()

    def get_retention_rates(self):
        # Enforces the thermodynamic boundary: 0 <= retention <= 1
        return torch.sigmoid(self.raw_inertia_weights)

    def forward(self, u_seq: torch.Tensor, h_0: torch.Tensor = None):
        """
        u_seq shape: (Batch, Sequence_Length, Input_Spatial_Dim)
        Returns the hidden state sequence (Y_F activity)
        """
        batch_size, seq_len, _ = u_seq.size()
        
        if h_0 is None:
            # Assume zero TF activity at t=0 if no root state is provided
            h_t = torch.zeros(batch_size, self.num_tfs, device=u_seq.device)
        else:
            h_t = h_0
            
        retention_rates = self.get_retention_rates()
        hidden_sequence = []
        
        # Unroll the RNN step-by-step
        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            
            # 1. External spatial forcing
            forcing = self.input_layer(u_t)
            
            # 2. Internal biological memory (element-wise multiplication enforces diagonal)
            memory = h_t * retention_rates
            
            # 3. Non-linear biological activation
            h_t = self.relu(forcing + memory)
            hidden_sequence.append(h_t.unsqueeze(1))
            
        # Shape: (Batch, Sequence_Length, Num_TFs)
        return torch.cat(hidden_sequence, dim=1)
