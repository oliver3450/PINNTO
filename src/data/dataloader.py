"""
dataloader.py — KNN-based spatial sequence dataset for the Spatial Mechanistic PINN.

Builds ordered sequences of beads using spatial nearest neighbors,
extracts spliced/unspliced RNA layers as empirical moment targets,
and returns batches ready for the three-loss training loop.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import scanpy as sc


class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, h5ad_path: str, seq_len: int = 50, num_genes: int = 200, num_fates: int = 2):
        """
        Args:
            h5ad_path:  Path to the processed .h5ad file with 'spliced'/'unspliced' layers
                        and adata.obsm['spatial'] coordinates.
            seq_len:    Number of spatial neighbors to include in each sequence (= S).
            num_genes:  Number of target genes to use (first `num_genes` columns).
            num_fates:  Number of terminal fates (for placeholder fate targets).
        """
        self.adata = sc.read_h5ad(h5ad_path)
        self.seq_len = seq_len
        self.num_genes = num_genes
        self.num_fates = num_fates

        # Spatial 2D coordinates from Spacemake/Open-ST
        self.coords = self.adata.obsm['spatial']

        # Spliced = mature mRNA (M), Unspliced = nascent pre-mRNA (N)
        self.M = self.adata.layers['spliced'][:, :num_genes]
        self.N = self.adata.layers['unspliced'][:, :num_genes]

        # Convert sparse to dense if needed
        if hasattr(self.M, "toarray"):
            self.M = self.M.toarray()
            self.N = self.N.toarray()

        # Precompute KNN graph: for each bead, find seq_len spatial neighbors
        self.nn = NearestNeighbors(n_neighbors=seq_len).fit(self.coords)
        _, self.neighbors = self.nn.kneighbors(self.coords)

        # Placeholder fate targets: uniform random Dirichlet-like soft probabilities
        # Replace with Palantir outputs from 01_run_palantir.py once available
        raw_fates = np.random.rand(self.adata.n_obs, seq_len, num_fates)
        self.fate_targets = raw_fates / raw_fates.sum(axis=-1, keepdims=True)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # Spatial neighbor indices form the sequence (ordered by proximity)
        seq_indices = self.neighbors[idx]  # (S,)

        # u_seq: 2D spatial coordinates of each neighbor bead — (S, 2)
        u_seq = torch.tensor(self.coords[seq_indices], dtype=torch.float32)

        # RNA count arrays for this spatial sequence
        N_seq = self.N[seq_indices]  # (S, G) — nascent/unspliced
        M_seq = self.M[seq_indices]  # (S, G) — mature/spliced

        # First moments: per-bead counts (mean over single bead = itself)
        nascent_mean = torch.tensor(N_seq, dtype=torch.float32)
        mature_mean  = torch.tensor(M_seq, dtype=torch.float32)

        # Second moments: variance computed across the neighbor window,
        # broadcast to (S, G) so each step carries the same variance estimate
        n_var = np.var(N_seq, axis=0, keepdims=True).repeat(self.seq_len, axis=0)
        m_var = np.var(M_seq, axis=0, keepdims=True).repeat(self.seq_len, axis=0)
        nascent_var = torch.tensor(n_var, dtype=torch.float32)
        mature_var  = torch.tensor(m_var, dtype=torch.float32)

        # Covariance placeholder: zero-initialized (no ground truth available)
        cov_nm = torch.zeros_like(nascent_mean)

        # Fate targets: (S, Num_Fates) soft probability vectors
        fate = torch.tensor(self.fate_targets[idx], dtype=torch.float32)

        return {
            "u_seq": u_seq,
            "empirical_moments": (nascent_mean, mature_mean, nascent_var, mature_var, cov_nm),
            "fate_targets": fate,
        }


def get_dataloader(
    h5ad_path: str,
    batch_size: int = 16,
    seq_len: int = 50,
    num_genes: int = 200,
    num_fates: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convenience function to build the DataLoader for training.

    Args:
        h5ad_path:  Path to processed .h5ad with spliced/unspliced layers.
        batch_size: Training batch size.
        seq_len:    Spatial KNN sequence length (must match config dt: 1/seq_len).
        num_genes:  Number of target genes (must match model num_target_genes).
        num_fates:  Number of terminal fates (must match model num_terminal_fates).
        shuffle:    Whether to shuffle between epochs.

    Returns:
        DataLoader yielding dicts with keys: u_seq, empirical_moments, fate_targets.
    """
    dataset = SpatialTranscriptomicsDataset(h5ad_path, seq_len, num_genes, num_fates)

    def collate_fn(batch):
        return {
            "u_seq": torch.stack([b["u_seq"] for b in batch]),
            "empirical_moments": tuple(
                torch.stack([b["empirical_moments"][i] for b in batch])
                for i in range(5)
            ),
            "fate_targets": torch.stack([b["fate_targets"] for b in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
    )
