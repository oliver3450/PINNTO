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

        # Spatial 2D coordinates — Open-ST uses 'X_xy_loc'/'xy_loc', Scanpy convention is 'spatial'
        SPATIAL_KEYS = ["spatial", "X_xy_loc", "xy_loc", "X_spatial", "coordinates"]
        spatial_key = next((k for k in SPATIAL_KEYS if k in self.adata.obsm), None)
        if spatial_key is None:
            raise KeyError(
                f"No spatial coordinates found. Tried: {SPATIAL_KEYS}. "
                f"Available: {list(self.adata.obsm.keys())}"
            )
        self.coords = self.adata.obsm[spatial_key]

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

        # Compute per-moment scaling factors from a random bead sample.
        # Used in compute_data_loss to give all 5 moment dimensions equal MSE weight.
        self.moment_scales = self._compute_moment_scales(n_sample=2000)
        print(f"Moment scales (std per dimension): {self.moment_scales.tolist()}")

        # Fate targets: use Palantir outputs if available, otherwise random placeholders
        if "palantir_fate_probs" in self.adata.obsm:
            # Real fate probabilities from Palantir: (n_cells, n_fates)
            cell_fates = self.adata.obsm["palantir_fate_probs"]
            if hasattr(cell_fates, "values"):
                cell_fates = cell_fates.values  # DataFrame -> numpy
            self.num_fates = cell_fates.shape[1]
            # Each cell's fate is broadcast to all sequence positions —
            # the KNN neighbors inherit the center cell's fate target
            self.fate_targets = np.tile(
                cell_fates[:, np.newaxis, :], (1, seq_len, 1)
            )  # (n_cells, seq_len, n_fates)
            print(f"Using Palantir fate probabilities: {self.num_fates} terminal states")
        else:
            raise KeyError(
                "FATAL: 'palantir_fate_probs' not found in adata.obsm. "
                "Do not train on random noise. Run scripts/00_master_preprocess.py first "
                "to compute real fate probabilities via Palantir."
            )

    def _compute_moment_scales(self, n_sample: int = 2000) -> torch.Tensor:
        """
        Estimate the population std of each of the 5 empirical moment dimensions
        from a random sample of beads.  Returns a (5,) float32 tensor:
            [nanstd(nascent_mean), nanstd(mature_mean),
             nanstd(nascent_var),  nanstd(mature_var),  nanstd(cov_nm)]

        Used by compute_data_loss to normalise each moment dimension so all 5
        contribute equally to L_data regardless of raw count scale.

        Means use all n_sample beads (just raw count arrays — no KNN needed).
        Variances use a KNN window subsample (1000 beads) to estimate spread.
        cov_nm is computed empirically: Cov(N,M) = E[NM] - E[N]E[M] per gene per window.
        """
        n = min(n_sample, self.adata.n_obs)
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(self.adata.n_obs, n, replace=False)

        def safe_std(arr: np.ndarray) -> float:
            """Nanstd across all elements; falls back to 1.0 if degenerate or all-NaN."""
            s = float(np.nanstd(arr))
            return s if s > 1e-6 else 1.0

        # --- Means: raw count arrays, shape (n, G) ---
        N_sample = self.N[idx].ravel()   # unspliced (nascent) counts
        M_sample = self.M[idx].ravel()   # spliced (mature) counts

        # --- Variances + Covariance: need KNN windows, use a smaller subsample ---
        n_var_sample = min(1000, n)
        var_idx = idx[:n_var_sample]

        n_var_vals, m_var_vals, cov_vals = [], [], []
        for i in var_idx:
            N_win = self.N[self.neighbors[i]]   # (seq_len, G)
            M_win = self.M[self.neighbors[i]]   # (seq_len, G)
            n_var_vals.append(np.var(N_win, axis=0))
            m_var_vals.append(np.var(M_win, axis=0))
            # Cov(N,M) = E[NM] - E[N]E[M]: NaN where window is all-zero for that gene
            cov_per_gene = (
                np.mean(N_win * M_win, axis=0)
                - np.mean(N_win, axis=0) * np.mean(M_win, axis=0)
            )
            cov_vals.append(cov_per_gene)

        scales = torch.tensor([
            safe_std(N_sample),                         # nascent_mean
            safe_std(M_sample),                         # mature_mean
            safe_std(np.concatenate(n_var_vals)),       # nascent_var
            safe_std(np.concatenate(m_var_vals)),       # mature_var
            safe_std(np.concatenate(cov_vals)),         # cov_nm — now computed, not hardcoded
        ], dtype=torch.float32)

        return scales

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

        # Empirical covariance: Cov(N, M) = E[N·M] − E[N]·E[M] per gene across window.
        # Genes where all counts are 0 in this window yield NaN → replace with 0.
        cov_gene = (
            np.mean(N_seq * M_seq, axis=0)
            - np.mean(N_seq, axis=0) * np.mean(M_seq, axis=0)
        )  # (G,)
        cov_gene = np.nan_to_num(cov_gene, nan=0.0)
        # Broadcast to (S, G) — same convention as the variance terms
        cov_nm = torch.tensor(
            np.broadcast_to(cov_gene[np.newaxis, :], (self.seq_len, self.num_genes)).copy(),
            dtype=torch.float32,
        )

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
    drop_last: bool = True,
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
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
