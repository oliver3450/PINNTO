import pandas as pd
import numpy as np
import torch
from typing import List, Tuple

def build_frozen_grn_matrix(
    tftg_path: str,
    expressed_tfs: List[str],
    expressed_target_genes: List[str],
    return_names: bool = False,
) -> torch.Tensor:
    """
    Builds the W_TFTG matrix for the PINN's output projection layer.
    Only includes genes that are physically present in your spatial dataset.
    """
    print(f"Loading TF-TG database from {tftg_path}...")
    df = pd.read_csv(tftg_path)

    # Normalise to uppercase for case-insensitive matching.
    # TFTGDB uses human-style ALL-CAPS (e.g. "AATF").
    # Mouse h5ad var['GeneName'] uses title-case (e.g. "Aatf").
    # Uppercasing both sides resolves the mismatch without altering stored indices.
    df_source_upper = df['source'].str.upper()
    df_target_upper = df['target'].str.upper()
    tfs_upper  = [g.upper() for g in expressed_tfs]
    genes_upper = [g.upper() for g in expressed_target_genes]

    # Filter: keep only edges where both endpoints are in the expressed gene sets
    mask = df_source_upper.isin(tfs_upper) & df_target_upper.isin(genes_upper)
    df_filtered = df[mask].copy()
    df_filtered['source_up'] = df_source_upper[mask].values
    df_filtered['target_up'] = df_target_upper[mask].values

    # Initialize an empty matrix of shape (Num_TFs, Num_Target_Genes)
    num_tfs = len(expressed_tfs)
    num_genes = len(expressed_target_genes)
    grn_matrix = np.zeros((num_tfs, num_genes), dtype=np.float32)

    # Index maps using uppercase keys → original position in expressed lists
    tf_to_idx   = {g.upper(): i for i, g in enumerate(expressed_tfs)}
    gene_to_idx = {g.upper(): i for i, g in enumerate(expressed_target_genes)}

    # Populate the matrix with interaction scores
    for _, row in df_filtered.iterrows():
        grn_matrix[tf_to_idx[row['source_up']], gene_to_idx[row['target_up']]] = \
            float(row['score'])

    sparsity = float((grn_matrix == 0).mean())
    print(f"Constructed W_TFTG Matrix: {num_tfs} TFs -> {num_genes} Target Genes")
    print(f"Matrix Sparsity: {sparsity * 100:.2f}%  ({len(df_filtered)} edges populated)")

    assert sparsity < 1.0, (
        f"GRN matrix is 100% sparse — no edges populated!\n"
        f"  TFTGDB source sample: {df['source'].head(3).tolist()}\n"
        f"  expressed_tfs sample: {expressed_tfs[:3]}\n"
        f"  Hint: check that expressed_genes.csv contains gene *symbols* "
        f"(e.g. 'Aatf'), not Ensembl IDs (e.g. 'ENSMUSG...')."
    )

    matrix = torch.tensor(grn_matrix, dtype=torch.float32)
    if return_names:
        return matrix, expressed_tfs, expressed_target_genes
    return matrix


def build_spatial_signaling_weights(
    ligrec_path: str,
    rectf_path: str,
    expressed_ligands: List[str],
    expressed_tfs: List[str]
) -> pd.DataFrame:
    """
    Used during CPU preprocessing (02_build_spatial.py) to calculate U_{F, i}.
    It mathematically bridges Ligand -> Receptor -> TF to find out how much
    a specific extracellular ligand ultimately drives a specific nuclear TF.
    """
    print("Mapping Extracellular Ligands to Intracellular TFs...")
    
    df_lr = pd.read_csv(ligrec_path) # source: Ligand, target: Receptor
    df_rtf = pd.read_csv(rectf_path) # source: Receptor, target: TF
    
    # Filter by what is actually expressed
    df_lr = df_lr[df_lr['source'].isin(expressed_ligands)]
    df_rtf = df_rtf[df_rtf['target'].isin(expressed_tfs)]
    
    # Merge the two databases on the Receptor (target of LR, source of RTF)
    merged = pd.merge(
        df_lr, 
        df_rtf, 
        left_on='target', 
        right_on='source',
        suffixes=('_LR', '_RTF')
    )
    
    # Rename for clarity:
    # source_LR = Ligand
    # target_LR / source_RTF = Receptor
    # target_RTF = Transcription Factor
    merged = merged.rename(columns={
        'source_LR': 'Ligand',
        'target_LR': 'Receptor',
        'target_RTF': 'TF',
        'score': 'Activation_Score'
    })
    
    # A single ligand might activate multiple receptors that converge on the same TF.
    # We group by Ligand and TF, summing the activation potentials.
    ligand_tf_weights = merged.groupby(['Ligand', 'TF'])['Activation_Score'].sum().reset_index()
    
    # Pivot into a dense matrix: Rows = Ligands, Columns = TFs
    # This matrix is what you use to multiply the neighborhood ligand concentrations
    # to calculate the total U_{F,i} vector for each cell.
    weight_matrix = ligand_tf_weights.pivot(index='Ligand', columns='TF', values='Activation_Score').fillna(0)
    
    return weight_matrix
