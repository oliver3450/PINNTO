import pandas as pd
import numpy as np
import torch
from typing import List, Tuple

def build_frozen_grn_matrix(
    tftg_path: str, 
    expressed_tfs: List[str], 
    expressed_target_genes: List[str]
) -> torch.Tensor:
    """
    Builds the W_TFTG matrix for the PINN's output projection layer.
    Only includes genes that are physically present in your spatial dataset.
    """
    print(f"Loading TF-TG database from {tftg_path}...")
    df = pd.read_csv(tftg_path)
    
    # Filter the database to ONLY include TFs and Genes actually in your data
    df = df[df['source'].isin(expressed_tfs) & df['target'].isin(expressed_target_genes)]
    
    # Initialize an empty matrix of shape (Num_TFs, Num_Target_Genes)
    num_tfs = len(expressed_tfs)
    num_genes = len(expressed_target_genes)
    grn_matrix = np.zeros((num_tfs, num_genes), dtype=np.float32)
    
    # Create mapping dictionaries for fast indexing
    tf_to_idx = {tf: i for i, tf in enumerate(expressed_tfs)}
    gene_to_idx = {gene: i for i, gene in enumerate(expressed_target_genes)}
    
    # Populate the matrix with the interaction scores
    for _, row in df.iterrows():
        tf_idx = tf_to_idx[row['source']]
        gene_idx = gene_to_idx[row['target']]
        
        # If your score is continuous, it uses it. If it's just '1', it acts as a binary mask.
        grn_matrix[tf_idx, gene_idx] = float(row['score'])
        
    print(f"Constructed W_TFTG Matrix: {num_tfs} TFs -> {num_genes} Target Genes")
    print(f"Matrix Sparsity: {(grn_matrix == 0).mean() * 100:.2f}%")
    
    return torch.tensor(grn_matrix, dtype=torch.float32)


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
