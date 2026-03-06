import os
import numpy as np
import anndata as ad
n_beads, n_genes = 400, 2
spatial_coords = np.random.rand(n_beads, 2) * 100
spliced = np.random.poisson(lam=5.0, size=(n_beads, n_genes)).astype(np.float32)
unspliced = np.random.poisson(lam=2.0, size=(n_beads, n_genes)).astype(np.float32)
adata = ad.AnnData(X=spliced)
adata.obsm['spatial'] = spatial_coords
adata.layers['spliced'] = spliced
adata.layers['unspliced'] = unspliced
adata.write_h5ad("data/processed/fake_spatial.h5ad")
