#!/usr/bin/env python3
"""
Train Stage A and B for mouse brain data (single GPU)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from threadpoolctl import threadpool_limits
threadpool_limits(limits=1, user_api='blas')

import sys
from pathlib import Path
import torch
import scanpy as sc
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.core_models_et_p3 import GEMSModel

def main():
    device = 'cuda'
    output_dir = Path('./gems_mousebrain_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("STAGE A + B TRAINING (SINGLE GPU)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    scdata = pd.read_csv('data/mousedata_2020/E1z2/simu_sc_counts.csv', index_col=0).T
    stdata = pd.read_csv('data/mousedata_2020/E1z2/simu_st_counts.csv', index_col=0).T
    spcoor = pd.read_csv('data/mousedata_2020/E1z2/simu_st_metadata.csv', index_col=0)
    
    # Create AnnData
    scadata = sc.AnnData(scdata)
    stadata = sc.AnnData(stdata)
    
    # Normalize
    sc.pp.normalize_total(scadata)
    sc.pp.log1p(scadata)
    sc.pp.normalize_total(stadata)
    sc.pp.log1p(stadata)
    
    stadata.obsm['spatial'] = spcoor[['coord_x', 'coord_y']].values
    
    # Get common genes
    common_genes = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    n_genes = len(common_genes)
    
    print(f"SC cells: {scadata.shape[0]}")
    print(f"ST spots: {stadata.shape[0]}")
    print(f"Common genes: {n_genes}")
    
    # Extract tensors
    sc_expr = scadata[:, common_genes].X
    st_expr = stadata[:, common_genes].X
    
    if hasattr(sc_expr, 'toarray'):
        sc_expr = sc_expr.toarray()
    if hasattr(st_expr, 'toarray'):
        st_expr = st_expr.toarray()
    
    sc_expr_tensor = torch.tensor(sc_expr, dtype=torch.float32)
    st_expr_tensor = torch.tensor(st_expr, dtype=torch.float32)
    st_coords_tensor = torch.tensor(stadata.obsm['spatial'], dtype=torch.float32)
    
    slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long)
    
    slides_dict = {0: (st_coords_tensor, st_expr_tensor)}
    st_gene_expr_dict = {0: st_expr_tensor}
    
    # Initialize model
    print("\nInitializing GEMS model...")
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=16,
        c_dim=256,
        n_heads=4,
        isab_m=64,
        device=device
    )
    
    # Stage A
    print("\n" + "="*70)
    print("STAGE A: Training Shared Encoder")
    print("="*70)
    
    model.train_stageA(
        st_gene_expr=st_expr_tensor,
        st_coords=st_coords_tensor,
        sc_gene_expr=sc_expr_tensor,
        slide_ids=slide_ids,
        n_epochs=1000,
        batch_size=256,
        lr=0.0001,
        sigma=None,
        alpha=0.8,
        ratio_start=0.0,
        ratio_end=1.0,
        mmdbatch=1.0,
        outf=str(output_dir)
    )
    
    # Stage B
    print("\n" + "="*70)
    print("STAGE B: Precomputing Geometric Targets")
    print("="*70)
    
    slides_dict_device = {
        sid: (coords.to(device), expr.to(device))
        for sid, (coords, expr) in slides_dict.items()
    }
    
    model.train_stageB(
        slides=slides_dict_device,
        outdir=str(output_dir / 'stage_b_cache')
    )
    
    # Save model
    print("\nSaving model...")
    model.save(str(output_dir / 'gems_model_mousebrain.pt'))
    
    # Save common genes
    torch.save({'common_genes': common_genes}, output_dir / 'common_genes.pt')
    
    print("\n" + "="*70)
    print("STAGE A + B COMPLETE!")
    print("="*70)
    print(f"Model saved: {output_dir / 'gems_model_mousebrain.pt'}")
    print(f"Stage B cache: {output_dir / 'stage_b_cache'}")
    print("\nNow run: python scripts/prepare_mousebrain_data.py")

if __name__ == '__main__':
    main()