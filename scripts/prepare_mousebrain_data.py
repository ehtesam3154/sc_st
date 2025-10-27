#!/usr/bin/env python3
"""
Prepare mouse brain data for DDP training.
Saves pre-encoded tensors and targets to .pt file for fast loading.

Usage:
    python scripts/prepare_mousebrain_data.py
"""

import sys
from pathlib import Path
import torch
import scanpy as sc
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.core_models_et_p1 import SharedEncoder
from model.core_models_et_p3 import GEMSModel

def main():
    print("Preparing mouse brain data for DDP training...")
    
    # Paths
    output_dir = Path('./gems_mousebrain_output')
    stagec_output = Path('./data/mousebrain_stageC_inputs.pt')
    stagec_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load trained model (after Stage A and B)
    model_path = output_dir / 'gems_model_mousebrain.pt'
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run single-GPU training first to complete Stage A and B")
        return
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load original data
    print("Loading original data...")
    import pandas as pd
    
    scdata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_sc_counts.csv', index_col=0).T
    stdata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_counts.csv', index_col=0).T
    
    scadata = sc.AnnData(scdata)
    stadata = sc.AnnData(stdata)
    
    sc.pp.normalize_total(scadata)
    sc.pp.log1p(scadata)
    sc.pp.normalize_total(stadata)
    sc.pp.log1p(stadata)
    
    # Get common genes
    common_genes = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    
    # Extract tensors
    sc_expr = scadata[:, common_genes].X
    st_expr = stadata[:, common_genes].X
    
    if hasattr(sc_expr, 'toarray'):
        sc_expr = sc_expr.toarray()
    if hasattr(st_expr, 'toarray'):
        st_expr = st_expr.toarray()
    
    sc_expr_tensor = torch.tensor(sc_expr, dtype=torch.float32)
    st_expr_tensor = torch.tensor(st_expr, dtype=torch.float32)
    
    # Load spatial coords
    spcoor = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_metadata.csv', index_col=0)
    st_coords = torch.tensor(spcoor[['coord_x', 'coord_y']].values, dtype=torch.float32)
    
    print(f"SC expression: {sc_expr_tensor.shape}")
    print(f"ST expression: {st_expr_tensor.shape}")
    print(f"Common genes: {len(common_genes)}")
    
    # Prepare data dict
    st_gene_expr_dict = {
        0: st_expr_tensor
    }
    
    # Load targets_dict from Stage B cache
    stage_b_cache = output_dir / 'stage_b_cache'
    if not stage_b_cache.exists():
        print(f"ERROR: Stage B cache not found at {stage_b_cache}")
        print("Please run single-GPU training to complete Stage B first")
        return
    
    print(f"Loading Stage B targets from: {stage_b_cache}")
    
    # Load targets for slide 0
    targets_0 = torch.load(stage_b_cache / 'slide_0.pt', map_location='cpu')
    targets_dict = {0: targets_0}
    
    # Save everything
    data_package = {
        'st_gene_expr_dict': st_gene_expr_dict,
        'sc_gene_expr': sc_expr_tensor,
        'targets_dict': targets_dict,
        'common_genes': common_genes,
        'st_coords': st_coords
    }
    
    print(f"\nSaving to: {stagec_output}")
    torch.save(data_package, stagec_output)
    
    print("✓ Data preparation complete!")
    print(f"\nNow you can run DDP training with:")
    print(f"  torchrun --standalone --nproc_per_node=2 scripts/train_stageC_ddp.py --cfg configs/mousebrain_stageC_ddp.yaml")


if __name__ == '__main__':
    main()