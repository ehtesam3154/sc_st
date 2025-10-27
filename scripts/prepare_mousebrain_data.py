#!/usr/bin/env python3
"""
Prepare mouse brain data for DDP training.
"""

import sys
from pathlib import Path
import torch
import scanpy as sc
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("Preparing mouse brain data for DDP training...")
    
    output_dir = Path('./gems_mousebrain_output')
    stagec_output = Path('./data/mousebrain_stageC_inputs.pt')
    stagec_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model checkpoint
    model_path = output_dir / 'gems_model_mousebrain.pt'
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load original data
    print("Loading original data...")
    scdata = pd.read_csv('data/mousedata_2020/E1z2/simu_sc_counts.csv', index_col=0).T
    stdata = pd.read_csv('data/mousedata_2020/E1z2/simu_st_counts.csv', index_col=0).T
    
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
    
    print(f"SC expression: {sc_expr_tensor.shape}")
    print(f"ST expression: {st_expr_tensor.shape}")
    print(f"Common genes: {len(common_genes)}")
    
    st_gene_expr_dict = {0: st_expr_tensor}
    
    # Get targets_dict from checkpoint
    print("Loading targets from checkpoint...")
    if 'targets_dict' in checkpoint:
        targets_dict = checkpoint['targets_dict']
        print(f"  Loaded targets_dict from checkpoint")
    else:
        print("  ERROR: targets_dict not found in checkpoint!")
        print("  The model needs to save targets_dict in train_stageB")
        
        # Try to load from stage_b_cache as fallback
        stage_b_cache = output_dir / 'stage_b_cache'
        if stage_b_cache.exists():
            print(f"  Fallback: Loading from {stage_b_cache}")
            targets_0 = torch.load(stage_b_cache / 'slide_0.pt', map_location='cpu')
            targets_dict = {0: targets_0}
        else:
            print("  ERROR: Neither checkpoint nor stage_b_cache contains targets")
            print("  Please re-run Stage B training")
            return
    
    # Save everything
    data_package = {
        'st_gene_expr_dict': st_gene_expr_dict,
        'sc_gene_expr': sc_expr_tensor,
        'targets_dict': targets_dict,
        'common_genes': common_genes
    }
    
    print(f"\nSaving to: {stagec_output}")
    torch.save(data_package, stagec_output)
    
    print("✓ Data preparation complete!")
    print(f"\nNow run DDP training:")
    print(f"  export CUDA_VISIBLE_DEVICES=0,1")
    print(f"  torchrun --standalone --nproc_per_node=2 scripts/train_stageC_ddp.py --cfg configs/mousebrain_stageC_ddp.yaml")


if __name__ == '__main__':
    main()