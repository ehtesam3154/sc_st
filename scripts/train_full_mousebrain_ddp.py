#!/usr/bin/env python3
"""
Full GEMS training pipeline (Stage A + B + C) with DDP for mouse brain.
Stage A & B run on single GPU, Stage C uses DDP.
"""

import os
import sys
from pathlib import Path
import argparse
import torch

# Set path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.core_models_et_p3 import GEMSModel
import scanpy as sc
import pandas as pd


def load_mousebrain_data():
    """Load mouse brain data."""
    print("Loading mouse brain data...")
    
    scdata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_sc_counts.csv', index_col=0).T
    stdata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_counts.csv', index_col=0).T
    
    scadata = sc.AnnData(scdata)
    stadata = sc.AnnData(stdata)
    
    sc.pp.normalize_total(scadata)
    sc.pp.log1p(scadata)
    sc.pp.normalize_total(stadata)
    sc.pp.log1p(stadata)
    
    spcoor = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_metadata.csv', index_col=0)
    stadata.obsm['spatial'] = spcoor[['coord_x', 'coord_y']].values
    
    return scadata, stadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output_mousebrain_ddp')
    parser.add_argument('--n_epochs_stageA', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--n_min', type=int, default=128)
    parser.add_argument('--n_max', type=int, default=256)
    parser.add_argument('--num_st_samples', type=int, default=100)
    parser.add_argument('--num_sc_samples', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_timesteps', type=int, default=400)
    parser.add_argument('--sigma_min', type=float, default=0.1)
    parser.add_argument('--sigma_max', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--D_latent', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load data
    scadata, stadata = load_mousebrain_data()
    
    # Get common genes
    common_genes = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    n_genes = len(common_genes)
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
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=args.D_latent,
        c_dim=256,
        n_heads=4,
        isab_m=64,
        device=device
    )
    
    # STAGE A
    print("\n" + "="*70)
    print("STAGE A: Training Shared Encoder")
    print("="*70)
    
    model.train_stageA(
        st_gene_expr=st_expr_tensor,
        st_coords=st_coords_tensor,
        sc_gene_expr=sc_expr_tensor,
        slide_ids=slide_ids,
        n_epochs=args.n_epochs_stageA,
        batch_size=256,
        lr=0.0001,
        sigma=None,
        alpha=0.8,
        ratio_start=0.0,
        ratio_end=1.0,
        mmdbatch=1.0,
        outf=str(output_dir)
    )
    
    # STAGE B
    print("\n" + "="*70)
    print("STAGE B: Precomputing Targets")
    print("="*70)
    
    slides_dict_device = {
        sid: (coords.to(device), expr.to(device))
        for sid, (coords, expr) in slides_dict.items()
    }
    
    model.train_stageB(
        slides=slides_dict_device,
        outdir=str(output_dir / 'stage_b_cache')
    )
    
    # Save model after Stage A & B
    model.save(str(output_dir / 'model_after_stageAB.pt'))
    
    # Save data package for Stage C DDP
    data_package = {
        'st_gene_expr_dict': st_gene_expr_dict,
        'sc_gene_expr': sc_expr_tensor,
        'targets_dict': model.targets_dict,
        'common_genes': common_genes,
        'model_config': {
            'n_genes': n_genes,
            'n_embedding': [512, 256, 128],
            'D_latent': args.D_latent,
            'c_dim': 256,
            'n_heads': 4,
            'isab_m': 64
        }
    }
    
    torch.save(data_package, output_dir / 'stageC_inputs.pt')
    print(f"\n✓ Stage A & B complete. Data saved to {output_dir / 'stageC_inputs.pt'}")
    print(f"\nNow run Stage C with DDP:")
    print(f"  export CUDA_VISIBLE_DEVICES=0,1")
    print(f"  torchrun --standalone --nproc_per_node=2 scripts/train_stageC_ddp.py \\")
    print(f"    --data_pt {output_dir / 'stageC_inputs.pt'} \\")
    print(f"    --output_dir {output_dir / 'checkpoints'} \\")
    print(f"    --n_epochs {args.n_epochs} \\")
    print(f"    --batch_size {args.batch_size} \\")
    print(f"    --n_min {args.n_min} \\")
    print(f"    --n_max {args.n_max} \\")
    print(f"    --num_st_samples {args.num_st_samples} \\")
    print(f"    --num_sc_samples {args.num_sc_samples} \\")
    print(f"    --n_timesteps {args.n_timesteps} \\")
    print(f"    --sigma_min {args.sigma_min} \\")
    print(f"    --sigma_max {args.sigma_max} \\")
    print(f"    --lr {args.lr}")


if __name__ == '__main__':
    main()