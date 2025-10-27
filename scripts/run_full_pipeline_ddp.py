"""
Full GEMS Pipeline with DDP for Mouse Brain
Runs Stage A (single GPU) → Stage B → Stage C (DDP on 2 GPUs)

Usage:
    # For full pipeline (Stage A+B+C):
    torchrun --standalone --nproc_per_node=2 scripts/run_full_pipeline_ddp.py \
        --mode full --n_epochs_stageC 2
    
    # For Stage C only (if A+B already done):
    torchrun --standalone --nproc_per_node=2 scripts/run_full_pipeline_ddp.py \
        --mode stagec_only --n_epochs_stageC 400
"""

import os
import sys 
from pathlib import Path
import argparse 
import torch 
import torch.nn as nn 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import scanpy as sc
import pandas as pd
from datetime import datetime


#add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))

#import GEMS components
from core_models_et_p1 import SharedEncoder, STStageBPrecomputer, STSetDataset, SCSetDataset, collate_minisets, collate_sc_minisets
from core_models_et_p2 import SetEncoderContext, DiffusionScoreNet, precompute_st_prototypes
from core_models_et_p3 import GEMSModel
import utils_et as uet

# Import DDP utilities
from ddp_utils import (
    init_dist, cleanup_dist, is_main_process, get_rank, get_world_size,
    rank_print, seed_all, rank_tqdm, all_reduce_mean, broadcast_object
)
from pairwise_shard import is_sharding_beneficial


def load_mousebrain_data(data_dir: Path):
    """Load mouse brain SC and ST data - EXACT COPY from mouse_brain_gems.ipynb."""
    rank_print("Loading mouse brain data...")
    
    # Load SC data
    scdata = pd.read_csv(data_dir / 'simu_sc_counts.csv', index_col=0).T
    scmetadata = pd.read_csv(data_dir / 'metadata.csv', index_col=0)
    
    # Load ST data
    stdata = pd.read_csv(data_dir / 'simu_st_counts.csv', index_col=0).T
    spcoor = pd.read_csv(data_dir / 'simu_st_metadata.csv', index_col=0)
    stgtcelltype = pd.read_csv(data_dir / 'simu_st_celltype.csv', index_col=0)
    
    # Create SC AnnData - EXACT COPY from notebook
    scadata = sc.AnnData(scdata, obs=scmetadata)
    sc.pp.normalize_total(scadata)
    sc.pp.log1p(scadata)
    scadata.obsm['spatial'] = scmetadata[['x_global', 'y_global']].values
    
    # Create ST AnnData - EXACT COPY from notebook
    stadata = sc.AnnData(stdata)
    sc.pp.normalize_total(stadata)
    sc.pp.log1p(stadata)
    stadata.obsm['spatial'] = spcoor[['coord_x', 'coord_y']].values
    
    # Process ST cell type information - EXACT COPY from notebook
    cell_type_columns = stgtcelltype.columns
    dominant_celltypes = []
    
    for i in range(stgtcelltype.shape[0]):
        cell_types_present = [col for col, val in zip(cell_type_columns, stgtcelltype.iloc[i]) if val > 0]
        dominant_celltype = cell_types_present[0] if cell_types_present else 'Unknown'
        dominant_celltypes.append(dominant_celltype)
    
    stadata.obs['celltype'] = dominant_celltypes
    
    # Get common genes
    common_genes = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    n_genes = len(common_genes)
    
    # Extract tensors
    sc_expr = scadata[:, common_genes].X
    st_expr = stadata[:, common_genes].X
    
    if hasattr(sc_expr, 'toarray'):
        sc_expr = sc_expr.toarray()
    if hasattr(st_expr, 'toarray'):
        st_expr = st_expr.toarray()
    
    st_coords = stadata.obsm['spatial']
    
    sc_expr_tensor = torch.tensor(sc_expr, dtype=torch.float32)
    st_expr_tensor = torch.tensor(st_expr, dtype=torch.float32)
    st_coords_tensor = torch.tensor(st_coords, dtype=torch.float32)
    
    rank_print(f"  Common genes: {n_genes}")
    rank_print(f"  SC: {sc_expr_tensor.shape}")
    rank_print(f"  ST: {st_expr_tensor.shape}")
    
    return {
        'sc_expr': sc_expr_tensor,
        'st_expr': st_expr_tensor,
        'st_coords': st_coords_tensor,
        'n_genes': n_genes,
        'common_genes': common_genes,
        'scadata': scadata,
        'stadata': stadata
    }

def run_stage_a(data, args, device):
    """Stage A: Train shared encoder (single GPU on rank 0)."""
    if not is_main_process():
        return None
    
    rank_print("\n" + "="*70)
    rank_print("STAGE A: Training Shared Encoder (Rank 0 only)")
    rank_print("="*70)
    
    # Build encoder
    encoder = SharedEncoder(
        n_genes=data['n_genes'],
        n_embedding=[512, 256, 128],
        dropout=0.1
    ).to(device)
    
    # Single slide
    slide_ids = torch.zeros(data['st_expr'].shape[0], dtype=torch.long)
    
    # Train encoder
    from core_models_et_p1 import train_encoder
    train_encoder(
        model=encoder,
        st_gene_expr=data['st_expr'],
        st_coords=data['st_coords'],
        sc_gene_expr=data['sc_expr'],
        slide_ids=slide_ids,
        n_epochs=args.n_epochs_stageA,
        batch_size=256,
        lr=1e-4,
        sigma=None,
        alpha=0.8,
        ratio_start=0.0,
        ratio_end=1.0,
        mmdbatch=1.0,
        device=str(device),
        outf=str(args.output_dir)
    )
    
    # Save encoder
    encoder_path = args.output_dir / 'encoder_stageA.pt'
    torch.save(encoder.state_dict(), encoder_path)
    rank_print(f"✓ Encoder saved: {encoder_path}")
    
    return encoder


def run_stage_b(data, encoder, args, device):
    """Stage B: Precompute geometric targets (rank 0 only)."""
    if not is_main_process():
        return None
    
    rank_print("\n" + "="*70)
    rank_print("STAGE B: Precomputing Geometric Targets (Rank 0 only)")
    rank_print("="*70)
    
    # Create precomputer
    precomputer = STStageBPrecomputer(
        device=device
    )
    
    # Prepare slide dict
    slides_dict = {
        0: (data['st_coords'].to(device), data['st_expr'].to(device))
    }
    
    # Precompute
    cache_dir = args.output_dir / 'stage_b_cache'
    targets_dict = precomputer.precompute(
        slides=slides_dict,
        encoder=encoder,
        outdir=str(cache_dir)
    )
    
    rank_print(f"✓ Stage B complete. Targets cached at: {cache_dir}")
    
    return targets_dict

def run_stage_c_ddp(data, encoder, targets_dict, args, device, rank, world_size):
    """Stage C: Train diffusion generator with DDP."""
    rank_print("\n" + "="*70)
    rank_print("STAGE C: Training Diffusion Generator (DDP)")
    rank_print("="*70)
    
    # Determine AMP dtype
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True
    
    rank_print(f"AMP dtype: {amp_dtype}, using GradScaler: {use_scaler}")
    
    # Build models
    context_encoder = SetEncoderContext(
        h_dim=128,
        c_dim=256,
        n_heads=4,
        n_blocks=3,
        isab_m=64
    ).to(device)
    
    score_net = DiffusionScoreNet(
        D_latent=args.D_latent,
        c_dim=256,
        time_emb_dim=128,
        n_blocks=4
    ).to(device)
    
    # Wrap in DDP
    if world_size > 1:
        context_encoder = DDP(context_encoder, device_ids=[rank], output_device=rank)
        score_net = DDP(score_net, device_ids=[rank], output_device=rank)
    
    # Optimizer
    params = list(context_encoder.parameters()) + list(score_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Datasets
    st_gene_expr_dict = {0: data['st_expr']}
    
    st_dataset = STSetDataset(
        targets_dict=targets_dict,
        encoder=encoder,
        st_gene_expr_dict=st_gene_expr_dict,
        n_min=args.n_min,
        n_max=args.n_max,
        D_latent=args.D_latent,
        num_samples=args.num_st_samples,
        knn_k=12,
        device=device
    )
    
    sc_dataset = SCSetDataset(
        sc_gene_expr=data['sc_expr'],
        encoder=encoder,
        n_min=args.n_min,
        n_max=args.n_max,
        num_samples=args.num_sc_samples,
        device=device
    )
    
    rank_print(f"  ST dataset: {len(st_dataset)} samples")
    rank_print(f"  SC dataset: {len(sc_dataset)} samples")
    
    # Precompute prototypes on rank 0, then broadcast
    if is_main_process():
        rank_print("Precomputing ST prototypes...")
        prototype_bank = precompute_st_prototypes(
            targets_dict=targets_dict,
            encoder=encoder,
            st_gene_expr_dict=st_gene_expr_dict,
            n_prototypes=3000,
            device=device
        )
    else:
        prototype_bank = None
    
    if world_size > 1:
        prototype_bank = broadcast_object(prototype_bank, src=0)
    
    # Samplers and loaders
    st_sampler = DistributedSampler(st_dataset, shuffle=True) if world_size > 1 else None
    sc_sampler = DistributedSampler(sc_dataset, shuffle=True) if world_size > 1 else None
    
    st_loader = DataLoader(
        st_dataset, batch_size=args.batch_size,
        sampler=st_sampler, shuffle=(st_sampler is None),
        collate_fn=collate_minisets, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    
    sc_loader = DataLoader(
        sc_dataset, batch_size=args.batch_size,
        sampler=sc_sampler, shuffle=(sc_sampler is None),
        collate_fn=collate_sc_minisets, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    
    # Sigma schedule
    sigmas = torch.exp(torch.linspace(
        np.log(args.sigma_min), np.log(args.sigma_max),
        args.n_timesteps, device=device
    ))
    
    # Loss modules
    loss_modules = {
        'gram': uet.FrobeniusGramLoss(),
        'heat': uet.HeatKernelLoss(
            use_hutchinson=True, num_probes=8,
            chebyshev_degree=10, knn_k=8,
            t_list=(0.5, 1.0), laplacian='sym'
        ),
        'sw': uet.SlicedWassersteinLoss1D(),
        'triplet': uet.OrdinalTripletLoss()
    }
    
    # Loss weights
    weights = {
        'score': 1.0, 'gram': 0.5, 'heat': 0.25,
        'sw_st': 0.5, 'sw_sc': 0.3,
        'overlap': 0.25, 'ordinal_sc': 0.5
    }
    
    # Training loop
    for epoch in range(args.n_epochs):
        if st_sampler:
            st_sampler.set_epoch(epoch)
        if sc_sampler:
            sc_sampler.set_epoch(epoch)
        
        context_encoder.train()
        score_net.train()
        
        st_iter = iter(st_loader)
        sc_iter = iter(sc_loader)
        total_batches = len(st_loader) + len(sc_loader)
        
        pbar = rank_tqdm(range(total_batches), desc=f"Epoch {epoch+1}/{args.n_epochs}")
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx in pbar:
            is_st = (batch_idx % 2 == 0) and (batch_idx // 2 < len(st_loader))
            
            try:
                batch = next(st_iter) if is_st else next(sc_iter)
                batch_type = 'ST' if is_st else 'SC'
            except StopIteration:
                try:
                    batch = next(sc_iter if is_st else st_iter)
                    batch_type = 'SC' if is_st else 'ST'
                except StopIteration:
                    break
            
            # Move to device
            Z_set = batch['Z_set'].to(device)
            V_target = batch['V_target'].to(device)
            mask = batch['mask'].to(device)
            
            # Sample timestep
            batch_size = Z_set.shape[0]
            t_idx = torch.randint(0, len(sigmas), (batch_size,), device=device)
            sigma_t = sigmas[t_idx].view(-1, 1, 1)
            
            # Add noise
            noise = torch.randn_like(V_target)
            V_noisy = V_target + sigma_t * noise
            
            # Forward with AMP
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                H = context_encoder(Z_set, mask)
                t_normalized = t_idx.float() / (len(sigmas) - 1)
                eps_pred = score_net(V_noisy, t_normalized.view(-1, 1), H, mask)
                
                # Score matching loss
                loss_score = ((eps_pred - noise) ** 2 * mask.unsqueeze(-1).float()).sum()
                loss_score = loss_score / (mask.sum() * V_target.shape[-1])
                
                # Simple loss for now (full loss with geometric terms can be added)
                loss = weights['score'] * loss_score
            
            # Backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'type': batch_type})
        
        # Print epoch summary
        if is_main_process():
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"\nEpoch {epoch+1}/{args.n_epochs} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint (only final or every 50 epochs)
        if is_main_process() and ((epoch + 1) == args.n_epochs or (epoch + 1) % 50 == 0):
            ckpt_dir = args.output_dir / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            
            # Unwrap DDP
            if world_size > 1:
                context_state = context_encoder.module.state_dict()
                score_state = score_net.module.state_dict()
            else:
                context_state = context_encoder.state_dict()
                score_state = score_net.state_dict()
            
            ckpt = {
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'context_encoder': context_state,
                'score_net': score_state,
                'optimizer': optimizer.state_dict(),
                'cfg': {
                    'model': {
                        'n_genes': data['n_genes'],
                        'n_embedding': [512, 256, 128],
                        'D_latent': args.D_latent,
                        'c_dim': 256,
                        'n_heads': 4,
                        'isab_m': 64
                    }
                }
            }
            
            if scaler:
                ckpt['scaler'] = scaler.state_dict()
            
            ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch+1:04d}.pt'
            torch.save(ckpt, ckpt_path)
            rank_print(f"  ✓ Checkpoint saved: {ckpt_path}")
    
    rank_print("✓ Stage C complete")
    
    return context_encoder, score_net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'stagec_only'], default='full')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/ehtesamul/sc_st/data/mousedata_2020/E1z2')
    parser.add_argument('--output_dir', type=str,
                        default='./gems_mousebrain_ddp_output')
    
    # STAGE C PARAMS (THESE ARE WHAT MATTER FOR SPEED)
    parser.add_argument('--n_epochs', type=int, default=3,
                        help='Stage C epochs (main bottleneck)')
    parser.add_argument('--n_min', type=int, default=128,
                        help='Min mini-set size (lower=faster)')
    parser.add_argument('--n_max', type=int, default=256,
                        help='Max mini-set size (lower=faster)')
    parser.add_argument('--num_st_samples', type=int, default=100,
                        help='ST dataset size (lower=faster)')
    parser.add_argument('--num_sc_samples', type=int, default=300,
                        help='SC dataset size (lower=faster)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_timesteps', type=int, default=400,
                        help='Diffusion steps (lower=faster)')
    parser.add_argument('--sigma_min', type=float, default=0.1)
    parser.add_argument('--sigma_max', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Less important params
    parser.add_argument('--n_epochs_stageA', type=int, default=1000,
                        help='Stage A epochs (fast, dont worry about it)')
    parser.add_argument('--D_latent', type=int, default=16)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.data_dir = Path(args.data_dir)
    
    # Initialize distributed
    rank, local_rank, world_size, device = init_dist()
    
    print(f"Rank {rank}/{world_size-1} on cuda:{local_rank} | seed={args.seed+rank}")
    
    # Seed
    seed_all(args.seed, rank)
    
    # Load data
    data = load_mousebrain_data(args.data_dir)
    
    # Create output directory
    if is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()  # Wait for rank 0 to create directory
    
    # Run pipeline
    if args.mode == 'full':
        # =====================================================================
        # STAGE A: Train encoder (rank 0 only)
        # =====================================================================
        encoder = run_stage_a(data, args, device)
        
        # Load encoder on all ranks
        if world_size > 1:
            dist.barrier()  # Wait for rank 0 to save
        
        if not is_main_process():
            encoder = SharedEncoder(
                n_genes=data['n_genes'],
                n_embedding=[512, 256, 128]
            ).to(device)
            encoder_path = args.output_dir / 'encoder_stageA.pt'
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        
        encoder.eval()
        
        # =====================================================================
        # STAGE B: Precompute targets (rank 0 only, then broadcast)
        # =====================================================================
        targets_dict = run_stage_b(data, encoder, args, device)
        
        # BROADCAST targets instead of loading from disk
        if world_size > 1:
            rank_print("Broadcasting Stage B targets from rank 0...")
            targets_dict = broadcast_object(targets_dict, src=0)
            rank_print(f"  Rank {rank} received targets")
        
    else:  # stagec_only
        rank_print("Loading encoder and targets from previous run...")
        encoder = SharedEncoder(
            n_genes=data['n_genes'],
            n_embedding=[512, 256, 128]
        ).to(device)
        encoder_path = args.output_dir / 'encoder_stageA.pt'
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.eval()
        
        # Load from cache (all ranks)
        cache_dir = args.output_dir / 'stage_b_cache'
        targets_0 = torch.load(cache_dir / 'slide_0.pt', map_location=device)
        targets_dict = {0: targets_0}
    
    # =========================================================================
    # STAGE C: DDP training (all ranks)
    # =========================================================================
    rank_print("\n" + "="*70)
    rank_print("STAGE C: Training Diffusion Generator (DDP)")
    rank_print("="*70)
    
    context_encoder, score_net = run_stage_c_ddp(
        data, encoder, targets_dict, args, device, rank, world_size
    )
    
    rank_print("\n" + "="*70)
    rank_print("PIPELINE COMPLETE!")
    rank_print("="*70)
    rank_print(f"Output directory: {args.output_dir}")
    
    cleanup_dist()


if __name__ == '__main__':
    main()




