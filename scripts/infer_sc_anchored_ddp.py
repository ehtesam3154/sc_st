#!/usr/bin/env python3
"""
Parallel Anchor-Conditioned SC Inference with PyTorch DDP
Distributed inference script for GEMS model across 2 GPUs

Usage:
    torchrun --standalone --nproc_per_node=2 scripts/infer_sc_anchored_ddp.py \
        --ckpt path/to/checkpoint.pt \
        --sc_expr path/to/sc_expr.pt \
        --out path/to/output.pt \
        --anchor_size 384 \
        --batch_size 512
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import gc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GEMS components
from model.core_models_et_p1 import SharedEncoder
from model.core_models_et_p2 import SetEncoderContext, DiffusionScoreNet
import model.utils_et as uet

# Import DDP utilities
from model.ddp_utils import (
    init_dist, cleanup_dist, is_main_process, get_rank, get_world_size,
    rank_print, seed_all, rank_tqdm, broadcast_object, all_gather_objects
)


def load_models_from_checkpoint(
    ckpt_path: str,
    device: torch.device
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    """
    Load encoder, context_encoder, and score_net from checkpoint.
    
    Returns:
        (encoder, context_encoder, score_net, cfg)
    """
    rank_print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    cfg = ckpt['cfg']
    
    # Build models
    encoder = SharedEncoder(
        n_genes=cfg['model']['n_genes'],
        n_embedding=cfg['model']['n_embedding']
    ).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    
    context_encoder = SetEncoderContext(
        h_dim=cfg['model']['n_embedding'][-1],
        c_dim=cfg['model']['c_dim'],
        n_heads=cfg['model']['n_heads'],
        n_blocks=3,
        isab_m=cfg['model']['isab_m']
    ).to(device)
    
    # Handle DDP-wrapped state dict
    context_state = ckpt['context_encoder']
    if list(context_state.keys())[0].startswith('module.'):
        # Unwrap DDP prefix
        context_state = {k.replace('module.', ''): v for k, v in context_state.items()}
    context_encoder.load_state_dict(context_state)
    context_encoder.eval()
    
    score_net = DiffusionScoreNet(
        D_latent=cfg['model']['D_latent'],
        c_dim=cfg['model']['c_dim'],
        time_emb_dim=128,
        n_blocks=4
    ).to(device)
    
    score_state = ckpt['score_net']
    if list(score_state.keys())[0].startswith('module.'):
        score_state = {k.replace('module.', ''): v for k, v in score_state.items()}
    score_net.load_state_dict(score_state)
    score_net.eval()
    
    rank_print(f"  Models loaded successfully")
    rank_print(f"  D_latent: {cfg['model']['D_latent']}")
    
    return encoder, context_encoder, score_net, cfg


def encode_all_sc_cells(
    sc_gene_expr: torch.Tensor,
    encoder: nn.Module,
    device: torch.device,
    encode_batch_size: int = 1024
) -> torch.Tensor:
    """
    Encode all SC cells in batches.
    
    Returns:
        Z_all: (n_sc, h_dim) embeddings on CPU
    """
    n_sc = sc_gene_expr.shape[0]
    Z_all = []
    
    with torch.no_grad():
        for i in range(0, n_sc, encode_batch_size):
            batch = sc_gene_expr[i:i+encode_batch_size].to(device)
            z = encoder(batch).cpu()
            Z_all.append(z)
            
            # Clear cache periodically
            if (i // encode_batch_size) % 10 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    Z_all = torch.cat(Z_all, dim=0)
    return Z_all


def select_anchors_fps(
    Z_all: torch.Tensor,
    anchor_size: int
) -> torch.Tensor:
    """
    Select anchor indices using Farthest Point Sampling.
    
    Args:
        Z_all: (n_sc, h_dim) embeddings
        anchor_size: Number of anchors
    
    Returns:
        anchor_idx: (anchor_size,) long tensor of indices
    """
    n_sc = Z_all.shape[0]
    anchor_size = min(anchor_size, n_sc)
    
    # Normalize for FPS
    Z_unit = Z_all / (Z_all.norm(dim=1, keepdim=True) + 1e-8)
    
    anchor_idx = uet.farthest_point_sampling(Z_unit, anchor_size, device='cpu')
    
    return anchor_idx


def sample_anchor_positions(
    Z_A: torch.Tensor,
    context_encoder: nn.Module,
    score_net: nn.Module,
    n_timesteps: int,
    sigma_min: float,
    sigma_max: float,
    D_latent: int,
    device: torch.device,
    guidance_scale: float = 8.0
) -> torch.Tensor:
    """
    Sample latent positions for anchor cells using reverse diffusion.
    
    Args:
        Z_A: (1, n_anchor, h_dim) anchor embeddings
        context_encoder: Context encoder model
        score_net: Score network model
        n_timesteps: Number of diffusion steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        D_latent: Latent dimension
        device: Device to run on
        guidance_scale: CFG guidance scale
    
    Returns:
        V_A: (n_anchor, D_latent) anchor positions
    """
    n_anchor = Z_A.shape[1]
    mask_A = torch.ones(1, n_anchor, dtype=torch.bool, device=device)
    
    # Compute context
    with torch.no_grad(), torch.cuda.amp.autocast():
        H_A = context_encoder(Z_A, mask_A)
    
    # Sigma schedule (decreasing: sigma_max → sigma_min)
    sigmas = torch.exp(torch.linspace(
        np.log(sigma_max), np.log(sigma_min),
        n_timesteps, device=device
    ))
    
    # Initialize with high noise
    V_t = torch.randn(1, n_anchor, D_latent, device=device) * sigmas[0]
    
    # Reverse diffusion
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, sigma_t in enumerate(sigmas):
            t_normalized = torch.full((1, 1), i / (n_timesteps - 1), device=device)
            
            # Unconditional score (no context)
            H_uncond = torch.zeros_like(H_A)
            eps_uncond = score_net(V_t, t_normalized, H_uncond, mask_A)
            
            # Conditional score (with context)
            eps_cond = score_net(V_t, t_normalized, H_A, mask_A)
            
            # Classifier-free guidance
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # Denoising step
            if i < n_timesteps - 1:
                sigma_next = sigmas[i + 1]
                V_t = V_t - sigma_t * eps
                
                # Add noise for next step (Langevin dynamics)
                noise_scale = torch.sqrt(torch.clamp((sigma_next**2 / sigma_t**2) * sigma_t**2, min=0))
                V_t = V_t + noise_scale * torch.randn_like(V_t)
            else:
                # Final step - no noise
                V_t = V_t - sigma_t * eps
    
    V_A = V_t.squeeze(0).cpu()
    return V_A


def process_batch_with_frozen_anchors(
    Z_batch: torch.Tensor,
    Z_A: torch.Tensor,
    V_A: torch.Tensor,
    context_encoder: nn.Module,
    score_net: nn.Module,
    n_timesteps: int,
    sigma_min: float,
    sigma_max: float,
    D_latent: int,
    device: torch.device,
    guidance_scale: float = 8.0,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Sample positions for a batch of cells with frozen anchor positions.
    
    Args:
        Z_batch: (n_batch, h_dim) embeddings for batch cells
        Z_A: (n_anchor, h_dim) anchor embeddings
        V_A: (n_anchor, D_latent) anchor positions (frozen)
        ... other args same as sample_anchor_positions
        eta: Stochasticity factor (0=deterministic, 1=full noise)
    
    Returns:
        V_batch: (n_batch, D_latent) sampled positions for batch cells
    """
    n_anchor = Z_A.shape[0]
    n_batch = Z_batch.shape[0]
    n_total = n_anchor + n_batch
    
    # Combine anchors + batch
    Z_combined = torch.cat([Z_A, Z_batch], dim=0).unsqueeze(0).to(device)  # (1, n_total, h_dim)
    mask_combined = torch.ones(1, n_total, dtype=torch.bool, device=device)
    
    # Context encoding
    with torch.no_grad(), torch.cuda.amp.autocast():
        H_combined = context_encoder(Z_combined, mask_combined)
    
    # Initialize V: anchors are fixed, batch starts with noise
    V_A_device = V_A.to(device)
    V_batch_init = torch.randn(n_batch, D_latent, device=device) * sigma_max
    V_t = torch.cat([V_A_device, V_batch_init], dim=0).unsqueeze(0)  # (1, n_total, D_latent)
    
    # Freeze mask: only update batch cells
    upd_mask = torch.zeros(1, n_total, 1, device=device)
    upd_mask[:, n_anchor:, :] = 1.0  # Only update non-anchor cells
    
    # Sigma schedule
    sigmas = torch.exp(torch.linspace(
        np.log(sigma_max), np.log(sigma_min),
        n_timesteps, device=device
    ))
    
    # Reverse diffusion
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, sigma_t in enumerate(sigmas):
            t_normalized = torch.full((1, 1), i / (n_timesteps - 1), device=device)
            
            # Unconditional
            H_uncond = torch.zeros_like(H_combined)
            eps_uncond = score_net(V_t, t_normalized, H_uncond, mask_combined)
            
            # Conditional
            eps_cond = score_net(V_t, t_normalized, H_combined, mask_combined)
            
            # CFG
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # Update step
            if i < n_timesteps - 1:
                sigma_next = sigmas[i + 1]
                V_new = V_t - sigma_t * eps
                
                # Add noise with eta factor
                if eta > 0:
                    noise_scale = eta * torch.sqrt(torch.clamp((sigma_next**2 / sigma_t**2) * sigma_t**2, min=0))
                    V_new = V_new + noise_scale * torch.randn_like(V_new)
            else:
                V_new = V_t - sigma_t * eps
            
            # Apply freeze mask (only update non-anchor positions)
            V_t = upd_mask * V_new + (1.0 - upd_mask) * V_t
    
    # Extract batch positions only
    V_batch = V_t.squeeze(0)[n_anchor:, :].cpu()
    return V_batch


def split_work_across_ranks(
    non_anchor_idx: torch.Tensor,
    batch_size: int,
    rank: int,
    world_size: int
) -> List[Tuple[int, torch.Tensor]]:
    """
    Split non-anchor cells into batches and assign to ranks.
    
    Args:
        non_anchor_idx: Indices of non-anchor cells
        batch_size: Batch size
        rank: Current rank
        world_size: Total ranks
    
    Returns:
        List of (batch_id, indices) tuples for this rank
    """
    n_non_anchor = len(non_anchor_idx)
    n_batches = (n_non_anchor + batch_size - 1) // batch_size
    
    my_batches = []
    for batch_id in range(n_batches):
        if batch_id % world_size == rank:
            start = batch_id * batch_size
            end = min(start + batch_size, n_non_anchor)
            batch_idx = non_anchor_idx[start:end]
            my_batches.append((batch_id, batch_idx))
    
    return my_batches


def main():
    parser = argparse.ArgumentParser(description='Parallel Anchored SC Inference')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--sc_expr', type=str, required=True, help='Path to SC expression tensor (.pt)')
    parser.add_argument('--out', type=str, required=True, help='Output path (.pt)')
    parser.add_argument('--anchor_size', type=int, default=384, help='Number of anchor cells')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (excluding anchors)')
    parser.add_argument('--n_timesteps', type=int, default=160, help='Number of diffusion steps')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='Minimum noise level')
    parser.add_argument('--sigma_max', type=float, default=50.0, help='Maximum noise level')
    parser.add_argument('--guidance_scale', type=float, default=8.0, help='CFG guidance scale')
    parser.add_argument('--eta', type=float, default=0.0, help='Stochasticity (0=deterministic)')
    parser.add_argument('--save_edm', action='store_true', help='Save full EDM (can be large)')
    parser.add_argument('--save_coords', action='store_true', default=True, help='Compute and save coordinates')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, local_rank, world_size, device = init_dist()
    
    # Print startup info
    if world_size > 1:
        print(f"Rank {rank}/{world_size-1} on cuda:{local_rank} | seed={args.seed+rank}")
    else:
        rank_print(f"Single-GPU mode on {device}")
    
    # Seed
    seed_all(args.seed, rank)
    
    rank_print("\n" + "="*70)
    rank_print("PARALLEL ANCHOR-CONDITIONED SC INFERENCE")
    rank_print("="*70)
    
    # Load models (all ranks)
    encoder, context_encoder, score_net, cfg = load_models_from_checkpoint(args.ckpt, device)
    D_latent = cfg['model']['D_latent']
    
    # Load SC expression data
    rank_print(f"Loading SC expression: {args.sc_expr}")
    sc_gene_expr = torch.load(args.sc_expr, map_location='cpu')
    n_sc = sc_gene_expr.shape[0]
    rank_print(f"  SC cells: {n_sc}")
    rank_print(f"  Genes: {sc_gene_expr.shape[1]}")
    
    # Clear cache before inference
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    # =========================================================================
    # STEP 1: Encode all cells (all ranks do this - could optimize later)
    # =========================================================================
    rank_print("\nStep 1: Encoding SC cells...")
    Z_all = encode_all_sc_cells(sc_gene_expr, encoder, device)
    rank_print(f"  Encoded shape: {Z_all.shape}")
    
    # =========================================================================
    # STEP 2: Select anchors (rank 0 only, then broadcast)
    # =========================================================================
    if is_main_process():
        rank_print("\nStep 2: Selecting anchors via FPS...")
        anchor_idx = select_anchors_fps(Z_all, args.anchor_size)
        rank_print(f"  Selected {len(anchor_idx)} anchors")
        
        # Get non-anchor indices
        non_anchor_idx = torch.tensor(
            [i for i in range(n_sc) if i not in set(anchor_idx.tolist())],
            dtype=torch.long
        )
        rank_print(f"  Remaining cells: {len(non_anchor_idx)}")
    else:
        anchor_idx = None
        non_anchor_idx = None
    
    # Broadcast indices
    if world_size > 1:
        anchor_idx = broadcast_object(anchor_idx, src=0)
        non_anchor_idx = broadcast_object(non_anchor_idx, src=0)
    
    # =========================================================================
    # STEP 3: Sample anchor positions (rank 0 only, then broadcast)
    # =========================================================================
    if is_main_process():
        rank_print("\nStep 3: Sampling anchor positions...")
        Z_A = Z_all[anchor_idx].unsqueeze(0).to(device)
        
        V_A = sample_anchor_positions(
            Z_A, context_encoder, score_net,
            args.n_timesteps, args.sigma_min, args.sigma_max,
            D_latent, device, args.guidance_scale
        )
        
        rank_print(f"  Anchor positions sampled: {V_A.shape}")
    else:
        V_A = None
    
    # Broadcast anchor positions
    if world_size > 1:
        V_A = broadcast_object(V_A, src=0)
    
    # =========================================================================
    # STEP 4: Process remaining cells with frozen anchors (parallel)
    # =========================================================================
    rank_print(f"\nStep 4: Processing remaining cells (Rank {rank})...")
    
    # Split work
    my_batches = split_work_across_ranks(non_anchor_idx, args.batch_size, rank, world_size)
    
    rank_print(f"  Rank {rank} processing {len(my_batches)} batches")
    
    # Process batches
    my_results = []  # List of (indices, V_batch) tuples
    
    Z_A = Z_all[anchor_idx]  # Keep on CPU for now
    
    pbar = rank_tqdm(my_batches, desc=f"Batching")
    
    for batch_id, batch_idx in pbar:
        Z_batch = Z_all[batch_idx]  # (n_batch, h_dim)
        
        V_batch = process_batch_with_frozen_anchors(
            Z_batch, Z_A, V_A,
            context_encoder, score_net,
            args.n_timesteps, args.sigma_min, args.sigma_max,
            D_latent, device, args.guidance_scale, args.eta
        )
        
        my_results.append({
            'indices': batch_idx.cpu(),
            'V': V_batch.cpu()
        })
        
        # Clear cache periodically
        if batch_id % 5 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    rank_print(f"  Rank {rank} completed {len(my_results)} batches")
    
    # =========================================================================
    # STEP 5: Gather results to rank 0
    # =========================================================================
    rank_print("\nStep 5: Gathering results...")
    
    if world_size > 1:
        all_results = all_gather_objects(my_results)
        # Flatten list of lists
        if is_main_process():
            all_results = [item for sublist in all_results for item in sublist]
    else:
        all_results = my_results
    
    # =========================================================================
    # STEP 6: Assemble final result (rank 0 only)
    # =========================================================================
    if is_main_process():
        rank_print("\nStep 6: Assembling final coordinates...")
        
        # Initialize full V tensor
        V_0_full = torch.empty(n_sc, D_latent)
        
        # Place anchors
        V_0_full[anchor_idx] = V_A
        
        # Place non-anchor results
        for result in all_results:
            idx = result['indices']
            V = result['V']
            V_0_full[idx] = V
        
        # Global centering (CRITICAL for EDM consistency)
        V_0_full = V_0_full - V_0_full.mean(dim=0, keepdim=True)
        
        rank_print(f"  Final latent shape: {V_0_full.shape}")
        
        # Compute EDM
        rank_print("\nStep 7: Computing EDM...")
        G = V_0_full @ V_0_full.t()
        diag = torch.diag(G).unsqueeze(1)
        D = torch.sqrt(torch.clamp(diag + diag.t() - 2 * G, min=0))
        D_edm = uet.edm_project(D)
        
        rank_print(f"  EDM shape: {D_edm.shape}")
        
        # Prepare output
        output = {
            'latent': V_0_full.cpu(),
            'anchor_idx': anchor_idx.cpu(),
            'non_anchor_idx': non_anchor_idx.cpu(),
            'cfg': cfg,
            'args': vars(args)
        }
        
        if args.save_edm:
            output['D_edm'] = D_edm.cpu()
            rank_print("  ✓ EDM saved")
        
        # Compute coordinates if requested
        if args.save_coords:
            rank_print("\nStep 8: Computing MDS coordinates...")
            
            try:
                n = D_edm.shape[0]
                J = torch.eye(n) - torch.ones(n, n) / n
                B = -0.5 * J @ (D_edm ** 2) @ J
                coords = uet.classical_mds(B, d_out=2)
                coords_canon = uet.canonicalize_coords(coords)
                
                output['coords'] = coords.cpu()
                output['coords_canon'] = coords_canon.cpu()
                
                rank_print(f"  Coordinates shape: {coords_canon.shape}")
            except Exception as e:
                rank_print(f"  ⚠️  MDS failed: {e}")
        
        # Save output
        rank_print(f"\nSaving output to: {args.out}")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write
        temp_path = out_path.with_suffix('.pt.tmp')
        torch.save(output, temp_path)
        temp_path.rename(out_path)
        
        rank_print("  ✓ Output saved")
        
        # Print summary
        rank_print("\n" + "="*70)
        rank_print("INFERENCE COMPLETE")
        rank_print("="*70)
        rank_print(f"Output: {args.out}")
        rank_print(f"  Cells: {n_sc}")
        rank_print(f"  Anchors: {len(anchor_idx)}")
        rank_print(f"  Latent dim: {D_latent}")
        if args.save_coords:
            rank_print(f"  Coordinates: ✓")
        if args.save_edm:
            rank_print(f"  EDM: ✓")
    
    # Cleanup
    cleanup_dist()


if __name__ == '__main__':
    main()