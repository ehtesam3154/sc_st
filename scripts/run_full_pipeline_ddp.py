"""
Full GEMS Pipeline with DDP for Mouse Brain
Runs Stage A (single GPU) → Stage B → Stage C (DDP on 2 GPUs)
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

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))

# Import GEMS components
from core_models_et_p1 import SharedEncoder, STStageBPrecomputer, STSetDataset, SCSetDataset, collate_minisets, collate_sc_minisets
from core_models_et_p2 import SetEncoderContext, DiffusionScoreNet, precompute_st_prototypes, MetricSetGenerator
import utils_et as uet

# Import DDP utilities
from ddp_utils import (
    init_dist, cleanup_dist, is_main_process, get_rank, get_world_size,
    rank_print, seed_all, all_reduce_mean, broadcast_object
)


def load_mousebrain_data(data_dir: Path):
    """Load mouse brain SC and ST data."""
    rank_print("Loading mouse brain data...")
    
    scdata = pd.read_csv(data_dir / 'simu_sc_counts.csv', index_col=0).T
    scmetadata = pd.read_csv(data_dir / 'metadata.csv', index_col=0)
    stdata = pd.read_csv(data_dir / 'simu_st_counts.csv', index_col=0).T
    spcoor = pd.read_csv(data_dir / 'simu_st_metadata.csv', index_col=0)
    
    scadata = sc.AnnData(scdata, obs=scmetadata)
    sc.pp.normalize_total(scadata)
    sc.pp.log1p(scadata)
    scadata.obsm['spatial'] = scmetadata[['x_global', 'y_global']].values
    
    stadata = sc.AnnData(stdata)
    sc.pp.normalize_total(stadata)
    sc.pp.log1p(stadata)
    stadata.obsm['spatial'] = spcoor[['coord_x', 'coord_y']].values
    
    common_genes = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    n_genes = len(common_genes)
    
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
        'n_genes': n_genes
    }


def run_stage_a(data, args, device):
    """Stage A: Train shared encoder (rank 0 only)."""
    if not is_main_process():
        return None
    
    rank_print("\n" + "="*70)
    rank_print("STAGE A: Training Shared Encoder (Rank 0 only)")
    rank_print("="*70)
    
    encoder = SharedEncoder(
        n_genes=data['n_genes'],
        n_embedding=[512, 256, 128],
        dropout=0.1
    ).to(device)
    
    slide_ids = torch.zeros(data['st_expr'].shape[0], dtype=torch.long)
    
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
    
    precomputer = STStageBPrecomputer(device=device)
    slides_dict = {0: (data['st_coords'].to(device), data['st_expr'].to(device))}
    cache_dir = args.output_dir / 'stage_b_cache'
    
    targets_dict = precomputer.precompute(
        slides=slides_dict,
        encoder=encoder,
        outdir=str(cache_dir)
    )
    
    rank_print(f"✓ Stage B complete. Targets cached at: {cache_dir}")
    
    # Move to CPU for broadcasting
    rank_print("Moving targets to CPU for broadcasting...")
    for slide_id, targets in targets_dict.items():
        targets.y_hat = targets.y_hat.cpu()
        targets.G = targets.G.cpu()
        targets.D = targets.D.cpu()
        targets.H = targets.H.cpu()
        targets.H_bins = targets.H_bins.cpu()
        if hasattr(targets, 'L') and targets.L is not None:
            if targets.L.is_sparse:
                targets.L = targets.L.to_dense().cpu()
            else:
                targets.L = targets.L.cpu()
        if hasattr(targets, 'triplets') and targets.triplets is not None:
            targets.triplets = targets.triplets.cpu()
    
    return targets_dict


def run_stage_c_ddp(data, encoder, targets_dict, args, device, rank, world_size):
    """Stage C: Train diffusion generator with DDP."""
    rank_print("\n" + "="*70)
    rank_print("STAGE C: Training Diffusion Generator (DDP)")
    rank_print("="*70)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
    
    # Move targets to device
    rank_print(f"Moving targets to cuda:{rank}...")
    if targets_dict is not None:
        for slide_id, targets in targets_dict.items():
            targets.y_hat = targets.y_hat.to(device)
            targets.G = targets.G.to(device)
            targets.D = targets.D.to(device)
            targets.H = targets.H.to(device)
            targets.H_bins = targets.H_bins.to(device)
            if hasattr(targets, 'L') and targets.L is not None:
                targets.L = targets.L.to(device)
            if hasattr(targets, 'triplets') and targets.triplets is not None:
                targets.triplets = targets.triplets.to(device)
    
    torch.cuda.synchronize(device)
    rank_print("✓ Targets moved to device")
    
    # Determine AMP dtype
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True
    
    rank_print(f"AMP dtype: {amp_dtype}, using GradScaler: {use_scaler}")
    
    # Build models
    rank_print("Building models...")
    context_encoder = SetEncoderContext(
        h_dim=128, c_dim=256, n_heads=4, n_blocks=3, isab_m=64
    ).to(device)
    
    generator = MetricSetGenerator(
        c_dim=256, D_latent=args.D_latent, n_heads=4, n_blocks=2, isab_m=64
    ).to(device)
    
    score_net = DiffusionScoreNet(
        D_latent=args.D_latent, c_dim=256, time_emb_dim=128, n_blocks=4
    ).to(device)
    
    torch.cuda.synchronize(device)
    rank_print("✓ Models created")
    
    # Wrap in DDP
    if world_size > 1:
        dist.barrier()
        rank_print("Wrapping models in DDP...")
        context_encoder = DDP(context_encoder, device_ids=[rank], output_device=rank)
        generator = DDP(generator, device_ids=[rank], output_device=rank)
        score_net = DDP(score_net, device_ids=[rank], output_device=rank)
        rank_print("✓ Models wrapped in DDP")
    
    # Optimizer
    params = (
        list(context_encoder.parameters()) + 
        list(generator.parameters()) + 
        list(score_net.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Ensure encoder is on device
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Create datasets
    st_gene_expr_dict = {0: data['st_expr'].to(device)}
    
    rank_print("Creating datasets...")
    st_dataset = STSetDataset(
        targets_dict=targets_dict,
        encoder=encoder,
        st_gene_expr_dict=st_gene_expr_dict,
        n_min=args.n_min,
        n_max=args.n_max,
        D_latent=args.D_latent,
        num_samples=args.num_st_samples,
        knn_k=12,
        device=str(device)
    )
    
    sc_dataset = SCSetDataset(
        sc_gene_expr=data['sc_expr'].to(device),
        encoder=encoder,
        n_min=args.n_min,
        n_max=args.n_max,
        num_samples=args.num_sc_samples,
        device=str(device)
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
            device=str(device)
        )
        # Move to CPU for broadcasting
        prototype_bank['centroids'] = prototype_bank['centroids'].cpu()
        for proto in prototype_bank['prototypes']:
            proto['centroid_Z'] = proto['centroid_Z'].cpu()
            proto['hist'] = proto['hist'].cpu()
            proto['bins'] = proto['bins'].cpu()
            proto['D'] = proto['D'].cpu()
        rank_print("✓ Prototypes computed")
    else:
        prototype_bank = None
    
    # Broadcast prototypes
    if world_size > 1:
        rank_print("Broadcasting prototypes...")
        prototype_bank = broadcast_object(prototype_bank, src=0)
        if prototype_bank is not None:
            prototype_bank['centroids'] = prototype_bank['centroids'].to(device)
            for proto in prototype_bank['prototypes']:
                proto['centroid_Z'] = proto['centroid_Z'].to(device)
                proto['hist'] = proto['hist'].to(device)
                proto['bins'] = proto['bins'].to(device)
                proto['D'] = proto['D'].to(device)
        rank_print("✓ Prototypes received")
    
    # Create samplers and dataloaders
    st_sampler = DistributedSampler(
        st_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    sc_sampler = DistributedSampler(
        sc_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    st_loader = DataLoader(
        st_dataset, batch_size=args.batch_size,
        sampler=st_sampler, shuffle=(st_sampler is None),
        collate_fn=collate_minisets, num_workers=0, pin_memory=False
    )
    
    sc_loader = DataLoader(
        sc_dataset, batch_size=args.batch_size,
        sampler=sc_sampler, shuffle=(sc_sampler is None),
        collate_fn=collate_sc_minisets, num_workers=0, pin_memory=False
    )
    
    # Sigma schedule
    sigmas = torch.exp(torch.linspace(
        np.log(args.sigma_min), np.log(args.sigma_max),
        args.n_timesteps, device=device
    ))
    
    # Loss modules
    gram_loss = uet.FrobeniusGramLoss()
    
    # Training loop
    rank_print(f"\nStarting Stage C training for {args.n_epochs} epochs...")
    
    for epoch in range(args.n_epochs):
        if world_size > 1:
            st_sampler.set_epoch(epoch)
            sc_sampler.set_epoch(epoch)
        
        context_encoder.train()
        generator.train()
        score_net.train()
        
        epoch_losses = []
        
        st_iter = iter(st_loader)
        sc_iter = iter(sc_loader)
        
        for batch_idx in range(max(len(st_loader), len(sc_loader))):
            # Get batch (alternate ST/SC)
            is_st = (batch_idx % 2 == 0) and (batch_idx // 2 < len(st_loader))
            
            try:
                if is_st:
                    batch = next(st_iter)
                else:
                    batch = next(sc_iter)
            except StopIteration:
                try:
                    batch = next(sc_iter) if is_st else next(st_iter)
                except StopIteration:
                    break
            
            # Move batch to device
            Z_set = batch['Z_set'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP (use torch.amp instead of torch.cuda.amp)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                # Context encoding
                H = context_encoder(Z_set, mask)
                
                # Generate V_0
                V_0 = generator(H, mask)
                
                # Sample timestep
                batch_size_real = Z_set.shape[0]
                t_idx = torch.randint(0, args.n_timesteps, (batch_size_real,), device=device)
                t_norm = t_idx.float() / args.n_timesteps
                sigma_t = sigmas[t_idx].view(-1, 1, 1)
                
                # Add noise
                eps = torch.randn_like(V_0)
                V_t = V_0 + sigma_t * eps
                V_t = V_t * mask.unsqueeze(-1).float()
                
                # Predict noise
                eps_pred = score_net(V_t, t_norm.unsqueeze(1), H, mask)
                
                # Score loss
                loss_score = ((eps_pred - eps) ** 2 * mask.unsqueeze(-1).float()).sum()
                loss_score = loss_score / (mask.sum() * V_0.shape[-1])
                
                # Gram loss (only for ST batches)
                loss_gram = torch.tensor(0.0, device=device)
                if is_st and 'G_target' in batch:
                    V_pred = V_t - sigma_t * eps_pred
                    G_target = batch['G_target'].to(device)
                    loss_gram = gram_loss(V_pred, G_target, mask)
                
                # Total loss
                loss = loss_score + 0.5 * loss_gram
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Aggregate loss
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if world_size > 1:
            avg_loss = all_reduce_mean(avg_loss)
        
        if is_main_process():
            rank_print(f"Epoch {epoch+1}/{args.n_epochs} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if is_main_process() and ((epoch + 1) == args.n_epochs or (epoch + 1) % 50 == 0):
            ckpt_dir = args.output_dir / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            
            if world_size > 1:
                context_state = context_encoder.module.state_dict()
                generator_state = generator.module.state_dict()
                score_state = score_net.module.state_dict()
            else:
                context_state = context_encoder.state_dict()
                generator_state = generator.state_dict()
                score_state = score_net.state_dict()
            
            ckpt = {
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'context_encoder': context_state,
                'generator': generator_state,
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
        
        if world_size > 1:
            dist.barrier()
    
    rank_print("✓ Stage C complete")
    
    return context_encoder, score_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'stagec_only'], default='full')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/ehtesamul/sc_st/data/mousedata_2020/E1z2')
    parser.add_argument('--output_dir', type=str,
                        default='./gems_mousebrain_ddp_output')
    
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
    parser.add_argument('--n_epochs_stageA', type=int, default=1000)
    parser.add_argument('--D_latent', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.data_dir = Path(args.data_dir)
    
    # Initialize distributed
    rank, local_rank, world_size, device = init_dist()
    print(f"Rank {rank}/{world_size-1} on cuda:{local_rank} | seed={args.seed+rank}")
    
    seed_all(args.seed, rank)
    data = load_mousebrain_data(args.data_dir)
    
    if is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    if args.mode == 'full':
        encoder = run_stage_a(data, args, device)
        
        if world_size > 1:
            dist.barrier()
        
        if not is_main_process():
            encoder = SharedEncoder(
                n_genes=data['n_genes'],
                n_embedding=[512, 256, 128]
            ).to(device)
            encoder_path = args.output_dir / 'encoder_stageA.pt'
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        
        encoder.eval()
        
        targets_dict = run_stage_b(data, encoder, args, device)
        
        if world_size > 1:
            rank_print("Broadcasting Stage B targets from rank 0...")
            targets_dict = broadcast_object(targets_dict, src=0)
            rank_print(f"  Rank {rank} received targets")
    
    else:
        rank_print("Loading encoder and targets from previous run...")
        encoder = SharedEncoder(
            n_genes=data['n_genes'],
            n_embedding=[512, 256, 128]
        ).to(device)
        encoder_path = args.output_dir / 'encoder_stageA.pt'
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.eval()
        
        cache_dir = args.output_dir / 'stage_b_cache'
        targets_0 = torch.load(cache_dir / 'slide_0.pt', map_location='cpu')
        targets_dict = {0: targets_0}
    
    context_encoder, score_net = run_stage_c_ddp(
        data, encoder, targets_dict, args, device, rank, world_size
    )
    
    rank_print("\n" + "="*70)
    rank_print("PIPELINE COMPLETE!")
    rank_print("="*70)
    
    cleanup_dist()


if __name__ == '__main__':
    main()