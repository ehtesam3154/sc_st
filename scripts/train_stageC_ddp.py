'''
stage C diffusion training with pytorch ddp
distributed training script for GEMS model across 2 gpus (i have 2 rn)

usage:
    torchrun --standalone --nproc_per_node=2 train_stageC_ddp.py --cfg configs/mousebrain_stageC_ddp.eval
'''

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from datetime import datetime

#add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

#import GEMS checkpoint
from core_models_et_p1 import SharedEncoder, STSetDataset, SCSetDataset, collate_minisets, collate_sc_minisets
from core_models_et_p2 import SetEncoderContext, DiffusionScoreNet, precompute_st_prototypes
from core_models_et_p3 import GEMSModel
import utils_et as uet

#import DDP utilities 
from ddp_utils import (
    init_dist, cleanup_dist, is_main_process, get_rank, get_world_size,
    rank_print, seed_all, rank_tqdm, all_reduce_mean
)
from pairwise_shard import is_sharding_beneficial

def load_config(cfg_path: str, cli_overrides: Dict[str, any]) -> Dict[str, Any]:
    '''load YAML config and apply CLI overrides'''
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    #apply CLI overrides (simple nested dict update)
    for key, value in cli_overrides.item():
        if '.' in key:
            #handle nested keys like 'training.lr'
            parts = key.split('.')
            d = cfg
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        else:
            cfg[key] = value

    return cfg

def load_stage_c_data(cfg: Dict[str, Any], device=torch.device):
    '''
    load stage C training data from .pt file or h5ad files
    
    returns:
        (st_gene_expr_dict, sc_gene_expr, targets_dict)
    '''

    rank_print('loading stage C data......')

    #check for precomputed .pt file first
    if cfg['data'].get('stagec_pt') and Path(cfg['data']['stagec_pt']).exists():
        rank_print(f'loading from: {cfg['data']['stagec_pt']}')
        data = torch.load(cfg['data']['stagec_pt'], map_location='cpu')

        st_gene_expr_dict = data['st_gene_expr_dict']
        sc_gene_expr = data['sc_gene_expr']
        targets_dict = data['targets_dict']

        rank_print(f' SC expression: {sc_gene_expr.shape}')
        rank_print(f' ST slides: {list(st_gene_expr_dict.keys())}')

        return st_gene_expr_dict, sc_gene_expr, targets_dict
    
    else:
        raise NotImplementedError(
            "h5ad fallback not implemented. Please save data as .pt file:\n"
            "  torch.save({'st_gene_expr_dict': ..., 'sc_gene_expr': ..., 'targets_dict': ...}, 'stagec_inputs.pt')"
        )
    
def build_models(cfg: Dict[str, Any], device: torch.device, rank: int):
    '''build encoder, context_encoder, generator, and score_net.'''
    rank_print('building models.....')

    #encoder (frozen - loaded from checkpoint)
    encoder = SharedEncoder(
        n_genes= cfg['model']['n_genes'],
        n_embedding=cfg['model']['n_embedding']
    ).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    #context encoder
    context_encoder = SetEncoderContext(
        h_dim = cfg['model']['n_embedding'][-1],
        c_dim = cfg['model']['c_dim'],
        n_heads = cfg['model']['n_heads'],
        n_blocks = 3,
        isab_m = cfg['model']['isab_m']
    ).to(device)

    #score net
    score_net = DiffusionScoreNet(
        D_latent=cfg['model']['D_latent'],
        c_dim = cfg['model']['c_dim'],
        time_emb_dim=128,
        n_blocks=4
    ).to(device)

    rank_print(f"  Encoder: frozen, {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
    rank_print(f"  Context Encoder: {sum(p.numel() for p in context_encoder.parameters())/1e6:.2f}M params")
    rank_print(f"  Score Net: {sum(p.numel() for p in score_net.parameters())/1e6:.2f}M params")
    
    return encoder, context_encoder, score_net

def wrap_ddp(model: nn.Module, device: torch.device, local_rank: int) -> nn.Module:
    '''
    wrap model in DDP if distributed
    '''
    if get_world_size > 1:
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
def load_checkpoint(
        cfg: Dict[str, Any],
        encoder: nn.Module,
        context_encoder: nn.Module,
        score_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device
) -> int:
    '''
    load checkpoint if resume path is provided. returns start_epoch
    '''

    resume_path = cfg['checkpoint'].get['resume']

    if resume_path and Path(resume_path).exists():
        rank_print(f'resuming from checkpoint: [resume_path]')
        ckpt = torch.load(resume_path, map_location=device)

        #load model states (handle DDP wrapper)
        encoder.load_state_dict(ckpt['encoder'])

        if get_world_size() > 1:
            context_encoder.module.load_state_dict(ckpt['context_encoder'])
            score_net.module.load_state_dict(ckpt['score_net'])
        else:
            context_encoder.load_state_dict(ckpt['context_encoder'])
            score_net.load_state_dict(ckpt['score_net'])

        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt['score_net'])

        start_epoch = ckpt.get('epoch', 0) + 1
        rank_print(f'resuming from epoch {start_epoch}')
        return start_epoch
    
    return 0


def save_checkpoint(
        cfg: Dict[str, Any],
        epoch: int,
        encoder: nn.Module,
        context_encoder: nn.Module,
        score_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler]
):
    '''
    save checkpoint (rank 0 only)
    '''
    if not is_main_process():
        return
    
    out_dir = Path(cfg['checkpoint']['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    #prepare state dict (unwrap ddp)
    if get_world_size() > 1:
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
        'cfg': cfg
    }

    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()

    #atomic write
    temp_path = out_dir / f'checkpoint_epoch{epoch:04d}.pt.tmp'
    final_path = out_dir / f'checkpoint_epoch{epoch:04d}.pt'

    torch.save(ckpt, temp_path)
    temp_path.rename(final_path)

    rank_print(f' checkpoint saved: {final_path}')

    #clean up old checkpoints
    keep_last = cfg['checkpoint'].get('keep_last', 5)
    checkpoints = sorted(out_dir.glob('checkpoint_epoch*.pt'))
    if len(checkpoints) > keep_last:
        for old_ckpt in checkpoints[:-keep_last]:
            old_ckpt.unlink()

def compute_losses_with_optional_sharding(
        V_pred: torch.Tensor,
        targets: Dict[str, Any],
        mask: torch.Tensor,
        cfg: Dict[str, Any],
        loss_modules: Dict[str, Any],
        rank: int,
        world_size: int
) -> Dict[str, torch.Tensor]:
    ''' 
    compute all losses with optional O(N^2) sharding for large mini-sets
    
    Falls back to original implementation for small N or single GPU
    '''

    n = V_pred.shape[1]
    losses = {}

    #check if sharding should be used
    use_sharding = (
        cfg['pairwise_shard']['enable'] and
        is_sharding_beneficial(
            n,
            world_size,
            cfg['pairwise_shard']['threshold_N']
        )
    )

    #gram loss
    if 'G_target' in targets:
        if use_sharding:
            from model.pairwise_shard import sharded_gram_loss
            losses['gram'] = sharded_gram_loss(
                V_pred,
                targets['G_target'],
                mask,
                rank,
                world_size,
                block_size = cfg['pairwise_shard']['threshold_N']
            )

        else:
            losses['gram'] = loss_modules['gram'](V_pred, targets['G_target'], mask)

        return losses

def train_one_epoch(
        epoch: int,
        st_loader: DataLoader,
        sc_loader: DataLoader,
        encoder: nn.Module,
        context_encoder: nn.Module,
        score_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        cfg: Dict[str, Any],
        loss_modules: Dict[str, Any],
        sigmas: torch.Tensor,
        prototype_bank: Dict[str, Any],
        device: torch.device,
        rank: int,
        world_size: int,
        amp_dtype: torch.dtype
) -> Dict[str, float]:
    '''
    train for one epoch with mixed ST/SC batches
    '''
    context_encoder.train()
    score_net.train()

    #combine loaders
    st_iter = iter(st_loader)
    sc_iter = iter(sc_loader)

    total_batches = len(st_loader) + len(sc_loader)

    #metrics
    metrics = {
        'loss_total': 0.0,
        'loss_score': 0.0,
        'loss_gram': 0.0,
        'loss_heat': 0.0,
        'loss_sw': 0.0,
        'loss_triplet': 0.0,
        'n_st': 0,
        'n_sc': 0
    }

    #progress bar
    pbar = rank_tqdm(
        range(total_batches),
        desc=f'epoch {epoch+1}/{cfg['training']['n_epochs']}'
    )

    for batch_idx in pbar:
        #alternate st/sc
        is_st = (batch_idx % 2 == 0) and (batch_idx // 2 < len(st_loader))

        try:
            if is_st:
                batch = next(st_iter)
                batch_type = 'ST'
            else:
                batch = next(sc_iter)
                batch_type = 'SC'
        except StopIteration:
            #one loader exhausted, drain the other
            try:
                if is_st:
                    batch = next(sc_iter)
                    batch_type = 'SC'
                else:
                    batch = next(st_iter)
                    batch_type = 'ST'
            except StopIteration:
                break

        #move to device
        Z_set = batch['Z_set'].to(device)
        V_target = batch['Z_target'].to(device)
        mask = batch['mask'].to(device)
        n = batch['n']

        #move targets
        targets ={}
        for key in ['G_target', 'D_target', 'H_target', 'H_bins', 'triplets']:
            if key in batch:
                targets[key] = batch[key].to(device)
        if 'L_info' in batch:
            L = batch['L_info']['L'].to(device)
            targets['L_info'] = {'L': L, 't_list': batch['L_info']['t_list']}

        #sample timestep
        batch_size = Z_set.shape[0]
        t_idx = torch.randint(0, len(sigmas), (batch_size,), device=device)
        sigma_t = sigmas[t_idx].view(-1, 1, 1)

        #add noise
        noise = torch.randn_like(V_target)
        V_noisy = V_target + sigma_t * noise 

        #forward pass with AMP
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            #context encoding
            H = context_encoder(Z_set, mask)

            #score prediction
            t_normalized = t_idx.float() / (len(sigmas) - 1)
            t_normalized = t_normalized.view(-1, 1)
            eps_pred = score_net(V_noisy, t_normalized, H, mask)

            #score matching loss
            loss_score = ((eps_pred - noise) ** 2 * mask.unsqueeze(-1).float()).sum()
            loss_score = loss_score / (mask.sum() * V_target.shape[-1])

            #denoised prediction for other losses
            V_pred = V_noisy - sigma_t * eps_pred

            #compute geometric losses
            geom_losses = compute_losses_with_optional_sharding(
                V_pred, targets, mask, cfg, loss_modules, rank, world_size
            )

            #total size
            weights = cfg['training']['loss_weights']
            loss = weights['score'] * loss_score

            for key, val in geom_losses.items():
                if key == 'gram':
                    loss += weights['gram'] * val
                elif key == 'heat':
                    loss += weights['heat'] * val
                elif key == 'sw_st':
                    loss += weights['sw_st'] * val if batch_type =='ST' else weights.get('sw_sc', 0.3) * val
                elif key == 'triplet':
                    loss += weights.get('ordinal_sc', 0.5) * val if batch_type == 'SC' else 0.0

        #backward
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        #update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item(): .4f}',
            'type': batch_type
        })

    #average metrics
    n_batches = metrics['n_st'] + metrics['n_sc']
    for key in metrics:
        if key.startswith('loss'):
            metrics[key] /= max(n_batches, 1)

    return metrics

def main():
    parser = argparse.ArgumentParser(description='stage C DDP training')
    parser.add_argument('--cfg', dtype=str, required=True, help='path to config YAML')
    parser.add_argument('--batch_size', type=int, help='override batch size')
    parser.add_argument('--n_epochs', dtype=int, help='override number of epochs')
    parser.add_argument('--lr', dtype=float, help='override learning rate')
    parser.add_argument('--resume', dtype=str, help='override resume checkpoint path')

    args, unknown = parser.parse_known_args()

    #parse unknown args as overrides (e.g --training.lr 1e-3)
    cli_overrides = {}
    if args.batch_size:
        cli_overrides['training.batch_size'] = args.batch_size
    if args.n_epochs:
        cli_overrides['training.n_epochs'] = args.n_epochs
    if args.lr:
        cli_overrides['training.lr'] = args.lr
    if args.resume:
        cli_overrides['checkpoint.resume'] = args.resume

    #load config
    cfg = load_config(args.cfg, cli_overrides)

    #init distributed
    rank, local_rank, world_size, device= init_dist()

    #print startup info
    if world_size > 1:
        print(f'rank {rank}/{world_size-1} on cuda: {local_rank} | seed={cfg['device']['seed'] + rank}')
    else:
        rank_print(f'single-GPU mode on {device}')

    #seed
    seed_all(cfg['device']['seed'], rank)

    #determine amp dtype
    amp_dtype_str = cfg['device']['amp_dtype']
    if amp_dtype_str == 'auto':
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif amp_dtype_str == 'bf16':
        amp_dtype = torch.bfloat16
    elif amp_dtype_str == 'fp16':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    use_scaler = (amp_dtype == torch.float16)
    rank_print(f'AMP dtype: {amp_dtype}, using GradScaler: {use_scaler}')

    #enable tf32 for ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backen.cudnn.allow_tf32 = True

    #load data
    st_gene_expr_dict, sc_gene_expr, targets_dict = load_stage_c_data(cfg, device)

    #build models
    encoder, context_encoder, score_net = build_models(cfg, device, rank)

    #load encoder weights (stage A checkpoint)
    encoder_ckpt = Path(cfg['checkpoint']['out_dir']).parent / 'encoder_stageA.pt'
    if encoder_ckpt.exists():
        rank_print(f'loading encoder from: {encoder_ckpt}')
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    else:
        rank_print('ENCODER checkpoint not found, using random weights')

    #wrap in ddp
    context_encoder = wrap_ddp(context_encoder, device, local_rank)
    score_net = wrap_ddp(score_net, device, local_rank)

    #optimizer
    params = list(context_encoder.parameters()) + list(score_net.parameters)
    optimizer = torch.optim.AdamW(
        params,
        lr = cfg['training']['lr'],
        weight_decay = cfg['training']['weight_decay']
    )

    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint(cfg, encoder, context_encoder, score_net, optimizer, scaler, device)
    
    # Create datasets
    rank_print("Creating datasets...")
    
    st_dataset = STSetDataset(
        targets_dict=targets_dict,
        encoder=encoder,
        st_gene_expr_dict=st_gene_expr_dict,
        n_min=cfg['dataset']['n_min'],
        n_max=cfg['dataset']['n_max'],
        D_latent=cfg['model']['D_latent'],
        num_samples=cfg['dataset']['num_st_samples'],
        knn_k=cfg['dataset']['knn_k'],
        device=device
    )
    
    sc_dataset = SCSetDataset(
        sc_gene_expr=sc_gene_expr,
        encoder=encoder,
        n_min=cfg['dataset']['n_min'],
        n_max=cfg['dataset']['n_max'],
        num_samples=cfg['dataset']['num_sc_samples'],
        device=device
    )
    
    rank_print(f"  ST dataset: {len(st_dataset)} samples")
    rank_print(f"  SC dataset: {len(sc_dataset)} samples")
    
    # Precompute prototypes (rank 0 only, then broadcast)
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
    
    # Broadcast prototype bank
    if world_size > 1:
        from model.ddp_utils import broadcast_object
        prototype_bank = broadcast_object(prototype_bank, src=0)
    
    # Create samplers and loaders
    st_sampler = DistributedSampler(st_dataset, shuffle=True) if world_size > 1 else None
    sc_sampler = DistributedSampler(sc_dataset, shuffle=True) if world_size > 1 else None
    
    st_loader = DataLoader(
        st_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=st_sampler,
        shuffle=(st_sampler is None),
        collate_fn=collate_minisets,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    sc_loader = DataLoader(
        sc_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=sc_sampler,
        shuffle=(sc_sampler is None),
        collate_fn=collate_sc_minisets,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Sigma schedule
    sigmas = torch.exp(torch.linspace(
        np.log(cfg['training']['sigma_min']),
        np.log(cfg['training']['sigma_max']),
        cfg['training']['n_timesteps'],
        device=device
    ))
    
    # Loss modules
    loss_modules = {
        'gram': uet.FrobeniusGramLoss(),
        'heat': uet.HeatKernelLoss(
            use_hutchinson=True,
            num_probes=8,
            chebyshev_degree=10,
            knn_k=8,
            t_list=(0.5, 1.0),
            laplacian='sym'
        ),
        'sw': uet.SlicedWassersteinLoss1D(),
        'triplet': uet.OrdinalTripletLoss()
    }
    
    # Training loop
    rank_print("\n" + "="*70)
    rank_print("STARTING STAGE C TRAINING (DDP)")
    rank_print("="*70)
    
    for epoch in range(start_epoch, cfg['training']['n_epochs']):
        # Set epoch for samplers
        if st_sampler:
            st_sampler.set_epoch(epoch)
        if sc_sampler:
            sc_sampler.set_epoch(epoch)
        
        # Train one epoch
        metrics = train_one_epoch(
            epoch, st_loader, sc_loader,
            encoder, context_encoder, score_net,
            optimizer, scaler, cfg, loss_modules,
            sigmas, prototype_bank, device,
            rank, world_size, amp_dtype
        )
        
        # Print summary on rank 0
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{cfg['training']['n_epochs']} Summary:")
            print(f"  Loss: {metrics['loss_total']:.4f}")
            print(f"  Score: {metrics['loss_score']:.4f}")
            print(f"  Gram: {metrics['loss_gram']:.4f}")
            print(f"  Heat: {metrics['loss_heat']:.4f}")
            print(f"  ST batches: {metrics['n_st']}, SC batches: {metrics['n_sc']}")
        
        # Save checkpoint
        if (epoch + 1) % cfg['checkpoint']['save_every'] == 0:
            save_checkpoint(cfg, epoch, encoder, context_encoder, score_net, optimizer, scaler)
    
    # Final save
    save_checkpoint(cfg, cfg['training']['n_epochs'] - 1, encoder, context_encoder, score_net, optimizer, scaler)
    
    rank_print("\n" + "="*70)
    rank_print("TRAINING COMPLETE")
    rank_print("="*70)
    
    cleanup_dist()


if __name__ == '__main__':
    main()








    
