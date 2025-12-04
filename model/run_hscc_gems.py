import os
import torch
from lightning.fabric import Fabric
import scanpy as sc
from core_models_et_p3 import GEMSModel
import argparse
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import utils_et as uet
import torch.distributed as dist
from scipy.ndimage import gaussian_filter1d
import os





def ensure_disk_space(path, min_gb=10.0):
    """Check free space, raise error if below threshold."""
    path = Path(path)
    if not path.exists():
        path = Path.home()

    stat = shutil.disk_usage(str(path))
    avail_gb = stat.free / (1024**3)

    print(f"[disk] Available space on {path.anchor or path}: {avail_gb:.1f} GB")

    if avail_gb < min_gb:
        raise RuntimeError(
            f"Only {avail_gb:.1f} GB free on filesystem for {path}. "
            f"Please delete old checkpoints/plots/logs before running inference."
        )
    
def parse_args():
    parser = argparse.ArgumentParser(description='GEMS Training with Lightning Fabric - hSCC 2 Slides')
    
    # Training config
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--stageA_epochs', type=int, default=1000)
    parser.add_argument('--stageC_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='gems_hscc_2slides_output')
    parser.add_argument('--num_st_samples', type=int, default=4000)
    parser.add_argument('--num_sc_samples', type=int, default=9000)
    
    # Geometry-aware diffusion parameters
    parser.add_argument('--use_canonicalize', action='store_true', default=True)
    parser.add_argument('--use_dist_bias', action='store_true', default=True)
    parser.add_argument('--dist_bins', type=int, default=24)
    parser.add_argument('--dist_head_shared', action='store_true', default=True)
    parser.add_argument('--use_angle_features', action='store_true', default=True)
    parser.add_argument('--angle_bins', type=int, default=8)
    parser.add_argument('--knn_k', type=int, default=12)
    parser.add_argument('--self_conditioning', action='store_true', default=True)
    parser.add_argument('--sc_feat_mode', type=str, default='concat', choices=['concat', 'mlp'])
    parser.add_argument('--landmarks_L', type=int, default=16)

    # Early stopping
    parser.add_argument('--enable_early_stop', action='store_true', default=False,
                        help='Enable early stopping')
    parser.add_argument('--early_stop_min_epochs', type=int, default=12,
                        help='Minimum epochs before early stopping kicks in')
    parser.add_argument('--early_stop_patience', type=int, default=6,
                        help='Epochs to wait without improvement before stopping')
    parser.add_argument('--early_stop_threshold', type=float, default=0.01,
                        help='Relative improvement threshold (e.g., 0.01 = 1%)')

    parser.add_argument('--sc_finetune_epochs', type=int, default=None, 
                        help='SC fine-tune epochs (default: auto = 50%% of ST best epoch, clamped [10,50])')

    
    return parser.parse_args()

def load_hscc_data():
    """
    Load hSCC data: 2 ST slides for training, 1 ST slide for inference.
    Follows hSCC_gems.ipynb data loading structure.
    """
    print("Loading hSCC data...")
    
    # Load SC data (used as test data, like ST2 in mouse_brain)
    scadata = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP2.h5ad')
    
    # Load 3 ST datasets (use first 2 for training, 3rd for inference)
    stadata1 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2.h5ad')
    stadata2 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep2.h5ad')
    stadata3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep3.h5ad')
    
    # Normalize and log transform
    print("Normalizing data...")
    for adata in [scadata, stadata1, stadata2, stadata3]:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    
    # Create rough cell types for SC data (from hSCC_gems.ipynb)
    print("Creating rough cell types...")
    scadata.obs['celltype'] = scadata.obs['level1_celltype'].astype(str)
    scadata.obs.loc[scadata.obs['level1_celltype']=='CLEC9A', 'celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='CD1C', 'celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='ASDC', 'celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='PDC', 'celltype'] = 'PDC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='MDSC', 'celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='LC', 'celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='Mac', 'celltype'] = 'Myeloid cell'
    scadata.obs.loc[scadata.obs['level1_celltype']=='Tcell', 'celltype'] = 'T cell'
    scadata.obs.loc[scadata.obs['level2_celltype']=='TSK', 'celltype'] = 'TSK'
    scadata.obs.loc[scadata.obs['level2_celltype'].isin(['Tumor_KC_Basal', 'Tumor_KC_Diff', 'Tumor_KC_Cyc']), 'celltype'] = 'NonTSK'
    
    print(f"SC data loaded: {scadata.shape[0]} cells, {scadata.shape[1]} genes")
    print(f"ST slide 1: {stadata1.shape[0]} spots, {stadata1.shape[1]} genes")
    print(f"ST slide 2: {stadata2.shape[0]} spots, {stadata2.shape[1]} genes")
    print(f"ST slide 3 (inference): {stadata3.shape[0]} spots, {stadata3.shape[1]} genes")
    
    return scadata, stadata1, stadata2, stadata3


def main(args=None):
    # Parse args if not provided
    if args is None:
        args = parse_args()
    
    # Extract parameters
    devices = args.devices
    precision = args.precision
    stageA_epochs = args.stageA_epochs
    stageB_outdir = "gems_stageB_cache"
    stageC_epochs = args.stageC_epochs
    stageC_batch = args.batch_size
    lr = args.lr
    outdir = args.outdir

    fabric = Fabric(accelerator="gpu", devices=devices, strategy="ddp", precision=precision)
    fabric.launch()

    # ---------- Load data on all ranks (OK) ----------
    scadata, stadata1, stadata2, stadata3 = load_hscc_data()

    # ---------- Get common genes ----------
    common = sorted(list(set(scadata.var_names) & set(stadata1.var_names) & 
                         set(stadata2.var_names) & set(stadata3.var_names)))
    n_genes = len(common)
    
    if fabric.is_global_zero:
        print(f"Common genes across all datasets: {n_genes}")

    # ---------- Build model (same across ranks) ----------
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=32,
        c_dim=256,
        n_heads=4,
        isab_m=64,
        device=str(fabric.device),
        use_canonicalize=args.use_canonicalize,
        use_dist_bias=args.use_dist_bias,
        dist_bins=args.dist_bins,
        dist_head_shared=args.dist_head_shared,
        use_angle_features=args.use_angle_features,
        angle_bins=args.angle_bins,
        knn_k=args.knn_k,
        self_conditioning=args.self_conditioning,
        sc_feat_mode=args.sc_feat_mode,
        landmarks_L=args.landmarks_L,
    )

    # ---------- Stage A & B on rank-0 only ----------
    if fabric.is_global_zero:
        # Extract tensors
        
        # SC expression
        X_sc = scadata[:, common].X
        if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
        sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
        
        # ST expression from 2 training slides (stadata1, stadata2)
        X_st1 = stadata1[:, common].X
        X_st2 = stadata2[:, common].X
        if hasattr(X_st1, "toarray"): X_st1 = X_st1.toarray()
        if hasattr(X_st2, "toarray"): X_st2 = X_st2.toarray()
        
        # Combine training ST data
        st_expr = torch.tensor(np.vstack([X_st1, X_st2]), dtype=torch.float32, device=fabric.device)
        
        # ST coordinates from 2 training slides
        st_coords1 = stadata1.obsm['spatial']
        st_coords2 = stadata2.obsm['spatial']
        st_coords_raw = torch.tensor(np.vstack([st_coords1, st_coords2]), 
                                     dtype=torch.float32, device=fabric.device)
        
        # Slide IDs (0 for slide1, 1 for slide2)
        slide_ids = torch.tensor(
            np.concatenate([
                np.zeros(X_st1.shape[0], dtype=int),
                np.ones(X_st2.shape[0], dtype=int)
            ]),
            dtype=torch.long, device=fabric.device
        )
        
        # Per-slide canonicalization
        st_coords, st_mu, st_scale = uet.canonicalize_st_coords_per_slide(
            st_coords_raw, slide_ids
        )

        # ========== DEBUG CODE START ==========
        print("\n" + "="*60)
        print("COORDINATE NORMALIZATION DEBUG")
        print("="*60)
        print(f"Raw coords shape: {st_coords_raw.shape}")
        print(f"Raw coords stats:")
        print(f"  mean: {st_coords_raw.mean(dim=0).cpu().numpy()}")
        print(f"  std:  {st_coords_raw.std(dim=0).cpu().numpy()}")
        print(f"  min:  {st_coords_raw.min(dim=0)[0].cpu().numpy()}")
        print(f"  max:  {st_coords_raw.max(dim=0)[0].cpu().numpy()}")

        centered_raw = st_coords_raw - st_coords_raw.mean(dim=0)
        rms_raw = centered_raw.pow(2).sum(dim=1).mean().sqrt().item()
        print(f"  RMS radius: {rms_raw:.4f}")

        print(f"\nAfter canonicalize_st_coords_per_slide:")
        print(f"  st_mu: {st_mu.cpu().numpy()}")
        print(f"  st_scale: {st_scale.cpu().numpy()}")
        print(f"\nCanonical coords stats:")
        print(f"  mean: {st_coords.mean(dim=0).cpu().numpy()}")
        print(f"  std:  {st_coords.std(dim=0).cpu().numpy()}")
        print(f"  min:  {st_coords.min(dim=0)[0].cpu().numpy()}")
        print(f"  max:  {st_coords.max(dim=0)[0].cpu().numpy()}")

        rms_canon = st_coords.pow(2).sum(dim=1).mean().sqrt().item()
        print(f"  RMS radius: {rms_canon:.4f} (should be ~1.0 if unit RMS)")

        # Check pairwise distances
        D = torch.cdist(st_coords, st_coords)
        triu_mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        D_upper = D[triu_mask]
        print(f"\nPairwise distances after canonicalization:")
        print(f"  p50: {D_upper.quantile(0.50).item():.6f}")
        print(f"  p90: {D_upper.quantile(0.90).item():.6f}")
        print(f"  p95: {D_upper.quantile(0.95).item():.6f}")
        print(f"  max: {D_upper.max().item():.6f}")
        print("="*60 + "\n")
        # ========== DEBUG CODE END ==========
        
        # Save canonicalization stats for later denormalization
        os.makedirs(outdir, exist_ok=True)
        torch.save(
            {"mu": st_mu.cpu(), "scale": st_scale.cpu()},
            os.path.join(outdir, "st_slide_canon_stats.pt"),
        )
        print(f"[Rank-0] Per-slide canonicalization: {len(torch.unique(slide_ids))} slides")

        print("\n=== Stage A (single GPU, rank-0) ===")
        model.train_stageA(
            st_gene_expr=st_expr,
            st_coords=st_coords,
            sc_gene_expr=sc_expr,
            slide_ids=slide_ids,
            n_epochs=stageA_epochs,
            batch_size=256,
            lr=1e-4,
            sigma=0.75,
            alpha=0.8,
            ratio_start=0.0,
            ratio_end=1.0,
            mmdbatch=1000,
            outf=outdir,
        )

        print("\n=== Stage B (single GPU, rank-0) ===")
        # Create slides_dict for 2 training slides
        st1_coords_canon = st_coords[slide_ids == 0]
        st1_expr = st_expr[slide_ids == 0]
        st2_coords_canon = st_coords[slide_ids == 1]
        st2_expr = st_expr[slide_ids == 1]
        
        slides_dict = {
            0: (st1_coords_canon, st1_expr),
            1: (st2_coords_canon, st2_expr)
        }
        
        model.train_stageB(
            slides=slides_dict,
            outdir=stageB_outdir,
        )

        # Save a checkpoint after Stage A/B for other ranks to load
        ckpt_ab = {
            "encoder": model.encoder.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "score_net": model.score_net.state_dict(),
        }
        os.makedirs(outdir, exist_ok=True)
        torch.save(ckpt_ab, os.path.join(outdir, "ab_init.pt"))

    # Sync and make sure all ranks see the Stage A/B weights
    fabric.barrier()
    if not fabric.is_global_zero:
        # Load A/B weights
        path = os.path.join(outdir, "ab_init.pt")
        ck = torch.load(path, map_location=fabric.device)
        model.encoder.load_state_dict(ck["encoder"])
        model.context_encoder.load_state_dict(ck["context_encoder"])
        model.generator.load_state_dict(ck["generator"])
        model.score_net.load_state_dict(ck["score_net"])

        # CRITICAL: Non-rank-0 processes need to recompute Stage B
        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")
        
        # SC expression
        X_sc = scadata[:, common].X
        if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
        
        # ST expression from 2 training slides
        X_st1 = stadata1[:, common].X
        X_st2 = stadata2[:, common].X
        if hasattr(X_st1, "toarray"): X_st1 = X_st1.toarray()
        if hasattr(X_st2, "toarray"): X_st2 = X_st2.toarray()
        
        st_expr_rank = torch.tensor(np.vstack([X_st1, X_st2]), dtype=torch.float32)
        
        # ST coordinates
        st_coords1 = stadata1.obsm['spatial']
        st_coords2 = stadata2.obsm['spatial']
        st_coords_raw_rank = torch.tensor(np.vstack([st_coords1, st_coords2]), dtype=torch.float32)
        
        slide_ids_rank = torch.tensor(
            np.concatenate([
                np.zeros(X_st1.shape[0], dtype=int),
                np.ones(X_st2.shape[0], dtype=int)
            ]),
            dtype=torch.long
        )
        
        st_coords_rank, _, _ = uet.canonicalize_st_coords_per_slide(
            st_coords_raw_rank, slide_ids_rank
        )
        print(f"[Rank {fabric.global_rank}] Applied per-slide canonicalization")
        
        st1_coords_canon = st_coords_rank[slide_ids_rank == 0]
        st1_expr = st_expr_rank[slide_ids_rank == 0]
        st2_coords_canon = st_coords_rank[slide_ids_rank == 1]
        st2_expr = st_expr_rank[slide_ids_rank == 1]
        
        slides_dict_rank = {
            0: (st1_coords_canon, st1_expr),
            1: (st2_coords_canon, st2_expr)
        }

        model.train_stageB(
            slides=slides_dict_rank,
            outdir=stageB_outdir,
        )

    # ---------- Stage C (multi-GPU with Fabric) ----------
    print("\n=== Stage C (DDP across GPUs) ===")
    # Rebuild tensors on each rank (cheap)
    
    # SC expression
    X_sc = scadata[:, common].X
    if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
    sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
    
    # ST expression from 2 training slides
    X_st1 = stadata1[:, common].X
    X_st2 = stadata2[:, common].X
    if hasattr(X_st1, "toarray"): X_st1 = X_st1.toarray()
    if hasattr(X_st2, "toarray"): X_st2 = X_st2.toarray()
    
    st_expr = torch.tensor(np.vstack([X_st1, X_st2]), dtype=torch.float32, device=fabric.device)
    
    # ST coordinates
    st_coords1 = stadata1.obsm['spatial']
    st_coords2 = stadata2.obsm['spatial']
    st_coords_raw = torch.tensor(np.vstack([st_coords1, st_coords2]), 
                                 dtype=torch.float32, device=fabric.device)
    
    slide_ids = torch.tensor(
        np.concatenate([
            np.zeros(X_st1.shape[0], dtype=int),
            np.ones(X_st2.shape[0], dtype=int)
        ]),
        dtype=torch.long, device=fabric.device
    )
    
    st_coords, _, _ = uet.canonicalize_st_coords_per_slide(
        st_coords_raw, slide_ids
    )
    print(f"[Rank {fabric.global_rank}] Stage C: Applied per-slide canonicalization")

    # Create st_gene_expr_dict for Stage C
    st1_expr_tensor = torch.tensor(X_st1, dtype=torch.float32, device=fabric.device)
    st2_expr_tensor = torch.tensor(X_st2, dtype=torch.float32, device=fabric.device)
    st_gene_expr_dict = {
        0: st1_expr_tensor,
        1: st2_expr_tensor
    }

    # ========== PHASE 1: ST-ONLY TRAINING ==========
    print("\n" + "="*70)
    print("PHASE 1: Training with ST data ONLY (fix geometry)")
    print("="*70)
    
    history_st = model.train_stageC(
        st_gene_expr_dict=st_gene_expr_dict,
        sc_gene_expr=sc_expr,
        n_min=96, n_max=384,
        num_st_samples=args.num_st_samples,
        num_sc_samples=0,  # DISABLE SC in phase 1
        n_epochs=stageC_epochs,
        batch_size=stageC_batch,
        lr=lr,
        n_timesteps=500,
        sigma_min=0.01,
        sigma_max=3.0,
        outf=outdir,
        fabric=fabric,
        precision=precision,
        phase_name="ST-only",
        enable_early_stop=args.enable_early_stop,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
    )
    
    fabric.barrier()
    
    # ========== SAVE ST CHECKPOINT (Phase 1 complete) ==========
    if fabric.is_global_zero:
        # Extract best epoch from history
        if history_st and history_st.get('early_stopped', False):
            E_ST_best = history_st['early_stop_info']['epoch']
            print(f"\n[Phase 1] Early stopped at epoch {E_ST_best}")
        else:
            E_ST_best = len(history_st['epoch']) if history_st else stageC_epochs
            print(f"\n[Phase 1] Completed all {E_ST_best} epochs")
        
        # Save Phase 1 checkpoint
        checkpoint_path = os.path.join(outdir, "phase1_st_checkpoint.pt")
        checkpoint = {
            'encoder': model.encoder.state_dict(),
            'context_encoder': model.context_encoder.module.state_dict() if hasattr(model.context_encoder, 'module') else model.context_encoder.state_dict(),
            'generator': model.generator.module.state_dict() if hasattr(model.generator, 'module') else model.generator.state_dict(),
            'score_net': model.score_net.module.state_dict() if hasattr(model.score_net, 'module') else model.score_net.state_dict(),
            'E_ST_best': E_ST_best,
            'lr_ST': lr,
            'history_st': history_st,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved Phase 1 checkpoint: {checkpoint_path}")
        print(f"  - Best ST epoch: {E_ST_best}")
        print(f"  - ST learning rate: {lr:.2e}")
    
    fabric.barrier()
    
    # ========== PHASE 2: SC FINE-TUNING WITH MIXED ST+SC BATCHES ==========
    if args.num_sc_samples > 0:
        print("\n" + "="*70)
        print("PHASE 2: Fine-tuning with SC data (mixed ST+SC regime)")
        print("="*70)
        
        # Derive SC hyperparameters from ST results
        if fabric.is_global_zero:
            if history_st and history_st.get('early_stopped', False):
                E_ST_best = history_st['early_stop_info']['epoch']
            else:
                E_ST_best = len(history_st['epoch']) if history_st else stageC_epochs
        else:
            E_ST_best = stageC_epochs  # fallback for non-rank0
 
        # CRITICAL FIX: Broadcast E_ST_best to all ranks to ensure sync
        E_ST_best_tensor = torch.tensor([E_ST_best], dtype=torch.long, device=fabric.device)
        if dist.is_initialized():
            dist.broadcast(E_ST_best_tensor, src=0)
        E_ST_best = int(E_ST_best_tensor.item())
 
        if fabric.is_global_zero:
            print(f"[Phase 2] Synchronized E_ST_best={E_ST_best} across all ranks")
        
        # Auto-compute SC epochs if not provided
        if args.sc_finetune_epochs is None:
            epochs_finetune = int(0.5 * E_ST_best)
            epochs_finetune = max(10, min(50, epochs_finetune))
        else:
            epochs_finetune = args.sc_finetune_epochs
        
        # Lower learning rate for fine-tuning
        lr_finetune = lr / 3.0
        
        # Keep some ST samples to anchor geometry (1/3 of Phase 1)
        num_st_finetune = args.num_st_samples // 2
        
        if fabric.is_global_zero:
            print(f"[Phase 2] Config:")
            print(f"  - Epochs: {epochs_finetune}")
            print(f"  - Learning rate: {lr_finetune:.2e} (1/3 of Phase 1)")
            print(f"  - ST samples: {num_st_finetune} (1/3 of Phase 1, to anchor geometry)")
            print(f"  - SC samples: {args.num_sc_samples}")
        
        # ========== FREEZE GENERATOR ONLY ==========
        if fabric.is_global_zero:
            print(f"\n[Phase 2] Freezing generator...")
        
        for param in model.generator.parameters():
            param.requires_grad = False
        
        if fabric.is_global_zero:
            print(f"  ✓ Generator frozen")
            print(f"  ✓ Score_net trainable")
            print(f"  ✓ Context_encoder trainable")
        
        training_history = model.train_stageC(
            st_gene_expr_dict=st_gene_expr_dict,
            sc_gene_expr=sc_expr,
            n_min=96, n_max=384,
            num_st_samples=num_st_finetune,
            num_sc_samples=args.num_sc_samples,
            n_epochs=epochs_finetune,
            batch_size=stageC_batch,
            lr=lr_finetune,
            n_timesteps=500,
            sigma_min=0.01,
            sigma_max=3.0,
            outf=outdir,
            fabric=fabric,
            precision=precision,
            phase_name="SC Fine-tune",
            enable_early_stop=False,
        )
        
        fabric.barrier()
        
        if fabric.is_global_zero:
            print("\n" + "="*70)
            print("PHASE 2 COMPLETE")
            print("="*70)
            
            checkpoint_path = os.path.join(outdir, "phase2_sc_finetuned_checkpoint.pt")
            checkpoint = {
                'encoder': model.encoder.state_dict(),
                'context_encoder': model.context_encoder.module.state_dict() if hasattr(model.context_encoder, 'module') else model.context_encoder.state_dict(),
                'generator': model.generator.module.state_dict() if hasattr(model.generator, 'module') else model.generator.state_dict(),
                'score_net': model.score_net.module.state_dict() if hasattr(model.score_net, 'module') else model.score_net.state_dict(),
                'E_ST_best': E_ST_best,
                'epochs_finetune': epochs_finetune,
                'lr_finetune': lr_finetune,
                'history_st': history_st,
                'history_sc': training_history,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        fabric.barrier()
    else:
        print("\n[INFO] Skipping Phase 2 (num_sc_samples=0)")
        training_history = history_st

    # Single barrier after training
    fabric.barrier()

    # Debug prints
    if fabric.is_global_zero:
        print(f"[DEBUG Rank-0] Training complete, got history")
    else:
        print(f"[DEBUG Rank-{fabric.global_rank}] Training complete, waiting")

    # Early stop info (only once!)
    if fabric.is_global_zero and training_history and training_history.get('early_stopped', False):
        info = training_history['early_stop_info']
        print(f"\n{'='*80}")
        print(f"Training stopped early at epoch {info['epoch']}")
        print(f"Best validation metric: {info['best_metric']:.4f}")
        print(f"{'='*80}\n")

    if not fabric.is_global_zero:
        return

    # ---------- Inference (rank-0 only, single GPU) ----------
    # Now use stadata3 (3rd slide) for inference!
    if fabric.is_global_zero:
        print("[DEBUG Rank-0] Starting inference on 3rd slide (stadata3)...")
        
        try:
            # Check if wrapped and unwrap to raw PyTorch module
            if hasattr(model.encoder, 'module'):
                model.encoder = model.encoder.module
            if hasattr(model.context_encoder, 'module'):
                print("[DEBUG Rank-0] Unwrapping context_encoder from DDP...")
                model.context_encoder = model.context_encoder.module
            if hasattr(model.score_net, 'module'):
                print("[DEBUG Rank-0] Unwrapping score_net from DDP...")
                model.score_net = model.score_net.module
            print("[DEBUG Rank-0] Models unwrapped, ready for single-GPU inference")
        except Exception as e:
            print(f"[DEBUG Rank-0] Unwrap failed (maybe not wrapped?): {e}")


        
        # Create timestamp for all outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ensure_disk_space(outdir, min_gb=10.0)

        # Prepare 3rd slide (inference slide) expression data
        X_st3 = stadata3[:, common].X
        if hasattr(X_st3, "toarray"): X_st3 = X_st3.toarray()
        st3_expr = torch.tensor(X_st3, dtype=torch.float32, device=fabric.device)

        print("\n=== Inference (rank-0) on 3rd Slide with Multi-Sample Ranking ===")

        if True:
            # Multi-sample inference with quality ranking
            K_samples = 1  # Number of samples to generate
            all_results = []
            all_scores = []

            for k in range(K_samples):
                print(f"\n  Generating sample {k+1}/{K_samples}...")
                
                # Set different seed for each sample
                torch.manual_seed(42 + k)

                n_cells = st3_expr.shape[0]

                sample_results = model.infer_sc_patchwise(
                    sc_gene_expr=st3_expr,
                    n_timesteps_sample=400,
                    return_coords=True,
                    patch_size=256,
                    coverage_per_cell=6.0,
                    n_align_iters=10,
                    eta=0.0,
                    guidance_scale=2.0,
                    sigma_min=0.01,
                    sigma_max=3.0,
                )
                
                # Compute EDM cone penalty as quality score (lower is better)
                coords_sample = sample_results['coords_canon']
                mask = torch.ones(coords_sample.shape[0], dtype=torch.bool, device=coords_sample.device)
                cone_penalty = uet.edm_cone_penalty_from_V(
                    coords_sample.unsqueeze(0), 
                    mask.unsqueeze(0)
                ).item()
                
                all_results.append(sample_results)
                all_scores.append(cone_penalty)
                
                print(f"  Sample {k+1} EDM cone penalty: {cone_penalty:.6f}")

            # Select best sample (lowest penalty)
            best_idx = int(np.argmin(all_scores))
            results = all_results[best_idx]

            print(f"\n✓ Selected sample {best_idx+1} with lowest cone penalty: {all_scores[best_idx]:.6f}")
            print(f"  Mean penalty: {np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")

            # Extract results from best sample
            D_edm = results['D_edm'].cpu().numpy()
            coords = results['coords'].cpu().numpy()
            coords_canon = results['coords_canon'].cpu().numpy()
            
            print(f"\nInference complete!")
            print(f"  D_edm shape: {D_edm.shape}")
            print(f"  Coordinates shape: {coords_canon.shape}")

        # ============================================================================
        # SINGLE-PATCH INFERENCE (for comparison with patchwise)
        # ============================================================================
        print("\n" + "="*70)
        print("SINGLE-PATCH INFERENCE (NO STITCHING)")
        print("="*70)
        
        torch.manual_seed(42)  # Same seed as patchwise for fair comparison
        
        # Compute ST p95 target using training slides
        D_st_list = []
        for coords_st in [st_coords1, st_coords2]:
            D = torch.cdist(
                torch.tensor(coords_st, dtype=torch.float32),
                torch.tensor(coords_st, dtype=torch.float32)
            )
            D_st_list.append(D)
        D_st_combined = torch.cat([D[torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)] 
                                   for D in D_st_list])
        target_st_p95 = D_st_combined.quantile(0.95).item()
        
        results_single = model.infer_sc_single_patch(
            sc_gene_expr=st3_expr,
            n_timesteps_sample=500,
            sigma_min=0.01,
            sigma_max=3.0,
            guidance_scale=2.0,
            eta=0.0,
            target_st_p95=target_st_p95,
            return_coords=True,
            DEBUG_FLAG=True,
        )
        
        # Extract results
        D_edm_single = results_single['D_edm'].cpu().numpy()
        coords_single = results_single['coords'].cpu().numpy()
        coords_canon_single = results_single['coords_canon'].cpu().numpy()
        
        print(f"\n✓ Single-patch inference complete!")
        print(f"  D_edm shape: {D_edm_single.shape}")
        print(f"  Coordinates shape: {coords_canon_single.shape}")
        
        # ============================================================================
        # SAVE SINGLE-PATCH RESULTS
        # ============================================================================
        results_single_filename = f"st3_inference_SINGLE_PATCH_{timestamp}.pt"
        results_single_path = os.path.join(outdir, results_single_filename)
        torch.save(results_single, results_single_path)
        print(f"✓ Saved single-patch results: {results_single_path}")
        
        # Save processed results
        results_single_processed = {
            'D_edm': D_edm_single,
            'coords': coords_single,
            'coords_canon': coords_canon_single,
            'n_cells': coords_canon_single.shape[0],
            'timestamp': timestamp,
            'method': 'single_patch',
            'model_config': {
                'n_genes': model.n_genes,
                'D_latent': model.D_latent,
                'c_dim': model.c_dim,
            }
        }
        processed_single_filename = f"st3_inference_SINGLE_PATCH_processed_{timestamp}.pt"
        processed_single_path = os.path.join(outdir, processed_single_filename)
        torch.save(results_single_processed, processed_single_path)
        print(f"✓ Saved processed single-patch results: {processed_single_path}")
        
        # ============================================================================
        # VISUALIZE SINGLE-PATCH RESULTS
        # ============================================================================
        print("\n=== Creating single-patch visualizations ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Single-patch coordinates colored by cell type
        if 'celltype' in stadata3.obs.columns:
            cell_types = stadata3.obs['celltype']
            unique_types = cell_types.unique()
            colors = sns.color_palette("husl", len(unique_types))
            color_map = dict(zip(unique_types, colors))
            
            for ct in unique_types:
                mask = (cell_types == ct).values
                axes[0].scatter(
                    coords_canon_single[mask, 0], 
                    coords_canon_single[mask, 1],
                    s=3, 
                    alpha=0.7, 
                    label=ct,
                    c=[color_map[ct]]
                )
            
            axes[0].set_title('Single-Patch Coordinates (by cell type)', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('Dim 1', fontsize=12)
            axes[0].set_ylabel('Dim 2', fontsize=12)
            axes[0].legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].scatter(coords_canon_single[:, 0], coords_canon_single[:, 1], s=3, alpha=0.7, c='steelblue')
            axes[0].set_title('Single-Patch Coordinates', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('Dim 1', fontsize=12)
            axes[0].set_ylabel('Dim 2', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        
        axes[0].axis('equal')
        
        # Plot 2: Distance distribution
        upper_tri_idx = np.triu_indices_from(D_edm_single, k=1)
        distances_single = D_edm_single[upper_tri_idx]
        
        axes[1].hist(distances_single, bins=100, alpha=0.7, edgecolor='black', color='green')
        axes[1].set_title('Pairwise Distance Distribution (Single-Patch)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Distance', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].axvline(distances_single.mean(), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {distances_single.mean():.2f}')
        axes[1].axvline(np.median(distances_single), color='b', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(distances_single):.2f}')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_single_filename = f"gems_inference_SINGLE_PATCH_{timestamp}.png"
        plot_single_path = os.path.join(outdir, plot_single_filename)
        plt.savefig(plot_single_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved single-patch plot: {plot_single_path}")
        plt.close()
        
        # ============================================================================
        # COMPARISON PLOT: Patchwise vs Single-Patch
        # ============================================================================
        print("\n=== Creating comparison plot ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Patchwise (left)
        if 'celltype' in stadata3.obs.columns:
            for ct in unique_types:
                mask = (cell_types == ct).values
                axes[0].scatter(
                    coords_canon[mask, 0], 
                    coords_canon[mask, 1],
                    s=3, alpha=0.7, label=ct, c=[color_map[ct]]
                )
            axes[0].legend(markerscale=3, fontsize=8, loc='best', framealpha=0.9)
        else:
            axes[0].scatter(coords_canon[:, 0], coords_canon[:, 1], s=3, alpha=0.7, c='steelblue')
        
        axes[0].set_title('PATCHWISE (with alignment)', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Dim 1', fontsize=12)
        axes[0].set_ylabel('Dim 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # Single-patch (right)
        if 'celltype' in stadata3.obs.columns:
            for ct in unique_types:
                mask = (cell_types == ct).values
                axes[1].scatter(
                    coords_canon_single[mask, 0], 
                    coords_canon_single[mask, 1],
                    s=3, alpha=0.7, label=ct, c=[color_map[ct]]
                )
            axes[1].legend(markerscale=3, fontsize=8, loc='best', framealpha=0.9)
        else:
            axes[1].scatter(coords_canon_single[:, 0], coords_canon_single[:, 1], s=3, alpha=0.7, c='green')
        
        axes[1].set_title('SINGLE-PATCH (no alignment)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Dim 1', fontsize=12)
        axes[1].set_ylabel('Dim 2', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        plt.tight_layout()
        
        comparison_filename = f"gems_COMPARISON_patchwise_vs_single_{timestamp}.png"
        comparison_path = os.path.join(outdir, comparison_filename)
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot: {comparison_path}")
        plt.close()
        
        # ============================================================================
        # COMPARISON STATISTICS
        # ============================================================================
        print("\n" + "="*70)
        print("COMPARISON: PATCHWISE vs SINGLE-PATCH")
        print("="*70)

        # Compute distances from D_edm (upper triangle)
        upper_tri_idx = np.triu_indices_from(D_edm, k=1)
        distances = D_edm[upper_tri_idx]
        
        print(f"\nPATCHWISE:")
        print(f"  Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]")
        print(f"  Distance stats: mean={distances.mean():.4f}, median={np.median(distances):.4f}, std={distances.std():.4f}")
        
        print(f"\nSINGLE-PATCH:")
        print(f"  Coordinate range: [{coords_canon_single.min():.2f}, {coords_canon_single.max():.2f}]")
        print(f"  Distance stats: mean={distances_single.mean():.4f}, median={np.median(distances_single):.4f}, std={distances_single.std():.4f}")
        
        # Compute geometry metrics
        def compute_geometry_metrics(coords):
            """Compute dimensionality and anisotropy metrics."""
            coords_centered = coords - coords.mean(axis=0)
            cov = np.cov(coords_centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]  # Descending order
            
            # Effective dimensionality (participation ratio)
            dim_eff = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
            
            # Anisotropy ratio (top 2 eigenvalues)
            if len(eigvals) >= 2:
                aniso = eigvals[0] / (eigvals[1] + 1e-8)
            else:
                aniso = 1.0
            
            return dim_eff, aniso, eigvals
        
        dim_pw, aniso_pw, eig_pw = compute_geometry_metrics(coords_canon)
        dim_sp, aniso_sp, eig_sp = compute_geometry_metrics(coords_canon_single)
        
        print(f"\nGEOMETRY METRICS:")
        print(f"  Patchwise:    eff_dim={dim_pw:.2f}, anisotropy={aniso_pw:.2f}, top_eigs={eig_pw[:3]}")
        print(f"  Single-patch: eff_dim={dim_sp:.2f}, anisotropy={aniso_sp:.2f}, top_eigs={eig_sp[:3]}")
        print("="*70 + "\n")

        
        # ============================================================================
        # SAVE RESULTS WITH DATETIME
        # ============================================================================
        
        # Save raw results
        results_filename = f"st3_inference_{timestamp}.pt"
        results_path = os.path.join(outdir, results_filename)
        torch.save(results, results_path)
        print(f"\n✓ Saved raw results: {results_path}")
        
        # Save processed results with metadata
        results_processed = {
            'D_edm': D_edm,
            'coords': coords,
            'coords_canon': coords_canon,
            'n_cells': coords_canon.shape[0],
            'timestamp': timestamp,
            'model_config': {
                'n_genes': model.n_genes,
                'D_latent': model.D_latent,
                'c_dim': model.c_dim,
            }
        }
        processed_filename = f"st3_inference_processed_{timestamp}.pt"
        processed_path = os.path.join(outdir, processed_filename)
        torch.save(results_processed, processed_path)
        print(f"✓ Saved processed results: {processed_path}")
        
        # ============================================================================
        # ADD COORDINATES TO ANNDATA
        # ============================================================================
        
        # Add GEMS coordinates to stadata3
        stadata3.obsm['X_gems'] = coords_canon
        
        # Save AnnData with GEMS coordinates
        adata_filename = f"stadata3_with_gems_{timestamp}.h5ad"
        adata_path = os.path.join(outdir, adata_filename)
        stadata3.write_h5ad(adata_path)
        print(f"✓ Saved AnnData with GEMS coords: {adata_path}")
        
        # ============================================================================
        # VISUALIZATION
        # ============================================================================
        
        print("\n=== Creating visualizations ===")
        
        # Figure 1: Cell type colored scatter (matplotlib)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: GEMS coordinates colored by cell type (if available)
        if 'celltype' in stadata3.obs.columns:
            cell_types = stadata3.obs['celltype']
            unique_types = cell_types.unique()
            
            # Use a nice color palette
            colors = sns.color_palette("husl", len(unique_types))
            color_map = dict(zip(unique_types, colors))
            
            for ct in unique_types:
                mask = (cell_types == ct).values
                axes[0].scatter(
                    coords_canon[mask, 0], 
                    coords_canon[mask, 1],
                    s=3, 
                    alpha=0.7, 
                    label=ct,
                    c=[color_map[ct]]
                )
            
            axes[0].set_title('GEMS Coordinates - Slide 3 (by cell type)', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('GEMS Dim 1', fontsize=12)
            axes[0].set_ylabel('GEMS Dim 2', fontsize=12)
            axes[0].legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].scatter(coords_canon[:, 0], coords_canon[:, 1], s=3, alpha=0.7, c='steelblue')
            axes[0].set_title('GEMS Coordinates - Slide 3', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('GEMS Dim 1', fontsize=12)
            axes[0].set_ylabel('GEMS Dim 2', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        
        axes[0].axis('equal')
        
        # Plot 2: Distance distribution
        upper_tri_idx = np.triu_indices_from(D_edm, k=1)
        distances = D_edm[upper_tri_idx]
        
        axes[1].hist(distances, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        axes[1].set_title('Pairwise Distance Distribution (EDM)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Distance', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].axvline(distances.mean(), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {distances.mean():.2f}')
        axes[1].axvline(np.median(distances), color='g', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(distances):.2f}')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save matplotlib figure
        plot_filename = f"gems_inference_results_{timestamp}.png"
        plot_path = os.path.join(outdir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved matplotlib plot: {plot_path}")
        plt.close()
        
        # ============================================================================
        # SCANPY VISUALIZATION
        # ============================================================================
        
        if 'celltype' in stadata3.obs.columns:
            # Create scanpy-style embedding plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sc.settings.set_figure_params(dpi=150, frameon=False)
            
            # Use scanpy's embedding plot
            sc.pl.embedding(
                stadata3, 
                basis='gems',  # This uses obsm['X_gems']
                color='celltype',
                title='GEMS Embedding - Slide 3 (Scanpy)',
                save=False,
                show=False,
                ax=ax,
                size=20,
                alpha=0.8,
                legend_loc='on data',
                legend_fontsize=8
            )
            
            # Save scanpy plot
            scanpy_plot_filename = f"gems_embedding_scanpy_{timestamp}.png"
            scanpy_plot_path = os.path.join(outdir, scanpy_plot_filename)
            plt.savefig(scanpy_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved scanpy plot: {scanpy_plot_path}")
            plt.close()
            
            # Also create a high-quality version with better styling
            fig, ax = plt.subplots(figsize=(12, 10))
            sc.pl.embedding(
                stadata3, 
                basis='gems',
                color='celltype',
                title='',
                save=False,
                show=False,
                ax=ax,
                size=30,
                alpha=0.7,
                frameon=True,
                legend_loc='right margin',
                legend_fontsize=10
            )
            ax.set_xlabel('GEMS-1', fontsize=14, fontweight='bold')
            ax.set_ylabel('GEMS-2', fontsize=14, fontweight='bold')
            
            hq_plot_filename = f"gems_embedding_hq_{timestamp}.png"
            hq_plot_path = os.path.join(outdir, hq_plot_filename)
            plt.savefig(hq_plot_path, dpi=600, bbox_inches='tight')
            print(f"✓ Saved high-quality plot: {hq_plot_path}")
            plt.close()
        
        # ============================================================================
        # SUMMARY STATISTICS
        # ============================================================================
        
        print("\n" + "="*70)
        print("INFERENCE SUMMARY")
        print("="*70)
        print(f"Timestamp: {timestamp}")
        print(f"Number of spots (slide 3): {coords_canon.shape[0]}")
        print(f"Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]")
        print(f"Distance statistics:")
        print(f"  Mean: {distances.mean():.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        print(f"  Std: {distances.std():.4f}")
        print(f"  Min: {distances.min():.4f}")
        print(f"  Max: {distances.max():.4f}")
        
        if 'celltype' in stadata3.obs.columns:
            print(f"\nCell type distribution:")
            for ct, count in stadata3.obs['celltype'].value_counts().items():
                print(f"  {ct}: {count} spots ({count/len(stadata3)*100:.1f}%)")
        
        print("\n" + "="*70)
        print("All outputs saved to:", outdir)
        print("="*70)
        
        # Create a summary file
        summary_filename = f"inference_summary_{timestamp}.txt"
        summary_path = os.path.join(outdir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write("GEMS Inference Summary - hSCC Slide 3\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Number of spots: {coords_canon.shape[0]}\n")
            f.write(f"Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]\n")
            f.write(f"\nDistance statistics:\n")
            f.write(f"  Mean: {distances.mean():.4f}\n")
            f.write(f"  Median: {np.median(distances):.4f}\n")
            f.write(f"  Std: {distances.std():.4f}\n")
            f.write(f"  Min: {distances.min():.4f}\n")
            f.write(f"  Max: {distances.max():.4f}\n")
            if 'celltype' in stadata3.obs.columns:
                f.write(f"\nCell type distribution:\n")
                for ct, count in stadata3.obs['celltype'].value_counts().items():
                    f.write(f"  {ct}: {count} spots ({count/len(stadata3)*100:.1f}%)\n")
        
        print(f"✓ Saved summary: {summary_path}")
        print("[DEBUG Rank-0] Inference complete!")
        
    if fabric.is_global_zero:
        print("\n=== Plotting Stage C training losses ===")
        
        # ============================================================================
        # PLOT PHASE 1 (ST-ONLY) LOSSES
        # ============================================================================
        if history_st is not None and len(history_st['epoch']) > 0:
            print("\n--- Plotting Phase 1 (ST-only) Losses ---")
            epochs = history_st['epoch']
            losses = history_st['epoch_avg']
            
            # ST-specific losses
            st_loss_names = ['total', 'score', 'gram', 'gram_scale', 'heat', 'sw_st', 'st_dist', 'edm_tail', 'gen_align', 'dim', 'triangle', 'radial', 'repel', 'shape']
            st_colors = ['black', 'blue', 'red', 'orange', 'green', 'purple', 'magenta', 'cyan', 'lime', 'brown', 'pink', 'gray', 'darkred', 'darkblue']

            
            # Increased grid to fit new losses
            fig, axes = plt.subplots(5, 3, figsize=(20, 25))

            fig.suptitle('Phase 1: ST-Only Training Losses', fontsize=18, fontweight='bold', y=0.995)
            
            axes = axes.flatten()
            for idx, (name, color) in enumerate(zip(st_loss_names, st_colors)):
                if name in losses and len(losses[name]) > 0:
                    ax = axes[idx]
                    ax.plot(epochs, losses[name], color=color, linewidth=2, alpha=0.7, marker='o', markersize=4)
                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title(f'{name.upper()} Loss', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    if len(epochs) > 10:
                        smoothed = gaussian_filter1d(losses[name], sigma=2)
                        ax.plot(epochs, smoothed, '--', color=color, linewidth=2.5, alpha=0.5, label='Trend')
                        ax.legend(fontsize=10)
            
            # Hide unused subplot
            axes[14].axis('off')

            plt.tight_layout()
            st_plot_filename = f"stageC_phase1_ST_losses_{timestamp}.png"
            st_plot_path = os.path.join(outdir, st_plot_filename)
            plt.savefig(st_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Phase 1 (ST) loss plot: {st_plot_path}")
            plt.close()
        
        # ============================================================================
        # PLOT PHASE 2 (SC FINE-TUNE) LOSSES (if it ran)
        # ============================================================================
        if args.num_sc_samples > 0 and training_history is not None and len(training_history['epoch']) > 0:
            print("\n--- Plotting Phase 2 (SC Fine-tune) Losses ---")
            epochs = training_history['epoch']
            losses = training_history['epoch_avg']
            
            # SC-specific losses
            sc_loss_names = ['total', 'score', 'sw_sc', 'overlap', 'ordinal_sc']
            sc_colors = ['black', 'blue', 'purple', 'brown', 'pink']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Phase 2: SC Fine-tuning Losses', fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()
            
            for idx, (name, color) in enumerate(zip(sc_loss_names, sc_colors)):
                if name in losses and len(losses[name]) > 0:
                    ax = axes[idx]
                    ax.plot(epochs, losses[name], color=color, linewidth=2, alpha=0.7, marker='o', markersize=4)
                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title(f'{name.upper()} Loss', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    if len(epochs) > 10:
                        smoothed = gaussian_filter1d(losses[name], sigma=2)
                        ax.plot(epochs, smoothed, '--', color=color, linewidth=2.5, alpha=0.5, label='Trend')
                        ax.legend(fontsize=10)
            
            # Hide empty subplot
            axes[5].axis('off')
            
            plt.tight_layout()
            sc_plot_filename = f"stageC_phase2_SC_losses_{timestamp}.png"
            sc_plot_path = os.path.join(outdir, sc_plot_filename)
            plt.savefig(sc_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Phase 2 (SC) loss plot: {sc_plot_path}")
            plt.close()
        
        # ============================================================================
        # SUMMARY STATISTICS
        # ============================================================================
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        if history_st is not None:
            print("\n--- Phase 1: ST-Only Training ---")
            st_losses = history_st['epoch_avg']
            print(f"Total epochs: {len(history_st['epoch'])}")
            if 'total' in st_losses and len(st_losses['total']) > 0:
                print(f"Final total loss: {st_losses['total'][-1]:.4f}")
                print(f"Final ST geometry losses:")
                for name in ['gram', 'heat', 'edm_tail', 'gen_align', 'dim', 'triangle', 'radial', 'repel', 'shape']:
                    if name in st_losses and len(st_losses[name]) > 0:
                        print(f"  {name}: {st_losses[name][-1]:.4f}")

        
        if args.num_sc_samples > 0 and training_history is not None:
            print("\n--- Phase 2: SC Fine-tuning ---")
            sc_losses = training_history['epoch_avg']
            print(f"Total epochs: {len(training_history['epoch'])}")
            if 'total' in sc_losses and len(sc_losses['total']) > 0:
                print(f"Final total loss: {sc_losses['total'][-1]:.4f}")
                print(f"Final SC losses:")
                for name in ['sw_sc', 'overlap', 'ordinal_sc']:
                    if name in sc_losses and len(sc_losses[name]) > 0:
                        print(f"  {name}: {sc_losses[name][-1]:.4f}")
        
        print("="*70)

if __name__ == "__main__":
    args = parse_args()
    main(args)

    os._exit(0)