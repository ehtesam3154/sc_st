# run_mouse_brain_lightning.py
import os
import torch
from lightning.fabric import Fabric
import scanpy as sc
from core_models_et_p3 import GEMSModel
# from mouse_brain import train_gems_mousebrain  # optional, if you want its helpers
import sys

import argparse

import shutil
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='GEMS Training with Lightning Fabric')
    
    # Training config
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--stageA_epochs', type=int, default=1000)
    parser.add_argument('--stageC_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='gems_mousebrain_output')
    parser.add_argument('--num_st_samples', type=int, default=4000)  # ADD THIS
    parser.add_argument('--num_sc_samples', type=int, default=9000)  # ADD THIS
    
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

def load_mouse_data():
    # ST1 as training ST data (with coordinates)
    st_counts = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st1_counts_et.csv'
    st_meta   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st1_metadata_et.csv'
    st_ct     = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st1_celltype_et.csv'
    
    # ST2 as test SC data (coordinates hidden, used for evaluation)
    sc_counts = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st2_counts_et.csv'
    sc_meta   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st2_metadata_et.csv'
    sc_ct     = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st2_celltype_et.csv'

    import pandas as pd
    import anndata as ad
    
    # Load ST1 data (training)
    print("Loading ST1 (training ST data)...")
    st_expr_df = pd.read_csv(st_counts, index_col=0)
    st_meta_df = pd.read_csv(st_meta, index_col=0)
    st_ct_df = pd.read_csv(st_ct, index_col=0)
    
    stadata = ad.AnnData(X=st_expr_df.values.T)
    stadata.obs_names = st_expr_df.columns
    stadata.var_names = st_expr_df.index
    stadata.obsm['spatial'] = st_meta_df[['coord_x', 'coord_y']].values
    stadata.obs['celltype_mapped_refined'] = st_ct_df.idxmax(axis=1).values
    stadata.obsm['celltype_proportions'] = st_ct_df.values
    
    print(f"ST1 loaded: {stadata.shape[0]} spots, {stadata.shape[1]} genes")
    
    # Load ST2 data (test "SC" data)
    print("Loading ST2 (test SC data)...")
    sc_expr_df = pd.read_csv(sc_counts, index_col=0)
    sc_meta_df = pd.read_csv(sc_meta, index_col=0)
    sc_ct_df = pd.read_csv(sc_ct, index_col=0)
    
    scadata = ad.AnnData(X=sc_expr_df.values.T)
    scadata.obs_names = sc_expr_df.columns
    scadata.var_names = sc_expr_df.index
    scadata.obsm['spatial_gt'] = sc_meta_df[['coord_x', 'coord_y']].values
    scadata.obs['celltype_mapped_refined'] = sc_ct_df.idxmax(axis=1).values
    scadata.obsm['celltype_proportions'] = sc_ct_df.values
    
    print(f"ST2 loaded: {scadata.shape[0]} cells, {scadata.shape[1]} genes")
    print(f"Ground truth coordinates stored in scadata.obsm['spatial_gt']")
    
    return scadata, stadata

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
    scadata, stadata = load_mouse_data()

    # ---------- Build model (same across ranks) ----------
    n_genes = len(sorted(list(set(scadata.var_names) & set(stadata.var_names))))

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
        # Extract tensors like mouse_brain.py does
        import numpy as np
        import utils_et as uet
        
        common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
        X_sc = scadata[:, common].X
        X_st = stadata[:, common].X
        if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
        if hasattr(X_st, "toarray"): X_st = X_st.toarray()
        sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
        st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)
        
        # NEW: Per-slide canonicalization BEFORE Stage A/B
        st_coords_raw = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
        slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)
        
        st_coords, st_mu, st_scale = uet.canonicalize_st_coords_per_slide(
            st_coords_raw, slide_ids
        )
        
        # Save canonicalization stats for later denormalization
        os.makedirs(outdir, exist_ok=True)
        torch.save(
            {"mu": st_mu.cpu(), "scale": st_scale.cpu()},
            os.path.join(outdir, "st_slide_canon_stats.pt"),
        )
        print(f"[Rank-0] Per-slide canonicalization: scale={st_scale[0].item():.4f}")

        print("\n=== Stage A (single GPU, rank-0) ===")
        model.train_stageA(
            st_gene_expr=st_expr,
            st_coords=st_coords,
            sc_gene_expr=sc_expr,
            slide_ids=slide_ids,
            n_epochs=stageA_epochs,
            batch_size=256,
            lr=1e-4,
            sigma=None,
            alpha=0.8,
            ratio_start=0.0,
            ratio_end=1.0,
            mmdbatch=1.0,
            outf=outdir,
        )

        print("\n=== Stage B (single GPU, rank-0) ===")
        slides_dict = {0: (st_coords, st_expr)}
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
        # because targets_dict is not serializable (has sparse tensors, etc.)
        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")
        import utils_et as uet
        
        common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
        X_st = stadata[:, common].X
        if hasattr(X_st, "toarray"): X_st = X_st.toarray()
        st_expr_rank = torch.tensor(X_st, dtype=torch.float32)
        
        # NEW: Same per-slide canonicalization as rank-0
        st_coords_raw_rank = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32)
        slide_ids_rank = torch.zeros(st_expr_rank.shape[0], dtype=torch.long)
        
        st_coords_rank, _, _ = uet.canonicalize_st_coords_per_slide(
            st_coords_raw_rank, slide_ids_rank
        )
        print(f"[Rank {fabric.global_rank}] Applied per-slide canonicalization")
        
        slides_dict_rank = {0: (st_coords_rank, st_expr_rank)}

        model.train_stageB(
            slides=slides_dict_rank,
            outdir=stageB_outdir,
        )

    # ---------- Stage C (multi-GPU with Fabric) ----------
    print("\n=== Stage C (DDP across GPUs) ===")
    # Rebuild tensors on each rank (cheap)
    import utils_et as uet
    
    common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    X_sc = scadata[:, common].X
    X_st = stadata[:, common].X
    if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
    if hasattr(X_st, "toarray"): X_st = X_st.toarray()
    sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
    st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)
    
    # NEW: Use same per-slide canonicalization for Stage C
    st_coords_raw = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
    slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)
    
    st_coords, _, _ = uet.canonicalize_st_coords_per_slide(
        st_coords_raw, slide_ids
    )
    print(f"[Rank {fabric.global_rank}] Stage C: Applied per-slide canonicalization")

    st_gene_expr_dict = {0: st_expr}

    # training_history = model.train_stageC(
    #     st_gene_expr_dict=st_gene_expr_dict,
    #     sc_gene_expr=sc_expr,
    #     n_min=64, n_max=384,
    #     num_st_samples=args.num_st_samples,
    #     num_sc_samples=args.num_sc_samples,
    #     n_epochs=stageC_epochs,
    #     batch_size=stageC_batch,   # per-GPU batch; global batch = this * #GPUs
    #     lr=lr,
    #     n_timesteps=500,
    #     sigma_min=0.01,
    #     sigma_max=7.0,
    #     outf=outdir,
    #     fabric=fabric,
    #     precision=precision,
    #     # Early stopping
    #     enable_early_stop=args.enable_early_stop,
    #     early_stop_min_epochs=args.early_stop_min_epochs,
    #     early_stop_patience=args.early_stop_patience,
    #     early_stop_threshold=args.early_stop_threshold,
    # )

    # ========== PHASE 1: ST-ONLY TRAINING ==========
    print("\n" + "="*70)
    print("PHASE 1: Training with ST data ONLY (fix geometry)")
    print("="*70)
    
    history_st = model.train_stageC(
        st_gene_expr_dict=st_gene_expr_dict,
        sc_gene_expr=sc_expr,
        n_min=64, n_max=384,
        num_st_samples=args.num_st_samples,
        num_sc_samples=0,  # DISABLE SC in phase 1
        n_epochs=stageC_epochs,
        batch_size=stageC_batch,
        lr=lr,
        n_timesteps=500,
        sigma_min=0.01,
        sigma_max=7.0,
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
        import torch.distributed as dist
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
            n_min=64, n_max=384,
            num_st_samples=num_st_finetune,  # ← CHANGED from 0
            num_sc_samples=args.num_sc_samples,
            n_epochs=epochs_finetune,
            batch_size=stageC_batch,
            lr=lr_finetune,
            n_timesteps=500,
            sigma_min=0.01,
            sigma_max=7.0,
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
    if fabric.is_global_zero:
        print("[DEBUG Rank-0] Starting inference...")
        
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

        from datetime import datetime
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create timestamp for all outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ensure_disk_space(outdir, min_gb=10.0)

        print("\n=== Inference (rank-0) with Multi-Sample Ranking ===")

        if True:

            # Multi-sample inference with quality ranking
            K_samples = 1  # Number of samples to generate
            all_results = []
            all_scores = []

            import utils_et as uet

            for k in range(K_samples):
                print(f"\n  Generating sample {k+1}/{K_samples}...")
                
                # Set different seed for each sample
                torch.manual_seed(42 + k)

                # sample_results = model.infer_sc_anchored(
                #     sc_gene_expr=sc_expr,
                #     n_timesteps_sample=300,
                #     return_coords=True,
                #     anchor_size=640,
                #     batch_size=512,      # ← REDUCED from 512 to avoid OOM
                #     eta=0.0,
                #     guidance_scale=7.0,
                #     sigma_min=0.01,      # ← ADD THIS
                #     sigma_max=5.0,      # ← MATCH TRAINING (was 5.0, should be 53.0)
                # )

                n_cells = sc_expr.shape[0]

                sample_results = model.infer_sc_patchwise(
                    sc_gene_expr=sc_expr,
                    n_timesteps_sample=300,
                    return_coords=True,
                    # patch_size=512,          # was batch_size; also your Stage D batch size
                    patch_size=n_cells,
                    coverage_per_cell=5.0,   # you can tune 3–6
                    n_align_iters=10,        # can tune 5–15
                    eta=0.0,
                    guidance_scale=2.0,
                    sigma_min=0.01,
                    sigma_max=7.0,
                )
                
                # Compute EDM cone penalty as quality score (lower is better)
                coords_sample = sample_results['coords_canon']
                mask = torch.ones(coords_sample.shape[0], dtype=torch.bool, device=coords_sample.device)
                cone_penalty = uet.edm_cone_penalty_from_V(
                    coords_sample.unsqueeze(0), 
                    mask.unsqueeze(0)
                ).item()
                # cone_penalty = uet.edm_cone_penalty_from_V(coords_sample.unsqueeze(0), mask.unsqueeze(0))
                
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
            
            # Extract results
            D_edm = results['D_edm'].cpu().numpy()
            coords = results['coords'].cpu().numpy()
            coords_canon = results['coords_canon'].cpu().numpy()
            
            print(f"\nInference complete!")
            print(f"  D_edm shape: {D_edm.shape}")
            print(f"  Coordinates shape: {coords_canon.shape}")

        if False:
            print("\n=== Inference (rank-0) with Multi-Sample Ranking [RANDOM MODE] ===")

            # Get number of cells from sc_expr
            n_cells = sc_expr.shape[0]
            D_latent = 16  # matches your model config

            # Multi-sample inference with quality ranking
            K_samples = 1  # Number of samples to generate
            all_results = []
            all_scores = []
            import utils_et as uet

            for k in range(K_samples):
                print(f"\n  Generating sample {k+1}/{K_samples}...")
                
                # Set different seed for each sample
                torch.manual_seed(42 + k)
                
                # ===== RANDOM RESULTS INSTEAD OF INFERENCE =====
                # Generate random coordinates in latent space
                coords_random = torch.randn(n_cells, D_latent, device=fabric.device)
                
                # Generate random canonicalized coordinates
                coords_canon_random = torch.randn(n_cells, D_latent, device=fabric.device)
                
                # Generate random distance matrix
                D_edm_random = torch.cdist(coords_canon_random.unsqueeze(0), 
                                        coords_canon_random.unsqueeze(0)).squeeze(0)
                
                # Create fake results dictionary matching expected format
                sample_results = {
                    'D_edm': D_edm_random,
                    'coords': coords_random,
                    'coords_canon': coords_canon_random
                }
                # ===== END RANDOM RESULTS =====
                
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
        # SAVE RESULTS WITH DATETIME
        # ============================================================================
        
        # Save raw results
        results_filename = f"sc_inference_{timestamp}.pt"
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
        processed_filename = f"sc_inference_processed_{timestamp}.pt"
        processed_path = os.path.join(outdir, processed_filename)
        torch.save(results_processed, processed_path)
        print(f"✓ Saved processed results: {processed_path}")
        
        # ============================================================================
        # ADD COORDINATES TO ANNDATA
        # ============================================================================
        
        # Add GEMS coordinates to scadata
        scadata.obsm['X_gems'] = coords_canon
        
        # Save AnnData with GEMS coordinates
        adata_filename = f"scadata_with_gems_{timestamp}.h5ad"
        adata_path = os.path.join(outdir, adata_filename)
        scadata.write_h5ad(adata_path)
        print(f"✓ Saved AnnData with GEMS coords: {adata_path}")
        
        # ============================================================================
        # VISUALIZATION
        # ============================================================================
        
        print("\n=== Creating visualizations ===")
        
        # Figure 1: Cell type colored scatter (matplotlib)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: GEMS coordinates colored by cell type
        if 'celltype' in scadata.obs.columns:
            cell_types = scadata.obs['celltype']
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
            
            axes[0].set_title('GEMS Coordinates (by cell type)', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('GEMS Dim 1', fontsize=12)
            axes[0].set_ylabel('GEMS Dim 2', fontsize=12)
            axes[0].legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].scatter(coords_canon[:, 0], coords_canon[:, 1], s=3, alpha=0.7, c='steelblue')
            axes[0].set_title('GEMS Coordinates', fontsize=16, fontweight='bold')
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
        
        if 'celltype' in scadata.obs.columns:
            # Create scanpy-style embedding plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            import scanpy as sc
            sc.settings.set_figure_params(dpi=150, frameon=False)
            
            # Use scanpy's embedding plot
            sc.pl.embedding(
                scadata, 
                basis='gems',  # This uses obsm['X_gems']
                color='celltype',
                title='GEMS Embedding (Scanpy)',
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
                scadata, 
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
        print(f"Number of cells: {coords_canon.shape[0]}")
        print(f"Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]")
        print(f"Distance statistics:")
        print(f"  Mean: {distances.mean():.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        print(f"  Std: {distances.std():.4f}")
        print(f"  Min: {distances.min():.4f}")
        print(f"  Max: {distances.max():.4f}")
        
        if 'celltype' in scadata.obs.columns:
            print(f"\nCell type distribution:")
            for ct, count in scadata.obs['celltype'].value_counts().items():
                print(f"  {ct}: {count} cells ({count/len(scadata)*100:.1f}%)")
        
        print("\n" + "="*70)
        print("All outputs saved to:", outdir)
        print("="*70)
        
        # Create a summary file
        summary_filename = f"inference_summary_{timestamp}.txt"
        summary_path = os.path.join(outdir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write("GEMS Inference Summary\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Number of cells: {coords_canon.shape[0]}\n")
            f.write(f"Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]\n")
            f.write(f"\nDistance statistics:\n")
            f.write(f"  Mean: {distances.mean():.4f}\n")
            f.write(f"  Median: {np.median(distances):.4f}\n")
            f.write(f"  Std: {distances.std():.4f}\n")
            f.write(f"  Min: {distances.min():.4f}\n")
            f.write(f"  Max: {distances.max():.4f}\n")
            if 'celltype' in scadata.obs.columns:
                f.write(f"\nCell type distribution:\n")
                for ct, count in scadata.obs['celltype'].value_counts().items():
                    f.write(f"  {ct}: {count} cells ({count/len(scadata)*100:.1f}%)\n")
        
        print(f"✓ Saved summary: {summary_path}")
        print("[DEBUG Rank-0] Inference complete!")

    # # === CLEAN SYNC POINT ===
    # # === FINAL SYNC POINT ===
    # # Rank-1 has been waiting here while rank-0 did inference
    # print(f"[DEBUG Rank-{fabric.global_rank}] Reaching final sync point", flush=True)
    # torch.cuda.synchronize()
    # if fabric.world_size > 1:
    #     fabric.barrier()
    # print(f"[DEBUG Rank-{fabric.global_rank}] Passed final sync point", flush=True)

    # Add this right after the inference block
    # if not fabric.is_global_zero:
    #     print(f"[DEBUG Rank-{fabric.global_rank}] Inference skipped, ready for sync", flush=True)
    #     print(f"[DEBUG Rank-{fabric.global_rank}] Reached post-inference point")


    #     # ============================================================================
    #     # PLOT STAGE C TRAINING LOSSES
    #     # ============================================================================
        
    #     print("\n=== Loading and plotting Stage C training losses ===")
        
    #     # ============================================================================
    #     # PLOT STAGE C TRAINING LOSSES (NO CHECKPOINT NEEDED)
    #     # ============================================================================

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
            st_loss_names = ['total', 'score', 'gram', 'gram_scale', 'heat', 'sw_st', 'st_dist', 'edm_tail', 'gen_align']
            st_colors = ['black', 'blue', 'red', 'orange', 'green', 'purple', 'magenta', 'cyan', 'lime']
            
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle('Phase 1: ST-Only Training Losses', fontsize=18, fontweight='bold', y=0.995)
            
            for idx, (name, color) in enumerate(zip(st_loss_names, st_colors)):
                if name in losses and len(losses[name]) > 0:
                    ax = axes[idx // 3, idx % 3]
                    ax.plot(epochs, losses[name], color=color, linewidth=2, alpha=0.7, marker='o', markersize=4)
                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title(f'{name.upper()} Loss', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    if len(epochs) > 10:
                        from scipy.ndimage import gaussian_filter1d
                        smoothed = gaussian_filter1d(losses[name], sigma=2)
                        ax.plot(epochs, smoothed, '--', color=color, linewidth=2.5, alpha=0.5, label='Trend')
                        ax.legend(fontsize=10)
            
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
            
            # SC-specific losses (no gram/heat/edm_tail/gen_align)
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
                        from scipy.ndimage import gaussian_filter1d
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
                for name in ['gram', 'heat', 'edm_tail', 'gen_align']:
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

    import os
    os._exit(0)
