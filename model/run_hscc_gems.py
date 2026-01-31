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

    # Miniset sampling parameters
    parser.add_argument('--pool_mult', type=float, default=2.0,
                        help='Pool multiplier for stochastic miniset sampling. '
                             'Lower values (1.5-2.0) give tighter spatial locality. '
                             'For small slides (<1000 spots), use 1.5-2.0 to avoid pool covering entire slide.')
    parser.add_argument('--stochastic_tau', type=float, default=1.0,
                        help='Temperature for stochastic sampling within pool (lower = more deterministic)')

    # Early stopping
    parser.add_argument('--enable_early_stop', action='store_true', default=False,
                        help='Enable early stopping (legacy, only used when curriculum is disabled)')
    parser.add_argument('--early_stop_min_epochs', type=int, default=12,
                        help='Minimum epochs before early stopping kicks in')
    parser.add_argument('--early_stop_patience', type=int, default=6,
                        help='Epochs to wait without improvement before stopping')
    parser.add_argument('--early_stop_threshold', type=float, default=0.01,
                        help='Relative improvement threshold (e.g., 0.01 = 1%%)')

    # ========== CURRICULUM EARLY STOPPING ==========
    parser.add_argument('--curriculum_target_stage', type=int, default=2,
                        help='Target curriculum stage for Gate A (0-indexed, default 2 = final stage for 3-stage curriculum)')
    parser.add_argument('--curriculum_min_epochs', type=int, default=100,
                        help='Minimum epochs before three-gate stopping is allowed')
    parser.add_argument('--curriculum_early_stop', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable three-gate curriculum-aware early stopping (default: enabled)')
    parser.add_argument('--use_legacy_curriculum', action=argparse.BooleanOptionalAction, default=False,
                        help='Use legacy 7-stage curriculum [0.3,...,17.0] instead of new 3-stage [1,2,3]')
    # ========== DATA-DEPENDENT SIGMA CAP ==========
    parser.add_argument('--sigma_cap_safe_mult', type=float, default=3.0,
                        help='Safety cap multiplier: σ_cap_safe = mult × σ0 (default 3.0, allows 3×σ_data)')
    parser.add_argument('--sigma_cap_abs_max', type=float, default=None,
                        help='Optional absolute max σ_cap (e.g., 1.5). None = no absolute limit.')
    parser.add_argument('--sigma_cap_abs_min', type=float, default=None,
                        help='Optional absolute min σ_cap. None = no absolute floor.')

    parser.add_argument('--sc_finetune_epochs', type=int, default=None, 
                        help='SC fine-tune epochs (default: auto = 50%% of ST best epoch, clamped [10,50])')
    
    # ========== COMPETITOR TRAINING FLAGS (ChatGPT hypothesis test) ==========
    parser.add_argument('--compete_train', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable competitor training (expand candidate pool)')
    parser.add_argument('--compete_n_extra', type=int, default=128,
                        help='Total extra points to add beyond core patch')
    parser.add_argument('--compete_n_rand', type=int, default=64,
                        help='Number of random distractors from outside patch')
    parser.add_argument('--compete_n_hard', type=int, default=64,
                        help='Number of hard negatives (expression-similar distractors)')
    parser.add_argument('--compete_use_pos_closure', action=argparse.BooleanOptionalAction, default=True,
                        help='Include GT spatial neighbors missing from core patch')
    parser.add_argument('--compete_k_pos', type=int, default=10,
                        help='Number of GT spatial neighbors to try to include per core point')
    parser.add_argument('--compete_expr_knn_k', type=int, default=50,
                        help='Top-K expression neighbors for hard negative pool')
    parser.add_argument('--compete_anchor_only', action=argparse.BooleanOptionalAction, default=True,
                        help='Restrict NCA loss anchors to core points only')
    parser.add_argument('--compete_diag_every', type=int, default=200,
                        help='Steps between competitor diagnostic prints')

    # ========== NEW: Anchored training flags ==========
    parser.add_argument('--anchor_train', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable anchored conditional training')
    parser.add_argument('--anchor_p_uncond', type=float, default=0.50,
                        help='Fraction of batches with zero anchors (preserves unconditional capability)')
    parser.add_argument('--anchor_frac_min', type=float, default=0.10,
                        help='Min fraction of points to use as anchors')
    parser.add_argument('--anchor_frac_max', type=float, default=0.30,
                        help='Max fraction of points to use as anchors')
    parser.add_argument('--anchor_min', type=int, default=8,
                        help='Minimum number of anchor points')
    parser.add_argument('--anchor_max', type=int, default=96,
                        help='Maximum number of anchor points')
    parser.add_argument('--anchor_mode', type=str, default='ball', choices=['ball', 'uniform', 'knn_bfs'],
                        help='Anchor sampling mode')
    parser.add_argument('--anchor_clamp_clean', action=argparse.BooleanOptionalAction, default=True,
                        help='Clamp anchor positions to clean values in noisy input')
    parser.add_argument('--anchor_mask_score_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Mask score loss to unknown points only')
    parser.add_argument('--anchor_warmup_steps', type=int, default=0,
                        help='Steps to linearly ramp anchored probability (0=no warmup)')
    parser.add_argument('--anchor_debug_every', type=int, default=200,
                        help='Debug print frequency for anchor stats')

    # ---- Stage C resume ----
    parser.add_argument('--resume_stageC_ckpt', type=str, default=None,
                        help='Path to a full Stage C checkpoint to resume from (must include generator).')
    parser.add_argument('--resume_reset_optimizer', action=argparse.BooleanOptionalAction, default=False,
                        help='If set, ignore optimizer state when resuming Stage C.')
    parser.add_argument('--skip_stageA', action=argparse.BooleanOptionalAction, default=False,
                        help='Skip Stage A training (useful when resuming from a full checkpoint).')

    
    # ========== ANCHOR GEOMETRY LOSSES ==========
    parser.add_argument('--anchor_geom_losses', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply anchor-clamping to structure losses (default: True)')
    parser.add_argument('--anchor_geom_mode', type=str, default='clamp_only',
                        choices=['clamp_only', 'clamp_and_mask'],
                        help='Anchor geometry mode (clamp_only recommended)')
    parser.add_argument('--anchor_geom_min_unknown', type=int, default=8,
                        help='Minimum unknown points to use anchor_geom')
    parser.add_argument('--anchor_geom_debug_every', type=int, default=200,
                        help='Debug logging interval for anchor geometry')
    
    # ========== INFERENCE MODE ==========
    parser.add_argument('--inference_mode', type=str, default='unanchored',
                        choices=['unanchored', 'anchored'],
                        help='Inference mode: unanchored (legacy) or anchored (new sequential)')
    
    # ========== CONTEXT REPLACEMENT INVARIANCE ==========
    parser.add_argument('--ctx_replace_variant', type=str, default='permute',
                        choices=['permute', 'hard'],
                        help='Context replacement variant: permute (random shuffle) or hard (most similar)')
    parser.add_argument('--ctx_loss_weight', type=float, default=0.0,
                        help='Context replacement invariance loss weight (lambda_ctx)')
    parser.add_argument('--ctx_replace_p', type=float, default=0.5,
                        help='Probability of applying context replacement per batch')
    parser.add_argument('--ctx_snr_thresh', type=float, default=0.3,
                        help='SNR threshold for context loss gating (0.3-0.5 recommended)')
    parser.add_argument('--ctx_warmup_steps', type=int, default=1000,
                        help='Warmup steps for context loss weight')
    parser.add_argument('--ctx_debug_every', type=int, default=100,
                        help='Debug print frequency for context replacement loss')

    # ========== SELF-CONDITIONING MODE ==========
    parser.add_argument('--self_cond_mode', type=str, default='standard',
                        choices=['none', 'standard'],
                        help='Self-conditioning mode: none (disabled) or standard (two-pass)')

    # ========== RESIDUAL DIFFUSION ==========
    parser.add_argument('--use_residual_diffusion', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable residual diffusion: diffuse R = V_target - V_base instead of V_target')
    parser.add_argument('--sigma_resid_recompute_step', type=int, default=3000,
                        help='Step at which to recompute sigma_data_resid (after generator warmup)')

    # ========== PAIRED OVERLAP TRAINING (Candidate 1) ==========
    parser.add_argument('--train_pair_overlap', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable paired overlapping minisets for overlap-consistency training')
    parser.add_argument('--pair_overlap_alpha', type=float, default=0.5,
                        help='Fraction of core points that overlap between paired minisets (0.3-0.7 recommended)')
    parser.add_argument('--pair_overlap_min_I', type=int, default=16,
                        help='Minimum number of overlapping core points required')
    parser.add_argument('--overlap_loss_weight_shape', type=float, default=1.0,
                        help='Weight for overlap shape consistency loss (scale-free Gram)')
    parser.add_argument('--overlap_loss_weight_scale', type=float, default=0.5,
                        help='Weight for overlap scale consistency loss (log-trace difference)')
    parser.add_argument('--overlap_loss_weight_kl', type=float, default=1.0,
                        help='Weight for overlap KL neighbor distribution loss (targets Jaccard)')
    parser.add_argument('--overlap_kl_tau', type=float, default=0.5,
                        help='Temperature for soft neighbor distribution in KL loss')
    parser.add_argument('--overlap_sigma_thresh', type=float, default=0.5,
                        help='Max sigma for overlap loss (SNR gating - only apply at low/mid noise)')
    parser.add_argument('--disable_ctx_loss_when_overlap', action=argparse.BooleanOptionalAction, default=True,
                        help='Disable context replacement loss when using paired overlap training')
    parser.add_argument('--overlap_debug_every', type=int, default=100,
                        help='Debug print frequency for overlap loss')

    # ========== NEW: Stage A VICReg + Adversary Arguments ==========
    parser.add_argument('--stageA_obj', type=str, default='geom',
                        choices=['geom', 'vicreg_adv'],
                        help='Stage A objective: geom (coord-based) or vicreg_adv (SSL)')
    
    # VICReg loss weights
    parser.add_argument('--vicreg_lambda_inv', type=float, default=25.0,
                        help='VICReg invariance loss weight')
    parser.add_argument('--vicreg_lambda_var', type=float, default=25.0,
                        help='VICReg variance loss weight')
    parser.add_argument('--vicreg_lambda_cov', type=float, default=1.0,
                        help='VICReg covariance loss weight')
    parser.add_argument('--vicreg_gamma', type=float, default=1.0,
                        help='VICReg target std (variance hinge threshold)')
    parser.add_argument('--vicreg_eps', type=float, default=1e-4,
                        help='VICReg numerical stability epsilon')
    parser.add_argument('--vicreg_project_dim', type=int, default=256,
                        help='VICReg projector output dimension')
    parser.add_argument('--vicreg_use_projector', action=argparse.BooleanOptionalAction, 
                        default=True,
                        help='Use projector head for VICReg')
    parser.add_argument('--vicreg_float32_stats', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Compute VICReg var/cov in fp32')
    parser.add_argument('--vicreg_ddp_gather', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Gather embeddings across GPUs for VICReg var/cov')
    
    # Expression augmentations
    parser.add_argument('--aug_gene_dropout', type=float, default=0.2,
                        help='Gene dropout probability')
    parser.add_argument('--aug_gauss_std', type=float, default=0.01,
                        help='Gaussian noise std')
    parser.add_argument('--aug_scale_jitter', type=float, default=0.2,
                        help='Scale jitter range [1-j, 1+j]')
    
    # Slide adversary
    parser.add_argument('--adv_slide_weight', type=float, default=1.0,
                        help='Slide adversary loss weight')
    parser.add_argument('--adv_warmup_epochs', type=int, default=10,
                        help='Adversary warmup epochs (alpha=0)')
    parser.add_argument('--adv_ramp_epochs', type=int, default=20,
                        help='Adversary ramp epochs (0 to alpha_max)')
    parser.add_argument('--grl_alpha_max', type=float, default=1.0,
                        help='Maximum GRL alpha value')
    parser.add_argument('--disc_hidden', type=int, default=256,
                        help='Discriminator hidden dimension')
    parser.add_argument('--disc_dropout', type=float, default=0.1,
                        help='Discriminator dropout')
    
    # Balanced slide sampling
    parser.add_argument('--stageA_balanced_slides', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Balance batch sampling across slides in Stage A')
    
    return parser.parse_args()


def load_hscc_data():
    """
    Load hSCC data: 2 ST slides for training, 1 ST slide for inference.
    Follows hSCC_gems.ipynb data loading structure.
    """
    print("Loading hSCC data...")
    
    # Load SC data (used as test data, like ST2 in mouse_brain)
    # scadata = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP2.h5ad')
    # Treat ST3 as the "SC" domain for now (so we have GT coords)
    scadata = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep3.h5ad')

    
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
    if 'level1_celltype' in scadata.obs.columns:
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
    else:
        print("No celltype metadata in scadata (ST3). Skipping celltype relabeling.")

    
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
    
    # ========== NEW: Auto-switch to anchored output directory if anchor training enabled ==========
    if args.anchor_train and 'anchored' not in outdir:
        outdir = outdir.replace('gems_hscc_2slides_output', 'gems_hscc_2slides_output_anchored')
        print(f"[ANCHOR-TRAIN] Auto-switching output dir to: {outdir}")

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
        isab_m=128,
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
        # ========== NEW: Anchored training configuration ==========
        anchor_train=args.anchor_train,
        anchor_geom_losses=args.anchor_geom_losses,
        anchor_geom_mode=args.anchor_geom_mode,
        anchor_geom_min_unknown=args.anchor_geom_min_unknown
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

        resume_ckpt_path = args.resume_stageC_ckpt
        skip_stageA = args.skip_stageA or (resume_ckpt_path is not None)

        if resume_ckpt_path and fabric.is_global_zero:
            print(f"[RESUME] Loading full checkpoint: {resume_ckpt_path}")
            ckpt = torch.load(resume_ckpt_path, map_location=fabric.device, weights_only=False)
            if 'encoder' in ckpt:
                model.encoder.load_state_dict(ckpt['encoder'])
            if 'context_encoder' in ckpt:
                model.context_encoder.load_state_dict(ckpt['context_encoder'])
            if 'generator' in ckpt:
                model.generator.load_state_dict(ckpt['generator'])
            if 'score_net' in ckpt:
                model.score_net.load_state_dict(ckpt['score_net'])
            print("[RESUME] Loaded encoder/context/generator/score_net (if present).")

        if fabric.is_global_zero:
            if skip_stageA:
                print("[Stage A] Skipped (resume or --skip_stageA enabled).")
            else:
                print("\n=== Stage A (single GPU, rank-0) ===")
                model.train_stageA(
                    st_gene_expr=st_expr,
                    st_coords=st_coords,
                    sc_gene_expr=sc_expr,
                    slide_ids=slide_ids,
                    n_epochs=stageA_epochs,
                    batch_size=256,
                    lr=1e-3,
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
        ck = torch.load(path, map_location=fabric.device, weights_only=False)
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
        n_min=96, n_max=448,
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
        curriculum_target_stage=args.curriculum_target_stage,
        curriculum_min_epochs=args.curriculum_min_epochs,
        curriculum_early_stop=args.curriculum_early_stop,
        use_legacy_curriculum=args.use_legacy_curriculum,
        # ========== DATA-DEPENDENT SIGMA CAP ==========
        sigma_cap_safe_mult=args.sigma_cap_safe_mult,
        sigma_cap_abs_max=args.sigma_cap_abs_max,
        sigma_cap_abs_min=args.sigma_cap_abs_min,
        # ========== NEW: Context augmentation ==========
        z_noise_std=0.0,       # No noise for Phase 1 (focus on clean geometry)
        z_dropout_rate=0.0,    # No dropout for Phase 1
        aug_prob=0.0,          # No augmentation for Phase 1
        # ========== COMPETITOR TRAINING (ChatGPT A/B test) ==========
        compete_train=args.compete_train,
        compete_n_extra=args.compete_n_extra,
        compete_n_rand=args.compete_n_rand,
        compete_n_hard=args.compete_n_hard,
        compete_use_pos_closure=args.compete_use_pos_closure,
        compete_k_pos=args.compete_k_pos,
        compete_expr_knn_k=args.compete_expr_knn_k,
        compete_anchor_only=args.compete_anchor_only,
        compete_diag_every=args.compete_diag_every,
        # ========== ANCHORED TRAINING ==========
        anchor_train=args.anchor_train,
        anchor_p_uncond=args.anchor_p_uncond,
        anchor_frac_min=args.anchor_frac_min,
        anchor_frac_max=args.anchor_frac_max,
        anchor_min=args.anchor_min,
        anchor_max=args.anchor_max,
        anchor_mode=args.anchor_mode,
        anchor_clamp_clean=args.anchor_clamp_clean,
        anchor_mask_score_loss=args.anchor_mask_score_loss,
        anchor_warmup_steps=args.anchor_warmup_steps,
        anchor_debug_every=args.anchor_debug_every,
        # ---- Resume Stage C ----
        resume_stageC_ckpt=args.resume_stageC_ckpt,
        resume_reset_optimizer=args.resume_reset_optimizer,
        # ========== CONTEXT REPLACEMENT INVARIANCE ==========
        ctx_replace_variant=args.ctx_replace_variant,
        ctx_loss_weight=args.ctx_loss_weight,
        ctx_replace_p=args.ctx_replace_p,
        ctx_snr_thresh=args.ctx_snr_thresh,
        ctx_warmup_steps=args.ctx_warmup_steps,
        ctx_debug_every=args.ctx_debug_every,
        # ========== SELF-CONDITIONING MODE ==========
        self_cond_mode=args.self_cond_mode,
        # ========== RESIDUAL DIFFUSION ==========
        use_residual_diffusion=args.use_residual_diffusion,
        sigma_resid_recompute_step=args.sigma_resid_recompute_step,
        # ========== PAIRED OVERLAP TRAINING (Candidate 1) ==========
        train_pair_overlap=args.train_pair_overlap,
        pair_overlap_alpha=args.pair_overlap_alpha,
        pair_overlap_min_I=args.pair_overlap_min_I,
        overlap_loss_weight_shape=args.overlap_loss_weight_shape,
        overlap_loss_weight_scale=args.overlap_loss_weight_scale,
        overlap_loss_weight_kl=args.overlap_loss_weight_kl,
        overlap_kl_tau=args.overlap_kl_tau,
        overlap_sigma_thresh=args.overlap_sigma_thresh,
        disable_ctx_loss_when_overlap=args.disable_ctx_loss_when_overlap,
        overlap_debug_every=args.overlap_debug_every,
        # ========== MINISET SAMPLING ==========
        pool_mult=args.pool_mult,
        stochastic_tau=args.stochastic_tau,
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
            # ========== UNWRAP DDP MODULES WHEN SAVING ==========
            'context_encoder': model.context_encoder.module.state_dict() if hasattr(model.context_encoder, 'module') else model.context_encoder.state_dict(),
            'generator': model.generator.module.state_dict() if hasattr(model.generator, 'module') else model.generator.state_dict(),
            'score_net': model.score_net.module.state_dict() if hasattr(model.score_net, 'module') else model.score_net.state_dict(),
            # ========== NEW: Save EDM parameters ==========
            'sigma_data': getattr(model, 'sigma_data', None),
            'sigma_min': getattr(model, 'sigma_min', None),
            'sigma_max': getattr(model, 'sigma_max', None),
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
            # ========== CONTEXT AUGMENTATION (Phase 2 can have slight augmentation) ==========
            z_noise_std=0.0,       # Optional: could add 0.02 for robustness
            z_dropout_rate=0.0,    # Optional: could add 0.1 for robustness
            aug_prob=0.0,          # Optional: could add 0.5 for robustness
            # ========== PAIRED OVERLAP TRAINING (Candidate 1) ==========
            # Note: In Phase 2 (SC fine-tune), overlap training may be less relevant
            # but we pass through for consistency
            train_pair_overlap=args.train_pair_overlap,
            pair_overlap_alpha=args.pair_overlap_alpha,
            pair_overlap_min_I=args.pair_overlap_min_I,
            overlap_loss_weight_shape=args.overlap_loss_weight_shape,
            overlap_loss_weight_scale=args.overlap_loss_weight_scale,
            overlap_loss_weight_kl=args.overlap_loss_weight_kl,
            overlap_kl_tau=args.overlap_kl_tau,
            overlap_sigma_thresh=args.overlap_sigma_thresh,
            disable_ctx_loss_when_overlap=args.disable_ctx_loss_when_overlap,
            overlap_debug_every=args.overlap_debug_every,
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
                # ========== NEW: Save EDM parameters ==========
                'sigma_data': getattr(model, 'sigma_data', None),
                'sigma_min': getattr(model, 'sigma_min', None),
                'sigma_max': getattr(model, 'sigma_max', None),
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
    
    # ========== CRITICAL: DDP TEARDOWN DISCIPLINE ==========
    # Non-rank-0 processes MUST exit cleanly to avoid inference hangs
    if not fabric.is_global_zero:
        print(f"[Rank {fabric.global_rank}] Training complete. Exiting (rank-0 will handle inference).")
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        import sys
        sys.exit(0)  # Fully exit non-rank-0 processes
    
    # ========== CRITICAL: Destroy DDP on rank-0 before single-GPU inference ==========
    print("[Rank-0] Destroying DDP process group...")
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
        print("[Rank-0] ✓ DDP process group destroyed")
    
    # Clear CUDA cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[Rank-0] GPU memory after DDP cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # ========== CRITICAL: Unwrap ALL DDP modules ==========
    if hasattr(model.encoder, 'module'):
        model.encoder = model.encoder.module
    if hasattr(model.context_encoder, 'module'):
        model.context_encoder = model.context_encoder.module
        print("[Rank-0] ✓ context_encoder unwrapped from DDP")
    if hasattr(model.score_net, 'module'):
        model.score_net = model.score_net.module
        print("[Rank-0] ✓ score_net unwrapped from DDP")
    if hasattr(model.generator, 'module'):
        model.generator = model.generator.module
        print("[Rank-0] ✓ generator unwrapped from DDP")
    print("[Rank-0] All modules unwrapped, ready for single-GPU inference\n")
    

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
    
    # # ========== COMPUTE CORAL TRANSFORMATION PARAMETERS ==========
    # print("\n" + "="*70)
    # print("COMPUTING CORAL TRANSFORMATION PARAMETERS")
    # print("="*70)
    
    # # Rebuild st_gene_expr_dict for CORAL computation (it's still in scope but make sure)
    # st1_expr_tensor = torch.tensor(X_st1, dtype=torch.float32, device=fabric.device)
    # st2_expr_tensor = torch.tensor(X_st2, dtype=torch.float32, device=fabric.device)
    # st_gene_expr_dict = {
    #     0: st1_expr_tensor,
    #     1: st2_expr_tensor
    # }
    
    # # Compute ST distribution from training slides
    # model.compute_coral_params_from_st(
    #     st_gene_expr_dict=st_gene_expr_dict,
    #     n_samples=2000,
    #     n_min=96,
    #     n_max=384,
    # )
    
    # # Compute SC distribution and build CORAL transform
    # # For HSCC: use SC data for transform (even though we infer on ST3)
    # model.build_coral_transform(
    #     sc_gene_expr=sc_expr,
    #     n_samples=2000,
    #     n_min=96,
    #     n_max=384,
    #     shrink=0.01,
    #     eps=1e-5,
    # )
    
    # print("✓ CORAL transformation ready for inference")


    # ---------- Inference (rank-0 only, single GPU) ----------
    # Now use stadata3 (3rd slide) for inference!
    if fabric.is_global_zero:
        print("[DEBUG Rank-0] Starting inference on 3rd slide (stadata3)...")

        # ===================================================================
        # GT COORDS: HSCC slide 3 doesn't have GT, so disable debug_knn
        # ===================================================================
        gt_coords_norm = None
        if 'spatial_gt' in stadata3.obsm:
            # If GT coords exist, load and normalize them
            gt_coords_raw = stadata3.obsm['spatial_gt']
            gt_coords_tensor = torch.tensor(gt_coords_raw, dtype=torch.float32, device=fabric.device)
            slide_ids_gt = torch.zeros(gt_coords_tensor.shape[0], dtype=torch.long, device=fabric.device)
            
            gt_coords_norm, gt_mu, gt_scale = uet.canonicalize_st_coords_per_slide(
                gt_coords_tensor, slide_ids_gt
            )
            
            print(f"✓ GT coords available and normalized: scale={gt_scale[0].item():.4f}")
            print(f"✓ GT coords RMS: {gt_coords_norm.pow(2).mean().sqrt().item():.4f}")
        else:
            print("✓ GT coords not available for slide 3 - disabling debug_knn")
        

        
        # Create timestamp for all outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ensure_disk_space(outdir, min_gb=10.0)

        # Prepare 3rd slide (inference slide) expression data
        X_st3 = stadata3[:, common].X
        if hasattr(X_st3, "toarray"): X_st3 = X_st3.toarray()
        st3_expr = torch.tensor(X_st3, dtype=torch.float32, device=fabric.device)

        print("\n=== Inference (rank-0) on 3rd Slide with Multi-Sample Ranking ===")

        # ===================================================================
        # LOAD CHECKPOINT AND RESTORE EDM PARAMETERS
        # ===================================================================
        # Load most recent checkpoint (Phase 2 if exists, else Phase 1)
        checkpoint_path_p2 = os.path.join(outdir, "phase2_sc_finetuned_checkpoint.pt")
        checkpoint_path_p1 = os.path.join(outdir, "phase1_st_checkpoint.pt")
        
        if os.path.exists(checkpoint_path_p2):
            checkpoint = torch.load(checkpoint_path_p2, map_location=fabric.device, weights_only=False)
            print(f"✓ Loaded checkpoint: phase2_sc_finetuned_checkpoint.pt")
        elif os.path.exists(checkpoint_path_p1):
            checkpoint = torch.load(checkpoint_path_p1, map_location=fabric.device, weights_only=False)
            print(f"✓ Loaded checkpoint: phase1_st_checkpoint.pt")
        else:
            checkpoint = {}
            print("⚠️ No checkpoint found - using current model state")
        
        # Restore EDM parameters from checkpoint
        if 'sigma_data' in checkpoint:
            model.sigma_data = checkpoint['sigma_data']
            print(f"✓ Restored sigma_data: {model.sigma_data:.4f}")
        if 'sigma_min' in checkpoint:
            model.sigma_min = checkpoint['sigma_min']
            print(f"✓ Restored sigma_min: {model.sigma_min:.6f}")
        if 'sigma_max' in checkpoint:
            model.sigma_max = checkpoint['sigma_max']
            print(f"✓ Restored sigma_max: {model.sigma_max:.2f}")
        
        if True:
            # Multi-sample inference with quality ranking
            K_samples = 1  # Number of samples to generate
            all_results = []
            all_scores = []

            with torch.no_grad():
                for k in range(K_samples):
                    print(f"\n  Generating sample {k+1}/{K_samples}...")
                    
                    # Set different seed for each sample
                    torch.manual_seed(42 + k)

                    n_cells = st3_expr.shape[0]

                    sample_results = model.infer_sc_patchwise(
                        sc_gene_expr=st3_expr,
                        n_timesteps_sample=500,  # Use 600 like mouse brain (you had 400)
                        return_coords=True,
                        patch_size=192,  # Use 192 like mouse brain (you had 256)
                        coverage_per_cell=6.0,
                        n_align_iters=15,  # Use 15 like mouse brain (you had 10)
                        eta=0.0,
                        guidance_scale=2.0,
                        # GT coords (None for HSCC slide 3)
                        gt_coords=gt_coords_norm,
                        debug_knn=(gt_coords_norm is not None),  # Only if GT available
                        debug_max_patches=15,
                        debug_k_list=(10, 20),
                        # ST-style stochastic sampling
                        pool_mult=2.0,
                        stochastic_tau=0.8,
                        tau_mode="adaptive_kth",
                        ensure_connected=True,
                        local_refine=False,
                        # V2 pipeline for residual diffusion
                        use_v2_pipeline=args.use_residual_diffusion,
                        use_residual_diffusion=args.use_residual_diffusion,
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
        # print("\n" + "="*70)
        # print("SINGLE-PATCH INFERENCE (NO STITCHING)")
        # print("="*70)
        
        # torch.manual_seed(42)  # Same seed as patchwise for fair comparison
        
        # # Compute ST p95 target using training slides
        # D_st_list = []
        # for coords_st in [st_coords1, st_coords2]:
        #     D = torch.cdist(
        #         torch.tensor(coords_st, dtype=torch.float32),
        #         torch.tensor(coords_st, dtype=torch.float32)
        #     )
        #     D_st_list.append(D)
        # D_st_combined = torch.cat([D[torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)] 
        #                            for D in D_st_list])
        # target_st_p95 = D_st_combined.quantile(0.95).item()
        
        # results_single = model.infer_sc_single_patch(
        #     sc_gene_expr=st3_expr,
        #     n_timesteps_sample=500,
        #     sigma_min=0.01,
        #     sigma_max=3.0,
        #     guidance_scale=2.0,
        #     eta=0.0,
        #     target_st_p95=target_st_p95,
        #     return_coords=True,
        #     DEBUG_FLAG=True,
        # )
        
        # # Extract results
        # D_edm_single = results_single['D_edm'].cpu().numpy()
        # coords_single = results_single['coords'].cpu().numpy()
        # coords_canon_single = results_single['coords_canon'].cpu().numpy()
        
        # print(f"\n✓ Single-patch inference complete!")
        # print(f"  D_edm shape: {D_edm_single.shape}")
        # print(f"  Coordinates shape: {coords_canon_single.shape}")
        
        # # ============================================================================
        # # SAVE SINGLE-PATCH RESULTS
        # # ============================================================================
        # results_single_filename = f"st3_inference_SINGLE_PATCH_{timestamp}.pt"
        # results_single_path = os.path.join(outdir, results_single_filename)
        # torch.save(results_single, results_single_path)
        # print(f"✓ Saved single-patch results: {results_single_path}")
        
        # # Save processed results
        # results_single_processed = {
        #     'D_edm': D_edm_single,
        #     'coords': coords_single,
        #     'coords_canon': coords_canon_single,
        #     'n_cells': coords_canon_single.shape[0],
        #     'timestamp': timestamp,
        #     'method': 'single_patch',
        #     'model_config': {
        #         'n_genes': model.n_genes,
        #         'D_latent': model.D_latent,
        #         'c_dim': model.c_dim,
        #     }
        # }
        # processed_single_filename = f"st3_inference_SINGLE_PATCH_processed_{timestamp}.pt"
        # processed_single_path = os.path.join(outdir, processed_single_filename)
        # torch.save(results_single_processed, processed_single_path)
        # print(f"✓ Saved processed single-patch results: {processed_single_path}")
        
        # ============================================================================
        # VISUALIZE SINGLE-PATCH RESULTS
        # ============================================================================
        # print("\n=== Creating single-patch visualizations ===")
        
        # fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # # Plot 1: Single-patch coordinates colored by cell type
        # if 'celltype' in stadata3.obs.columns:
        #     cell_types = stadata3.obs['celltype']
        #     unique_types = cell_types.unique()
        #     colors = sns.color_palette("husl", len(unique_types))
        #     color_map = dict(zip(unique_types, colors))
            
        #     for ct in unique_types:
        #         mask = (cell_types == ct).values
        #         axes[0].scatter(
        #             coords_canon_single[mask, 0], 
        #             coords_canon_single[mask, 1],
        #             s=3, 
        #             alpha=0.7, 
        #             label=ct,
        #             c=[color_map[ct]]
        #         )
            
        #     axes[0].set_title('Single-Patch Coordinates (by cell type)', fontsize=16, fontweight='bold')
        #     axes[0].set_xlabel('Dim 1', fontsize=12)
        #     axes[0].set_ylabel('Dim 2', fontsize=12)
        #     axes[0].legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
        #     axes[0].grid(True, alpha=0.3)
        # else:
        #     axes[0].scatter(coords_canon_single[:, 0], coords_canon_single[:, 1], s=3, alpha=0.7, c='steelblue')
        #     axes[0].set_title('Single-Patch Coordinates', fontsize=16, fontweight='bold')
        #     axes[0].set_xlabel('Dim 1', fontsize=12)
        #     axes[0].set_ylabel('Dim 2', fontsize=12)
        #     axes[0].grid(True, alpha=0.3)
        
        # axes[0].axis('equal')
        
        # # Plot 2: Distance distribution
        # upper_tri_idx = np.triu_indices_from(D_edm_single, k=1)
        # distances_single = D_edm_single[upper_tri_idx]
        
        # axes[1].hist(distances_single, bins=100, alpha=0.7, edgecolor='black', color='green')
        # axes[1].set_title('Pairwise Distance Distribution (Single-Patch)', fontsize=16, fontweight='bold')
        # axes[1].set_xlabel('Distance', fontsize=12)
        # axes[1].set_ylabel('Count', fontsize=12)
        # axes[1].axvline(distances_single.mean(), color='r', linestyle='--', linewidth=2, 
        #                label=f'Mean: {distances_single.mean():.2f}')
        # axes[1].axvline(np.median(distances_single), color='b', linestyle='--', linewidth=2, 
        #                label=f'Median: {np.median(distances_single):.2f}')
        # axes[1].legend(fontsize=10)
        # axes[1].grid(True, alpha=0.3, axis='y')
        
        # plt.tight_layout()
        
        # plot_single_filename = f"gems_inference_SINGLE_PATCH_{timestamp}.png"
        # plot_single_path = os.path.join(outdir, plot_single_filename)
        # plt.savefig(plot_single_path, dpi=300, bbox_inches='tight')
        # print(f"✓ Saved single-patch plot: {plot_single_path}")
        # plt.close()
        
        # ============================================================================
        # COMPARISON PLOT: Patchwise vs Single-Patch
        # ============================================================================
        # print("\n=== Creating comparison plot ===")
        
        # fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # # Patchwise (left)
        # if 'celltype' in stadata3.obs.columns:
        #     for ct in unique_types:
        #         mask = (cell_types == ct).values
        #         axes[0].scatter(
        #             coords_canon[mask, 0], 
        #             coords_canon[mask, 1],
        #             s=3, alpha=0.7, label=ct, c=[color_map[ct]]
        #         )
        #     axes[0].legend(markerscale=3, fontsize=8, loc='best', framealpha=0.9)
        # else:
        #     axes[0].scatter(coords_canon[:, 0], coords_canon[:, 1], s=3, alpha=0.7, c='steelblue')
        
        # axes[0].set_title('PATCHWISE (with alignment)', fontsize=16, fontweight='bold')
        # axes[0].set_xlabel('Dim 1', fontsize=12)
        # axes[0].set_ylabel('Dim 2', fontsize=12)
        # axes[0].grid(True, alpha=0.3)
        # axes[0].axis('equal')
        
        # # Single-patch (right)
        # if 'celltype' in stadata3.obs.columns:
        #     for ct in unique_types:
        #         mask = (cell_types == ct).values
        #         axes[1].scatter(
        #             coords_canon_single[mask, 0], 
        #             coords_canon_single[mask, 1],
        #             s=3, alpha=0.7, label=ct, c=[color_map[ct]]
        #         )
        #     axes[1].legend(markerscale=3, fontsize=8, loc='best', framealpha=0.9)
        # else:
        #     axes[1].scatter(coords_canon_single[:, 0], coords_canon_single[:, 1], s=3, alpha=0.7, c='green')
        
        # axes[1].set_title('SINGLE-PATCH (no alignment)', fontsize=16, fontweight='bold')
        # axes[1].set_xlabel('Dim 1', fontsize=12)
        # axes[1].set_ylabel('Dim 2', fontsize=12)
        # axes[1].grid(True, alpha=0.3)
        # axes[1].axis('equal')
        
        # plt.tight_layout()
        
        # comparison_filename = f"gems_COMPARISON_patchwise_vs_single_{timestamp}.png"
        # comparison_path = os.path.join(outdir, comparison_filename)
        # plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        # print(f"✓ Saved comparison plot: {comparison_path}")
        # plt.close()
        
        # # ============================================================================
        # # COMPARISON STATISTICS
        # # ============================================================================
        # print("\n" + "="*70)
        # print("COMPARISON: PATCHWISE vs SINGLE-PATCH")
        # print("="*70)

        # # Compute distances from D_edm (upper triangle)
        # upper_tri_idx = np.triu_indices_from(D_edm, k=1)
        # distances = D_edm[upper_tri_idx]
        
        # print(f"\nPATCHWISE:")
        # print(f"  Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]")
        # print(f"  Distance stats: mean={distances.mean():.4f}, median={np.median(distances):.4f}, std={distances.std():.4f}")
        
        # print(f"\nSINGLE-PATCH:")
        # print(f"  Coordinate range: [{coords_canon_single.min():.2f}, {coords_canon_single.max():.2f}]")
        # print(f"  Distance stats: mean={distances_single.mean():.4f}, median={np.median(distances_single):.4f}, std={distances_single.std():.4f}")
        
        # # Compute geometry metrics
        # def compute_geometry_metrics(coords):
        #     """Compute dimensionality and anisotropy metrics."""
        #     coords_centered = coords - coords.mean(axis=0)
        #     cov = np.cov(coords_centered.T)
        #     eigvals = np.linalg.eigvalsh(cov)
        #     eigvals = np.sort(eigvals)[::-1]  # Descending order
            
        #     # Effective dimensionality (participation ratio)
        #     dim_eff = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
            
        #     # Anisotropy ratio (top 2 eigenvalues)
        #     if len(eigvals) >= 2:
        #         aniso = eigvals[0] / (eigvals[1] + 1e-8)
        #     else:
        #         aniso = 1.0
            
        #     return dim_eff, aniso, eigvals
        
        # dim_pw, aniso_pw, eig_pw = compute_geometry_metrics(coords_canon)
        # dim_sp, aniso_sp, eig_sp = compute_geometry_metrics(coords_canon_single)
        
        # print(f"\nGEOMETRY METRICS:")
        # print(f"  Patchwise:    eff_dim={dim_pw:.2f}, anisotropy={aniso_pw:.2f}, top_eigs={eig_pw[:3]}")
        # print(f"  Single-patch: eff_dim={dim_sp:.2f}, anisotropy={aniso_sp:.2f}, top_eigs={eig_sp[:3]}")
        # print("="*70 + "\n")

        
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