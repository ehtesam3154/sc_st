#!/usr/bin/env python3
"""
hscc_p10_all.py

Train on P10 ST1 and ST2, with multi-patient encoder alignment (P2 + P10 SC data).
Phase 1 only: Stage A + Stage B + Stage C diffusion training.
No finetuning, no inference.

Usage:
    # With YAML config:
    python hscc_p10_all.py --config configs/hscc_p10_all.yaml

    # Or with CLI args:
    python hscc_p10_all.py --devices 2 --stageC_epochs 200
"""

import os
import sys
import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from lightning.fabric import Fabric

from core_models_et_p3 import GEMSModel
from core_models_et_p1 import SharedEncoder, train_encoder
import utils_et as uet
from ssl_utils import filter_informative_genes, set_seed


# ============================================================================
# STAGE A HYPERPARAMETERS (adjust here instead of core_models_et_p1.py)
# ============================================================================
PATIENT_CORAL_WEIGHT = 10.0   # Cross-patient alignment weight
ADV_SLIDE_WEIGHT = 50.0       # Slide adversary weight
ADV_WARMUP_EPOCHS = 50        # Epochs before adversary kicks in
ADV_RAMP_EPOCHS = 200         # Epochs to ramp adversary from 0 to max
# ============================================================================


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
            f"Please delete old checkpoints/plots/logs before running."
        )


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


def parse_args():
    parser = argparse.ArgumentParser(description='GEMS Training - P2 All (Multi-patient Stage A)')

    # Config file (loaded first, then CLI args override)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # Training config
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--stageA_epochs', type=int, default=1200)
    parser.add_argument('--stageC_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='gems_hscc_p10all_output')
    parser.add_argument('--num_st_samples', type=int, default=4000)

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
                        help='Pool multiplier for stochastic miniset sampling.')
    parser.add_argument('--stochastic_tau', type=float, default=1.0,
                        help='Temperature for stochastic sampling within pool')

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
                        help='Target curriculum stage for Gate A (0-indexed)')
    parser.add_argument('--curriculum_min_epochs', type=int, default=100,
                        help='Minimum epochs before three-gate stopping is allowed')
    parser.add_argument('--curriculum_early_stop', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable three-gate curriculum-aware early stopping')
    parser.add_argument('--use_legacy_curriculum', action=argparse.BooleanOptionalAction, default=False,
                        help='Use legacy 7-stage curriculum instead of new 3-stage')

    # ========== DATA-DEPENDENT SIGMA CAP ==========
    parser.add_argument('--sigma_cap_safe_mult', type=float, default=None,
                        help='Safety cap multiplier: sigma_cap_safe = mult * sigma0')
    parser.add_argument('--sigma_cap_abs_max', type=float, default=None,
                        help='Optional absolute max sigma_cap')
    parser.add_argument('--sigma_cap_abs_min', type=float, default=None,
                        help='Optional absolute min sigma_cap')

    # ========== COMPETITOR TRAINING FLAGS ==========
    parser.add_argument('--compete_train', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable competitor training')
    parser.add_argument('--compete_n_extra', type=int, default=128)
    parser.add_argument('--compete_n_rand', type=int, default=64)
    parser.add_argument('--compete_n_hard', type=int, default=64)
    parser.add_argument('--compete_use_pos_closure', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compete_k_pos', type=int, default=10)
    parser.add_argument('--compete_expr_knn_k', type=int, default=50)
    parser.add_argument('--compete_anchor_only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compete_diag_every', type=int, default=200)

    # ========== ANCHORED TRAINING FLAGS ==========
    parser.add_argument('--anchor_train', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable anchored conditional training')
    parser.add_argument('--anchor_p_uncond', type=float, default=0.50)
    parser.add_argument('--anchor_frac_min', type=float, default=0.10)
    parser.add_argument('--anchor_frac_max', type=float, default=0.30)
    parser.add_argument('--anchor_min', type=int, default=8)
    parser.add_argument('--anchor_max', type=int, default=96)
    parser.add_argument('--anchor_mode', type=str, default='ball', choices=['ball', 'uniform', 'knn_bfs'])
    parser.add_argument('--anchor_clamp_clean', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--anchor_mask_score_loss', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--anchor_warmup_steps', type=int, default=0)
    parser.add_argument('--anchor_debug_every', type=int, default=200)

    # ---- Stage C resume ----
    parser.add_argument('--resume_stageC_ckpt', type=str, default=None,
                        help='Path to a full Stage C checkpoint to resume from.')
    parser.add_argument('--resume_reset_optimizer', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--skip_stageA', action=argparse.BooleanOptionalAction, default=False,
                        help='Skip Stage A training.')

    # ========== ANCHOR GEOMETRY LOSSES ==========
    parser.add_argument('--anchor_geom_losses', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--anchor_geom_mode', type=str, default='clamp_only',
                        choices=['clamp_only', 'clamp_and_mask'])
    parser.add_argument('--anchor_geom_min_unknown', type=int, default=8)
    parser.add_argument('--anchor_geom_debug_every', type=int, default=200)

    # ========== GENERATOR CAPACITY ==========
    parser.add_argument('--gen_n_blocks', type=int, default=2)
    parser.add_argument('--gen_isab_m', type=int, default=None)

    # ========== INFERENCE MODE ==========
    parser.add_argument('--inference_mode', type=str, default='unanchored',
                        choices=['unanchored', 'anchored'])

    # ========== CONTEXT REPLACEMENT INVARIANCE ==========
    parser.add_argument('--ctx_replace_variant', type=str, default='permute',
                        choices=['permute', 'hard'])
    parser.add_argument('--ctx_loss_weight', type=float, default=0.0)
    parser.add_argument('--ctx_replace_p', type=float, default=0.5)
    parser.add_argument('--ctx_snr_thresh', type=float, default=0.3)
    parser.add_argument('--ctx_warmup_steps', type=int, default=1000)
    parser.add_argument('--ctx_debug_every', type=int, default=100)

    # ========== SELF-CONDITIONING MODE ==========
    parser.add_argument('--self_cond_mode', type=str, default='standard',
                        choices=['none', 'standard'])

    # ========== RESIDUAL DIFFUSION ==========
    parser.add_argument('--use_residual_diffusion', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sigma_resid_recompute_step', type=int, default=3000)

    # ========== PAIRED OVERLAP TRAINING ==========
    parser.add_argument('--train_pair_overlap', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pair_overlap_alpha', type=float, default=0.5)
    parser.add_argument('--pair_overlap_min_I', type=int, default=16)
    parser.add_argument('--overlap_loss_weight_shape', type=float, default=1.0)
    parser.add_argument('--overlap_loss_weight_scale', type=float, default=0.5)
    parser.add_argument('--overlap_loss_weight_kl', type=float, default=1.0)
    parser.add_argument('--overlap_kl_tau', type=float, default=0.5)
    parser.add_argument('--overlap_sigma_thresh', type=float, default=0.5)
    parser.add_argument('--disable_ctx_loss_when_overlap', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overlap_debug_every', type=int, default=100)

    # ========== Stage A VICReg + Adversary Arguments ==========
    parser.add_argument('--stageA_obj', type=str, default='vicreg_adv',
                        choices=['geom', 'vicreg_adv'],
                        help='Stage A objective (default: vicreg_adv for multi-patient)')

    # VICReg loss weights
    parser.add_argument('--vicreg_lambda_inv', type=float, default=25.0)
    parser.add_argument('--vicreg_lambda_var', type=float, default=25.0)
    parser.add_argument('--vicreg_lambda_cov', type=float, default=1.0)
    parser.add_argument('--vicreg_gamma', type=float, default=1.0)
    parser.add_argument('--vicreg_eps', type=float, default=1e-4)
    parser.add_argument('--vicreg_project_dim', type=int, default=256)
    parser.add_argument('--vicreg_use_projector', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vicreg_float32_stats', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vicreg_ddp_gather', action=argparse.BooleanOptionalAction, default=True)

    # Expression augmentations
    parser.add_argument('--aug_gene_dropout', type=float, default=0.2)
    parser.add_argument('--aug_gauss_std', type=float, default=0.01)
    parser.add_argument('--aug_scale_jitter', type=float, default=0.2)

    # Slide adversary (note: actual weights come from module-level constants)
    parser.add_argument('--grl_alpha_max', type=float, default=1.0)
    parser.add_argument('--disc_hidden', type=int, default=256)
    parser.add_argument('--disc_dropout', type=float, default=0.1)

    # Balanced slide sampling
    parser.add_argument('--stageA_balanced_slides', action=argparse.BooleanOptionalAction, default=True)

    # First parse to check for config file
    args, remaining = parser.parse_known_args()

    # If config file provided, load it and set as defaults
    if args.config is not None:
        config = load_config(args.config)
        # Convert YAML keys (may use underscores) to match argparse
        parser.set_defaults(**config)
        print(f"[CONFIG] Loaded config from: {args.config}")

    # Re-parse with config as defaults (CLI args override config)
    args = parser.parse_args()

    return args


def load_multipatient_data():
    """
    Load multi-patient hSCC data:
    - ST training: P10_ST1, P10_ST2
    - SC domain (for Stage A encoder alignment): P2_ST3, P2_SC, P10_ST3, P10_SC
    """
    print("Loading multi-patient hSCC data...")

    # ========== P10 TRAINING DATA ==========
    # ST training slides (P10)
    stadata1 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP10rep1.h5ad')
    stadata2 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP10rep2.h5ad')

    # P2 SC domain sources (for encoder alignment)
    p2_st3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep3.h5ad')
    p2_sc = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP2.h5ad')

    # ========== P10 SC domain sources ==========
    p10_st3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP10rep3.h5ad')
    p10_sc = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP10.h5ad')

    # Normalize and log transform
    print("Normalizing data...")
    all_data = [stadata1, stadata2, p2_st3, p2_sc, p10_st3, p10_sc]
    for adata in all_data:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    print(f"  P10_ST1: {stadata1.n_obs} cells")
    print(f"  P10_ST2: {stadata2.n_obs} cells")
    print(f"  P2_ST3: {p2_st3.n_obs} cells")
    print(f"  P2_SC:  {p2_sc.n_obs} cells")
    print(f"  P10_ST3: {p10_st3.n_obs} cells")
    print(f"  P10_SC:  {p10_sc.n_obs} cells")

    return stadata1, stadata2, p2_st3, p2_sc, p10_st3, p10_sc


def main(args=None):
    if args is None:
        args = parse_args()

    # Set seed for reproducibility (before any model creation)
    set_seed(args.seed)
    print(f"[SEED] Set random seed to {args.seed}")

    # Extract parameters
    devices = args.devices
    precision = args.precision
    stageA_epochs = args.stageA_epochs
    stageB_outdir = "gems_stageB_cache"
    stageC_epochs = args.stageC_epochs
    stageC_batch = args.batch_size
    lr = args.lr
    outdir = args.outdir

    # ========== AUTO-COMPUTE sigma_cap_safe_mult FROM curriculum_target_stage ==========
    curriculum_mults = [1.0, 2.0, 3.0, 4.0]
    target_stage_mult = curriculum_mults[min(args.curriculum_target_stage, len(curriculum_mults)-1)]
    if args.sigma_cap_safe_mult is None:
        args.sigma_cap_safe_mult = target_stage_mult
        print(f"[AUTO] sigma_cap_safe_mult set to {args.sigma_cap_safe_mult:.1f} (from curriculum_target_stage={args.curriculum_target_stage})")
    elif args.sigma_cap_safe_mult < target_stage_mult:
        print(f"\n{'='*70}")
        print(f"[WARNING] sigma_cap_safe_mult={args.sigma_cap_safe_mult:.1f} < curriculum_target_stage mult={target_stage_mult:.1f}")
        print(f"          This will PREVENT training from reaching the target stage's full sigma_cap!")
        print(f"{'='*70}\n")

    # Auto-switch to anchored output directory if anchor training enabled
    if args.anchor_train and 'anchored' not in outdir:
        outdir = outdir.replace('gems_hscc_p10all_output', 'gems_hscc_p10all_output_anchored')
        print(f"[ANCHOR-TRAIN] Auto-switching output dir to: {outdir}")

    fabric = Fabric(accelerator="gpu", devices=devices, strategy="ddp", precision=precision)
    fabric.launch()

    # ---------- Load data on all ranks ----------
    stadata1, stadata2, p2_st3, p2_sc, p10_st3, p10_sc = load_multipatient_data()

    # ---------- Get common genes across ALL 6 sources ----------
    common = sorted(list(
        set(stadata1.var_names) &
        set(stadata2.var_names) &
        set(p2_st3.var_names) &
        set(p2_sc.var_names) &
        set(p10_st3.var_names) &
        set(p10_sc.var_names)
    ))

    if fabric.is_global_zero:
        print(f"Common genes across all 6 sources: {len(common)}")

    # ---------- Filter to informative genes ----------
    def _extract(adata, genes):
        X = adata[:, genes].X
        return X.toarray() if hasattr(X, 'toarray') else X

    sources_for_filter = {
        'P2_ST1': _extract(stadata1, common),
        'P2_ST2': _extract(stadata2, common),
        'P2_ST3': _extract(p2_st3, common),
        'P2_SC': _extract(p2_sc, common),
        'P10_ST3': _extract(p10_st3, common),
        'P10_SC': _extract(p10_sc, common),
    }
    common = filter_informative_genes(sources_for_filter, common, verbose=fabric.is_global_zero)

    n_genes = len(common)
    if fabric.is_global_zero:
        print(f"Genes after filtering: {n_genes}")

    # ---------- Build model ----------
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
        anchor_train=args.anchor_train,
        anchor_geom_losses=args.anchor_geom_losses,
        anchor_geom_mode=args.anchor_geom_mode,
        anchor_geom_min_unknown=args.anchor_geom_min_unknown,
        gen_n_blocks=args.gen_n_blocks,
        gen_isab_m=args.gen_isab_m,
    )

    # ---------- Stage A & B on rank-0 only ----------
    if fabric.is_global_zero:
        # ========== ST DOMAIN: P2_ST1 + P2_ST2 ==========
        X_st1 = stadata1[:, common].X
        X_st2 = stadata2[:, common].X
        if hasattr(X_st1, "toarray"): X_st1 = X_st1.toarray()
        if hasattr(X_st2, "toarray"): X_st2 = X_st2.toarray()

        st_expr = torch.tensor(np.vstack([X_st1, X_st2]), dtype=torch.float32, device=fabric.device)

        st_coords1 = stadata1.obsm['spatial']
        st_coords2 = stadata2.obsm['spatial']
        st_coords_raw = torch.tensor(np.vstack([st_coords1, st_coords2]),
                                     dtype=torch.float32, device=fabric.device)

        # ST slide IDs (0 for ST1, 1 for ST2)
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

        # Debug print
        print("\n" + "="*60)
        print("COORDINATE NORMALIZATION DEBUG")
        print("="*60)
        print(f"Raw coords shape: {st_coords_raw.shape}")
        print(f"After canonicalize_st_coords_per_slide:")
        print(f"  st_mu: {st_mu.cpu().numpy()}")
        print(f"  st_scale: {st_scale.cpu().numpy()}")
        rms_canon = st_coords.pow(2).sum(dim=1).mean().sqrt().item()
        print(f"  RMS radius: {rms_canon:.4f} (should be ~1.0)")
        print("="*60 + "\n")

        # Save canonicalization stats
        os.makedirs(outdir, exist_ok=True)
        torch.save(
            {"mu": st_mu.cpu(), "scale": st_scale.cpu()},
            os.path.join(outdir, "st_slide_canon_stats.pt"),
        )
        print(f"[Rank-0] Per-slide canonicalization: {len(torch.unique(slide_ids))} slides")

        # ========== SC DOMAIN: P2_ST3 + P2_SC + P10_ST3 + P10_SC ==========
        X_p2_st3 = p2_st3[:, common].X
        X_p2_sc = p2_sc[:, common].X
        X_p10_st3 = p10_st3[:, common].X
        X_p10_sc = p10_sc[:, common].X

        if hasattr(X_p2_st3, "toarray"): X_p2_st3 = X_p2_st3.toarray()
        if hasattr(X_p2_sc, "toarray"): X_p2_sc = X_p2_sc.toarray()
        if hasattr(X_p10_st3, "toarray"): X_p10_st3 = X_p10_st3.toarray()
        if hasattr(X_p10_sc, "toarray"): X_p10_sc = X_p10_sc.toarray()

        sc_expr = torch.tensor(
            np.vstack([X_p2_st3, X_p2_sc, X_p10_st3, X_p10_sc]),
            dtype=torch.float32, device=fabric.device
        )

        # SC slide IDs: 0=P2_ST3, 1=P2_SC, 2=P10_ST3, 3=P10_SC
        n_p2_st3 = X_p2_st3.shape[0]
        n_p2_sc = X_p2_sc.shape[0]
        n_p10_st3 = X_p10_st3.shape[0]
        n_p10_sc = X_p10_sc.shape[0]

        sc_slide_ids = torch.tensor(
            np.concatenate([
                np.zeros(n_p2_st3, dtype=int),
                np.ones(n_p2_sc, dtype=int),
                np.full(n_p10_st3, 2, dtype=int),
                np.full(n_p10_sc, 3, dtype=int),
            ]),
            dtype=torch.long, device=fabric.device
        )

        # SC patient IDs: 0=P2, 1=P10
        sc_patient_ids = torch.tensor(
            np.concatenate([
                np.zeros(n_p2_st3, dtype=int),      # P2_ST3 -> patient 0
                np.zeros(n_p2_sc, dtype=int),       # P2_SC -> patient 0
                np.ones(n_p10_st3, dtype=int),      # P10_ST3 -> patient 1
                np.ones(n_p10_sc, dtype=int),       # P10_SC -> patient 1
            ]),
            dtype=torch.long, device=fabric.device
        )

        print(f"\n[Stage A] Multi-patient SC domain setup:")
        print(f"  SC total: {sc_expr.shape[0]} cells")
        print(f"  SC slides: {len(torch.unique(sc_slide_ids))} (P2_ST3, P2_SC, P10_ST3, P10_SC)")
        print(f"  SC patients: {len(torch.unique(sc_patient_ids))} (P2, P10)")
        print(f"  patient_coral_weight: {PATIENT_CORAL_WEIGHT}")
        print(f"  adv_slide_weight: {ADV_SLIDE_WEIGHT}")

        # ========== STAGE A ==========
        resume_ckpt_path = args.resume_stageC_ckpt
        skip_stageA = args.skip_stageA or (resume_ckpt_path is not None)

        if resume_ckpt_path:
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

        if skip_stageA:
            print("[Stage A] Skipped (resume or --skip_stageA enabled).")
        else:
            print("\n=== Stage A (single GPU, rank-0) ===")
            print("Using train_encoder directly for multi-patient support")

            # Create encoder for Stage A training
            encoder_stageA = SharedEncoder(
                n_genes=n_genes,
                n_embedding=[512, 256, 128],
                dropout=0.1
            ).to(fabric.device)

            # Call train_encoder directly with all parameters
            encoder_stageA, projector, discriminator, hist = train_encoder(
                model=encoder_stageA,
                st_gene_expr=st_expr,
                st_coords=st_coords,
                sc_gene_expr=sc_expr,
                slide_ids=slide_ids,
                sc_slide_ids=sc_slide_ids,
                sc_patient_ids=sc_patient_ids,
                n_epochs=stageA_epochs,
                batch_size=256,
                lr=1e-3,
                device=str(fabric.device),
                outf=outdir,
                # Stage A objective
                stageA_obj=args.stageA_obj,
                # VICReg params
                vicreg_lambda_inv=args.vicreg_lambda_inv,
                vicreg_lambda_var=args.vicreg_lambda_var,
                vicreg_lambda_cov=args.vicreg_lambda_cov,
                vicreg_gamma=args.vicreg_gamma,
                vicreg_eps=args.vicreg_eps,
                vicreg_project_dim=args.vicreg_project_dim,
                vicreg_use_projector=args.vicreg_use_projector,
                vicreg_float32_stats=args.vicreg_float32_stats,
                vicreg_ddp_gather=False,  # Single GPU for Stage A
                # Augmentation params
                aug_gene_dropout=args.aug_gene_dropout,
                aug_gauss_std=args.aug_gauss_std,
                aug_scale_jitter=args.aug_scale_jitter,
                # Adversary params (using module-level constants)
                adv_slide_weight=ADV_SLIDE_WEIGHT,
                adv_warmup_epochs=ADV_WARMUP_EPOCHS,
                adv_ramp_epochs=ADV_RAMP_EPOCHS,
                grl_alpha_max=args.grl_alpha_max,
                disc_hidden=args.disc_hidden,
                disc_dropout=args.disc_dropout,
                # Multi-patient CORAL
                patient_coral_weight=PATIENT_CORAL_WEIGHT,
                # Balanced sampling
                stageA_balanced_slides=args.stageA_balanced_slides,
                # Representation mode
                adv_representation_mode='clean',
                adv_use_layernorm=True,
                adv_log_diagnostics=True,
                adv_log_grad_norms=False,
                # Local alignment
                use_local_align=True,
                local_align_bidirectional=True,
                local_align_weight=4.0,
                local_align_tau_z=0.07,
                # Return auxiliary outputs
                return_aux=True,
                # Use best checkpoint
                use_best_checkpoint=True,
                # Reproducibility
                seed=args.seed,
            )

            # Assign trained encoder to model
            model.encoder = encoder_stageA

            # Freeze encoder for subsequent stages
            model.encoder.eval()
            for param in model.encoder.parameters():
                param.requires_grad = False

            print("Stage A complete. Encoder frozen.")

        # ========== STAGE B ==========
        print("\n=== Stage B (single GPU, rank-0) ===")
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

        # Save checkpoint after Stage A/B for other ranks to load
        ckpt_ab = {
            "encoder": model.encoder.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "score_net": model.score_net.state_dict(),
        }
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

        # Recompute Stage B targets on non-rank-0
        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")

        X_st1 = stadata1[:, common].X
        X_st2 = stadata2[:, common].X
        if hasattr(X_st1, "toarray"): X_st1 = X_st1.toarray()
        if hasattr(X_st2, "toarray"): X_st2 = X_st2.toarray()

        st_expr_rank = torch.tensor(np.vstack([X_st1, X_st2]), dtype=torch.float32)

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

    # Rebuild tensors on each rank
    X_st1 = stadata1[:, common].X
    X_st2 = stadata2[:, common].X
    if hasattr(X_st1, "toarray"): X_st1 = X_st1.toarray()
    if hasattr(X_st2, "toarray"): X_st2 = X_st2.toarray()

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

    st_coords, _, _ = uet.canonicalize_st_coords_per_slide(st_coords_raw, slide_ids)
    print(f"[Rank {fabric.global_rank}] Stage C: Applied per-slide canonicalization")

    # Create st_gene_expr_dict for Stage C
    st1_expr_tensor = torch.tensor(X_st1, dtype=torch.float32, device=fabric.device)
    st2_expr_tensor = torch.tensor(X_st2, dtype=torch.float32, device=fabric.device)
    st_gene_expr_dict = {
        0: st1_expr_tensor,
        1: st2_expr_tensor
    }

    # SC expression for Stage C (all 4 sources combined)
    X_p2_st3 = p2_st3[:, common].X
    X_p2_sc = p2_sc[:, common].X
    X_p10_st3 = p10_st3[:, common].X
    X_p10_sc = p10_sc[:, common].X
    if hasattr(X_p2_st3, "toarray"): X_p2_st3 = X_p2_st3.toarray()
    if hasattr(X_p2_sc, "toarray"): X_p2_sc = X_p2_sc.toarray()
    if hasattr(X_p10_st3, "toarray"): X_p10_st3 = X_p10_st3.toarray()
    if hasattr(X_p10_sc, "toarray"): X_p10_sc = X_p10_sc.toarray()

    sc_expr = torch.tensor(
        np.vstack([X_p2_st3, X_p2_sc, X_p10_st3, X_p10_sc]),
        dtype=torch.float32, device=fabric.device
    )

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
        sigma_cap_safe_mult=args.sigma_cap_safe_mult,
        sigma_cap_abs_max=args.sigma_cap_abs_max,
        sigma_cap_abs_min=args.sigma_cap_abs_min,
        z_noise_std=0.0,
        z_dropout_rate=0.0,
        aug_prob=0.0,
        compete_train=args.compete_train,
        compete_n_extra=args.compete_n_extra,
        compete_n_rand=args.compete_n_rand,
        compete_n_hard=args.compete_n_hard,
        compete_use_pos_closure=args.compete_use_pos_closure,
        compete_k_pos=args.compete_k_pos,
        compete_expr_knn_k=args.compete_expr_knn_k,
        compete_anchor_only=args.compete_anchor_only,
        compete_diag_every=args.compete_diag_every,
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
        resume_stageC_ckpt=args.resume_stageC_ckpt,
        resume_reset_optimizer=args.resume_reset_optimizer,
        ctx_replace_variant=args.ctx_replace_variant,
        ctx_loss_weight=args.ctx_loss_weight,
        ctx_replace_p=args.ctx_replace_p,
        ctx_snr_thresh=args.ctx_snr_thresh,
        ctx_warmup_steps=args.ctx_warmup_steps,
        ctx_debug_every=args.ctx_debug_every,
        self_cond_mode=args.self_cond_mode,
        use_residual_diffusion=args.use_residual_diffusion,
        sigma_resid_recompute_step=args.sigma_resid_recompute_step,
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
        pool_mult=args.pool_mult,
        stochastic_tau=args.stochastic_tau,
    )

    fabric.barrier()

    # ========== SAVE CHECKPOINT (Phase 1 complete) ==========
    if fabric.is_global_zero:
        if history_st and history_st.get('early_stopped', False):
            E_ST_best = history_st['early_stop_info']['epoch']
            print(f"\n[Phase 1] Early stopped at epoch {E_ST_best}")
        else:
            E_ST_best = len(history_st['epoch']) if history_st else stageC_epochs
            print(f"\n[Phase 1] Completed all {E_ST_best} epochs")

        checkpoint_path = os.path.join(outdir, "phase1_st_checkpoint.pt")
        checkpoint = {
            'encoder': model.encoder.state_dict(),
            'context_encoder': model.context_encoder.module.state_dict() if hasattr(model.context_encoder, 'module') else model.context_encoder.state_dict(),
            'generator': model.generator.module.state_dict() if hasattr(model.generator, 'module') else model.generator.state_dict(),
            'score_net': model.score_net.module.state_dict() if hasattr(model.score_net, 'module') else model.score_net.state_dict(),
            'sigma_data': getattr(model, 'sigma_data', None),
            'sigma_min': getattr(model, 'sigma_min', None),
            'sigma_max': getattr(model, 'sigma_max', None),
            'E_ST_best': E_ST_best,
            'lr_ST': lr,
            'history_st': history_st,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved Phase 1 checkpoint: {checkpoint_path}")
        print(f"  - Best ST epoch: {E_ST_best}")
        print(f"  - ST learning rate: {lr:.2e}")

    fabric.barrier()

    # ========== PLOT PHASE 1 LOSSES (only non-zero weighted) ==========
    if fabric.is_global_zero:
        print("\n=== Plotting Phase 1 training losses ===")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if history_st is not None and len(history_st['epoch']) > 0:
            epochs = history_st['epoch']
            losses = history_st['epoch_avg']

            # Dynamically find non-zero losses
            active_losses = []
            for name, vals in losses.items():
                if len(vals) > 0 and any(abs(v) > 1e-9 for v in vals):
                    active_losses.append(name)

            # Sort for consistent ordering (total first, then alphabetical)
            if 'total' in active_losses:
                active_losses.remove('total')
                active_losses = ['total'] + sorted(active_losses)
            else:
                active_losses = sorted(active_losses)

            print(f"Active losses to plot: {active_losses}")

            # Create subplot grid
            n_losses = len(active_losses)
            n_cols = 3
            n_rows = (n_losses + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            fig.suptitle('Phase 1: ST-Only Training Losses (Non-Zero Only)',
                        fontsize=18, fontweight='bold', y=0.995)

            axes = axes.flatten() if n_losses > 1 else [axes]

            # Color cycle
            colors = plt.cm.tab20(np.linspace(0, 1, 20))

            for idx, name in enumerate(active_losses):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                color = colors[idx % len(colors)]

                vals = losses[name]
                ax.plot(epochs, vals, color=color, linewidth=2, alpha=0.7, marker='o', markersize=3)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Loss', fontsize=12)
                ax.set_title(f'{name.upper()}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                if len(epochs) > 10:
                    smoothed = gaussian_filter1d(vals, sigma=2)
                    ax.plot(epochs, smoothed, '--', color=color, linewidth=2.5, alpha=0.5, label='Trend')
                    ax.legend(fontsize=10)

            # Hide unused subplots
            for idx in range(n_losses, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plot_filename = f"stageC_phase1_losses_{timestamp}.png"
            plot_path = os.path.join(outdir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Phase 1 loss plot: {plot_path}")
            plt.close()

        # ========== SUMMARY ==========
        print("\n" + "="*70)
        print("TRAINING COMPLETE (Phase 1 only)")
        print("="*70)
        print(f"\nOutput directory: {outdir}")
        print(f"Checkpoints saved:")
        print(f"  - {os.path.join(outdir, 'ab_init.pt')}")
        print(f"  - {os.path.join(outdir, 'phase1_st_checkpoint.pt')}")
        print(f"  - {os.path.join(outdir, 'encoder_final_new.pt')} (from Stage A)")
        print(f"\nStage A config:")
        print(f"  - patient_coral_weight: {PATIENT_CORAL_WEIGHT}")
        print(f"  - adv_slide_weight: {ADV_SLIDE_WEIGHT}")
        print(f"  - adv_warmup_epochs: {ADV_WARMUP_EPOCHS}")
        print(f"  - adv_ramp_epochs: {ADV_RAMP_EPOCHS}")
        print("="*70)


if __name__ == "__main__":
    main()
