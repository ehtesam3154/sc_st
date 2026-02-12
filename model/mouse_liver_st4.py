#!/usr/bin/env python3
"""
liver_st_st4.py

Train on liver ST1, ST2, ST3 (ST domain) with ST4 (inference domain) for encoder alignment.
Phase 1 only: Stage A + Stage B + Stage C diffusion training.
Same patient (mouse) — cross-slide gap only.

Usage:
    python liver_st_st4.py --config configs/liver_st_st4.yaml
    python liver_st_st4.py --devices 2 --stageC_epochs 200
"""

import os
import sys
import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
import scanpy as sc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from lightning.fabric import Fabric

from core_models_et_p3 import GEMSModel
from core_models_et_p1 import SharedEncoder, train_encoder
import utils_et as uet
from ssl_utils import set_seed, filter_informative_genes


# ============================================================================
# STAGE A HYPERPARAMETERS
# ============================================================================
PATIENT_CORAL_WEIGHT = 0.0        # Same patient (mouse)
ADV_SLIDE_WEIGHT = 75.0           # Slide adversary weight
ADV_WARMUP_EPOCHS = 50
ADV_RAMP_EPOCHS = 200
MMD_WEIGHT = 30.0                 # MMD for cross-slide alignment
CORAL_RAW_WEIGHT = 2.0            # CORAL on raw embeddings
SOURCE_CORAL_WEIGHT = 75.0        # Source CORAL weight
LOCAL_ALIGN_WEIGHT = 6.0          # Local alignment weight

# ---- Stage A guard ----
ENABLE_STAGEA_GUARD = True
STAGEA_GUARD_MAX_SAMPLES = 2000
STAGEA_GUARD_K = 20
STAGEA_GUARD_MARGIN = 0.08
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
    parser = argparse.ArgumentParser(description='GEMS Training - Liver Cross-Slide')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # Training config
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stageA_epochs', type=int, default=2000)
    parser.add_argument('--stageC_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='gems_liver_st4')
    parser.add_argument('--num_st_samples', type=int, default=6000)

    # Geometry-aware diffusion
    parser.add_argument('--use_canonicalize', action='store_true', default=True)
    parser.add_argument('--use_dist_bias', action='store_true', default=True)
    parser.add_argument('--dist_bins', type=int, default=24)
    parser.add_argument('--dist_head_shared', action='store_true', default=True)
    parser.add_argument('--use_angle_features', action='store_true', default=True)
    parser.add_argument('--angle_bins', type=int, default=8)
    parser.add_argument('--knn_k', type=int, default=15)
    parser.add_argument('--self_conditioning', action='store_true', default=True)
    parser.add_argument('--sc_feat_mode', type=str, default='concat', choices=['concat', 'mlp'])
    parser.add_argument('--landmarks_L', type=int, default=16)

    # Miniset sampling
    parser.add_argument('--pool_mult', type=float, default=2.0)
    parser.add_argument('--stochastic_tau', type=float, default=1.0)

    # Early stopping
    parser.add_argument('--enable_early_stop', action='store_true', default=False)
    parser.add_argument('--early_stop_min_epochs', type=int, default=120)
    parser.add_argument('--early_stop_patience', type=int, default=6)
    parser.add_argument('--early_stop_threshold', type=float, default=0.01)

    # Curriculum
    parser.add_argument('--curriculum_target_stage', type=int, default=3)
    parser.add_argument('--curriculum_min_epochs', type=int, default=200)
    parser.add_argument('--curriculum_early_stop', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_legacy_curriculum', action=argparse.BooleanOptionalAction, default=False)

    # Sigma cap
    parser.add_argument('--sigma_cap_safe_mult', type=float, default=None)
    parser.add_argument('--sigma_cap_abs_max', type=float, default=None)
    parser.add_argument('--sigma_cap_abs_min', type=float, default=None)

    # Competitor training
    parser.add_argument('--compete_train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--compete_n_extra', type=int, default=128)
    parser.add_argument('--compete_n_rand', type=int, default=64)
    parser.add_argument('--compete_n_hard', type=int, default=64)
    parser.add_argument('--compete_use_pos_closure', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compete_k_pos', type=int, default=10)
    parser.add_argument('--compete_expr_knn_k', type=int, default=50)
    parser.add_argument('--compete_anchor_only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compete_diag_every', type=int, default=200)

    # Anchored training
    parser.add_argument('--anchor_train', action=argparse.BooleanOptionalAction, default=False)
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

    # Stage C resume
    parser.add_argument('--resume_stageC_ckpt', type=str, default=None)
    parser.add_argument('--resume_reset_optimizer', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--skip_stageA', action=argparse.BooleanOptionalAction, default=False)

    # Anchor geometry losses
    parser.add_argument('--anchor_geom_losses', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--anchor_geom_mode', type=str, default='clamp_only',
                        choices=['clamp_only', 'clamp_and_mask'])
    parser.add_argument('--anchor_geom_min_unknown', type=int, default=8)
    parser.add_argument('--anchor_geom_debug_every', type=int, default=200)

    # Generator capacity
    parser.add_argument('--gen_n_blocks', type=int, default=4)
    parser.add_argument('--gen_isab_m', type=int, default=128)

    # Context replacement invariance
    parser.add_argument('--ctx_replace_variant', type=str, default='permute', choices=['permute', 'hard'])
    parser.add_argument('--ctx_loss_weight', type=float, default=0.0)
    parser.add_argument('--ctx_replace_p', type=float, default=0.5)
    parser.add_argument('--ctx_snr_thresh', type=float, default=0.3)
    parser.add_argument('--ctx_warmup_steps', type=int, default=1000)
    parser.add_argument('--ctx_debug_every', type=int, default=100)

    # Self-conditioning mode
    parser.add_argument('--self_cond_mode', type=str, default='standard', choices=['none', 'standard'])

    # Residual diffusion
    parser.add_argument('--use_residual_diffusion', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sigma_resid_recompute_step', type=int, default=3000)

    # Paired overlap training
    parser.add_argument('--train_pair_overlap', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pair_overlap_alpha', type=float, default=0.5)
    parser.add_argument('--pair_overlap_min_I', type=int, default=20)
    parser.add_argument('--overlap_loss_weight_shape', type=float, default=1.0)
    parser.add_argument('--overlap_loss_weight_scale', type=float, default=0.5)
    parser.add_argument('--overlap_loss_weight_kl', type=float, default=1.0)
    parser.add_argument('--overlap_kl_tau', type=float, default=0.5)
    parser.add_argument('--overlap_sigma_thresh', type=float, default=1.0)
    parser.add_argument('--disable_ctx_loss_when_overlap', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overlap_debug_every', type=int, default=100)

    # Stage A VICReg + Adversary
    parser.add_argument('--stageA_obj', type=str, default='vicreg_adv', choices=['geom', 'vicreg_adv'])
    parser.add_argument('--vicreg_lambda_inv', type=float, default=25.0)
    parser.add_argument('--vicreg_lambda_var', type=float, default=50.0)
    parser.add_argument('--vicreg_lambda_cov', type=float, default=1.0)
    parser.add_argument('--vicreg_gamma', type=float, default=1.0)
    parser.add_argument('--vicreg_eps', type=float, default=1e-4)
    parser.add_argument('--vicreg_project_dim', type=int, default=256)
    parser.add_argument('--vicreg_use_projector', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--vicreg_float32_stats', action=argparse.BooleanOptionalAction, default=True)

    # Expression augmentations
    parser.add_argument('--aug_gene_dropout', type=float, default=0.25)
    parser.add_argument('--aug_gauss_std', type=float, default=0.01)
    parser.add_argument('--aug_scale_jitter', type=float, default=0.1)

    # Slide adversary
    parser.add_argument('--grl_alpha_max', type=float, default=1.0)
    parser.add_argument('--disc_hidden', type=int, default=512)
    parser.add_argument('--disc_dropout', type=float, default=0.1)

    # Balanced slide sampling
    parser.add_argument('--stageA_balanced_slides', action=argparse.BooleanOptionalAction, default=True)

    # First parse to check for config file
    args, remaining = parser.parse_known_args()

    if args.config is not None:
        config = load_config(args.config)
        parser.set_defaults(**config)
        print(f"[CONFIG] Loaded config from: {args.config}")

    args = parser.parse_args()

    return args


def load_liver_data():
    """
    Load mouse liver data:
    - ST training: ST1, ST2, ST3
    - Inference target: ST4
    """
    print("Loading mouse liver data...")

    st1 = sc.read_h5ad('/home/ehtesamul/sc_st/data/liver/stadata1.h5ad')
    st2 = sc.read_h5ad('/home/ehtesamul/sc_st/data/liver/stadata2.h5ad')
    st3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/liver/stadata3.h5ad')
    st4 = sc.read_h5ad('/home/ehtesamul/sc_st/data/liver/stadata4.h5ad')

    print("Normalizing data...")
    all_data = [st1, st2, st3, st4]
    for adata in all_data:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    print(f"  liver_ST1: {st1.n_obs} spots")
    print(f"  liver_ST2: {st2.n_obs} spots")
    print(f"  liver_ST3: {st3.n_obs} spots")
    print(f"  liver_ST4: {st4.n_obs} spots (inference target)")

    return st1, st2, st3, st4


def main(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    print(f"[SEED] Set random seed to {args.seed}")

    devices = args.devices
    precision = args.precision
    stageA_epochs = args.stageA_epochs
    stageB_outdir = "gems_stageB_cache"
    stageC_epochs = args.stageC_epochs
    stageC_batch = args.batch_size
    lr = args.lr
    outdir = args.outdir

    # Auto-compute sigma_cap_safe_mult
    curriculum_mults = [1.0, 2.0, 3.0, 4.0]
    target_stage_mult = curriculum_mults[min(args.curriculum_target_stage, len(curriculum_mults)-1)]
    if args.sigma_cap_safe_mult is None:
        args.sigma_cap_safe_mult = target_stage_mult
        print(f"[AUTO] sigma_cap_safe_mult set to {args.sigma_cap_safe_mult:.1f}")

    fabric = Fabric(accelerator="gpu", devices=devices, strategy="ddp", precision=precision)
    fabric.launch()

    # Load data on all ranks
    st1, st2, st3, st4 = load_liver_data()

    # Common genes across all 4 sources
    common = sorted(list(
        set(st1.var_names) &
        set(st2.var_names) &
        set(st3.var_names) &
        set(st4.var_names)
    ))

    if fabric.is_global_zero:
        print(f"Common genes across all 4 sources: {len(common)}")

    def _extract_np(adata, genes):
        X = adata[:, genes].X
        return X.toarray() if hasattr(X, 'toarray') else X

    sources_for_filter = {
        'liver_ST1': _extract_np(st1, common),
        'liver_ST2': _extract_np(st2, common),
        'liver_ST3': _extract_np(st3, common),
        'liver_ST4': _extract_np(st4, common),
    }
    common = filter_informative_genes(sources_for_filter, common, max_zero_frac=0.85, min_variance=0.01, verbose=fabric.is_global_zero)

    n_genes = len(common)
    if fabric.is_global_zero:
        print(f"Genes after filtering: {n_genes}")

    # Build model
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

    # Stage A & B on rank-0 only
    if fabric.is_global_zero:
        def _extract(adata, genes):
            X = adata[:, genes].X
            return X.toarray() if hasattr(X, 'toarray') else X

        # ========== ST DOMAIN: ST1 + ST2 + ST3 ==========
        X_st1 = _extract(st1, common)
        X_st2 = _extract(st2, common)
        X_st3 = _extract(st3, common)

        st_expr = torch.tensor(np.vstack([X_st1, X_st2, X_st3]), dtype=torch.float32, device=fabric.device)

        st_coords_raw = torch.tensor(
            np.vstack([st1.obsm['spatial'], st2.obsm['spatial'], st3.obsm['spatial']]),
            dtype=torch.float32, device=fabric.device
        )

        slide_ids = torch.tensor(
            np.concatenate([
                np.zeros(X_st1.shape[0], dtype=int),
                np.ones(X_st2.shape[0], dtype=int),
                np.full(X_st3.shape[0], 2, dtype=int)
            ]),
            dtype=torch.long, device=fabric.device
        )

        st_coords, st_mu, st_scale = uet.canonicalize_st_coords_per_slide(st_coords_raw, slide_ids)

        print("\n" + "=" * 60)
        print("COORDINATE NORMALIZATION DEBUG")
        print("=" * 60)
        print(f"Raw coords shape: {st_coords_raw.shape}")
        print(f"After canonicalize_st_coords_per_slide:")
        print(f"  st_mu: {st_mu.cpu().numpy()}")
        print(f"  st_scale: {st_scale.cpu().numpy()}")
        rms_canon = st_coords.pow(2).sum(dim=1).mean().sqrt().item()
        print(f"  RMS radius: {rms_canon:.4f} (should be ~1.0)")
        print("=" * 60 + "\n")

        os.makedirs(outdir, exist_ok=True)
        torch.save(
            {"mu": st_mu.cpu(), "scale": st_scale.cpu()},
            os.path.join(outdir, "st_slide_canon_stats.pt"),
        )
        print(f"[Rank-0] Per-slide canonicalization: {len(torch.unique(slide_ids))} slides")

        # ========== INFERENCE DOMAIN: ST4 (treated as SC) ==========
        X_st4 = _extract(st4, common)
        inf_expr = torch.tensor(X_st4, dtype=torch.float32, device=fabric.device)

        n_inf = X_st4.shape[0]
        inf_slide_ids = torch.zeros(n_inf, dtype=torch.long, device=fabric.device)
        inf_patient_ids = torch.zeros(n_inf, dtype=torch.long, device=fabric.device)  # same patient

        st_source_ids = torch.tensor(
            np.concatenate([
                np.zeros(X_st1.shape[0], dtype=int),
                np.ones(X_st2.shape[0], dtype=int),
                np.full(X_st3.shape[0], 2, dtype=int)
            ]),
            dtype=torch.long, device=fabric.device
        )

        inf_source_ids = torch.full((n_inf,), 3, dtype=torch.long, device=fabric.device)

        print(f"\n[Stage A] Liver cross-slide setup:")
        print(f"  ST total: {st_expr.shape[0]} spots (3 slides)")
        print(f"  ST4 (inference): {inf_expr.shape[0]} spots")
        print(f"  adv_slide_weight: {ADV_SLIDE_WEIGHT}")
        print(f"  mmd_weight: {MMD_WEIGHT}")
        print(f"  coral_raw_weight: {CORAL_RAW_WEIGHT}")
        print(f"  source_coral_weight: {SOURCE_CORAL_WEIGHT}")
        print(f"  local_align_weight: {LOCAL_ALIGN_WEIGHT}")

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
            print("Using train_encoder for liver cross-slide alignment")

            encoder_stageA = SharedEncoder(
                n_genes=n_genes,
                n_embedding=[512, 256, 128],
                dropout=0.1
            ).to(fabric.device)

            encoder_stageA, projector, discriminator, hist = train_encoder(
                inference_dropout_prob=0.5,
                model=encoder_stageA,
                st_gene_expr=st_expr,
                st_coords=st_coords,
                sc_gene_expr=inf_expr,
                slide_ids=slide_ids,
                sc_slide_ids=inf_slide_ids,
                sc_patient_ids=inf_patient_ids,
                n_epochs=stageA_epochs,
                batch_size=256,
                lr=1e-4,
                device=str(fabric.device),
                outf=outdir,
                st_source_ids=st_source_ids,
                sc_source_ids=inf_source_ids,
                use_source_adversary=False,
                source_coral_weight=SOURCE_CORAL_WEIGHT,
                stageA_obj='vicreg_adv',
                vicreg_lambda_inv=args.vicreg_lambda_inv,
                vicreg_lambda_var=args.vicreg_lambda_var,
                vicreg_lambda_cov=args.vicreg_lambda_cov,
                vicreg_gamma=args.vicreg_gamma,
                vicreg_eps=args.vicreg_eps,
                vicreg_project_dim=args.vicreg_project_dim,
                vicreg_use_projector=args.vicreg_use_projector,
                vicreg_float32_stats=args.vicreg_float32_stats,
                vicreg_ddp_gather=False,
                aug_gene_dropout=args.aug_gene_dropout,
                aug_gauss_std=args.aug_gauss_std,
                aug_scale_jitter=args.aug_scale_jitter,
                adv_slide_weight=ADV_SLIDE_WEIGHT,
                patient_coral_weight=PATIENT_CORAL_WEIGHT,
                mmd_weight=MMD_WEIGHT,
                mmd_use_l2norm=True,
                mmd_ramp=True,
                adv_warmup_epochs=ADV_WARMUP_EPOCHS,
                adv_ramp_epochs=ADV_RAMP_EPOCHS,
                grl_alpha_max=args.grl_alpha_max,
                disc_hidden=args.disc_hidden,
                disc_dropout=args.disc_dropout,
                stageA_balanced_slides=args.stageA_balanced_slides,
                adv_representation_mode='clean',
                adv_use_layernorm=False,
                adv_log_diagnostics=True,
                adv_log_grad_norms=False,
                use_local_align=True,
                return_aux=True,
                local_align_bidirectional=True,
                local_align_weight=LOCAL_ALIGN_WEIGHT,
                local_align_tau_z=0.07,
                seed=args.seed,
                use_best_checkpoint=True,
                coral_raw_weight=CORAL_RAW_WEIGHT,
                knn_weight=0.0
            )

            model.encoder = encoder_stageA

            set_seed(args.seed)

            model.encoder.eval()
            for param in model.encoder.parameters():
                param.requires_grad = False

            print("Stage A complete. Encoder frozen.")

            # ===== Stage A guard: 4-class probe on Z_ln =====
            if ENABLE_STAGEA_GUARD:
                print("\n[Stage A GUARD] Checking 4-class probe (Z_ln)...")
                encoder_stageA.eval()
                with torch.no_grad():
                    n_s1 = min(STAGEA_GUARD_MAX_SAMPLES, X_st1.shape[0])
                    n_s2 = min(STAGEA_GUARD_MAX_SAMPLES, X_st2.shape[0])
                    n_s3 = min(STAGEA_GUARD_MAX_SAMPLES, X_st3.shape[0])
                    n_s4 = min(STAGEA_GUARD_MAX_SAMPLES, X_st4.shape[0])

                    idx1 = torch.randperm(X_st1.shape[0], device=fabric.device)[:n_s1]
                    idx2 = torch.randperm(X_st2.shape[0], device=fabric.device)[:n_s2]
                    idx3 = torch.randperm(X_st3.shape[0], device=fabric.device)[:n_s3]
                    idx4 = torch.randperm(X_st4.shape[0], device=fabric.device)[:n_s4]

                    Z_s1 = encoder_stageA(torch.tensor(X_st1, dtype=torch.float32, device=fabric.device)[idx1])
                    Z_s2 = encoder_stageA(torch.tensor(X_st2, dtype=torch.float32, device=fabric.device)[idx2])
                    Z_s3 = encoder_stageA(torch.tensor(X_st3, dtype=torch.float32, device=fabric.device)[idx3])
                    Z_s4 = encoder_stageA(torch.tensor(X_st4, dtype=torch.float32, device=fabric.device)[idx4])

                    Z_all = torch.cat([Z_s1, Z_s2, Z_s3, Z_s4], dim=0)
                    Z_all = F.layer_norm(Z_all, (Z_all.shape[1],))
                    Z_all = F.normalize(Z_all, dim=1)

                    labels = torch.cat([
                        torch.full((Z_s1.shape[0],), 0, dtype=torch.long, device=fabric.device),
                        torch.full((Z_s2.shape[0],), 1, dtype=torch.long, device=fabric.device),
                        torch.full((Z_s3.shape[0],), 2, dtype=torch.long, device=fabric.device),
                        torch.full((Z_s4.shape[0],), 3, dtype=torch.long, device=fabric.device),
                    ])

                Z_np = Z_all.cpu().numpy()
                y_np = labels.cpu().numpy()
                probe = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
                probe.fit(Z_np, y_np)
                pred = probe.predict(Z_np)
                bal_acc = balanced_accuracy_score(y_np, pred)
                chance = 1.0 / 4.0

                print(f"[Stage A GUARD] balanced acc = {bal_acc:.4f} (chance={chance:.4f})")
                if bal_acc > 0.50:
                    print("[Stage A GUARD] ⚠️ Poor mixing detected. Stopping before Stage B/C.")
                    raise SystemExit(1)
                else:
                    print("[Stage A GUARD] ✓ Mixing OK. Proceeding to Stage B/C.")

        # ========== STAGE B ==========
        print("\n=== Stage B (single GPU, rank-0) ===")
        st1_coords_canon = st_coords[slide_ids == 0]
        st1_expr_tensor = st_expr[slide_ids == 0]
        st2_coords_canon = st_coords[slide_ids == 1]
        st2_expr_tensor = st_expr[slide_ids == 1]
        st3_coords_canon = st_coords[slide_ids == 2]
        st3_expr_tensor = st_expr[slide_ids == 2]

        slides_dict = {
            0: (st1_coords_canon, st1_expr_tensor),
            1: (st2_coords_canon, st2_expr_tensor),
            2: (st3_coords_canon, st3_expr_tensor)
        }

        model.train_stageB(
            slides=slides_dict,
            outdir=stageB_outdir,
        )

        ckpt_ab = {
            "encoder": model.encoder.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "score_net": model.score_net.state_dict(),
        }
        torch.save(ckpt_ab, os.path.join(outdir, "ab_init.pt"))

    # Sync all ranks
    fabric.barrier()

    if not fabric.is_global_zero:
        path = os.path.join(outdir, "ab_init.pt")
        ck = torch.load(path, map_location=fabric.device, weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.context_encoder.load_state_dict(ck["context_encoder"])
        model.generator.load_state_dict(ck["generator"])
        model.score_net.load_state_dict(ck["score_net"])

        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")

        def _extract(adata, genes):
            X = adata[:, genes].X
            return X.toarray() if hasattr(X, 'toarray') else X

        X_st1 = _extract(st1, common)
        X_st2 = _extract(st2, common)
        X_st3 = _extract(st3, common)

        st_coords_raw_rank = torch.tensor(
            np.vstack([st1.obsm['spatial'], st2.obsm['spatial'], st3.obsm['spatial']]),
            dtype=torch.float32
        )

        slide_ids_rank = torch.tensor(
            np.concatenate([
                np.zeros(X_st1.shape[0], dtype=int),
                np.ones(X_st2.shape[0], dtype=int),
                np.full(X_st3.shape[0], 2, dtype=int)
            ]),
            dtype=torch.long
        )

        st_coords_rank, _, _ = uet.canonicalize_st_coords_per_slide(st_coords_raw_rank, slide_ids_rank)
        print(f"[Rank {fabric.global_rank}] Applied per-slide canonicalization")

        st1_coords_canon = st_coords_rank[slide_ids_rank == 0]
        st1_expr = torch.tensor(X_st1, dtype=torch.float32)
        st2_coords_canon = st_coords_rank[slide_ids_rank == 1]
        st2_expr = torch.tensor(X_st2, dtype=torch.float32)
        st3_coords_canon = st_coords_rank[slide_ids_rank == 2]
        st3_expr = torch.tensor(X_st3, dtype=torch.float32)

        slides_dict_rank = {
            0: (st1_coords_canon, st1_expr),
            1: (st2_coords_canon, st2_expr),
            2: (st3_coords_canon, st3_expr)
        }

        model.train_stageB(
            slides=slides_dict_rank,
            outdir=stageB_outdir,
        )

    # ---------- Stage C (multi-GPU with Fabric) ----------
    print("\n=== Stage C (DDP across GPUs) ===")

    def _extract(adata, genes):
        X = adata[:, genes].X
        return X.toarray() if hasattr(X, 'toarray') else X

    X_st1 = _extract(st1, common)
    X_st2 = _extract(st2, common)
    X_st3 = _extract(st3, common)
    X_st4 = _extract(st4, common)

    st_coords_raw = torch.tensor(
        np.vstack([st1.obsm['spatial'], st2.obsm['spatial'], st3.obsm['spatial']]),
        dtype=torch.float32, device=fabric.device
    )

    slide_ids = torch.tensor(
        np.concatenate([
            np.zeros(X_st1.shape[0], dtype=int),
            np.ones(X_st2.shape[0], dtype=int),
            np.full(X_st3.shape[0], 2, dtype=int)
        ]),
        dtype=torch.long, device=fabric.device
    )

    st_coords, _, _ = uet.canonicalize_st_coords_per_slide(st_coords_raw, slide_ids)
    print(f"[Rank {fabric.global_rank}] Stage C: Applied per-slide canonicalization")

    st1_expr_tensor = torch.tensor(X_st1, dtype=torch.float32, device=fabric.device)
    st2_expr_tensor = torch.tensor(X_st2, dtype=torch.float32, device=fabric.device)
    st3_expr_tensor = torch.tensor(X_st3, dtype=torch.float32, device=fabric.device)
    st_gene_expr_dict = {
        0: st1_expr_tensor,
        1: st2_expr_tensor,
        2: st3_expr_tensor
    }

    # ST4 as inference expression for Stage C
    inf_expr = torch.tensor(X_st4, dtype=torch.float32, device=fabric.device)

    # ========== PHASE 1: ST-ONLY TRAINING ==========
    print("\n" + "=" * 70)
    print("PHASE 1: Training with ST data ONLY (fix geometry)")
    print("=" * 70)

    history_st = model.train_stageC(
        st_gene_expr_dict=st_gene_expr_dict,
        sc_gene_expr=inf_expr,
        n_min=96, n_max=384,
        num_st_samples=args.num_st_samples,
        num_sc_samples=0,
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

    # ========== SAVE CHECKPOINT ==========
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

    # ========== PLOT PHASE 1 LOSSES ==========
    if fabric.is_global_zero:
        print("\n=== Plotting Phase 1 training losses ===")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if history_st is not None and len(history_st['epoch']) > 0:
            epochs = history_st['epoch']
            losses = history_st['epoch_avg']

            active_losses = []
            for name, vals in losses.items():
                if len(vals) > 0 and any(abs(v) > 1e-9 for v in vals):
                    active_losses.append(name)

            if 'total' in active_losses:
                active_losses.remove('total')
                active_losses = ['total'] + sorted(active_losses)
            else:
                active_losses = sorted(active_losses)

            print(f"Active losses to plot: {active_losses}")

            n_losses = len(active_losses)
            n_cols = 3
            n_rows = (n_losses + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            fig.suptitle('Phase 1: ST-Only Training Losses (Non-Zero Only)',
                         fontsize=18, fontweight='bold', y=0.995)

            axes = axes.flatten() if n_losses > 1 else [axes]

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

            for idx in range(n_losses, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plot_filename = f"stageC_phase1_losses_{timestamp}.png"
            plot_path = os.path.join(outdir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Phase 1 loss plot: {plot_path}")
            plt.close()

        # ========== SUMMARY ==========
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE (Phase 1 only)")
        print("=" * 70)
        print(f"\nOutput directory: {outdir}")
        print(f"Checkpoints saved:")
        print(f"  - {os.path.join(outdir, 'ab_init.pt')}")
        print(f"  - {os.path.join(outdir, 'phase1_st_checkpoint.pt')}")
        print(f"\nStage A config:")
        print(f"  - adv_slide_weight: {ADV_SLIDE_WEIGHT}")
        print(f"  - mmd_weight: {MMD_WEIGHT}")
        print(f"  - coral_raw_weight: {CORAL_RAW_WEIGHT}")
        print(f"  - source_coral_weight: {SOURCE_CORAL_WEIGHT}")
        print(f"  - local_align_weight: {LOCAL_ALIGN_WEIGHT}")
        print("=" * 70)


if __name__ == "__main__":
    main()