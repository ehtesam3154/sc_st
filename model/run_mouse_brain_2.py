# run_mouse_brain_2.py - Updated to match run_hscc_gems.py pipeline
import os
import torch
from lightning.fabric import Fabric
import scanpy as sc
from core_models_et_p3 import GEMSModel
from scipy.ndimage import gaussian_filter1d
import torch.distributed as dist
import pandas as pd
import anndata as ad
import numpy as np
import utils_et as uet
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil
from pathlib import Path


def ensure_disk_space(path, min_gb=1.0):
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
    parser = argparse.ArgumentParser(description='GEMS Training with Lightning Fabric - Mouse Brain')

    # Training config
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--stageA_epochs', type=int, default=1000)
    parser.add_argument('--stageC_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='gems_mousebrain_output')
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
    parser.add_argument('--curriculum_target_stage', type=int, default=6,
                        help='Target curriculum stage for Gate A (0-indexed, default 6 = full curriculum)')
    parser.add_argument('--curriculum_min_epochs', type=int, default=100,
                        help='Minimum epochs before three-gate stopping is allowed')
    parser.add_argument('--curriculum_early_stop', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable three-gate curriculum-aware early stopping (default: enabled)')

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

    # ========== RESIDUAL DIFFUSION ==========
    parser.add_argument('--use_residual_diffusion', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable residual diffusion: diffuse R = V_target - V_base instead of V_target')

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


# ============================================================================
# DATA LOADING - MOUSE BRAIN SPECIFIC (UNCHANGED)
# ============================================================================
def load_mouse_data():
    # ST1 as training ST data (with coordinates)
    st_counts = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st1_counts_et.csv'
    st_meta   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st1_metadata_et.csv'
    st_ct     = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st1_celltype_et.csv'

    # ST2 as test SC data (coordinates hidden, used for evaluation)
    sc_counts = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st2_counts_et.csv'
    sc_meta   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st2_metadata_et.csv'
    sc_ct     = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st2_celltype_et.csv'

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

    # ========== NEW: Auto-switch to anchored output directory if anchor training enabled ==========
    if args.anchor_train and 'anchored' not in outdir:
        outdir = outdir.replace('gems_mousebrain_output', 'gems_mousebrain_output_anchored')
        print(f"[ANCHOR-TRAIN] Auto-switching output dir to: {outdir}")

    fabric = Fabric(accelerator="gpu", devices=devices, strategy="ddp", precision=precision)
    fabric.launch()

    # ---------- Load data on all ranks (OK) ----------
    scadata, stadata = load_mouse_data()

    # ---------- Get common genes ----------
    common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
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
        X_sc = scadata[:, common].X
        X_st = stadata[:, common].X
        if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
        if hasattr(X_st, "toarray"): X_st = X_st.toarray()
        sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
        st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)

        # Per-slide canonicalization
        st_coords_raw = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
        slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)

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
        ck = torch.load(path, map_location=fabric.device, weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.context_encoder.load_state_dict(ck["context_encoder"])
        model.generator.load_state_dict(ck["generator"])
        model.score_net.load_state_dict(ck["score_net"])

        # CRITICAL: Non-rank-0 processes need to recompute Stage B
        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")

        common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
        X_st = stadata[:, common].X
        if hasattr(X_st, "toarray"): X_st = X_st.toarray()
        st_expr_rank = torch.tensor(X_st, dtype=torch.float32)

        # Same per-slide canonicalization as rank-0
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

    common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    X_sc = scadata[:, common].X
    X_st = stadata[:, common].X
    if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
    if hasattr(X_st, "toarray"): X_st = X_st.toarray()
    sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
    st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)

    # Use same per-slide canonicalization for Stage C
    st_coords_raw = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
    slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)

    st_coords, _, _ = uet.canonicalize_st_coords_per_slide(
        st_coords_raw, slide_ids
    )
    print(f"[Rank {fabric.global_rank}] Stage C: Applied per-slide canonicalization")

    st_gene_expr_dict = {0: st_expr}

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
        # ========== RESIDUAL DIFFUSION ==========
        use_residual_diffusion=args.use_residual_diffusion,
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
        if dist.is_initialized():
            dist.destroy_process_group()
        import sys
        sys.exit(0)  # Fully exit non-rank-0 processes

    # ========== CRITICAL: Destroy DDP on rank-0 before single-GPU inference ==========
    print("[Rank-0] Destroying DDP process group...")
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


    # ---------- Inference (rank-0 only, single GPU) ----------
    if fabric.is_global_zero:
        print("[DEBUG Rank-0] Starting inference...")

        # Create timestamp for all outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ensure_disk_space(outdir, min_gb=10.0)

        # ===================================================================
        # LOAD AND NORMALIZE GT COORDS FOR DEBUG KNN TRACKING
        # ===================================================================
        gt_coords_raw = scadata.obsm['spatial_gt']
        gt_coords_tensor = torch.tensor(gt_coords_raw, dtype=torch.float32, device=fabric.device)
        slide_ids_gt = torch.zeros(gt_coords_tensor.shape[0], dtype=torch.long, device=fabric.device)

        gt_coords_norm, gt_mu, gt_scale = uet.canonicalize_st_coords_per_slide(
            gt_coords_tensor, slide_ids_gt
        )

        print(f"✓ GT coords normalized: scale={gt_scale[0].item():.4f}")
        print(f"✓ GT coords RMS: {gt_coords_norm.pow(2).mean().sqrt().item():.4f}")


        print("\n=== Inference (rank-0) with Multi-Sample Ranking ===")

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

                    n_cells = sc_expr.shape[0]

                    sample_results = model.infer_sc_patchwise(
                        sc_gene_expr=sc_expr,
                        n_timesteps_sample=500,  # Use 600 like mouse brain (you had 400)
                        return_coords=True,
                        patch_size=192,  # Use 192 like mouse brain (you had 256)
                        coverage_per_cell=6.0,
                        n_align_iters=15,  # Use 15 like mouse brain (you had 10)
                        eta=0.0,
                        guidance_scale=2.0,
                        # GT coords for evaluation
                        gt_coords=gt_coords_norm,
                        debug_knn=True,
                        debug_max_patches=15,
                        debug_k_list=(10, 20),
                        # ST-style stochastic sampling
                        pool_mult=2.0,
                        stochastic_tau=0.8,
                        tau_mode="adaptive_kth",
                        ensure_connected=True,
                        local_refine=False,
                        # Anchored sequential sampling
                        inference_mode=args.inference_mode,  # "anchored" or "unanchored"
                        anchor_sampling_mode="align_vote_only" if args.inference_mode == "anchored" else "off",
                        commit_frac=0.6,
                        seq_align_dim=32,
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

        # Plot 1: GEMS coordinates colored by cell type (if available)
        if 'celltype_mapped_refined' in scadata.obs.columns:
            cell_types = scadata.obs['celltype_mapped_refined']
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

        if 'celltype_mapped_refined' in scadata.obs.columns:
            # Create scanpy-style embedding plot
            fig, ax = plt.subplots(figsize=(10, 8))

            sc.settings.set_figure_params(dpi=150, frameon=False)

            # Use scanpy's embedding plot
            sc.pl.embedding(
                scadata,
                basis='gems',  # This uses obsm['X_gems']
                color='celltype_mapped_refined',
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
                color='celltype_mapped_refined',
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

        if 'celltype_mapped_refined' in scadata.obs.columns:
            print(f"\nCell type distribution:")
            for ct, count in scadata.obs['celltype_mapped_refined'].value_counts().items():
                print(f"  {ct}: {count} cells ({count/len(scadata)*100:.1f}%)")

        print("\n" + "="*70)
        print("All outputs saved to:", outdir)
        print("="*70)

        # Create a summary file
        summary_filename = f"inference_summary_{timestamp}.txt"
        summary_path = os.path.join(outdir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write("GEMS Inference Summary - Mouse Brain\n")
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
            if 'celltype_mapped_refined' in scadata.obs.columns:
                f.write(f"\nCell type distribution:\n")
                for ct, count in scadata.obs['celltype_mapped_refined'].value_counts().items():
                    f.write(f"  {ct}: {count} cells ({count/len(scadata)*100:.1f}%)\n")

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
