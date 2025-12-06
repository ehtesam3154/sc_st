# run_mouse_brain_v2.py
# Stage C 2.0: Graph-aware losses, ST-only training, SC encoder fine-tuning
"""
Training Pipeline:
    Stage A: Shared encoder (ST ↔ SC alignment) - Rank-0 only
    Stage B: Precompute geometric targets - Rank-0 only  
    Stage C: ST-only diffusion training (graph-aware losses) - Multi-GPU
    Stage D: SC encoder fine-tuning (frozen geometry prior) - Multi-GPU
    Inference: Patchwise + Single-patch SC inference with evaluation
"""

import os
import sys
import torch
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from lightning.fabric import Fabric

from core_models_et_p3 import GEMSModel
from core_models_et_p1 import STSetDataset, SCSetDataset
import utils_et as uet


def ensure_disk_space(path, min_gb=10.0):
    """Check free space, raise error if below threshold."""
    path = Path(path)
    if not path.exists():
        path = Path.home()
    stat = shutil.disk_usage(str(path))
    avail_gb = stat.free / (1024**3)
    print(f"[disk] Available space on {path.anchor or path}: {avail_gb:.1f} GB")
    if avail_gb < min_gb:
        raise RuntimeError(f"Only {avail_gb:.1f} GB free. Please free up space.")


def parse_args():
    parser = argparse.ArgumentParser(description='GEMS v2.0 Training (Graph-Aware Losses)')
    
    # Hardware
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    
    # Stage A/B
    parser.add_argument('--stageA_epochs', type=int, default=1000)
    
    # Stage C (ST-only)
    parser.add_argument('--stageC_epochs', type=int, default=150)
    parser.add_argument('--num_st_samples', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Loss weights for Stage C
    parser.add_argument('--w_score', type=float, default=1.0)
    parser.add_argument('--w_edge', type=float, default=1.0)
    parser.add_argument('--w_repel', type=float, default=0.1)
    
    # SC fine-tuning
    parser.add_argument('--sc_finetune_epochs', type=int, default=50)
    parser.add_argument('--num_sc_samples', type=int, default=5000)
    parser.add_argument('--sc_lr', type=float, default=3e-5)
    parser.add_argument('--skip_sc_finetune', action='store_true', default=False)
    
    # Model architecture
    parser.add_argument('--use_canonicalize', action='store_true', default=True)
    parser.add_argument('--use_dist_bias', action='store_true', default=True)
    parser.add_argument('--dist_bins', type=int, default=24)
    parser.add_argument('--dist_head_shared', action='store_true', default=True)
    parser.add_argument('--use_angle_features', action='store_true', default=True)
    parser.add_argument('--angle_bins', type=int, default=8)
    parser.add_argument('--knn_k', type=int, default=12)
    parser.add_argument('--self_conditioning', action='store_true', default=True)
    parser.add_argument('--sc_feat_mode', type=str, default='concat')
    parser.add_argument('--landmarks_L', type=int, default=0)
    
    # Early stopping
    parser.add_argument('--enable_early_stop', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_min_epochs', type=int, default=20)
    
    # Inference
    parser.add_argument('--skip_inference', action='store_true', default=False)
    parser.add_argument('--n_timesteps_sample', type=int, default=500)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    
    # Output
    parser.add_argument('--outdir', type=str, default='gems_v2_output')
    
    return parser.parse_args()


def load_mouse_data():
    """Load mouse brain ST and SC data."""
    base_path = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2'
    
    import pandas as pd
    import anndata as ad
    
    # ST data (ST1)
    st_counts = f'{base_path}/simu_st1_counts_et.csv'
    st_meta = f'{base_path}/simu_st1_metadata_et.csv'
    st_ct = f'{base_path}/simu_st1_celltype_et.csv'
    
    st_expr_df = pd.read_csv(st_counts, index_col=0)
    st_meta_df = pd.read_csv(st_meta, index_col=0)
    st_ct_df = pd.read_csv(st_ct, index_col=0)
    
    stadata = ad.AnnData(X=st_expr_df.values.T)
    stadata.obs_names = st_expr_df.columns
    stadata.var_names = st_expr_df.index
    stadata.obsm['spatial'] = st_meta_df[['coord_x', 'coord_y']].values
    stadata.obs['celltype_mapped_refined'] = st_ct_df.idxmax(axis=1).values
    stadata.obsm['celltype_proportions'] = st_ct_df.values
    
    print(f"ST loaded: {stadata.shape[0]} spots, {stadata.shape[1]} genes")
    
    # SC data (ST2 as pseudo-SC with ground truth)
    sc_counts = f'{base_path}/simu_st2_counts_et.csv'
    sc_meta = f'{base_path}/simu_st2_metadata_et.csv'
    sc_ct = f'{base_path}/simu_st2_celltype_et.csv'
    
    sc_expr_df = pd.read_csv(sc_counts, index_col=0)
    sc_meta_df = pd.read_csv(sc_meta, index_col=0)
    sc_ct_df = pd.read_csv(sc_ct, index_col=0)
    
    scadata = ad.AnnData(X=sc_expr_df.values.T)
    scadata.obs_names = sc_expr_df.columns
    scadata.var_names = sc_expr_df.index
    scadata.obsm['spatial_gt'] = sc_meta_df[['coord_x', 'coord_y']].values
    scadata.obs['celltype_mapped_refined'] = sc_ct_df.idxmax(axis=1).values
    scadata.obsm['celltype_proportions'] = sc_ct_df.values
    
    print(f"SC loaded: {scadata.shape[0]} cells, {scadata.shape[1]} genes")
    
    return scadata, stadata


# ==============================================================================
# AUTOMATIC LOSS PLOTTING (adapts to any losses in history)
# ==============================================================================

def plot_training_history(history, phase_name, outdir, timestamp):
    """
    Automatically plot all losses found in history['epoch_avg'].
    Works for any number of loss components.
    """
    if history is None or 'epoch_avg' not in history:
        print(f"[Plot] No history for {phase_name}, skipping")
        return
    
    epochs = history['epoch']
    losses = history['epoch_avg']
    
    # Get all loss names (excluding 'total' for separate treatment)
    loss_names = [k for k in losses.keys() if len(losses[k]) > 0]
    if not loss_names:
        print(f"[Plot] No losses in {phase_name} history")
        return
    
    n_losses = len(loss_names)
    
    # Determine grid size
    n_cols = min(4, n_losses)
    n_rows = (n_losses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'{phase_name} Training Losses', fontsize=14, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_losses == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, name in enumerate(loss_names):
        ax = axes[idx]
        values = losses[name]
        color = colors[idx % len(colors)]
        
        ax.plot(epochs[:len(values)], values, color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'{name}', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        if values:
            ax.annotate(f'{values[-1]:.4f}', 
                       xy=(epochs[len(values)-1], values[-1]),
                       fontsize=9, alpha=0.7)
    
    # Hide unused subplots
    for idx in range(n_losses, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(outdir, f'{phase_name.lower().replace(" ", "_")}_losses_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_combined_losses(history_st, history_sc, outdir, timestamp):
    """Plot ST and SC losses side by side for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot ST total loss
    if history_st and 'total' in history_st['epoch_avg']:
        axes[0].plot(history_st['epoch'], history_st['epoch_avg']['total'], 
                    'b-', linewidth=2, label='ST Total')
        axes[0].set_title('Stage C: ST-Only Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot SC total loss
    if history_sc and 'total' in history_sc['epoch_avg']:
        axes[1].plot(history_sc['epoch'], history_sc['epoch_avg']['total'],
                    'r-', linewidth=2, label='SC Total')
        axes[1].set_title('Stage D: SC Encoder Fine-tuning')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Total Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(outdir, f'combined_losses_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ==============================================================================
# INFERENCE AND EVALUATION
# ==============================================================================

def run_inference_and_evaluate(model, sc_expr, scadata, outdir, timestamp, args):
    """Run inference and evaluate against ground truth."""
    from scipy.spatial.distance import cdist
    from scipy.stats import pearsonr, spearmanr
    
    print("\n" + "="*70)
    print("INFERENCE AND EVALUATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sc_expr = sc_expr.to(device)
    n_cells = sc_expr.shape[0]
    
    # Get ground truth
    gt_coords = scadata.obsm['spatial_gt']
    gt_edm = cdist(gt_coords, gt_coords)
    
    results_dict = {}
    
    # ===== SINGLE-PATCH INFERENCE =====
    print("\n--- Single-Patch Inference ---")
    torch.manual_seed(42)
    
    results_single = model.infer_sc_single_patch(
        sc_gene_expr=sc_expr,
        n_timesteps_sample=args.n_timesteps_sample,
        sigma_min=0.01,
        sigma_max=3.0,
        guidance_scale=args.guidance_scale,
        eta=0.0,
        target_st_p95=None,
        return_coords=True,
        DEBUG_FLAG=True,
    )
    
    coords_single = results_single['coords_canon'].cpu().numpy()
    pred_edm_single = cdist(coords_single, coords_single)
    
    # Flatten for correlation
    gt_flat = gt_edm[np.triu_indices(n_cells, k=1)]
    pred_flat_single = pred_edm_single[np.triu_indices(n_cells, k=1)]
    
    pearson_single = pearsonr(gt_flat, pred_flat_single)[0]
    spearman_single = spearmanr(gt_flat, pred_flat_single)[0]
    
    print(f"Single-Patch Results:")
    print(f"  Pearson:  {pearson_single:.4f}")
    print(f"  Spearman: {spearman_single:.4f}")
    
    results_dict['single_patch'] = {
        'coords': coords_single,
        'edm': pred_edm_single,
        'pearson': pearson_single,
        'spearman': spearman_single,
    }
    
    # ===== PATCHWISE INFERENCE =====
    print("\n--- Patchwise Inference ---")
    torch.manual_seed(42)
    
    results_patchwise = model.infer_sc_patchwise(
        sc_gene_expr=sc_expr,
        n_timesteps_sample=args.n_timesteps_sample,
        sigma_min=0.01,
        sigma_max=3.0,
        patch_size=384,
        coverage_per_cell=4.0,
        n_align_iters=10,
        eta=0.0,
        guidance_scale=args.guidance_scale,
        return_coords=True,
        debug_flag=True,
        debug_every=10,
    )
    
    coords_patchwise = results_patchwise['coords_canon'].cpu().numpy()
    pred_edm_patchwise = cdist(coords_patchwise, coords_patchwise)
    
    pred_flat_patchwise = pred_edm_patchwise[np.triu_indices(n_cells, k=1)]
    
    pearson_patchwise = pearsonr(gt_flat, pred_flat_patchwise)[0]
    spearman_patchwise = spearmanr(gt_flat, pred_flat_patchwise)[0]
    
    print(f"Patchwise Results:")
    print(f"  Pearson:  {pearson_patchwise:.4f}")
    print(f"  Spearman: {spearman_patchwise:.4f}")
    
    results_dict['patchwise'] = {
        'coords': coords_patchwise,
        'edm': pred_edm_patchwise,
        'pearson': pearson_patchwise,
        'spearman': spearman_patchwise,
    }
    
    # ===== kNN PRESERVATION =====
    print("\n--- kNN Preservation Metrics ---")
    for k in [10, 20, 50]:
        knn_single = compute_knn_preservation(gt_coords, coords_single, k)
        knn_patchwise = compute_knn_preservation(gt_coords, coords_patchwise, k)
        print(f"  k={k:2d}: Single={knn_single:.3f}, Patchwise={knn_patchwise:.3f}")
        results_dict[f'knn_{k}_single'] = knn_single
        results_dict[f'knn_{k}_patchwise'] = knn_patchwise
    
    # ===== GEOMETRY METRICS =====
    print("\n--- Geometry Metrics ---")
    dim_single, aniso_single = compute_geometry_metrics(coords_single)
    dim_patchwise, aniso_patchwise = compute_geometry_metrics(coords_patchwise)
    dim_gt, aniso_gt = compute_geometry_metrics(gt_coords)
    
    print(f"  GT:        eff_dim={dim_gt:.2f}, anisotropy={aniso_gt:.2f}")
    print(f"  Single:    eff_dim={dim_single:.2f}, anisotropy={aniso_single:.2f}")
    print(f"  Patchwise: eff_dim={dim_patchwise:.2f}, anisotropy={aniso_patchwise:.2f}")
    
    results_dict['geometry'] = {
        'gt': {'dim': dim_gt, 'aniso': aniso_gt},
        'single': {'dim': dim_single, 'aniso': aniso_single},
        'patchwise': {'dim': dim_patchwise, 'aniso': aniso_patchwise},
    }
    
    # ===== PLOT RESULTS =====
    plot_inference_results(gt_coords, coords_single, coords_patchwise, 
                          scadata, outdir, timestamp)
    
    # ===== SAVE RESULTS =====
    save_path = os.path.join(outdir, f'inference_results_{timestamp}.pt')
    torch.save(results_dict, save_path)
    print(f"\n✓ Saved inference results: {save_path}")
    
    # Save coordinates to AnnData
    scadata.obsm['X_gems_single'] = coords_single
    scadata.obsm['X_gems_patchwise'] = coords_patchwise
    adata_path = os.path.join(outdir, f'scadata_with_gems_{timestamp}.h5ad')
    scadata.write_h5ad(adata_path)
    print(f"✓ Saved AnnData: {adata_path}")
    
    return results_dict


def compute_knn_preservation(gt_coords, pred_coords, k):
    """Compute fraction of k-nearest neighbors preserved."""
    from sklearn.neighbors import NearestNeighbors
    
    nn_gt = NearestNeighbors(n_neighbors=k+1).fit(gt_coords)
    nn_pred = NearestNeighbors(n_neighbors=k+1).fit(pred_coords)
    
    _, idx_gt = nn_gt.kneighbors(gt_coords)
    _, idx_pred = nn_pred.kneighbors(pred_coords)
    
    # Exclude self (first neighbor)
    idx_gt = idx_gt[:, 1:]
    idx_pred = idx_pred[:, 1:]
    
    preservation = []
    for i in range(len(gt_coords)):
        overlap = len(set(idx_gt[i]) & set(idx_pred[i]))
        preservation.append(overlap / k)
    
    return np.mean(preservation)


def compute_geometry_metrics(coords):
    """Compute effective dimensionality and anisotropy."""
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    
    dim_eff = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    aniso = eigvals[0] / (eigvals[1] + 1e-8) if len(eigvals) >= 2 else 1.0
    
    return dim_eff, aniso


def plot_inference_results(gt_coords, coords_single, coords_patchwise, scadata, outdir, timestamp):
    """Plot spatial reconstructions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    celltypes = scadata.obs['celltype_mapped_refined'].values
    unique_types = np.unique(celltypes)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    color_map = {ct: colors[i] for i, ct in enumerate(unique_types)}
    cell_colors = [color_map[ct] for ct in celltypes]
    
    # Ground truth
    axes[0].scatter(gt_coords[:, 0], gt_coords[:, 1], c=cell_colors, s=5, alpha=0.7)
    axes[0].set_title('Ground Truth')
    axes[0].set_aspect('equal')
    
    # Single-patch
    axes[1].scatter(coords_single[:, 0], coords_single[:, 1], c=cell_colors, s=5, alpha=0.7)
    axes[1].set_title('Single-Patch Inference')
    axes[1].set_aspect('equal')
    
    # Patchwise
    axes[2].scatter(coords_patchwise[:, 0], coords_patchwise[:, 1], c=cell_colors, s=5, alpha=0.7)
    axes[2].set_title('Patchwise Inference')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    save_path = os.path.join(outdir, f'spatial_reconstruction_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main(args=None):
    if args is None:
        args = parse_args()
    
    # Setup Fabric for multi-GPU
    fabric = Fabric(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        precision=args.precision
    )
    fabric.launch()
    
    outdir = args.outdir
    stageB_outdir = "gems_stageB_cache"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========== LOAD DATA ==========
    scadata, stadata = load_mouse_data()
    
    common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    n_genes = len(common)
    print(f"Common genes: {n_genes}")
    
    # ========== BUILD MODEL ==========
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
    
    # ========== STAGE A & B (RANK-0 ONLY) ==========
    if fabric.is_global_zero:
        os.makedirs(outdir, exist_ok=True)
        
        # Extract tensors
        X_sc = scadata[:, common].X
        X_st = stadata[:, common].X
        if hasattr(X_sc, "toarray"):
            X_sc = X_sc.toarray()
        if hasattr(X_st, "toarray"):
            X_st = X_st.toarray()
        
        sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
        st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)
        
        # Canonicalize ST coordinates
        st_coords_raw = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
        slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)
        st_coords, st_mu, st_scale = uet.canonicalize_st_coords_per_slide(st_coords_raw, slide_ids)
        
        print(f"[Rank-0] ST coords canonicalized: scale={st_scale[0].item():.4f}")
        
        # Save canonicalization stats
        torch.save(
            {"mu": st_mu.cpu(), "scale": st_scale.cpu()},
            os.path.join(outdir, "st_slide_canon_stats.pt")
        )
        
        # ===== STAGE A =====
        print("\n" + "="*70)
        print("STAGE A: Encoder Training (Rank-0)")
        print("="*70)
        
        model.train_stageA(
            st_gene_expr=st_expr,
            st_coords=st_coords,
            sc_gene_expr=sc_expr,
            slide_ids=slide_ids,
            n_epochs=args.stageA_epochs,
            batch_size=256,
            lr=1e-4,
            sigma=None,
            alpha=0.8,
            ratio_start=0.0,
            ratio_end=1.0,
            mmdbatch=1.0,
            outf=outdir,
        )
        
        # ===== STAGE B =====
        print("\n" + "="*70)
        print("STAGE B: Precompute Targets (Rank-0)")
        print("="*70)
        
        slides_dict = {0: (st_coords, st_expr)}
        model.train_stageB(slides=slides_dict, outdir=stageB_outdir)
        
        # ===== PRECOMPUTE ST DISTANCE REFERENCE FOR SC FINE-TUNING =====
        print("\n[Rank-0] Precomputing ST distance reference...")
        st_coords_list = [st_coords.cpu()]
        D_st_reference = uet.compute_st_distance_reference(st_coords_list, max_samples=50000)
        torch.save(D_st_reference, os.path.join(outdir, "D_st_reference.pt"))
        print(f"[Rank-0] ST distance reference: {D_st_reference.shape[0]} samples")
        
        # Save checkpoint
        ckpt_ab = {
            "encoder": model.encoder.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "score_net": model.score_net.state_dict(),
        }
        torch.save(ckpt_ab, os.path.join(outdir, "ab_init.pt"))
        print(f"[Rank-0] Saved Stage A/B checkpoint")
    
    # ========== SYNC ALL RANKS ==========
    fabric.barrier()
    
    if not fabric.is_global_zero:
        # Load A/B weights on other ranks
        path = os.path.join(outdir, "ab_init.pt")
        ck = torch.load(path, map_location=fabric.device)
        model.encoder.load_state_dict(ck["encoder"])
        model.context_encoder.load_state_dict(ck["context_encoder"])
        model.generator.load_state_dict(ck["generator"])
        model.score_net.load_state_dict(ck["score_net"])
        
        # Recompute Stage B on non-rank-0
        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")
        X_st = stadata[:, common].X
        if hasattr(X_st, "toarray"):
            X_st = X_st.toarray()
        st_expr_rank = torch.tensor(X_st, dtype=torch.float32)
        st_coords_raw_rank = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32)
        slide_ids_rank = torch.zeros(st_expr_rank.shape[0], dtype=torch.long)
        st_coords_rank, _, _ = uet.canonicalize_st_coords_per_slide(st_coords_raw_rank, slide_ids_rank)
        slides_dict_rank = {0: (st_coords_rank, st_expr_rank)}
        model.train_stageB(slides=slides_dict_rank, outdir=stageB_outdir)
    
    fabric.barrier()
    
    # ========== PREPARE DATA FOR STAGE C ==========
    X_sc = scadata[:, common].X
    X_st = stadata[:, common].X
    if hasattr(X_sc, "toarray"):
        X_sc = X_sc.toarray()
    if hasattr(X_st, "toarray"):
        X_st = X_st.toarray()
    
    sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
    st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)
    st_coords_raw = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
    slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)
    st_coords, _, _ = uet.canonicalize_st_coords_per_slide(st_coords_raw, slide_ids)
    
    st_gene_expr_dict = {0: st_expr.cpu()}
    
    # ========== STAGE C: ST-ONLY TRAINING (MULTI-GPU) ==========
    print("\n" + "="*70)
    print("STAGE C: ST-Only Training (Graph-Aware Losses)")
    print("="*70)
    
    # Use the new v2 training function
    history_st = model.train_stageC_v2(
        st_gene_expr_dict=st_gene_expr_dict,
        n_min=96, 
        n_max=384,
        num_st_samples=args.num_st_samples,
        n_epochs=args.stageC_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_timesteps=500,
        sigma_min=0.01,
        sigma_max=3.0,
        outf=outdir,
        fabric=fabric,
        precision=args.precision,
        w_score=args.w_score,
        w_edge=args.w_edge,
        w_repel=args.w_repel,
        enable_early_stop=args.enable_early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_epochs=args.early_stop_min_epochs,
    )
    
    fabric.barrier()
    
    # ===== SAVE ST CHECKPOINT =====
    if fabric.is_global_zero:
        checkpoint_path = os.path.join(outdir, f"phase1_st_checkpoint_{timestamp}.pt")
        checkpoint = {
            'encoder': model.encoder.state_dict(),
            'context_encoder': model.context_encoder.module.state_dict() if hasattr(model.context_encoder, 'module') else model.context_encoder.state_dict(),
            'generator': model.generator.module.state_dict() if hasattr(model.generator, 'module') else model.generator.state_dict(),
            'score_net': model.score_net.module.state_dict() if hasattr(model.score_net, 'module') else model.score_net.state_dict(),
            'history_st': history_st,
            'timestamp': timestamp,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\n✓ Saved Phase 1 (ST) checkpoint: {checkpoint_path}")
        
        # Plot ST losses
        plot_training_history(history_st, "StageC_ST", outdir, timestamp)
    
    fabric.barrier()
    
    # ========== STAGE D: SC ENCODER FINE-TUNING (MULTI-GPU) ==========
    history_sc = None
    if not args.skip_sc_finetune and args.num_sc_samples > 0:
        print("\n" + "="*70)
        print("STAGE D: SC Encoder Fine-tuning (Frozen Geometry Prior)")
        print("="*70)
        
        # Load ST distance reference
        D_st_reference = torch.load(os.path.join(outdir, "D_st_reference.pt"))
        
        # Use the new v2 SC fine-tuning function
        history_sc = model.finetune_encoder_on_sc_v2(
            sc_gene_expr=sc_expr,
            D_st_reference=D_st_reference,
            n_min=96,
            n_max=384,
            num_sc_samples=args.num_sc_samples,
            n_epochs=args.sc_finetune_epochs,
            batch_size=args.batch_size,
            lr=args.sc_lr,
            outf=outdir,
            fabric=fabric,
            precision=args.precision,
        )
        
        fabric.barrier()
        
        # ===== SAVE FINAL CHECKPOINT =====
        if fabric.is_global_zero:
            checkpoint_path = os.path.join(outdir, f"final_checkpoint_{timestamp}.pt")
            checkpoint = {
                'encoder': model.encoder.state_dict() if not hasattr(model.encoder, 'module') else model.encoder.module.state_dict(),
                'context_encoder': model.context_encoder.module.state_dict() if hasattr(model.context_encoder, 'module') else model.context_encoder.state_dict(),
                'generator': model.generator.module.state_dict() if hasattr(model.generator, 'module') else model.generator.state_dict(),
                'score_net': model.score_net.module.state_dict() if hasattr(model.score_net, 'module') else model.score_net.state_dict(),
                'history_st': history_st,
                'history_sc': history_sc,
                'timestamp': timestamp,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✓ Saved final checkpoint: {checkpoint_path}")
            
            # Plot SC losses
            plot_training_history(history_sc, "StageD_SC", outdir, timestamp)
            plot_combined_losses(history_st, history_sc, outdir, timestamp)
    else:
        print("\n[INFO] Skipping SC fine-tuning")
    
    fabric.barrier()
    
    # ========== INFERENCE (RANK-0 ONLY) ==========
    if fabric.is_global_zero and not args.skip_inference:
        print("\n[Rank-0] Unwrapping models for inference...")
        
        # Unwrap DDP modules
        if hasattr(model.encoder, 'module'):
            model.encoder = model.encoder.module
        if hasattr(model.context_encoder, 'module'):
            model.context_encoder = model.context_encoder.module
        if hasattr(model.generator, 'module'):
            model.generator = model.generator.module
        if hasattr(model.score_net, 'module'):
            model.score_net = model.score_net.module
        
        ensure_disk_space(outdir, min_gb=5.0)
        
        # Run inference
        inference_results = run_inference_and_evaluate(
            model, sc_expr, scadata, outdir, timestamp, args
        )
        
        # ========== FINAL SUMMARY ==========
        print("\n" + "="*70)
        print("TRAINING AND INFERENCE COMPLETE")
        print("="*70)
        print(f"Output directory: {outdir}")
        print(f"Timestamp: {timestamp}")
        print(f"\nCheckpoints:")
        print(f"  - ab_init.pt (Stage A/B)")
        print(f"  - phase1_st_checkpoint_{timestamp}.pt (Stage C)")
        if not args.skip_sc_finetune:
            print(f"  - final_checkpoint_{timestamp}.pt (Stage D)")
        print(f"\nInference Results:")
        print(f"  Single-Patch: Pearson={inference_results['single_patch']['pearson']:.4f}")
        print(f"  Patchwise:    Pearson={inference_results['patchwise']['pearson']:.4f}")
        print("="*70)
    
    fabric.barrier()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    os._exit(0)