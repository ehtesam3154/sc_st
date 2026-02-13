#!/usr/bin/env python3
"""
Adapter Topology Evaluation
============================
Measures topology damage from adapter transforms and compares adapter strategies.

Implements:
  1. Topology damage metrics (overlap@K delta, physical distance delta, patch compactness)
  2. AffineAdapter (per-dimension scale+shift, closed-form or learned)
  3. ClosedFormCORALAdapter (whiten-color, no gradient)
  4. Full comparison: MLP vs Affine vs CORAL adapters
  5. Decision rule: accept only if topology damage < threshold

Usage (from notebook):
    from adapter_topology_eval import compare_all_adapters, evaluate_adapter_topology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_utils import coral_loss, mmd_rbf_loss, set_seed
from tie_robust_metrics import compute_tie_robust_metrics


# =============================================================================
# Restricted adapter classes
# =============================================================================

class AffineAdapter(nn.Module):
    """
    Per-dimension affine adapter: z' = a * z + b.

    Can be initialized via closed-form mean/std matching or learned.
    Much less expressive than an MLP — cannot scramble neighborhoods,
    only rescale/shift each dimension independently.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale + self.shift

    @staticmethod
    def fit_closed_form(
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        eps: float = 1e-8,
    ) -> 'AffineAdapter':
        """
        Closed-form per-dimension mean/std matching.

        a_d = std_target_d / (std_source_d + eps)
        b_d = mean_target_d - a_d * mean_source_d
        """
        mu_s = z_source.mean(dim=0)
        mu_t = z_target.mean(dim=0)
        std_s = z_source.std(dim=0)
        std_t = z_target.std(dim=0)

        a = std_t / (std_s + eps)
        b = mu_t - a * mu_s

        adapter = AffineAdapter(embed_dim=z_source.shape[1])
        with torch.no_grad():
            adapter.scale.copy_(a)
            adapter.shift.copy_(b)
        return adapter


class ClosedFormCORALAdapter(nn.Module):
    """
    Closed-form CORAL adapter: whiten source, color with target stats.

    z_adapted = (z - mu_s) @ Sigma_s^{-1/2} @ Sigma_t^{1/2} + mu_t

    This is a global linear map, so it preserves relative directions.
    Neighborhood scrambling is minimal — only rotations/stretches that
    align the covariance ellipsoids.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        # Store as buffers (not parameters — no gradient)
        self.register_buffer('mu_s', torch.zeros(embed_dim))
        self.register_buffer('mu_t', torch.zeros(embed_dim))
        self.register_buffer('W', torch.eye(embed_dim))  # combined whiten-color matrix

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu_s) @ self.W + self.mu_t

    @staticmethod
    def fit(
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        shrinkage: float = 0.01,
    ) -> 'ClosedFormCORALAdapter':
        """
        Compute the whiten-color transform with shrinkage regularization.

        W = Sigma_s^{-1/2} @ Sigma_t^{1/2}
        """
        d = z_source.shape[1]
        adapter = ClosedFormCORALAdapter(embed_dim=d)

        mu_s = z_source.mean(dim=0)
        mu_t = z_target.mean(dim=0)

        z_s_c = z_source - mu_s
        z_t_c = z_target - mu_t

        cov_s = (z_s_c.T @ z_s_c) / max(z_s_c.shape[0] - 1, 1)
        cov_t = (z_t_c.T @ z_t_c) / max(z_t_c.shape[0] - 1, 1)

        # Shrinkage for numerical stability
        eye = torch.eye(d, device=z_source.device)
        cov_s = (1 - shrinkage) * cov_s + shrinkage * eye
        cov_t = (1 - shrinkage) * cov_t + shrinkage * eye

        # Sigma_s^{-1/2} via eigendecomposition
        eigvals_s, eigvecs_s = torch.linalg.eigh(cov_s)
        eigvals_s = eigvals_s.clamp(min=1e-7)
        S_s_inv_half = eigvecs_s @ torch.diag(1.0 / eigvals_s.sqrt()) @ eigvecs_s.T

        # Sigma_t^{1/2} via eigendecomposition
        eigvals_t, eigvecs_t = torch.linalg.eigh(cov_t)
        eigvals_t = eigvals_t.clamp(min=1e-7)
        S_t_half = eigvecs_t @ torch.diag(eigvals_t.sqrt()) @ eigvecs_t.T

        W = S_s_inv_half @ S_t_half

        with torch.no_grad():
            adapter.mu_s.copy_(mu_s)
            adapter.mu_t.copy_(mu_t)
            adapter.W.copy_(W)

        return adapter


# =============================================================================
# Topology damage evaluation
# =============================================================================

def patch_compactness(coords: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """
    Mean diameter (max pairwise distance) of kNN patches in physical space.

    Args:
        coords:  (n, 2) physical coordinates
        knn_idx: (n, k) neighbor indices

    Returns:
        (n,) array of patch diameters
    """
    diameters = np.empty(len(knn_idx))
    for i in range(len(knn_idx)):
        pts = coords[knn_idx[i]]
        dists = cdist(pts, pts)
        diameters[i] = dists.max()
    return diameters


@torch.no_grad()
def evaluate_adapter_topology(
    z_before: torch.Tensor,
    z_after: torch.Tensor,
    coords: torch.Tensor,
    k: int = 20,
    n_compact_sample: int = 500,
) -> Dict[str, float]:
    """
    Measure topology damage from an adapter transform on a SINGLE slide.

    Compares embedding kNN structure before vs after applying the adapter.

    Args:
        z_before: (n, d) embeddings WITHOUT adapter
        z_after:  (n, d) embeddings WITH adapter
        coords:   (n, 2) physical coordinates
        k:        number of neighbors
        n_compact_sample: how many spots to sample for compactness (expensive)

    Returns:
        dict with:
          'overlap_before', 'overlap_after', 'delta_overlap':
              overlap@K with physical kNN
          'hit_before', 'hit_after', 'delta_hit':
              tie-robust hit@K
          'phys_dist_median_before', 'phys_dist_median_after', 'delta_phys_dist_median':
              median physical distance of embedding kNN
          'compactness_ratio_before', 'compactness_ratio_after', 'delta_compactness_ratio':
              ratio of emb-kNN patch diameter to phys-kNN patch diameter
    """
    n = z_before.shape[0]
    device = z_before.device

    # Ensure coords on same device
    if coords.device != device:
        coords = coords.to(device)

    # Physical distances and kNN
    D_phys = torch.cdist(coords.float(), coords.float())
    D_phys.fill_diagonal_(float('inf'))
    _, phys_knn = torch.topk(D_phys, k=k, dim=1, largest=False)

    results = {}

    for tag, z in [('before', z_before), ('after', z_after)]:
        # Tie-robust metrics
        metrics = compute_tie_robust_metrics(z, coords, k=k)
        results[f'overlap_{tag}'] = float(metrics['overlap_at_k'].mean())
        results[f'hit_{tag}'] = float(metrics['hit_within_radius'].mean())
        results[f'recall_{tag}'] = float(metrics['recall_into_shell'].mean())

        # Embedding kNN
        D_emb = torch.cdist(z, z)
        D_emb.fill_diagonal_(float('inf'))
        _, emb_knn = torch.topk(D_emb, k=k, dim=1, largest=False)

        # Median physical distance of embedding kNN
        emb_knn_flat = emb_knn.reshape(-1)
        row_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(emb_knn).reshape(-1)
        phys_dists_of_emb_knn = D_phys[row_idx, emb_knn_flat]
        # Filter out inf (shouldn't happen but safety)
        valid = phys_dists_of_emb_knn < float('inf')
        results[f'phys_dist_median_{tag}'] = float(phys_dists_of_emb_knn[valid].median())
        results[f'phys_dist_mean_{tag}'] = float(phys_dists_of_emb_knn[valid].mean())

        # Patch compactness (subsample for speed)
        n_sample = min(n_compact_sample, n)
        sample_idx = np.random.choice(n, n_sample, replace=False)

        coords_np = coords.cpu().numpy()
        emb_knn_np = emb_knn.cpu().numpy()
        phys_knn_np = phys_knn.cpu().numpy()

        emb_diameters = patch_compactness(coords_np, emb_knn_np[sample_idx])
        phys_diameters = patch_compactness(coords_np, phys_knn_np[sample_idx])

        # Compactness ratio: emb / phys (1.0 = same, >1 = emb patches are larger)
        ratio = np.median(emb_diameters) / max(np.median(phys_diameters), 1e-8)
        results[f'compactness_ratio_{tag}'] = float(ratio)
        results[f'compactness_emb_median_{tag}'] = float(np.median(emb_diameters))
        results[f'compactness_phys_median_{tag}'] = float(np.median(phys_diameters))

    # Deltas (after - before)
    for metric in ['overlap', 'hit', 'recall', 'phys_dist_median', 'phys_dist_mean', 'compactness_ratio']:
        results[f'delta_{metric}'] = results[f'{metric}_after'] - results[f'{metric}_before']

    return results


@torch.no_grad()
def compute_domain_metrics(
    z_st: torch.Tensor,
    z_sc: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute domain alignment quality metrics (no sklearn needed).

    Returns:
        dict with 'centroid_dist', 'coral', 'mmd', 'norm_gap_pct'
    """
    # Centroid distance
    mu_st = z_st.mean(dim=0)
    mu_sc = z_sc.mean(dim=0)
    centroid_dist = (mu_st - mu_sc).norm().item()

    # CORAL
    coral = coral_loss(z_st, z_sc).item()

    # MMD (subsample for speed)
    n_sub = min(1000, z_st.shape[0], z_sc.shape[0])
    mmd = mmd_rbf_loss(
        z_st[torch.randperm(z_st.shape[0])[:n_sub]],
        z_sc[torch.randperm(z_sc.shape[0])[:n_sub]],
    ).item()

    # Norm gap
    norm_st = z_st.norm(dim=1).mean().item()
    norm_sc = z_sc.norm(dim=1).mean().item()
    norm_gap_pct = abs(norm_st - norm_sc) / max(norm_st, 1e-8) * 100

    return {
        'centroid_dist': centroid_dist,
        'coral': coral,
        'mmd': mmd,
        'norm_gap_pct': norm_gap_pct,
        'norm_st_mean': norm_st,
        'norm_sc_mean': norm_sc,
    }


def compute_domain_classifier_acc(
    z_st: torch.Tensor,
    z_sc: torch.Tensor,
) -> float:
    """
    Domain classifier accuracy (ST vs SC) via logistic regression 5-fold CV.
    Lower = better alignment.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    Z = torch.cat([z_st.cpu(), z_sc.cpu()], dim=0)
    Z_norm = F.normalize(Z, dim=1).numpy()
    y = np.concatenate([np.zeros(z_st.shape[0]), np.ones(z_sc.shape[0])])

    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(cross_val_score(clf, Z_norm, y, cv=cv, scoring='balanced_accuracy').mean())


# =============================================================================
# Per-slide embedding norm diagnostics
# =============================================================================

@torch.no_grad()
def per_slide_norm_table(
    encoder: nn.Module,
    slide_exprs: Dict[str, torch.Tensor],
    adapter: Optional[nn.Module] = None,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """
    Print and return per-slide embedding norm statistics.

    Also checks: after L2-norm, all norms == 1.
    """
    encoder.eval()
    if adapter is not None:
        adapter.eval()

    print(f"\n  {'Slide':<18s} | {'n':>6s} | {'norm_mean':>9s} | {'norm_std':>8s} | "
          f"{'norm_cv':>7s} | {'per-dim std':>11s} | {'adapter':>7s}")
    print("  " + "-" * 85)

    results = {}
    for name, expr in slide_exprs.items():
        expr_dev = expr.to(device)

        # Raw (no adapter)
        parts = []
        for i in range(0, expr_dev.shape[0], 512):
            parts.append(encoder(expr_dev[i:i + 512]))
        z_raw = torch.cat(parts, dim=0)

        norms = z_raw.norm(dim=1)
        per_dim_std = z_raw.std(dim=0).mean()

        results[name] = {
            'n': z_raw.shape[0],
            'norm_mean': norms.mean().item(),
            'norm_std': norms.std().item(),
            'norm_cv': (norms.std() / norms.mean()).item(),
            'per_dim_std_mean': per_dim_std.item(),
        }

        print(f"  {name:<18s} | {z_raw.shape[0]:>6d} | {norms.mean():>9.3f} | "
              f"{norms.std():>8.3f} | {norms.std()/norms.mean():>7.3f} | "
              f"{per_dim_std:>11.3f} | {'no':>7s}")

        # With adapter
        if adapter is not None:
            z_adapted = adapter(z_raw)
            norms_a = z_adapted.norm(dim=1)
            per_dim_std_a = z_adapted.std(dim=0).mean()

            results[name + '_adapted'] = {
                'n': z_adapted.shape[0],
                'norm_mean': norms_a.mean().item(),
                'norm_std': norms_a.std().item(),
                'norm_cv': (norms_a.std() / norms_a.mean()).item(),
                'per_dim_std_mean': per_dim_std_a.item(),
            }

            print(f"  {name+'_adapted':<18s} | {z_adapted.shape[0]:>6d} | {norms_a.mean():>9.3f} | "
                  f"{norms_a.std():>8.3f} | {norms_a.std()/norms_a.mean():>7.3f} | "
                  f"{per_dim_std_a:>11.3f} | {'yes':>7s}")

        # L2-normalized check
        z_l2 = F.normalize(z_raw, dim=1)
        l2_norms = z_l2.norm(dim=1)
        if not torch.allclose(l2_norms, torch.ones_like(l2_norms), atol=1e-5):
            print(f"    WARNING: L2-normalized norms not exactly 1 for {name}")

    return results


# =============================================================================
# Full adapter comparison
# =============================================================================

@torch.no_grad()
def _encode_all(encoder, expr, device='cuda', chunk=512):
    """Encode all spots without adapter."""
    parts = []
    for i in range(0, expr.shape[0], chunk):
        parts.append(encoder(expr[i:i + chunk].to(device)))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def compare_all_adapters(
    encoder: nn.Module,
    st_gene_exprs: Dict[str, torch.Tensor],
    st_coords: Dict[str, torch.Tensor],
    inf_gene_exprs: Dict[str, torch.Tensor],
    inf_coords: Dict[str, torch.Tensor],
    mlp_adapter: Optional[nn.Module] = None,
    k: int = 20,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Compare MLP, Affine, and CORAL adapters on topology damage + domain alignment.

    Args:
        encoder:        Frozen SharedEncoder
        st_gene_exprs:  {slide_name: (n, n_genes)} ST training slides
        st_coords:      {slide_name: (n, 2)} ST coordinates
        inf_gene_exprs: {slide_name: (n, n_genes)} inference slides (treated as SC)
        inf_coords:     {slide_name: (n, 2)} inference coordinates (for eval only)
        mlp_adapter:    Pre-trained SCAdapter (MLP). If None, skipped.
        k:              number of neighbors
        device:         cuda/cpu
        seed:           random seed

    Returns:
        dict {adapter_name: {metrics_dict}} for each adapter type
    """
    set_seed(seed)
    encoder.eval().to(device)

    # === Pre-compute all frozen embeddings ===
    print("=" * 80)
    print("ADAPTER TOPOLOGY COMPARISON")
    print("=" * 80)

    print("\n  Encoding all slides (frozen encoder)...")
    z_st_dict = {}
    for name, expr in st_gene_exprs.items():
        z_st_dict[name] = _encode_all(encoder, expr, device)

    z_inf_dict = {}
    for name, expr in inf_gene_exprs.items():
        z_inf_dict[name] = _encode_all(encoder, expr, device)

    z_st_all = torch.cat(list(z_st_dict.values()), dim=0)
    z_inf_all = torch.cat(list(z_inf_dict.values()), dim=0)

    # === Build adapters ===
    adapters = {}

    # 1. Affine (closed-form)
    adapters['affine'] = AffineAdapter.fit_closed_form(z_inf_all, z_st_all).to(device)

    # 2. Closed-form CORAL
    adapters['coral'] = ClosedFormCORALAdapter.fit(z_inf_all, z_st_all).to(device)

    # 3. MLP (pre-trained, if provided)
    if mlp_adapter is not None:
        mlp_adapter.eval().to(device)
        adapters['mlp'] = mlp_adapter

    # 4. Identity baseline (no adaptation)
    adapters['none'] = nn.Identity().to(device)

    # === Evaluate each adapter ===
    all_results = {}

    for adapter_name, adapter in adapters.items():
        print(f"\n{'─' * 80}")
        print(f"  ADAPTER: {adapter_name}")
        print(f"{'─' * 80}")

        adapter_results = {
            'topology_per_slide': {},
            'domain_metrics': {},
        }

        # --- Topology damage on ST slides ---
        # Apply adapter to ST (stress test: adapter is "for SC" but we test distortion)
        print(f"\n  Topology damage on ST training slides (stress test):")
        print(f"  {'Slide':<15s} | {'ovl_bef':>7s} | {'ovl_aft':>7s} | {'delta_ovl':>9s} | "
              f"{'hit_bef':>7s} | {'hit_aft':>7s} | {'delta_hit':>9s} | "
              f"{'dist_bef':>8s} | {'dist_aft':>8s} | {'compact_bef':>11s} | {'compact_aft':>11s}")
        print("  " + "-" * 130)

        for slide_name in st_gene_exprs:
            z_raw = z_st_dict[slide_name]
            z_adapted = adapter(z_raw)
            coords = st_coords[slide_name]
            if isinstance(coords, np.ndarray):
                coords = torch.tensor(coords, dtype=torch.float32)
            coords = coords.to(device)

            topo = evaluate_adapter_topology(z_raw, z_adapted, coords, k=k)
            adapter_results['topology_per_slide'][slide_name] = topo

            print(f"  {slide_name:<15s} | {topo['overlap_before']:>7.4f} | {topo['overlap_after']:>7.4f} | "
                  f"{topo['delta_overlap']:>+9.4f} | "
                  f"{topo['hit_before']:>7.4f} | {topo['hit_after']:>7.4f} | "
                  f"{topo['delta_hit']:>+9.4f} | "
                  f"{topo['phys_dist_median_before']:>8.3f} | {topo['phys_dist_median_after']:>8.3f} | "
                  f"{topo['compactness_ratio_before']:>11.3f} | {topo['compactness_ratio_after']:>11.3f}")

        # --- Topology damage on inference slides ---
        if inf_coords:
            print(f"\n  Topology damage on inference slides:")
            print(f"  {'Slide':<15s} | {'ovl_bef':>7s} | {'ovl_aft':>7s} | {'delta_ovl':>9s} | "
                  f"{'hit_bef':>7s} | {'hit_aft':>7s} | {'delta_hit':>9s} | "
                  f"{'dist_bef':>8s} | {'dist_aft':>8s} | {'compact_bef':>11s} | {'compact_aft':>11s}")
            print("  " + "-" * 130)

            for slide_name in inf_gene_exprs:
                z_raw = z_inf_dict[slide_name]
                z_adapted = adapter(z_raw)
                coords = inf_coords[slide_name]
                if isinstance(coords, np.ndarray):
                    coords = torch.tensor(coords, dtype=torch.float32)
                coords = coords.to(device)

                topo = evaluate_adapter_topology(z_raw, z_adapted, coords, k=k)
                adapter_results['topology_per_slide'][slide_name + '_inf'] = topo

                print(f"  {slide_name:<15s} | {topo['overlap_before']:>7.4f} | {topo['overlap_after']:>7.4f} | "
                      f"{topo['delta_overlap']:>+9.4f} | "
                      f"{topo['hit_before']:>7.4f} | {topo['hit_after']:>7.4f} | "
                      f"{topo['delta_hit']:>+9.4f} | "
                      f"{topo['phys_dist_median_before']:>8.3f} | {topo['phys_dist_median_after']:>8.3f} | "
                      f"{topo['compactness_ratio_before']:>11.3f} | {topo['compactness_ratio_after']:>11.3f}")

        # --- Domain alignment metrics ---
        z_inf_adapted = adapter(z_inf_all)
        domain = compute_domain_metrics(z_st_all, z_inf_adapted)
        adapter_results['domain_metrics'] = domain

        # Domain classifier
        domain_acc = compute_domain_classifier_acc(z_st_all, z_inf_adapted)
        adapter_results['domain_classifier_acc'] = domain_acc

        print(f"\n  Domain alignment:")
        print(f"    Centroid dist:       {domain['centroid_dist']:.4f}")
        print(f"    CORAL loss:          {domain['coral']:.4f}")
        print(f"    MMD loss:            {domain['mmd']:.6f}")
        print(f"    Norm gap:            {domain['norm_gap_pct']:.1f}%")
        print(f"    Domain classifier:   {domain_acc:.4f}  (0.50 = chance)")

        all_results[adapter_name] = adapter_results

    # === Summary comparison table ===
    _print_comparison_summary(all_results, k)

    # === Decision rule ===
    _print_decision(all_results, k)

    return all_results


def _print_comparison_summary(all_results: Dict, k: int):
    """Print side-by-side summary table."""
    print(f"\n{'=' * 80}")
    print(f"ADAPTER COMPARISON SUMMARY (k={k})")
    print(f"{'=' * 80}")

    print(f"\n  {'Adapter':<12s} | {'domain_acc':>10s} | {'centroid':>8s} | {'CORAL':>8s} | "
          f"{'MMD':>8s} | {'norm_gap':>8s} | "
          f"{'mean_d_ovl':>10s} | {'mean_d_hit':>10s} | {'mean_d_compact':>14s}")
    print("  " + "-" * 110)

    for name, res in all_results.items():
        # Average topology deltas across all slides
        topos = res['topology_per_slide']
        if topos:
            mean_d_ovl = np.mean([t['delta_overlap'] for t in topos.values()])
            mean_d_hit = np.mean([t['delta_hit'] for t in topos.values()])
            mean_d_compact = np.mean([
                t['compactness_ratio_after'] - t['compactness_ratio_before']
                for t in topos.values()
            ])
        else:
            mean_d_ovl = mean_d_hit = mean_d_compact = 0.0

        dm = res['domain_metrics']
        da = res.get('domain_classifier_acc', float('nan'))

        print(f"  {name:<12s} | {da:>10.4f} | {dm['centroid_dist']:>8.4f} | "
              f"{dm['coral']:>8.4f} | {dm['mmd']:>8.6f} | "
              f"{dm['norm_gap_pct']:>7.1f}% | "
              f"{mean_d_ovl:>+10.4f} | {mean_d_hit:>+10.4f} | {mean_d_compact:>+14.4f}")


def _print_decision(all_results: Dict, k: int):
    """
    Decision rule:
      Accept adapter only if:
        1. Domain classifier acc < 0.60 (close to chance)
        2. |mean delta overlap@K| < 0.01 on ST slides
        3. Compactness ratio doesn't blow up (delta < 0.5)
    """
    print(f"\n{'=' * 80}")
    print(f"DECISION RULE")
    print(f"{'=' * 80}")

    DOMAIN_ACC_THRESH = 0.60
    DELTA_OVERLAP_THRESH = 0.01
    DELTA_COMPACT_THRESH = 0.5

    print(f"  Criteria:")
    print(f"    1. Domain classifier acc < {DOMAIN_ACC_THRESH} (close to chance)")
    print(f"    2. |mean delta overlap@{k}| < {DELTA_OVERLAP_THRESH} on ST slides")
    print(f"    3. Delta compactness ratio < {DELTA_COMPACT_THRESH}")
    print()

    best_name = None
    best_score = -float('inf')

    for name, res in all_results.items():
        if name == 'none':
            continue

        da = res.get('domain_classifier_acc', 1.0)
        topos = res['topology_per_slide']

        # Only ST slides for topology check (keys without '_inf' suffix)
        st_topos = {k_: v for k_, v in topos.items() if not k_.endswith('_inf')}
        if st_topos:
            mean_d_ovl = np.mean([t['delta_overlap'] for t in st_topos.values()])
            mean_d_compact = np.mean([
                t['compactness_ratio_after'] - t['compactness_ratio_before']
                for t in st_topos.values()
            ])
        else:
            mean_d_ovl = 0.0
            mean_d_compact = 0.0

        pass_domain = da < DOMAIN_ACC_THRESH
        pass_topo = abs(mean_d_ovl) < DELTA_OVERLAP_THRESH
        pass_compact = mean_d_compact < DELTA_COMPACT_THRESH

        status = "ACCEPT" if (pass_domain and pass_topo and pass_compact) else "REJECT"
        reasons = []
        if not pass_domain:
            reasons.append(f"domain_acc={da:.3f} >= {DOMAIN_ACC_THRESH}")
        if not pass_topo:
            reasons.append(f"|delta_ovl|={abs(mean_d_ovl):.4f} >= {DELTA_OVERLAP_THRESH}")
        if not pass_compact:
            reasons.append(f"delta_compact={mean_d_compact:.3f} >= {DELTA_COMPACT_THRESH}")

        print(f"  {name:<12s}: {status}")
        if reasons:
            for r in reasons:
                print(f"    FAIL: {r}")
        else:
            print(f"    All criteria passed.")

        # Score: prefer low domain_acc while having low topology damage
        if status == "ACCEPT":
            score = -da  # lower domain acc is better among accepted
            if score > best_score:
                best_score = score
                best_name = name

    print()
    if best_name:
        print(f"  >>> RECOMMENDED: {best_name}")
    else:
        print(f"  >>> NO ADAPTER PASSES ALL CRITERIA.")
        print(f"  >>> Consider: (a) relaxing thresholds, (b) more encoder training,")
        print(f"  >>>           (c) adding identity regularizer to MLP adapter.")

    print(f"{'=' * 80}")


# =============================================================================
# Convenience: train MLP adapter with identity regularizer
# =============================================================================

def train_mlp_adapter_with_identity_reg(
    z_st_frozen: torch.Tensor,
    z_sc_frozen: torch.Tensor,
    adapter_mode: str = 'mlp',
    adapter_dropout: float = 0.1,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    coral_weight: float = 10.0,
    mmd_weight: float = 10.0,
    identity_weight: float = 5.0,
    device: str = 'cuda',
    seed: int = 42,
    log_every: int = 50,
) -> nn.Module:
    """
    Train MLP adapter with identity regularizer on ST embeddings.

    L = coral_w * CORAL(z_st, g(z_sc))
      + mmd_w   * MMD(z_st, g(z_sc))
      + id_w    * ||g(z_st) - z_st||^2

    The identity term forces the adapter to be close to identity on the
    ST manifold, preventing arbitrary neighborhood warping.
    """
    from sc_adapter import SCAdapter

    set_seed(seed)

    embed_dim = z_st_frozen.shape[1]
    adapter = SCAdapter(embed_dim=embed_dim, mode=adapter_mode, dropout=adapter_dropout).to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_st = z_st_frozen.shape[0]
    n_sc = z_sc_frozen.shape[0]

    print(f"\n  Training MLP adapter WITH identity regularizer (weight={identity_weight})")
    print(f"  Epochs: {n_epochs}, LR: {lr}, CORAL: {coral_weight}, MMD: {mmd_weight}, L_id: {identity_weight}")

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        adapter.train()

        # Sample balanced batches
        st_idx = torch.randperm(n_st, device=device)[:batch_size]
        sc_idx = torch.randperm(n_sc, device=device)[:batch_size]

        z_st_batch = z_st_frozen[st_idx]
        z_sc_batch = z_sc_frozen[sc_idx]

        z_sc_adapted = adapter(z_sc_batch)

        # Alignment losses
        loss = torch.tensor(0.0, device=device)
        l_coral = coral_loss(z_st_batch, z_sc_adapted) if coral_weight > 0 else torch.tensor(0.0)
        l_mmd = mmd_rbf_loss(z_st_batch, z_sc_adapted) if mmd_weight > 0 else torch.tensor(0.0)
        loss = loss + coral_weight * l_coral + mmd_weight * l_mmd

        # Identity regularizer: g(z_st) should be close to z_st
        z_st_through = adapter(z_st_batch)
        l_id = F.mse_loss(z_st_through, z_st_batch)
        loss = loss + identity_weight * l_id

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"    e{epoch:4d} | loss={loss.item():.4f} coral={l_coral.item():.4f} "
                  f"mmd={l_mmd.item():.4f} L_id={l_id.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k_: v.cpu().clone() for k_, v in adapter.state_dict().items()}

    if best_state is not None:
        adapter.load_state_dict(best_state)
        adapter.to(device)
    adapter.eval()

    return adapter
