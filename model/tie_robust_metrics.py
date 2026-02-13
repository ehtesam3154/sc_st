#!/usr/bin/env python3
"""
Tie-Robust Locality Metrics & Pre-Diffusion Sanity Checks
==========================================================
Replaces overlap@20 with metrics that handle lattice ties correctly.

Metrics:
  1. hit_within_radius@k:  fraction of emb-kNN within physical k-th neighbor distance
  2. recall_into_shell@k:  fraction of emb-kNN within the full tied shell S(i)
  3. tie_multiplicity:     |S(i)| - k, explaining why overlap can't reach 1

Pre-diffusion checks:
  - Embedding norm mean/std per slide
  - Cosine similarity distribution: ST4 vs ST-train
  - Tie-robust metrics on ST4 (via adapter)

Usage (from notebook):
    from tie_robust_metrics import run_tie_robust_analysis, run_prediffusion_checks
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_models_et_p1 import SharedEncoder


@torch.no_grad()
def compute_tie_robust_metrics(
    z: torch.Tensor,
    coords: torch.Tensor,
    k: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute tie-robust locality metrics for one slide.

    Args:
        z:      (n, d) embeddings
        coords: (n, 2) physical coordinates
        k:      number of neighbors

    Returns dict with per-spot arrays:
        'hit_within_radius':   fraction of emb-kNN within d_k(i)
        'recall_into_shell':   fraction of emb-kNN within S(i)
        'tie_shell_size':      |S(i)| for each spot
        'tie_excess':          |S(i)| - k for each spot
        'overlap_at_k':        standard overlap for comparison
    """
    n = z.shape[0]
    assert n > k

    # Physical distances
    D_phys = torch.cdist(coords, coords)  # (n, n)
    D_phys.fill_diagonal_(float('inf'))

    # Physical k-th neighbor distance per spot
    D_phys_sorted, _ = D_phys.sort(dim=1)
    d_k = D_phys_sorted[:, k - 1]  # (n,) distance to k-th neighbor (0-indexed: k-1)

    # Physical kNN indices (for standard overlap comparison)
    _, phys_knn = torch.topk(D_phys, k=k, dim=1, largest=False)  # (n, k)

    # Tied shell S(i) = {j : d_phys(i,j) <= d_k(i)}
    # Add small epsilon for float tolerance
    eps = 1e-6
    shell_mask = D_phys <= (d_k.unsqueeze(1) + eps)  # (n, n) bool
    shell_sizes = shell_mask.sum(dim=1)  # (n,) = |S(i)|

    # Embedding kNN
    D_emb = torch.cdist(z, z)
    D_emb.fill_diagonal_(float('inf'))
    _, emb_knn = torch.topk(D_emb, k=k, dim=1, largest=False)  # (n, k)

    # Compute metrics per spot
    hit_within_radius = torch.zeros(n)
    recall_into_shell = torch.zeros(n)
    overlap_at_k = torch.zeros(n)

    for i in range(n):
        emb_set = set(emb_knn[i].cpu().tolist())
        phys_set = set(phys_knn[i].cpu().tolist())

        # hit_within_radius: emb neighbors within d_k(i)
        emb_neighbor_phys_dists = D_phys[i, emb_knn[i]]  # (k,)
        hits = (emb_neighbor_phys_dists <= d_k[i] + eps).sum().item()
        hit_within_radius[i] = hits / k

        # recall_into_shell: emb neighbors in S(i)
        shell_set = set(torch.where(shell_mask[i])[0].cpu().tolist())
        recall_into_shell[i] = len(emb_set & shell_set) / k

        # standard overlap
        overlap_at_k[i] = len(emb_set & phys_set) / k

    return {
        'hit_within_radius': hit_within_radius.numpy(),
        'recall_into_shell': recall_into_shell.numpy(),
        'tie_shell_size': shell_sizes.cpu().numpy(),
        'tie_excess': (shell_sizes - k).cpu().numpy(),
        'overlap_at_k': overlap_at_k.numpy(),
    }


def run_tie_robust_analysis(
    encoder: torch.nn.Module,
    st_gene_exprs: Dict[str, torch.Tensor],
    st_coords_raw: Dict[str, torch.Tensor],
    adapter: Optional[torch.nn.Module] = None,
    k: int = 20,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Run tie-robust metrics on multiple slides.

    Args:
        encoder:       SharedEncoder (frozen)
        st_gene_exprs: dict {slide_name: (n_i, n_genes) tensor}
        st_coords_raw: dict {slide_name: (n_i, 2) raw spatial coords}
        adapter:       Optional SCAdapter (applied if provided, for SC/inference data)
        k:             number of neighbors
        device:        cuda/cpu

    Returns dict {slide_name: metrics_dict}
    """
    encoder.eval()
    if adapter is not None:
        adapter.eval()

    results = {}

    for name in st_gene_exprs:
        expr = st_gene_exprs[name].to(device)
        coords = st_coords_raw[name]
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.to(device)

        # Compute embeddings
        with torch.no_grad():
            z_parts = []
            for i in range(0, expr.shape[0], 512):
                z_batch = encoder(expr[i:i + 512])
                if adapter is not None:
                    z_batch = adapter(z_batch)
                z_parts.append(z_batch)
            z = torch.cat(z_parts, dim=0)

        # Compute metrics
        metrics = compute_tie_robust_metrics(z, coords, k=k)
        results[name] = metrics

    return results


def print_tie_robust_report(results: Dict[str, Dict], k: int = 20):
    """Print formatted report of tie-robust metrics."""
    print(f"\n{'Slide':<15s} | {'hit@{k}':>8s} | {'recall@{k}':>10s} | {'overlap@{k}':>10s} | "
          f"{'|S| mean':>8s} | {'|S|-k mean':>10s} | {'|S|-k p50':>9s} | {'|S|-k p90':>9s} | n")
    print("  " + "-" * 110)

    for name, m in results.items():
        n = len(m['hit_within_radius'])
        print(f"  {name:<13s} | {m['hit_within_radius'].mean():>8.4f} | "
              f"{m['recall_into_shell'].mean():>10.4f} | "
              f"{m['overlap_at_k'].mean():>10.4f} | "
              f"{m['tie_shell_size'].mean():>8.1f} | "
              f"{m['tie_excess'].mean():>10.1f} | "
              f"{np.median(m['tie_excess']):>9.0f} | "
              f"{np.percentile(m['tie_excess'], 90):>9.0f} | "
              f"{n}")

    print()

    # Interpretation
    all_hit = np.concatenate([m['hit_within_radius'] for m in results.values()])
    all_recall = np.concatenate([m['recall_into_shell'] for m in results.values()])
    all_overlap = np.concatenate([m['overlap_at_k'] for m in results.values()])
    all_excess = np.concatenate([m['tie_excess'] for m in results.values()])

    print(f"  OVERALL:")
    print(f"    hit_within_radius@{k}:  {all_hit.mean():.4f}  (1.0 = embedding kNN as local as physical kNN)")
    print(f"    recall_into_shell@{k}:  {all_recall.mean():.4f}  (1.0 = all emb-kNN in the tied shell)")
    print(f"    overlap@{k}:            {all_overlap.mean():.4f}  (limited by tie multiplicity)")
    print(f"    tie excess |S|-k:       mean={all_excess.mean():.1f}, median={np.median(all_excess):.0f}")
    print(f"    => With {all_excess.mean():.0f} extra spots in the shell on average, "
          f"overlap is bounded by ~k/(k+{all_excess.mean():.0f}) = {k/(k + all_excess.mean()):.3f}")

    if all_hit.mean() > 0.90:
        print(f"\n  VERDICT: Embedding is SPATIALLY LOCAL at the correct scale.")
        print(f"  overlap@{k}={all_overlap.mean():.3f} is fully explained by lattice tie multiplicity.")
        print(f"  No further Phase 1 training is needed.")
    elif all_hit.mean() > 0.70:
        print(f"\n  VERDICT: Embedding is moderately spatially local. Some room for improvement.")
    else:
        print(f"\n  VERDICT: Embedding locality is weak. Consider more Phase 1 training.")


def run_prediffusion_checks(
    encoder: torch.nn.Module,
    adapter: torch.nn.Module,
    st_gene_exprs: Dict[str, torch.Tensor],
    st_coords_raw: Dict[str, torch.Tensor],
    inf_gene_exprs: Dict[str, torch.Tensor],
    inf_coords_raw: Dict[str, torch.Tensor],
    k: int = 20,
    device: str = 'cuda',
):
    """
    Pre-diffusion sanity checks:
      1. Embedding norm mean/std per slide
      2. Cosine similarity distribution: ST4 vs ST-train
      3. Tie-robust metrics on ST4 (via adapter)
    """
    encoder.eval()
    adapter.eval()

    print("=" * 70)
    print("PRE-DIFFUSION SANITY CHECKS")
    print("=" * 70)

    # === 1. Embedding norms per slide ===
    print(f"\n  [CHECK 1] Embedding norm statistics per slide")
    print(f"  {'Slide':<15s} | {'n':>6s} | {'norm_mean':>9s} | {'norm_std':>8s} | {'norm_cv':>7s} | {'uses adapter':>12s}")
    print(f"  " + "-" * 75)

    all_z = {}
    for name, expr in st_gene_exprs.items():
        with torch.no_grad():
            z = _encode_all(encoder, expr.to(device), adapter=None)
        norms = z.norm(dim=1)
        print(f"  {name:<15s} | {z.shape[0]:>6d} | {norms.mean():>9.3f} | {norms.std():>8.3f} | "
              f"{norms.std()/norms.mean():>7.3f} | {'no':>12s}")
        all_z[name] = z

    for name, expr in inf_gene_exprs.items():
        with torch.no_grad():
            z_raw = _encode_all(encoder, expr.to(device), adapter=None)
            z_adapted = _encode_all(encoder, expr.to(device), adapter=adapter)
        norms_raw = z_raw.norm(dim=1)
        norms_adapted = z_adapted.norm(dim=1)
        print(f"  {name:<15s} | {z_raw.shape[0]:>6d} | {norms_raw.mean():>9.3f} | {norms_raw.std():>8.3f} | "
              f"{norms_raw.std()/norms_raw.mean():>7.3f} | {'no (raw)':>12s}")
        print(f"  {name+'_adapted':<15s} | {z_adapted.shape[0]:>6d} | {norms_adapted.mean():>9.3f} | "
              f"{norms_adapted.std():>8.3f} | {norms_adapted.std()/norms_adapted.mean():>7.3f} | {'yes':>12s}")
        all_z[name] = z_raw
        all_z[name + '_adapted'] = z_adapted

    # === 2. Cosine similarity: ST4 vs ST-train ===
    print(f"\n  [CHECK 2] Cosine similarity: inference slides vs ST-train")

    z_st_cat = torch.cat([all_z[n] for n in st_gene_exprs], dim=0)
    z_st_norm = F.normalize(z_st_cat, dim=1)

    for name in inf_gene_exprs:
        # Raw (no adapter)
        z_inf_norm = F.normalize(all_z[name], dim=1)
        sims_raw = z_inf_norm @ z_st_norm.T  # (n_inf, n_st)
        top1_raw = sims_raw.max(dim=1).values
        mean_raw = sims_raw.mean()

        # Adapted
        z_inf_adapted_norm = F.normalize(all_z[name + '_adapted'], dim=1)
        sims_adapted = z_inf_adapted_norm @ z_st_norm.T
        top1_adapted = sims_adapted.max(dim=1).values
        mean_adapted = sims_adapted.mean()

        print(f"\n  {name} vs ST-train:")
        print(f"    Raw:     mean_sim={mean_raw:.4f}, top1_sim: mean={top1_raw.mean():.4f}, "
              f"min={top1_raw.min():.4f}, p5={torch.quantile(top1_raw, 0.05):.4f}")
        print(f"    Adapted: mean_sim={mean_adapted:.4f}, top1_sim: mean={top1_adapted.mean():.4f}, "
              f"min={top1_adapted.min():.4f}, p5={torch.quantile(top1_adapted, 0.05):.4f}")

    # === 3. Tie-robust metrics on inference slides ===
    print(f"\n  [CHECK 3] Tie-robust locality on inference slides (via adapter)")

    inf_metrics = {}
    for name, expr in inf_gene_exprs.items():
        coords = inf_coords_raw[name]
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.to(device)

        with torch.no_grad():
            z = _encode_all(encoder, expr.to(device), adapter=adapter)

        metrics = compute_tie_robust_metrics(z, coords, k=k)
        inf_metrics[name] = metrics

        print(f"\n  {name} (adapted, n={expr.shape[0]}):")
        print(f"    hit_within_radius@{k}: {metrics['hit_within_radius'].mean():.4f}")
        print(f"    recall_into_shell@{k}: {metrics['recall_into_shell'].mean():.4f}")
        print(f"    overlap@{k}:           {metrics['overlap_at_k'].mean():.4f}")
        print(f"    tie excess |S|-k:      mean={metrics['tie_excess'].mean():.1f}, "
              f"median={np.median(metrics['tie_excess']):.0f}")

    # === Summary ===
    print(f"\n" + "=" * 70)
    print(f"PRE-DIFFUSION CHECK SUMMARY")
    print(f"=" * 70)

    # Norm consistency
    st_norms = [all_z[n].norm(dim=1).mean().item() for n in st_gene_exprs]
    inf_adapted_norms = [all_z[n + '_adapted'].norm(dim=1).mean().item() for n in inf_gene_exprs]
    norm_gap = abs(np.mean(st_norms) - np.mean(inf_adapted_norms)) / np.mean(st_norms)
    print(f"  Norm gap (ST vs adapted SC): {norm_gap*100:.1f}%  {'PASS' if norm_gap < 0.2 else 'WARN'}")

    # Cosine similarity
    print(f"  Cosine sim (adapted SC → ST): mean={mean_adapted:.4f}  "
          f"{'PASS' if mean_adapted > 0.0 else 'WARN: negative mean similarity'}")

    # Locality on inference
    for name in inf_gene_exprs:
        h = inf_metrics[name]['hit_within_radius'].mean()
        print(f"  {name} hit@{k}: {h:.4f}  {'PASS' if h > 0.5 else 'WARN: low locality'}")

    print(f"\n  Ready for diffusion conditioning: "
          f"{'YES' if norm_gap < 0.2 and mean_adapted > -0.1 else 'REVIEW WARNINGS'}")
    print("=" * 70)


@torch.no_grad()
def _encode_all(encoder, expr, adapter=None, chunk=512):
    """Encode all expression data in chunks."""
    parts = []
    for i in range(0, expr.shape[0], chunk):
        z = encoder(expr[i:i + chunk])
        if adapter is not None:
            z = adapter(z)
        parts.append(z)
    return torch.cat(parts, dim=0)
