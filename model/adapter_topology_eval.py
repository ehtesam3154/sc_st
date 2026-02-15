#!/usr/bin/env python3
"""
Adapter Topology Evaluation
============================
Measures topology damage from adapter transforms and compares adapter strategies.

Implements:
  1. Topology damage metrics (overlap@K delta, physical distance delta, patch compactness)
  2. Inference-realistic patch quality test (mutual kNN + Jaccard graph, random walk patches)
  3. AffineAdapter (per-dimension scale+shift, closed-form)
  4. GlobalScaleShiftAdapter (single scalar + global shift, maximally restricted)
  5. ClosedFormCORALAdapter (whiten-color, no gradient)
  6. Structure-preserving MLP adapter training (cosine sim on ST kNN edges)
  7. Decision rule: accept only if topology damage < threshold

Usage (from notebook):
    from adapter_topology_eval import compare_all_adapters, evaluate_patch_quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import random
from typing import Dict, Optional, List, Tuple
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_utils import coral_loss, mmd_rbf_loss, set_seed
from tie_robust_metrics import compute_tie_robust_metrics


# =============================================================================
# Restricted adapter classes
# =============================================================================

class GlobalScaleShiftAdapter(nn.Module):
    """
    Most restricted adapter: z' = alpha * z + beta.

    alpha is a SINGLE scalar, beta is a global shift vector.
    Preserves cosine ordering exactly (only scales norms + shifts centroid).
    Cannot change relative directions at all.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.alpha + self.beta

    @staticmethod
    def fit_closed_form(
        z_source: torch.Tensor,
        z_target: torch.Tensor,
    ) -> 'GlobalScaleShiftAdapter':
        """
        Closed-form: match global mean norm, then shift centroid.

        alpha = mean(||z_t||) / mean(||z_s||)
        beta  = mean(z_t) - alpha * mean(z_s)
        """
        norm_s = z_source.norm(dim=1).mean()
        norm_t = z_target.norm(dim=1).mean()
        alpha = norm_t / (norm_s + 1e-8)

        mu_s = z_source.mean(dim=0)
        mu_t = z_target.mean(dim=0)
        beta = mu_t - alpha * mu_s

        adapter = GlobalScaleShiftAdapter(embed_dim=z_source.shape[1])
        with torch.no_grad():
            adapter.alpha.fill_(alpha.item())
            adapter.beta.copy_(beta)
        return adapter


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
# Inference-realistic patch quality test
# =============================================================================

def _build_locality_graph(
    Z: np.ndarray,
    k_Z: int = 40,
    k_sigma: int = 10,
    tau_jaccard: float = 0.10,
    min_shared: int = 5,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float], Dict]:
    """
    Build the same mutual-kNN + Jaccard-filtered weighted graph used by
    inference (build_locality_graph_v2 in core_models_et_p2.py).

    Returns:
        adj_list:     {node: [neighbors]}
        edge_weights: {(i,j): weight}
        diagnostics:  dict with graph stats
    """
    N = Z.shape[0]

    # Step 1: directed kNN
    nbrs = NearestNeighbors(n_neighbors=min(k_Z + 1, N), algorithm='auto').fit(Z)
    distances, indices = nbrs.kneighbors(Z)
    knn_sets = [set(indices[i, 1:k_Z + 1].tolist()) for i in range(N)]

    # Step 2: mutual filter
    mutual_edges = set()
    for i in range(N):
        for j in knn_sets[i]:
            if i in knn_sets[j]:
                mutual_edges.add((min(i, j), max(i, j)))

    # Step 3: Jaccard filter
    filtered_edges = []
    for (i, j) in mutual_edges:
        intersection = len(knn_sets[i] & knn_sets[j])
        union = len(knn_sets[i] | knn_sets[j])
        jaccard = intersection / (union + 1e-8)
        if jaccard >= tau_jaccard or intersection >= min_shared:
            filtered_edges.append((i, j))

    # Step 4: Gaussian edge weights with self-tuned bandwidth
    local_scales = distances[:, min(k_sigma, distances.shape[1] - 1)]
    Z_t = Z  # already numpy

    adj_list = {i: [] for i in range(N)}
    edge_weights = {}

    for (i, j) in filtered_edges:
        d_ij = np.linalg.norm(Z_t[i] - Z_t[j])
        sigma_i = max(local_scales[i], 1e-8)
        sigma_j = max(local_scales[j], 1e-8)
        w_ij = np.exp(-d_ij ** 2 / (sigma_i * sigma_j + 1e-8))

        adj_list[i].append(j)
        adj_list[j].append(i)
        edge_weights[(i, j)] = w_ij
        edge_weights[(j, i)] = w_ij

    # Diagnostics
    degrees = [len(adj_list[i]) for i in range(N)]
    isolated = sum(1 for d in degrees if d == 0)
    n_components = _count_components(adj_list, N)

    diagnostics = {
        'n_nodes': N,
        'n_directed_knn_edges': sum(len(s) for s in knn_sets),
        'n_mutual_edges': len(mutual_edges),
        'n_filtered_edges': len(filtered_edges),
        'mutuality_rate': len(mutual_edges) / max(sum(len(s) for s in knn_sets) // 2, 1),
        'jaccard_pass_rate': len(filtered_edges) / max(len(mutual_edges), 1),
        'mean_degree': np.mean(degrees),
        'median_degree': np.median(degrees),
        'isolated_nodes': isolated,
        'n_components': n_components,
    }

    return adj_list, edge_weights, diagnostics


def _count_components(adj_list: Dict[int, List[int]], N: int) -> int:
    """BFS-based connected component count."""
    visited = set()
    n_comp = 0
    for start in range(N):
        if start in visited:
            continue
        n_comp += 1
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for nb in adj_list[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return n_comp


def _sample_patches_random_walk(
    adj_list: Dict[int, List[int]],
    edge_weights: Dict[Tuple[int, int], float],
    N: int,
    patch_size: int = 256,
    n_patches: int = 20,
    seed: int = 42,
) -> List[List[int]]:
    """
    Sample patches via weighted random walk + sliding window,
    matching the inference routine (sample_patches_random_walk_v2).
    """
    rng = random.Random(seed)
    active_nodes = [i for i in range(N) if adj_list[i]]
    if not active_nodes:
        return []

    stride = max(1, patch_size // 2)
    walk_length = stride * (n_patches + 5) + patch_size

    # Weighted random walk
    current = rng.choice(active_nodes)
    walk = [current]
    for _ in range(walk_length - 1):
        neighbors = adj_list[current]
        if not neighbors:
            current = rng.choice(active_nodes)
        else:
            weights = [edge_weights.get((current, n), 0.01) for n in neighbors]
            total = sum(weights)
            weights = [w / total for w in weights]
            current = rng.choices(neighbors, weights=weights, k=1)[0]
        walk.append(current)

    # Sliding window extraction
    patches = []
    pos = 0
    while pos + patch_size <= len(walk) and len(patches) < n_patches:
        window = walk[pos:pos + patch_size]
        unique_nodes = list(dict.fromkeys(window))
        # BFS extend if too few unique
        if len(unique_nodes) < patch_size * 0.5:
            seen = set(unique_nodes)
            queue = list(unique_nodes)
            while len(unique_nodes) < patch_size and queue:
                node = queue.pop(0)
                for nb in adj_list[node]:
                    if nb not in seen:
                        seen.add(nb)
                        unique_nodes.append(nb)
                        queue.append(nb)
        unique_nodes = unique_nodes[:patch_size]
        if len(unique_nodes) >= patch_size * 0.3:
            patches.append(unique_nodes)
        pos += stride

    return patches


@torch.no_grad()
def evaluate_patch_quality(
    z: torch.Tensor,
    coords: torch.Tensor,
    k_Z: int = 40,
    tau_jaccard: float = 0.10,
    min_shared: int = 5,
    patch_size: int = 256,
    n_patches: int = 30,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Build the inference-style locality graph and sample patches.
    Report physical compactness of those patches.

    This directly tests whether patch sampling produces the same kind of
    local sets the diffusion model was trained on.

    Args:
        z:           (n, d) embeddings
        coords:      (n, 2) physical coordinates
        k_Z:         kNN for graph construction
        tau_jaccard:  Jaccard threshold
        min_shared:  minimum shared neighbors threshold
        patch_size:  target cells per patch
        n_patches:   how many patches to sample
        seed:        random seed

    Returns dict with:
        'graph_*':      graph diagnostics (n_components, mean_degree, etc.)
        'patch_diameter_median/p90':  physical diameter of patches
        'patch_radius_median/p90':    physical radius (max dist from centroid)
        'patch_nn_overlap_mean':      mean overlap of patch members with
                                       physical kNN (are patch members spatially close?)
    """
    Z_np = z.cpu().numpy()
    coords_np = coords.cpu().numpy()
    N = z.shape[0]

    # Build graph
    adj_list, edge_weights, graph_diag = _build_locality_graph(
        Z_np, k_Z=k_Z, tau_jaccard=tau_jaccard, min_shared=min_shared
    )

    # Sample patches
    patches = _sample_patches_random_walk(
        adj_list, edge_weights, N,
        patch_size=patch_size, n_patches=n_patches, seed=seed
    )

    results = {f'graph_{k}': v for k, v in graph_diag.items()}
    results['n_patches_sampled'] = len(patches)

    if not patches:
        results['patch_diameter_median'] = float('nan')
        results['patch_diameter_p90'] = float('nan')
        results['patch_radius_median'] = float('nan')
        results['patch_radius_p90'] = float('nan')
        results['patch_nn_overlap_mean'] = float('nan')
        return results

    # Compute physical kNN for overlap comparison
    k_phys = 20
    phys_nbrs = NearestNeighbors(n_neighbors=k_phys + 1, algorithm='auto').fit(coords_np)
    _, phys_knn_idx = phys_nbrs.kneighbors(coords_np)
    phys_knn_idx = phys_knn_idx[:, 1:]  # exclude self

    diameters = []
    radii = []
    nn_overlaps = []

    for patch in patches:
        patch_coords = coords_np[patch]

        # Diameter: max pairwise distance
        D = cdist(patch_coords, patch_coords)
        diameters.append(D.max())

        # Radius: max distance from centroid
        centroid = patch_coords.mean(axis=0)
        dists_to_centroid = np.linalg.norm(patch_coords - centroid, axis=1)
        radii.append(dists_to_centroid.max())

        # NN overlap: for each cell in patch, what fraction of its
        # physical k=20 neighbors are also in the patch?
        patch_set = set(patch)
        overlaps = []
        for cell in patch:
            phys_nbs = set(phys_knn_idx[cell].tolist())
            if phys_nbs:
                overlaps.append(len(phys_nbs & patch_set) / len(phys_nbs))
        if overlaps:
            nn_overlaps.append(np.mean(overlaps))

    diameters = np.array(diameters)
    radii = np.array(radii)

    results['patch_diameter_median'] = float(np.median(diameters))
    results['patch_diameter_p90'] = float(np.percentile(diameters, 90))
    results['patch_radius_median'] = float(np.median(radii))
    results['patch_radius_p90'] = float(np.percentile(radii, 90))
    results['patch_nn_overlap_mean'] = float(np.mean(nn_overlaps)) if nn_overlaps else float('nan')

    return results


# =============================================================================
# Topology damage evaluation (overlap@K based, from v1)
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

    Returns dict with before/after/delta for:
        overlap, hit, recall, phys_dist_median, compactness_ratio
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
    mu_st = z_st.mean(dim=0)
    mu_sc = z_sc.mean(dim=0)
    centroid_dist = (mu_st - mu_sc).norm().item()

    coral = coral_loss(z_st, z_sc).item()

    n_sub = min(1000, z_st.shape[0], z_sc.shape[0])
    mmd = mmd_rbf_loss(
        z_st[torch.randperm(z_st.shape[0])[:n_sub]],
        z_sc[torch.randperm(z_sc.shape[0])[:n_sub]],
    ).item()

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

        z_l2 = F.normalize(z_raw, dim=1)
        l2_norms = z_l2.norm(dim=1)
        if not torch.allclose(l2_norms, torch.ones_like(l2_norms), atol=1e-5):
            print(f"    WARNING: L2-normalized norms not exactly 1 for {name}")

    return results


# =============================================================================
# Full adapter comparison (v2: includes patch quality)
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
    run_patch_quality: bool = True,
    patch_size: int = 256,
    n_patches: int = 30,
) -> Dict[str, Dict]:
    """
    Compare adapters on topology damage + domain alignment + patch quality.

    Args:
        encoder:          Frozen SharedEncoder
        st_gene_exprs:    {slide_name: (n, n_genes)} ST training slides
        st_coords:        {slide_name: (n, 2)} ST coordinates
        inf_gene_exprs:   {slide_name: (n, n_genes)} inference slides
        inf_coords:       {slide_name: (n, 2)} inference coordinates (for eval)
        mlp_adapter:      Pre-trained SCAdapter (MLP). If None, skipped.
        k:                number of neighbors for overlap@K
        device:           cuda/cpu
        seed:             random seed
        run_patch_quality: if True, run inference-realistic patch quality test
        patch_size:       patch size for patch quality test
        n_patches:        number of patches to sample per slide
    """
    set_seed(seed)
    encoder.eval().to(device)

    # === Pre-compute all frozen embeddings ===
    print("=" * 80)
    print("ADAPTER TOPOLOGY COMPARISON (v2: with patch quality)")
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

    # 1. Global scale+shift (most restricted)
    adapters['global_ss'] = GlobalScaleShiftAdapter.fit_closed_form(z_inf_all, z_st_all).to(device)

    # 2. Per-dim affine
    adapters['affine'] = AffineAdapter.fit_closed_form(z_inf_all, z_st_all).to(device)

    # 3. Closed-form CORAL
    adapters['coral'] = ClosedFormCORALAdapter.fit(z_inf_all, z_st_all).to(device)

    # 4. MLP (pre-trained, if provided)
    if mlp_adapter is not None:
        mlp_adapter.eval().to(device)
        adapters['mlp'] = mlp_adapter

    # 5. Identity baseline (no adaptation)
    adapters['none'] = nn.Identity().to(device)

    # === Evaluate each adapter ===
    all_results = {}

    for adapter_name, adapter in adapters.items():
        print(f"\n{'─' * 80}")
        print(f"  ADAPTER: {adapter_name}")
        print(f"{'─' * 80}")

        adapter_results = {
            'topology_per_slide': {},
            'patch_quality_per_slide': {},
            'domain_metrics': {},
        }

        # --- Topology damage on ST slides ---
        print(f"\n  [1] Topology damage on ST training slides (stress test):")
        print(f"  {'Slide':<15s} | {'ovl_bef':>7s} | {'ovl_aft':>7s} | {'d_ovl':>7s} | "
              f"{'hit_bef':>7s} | {'hit_aft':>7s} | {'d_hit':>7s} | "
              f"{'dist_bef':>8s} | {'dist_aft':>8s} | {'cpt_bef':>7s} | {'cpt_aft':>7s}")
        print("  " + "-" * 110)

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
                  f"{topo['delta_overlap']:>+7.4f} | "
                  f"{topo['hit_before']:>7.4f} | {topo['hit_after']:>7.4f} | "
                  f"{topo['delta_hit']:>+7.4f} | "
                  f"{topo['phys_dist_median_before']:>8.1f} | {topo['phys_dist_median_after']:>8.1f} | "
                  f"{topo['compactness_ratio_before']:>7.3f} | {topo['compactness_ratio_after']:>7.3f}")

        # --- Topology damage on inference slides ---
        if inf_coords:
            print(f"\n  [2] Topology damage on inference slides:")
            print(f"  {'Slide':<15s} | {'ovl_bef':>7s} | {'ovl_aft':>7s} | {'d_ovl':>7s} | "
                  f"{'hit_bef':>7s} | {'hit_aft':>7s} | {'d_hit':>7s} | "
                  f"{'dist_bef':>8s} | {'dist_aft':>8s} | {'cpt_bef':>7s} | {'cpt_aft':>7s}")
            print("  " + "-" * 110)

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
                      f"{topo['delta_overlap']:>+7.4f} | "
                      f"{topo['hit_before']:>7.4f} | {topo['hit_after']:>7.4f} | "
                      f"{topo['delta_hit']:>+7.4f} | "
                      f"{topo['phys_dist_median_before']:>8.1f} | {topo['phys_dist_median_after']:>8.1f} | "
                      f"{topo['compactness_ratio_before']:>7.3f} | {topo['compactness_ratio_after']:>7.3f}")

        # --- Patch quality (inference-realistic) ---
        if run_patch_quality:
            print(f"\n  [3] Inference-realistic patch quality (mutual-kNN + Jaccard graph → random walk):")
            print(f"  {'Slide':<15s} | {'n_comp':>6s} | {'mean_deg':>8s} | {'diam_p50':>8s} | "
                  f"{'diam_p90':>8s} | {'rad_p50':>7s} | {'rad_p90':>7s} | {'nn_ovl':>7s} | {'isolated':>8s}")
            print("  " + "-" * 100)

            # Test on all slides where we have coords
            all_slide_z = {}
            all_slide_coords = {}
            for name in st_gene_exprs:
                all_slide_z[name] = adapter(z_st_dict[name])
                c = st_coords[name]
                if isinstance(c, np.ndarray):
                    c = torch.tensor(c, dtype=torch.float32)
                all_slide_coords[name] = c
            for name in inf_gene_exprs:
                all_slide_z[name + '_inf'] = adapter(z_inf_dict[name])
                c = inf_coords[name]
                if isinstance(c, np.ndarray):
                    c = torch.tensor(c, dtype=torch.float32)
                all_slide_coords[name + '_inf'] = c

            for slide_name, z_adapted in all_slide_z.items():
                coords = all_slide_coords[slide_name]
                pq = evaluate_patch_quality(
                    z_adapted, coords,
                    patch_size=patch_size, n_patches=n_patches, seed=seed,
                )
                adapter_results['patch_quality_per_slide'][slide_name] = pq

                print(f"  {slide_name:<15s} | {pq['graph_n_components']:>6d} | "
                      f"{pq['graph_mean_degree']:>8.1f} | "
                      f"{pq['patch_diameter_median']:>8.1f} | "
                      f"{pq['patch_diameter_p90']:>8.1f} | "
                      f"{pq['patch_radius_median']:>7.1f} | "
                      f"{pq['patch_radius_p90']:>7.1f} | "
                      f"{pq['patch_nn_overlap_mean']:>7.4f} | "
                      f"{pq['graph_isolated_nodes']:>8d}")

        # --- Domain alignment metrics ---
        z_inf_adapted = adapter(z_inf_all)
        domain = compute_domain_metrics(z_st_all, z_inf_adapted)
        adapter_results['domain_metrics'] = domain

        domain_acc = compute_domain_classifier_acc(z_st_all, z_inf_adapted)
        adapter_results['domain_classifier_acc'] = domain_acc

        print(f"\n  [4] Domain alignment:")
        print(f"    Centroid dist:       {domain['centroid_dist']:.4f}")
        print(f"    CORAL loss:          {domain['coral']:.4f}")
        print(f"    MMD loss:            {domain['mmd']:.6f}")
        print(f"    Norm gap:            {domain['norm_gap_pct']:.1f}%")
        print(f"    Domain classifier:   {domain_acc:.4f}  (0.50 = chance)")

        all_results[adapter_name] = adapter_results

    # === Summary comparison table ===
    _print_comparison_summary(all_results, k, run_patch_quality)

    # === Decision rule ===
    _print_decision(all_results, k)

    return all_results


def _print_comparison_summary(all_results: Dict, k: int, include_patches: bool = True):
    """Print side-by-side summary table."""
    print(f"\n{'=' * 80}")
    print(f"ADAPTER COMPARISON SUMMARY (k={k})")
    print(f"{'=' * 80}")

    # Row 1: domain alignment
    print(f"\n  Domain alignment:")
    print(f"  {'Adapter':<12s} | {'domain_acc':>10s} | {'centroid':>8s} | {'CORAL':>8s} | "
          f"{'MMD':>10s} | {'norm_gap':>8s}")
    print("  " + "-" * 65)

    for name, res in all_results.items():
        dm = res['domain_metrics']
        da = res.get('domain_classifier_acc', float('nan'))
        print(f"  {name:<12s} | {da:>10.4f} | {dm['centroid_dist']:>8.4f} | "
              f"{dm['coral']:>8.4f} | {dm['mmd']:>10.6f} | {dm['norm_gap_pct']:>7.1f}%")

    # Row 2: topology damage (ST only)
    print(f"\n  Topology damage (ST slides only, after - before):")
    print(f"  {'Adapter':<12s} | {'mean_d_ovl':>10s} | {'mean_d_hit':>10s} | {'mean_d_compact':>14s}")
    print("  " + "-" * 55)

    for name, res in all_results.items():
        st_topos = {k_: v for k_, v in res['topology_per_slide'].items() if not k_.endswith('_inf')}
        if st_topos:
            mean_d_ovl = np.mean([t['delta_overlap'] for t in st_topos.values()])
            mean_d_hit = np.mean([t['delta_hit'] for t in st_topos.values()])
            mean_d_compact = np.mean([
                t['compactness_ratio_after'] - t['compactness_ratio_before']
                for t in st_topos.values()
            ])
        else:
            mean_d_ovl = mean_d_hit = mean_d_compact = 0.0
        print(f"  {name:<12s} | {mean_d_ovl:>+10.4f} | {mean_d_hit:>+10.4f} | {mean_d_compact:>+14.4f}")

    # Row 3: patch quality
    if include_patches:
        print(f"\n  Patch quality (inference-realistic, per adapter):")
        print(f"  {'Adapter':<12s} | {'diam_p50':>8s} | {'diam_p90':>8s} | {'nn_ovl':>7s} | "
              f"{'components':>10s} | {'mean_deg':>8s}")
        print("  " + "-" * 65)

        for name, res in all_results.items():
            pqs = res.get('patch_quality_per_slide', {})
            # Average across ST slides
            st_pqs = {k_: v for k_, v in pqs.items() if not k_.endswith('_inf')}
            if st_pqs:
                avg_diam_p50 = np.mean([v['patch_diameter_median'] for v in st_pqs.values()])
                avg_diam_p90 = np.mean([v['patch_diameter_p90'] for v in st_pqs.values()])
                avg_nn_ovl = np.mean([v['patch_nn_overlap_mean'] for v in st_pqs.values()
                                      if not np.isnan(v['patch_nn_overlap_mean'])])
                avg_comp = np.mean([v['graph_n_components'] for v in st_pqs.values()])
                avg_deg = np.mean([v['graph_mean_degree'] for v in st_pqs.values()])
                print(f"  {name:<12s} | {avg_diam_p50:>8.1f} | {avg_diam_p90:>8.1f} | "
                      f"{avg_nn_ovl:>7.4f} | {avg_comp:>10.1f} | {avg_deg:>8.1f}")
            else:
                print(f"  {name:<12s} | {'n/a':>8s} | {'n/a':>8s} | {'n/a':>7s} | {'n/a':>10s} | {'n/a':>8s}")


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

        if status == "ACCEPT":
            # Among accepted: prefer best patch nn_overlap on ST
            st_pqs = {k_: v for k_, v in res.get('patch_quality_per_slide', {}).items()
                      if not k_.endswith('_inf')}
            if st_pqs:
                score = np.mean([v['patch_nn_overlap_mean'] for v in st_pqs.values()
                                 if not np.isnan(v['patch_nn_overlap_mean'])])
            else:
                score = -da  # fallback: lower domain_acc is better
            if score > best_score:
                best_score = score
                best_name = name

    print()
    if best_name:
        print(f"  >>> RECOMMENDED: {best_name}")
    else:
        print(f"  >>> NO ADAPTER PASSES ALL CRITERIA.")
        print(f"  >>> Consider: (a) relaxing thresholds, (b) more encoder training,")
        print(f"  >>>           (c) adding structure-preserving regularizer to MLP adapter.")

    print(f"{'=' * 80}")


# =============================================================================
# Structure-preserving MLP adapter training
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

        st_idx = torch.randperm(n_st, device=device)[:batch_size]
        sc_idx = torch.randperm(n_sc, device=device)[:batch_size]

        z_st_batch = z_st_frozen[st_idx]
        z_sc_batch = z_sc_frozen[sc_idx]

        z_sc_adapted = adapter(z_sc_batch)

        loss = torch.tensor(0.0, device=device)
        l_coral = coral_loss(z_st_batch, z_sc_adapted) if coral_weight > 0 else torch.tensor(0.0)
        l_mmd = mmd_rbf_loss(z_st_batch, z_sc_adapted) if mmd_weight > 0 else torch.tensor(0.0)
        loss = loss + coral_weight * l_coral + mmd_weight * l_mmd

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


def train_structure_preserving_adapter(
    z_st_frozen: torch.Tensor,
    z_sc_frozen: torch.Tensor,
    st_knn_k: int = 20,
    adapter_mode: str = 'mlp',
    adapter_dropout: float = 0.1,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    coral_weight: float = 10.0,
    mmd_weight: float = 10.0,
    identity_weight: float = 5.0,
    structure_weight: float = 5.0,
    device: str = 'cuda',
    seed: int = 42,
    log_every: int = 50,
) -> nn.Module:
    """
    Train MLP adapter that preserves ST kNN structure.

    L = coral_w * CORAL(z_st, g(z_sc))
      + mmd_w   * MMD(z_st, g(z_sc))
      + id_w    * ||g(z_st) - z_st||^2
      + struct_w * L_structure(z_st, g(z_st))

    L_structure penalizes changes in cosine similarities on ST kNN edges:
      For each ST kNN edge (i,j):
        L_structure += (cos(g(z_i), g(z_j)) - cos(z_i, z_j))^2
    """
    from sc_adapter import SCAdapter

    set_seed(seed)

    embed_dim = z_st_frozen.shape[1]
    n_st = z_st_frozen.shape[0]
    n_sc = z_sc_frozen.shape[0]

    # Precompute ST kNN edges
    print(f"\n  Precomputing ST kNN graph (k={st_knn_k}) for structure preservation...")
    z_st_np = z_st_frozen.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=st_knn_k + 1, algorithm='auto').fit(z_st_np)
    _, st_knn_idx = nbrs.kneighbors(z_st_np)
    st_knn_idx = torch.tensor(st_knn_idx[:, 1:], device=device)  # (n_st, k), exclude self

    # Precompute original cosine similarities on kNN edges
    z_st_norm = F.normalize(z_st_frozen, dim=1)
    # For each i, cos(z_i, z_j) for j in kNN(i)
    # Shape: (n_st, k)
    st_knn_cos_orig = torch.zeros(n_st, st_knn_k, device=device)
    for j_col in range(st_knn_k):
        neighbor_idx = st_knn_idx[:, j_col]  # (n_st,)
        st_knn_cos_orig[:, j_col] = (z_st_norm * z_st_norm[neighbor_idx]).sum(dim=1)

    print(f"  ST kNN cosine sim: mean={st_knn_cos_orig.mean():.4f}, "
          f"std={st_knn_cos_orig.std():.4f}")

    adapter = SCAdapter(embed_dim=embed_dim, mode=adapter_mode, dropout=adapter_dropout).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\n  Training structure-preserving MLP adapter")
    print(f"  Epochs: {n_epochs}, LR: {lr}")
    print(f"  CORAL: {coral_weight}, MMD: {mmd_weight}, L_id: {identity_weight}, L_struct: {structure_weight}")

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        adapter.train()

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

        # Identity regularizer
        z_st_through = adapter(z_st_batch)
        l_id = F.mse_loss(z_st_through, z_st_batch) if identity_weight > 0 else torch.tensor(0.0)
        loss = loss + identity_weight * l_id

        # Structure preservation: cosine sim on ST kNN edges
        if structure_weight > 0:
            z_st_adapted_norm = F.normalize(z_st_through, dim=1)
            z_st_full_adapted_norm = F.normalize(adapter(z_st_frozen), dim=1)

            # For the batch indices, compute cos sim with their kNN neighbors (through adapter)
            knn_of_batch = st_knn_idx[st_idx]  # (batch_size, k)
            cos_orig_batch = st_knn_cos_orig[st_idx]  # (batch_size, k)

            # Adapted cosine similarities
            cos_adapted_batch = torch.zeros_like(cos_orig_batch)
            for j_col in range(st_knn_k):
                nb_idx = knn_of_batch[:, j_col]  # (batch_size,)
                nb_adapted = z_st_full_adapted_norm[nb_idx]  # (batch_size, d)
                cos_adapted_batch[:, j_col] = (z_st_adapted_norm * nb_adapted).sum(dim=1)

            l_struct = F.mse_loss(cos_adapted_batch, cos_orig_batch)
            loss = loss + structure_weight * l_struct
        else:
            l_struct = torch.tensor(0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"    e{epoch:4d} | loss={loss.item():.4f} coral={l_coral.item():.4f} "
                  f"mmd={l_mmd.item():.4f} L_id={l_id.item():.6f} L_struct={l_struct.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k_: v.cpu().clone() for k_, v in adapter.state_dict().items()}

    if best_state is not None:
        adapter.load_state_dict(best_state)
        adapter.to(device)
    adapter.eval()

    return adapter