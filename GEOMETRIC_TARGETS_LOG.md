# Geometric Target Redesign Log

## Problem Statement

The current pipeline (KDD submission) uses **Gram matrix targets** derived from ST spatial coordinates as supervision for a diffusion-based geometry generator. The model learns to predict relative spatial arrangements of cells given only gene expression.

**Current approach:**
- For a miniset of n ST spots with coordinates P:
  - Center: Y = P - mean(P)
  - Gram target: G = Y Y^T (n x n)
  - Factor: V_target = factor_from_gram(G, D_latent) via eigendecomposition (n x D_latent)
  - The diffusion model is trained to denoise V_target
- Auxiliary targets: geodesic distance matrix D, distance histogram H, ordinal triplets, graph Laplacian, persistent homology pairs

**Why we are reconsidering:**
1. **Rank degeneracy**: ST coordinates are 2D, so gram_from_coords(Y) has rank exactly 2. Using D_latent=32 is overkill — 30 dimensions are zeros. The geodesic Gram (from double-centering D^2) may have higher effective rank, but the Euclidean Gram used in miniset construction doesn't.
2. **Pixel gauge dependence**: ST coordinates are in arbitrary pixel/slide space. Scale, orientation, and resolution vary across slides and datasets. The Gram matrix encodes absolute inner products which scale quadratically with coordinate scale — this is not intrinsic geometry.
3. **Inconsistency in current code**: G_subset in miniset sampling uses gram_from_coords (Euclidean centered Gram), but D_subset uses geodesic distances. These encode different geometry when geodesic mode is on.
4. **Need a new story for next paper**: Want more defensible, intrinsic geometric supervision that doesn't rely on trusting pixel distances as ground truth.

**Goal:** Replace or augment Gram targets with targets that are:
- Intrinsic (invariant to rigid transforms, ideally weakly sensitive to global scale)
- Stable under subsampling and miniset boundary effects
- Compatible with diffusion-based geometry generation
- More informative than rank-2 Gram matrices

---

## Current Target Pipeline (Reference)

### Full-slide precomputation (STStageBPrecomputer.precompute)
- `core_models_et_p1.py:2235`
- Coordinates arrive pre-normalized via `canonicalize_st_coords_per_slide` (per-slide center + RMS-scale to ~1.0)
- Geodesic D: kNN graph (k=15) + shortest path (`utils_et.compute_geodesic_distances`)
- Gram: `gram_from_geodesic(D)` = -0.5 H D^2 H (double-centering) OR `gram_from_coords(y_hat)` = y_hat @ y_hat.T
- Laplacian: kNN graph (k=20) with RBF weights, symmetric normalized
- Triplets: random (i,j,k) with D[i,j] + margin < D[i,k]
- Persistent homology: ripser on pairwise distances

### Per-miniset construction (STSetDataset.__getitem__)
- `core_models_et_p1.py:2454`
- Sample local patch: random center, stochastic softmax kNN sampling
- **G_subset = y_hat_centered @ y_hat_centered.T** (always Euclidean, even in geodesic mode)
- **V_target = factor_from_gram(G_subset, D_latent=32)** (eigendecompose, top-D_latent)
- D_subset = D[indices][:, indices] (geodesic submatrix)
- H_subset, triplets_subset, L_subset, topo_info also computed per miniset

### Consumption in diffusion training (core_models_et_p2.py)
- V_target re-factorized from G_target every step (PATCH 3, gauge consistency)
- Main losses: score (denoising), gram (off-diagonal relative Frobenius), gram_scale (log-trace), edge, knn_nca
- Geometry losses gated by noise level (SNR proxy)
- Overlap consistency: shape (normalized Gram match), scale (log-trace), KL on soft neighbor distributions

---

## Diagnostic Experiments

### Experiment D1: Rank Analysis of Current Targets
**Question**: What is the effective rank of G_target / V_target across minisets?
**Method**: Sample 20 minisets, compute eigenvalue spectrum, fraction of variance in top-k
**Expected**: Euclidean G has rank exactly 2; geodesic G may have slightly higher effective rank
**Status**: pending

### Experiment D2: MDS Recovery Test
**Question**: Can we recover 2D coordinates from D_target alone (without G)?
**Method**: Classical MDS on geodesic D_subset → Procrustes-align to y_hat_subset → measure residual
**Expected**: If geodesic D ≈ Euclidean D for small patches, recovery should be near-perfect
**Status**: pending

### Experiment D3: Stability Under Subsampling
**Question**: How stable are the targets when we drop 20% of points from a miniset?
**Method**: Compute targets on full miniset and 80% subset, compare Gram correlation + kNN overlap
**Expected**: Stable targets → good supervision; unstable → noisy signal
**Status**: pending

### Experiment D4: Scale Sensitivity
**Question**: How do targets change under coordinate scaling?
**Method**: Scale y_hat by [0.5, 1.0, 2.0, 5.0], recompute G/V/D
**Expected**: G scales quadratically, D scales linearly; normalized versions should be invariant
**Status**: pending

### Experiment D5: Geodesic vs Euclidean Comparison
**Question**: How much do geodesic and Euclidean targets differ for typical minisets?
**Method**: Compute both for same minisets, compare off-diagonal correlation + rank
**Expected**: For small, convex patches: nearly identical. For patches near tissue boundaries: diverge
**Status**: pending

---

## New Target Proposals (from ChatGPT Pro)
*(to be added after receiving recommendations)*

---

## Notebook
Working notebook: `model/geometric_targets_v1.ipynb` (user-managed)
