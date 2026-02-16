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
**Status**: COMPLETE

### Experiment D2: MDS Recovery Test
**Question**: Can we recover 2D coordinates from D_target alone (without G)?
**Method**: Classical MDS on geodesic D_subset → Procrustes-align to y_hat_subset → measure residual
**Status**: COMPLETE

### Experiment D3: Stability Under Subsampling
**Question**: How stable are the targets when we drop 20% of points from a miniset?
**Method**: Compute targets on full miniset and 80% subset, compare Gram correlation + kNN overlap
**Status**: COMPLETE

### Experiment D4: Scale Sensitivity
**Question**: How do targets change under coordinate scaling?
**Method**: Scale y_hat by [0.5, 1.0, 2.0, 5.0], recompute G/V/D
**Status**: COMPLETE

### Experiment D5: Geodesic vs Euclidean Comparison
**Question**: How much do geodesic and Euclidean targets differ for typical minisets?
**Method**: Compute both for same minisets, compare off-diagonal correlation + rank
**Status**: COMPLETE

---

## D1-D5 Results

### Full-Slide Stats (Cell 0)
| Slide | Spots | D_euc range | D_geo range | G_euc rank | G_geo rank |
|-------|-------|-------------|-------------|------------|------------|
| ST1 | 1293 | [0.0000, 3.3180] | [0.0605, 2.0490] | 278 | 631 |
| ST2 | 1363 | [0.0000, 3.3750] | [0.0604, 2.2142] | 293 | 659 |
| ST3 | 1316 | [0.0000, 3.3468] | [0.0606, 2.0954] | 298 | 640 |

Note: Full-slide Euclidean Gram has rank >2 because coordinates are 2D but centering is per-slide (not per-miniset). Miniset-level Euclidean Gram is always exactly rank 2.

### Miniset-Level Results (Cells 1+3, 20 minisets)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| **Euc Gram rank** | **2.0** | 0.0 | 2.0 | 2.0 |
| **Geo Gram rank** | **64.3** | 17.2 | 32.0 | 81.0 |
| Euc effective rank | 1.84 | 0.14 | 1.53 | 1.99 |
| **Geo effective rank** | **5.32** | 0.64 | 4.13 | 6.37 |
| **Euc cum var top-2** | **100.0%** | 0.0 | 100.0 | 100.0 |
| Geo cum var top-2 | 79.5% | 2.0 | 75.6 | 84.1 |
| Geo cum var top-5 | 87.3% | 1.5 | 85.1 | 89.9 |
| Geo cum var top-10 | 92.3% | 1.3 | 90.6 | 94.9 |
| **MDS rel. residual** | **0.085** | 0.025 | 0.055 | 0.167 |
| Gram corr (80% subsample) | 0.9985 | 0.002 | 0.993 | 1.000 |
| kNN overlap (80% subsample) | 1.000 | 0.0 | 1.0 | 1.0 |
| Norm Gram scale-inv err | 0.0000 | 0.0 | 0.0 | 0.0 |
| **Geo-Euc D corr** | **0.980** | 0.004 | 0.970 | 0.984 |

Scale sensitivity (Gram trace ratios at coordinate scale [0.5x, 1x, 2x, 5x]):
- Expected: [0.25, 1.0, 4.0, 25.0]
- Observed: [0.250, 1.000, 4.000, 25.000] (exactly quadratic)

### Cell 2 Single-Miniset Visualization
- Euclidean Gram rank: exactly 2 (as expected)
- V[:,:2] Procrustes residual: 0.000000 (perfect reconstruction — rank 2 captures everything)
- MDS from D_geo Procrustes residual: 0.088 (8.8% — geodesic D encodes non-Euclidean structure)

---

## Key Findings

### Finding 1: Euclidean Gram is always rank 2 — massive waste in V_target

The current code builds `G_subset = y_hat_centered @ y_hat_centered.T` in `STSetDataset.__getitem__` (line 2726-2727). Since `y_hat` is 2D (spatial coordinates), this Gram matrix is always exactly rank 2. `factor_from_gram(G_subset, D_latent=32)` produces V_target with shape (n, 32) where **30 of 32 columns are pure zeros**.

The diffusion model is trained to denoise a 32-dimensional target where only 2 dimensions carry any signal. This is a massive waste of model capacity and likely causes training difficulty — the model must learn to output near-zero values in 30 dimensions while predicting meaningful geometry in only 2.

### Finding 2: Geodesic Gram has meaningful higher rank (~5.3)

When using `gram_from_geodesic(D)` (double-centering of squared geodesic distances), the effective rank rises to ~5.3. Top-2 eigenvalues capture only 79.5% of variance (vs 100% for Euclidean), meaning the remaining ~20% encodes real structural information: graph topology, local density variation, tissue boundary effects, and connectivity patterns that flat 2D coordinates miss.

This is the "curvature signature" of the tissue — geodesic distances through the kNN graph capture how tissue is connected, not just where spots are in pixel space.

### Finding 3: Bug — G and D are inconsistent in the current code

In `STSetDataset.__getitem__`:
- `G_subset` uses **Euclidean** centered coordinates (rank 2) — line 2726-2727
- `D_subset` uses **geodesic** distances — line 2735

These encode different geometry. The Gram matrix (which is the source of truth for V_target and the score loss) captures flat 2D structure, while the distance matrix (used for triplets, histograms, kNN) captures graph topology. This inconsistency means different losses in training are optimizing toward different geometric targets.

### Finding 4: MDS residual of 8.5% confirms geodesic adds info

Classical MDS on geodesic D_subset cannot perfectly recover the original 2D coordinates (8.5% relative error). This proves the geodesic distance matrix encodes non-Euclidean structure — it's not redundant with the Euclidean Gram. The divergence comes from:
- Graph connectivity effects (shortest path ≠ straight line)
- Density variation (sparse regions → longer geodesic detours)
- Tissue boundary effects (paths must go through connected tissue)

### Finding 5: Scale dependence is real but fixable

Raw Gram scales quadratically with coordinate scale (trace ratios = [0.25, 1.0, 4.0, 25.0] for scale factors [0.5, 1.0, 2.0, 5.0]). However, normalized Gram G/tr(G) is **perfectly** scale-invariant (error = 0.000000). This means the relative structure is preserved — only the absolute scale changes. The fix is simple: normalize.

### Finding 6: Targets are very stable under subsampling

Gram correlation under 20% subsampling = 0.9985, kNN overlap = 1.0. Whatever geometric targets we use, they won't be noisy or sensitive to miniset boundary effects. This is good news for training stability.

### Finding 7: Geodesic and Euclidean are 98% correlated for local patches

For typical minisets (local kNN patches, diameter ~1-2 in canonical coordinates), geodesic ≈ Euclidean with 98% correlation. The 2% divergence is meaningful (it's the structural info from Finding 2) but not dramatic. Larger or more boundary-spanning patches would show larger divergence.

---

## Implications for Redesign

### Minimum fixes (no full redesign needed)
1. **Switch G_subset to geodesic Gram**: Replace `y_centered @ y_centered.T` with `gram_from_geodesic(D_subset)` in `__getitem__`. Raises effective rank from 2 to ~5.3.
2. **Reduce D_latent**: From 32 to 8-10 (captures 90-92% of geodesic Gram variance). Eliminates wasted dimensions.
3. **Normalize G by trace**: For scale invariance. Use G/tr(G) as target, or equivalently normalize V_target.

### Full redesign considerations (for new paper)
- The geodesic Gram's effective rank of 5.3 means there IS meaningful geometry beyond flat 2D
- New targets should capture this structure: diffusion kernels, Laplacian spectra, ordinal triplets
- The key question is which representation is most defensible and compatible with diffusion training
- Need to evaluate ChatGPT Pro's recommendations against these empirical findings

---

## New Target Proposals (from ChatGPT Pro)
*(to be added after receiving recommendations)*

---

## Notebook
Working notebook: `model/geometric_targets_v1.ipynb` (user-managed)
