# Geometric Target Families — Experiment Tracker

## Overview
Replacing the dense Gram matrix supervision with sparse, scale-free, rotation-invariant targets for diffusion-based spatial transcriptomics reconstruction.

**Key principle**: The diffusion model still outputs coordinates `x̂₀ ∈ R^{n×d}`, but supervision is via **relational losses** (distances/affinities on edges), NOT regression to a canonical V_target. This eliminates Gram factorization, sign ambiguity, and scale collapse.

---

## Target Family Status

| Family | Name | Status | Priority | Notes |
|--------|------|--------|----------|-------|
| 1 | Fuzzy Neighborhood Graph (UMAP/SNE) | 🔨 Implementing | **PRIMARY** | Main loss — local geometry via CE on fuzzy edges |
| 2 | Random-Walk Operator Matching | 🔨 Implementing | **SECONDARY** | Meso-scale regularizer — multi-step transitions |
| 3 | Ordinal / Ring Triplets | 🔨 Implementing | Auxiliary | Multi-scale triplets for rank geometry |
| 4 | Scale-free Sparse Stress | ⏳ Pending | Ablation | Normalized distance matching on edges |
| 5 | LLE (Local Linear Reconstruction) | 🔨 Implementing | Auxiliary | Coord-free local affine structure |
| 6 | Distributional (Histograms/Ripley) | ⏳ Existing | Ablation | Already have distance histogram; weak alone |
| 7 | Topology / Persistence | ⏳ Existing | Optional | Already have PH pairs; hard to differentiate |

---

## Diagnostic Plan (Days 1-4)

### Diagnostic 0 — Invariance Checks
- [ ] Apply random rigid transforms (rotation, translation, scale) to miniset coords
- [ ] Verify target T(P) ≈ T(P') under rigid transforms (< 1e-6 relative error)
- [ ] Verify scale-normalized targets stable under s ∈ [0.5, 2]

### Diagnostic 1 — Perturbation Stability
- [ ] Jitter coords at 1%, 2%, 5% of median kNN radius
- [ ] Subsample 80%, 60% of points
- [ ] Measure: kNN Jaccard overlap, spectral distance, membership entropy
- [ ] Test k = 10, 20, 30

### Diagnostic 2 — Recoverability by Direct Optimization
- [ ] For each target family: initialize V ~ N(0,I), optimize with Adam for 2k steps
- [ ] Measure: Procrustes error, neighborhood recall@k, convergence stability
- [ ] 5 random inits per miniset to check local minima
- [ ] Acceptance: recall@15 > 0.6, collapse rate ≈ 0

### Diagnostic 3 — Uniqueness / Informativeness
- [ ] Compute compact signatures per miniset per family
- [ ] Check diversity (not collapsing to one signature)
- [ ] Regress against confounds (miniset size n, density, boundary fraction)

### Diagnostic 4 — Grid Artifact Sensitivity
- [ ] Create synthetic lattice minisets with same n and spacing
- [ ] Compare target signatures: real vs lattice

---

## Implementation Progress

### Stage B Changes (per-slide precomputation)
New fields added to `STTargets`:
- `sigma_i`: (n,) local bandwidth per node (k-th neighbor distance)
- `knn_spatial`: (n, k) spatial kNN indices — **already exists**

### Stage C Changes (per-miniset target construction)
New fields computed per miniset:
- `fuzzy_edges_pos`: (E_pos, 2) positive edge indices (from kNN)
- `fuzzy_mu_pos`: (E_pos,) target fuzzy memberships on positive edges
- `fuzzy_edges_neg`: (E_neg, 2) negative edge indices (sampled non-neighbors)
- `fuzzy_mu_neg`: (E_neg,) target memberships on negatives (= 0)
- `sigma_local`: (n,) local bandwidths for this miniset
- `rw_transitions`: dict with {s: sparse (n,n)} for s-step random-walk matrices
- `lle_weights`: (n, k) reconstruction weights per node
- `ring_triplets`: (T, 3) multi-scale ring triplets
- `multiscale_edges`: (E_ms, 2) union of kNN + ring + landmark edges

---

## Key Design Decisions

### Coordinate Dimensionality
- Use **d=2** for geometry losses (model outputs 2D coords or has 2D head)
- All relational losses computed on 2D space
- No Gram factorization needed → no sign ambiguity

### Fixed Bandwidths (Early Training)
- σ_i computed from ST coords (ground truth), NOT from predicted coords
- Prevents model from gaming bandwidth to satisfy loss trivially
- Can switch to dynamic σ later in training

### Edge Set Strategy
- **E_pos**: kNN edges from ST spatial graph (k=20)
- **E_neg**: Ring negatives (outside k=20, within k=60) + random non-neighbors
- **E_landmark**: Farthest-point landmarks for global anchoring (optional)

### Normalization Before Loss
- Center predicted coords: x̂ -= mean(x̂)
- Scale: divide by RMS radius → unit RMS
- This prevents collapse-by-shrinking

---

## Experiment Log

| Date | Experiment | Family | Result | Notes |
|------|-----------|--------|--------|-------|
| | | | | |

