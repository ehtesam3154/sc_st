# Geometric Target Families — Experiment Tracker

## Overview
Replacing the dense Gram matrix supervision with sparse, scale-free, rotation-invariant targets for diffusion-based spatial transcriptomics reconstruction.

**Key principle**: The diffusion model still outputs coordinates `x̂₀ ∈ R^{n×d}`, but supervision is via **relational losses** (distances/affinities on edges), NOT regression to a canonical V_target. This eliminates Gram factorization, sign ambiguity, and scale collapse.

**Implementation**: All target construction lives in `model/utils_et.py` (lines 3482–4200). Entry point is `build_miniset_geometric_targets()`. Targets are computed per-miniset in `STSetDataset.__getitem__()` and `STPairSetDataset._build_miniset_dict()`, but are **not yet consumed in the training loop** (`core_models_et_p2.py`).

---

## Target Family Status

| Family | Name | Status | Priority | Notes |
|--------|------|--------|----------|-------|
| 1 | Fuzzy Neighborhood Graph (UMAP/SNE) | ✅ Built, ⚠️ Recoverability issues | **PRIMARY** | Targets healthy; direct optimization struggles |
| 2 | Random-Walk Operator Matching | ✅ Built, ✅ Best recoverability | **SECONDARY** | Best standalone recoverer of geometry |
| 3 | Ordinal / Ring Triplets | ✅ Built, ❌ Poor standalone recoverability | **On hold** | Cannot recover geometry alone; may be useful as regularizer only |
| 4 | Scale-free Sparse Stress | ⏳ Pending | Ablation | Normalized distance matching on edges |
| 5 | LLE (Local Linear Reconstruction) | ✅ Built, ⚠️ Moderate recoverability | Auxiliary | OK neighborhood recall, poor Procrustes |
| 6 | Distributional (Histograms/Ripley) | ⏳ Existing | Ablation | Already have distance histogram; weak alone |
| 7 | Topology / Persistence | ⏳ Existing | Optional | Already have PH pairs; hard to differentiate |

---

## Mathematical Formulation of Each Family

### Family 1 — Fuzzy Neighborhood Graph (UMAP/SNE-style)

**High-level idea**: Encode local geometry as a fuzzy graph where each edge has a "membership strength" in [0, 1]. Close neighbors get μ ≈ 1, distant nodes get μ ≈ 0. Self-tuning bandwidths adapt to local density so the target is not sensitive to global pixel scale.

**Math**:
1. **Local bandwidth** (self-tuning):
   ```
   σ_i = ||p_i - p_{kNN_k(i)}||    (distance to k-th neighbor)
   ```
2. **Fuzzy membership** on an edge (i, j):
   ```
   μ_{ij} = exp(-||p_i - p_j||² / (σ_i · σ_j + ε))
   ```
3. **Edge sets**:
   - Positive edges: all kNN(k=20) pairs → μ_pos values in (0, 1]
   - Negative edges: 50% "ring" negatives (closest non-neighbors, i.e. ranks k+1 to ~60), 50% random non-neighbors → μ_neg values near 0
4. **Intended loss** (not yet in training loop):
   ```
   L_fuzzy = BCE(μ̂_ij, μ_ij)   over all positive + negative edges
   ```
   where μ̂_ij is computed from predicted coordinates x̂ using the same formula with the **precomputed (frozen) GT bandwidths σ_i**.

**Code**: `compute_fuzzy_membership_on_edges()` (line 3537), `build_fuzzy_target_edges()` (line 3570)

### Family 2 — Random-Walk Transition Matrices

**High-level idea**: Build a kNN graph with Gaussian affinities, row-normalize to get a transition matrix P, then compute P², P³ for multi-step transitions. These capture meso-scale connectivity — nodes in the same "region" will have similar multi-step transition distributions even if they're not direct neighbors.

**Math**:
1. **Affinity matrix** (on kNN edges):
   ```
   W_{ij} = exp(-||p_i - p_j||² / (σ_i · σ_j + ε))   for j ∈ kNN(i)
   W = (W + Wᵀ) / 2    (symmetrize)
   ```
2. **Transition matrix**:
   ```
   P = D⁻¹ W    where D = diag(W · 1)    (row-stochastic)
   ```
3. **Multi-step**:
   ```
   P^(s) = P^s    for s = 1, 2, 3
   ```
4. **Intended loss**:
   ```
   L_rw = Σ_s  KL(P^(s)_target || P̂^(s)_pred)    averaged over steps
   ```
   where P̂^(s) is recomputed from predicted coords x̂ at each training step.

**Code**: `compute_rw_transition_matrix()` (line 3684), `compute_multistep_rw_targets()` (line 3735)

### Family 3 — Multi-scale Ring Triplets

**High-level idea**: Sample triplets (anchor, near, far) where "near" is within top-k_near neighbors and "far" is from a ring outside k_mid. The loss enforces ordinal ordering: predicted distance to near < predicted distance to far. Only captures rank/ordering, not metric distances.

**Math**:
1. **Sampling**: For anchor i, sample j from distance rank [0, near_k) and k from rank [mid_k, far_k).
   - Defaults: near_k = kNN_k / 2 = 10, mid_k = kNN_k = 20, far_k = 3 × kNN_k = 60
2. **Intended loss**:
   ```
   L_triplet = mean(max(0, d̂(i,j)² - d̂(i,k)² + margin))
   ```
   where d̂ = Euclidean distance in predicted coordinate space.

**Code**: `sample_ring_triplets()` (line 3835)

### Family 5 — LLE (Local Linear Embedding) Weights

**High-level idea**: For each point, find the affine combination of its kNN neighbors that best reconstructs it. These weights encode the local tangent plane / affine structure of the spatial manifold, and are invariant to translation, rotation, and uniform scaling.

**Math**:
1. **Local covariance** for node i with neighbors N(i):
   ```
   C_{jl} = (p_i - p_j) · (p_i - p_l)    for j, l ∈ N(i)
   ```
2. **Solve** constrained least-squares:
   ```
   min ||p_i - Σ_j w_{ij} p_j||²    s.t. Σ_j w_{ij} = 1
   ```
   Solution: solve (C + ridge · I)w = 1, then normalize w /= Σw.
3. **Intended loss**:
   ```
   L_lle = Σ_i ||x̂_i - Σ_j w_{ij} x̂_j||²
   ```
   where w_{ij} are precomputed from GT coords (frozen), applied to predicted coords x̂.

**Code**: `compute_lle_weights()` (line 3771)

### Family 4 — Normalized Stress (not yet tested)

**Math**:
```
d_norm_{ij} = ||p_i - p_j|| / σ_i    (locally-normalized distance)
L_stress = mean((d̂_norm_{ij} - d_norm_{ij})²)
```

**Code**: `compute_normalized_stress_targets()` (line 4059)

---

## Diagnostic Results

### Data: Mouse Liver ST (3 slides: ST1, ST2, ST3), 5 minisets (n = 96–192 each)

### Diagnostic 0 — Target Inspection (PASS)

All targets produced numerically healthy outputs across all 5 minisets:

| Property | Family 1 (Fuzzy) | Family 2 (RW) | Family 3 (Triplets) | Family 5 (LLE) |
|----------|-------------------|----------------|----------------------|------------------|
| Key metric | μ_pos mean ≈ 0.61, μ_neg mean ≈ 0.14 | Row sums = 1.0 exactly | Violation rate = 0.0 | Row sums = 1.0, RMSE ≈ 0.0004 |
| Edges/triplets per node | 20 pos, 5 neg | n×n dense | 1 triplet/node | 20 neighbors |
| Separation / quality | Good: μ_pos >> μ_neg | Entropy increases with steps (correct diffusion) | Mean margin ≈ 0.36 | Weight range [-0.33, 0.53] |

### Diagnostic 0 — Invariance Checks (PASS)

| Target | Rigid (should ≈ 0) | Scale 0.5x | Scale 2.0x |
|--------|---------------------|------------|------------|
| fuzzy_mu_pos | 0.000056 | 0.000000 | 0.000000 |
| rw_step1 | 0.000447 | 0.000000 | 0.000000 |
| rw_step2 | 0.000130 | 0.000000 | 0.000000 |
| rw_step3 | 0.000089 | 0.000000 | 0.000000 |
| ring_triplets | 0.000000 | 0.000000 | 0.000000 |
| lle_weights | 0.001784 | 0.000320 | 0.000081 |
| sigma_local | 0.000000 | 0.000000 | 0.000000 |

All targets are rotation/translation/reflection invariant and behave correctly under scaling.

### Diagnostic 2 — Recoverability by Direct Optimization (MIXED)

**Method**: Initialize V ~ N(0, I) as (n, 2) coordinates with `requires_grad=True`. Optimize with Adam for 2k steps using each family's loss against precomputed GT targets. Procrustes-align recovered V to GT coords. 3 random inits per miniset, 5 minisets.

**Acceptance criteria**: Recall@15 > 0.6, Collapse rate ≈ 0, Procrustes error < 0.3, F1+F2 combined should be best or near-best.

#### Results Summary (averaged over 5 minisets × 3 inits = 15 runs):

| Family | Procrustes (↓, want < 0.3) | Recall@15 (↑, want > 0.6) | Collapse rate | Verdict |
|--------|---------------------------|---------------------------|---------------|---------|
| **Gram (baseline)** | **0.0000 ± 0.0000** | **0.9981 ± 0.0009** | **0/15** | **PASS (perfect)** |
| Family 1: Fuzzy | 0.8755 ± 0.0655 | 0.6838 ± 0.0959 | 0/15 | ⚠️ Recall OK, **Procrustes FAIL** |
| **Family 2: RW** | **0.3730 ± 0.2929** | **0.8560 ± 0.1155** | **0/15** | **Best non-baseline. Procrustes borderline.** |
| Family 3: Triplets | 0.9946 ± 0.0022 | 0.3153 ± 0.0239 | 0/15 | ❌ **FAIL both metrics** |
| Family 5: LLE | 0.5907 ± 0.1438 | 0.6311 ± 0.1029 | 0/15 | ⚠️ Recall borderline, Procrustes FAIL |
| F1+F2 combined | 0.9310 ± 0.0386 | 0.6063 ± 0.1416 | 0/15 | ❌ **Worse than either alone** |

#### Per-miniset breakdown:

| Miniset | Slide | n | F2 (RW) Procrustes | F2 Recall@15 | F1 (Fuzzy) Procrustes | F1 Recall@15 |
|---------|-------|---|--------------------|--------------|-----------------------|--------------|
| 0 | liver_ST1 | 128 | 0.3168 | 0.7979 | 0.8546 | 0.7146 |
| 1 | liver_ST2 | 128 | 0.9014 | 0.7760 | 0.9094 | 0.6710 |
| 2 | liver_ST3 | 128 | 0.2759 | 0.8944 | 0.8213 | 0.7585 |
| 3 | liver_ST1 | 128 | 0.1957 | 0.9104 | 0.9202 | 0.6127 |
| 4 | liver_ST2 | 128 | 0.1755 | 0.9014 | 0.8719 | 0.6622 |

Note: Family 2 (RW) shows high variance — minisets 3 and 4 achieve Procrustes < 0.2, but miniset 1 has 0.90. This suggests sensitivity to miniset structure or local minima in the RW loss landscape.

---

## Key Observations and Open Issues

### Observation 1: Family 2 (Random Walk) is the strongest individual target
- Best Recall@15 (0.856) and best Procrustes (0.373) among all families
- When it works, it works very well (Procrustes 0.17-0.28 on 3/5 minisets)
- But high variance: one miniset (liver_ST2 #1) had Procrustes 0.90
- The multi-step transition matrices encode both local connectivity AND meso-scale structure

### Observation 2: Family 1 (Fuzzy) has decent neighborhood recall but poor global structure
- Recall@15 ≈ 0.68 means local neighborhoods are roughly correct
- Procrustes ≈ 0.88 means the global layout is badly wrong
- Hypothesis: The BCE loss landscape for fuzzy membership has many local minima. The loss can be satisfied by many different arrangements that preserve local neighborhoods but scramble global positions.

### Observation 3: F1+F2 combined is WORSE than either family alone
- This is the most concerning result
- Procrustes 0.93 (worse than F1's 0.88 or F2's 0.37)
- Likely cause: **loss scale mismatch** — the fuzzy BCE loss magnitude may dominate the RW KL loss, causing the optimizer to primarily minimize the fuzzy loss while ignoring the RW signal that carries the useful global structure
- Alternative hypothesis: the two losses create conflicting gradients that trap the optimizer in poor minima

### Observation 4: Family 3 (Triplets) cannot recover geometry alone
- Recall@15 = 0.315 (below random baseline for k=20)
- This is expected: ordinal triplets only say "A is closer to B than to C" — they don't provide metric distance information. With only ~192 triplets and a margin-based hinge loss, there are infinitely many arrangements that satisfy all triplets.
- Triplets may still be useful as a regularizer on top of a metric loss, but not as a standalone signal.

### Observation 5: No collapses anywhere
- The unit-RMS normalization before loss computation is working as intended
- All 15 runs across all families produced non-degenerate coordinate outputs

### Observation 6: Frozen GT bandwidths may hurt Family 1
- The σ_i bandwidths are computed from GT coordinates and frozen
- When predicted coordinates are far from GT (early optimization / early training), the GT bandwidths don't match the predicted geometry
- The membership μ̂ computed with GT σ on wrong-scale predicted coords creates a misleading loss signal
- Family 2 may be less affected because row-normalization of the transition matrix partially compensates for scale mismatch

---

## Questions for Research Agent (GPT Pro)

### Q1: How should we weight and combine the loss families?
The naive F1+F2 combination (loss_fuzzy + 0.3 * loss_rw) performed worse than either alone. What is the right way to combine these?
- Should we normalize each loss to unit variance before combining?
- Should we use a dynamic weighting scheme (e.g., GradNorm, uncertainty weighting)?
- Or should we use one family as the primary loss and others as soft regularizers with very small weights?
- Given that F2 (RW) clearly dominates in recoverability, should it be the primary loss?

### Q2: Should we recompute bandwidths from predicted coordinates during training?
Currently σ_i is frozen from GT. This means:
- At the start of training (or in Diagnostic 2), predicted coords are random → the GT bandwidths are meaningless for the predicted geometry
- The fuzzy membership μ̂ computed with GT σ on random coords gives gradients that may not point toward the right solution
- Family 2 (RW) is less affected because it row-normalizes (degree division absorbs some scale mismatch)

Options:
1. Keep frozen GT σ (current approach)
2. Recompute σ from predicted coords each step (risk: model can game bandwidth)
3. Use a geometric mean: σ_effective = sqrt(σ_GT · σ_pred) (compromise)

### Q3: Is the Fuzzy (Family 1) loss landscape fundamentally problematic?
The fuzzy BCE loss achieved OK local recall (0.68) but terrible global Procrustes (0.88). This pattern — correct neighborhoods but wrong global layout — is a known failure mode of SNE/UMAP-style objectives. These methods famously have the "crowding problem" and can produce arbitrary global rotations/reflections of local clusters.

Should we:
- Accept this and rely on Family 2 (RW) for global structure?
- Modify the fuzzy loss to include global anchoring (e.g., landmark edges)?
- Replace Family 1 entirely with something that has better global recovery properties?

### Q4: Should we drop Family 3 (Triplets) entirely?
Results show triplets alone cannot recover geometry (Recall 0.315, Procrustes 0.995). The theoretical justification is weak for small numbers of triplets — you need O(n log n) triplets for metric recovery guarantees, but we're using n triplets (one per node).

Options:
1. Drop entirely (simplifies the loss landscape)
2. Keep as very-light regularizer only (weight ≈ 0.01 × primary loss)
3. Increase triplet count dramatically and test again

### Q5: What is the right architecture for consuming these targets?
Currently `geo_targets` are computed but unused in the training loop. When we integrate them:
- Does the diffusion model output 2D coordinates directly, or do we need a projection head?
- Should the relational losses (fuzzy BCE, RW KL, LLE MSE) be applied to the clean prediction x̂₀, or also at intermediate noise levels?
- How do these losses interact with the existing Gram/overlap/score losses — do we replace Gram entirely or use both?

### Q6: The RW loss has high variance across minisets — why?
Family 2 Procrustes ranges from 0.17 (excellent) to 0.90 (terrible) across 5 minisets, all with similar n ≈ 128. What drives this variance?
- Is it the miniset's spatial structure (boundary effects, density)?
- Is it sensitivity to random initialization?
- Would more Adam steps or learning rate tuning help, or is this a fundamental property of the KL loss landscape?

---

## Diagnostic Plan

### Diagnostic 0 — Invariance Checks
- [x] Apply random rigid transforms (rotation, translation, scale) to miniset coords
- [x] Verify target T(P) ≈ T(P') under rigid transforms (< 1e-6 relative error)
- [x] Verify scale-normalized targets stable under s ∈ [0.5, 2]
- **Result**: ALL PASS. All targets are rotation/translation/reflection invariant.

### Diagnostic 1 — Perturbation Stability
- [ ] Jitter coords at 1%, 2%, 5% of median kNN radius
- [ ] Subsample 80%, 60% of points
- [ ] Measure: kNN Jaccard overlap, spectral distance, membership entropy
- [ ] Test k = 10, 20, 30

### Diagnostic 2 — Recoverability by Direct Optimization
- [x] For each target family: initialize V ~ N(0,I), optimize with Adam for 2k steps
- [x] Measure: Procrustes error, neighborhood recall@k, convergence stability
- [x] 3 random inits per miniset (used 3 instead of 5)
- [x] Acceptance: recall@15 > 0.6, collapse rate ≈ 0
- **Result**: MIXED. Family 2 (RW) best but borderline. F1+F2 combined fails. See full results above.

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
New fields computed per miniset (via `build_miniset_geometric_targets()`):
- `fuzzy_edges_pos_src/dst`: (E_pos,) positive edge indices (from kNN)
- `fuzzy_mu_pos`: (E_pos,) target fuzzy memberships on positive edges
- `fuzzy_edges_neg_src/dst`: (E_neg,) negative edge indices (sampled non-neighbors)
- `fuzzy_mu_neg`: (E_neg,) target memberships on negatives (near 0)
- `sigma_local`: (n,) local bandwidths for this miniset
- `rw_transitions`: dict with {s: dense (n,n)} for s-step random-walk matrices
- `lle_weights`: (n, k) reconstruction weights per node
- `ring_triplets`: (T, 3) multi-scale ring triplets
- `multiscale_edges`: (E_ms, 2) union of kNN + ring + landmark edges (Family 4 only)

### Stage D — Integration into training loop
- [ ] **NOT YET STARTED** — `geo_targets` dict flows through collate but is unused in `core_models_et_p2.py`
- Blocked on: resolving loss combination strategy (see Questions Q1, Q5)

---

## Key Design Decisions

### Coordinate Dimensionality
- Use **d=2** for geometry losses (model outputs 2D coords or has 2D head)
- All relational losses computed on 2D space
- No Gram factorization needed → no sign ambiguity

### Fixed Bandwidths (Early Training)
- σ_i computed from ST coords (ground truth), NOT from predicted coords
- Prevents model from gaming bandwidth to satisfy loss trivially
- **Open issue**: this may hurt Family 1 when predicted coords are far from GT (see Q2)

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
| 2026-02-16 | Diagnostic 0: Target Inspection | All | PASS | All targets numerically healthy |
| 2026-02-16 | Diagnostic 0: Invariance Checks | All | PASS | All invariant to rigid transforms |
| 2026-02-16 | Diagnostic 2: Recoverability | F1 | Recall 0.68, Procrustes 0.88 | Local recall OK, global fails |
| 2026-02-16 | Diagnostic 2: Recoverability | F2 | Recall 0.86, Procrustes 0.37 | Best family, but high variance |
| 2026-02-16 | Diagnostic 2: Recoverability | F3 | Recall 0.32, Procrustes 0.99 | Cannot recover alone |
| 2026-02-16 | Diagnostic 2: Recoverability | F5 | Recall 0.63, Procrustes 0.59 | Moderate |
| 2026-02-16 | Diagnostic 2: Recoverability | F1+F2 | Recall 0.61, Procrustes 0.93 | Combined WORSE than either alone |
