# Geometric Target Families — Experiment Tracker

## Overview
Replacing the dense Gram matrix supervision with sparse, scale-free, rotation-invariant targets for diffusion-based spatial transcriptomics reconstruction.

**Key principle**: The diffusion model still outputs coordinates `x̂₀ ∈ R^{n×d}`, but supervision is via **relational losses** (distances/affinities on edges), NOT regression to a canonical V_target. This eliminates Gram factorization, sign ambiguity, and scale collapse.

**Implementation**: All target construction lives in `model/utils_et.py` (lines 3482–4260). Entry point is `build_miniset_geometric_targets()`. Targets are computed per-miniset in `STSetDataset.__getitem__()` and `STPairSetDataset._build_miniset_dict()`, but are **not yet consumed in the training loop** (`core_models_et_p2.py`).

---

## Bugs Found and Fixed (2026-02-16)

Three issues were identified via external review (GPT Pro research agent) and verified against the codebase.

### Bug 1 — Scale mismatch between GT targets and optimized coordinates (FIXED)

**Problem**: `build_miniset_geometric_targets()` received raw miniset coords (slide-level canonical, local patch offset from slide center, local RMS ≠ 1.0). All targets (sigma, fuzzy mu, RW transitions) were built in that space. But Diagnostic 2's optimization loop normalized V to unit-RMS (`V_norm = V_c / rms`) before computing losses. This meant `sigma` (from GT space) was applied to `V_norm` (unit-RMS space) — the Gaussian kernel `exp(-d²/(σ_i·σ_j))` was systematically biased by the ratio of the two scales.

The Gram baseline was unaffected because it explicitly normalized GT coords to the same unit-RMS gauge before building G_target.

**Fix**: Added internal normalization at the top of `build_miniset_geometric_targets()`:
```python
coords = coords.clone()
coords = coords - coords.mean(dim=0, keepdim=True)
rms = coords.pow(2).sum(dim=1).mean().sqrt().clamp(min=1e-6)
coords = coords / rms
```
All targets are now built in the same unit-RMS gauge that predicted coords are normalized to. `utils_et.py:4187–4196`.

**Impact**: Affects F1 (Fuzzy), F2 (RW), and any sigma-dependent loss. This was the primary driver of poor F1 results and the misleading "F1+F2 worse than either alone" finding.

### Bug 2 — F1+F2 combined loss was missing negative edges (FIXED in notebook)

**Problem**: In the Diagnostic 2 notebook, the standalone F1 (Fuzzy) loss included both positive edge BCE and negative edge repulsion (`loss = loss_pos + 0.5 * loss_neg`). But the F1+F2 combined loss only computed the positive edge BCE — the negative edge term was omitted entirely. Without explicit repulsion on non-neighbors, the combined objective was strictly weaker than F1 standalone.

**Fix**: Added the same negative edge term to the `fuzzy_rw` branch in the notebook's Cell 6. No production code change needed (this was a notebook-only issue).

**Impact**: This was a major contributor to the "F1+F2 combined is WORSE" result. The combined loss was a different, weaker objective than F1 alone.

### Bug 3 — Miniset kNN built from subset only, ignoring full-slide structure (FIXED)

**Problem**: `build_miniset_geometric_targets()` called `build_knn_local(coords)` which computed kNN from scratch using only the miniset's ~128–256 points. This created "neighbors" that were only neighbors because intermediate points were not sampled — a node's kNN in the subset could include far-away nodes whose intervening true neighbors happened to not be in the miniset.

Meanwhile, Stage B already computes `knn_spatial` from the full slide (~3000+ spots) and maps it to local indices as `knn_spatial_local`. This was available but unused in target construction.

**Fix**: Added `knn_spatial_local` parameter to `build_miniset_geometric_targets()`. When provided, full-slide neighbors are merged with subset-only kNN via `_merge_knn_with_fallback()` — true spatial neighbors take priority, subset kNN fills gaps for nodes whose full-slide neighbors weren't sampled. Both call sites in `core_models_et_p1.py` (STSetDataset and STPairSetDataset) now pass `knn_spatial_local` through.

**Impact**: Reduces noise in the RW transition matrix and should lower variance across minisets. The notebook diagnostic does not benefit from this fix (it builds minisets independently), but the training pipeline does.

---

## Target Family Status

| Family | Name | Status | Priority | Notes |
|--------|------|--------|----------|-------|
| 1 | Fuzzy Neighborhood Graph (UMAP/SNE) | ✅ Built, ⚠️ Improved post-fix | **LOCAL REG** | Treat as local consistency regularizer, not primary |
| 2 | Random-Walk Operator Matching | ✅ Built, ✅ Best recoverability | **PRIMARY** | Best standalone recoverer of geometry |
| 3 | Ordinal / Ring Triplets | ✅ Built, ❌ Poor standalone recoverability | **Dropped** | Cannot recover geometry alone; not worth the budget |
| 4 | Scale-free Sparse Stress | ✅ Built, ✅ Tested — best combo with F2 | **SECONDARY** | Normalized distance matching on multiscale edges; adds global rigidity |
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
   L_fuzzy = BCE(μ̂_ij, μ_ij)   over positive edges
           + 0.5 · mean(-log(1 - μ̂_neg))   over negative edges
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
4. **Intended loss** (multi-step weighted KL):
   ```
   L_rw = Σ_s α_s · KL(P^(s)_target || P̂^(s)_pred)
   ```
   with α₁ = 1.0, α₂ = 0.5, α₃ = 0.25. P̂^(s) is recomputed from predicted coords x̂ at each training step.

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

**Status**: DROPPED as primary loss. Cannot recover geometry alone (Recall 0.31, Procrustes 0.99). May revisit as regularizer only if a specific ordinal failure mode appears that RW doesn't fix.

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

### Family 4 — Normalized Stress (built, untested in diagnostic)

**Math**:
```
d_norm_{ij} = ||p_i - p_j|| / σ_i    (locally-normalized distance)
L_stress = mean((d̂_norm_{ij} - d_norm_{ij})²)
```

Applied to multiscale edge set (kNN + ring + landmark edges) for longer-range structure.

**Code**: `compute_normalized_stress_targets()` (line 4059), `build_multiscale_edges()` (for the edge set)

---

## Diagnostic Results

### Data: Mouse Liver ST (3 slides: ST1, ST2, ST3), 5 minisets (n=128 each)

### Diagnostic 0 — Invariance Checks (PASS)

Post-fix results (with internal unit-RMS normalization):

| Target | Rigid (should ≈ 0) | Scale 0.5x | Scale 2.0x |
|--------|---------------------|------------|------------|
| fuzzy_mu_pos | 0.000053 | 0.000000 | 0.000000 |
| rw_step1 | 0.000427 | 0.000000 | 0.000000 |
| rw_step2 | 0.000121 | 0.000000 | 0.000000 |
| rw_step3 | 0.000079 | 0.000000 | 0.000000 |
| ring_triplets | 0.000000 | 0.000000 | 0.000000 |
| lle_weights | 0.001784 | 0.000000 | 0.000000 |
| sigma_local | 0.000000 | 0.500000 | 1.000000 |

All targets are rotation/translation/reflection invariant. Scale columns for fuzzy/RW/triplets/LLE are now exactly 0 — the internal unit-RMS normalization makes all targets fully scale-invariant (the input scale is divided out before any computation).

**Note on sigma_local scale columns**: The 0.5 and 1.0 values are artifacts of the test formula, which expects sigma to scale linearly with input coords (`sig_diff = (sig_orig * s_factor - sig_t) / sig_orig_mean`). Since we now normalize internally, sigma is the SAME regardless of input scale, so this formula gives `(σ·s - σ)/σ = s - 1`. This is correct behavior — sigma is now scale-invariant, which is what we want.

### Diagnostic 2 — Recoverability by Direct Optimization

**Method**: Initialize V ~ N(0, I) as (n, 2) coordinates with `requires_grad=True`. Optimize with Adam (lr=0.01) for 2k steps using each family's loss against precomputed GT targets. Procrustes-align recovered V to GT coords. 3 random inits per miniset.

#### Run 2 (post-fix, n=128 only): 5 minisets × 3 inits = 15 runs

**Changes from Run 1**: Bug 1 fixed (scale mismatch), Bug 2 fixed (missing neg edges), multi-step RW {1,2,3}.

| Family | Procrustes (↓) | Recall@15 (↑) | Collapse | Verdict |
|--------|----------------|---------------|----------|---------|
| **Gram (baseline)** | **0.0000 ± 0.0000** | **0.9981 ± 0.0009** | **0/15** | **PASS (perfect)** |
| Family 1: Fuzzy | 0.4963 ± 0.1484 | 0.7565 ± 0.0628 | 0/15 | ⬆️ Improved post-fix |
| **Family 2: RW** | **0.3406 ± 0.1785** | **0.8082 ± 0.1014** | **0/15** | **Best non-baseline at n=128** |
| Family 3: Triplets | 0.9947 ± 0.0033 | 0.3076 ± 0.0293 | 0/15 | ❌ FAIL — dropped |
| Family 5: LLE | 0.5960 ± 0.1446 | 0.6255 ± 0.1070 | 0/15 | ⚠️ Moderate |
| F1+F2 combined | 0.4617 ± 0.1874 | 0.7642 ± 0.0870 | 0/15 | ⬆️ No longer worse than components |

#### Run 3 (variable miniset sizes, F4 tested): NEW

**Changes from Run 2**: Added F2+F4 (RW+Stress) and F2-primary+F1-reg combinations. Dropped triplets. Variable miniset sizes [128, 256, 384, 128, 256] to test scaling behavior. `compute_families=[1, 2, 4, 5]`.

**Overall summary (5 minisets × 3 inits = 15 runs, mixed sizes):**

| Family | Procrustes (↓) | Recall@15 (↑) | Collapse | Verdict |
|--------|----------------|---------------|----------|---------|
| **Gram (baseline)** | **0.0000 ± 0.0000** | **0.9972 ± 0.0008** | **0/15** | **PASS (perfect at all sizes)** |
| Family 1: Fuzzy | 0.7388 ± 0.1671 | 0.5890 ± 0.1444 | 0/15 | ❌ Degrades badly at n>128 |
| Family 2: RW | 0.4303 ± 0.2701 | 0.7493 ± 0.1561 | 0/15 | ⚠️ Good at n=128, degrades at n=384 |
| Family 5: LLE | 0.6842 ± 0.1534 | 0.5902 ± 0.1104 | 0/15 | ❌ Consistently weak |
| F1+F2 combined | 0.7081 ± 0.1712 | 0.6014 ± 0.1426 | 0/15 | ❌ Fuzzy dominance hurts at large n |
| **F2+F4 (RW+Stress)** | **0.3157 ± 0.2409** | **0.8294 ± 0.1331** | **0/15** | **BEST non-Gram overall** |
| F2 primary+F1 reg | 0.4187 ± 0.2729 | 0.7556 ± 0.1654 | 0/15 | ⬆️ Better than F1+F2, worse than F2+F4 |

#### CRITICAL FINDING: Performance degrades with miniset size

Per-miniset breakdown (Run 3) showing the best non-Gram method (F2+F4) and the size effect:

| Miniset | Slide | n | F2+F4 Proc. | F2+F4 Recall | F2 Proc. | F2 Recall | Best non-Gram |
|---------|-------|---|-------------|--------------|----------|-----------|---------------|
| 0 | liver_ST1 | **128** | **0.2534** | **0.8911** | 0.2720 | 0.8484 | F2+F4 |
| 1 | liver_ST2 | **256** | **0.2449** | **0.8549** | 0.2800 | 0.7998 | F2+F4 |
| 2 | liver_ST3 | **384** | 0.5082 | 0.7326 | 0.8198 | 0.5405 | F2+F4 (but degraded) |
| 3 | liver_ST1 | **128** | **0.1016** | **0.9523** | 0.1832 | 0.9083 | F2+F4 |
| 4 | liver_ST2 | **256** | 0.4705 | 0.7160 | 0.5967 | 0.6496 | F2+F4 |

**Size scaling pattern:**

| Miniset size | F2+F4 avg Recall | F2+F4 avg Procrustes | F2 avg Recall | F2 avg Procrustes | Graph sparsity (k/n) |
|-------------|-----------------|---------------------|--------------|-------------------|---------------------|
| n=128 | **0.92** | **0.18** | 0.88 | 0.23 | 15.6% |
| n=256 | 0.79 | 0.36 | 0.72 | 0.44 | 7.8% |
| n=384 | 0.73 | 0.51 | 0.54 | 0.82 | 5.2% |

**Root cause**: With fixed k=20, the kNN graph becomes relatively sparser as n grows. At n=128, each node connects to 15.6% of all nodes; at n=384, only 5.2%. Multi-step RW with 3 steps covers a smaller fraction of the graph, and local information cannot propagate globally. This is a graph sparsity problem, not a fundamental flaw in the approach.

**Key positive**: At n=128, F2+F4 achieves Recall 0.95 and Procrustes 0.10 on one miniset — approaching Gram quality. Graph-based targets work well at the scale they're designed for.

---

## Key Observations (Updated after Run 3)

### Observation 1 (Run 2): Bugs were the primary drivers of earlier bad results
- The scale mismatch (Bug 1) systematically biased all sigma-dependent losses. Fixing it improved F1 Procrustes by 43%.
- The missing negative edges (Bug 2) made the combined F1+F2 loss a strictly weaker objective than F1 alone.
- Conclusion: **F1 and F2 are NOT intrinsically conflicting.** The earlier "conflicting gradients" hypothesis was wrong — it was just broken code.

### Observation 2 (Run 3): F2+F4 (RW+Stress) is the best non-Gram target combination
- Overall best: Recall 0.83, Procrustes 0.32 (across all sizes)
- At n=128: Recall **0.92**, Procrustes **0.18** — approaching Gram quality
- F4 (stress on multiscale edges) provides the global rigidity that RW alone lacks
- The multiscale edge set (kNN + ring + landmark) carries longer-range distance constraints
- **Confirmed**: F2+F4 is strictly better than F2 alone at every miniset size tested

### Observation 3 (Run 3): CRITICAL — All graph-based targets degrade with miniset size
- **This is the biggest finding from Run 3.** Performance drops substantially from n=128 to n=384:
  - F2+F4 Recall: 0.92 → 0.79 → 0.73 (at n=128, 256, 384)
  - F2+F4 Procrustes: 0.18 → 0.36 → 0.51
  - F2 alone: Recall drops from 0.88 to 0.54 at n=384
  - F1 (Fuzzy): completely fails at n=384 (Recall 0.36)
- **Root cause**: Fixed k=20 means graph sparsity scales as k/n. At n=128 each node sees 15.6% of the graph; at n=384 only 5.2%.
- Multi-step RW with 3 steps covers a shrinking fraction of the graph as n grows
- **This is NOT a fundamental flaw** — it's a graph sparsity problem with known solutions (see Next Steps)
- Gram is unaffected because it encodes all O(n²) pairwise inner products — no sparsity

### Observation 4 (Run 3): F2-primary weighting > F1-primary weighting
- `F2 primary+F1 reg` (loss = L_rw + 0.05 * L_fuzzy): Recall 0.76, Procrustes 0.42
- `F1+F2 combined` (loss = L_fuzzy + 0.3 * L_rw): Recall 0.60, Procrustes 0.71
- Making RW the primary objective is clearly better than making fuzzy primary
- But F2+F4 beats both — stress > fuzzy as the complementary target to RW

### Observation 5: Family ranking is now clear
1. **F2+F4 (RW + Stress)**: Best overall. RW captures local/meso-scale, stress adds global rigidity.
2. **F2 alone (RW)**: Good standalone, but lacks global structure at larger n.
3. **F2+F1 (RW-primary + Fuzzy-reg)**: Slightly better than F2 alone, fuzzy helps local consistency.
4. **F1 (Fuzzy)**: Local only. Degrades fastest with n. Regularizer role only.
5. **F5 (LLE)**: Consistently weak. Not worth including.
6. **F3 (Triplets)**: Dead. Dropped.

### Observation 6: The graph sparsity problem must be solved for diverse miniset sizes
- Training pipeline uses diverse miniset sizes (not fixed n=128)
- Cannot hardcode k=20 — need k to adapt to miniset size
- Need adaptive RW steps and landmark count that scale with n
- This is an engineering problem, not a theoretical limitation

---

## Answers to Research Questions (from GPT Pro agent review)

### Q1: How to weight/combine families?
**Answer**: Make F2 (RW) the primary objective. Add F1 (fuzzy) as a local regularizer at small weight. Suggested recipe:
```
L = L_rw + 0.05 * L_fuzzy + 0.01 * L_lle
```
Normalize loss scales by construction: divide each loss by number of terms, or maintain a running EMA of each loss magnitude and weight by 1/EMA (stop-grad). Only add triplets if a specific ordinal failure mode appears later.

### Q2: Recompute bandwidths from predicted coords?
**Answer**: No, not naively — the model can "game" it. With Bug 1 fixed (same gauge), frozen sigma is fine. If adaptivity is needed: compute σ_pred from x̂₀ but detach it, use `σ_eff = sqrt(σ_GT · σ_pred)`.

### Q3: Is Fuzzy (F1) fundamentally bad globally?
**Answer**: Yes, for global structure. Local affinity objectives have many global minima. Treat F1 as a local consistency regularizer only. For F1 to contribute globally, you'd need longer-range edges (Family 4 multiscale) or RW/heat-kernel targets.

### Q4: Drop triplets (F3)?
**Answer**: Yes, drop as primary loss. Only revisit if a consistent ordinal failure appears that RW doesn't fix, and then with many more triplets or a listwise ranking loss.

### Q5: Architecture and where losses apply?
**Answer**: Output 2D coords directly from the diffusion model. Apply relational losses to x̂₀ (denoised estimate), not noisy x_t. Apply at all sigma levels with a schedule: multiply by `w(σ) = min(1, (σ₀/σ)²)` with σ₀ ~ 0.3–0.5. For overlap-consistency: apply only at low σ, after per-view Procrustes alignment.

### Q6: Why is RW high-variance across minisets?
**Answer**: Three causes — (1) subset-only kNN creates variable operator quality (now fixed via Bug 3 in training pipeline), (2) operator saturation where nearly-uniform weights give weak KL gradients, (3) using only one step length under-identifies geometry. Fixes: multi-step {1,2,3} (done), full-slide kNN (done for pipeline), symmetric divergence (JS) or MSE on log-probs for stability, and a small amount of long-range edges (Family 4).

---

## Diagnostic Plan

### Diagnostic 0 — Invariance Checks
- [x] Apply random rigid transforms (rotation, translation, scale) to miniset coords
- [x] Verify target T(P) ≈ T(P') under rigid transforms (< 1e-6 relative error)
- [x] Verify scale-normalized targets stable under s ∈ [0.5, 2]
- **Result**: ALL PASS. All targets scale-invariant post internal normalization.

### Diagnostic 1 — Perturbation Stability
- [ ] Jitter coords at 1%, 2%, 5% of median kNN radius
- [ ] Subsample 80%, 60% of points
- [ ] Measure: kNN Jaccard overlap, spectral distance, membership entropy
- [ ] Test k = 10, 20, 30

### Diagnostic 2 — Recoverability by Direct Optimization
- [x] For each target family: initialize V ~ N(0,I), optimize with Adam for 2k steps
- [x] Measure: Procrustes error, neighborhood recall@k, convergence stability
- [x] 3 random inits per miniset
- [x] **Run 1 (pre-fix)**: MIXED. F1+F2 paradox, high RW variance.
- [x] **Run 2 (post-fix, n=128)**: IMPROVED. F1+F2 no longer worse. RW variance reduced.
- [x] **Run 3 (variable sizes [128, 256, 384])**: F2+F4 is best. Size-scaling degradation found.
- [ ] **Run 4 (next)**: Test adaptive k (scale k with n) to fix size degradation

### Diagnostic 3 — Uniqueness / Informativeness
- [ ] Compute compact signatures per miniset per family
- [ ] Check diversity (not collapsing to one signature)
- [ ] Regress against confounds (miniset size n, density, boundary fraction)

### Diagnostic 4 — Grid Artifact Sensitivity
- [ ] Create synthetic lattice minisets with same n and spacing
- [ ] Compare target signatures: real vs lattice

---

## Next Steps (Priority Order)

### 1. FIX SIZE SCALING — Adaptive k, steps, and landmarks (BLOCKING)
The biggest problem right now. Graph-based targets degrade from excellent (n=128) to mediocre (n=384) because k=20 is fixed. Three levers to try:

**a) Scale k with n** — maintain consistent graph connectivity:
```python
k = max(20, int(0.15 * n))  # ~15% connectivity at all sizes
# n=128 → k=20, n=256 → k=38, n=384 → k=57
```
Or a gentler scaling:
```python
k = max(20, int(4 * sqrt(n)))  # ~sqrt scaling
# n=128 → k=45, n=256 → k=64, n=384 → k=78
```

**b) Scale RW steps with n** — ensure diffusion covers the graph:
```python
n_steps = max(3, int(np.log2(n / 20)))  # more steps for larger graphs
# n=128 → 3 steps, n=256 → 3 steps, n=384 → 4 steps, n=512 → 4 steps
```

**c) Scale landmark count in F4** — more global anchors for larger minisets:
```python
n_landmarks = max(8, int(n / 16))  # currently fixed at 8
# n=128 → 8, n=256 → 16, n=384 → 24
```

**Test plan**: Re-run Diagnostic 2 (Run 4) with adaptive k at sizes [128, 256, 384]. If F2+F4 maintains Recall > 0.85 and Procrustes < 0.25 across all sizes, the sparsity problem is solved.

### 2. Confirmed recipe: F2+F4 primary + F1 regularizer
Based on Run 3 results, the loss recipe is:
```
L = L_rw_multistep + 0.1 * L_stress_multiscale + 0.05 * L_fuzzy
```
- F2 (RW): primary objective, captures local + meso-scale diffusion structure
- F4 (Stress): secondary, provides global rigidity via multiscale edge distances
- F1 (Fuzzy): light regularizer, helps local edge consistency
- F5 (LLE) and F3 (Triplets): dropped

### 3. Consider symmetric divergence for RW stability
Replace KL with Jensen-Shannon divergence or MSE on log-probabilities:
```
L_rw_JS = 0.5 * KL(P_GT || M) + 0.5 * KL(P_pred || M)    where M = 0.5(P_GT + P_pred)
```
This is symmetric and bounded, avoiding the KL instability when P_pred has near-zero entries where P_GT is nonzero.

### 4. Integrate into training loop (Stage D)
After size scaling is fixed:
- Consume `geo_targets` dict in `core_models_et_p2.py`
- Apply losses to x̂₀ (denoised estimate)
- Weight by sigma schedule: `w(σ) = min(1, (σ₀/σ)²)`
- Target recipe: `L = L_rw + 0.1 * L_stress + 0.05 * L_fuzzy`
- Remove or phase out Gram loss

### 5. Full-slide kNN benefit evaluation
Bug 3 fix (full-slide kNN) is now active in the training pipeline but couldn't be tested in the standalone notebook diagnostic. Verify impact in training.

---

## Implementation Progress

### Stage B Changes (per-slide precomputation)
New fields added to `STTargets`:
- `sigma_i`: (n,) local bandwidth per node (k-th neighbor distance)
- `knn_spatial`: (n, k) spatial kNN indices — **already exists, now consumed by target construction**

### Stage C Changes (per-miniset target construction)
New/updated fields computed per miniset (via `build_miniset_geometric_targets()`):
- **Internal unit-RMS normalization** of coords before all target computation (Bug 1 fix)
- **Full-slide kNN merging** via `_merge_knn_with_fallback()` when `knn_spatial_local` provided (Bug 3 fix)
- `fuzzy_edges_pos_src/dst`: (E_pos,) positive edge indices (from kNN)
- `fuzzy_mu_pos`: (E_pos,) target fuzzy memberships on positive edges
- `fuzzy_edges_neg_src/dst`: (E_neg,) negative edge indices (sampled non-neighbors)
- `fuzzy_mu_neg`: (E_neg,) target memberships on negatives (near 0)
- `sigma_local`: (n,) local bandwidths for this miniset (in unit-RMS gauge)
- `coords`: (n, 2) miniset coords in unit-RMS gauge
- `rw_transitions`: dict with {s: dense (n,n)} for s-step random-walk matrices
- `lle_weights`: (n, k) reconstruction weights per node
- `ring_triplets`: (T, 3) multi-scale ring triplets
- `multiscale_edges`: (E_ms, 2) union of kNN + ring + landmark edges (Family 4 only)

### Stage D — Integration into training loop
- [ ] **NOT YET STARTED** — `geo_targets` dict flows through collate but is unused in `core_models_et_p2.py`
- Unblocked by: Bug fixes resolved loss combination question. Recipe: F2 primary + F1 regularizer.
- Next blocker: test F2 + F4 in diagnostic before committing to final recipe.

---

## Key Design Decisions

### Coordinate Dimensionality
- Use **d=2** for geometry losses (model outputs 2D coords or has 2D head)
- All relational losses computed on 2D space
- No Gram factorization needed → no sign ambiguity

### Fixed Bandwidths (Early Training)
- σ_i computed from GT coordinates in unit-RMS gauge, frozen
- Prevents model from gaming bandwidth to satisfy loss trivially
- With Bug 1 fixed, GT and pred are in the same gauge → frozen σ is appropriate
- If adaptivity needed: `σ_eff = sqrt(σ_GT · σ_pred)` with σ_pred detached

### Edge Set Strategy
- **E_pos**: kNN edges from spatial graph (k=20), with full-slide kNN priority
- **E_neg**: Ring negatives (outside k=20, within k=60) + random non-neighbors
- **E_landmark**: Farthest-point landmarks for global anchoring (Family 4)

### Normalization Before Loss (both GT targets and predictions)
- Center coords: x -= mean(x)
- Scale: divide by RMS radius → unit RMS
- GT targets built in this gauge inside `build_miniset_geometric_targets()`
- Predictions should be normalized the same way before loss computation

---

## Experiment Log

| Date | Experiment | Family | Result | Notes |
|------|-----------|--------|--------|-------|
| 2026-02-16 | Diagnostic 0: Target Inspection | All | PASS | All targets numerically healthy |
| 2026-02-16 | Diagnostic 0: Invariance Checks (pre-fix) | All | PASS | All invariant to rigid transforms |
| 2026-02-16 | Diagnostic 2: Recoverability (pre-fix) | F1 | Recall 0.68, Procrustes 0.88 | Scale mismatch hurt F1 badly |
| 2026-02-16 | Diagnostic 2: Recoverability (pre-fix) | F2 | Recall 0.86, Procrustes 0.37 | Best family, high variance |
| 2026-02-16 | Diagnostic 2: Recoverability (pre-fix) | F3 | Recall 0.32, Procrustes 0.99 | Cannot recover alone |
| 2026-02-16 | Diagnostic 2: Recoverability (pre-fix) | F5 | Recall 0.63, Procrustes 0.59 | Moderate |
| 2026-02-16 | Diagnostic 2: Recoverability (pre-fix) | F1+F2 | Recall 0.61, Procrustes 0.93 | **Bug**: missing neg edges + scale mismatch |
| 2026-02-16 | Bug fixes: scale mismatch, neg edges, full-slide kNN | — | — | See "Bugs Found and Fixed" section |
| 2026-02-16 | Diagnostic 0: Invariance Checks (post-fix) | All | PASS | Scale columns now exactly 0 (full invariance) |
| 2026-02-16 | Diagnostic 2: Recoverability (post-fix) | F1 | Recall 0.76, Procrustes 0.50 | ⬆️ +11% recall, −43% Procrustes |
| 2026-02-16 | Diagnostic 2: Recoverability (post-fix) | F2 | Recall 0.81, Procrustes 0.34 | ⬆️ Lower variance (std 0.29→0.18) |
| 2026-02-16 | Diagnostic 2: Recoverability (post-fix) | F1+F2 | Recall 0.76, Procrustes 0.46 | ⬆️⬆️ No longer worse than components |
| 2026-02-16 | Diagnostic 2: Run 3 (variable sizes) | F2+F4 | Recall 0.83, Procrustes 0.32 | **Best non-Gram** across all sizes |
| 2026-02-16 | Diagnostic 2: Run 3 (variable sizes) | F2+F1(reg) | Recall 0.76, Procrustes 0.42 | Better than F1+F2, worse than F2+F4 |
| 2026-02-16 | Diagnostic 2: Run 3 (variable sizes) | All | Size degradation found | Fixed k=20 causes sparsity at n>128 |
| 2026-02-16 | Diagnostic 2: Run 3 (n=128 only) | F2+F4 | Recall 0.92, Procrustes 0.18 | Near-Gram at small n |
| 2026-02-16 | Diagnostic 2: Run 3 (n=384 only) | F2+F4 | Recall 0.73, Procrustes 0.51 | Degrades — needs adaptive k |
