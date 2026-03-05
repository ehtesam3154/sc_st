# Augmentation Validation Experiments — Full Results

## Context

Based on your suggestion, we ran a 4-part offline validation suite to evaluate augmentation strategies for increasing training diversity in IPA-Lite without generating new minisets. The current setup has 3000 minisets from 2 slides (ST1+ST2), with ~2656 unique spots. The model memorizes training data. The canonical recipe (P2_aug) uses fixed noise σ=0.3 and dropout=0.1 on H, with standard cosine kNN candidate graph at k=max(20, ⌊0.15n⌋).

The three augmentation families tested were:
1. **QC-style randomization**: Per-batch randomized noise σ ~ U(0, 0.5) and dropout ~ U(0, 0.2), instead of fixed values.
2. **Candidate graph randomization**: Jittering the k parameter, applying mutual-kNN filtering with probability 0.5.
3. **H-space style jitter**: Low-rank additive shifts in a slide-style subspace estimated from per-slide embedding means + top PCs.

All tests were run offline on the existing frozen encoder embeddings (real_H_cache), precomputed targets (T_ref_cache), and spatial ground truth (edge_masks). No training was performed — these are purely diagnostic.

---

## Part 1: Baseline Characterization of Existing 3000 Minisets

### What we tested
We analyzed the existing 3000 training minisets + 150 validation minisets to understand the current diversity landscape: how uniformly spots are sampled, how much minisets overlap, and how varied the operator structure and embedding statistics are across minisets.

### Results

**Spot reuse (per slide):**

| Slide | Total spots | Unique used | Coverage | Mean count | ESS | ESS ratio | Gini |
|-------|-------------|-------------|----------|------------|-----|-----------|------|
| 0 (ST1) | 1293 | 1293 | 100.0% | 390.86 | 1259 | 0.974 | 0.093 |
| 1 (ST2) | 1363 | 1363 | 100.0% | 372.47 | 1325 | 0.972 | 0.097 |
| Combined | — | 1363 | — | — | 1327 | 0.973 | 0.087 |

Every single spot on both slides appears in at least one miniset. Usage is remarkably uniform (Gini ~0.09). Each spot appears ~370–390 times across 3000 minisets. The sampler has fully exhausted both slides — there is no remaining spatial diversity to extract by changing the sampling strategy.

**Miniset overlap (Jaccard, 2000 random pairs):**

| Comparison | Mean Jaccard | Median | Max | % > 0.5 |
|------------|-------------|--------|-----|---------|
| Within slide 0 | 0.1447 | 0.1395 | 0.3563 | 0.0% |
| Within slide 1 | 0.1312 | 0.1287 | 0.3649 | 0.0% |
| Cross-slide | 0.1284 | 0.1208 | 0.3032 | — |

Overlap is moderate and matches the theoretical expectation for uniform random sampling of ~336 spots from ~1300 total (expected Jaccard ≈ 0.15). No pair exceeds 0.5. Note: cross-slide Jaccard is measuring coincidental index overlap (different physical spots), not real overlap.

**Operator spectrum (T_ref^(1), 500 minisets):**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| λ₂ (2nd eigenvalue) | 0.9522 | 0.0092 | 0.9055 | 0.9697 |
| Mixing time τ = 1/(1-λ₂) | 21.52 | 3.17 | 10.58 | 33.02 |
| Effective rank | 140.64 | 41.25 | 57.71 | 201.93 |
| Miniset size | 335.7 | — | 96 | 576 |

The spectral structure is extremely homogeneous across minisets: λ₂ std is only 0.0092. Every miniset has nearly identical diffusion structure. Effective rank scales almost deterministically with miniset size (clean curve, not a cloud). This means the model sees the same "type" of geometry repeatedly — there isn't a mix of tightly-clustered, elongated, sparse, and dense regions.

**Embedding summary statistics (500 minisets):**

| Metric | Mean | Std |
|--------|------|-----|
| Per-miniset embedding variance | 0.5656 | 0.0508 |
| Avg pairwise L2 distance | 11.7995 | 0.5810 |
| Avg kNN cosine sim (k=10) | 0.7495 | 0.0149 |

Very tight distributions — you cannot distinguish minisets by their embedding summary stats. This is further confirmation that diversity is exhausted.

**Inter-slide centroid distances:**

| Pair | L2 distance | Centroid norm |
|------|------------|---------------|
| Slide 0 ↔ 1 | 0.7421 | ||μ₀|| = 3.80, ||μ₁|| = 3.90 |
| Slide 0 ↔ 2 | 0.4693 | ||μ₂|| = 3.82 |
| Slide 1 ↔ 2 | 0.5938 | — |

Inter-slide distances are 0.47–0.74 vs centroid norms of ~3.8–3.9 (12–19% relative shift). This calibrates the scale for style jitter: perturbations of α × h_rms ≈ 0.08–0.17 would simulate "seeing a different slide."

---

## Part 2: kNN Stability Under Augmentation Strategies

### What we tested
For 50 validation minisets × 10 random draws per condition, we applied each augmentation to H (and/or the candidate graph) and measured:
- **Spatial Jaccard (Sp_J)**: overlap between kNN(H_aug) and the spatial ground-truth kNN. Higher = more spatial signal preserved.
- **Spatial Recall (Sp_R)**: fraction of spatial neighbors present in the augmented graph.
- **Orig Jaccard (Orig_J)**: overlap between kNN(H_aug) and kNN(H_original). Lower = more perturbation from baseline.

### 2A: H-augmentation results

| Condition | Sp_Jacc | Sp_Recall | Orig_Jacc | Orig_Recall | Density |
|-----------|---------|-----------|-----------|-------------|---------|
| no_aug | 0.6275±0.022 | 0.7758±0.019 | 1.0000±0.000 | 1.0000±0.000 | 0.1858 |
| fixed_aug (σ=0.3, drop=0.1) | 0.5888±0.021 | 0.7473±0.018 | 0.7694±0.014 | 0.8707±0.009 | 0.1869 |
| qc_random (σ~U(0,0.5), drop~U(0,0.2)) | 0.5899±0.031 | 0.7478±0.025 | 0.7860±0.072 | 0.8794±0.045 | 0.1868 |
| style_α=0.05 | 0.6275±0.022 | 0.7758±0.019 | 0.9924±0.003 | 0.9962±0.002 | 0.1858 |
| style_α=0.10 | 0.6275±0.023 | 0.7759±0.019 | 0.9861±0.005 | 0.9931±0.003 | 0.1859 |
| style_α=0.20 | 0.6272±0.022 | 0.7757±0.019 | 0.9733±0.008 | 0.9866±0.005 | 0.1859 |
| combined_α=0.10 | 0.5885±0.030 | 0.7468±0.024 | 0.7827±0.070 | 0.8775±0.044 | 0.1868 |

Key observations:
- **Style jitter is essentially a no-op for graph diversity.** At α=0.20, Sp_J = 0.6272 vs baseline 0.6275 — identical. Orig_J only drops to 0.97. This is because style jitter adds a uniform shift to ALL nodes; cosine kNN depends on relative directions between nodes, not absolute positions, so pairwise similarities barely change.
- **fixed_aug and qc_random are equivalent in mean effect** (Sp_J ≈ 0.589 for both). But qc_random has higher variance (std 0.031 vs 0.021), meaning the model sees a wider distribution of input conditions across training steps.
- **The spatial locality degradation from noise+dropout is modest**: 0.776 → 0.748 spatial recall (~3.6% relative drop).

### 2B: Graph augmentation results (original, non-size-adaptive)

| Condition | Sp_Jacc | Sp_Recall | Orig_Jacc | Orig_Recall | Density | Edges |
|-----------|---------|-----------|-----------|-------------|---------|-------|
| graph_standard | 0.6275±0.022 | 0.7758±0.019 | 1.0000±0.000 | 1.0000±0.000 | 0.1858 | 12708 |
| graph_k_jitter (k∈{80,100,120}) | 0.4429±0.136 | 0.9112±0.069 | 0.5466±0.224 | 0.9989±0.008 | 0.4327 | 20314 |
| graph_mutual_knn (p=0.5) | 0.6028±0.034 | 0.7041±0.083 | 0.8561±0.165 | 0.8561±0.165 | 0.1586 | 11029 |
| graph_k_jit+mutual | 0.4771±0.140 | 0.8696±0.108 | 0.5835±0.214 | 0.9518±0.088 | 0.3821 | 16872 |

Key observations:
- **Fixed k∈{80,100,120} is too aggressive.** For small minisets (n=96), k=80 means each node connects to 83% of all others — nearly a complete graph. Density jumps from 0.186 to 0.433. The enormous variance (std=0.136) comes from mixing near-complete graphs on small minisets with reasonable sparse graphs on large ones. This must be size-adaptive.
- **Mutual-kNN is well-behaved.** Density drops from 0.186 to 0.159 (prunes ~15% of edges), spatial recall drops from 0.776 to 0.704 (increases precision at cost of recall). Moderate variance from the 50% coin flip.
- **Recall vs Jaccard distinction matters**: k_jitter has recall 0.91 but Jaccard 0.44 — the denser graph trivially includes more spatial neighbors but also includes far more non-spatial junk. Jaccard is the honest metric because it penalizes false positives.

### 2C: Combined H + graph augmentation

| Condition | Sp_Jacc | Sp_Recall | Orig_Jacc | Orig_Recall |
|-----------|---------|-----------|-----------|-------------|
| baseline (no aug, std graph) | 0.6275±0.022 | 0.7758±0.019 | 1.0000±0.000 | 1.0000±0.000 |
| fixed_aug + std_graph | 0.5883±0.021 | 0.7468±0.017 | 0.7697±0.013 | 0.8708±0.009 |
| qc_random + k_jit+mutual | 0.4593±0.129 | 0.8620±0.106 | 0.5510±0.186 | 0.9382±0.083 |
| style_0.10 + k_jit+mutual | 0.4737±0.140 | 0.8746±0.104 | 0.5819±0.218 | 0.9581±0.080 |
| combined_α0.1 + k_jit+mutual | 0.4583±0.124 | 0.8573±0.113 | 0.5453±0.176 | 0.9307±0.095 |

---

## Part 3: Slide Leakage + Outlier Analysis + Corrected Graph Augmentation

### 3A: Slide leakage probe

**What we tested:** Trained a logistic regression (5-fold CV) to predict slide ID from miniset-level summary statistics (first 20 dims of mean embedding, first 20 dims of variance, pairwise distance, cosine similarity, PCA eigenvalues) under each augmentation condition.

**Results:** AUC = 1.0000 ± 0.0000 for ALL conditions. Zero reduction from any augmentation. Top features driving prediction were per-dimension variance statistics (var_15, var_3, var_8, var_11).

**Follow-up investigation (Part 3A-FIX):** We suspected this was a statistical aggregation artifact rather than an encoder flaw, because the shared encoder was specifically designed with per-slide per-gene mean centering to remove batch effects (documented slide acc dropping from 98.5% → 40.0% at the spot level for 3 slides). We ran four targeted tests:

**TEST 1 — Per-spot slide classification:**
We classified individual spot embeddings (not miniset summaries) for slide identity.
- Per-SPOT balanced accuracy: **0.6652 ± 0.0076** (2 slides, chance = 0.50)
- Per-SPOT AUC: **0.7246 ± 0.0039**

This is consistent with the encoder design document reporting 40% accuracy for 3 slides — 40% with 3 classes maps to approximately 66% with 2 classes since the task is easier with fewer classes.

**TEST 2 — AUC vs aggregation size:**
We pooled n random spots from the same slide into a single mean-embedding summary vector, then classified. This isolates the effect of averaging:

| n (spots averaged) | AUC |
|--------------------|-----|
| 1 | 0.7051 |
| 3 | 0.8231 |
| 5 | 0.8895 |
| 10 | 0.9527 |
| 20 | 0.9919 |
| 50 | 0.9994 |
| 100 | 1.0000 |
| 200 | 1.0000 |
| 300 | 1.0000 |

AUC climbs smoothly from 0.70 to 1.00 purely as a function of how many spots are averaged. At n=1 (single spot), AUC is only 0.705. By n=50, it's already 0.999. This is the Central Limit Theorem: each spot carries a tiny slide signal (66% vs 50% chance), and when you average 300 spots, noise shrinks by √300 ≈ 17× while the signal stays constant. The tiny per-spot bias becomes a high-confidence miniset-level signal.

**TEST 3 — Synthetic control:**
We created synthetic embeddings with controlled per-spot separability and measured aggregated AUC at n=300:

| Target | Actual per-spot acc | n=300 aggregated AUC |
|--------|--------------------|--------------------|
| ~50% (perfect encoder) | 0.550 | 1.0000 |
| ~55% (tiny signal) | 0.555 | 1.0000 |
| ~60% (small signal) | 0.579 | 1.0000 |
| ~70% (moderate signal) | 0.655 | 1.0000 |

Even a synthetic encoder with 55% per-spot accuracy (barely above chance) produces AUC=1.0 when aggregating 300 spots. This proves the phenomenon is universal and not specific to our encoder. The only way to achieve miniset-level AUC < 1.0 would be per-spot accuracy of exactly 50.0% — zero residual slide information whatsoever.

**TEST 4 — Per-spot feature importance:**
The per-spot classifier's coefficient structure shows:
- Max/mean |coef| ratio: **3.59×**
- Signal is diffuse across many embedding dimensions, not concentrated in a few broken dimensions
- Top dimensions: dim_38 (1.50), dim_117 (1.30), dim_102 (1.20), dim_5 (1.16)

This is consistent with the encoder design doc's H4 finding: NCE induces slide-specific covariance structure (41° principal angle divergence), creating a weak, distributed slide signal across many dimensions. This is the unavoidable cost of keeping NCE, which provides the 67% spatial overlap.

**Conclusion on slide leakage:** The miniset-level AUC=1.0 is a CLT aggregation artifact, not an encoder flaw. The encoder's per-spot slide separation is 66% (2-class), consistent with the documented 40% (3-class) from the v5 encoder design. The residual per-spot signal is the minimum achievable while retaining NCE for spatial locality. No augmentation strategy operating on frozen encoder outputs can or should fix this. The slide leakage metric is uninformative for evaluating augmentation strategies.

### 3B: Embedding outlier analysis (original, cross-slide — FLAWED)

**What we tested:** Built a reference pool from slides 0+1 (training), fitted LOF and kNN distance models, then tested augmented val minisets (slide 2) for outlier rates.

**Results:**

| Condition | LOF outlier % | kNN outlier % | Mean kNN dist |
|-----------|--------------|--------------|---------------|
| no_aug | 71.35% | 51.53% | 6.9813 |
| fixed_aug | 99.59% | 97.11% | 8.3246 |
| qc_random | 96.22% | 90.24% | 8.3012 |
| style_α=0.05 | 71.44% | 51.33% | 6.9819 |
| style_α=0.10 | 71.39% | 51.46% | 6.9828 |
| style_α=0.20 | 71.63% | 51.78% | 6.9900 |
| style_α=0.40 | 71.97% | 52.42% | 7.0084 |
| combined_α=0.10 | 95.79% | 89.25% | 8.2465 |

**Methodological flaw:** The no_aug baseline already shows 71% LOF outlier rate because the reference pool is slides 0+1 (training) and test is slide 2 (validation). The test measures cross-slide distribution shift, not augmentation damage. These results are not usable for evaluating augmentation safety. This was corrected in Part 4A.

### 3C: Corrected size-adaptive graph augmentation

**What we tested:** Replaced the fixed k∈{80,100,120} from Part 2B with size-adaptive k ranges: k = ⌊mult × n⌋ where mult is sampled from a specified range. Also tested mutual-kNN filtering applied with probability 0.5.

**Results:**

| Condition | Sp_Jacc | Sp_Recall | Orig_Jacc | Density |
|-----------|---------|-----------|-----------|---------|
| standard (k=0.15n) | 0.6275±0.022 | 0.7758±0.019 | 1.0000±0.000 | 0.1858 |
| narrow (0.12–0.18n) | 0.6190±0.019 | 0.7577±0.048 | 0.8916±0.065 | 0.1789 |
| wide (0.10–0.25n) | 0.5905±0.042 | 0.7987±0.084 | 0.7936±0.111 | 0.2119 |
| very_wide (0.08–0.30n) | 0.5637±0.058 | 0.7992±0.116 | 0.7200±0.131 | 0.2257 |
| narrow+mutual (p=0.5) | 0.5852±0.051 | 0.6736±0.103 | 0.7625±0.149 | 0.1482 |
| wide+mutual (p=0.5) | 0.5831±0.049 | 0.7214±0.120 | 0.7436±0.122 | 0.1736 |

Combined with H-augmentation:

| Condition | Sp_Jacc | Sp_Recall | Orig_Jacc | Orig_Recall |
|-----------|---------|-----------|-----------|-------------|
| baseline | 0.6275±0.022 | 0.7758±0.019 | 1.0000±0.000 | 1.0000±0.000 |
| fixed_aug + std_graph | 0.5889±0.021 | 0.7473±0.018 | 0.7697±0.013 | 0.8708±0.009 |
| qc_random + std_graph | 0.5878±0.032 | 0.7461±0.026 | 0.7792±0.070 | 0.8752±0.044 |
| qc_random + wide+mutual | 0.5498±0.054 | 0.7019±0.124 | 0.6753±0.099 | 0.7916±0.157 |
| qc_random + narrow+mutual | 0.5548±0.049 | 0.6529±0.099 | 0.6836±0.104 | 0.7382±0.140 |
| combined_α0.1 + wide+mutual | 0.5480±0.054 | 0.6927±0.127 | 0.6706±0.099 | 0.7798±0.160 |

Key observations:
- Size-adaptive k fixes the problem from Part 2B. Density stays reasonable (0.17–0.23) instead of exploding to 0.43.
- Wide (0.10–0.25n) gives meaningful perturbation (~6% Sp_J drop from baseline) with moderate variance.
- Mutual-kNN prunes aggressively (density 0.148–0.174 vs baseline 0.186), trading recall for precision.
- Wide+mutual has slightly better recall than narrow+mutual (0.721 vs 0.674) because the wider k compensates for mutual-kNN pruning.

---

## Part 4: Corrected Outlier Test + Diversity Analysis + Gate Compatibility

### 4A: Corrected same-slide outlier test

**What we tested:** Fixed the Part 3B flaw by building per-slide reference pools from clean H (slide 0 reference tests slide 0 augmented, slide 1 reference tests slide 1 augmented). 50 training minisets × 5 draws.

**Reference pool statistics:**
- Slide 0: 10000 embeddings, kNN-10 p95 = 6.7017
- Slide 1: 10000 embeddings, kNN-10 p95 = 6.5450
- Clean baseline mean kNN-10 distance: 3.9643

**Results:**

| Condition | LOF outlier % | kNN outlier % | Mean kNN dist | Δ dist vs clean |
|-----------|--------------|--------------|---------------|-----------------|
| no_aug | 1.60% ± 0.6 | 6.04% ± 2.9 | 3.9643 | +0.0000 |
| fixed_aug | 67.63% ± 4.5 | 53.80% ± 7.3 | 6.5027 | +2.5384 |
| qc_random | 57.79% ± 31.9 | 50.74% ± 23.2 | 6.4142 | +2.4499 |
| style_α=0.10 | 1.53% ± 0.6 | 6.20% ± 2.9 | 4.0340 | +0.0697 |
| style_α=0.20 | 1.62% ± 0.7 | 6.28% ± 3.0 | 4.1081 | +0.1438 |
| combined_α=0.10 | 54.60% ± 33.3 | 48.45% ± 24.3 | 6.3286 | +2.3643 |

Key observations:
- With same-slide reference, clean baseline is healthy: 1.6% LOF, 6% kNN outlier. This confirms Part 3B was measuring cross-slide shift, not augmentation damage.
- fixed_aug and qc_random push 58–68% of nodes into outlier territory, with kNN distances jumping from 3.96 to ~6.4 (+2.5 shift). This is a large displacement from the clean manifold. However, the current P2_aug recipe already uses exactly this level of noise (σ=0.3, dropout=0.1) and it is the best-performing configuration. The model has learned to handle this level of perturbation.
- Style jitter barely moves the needle: 1.5% LOF, +0.07 kNN distance shift at α=0.10. Confirms Part 2's finding that style jitter is too weak to matter.

### 4B: Effective diversity — do repeated augmentation draws look different?

**What we tested:** For 30 val minisets × 20 augmented draws each, we measured how different the resulting kNN graphs and summary statistics are between different draws of the same miniset. This directly answers: "does the model see something genuinely different each epoch when it revisits the same miniset?"

**Metrics:**
- **Graph_J_across**: Jaccard between two random augmented draws of the same miniset. 1.0 = identical graphs every draw (no diversity). Lower = more diversity.
- **Edge_flip_%**: Fraction of all possible edges that differ between two draws.
- **Density_std**: How much graph sparsity varies across draws.

**Results:**

| Condition | Graph_J_across | μ_shift_std | cos_sim_std | density_std | edge_flip_% |
|-----------|---------------|-------------|-------------|-------------|-------------|
| fixed_aug + std_graph | 0.6896±0.016 | 0.0173 | 0.00232 | 0.00079 | 6.89% |
| qc_random + std_graph | 0.6941±0.060 | 0.0769 | 0.06117 | 0.00092 | 6.81% |
| qc_random + wide_mutual | 0.5856±0.106 | 0.0755 | 0.05950 | 0.05454 | 9.19% |
| combined_α0.1 + wide_mutual | 0.5971±0.105 | 0.0757 | 0.06202 | 0.05267 | 9.04% |

Key observations:
- **fixed_aug + std_graph** gives graph Jaccard 0.690 — meaning ~31% of graph structure differs between two visits to the same miniset. But density_std is 0.00079 — essentially zero. The topology varies (noise changes cosine similarities near kNN boundaries) but overall sparsity is constant.
- **qc_random + std_graph** is nearly identical in graph diversity (0.694 Jaccard) despite the randomized noise schedule. The randomization doesn't meaningfully increase graph diversity when the graph is built from fixed-k cosine kNN. The higher std (0.060 vs 0.016) reflects draw-to-draw noise schedule variation.
- **qc_random + wide_mutual** is the clear diversity winner: graph Jaccard drops to 0.586 (vs 0.694), edge flip rate jumps from 6.8% to 9.2%, and critically density_std is 0.0545 — graph sparsity itself varies across draws. Some draws are dense (k ≈ 0.25n), others sparse (k ≈ 0.10n), and the mutual-kNN coin flip adds another layer.
- **Adding style jitter (combined_α0.1)** gives slightly worse diversity (0.597 vs 0.586). Confirms style jitter contributes nothing to graph diversity.

### 4C: Gate compatibility under augmentation

**What we tested:** Pretrained a fresh EdgeGateSoft (200 steps on clean H) and evaluated it on augmented inputs to check whether the gate can still identify spatial neighbors under augmentation. This tests whether augmentation breaks the gate's ability to function.

**Metric:** Precision@30 — for each node, do the top-30 gate predictions overlap with the spatial ground-truth neighbors?

**Results:**

| Condition | Gate KL | Prec@30 | Density |
|-----------|---------|---------|---------|
| clean H, std graph | nan | 0.8618 | 0.1857 |
| fixed_aug, std graph | nan | 0.8290 | 0.1866 |
| qc_random, std graph | nan | 0.8286 | 0.1866 |
| qc_random, wide+mutual | nan | 0.8157 | 0.1724 |
| combined_α0.1, wide+mutual | nan | 0.8123 | 0.1672 |

(Gate KL is nan because val minisets are from slide 2 and don't have T_ref_cache entries; this is harmless.)

**Oracle energy floor:** 0.046305 ± 0.002251 (target coordinates through the loss pipeline — independent of H, confirms targets are self-consistent).

Key observations:
- Gate precision is remarkably stable. Clean H gives 0.862, the most aggressive augmentation (combined_α0.1 + wide+mutual) only drops to 0.812 — a 5.8% relative decline. The gate can still identify spatial neighbors under all tested augmentation conditions.
- Density shifts modestly (0.186 → 0.167) under wide+mutual, which is expected and well-behaved.

---

## Combined Decision Table (All Evidence)

| Strategy | Sp_J | Outlier % | Diversity (Graph_J) | Gate P@30 |
|----------|------|-----------|-------------------|-----------|
| Current (fixed_aug, std graph) | 0.5888 | 67.6% | 0.6896 | 0.8290 |
| qc_random + std_graph | 0.5899 | 57.8% | 0.6941 | 0.8286 |
| qc_random + wide+mutual | 0.5498 | 57.8% | 0.5856 | 0.8157 |
| combined_α0.1 + wide+mutual | 0.5480 | 54.6% | 0.5971 | 0.8123 |

Note: Slide leakage (AUC) removed from table — proven to be a CLT aggregation artifact, uninformative for augmentation evaluation (see Part 3A-FIX).

Note: Outlier rates (57–68%) look high but are expected and acceptable — the current best-performing P2_aug recipe already operates at 67.6% and the model handles it. These measure displacement from the clean manifold, not training failure.

---

## Questions for Research Agent

1. **On QC-random vs fixed augmentation:** Parts 2A and 4B show that qc_random and fixed_aug give nearly identical mean spatial Jaccard (~0.589) and graph diversity (0.694 vs 0.690). The only difference is that qc_random has higher draw-to-draw variance in summary statistics (μ_shift_std 0.077 vs 0.017, cos_sim_std 0.061 vs 0.002). Is this higher input-level variance sufficient to reduce memorization, even though the graph structure is essentially the same? Or do we need graph-level diversity (wide+mutual) to see actual training improvements?

2. **On the outlier rate:** With same-slide reference, fixed_aug and qc_random push 58–68% of node embeddings into the outlier region (kNN distance jumps from 3.96 to ~6.4). The current P2_aug recipe already uses this level of noise and it's the best-performing configuration. But does this mean the model is learning to be robust to noise, or is it learning to ignore the input embeddings entirely and relying mostly on the graph structure? If the latter, augmenting the graph (wide+mutual) might actually be counterproductive because it removes the one stable signal source.

3. **On graph diversity vs stability trade-off:** wide+mutual gives the best diversity (Graph_J=0.586, edge_flip=9.2%) but also the highest variance (std=0.106) and lowest spatial Jaccard (0.550). Gate precision only drops to 0.816, which seems safe. But during actual training with gradient updates (not just offline evaluation), could this level of graph instability cause optimization issues (noisy gradients, slow convergence)? Would a more conservative option like narrow+mutual (Graph_J not measured but expected ~0.65, lower variance) be a safer first experiment?

4. **On style jitter — is there any use case?** All our tests show style jitter has zero effect on graph diversity, zero effect on spatial Jaccard, zero effect on slide leakage, and negligible outlier displacement (+0.07 kNN distance). The only pathway it could help is through the gate MLP, which sees absolute embedding values. But we didn't test whether the gate's behavior meaningfully changes with style jitter at training time (our gate compatibility test used a gate pretrained on clean H). Is there a scenario where style jitter helps the gate generalize, even though it doesn't help the graph?

5. **On the fundamental diversity bottleneck from Part 1:** The existing 3000 minisets already have 100% spot coverage, ESS ratio 0.97, and homogeneous spectral structure (λ₂ std = 0.009). Augmentation can make each visit to a miniset look different (per-epoch variation), but it cannot create new spatial neighborhoods or new operator structures — those are fixed by the slide geometry. Given this, is there a ceiling on how much augmentation alone can help? At what point should we invest in adding a third slide (e.g., ST3 from the encoder training set) to the miniset pool instead?

6. **Concrete implementation question:** If we proceed with qc_random + wide+mutual as the augmentation recipe, the training loop needs to rebuild the candidate graph from H_aug on every forward pass (since k is randomized and mutual-kNN is stochastic). Currently the candidate graph is precomputed and cached (candidate_masks_std). Rebuilding cosine kNN at every step for minisets of size 96–576 adds compute. Is this acceptable, or should we precompute multiple candidate graph variants per miniset and sample from them?
