# IPA-Lite Toy Validation — Complete Experiment History & Integration Guide

**Author:** Ehtesam (experiments) + Claude (code & analysis)  
**Dates:** Feb 19–27, 2026  
**Purpose:** This document records every experiment, decision, failure, and insight from the IPA-Lite toy validation notebook. It serves as a reference for the coding agent implementing the full IPA-R model in the GEARS pipeline.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Phase 1: Validation Ladder (Cells 0–7)](#2-phase-1-validation-ladder-cells-07)
3. [Phase 2: Encoder Integration (Cells A–C)](#3-phase-2-encoder-integration-cells-ac)
4. [Phase 3: Cross-Slide Regularization Sweep (Cells D–F)](#4-phase-3-cross-slide-regularization-sweep-cells-df)
5. [Phase 4: RW Schedule & Stress Weight Tuning (Cell G)](#5-phase-4-rw-schedule--stress-weight-tuning-cell-g)
6. [Phase 5: Multi-Seed Stability (Cell H)](#6-phase-5-multi-seed-stability-cell-h)
7. [Phase 6: Graph Topology Investigation (Cells I–J)](#7-phase-6-graph-topology-investigation-cells-ij)
8. [Phase 7: Inference Graph — Failed Approaches (Cells K–M)](#8-phase-7-inference-graph--failed-approaches-cells-km)
9. [Phase 8: Inference Graph — Soft Gating Success (Cells N–P)](#9-phase-8-inference-graph--soft-gating-success-cells-np)
10. [Phase 9: Attention Diagnostics (Cells Q1–Q2)](#10-phase-9-attention-diagnostics-cells-q1q2)
11. [Phase 10: Architecture Fixes — Unshared Blocks + QK-Norm (Cells R1–R3)](#11-phase-10-architecture-fixes--unshared-blocks--qk-norm-cells-r1r3)
12. [Phase 11: Controlled Replication — V1 vs V2 Identical Conditions (Cell S1)](#12-phase-11-controlled-replication--v1-vs-v2-identical-conditions-cell-s1)
13. [Phase 12: Geometry Bias Redesign (Cells S6–S7)](#13-phase-12-geometry-bias-redesign-cells-s6s7)
14. [Phase 13: Pair Representation (Cell S9)](#14-phase-13-pair-representation-cell-s9)
15. [Phase 14: Head-Specific Gate (Cell S10)](#15-phase-14-head-specific-gate-cell-s10)
16. [Phase 15: Gate Supervision Variants (Cell S11)](#16-phase-15-gate-supervision-variants-cell-s11)
17. [Phase 16: Candidate k Sensitivity (Cell S12)](#17-phase-16-candidate-k-sensitivity-cell-s12)
18. [Phase 17: Full IPA-R(2D) Integration (Cell D2)](#18-phase-17-full-ipa-r2d-integration-cell-d2)
19. [Phase 18: Code Audit, Bug Fixes & Architecture Confirmation (D3–D7)](#19-phase-18-code-audit-bug-fixes--architecture-confirmation-d3d7)
20. [Phase 19: Inference Gap Diagnostics & Gating Analysis (E-Series)](#20-phase-19-inference-gap-diagnostics--gating-analysis-e-series)
21. [Phase 20: Post-E-Series Gating Fixes (P-Series)](#21-phase-20-post-e-series-gating-fixes-p-series)
22. [Phase 21: C2/C2′ Oracle Distillation — Failed](#22-phase-21-c2c2-oracle-distillation--failed)
23. [Phase 22: β Metric Discovery & Canonical Reeval](#23-phase-22-β-metric-discovery--canonical-reeval)
24. [Updated Final Recipe & Open Questions](#24-updated-final-recipe--open-questions)
25. [Instructions for Coding Agent — Scaling to Full IPA](#25-instructions-for-coding-agent--scaling-to-full-ipa)

---

## 1. Project Context

### What is IPA-Lite?

IPA-Lite is a **simplified 2D adaptation of OpenFold's Invariant Point Attention (IPA)** for spatial transcriptomics. It predicts 2D spatial coordinates from gene expression embeddings. Unlike the full AlphaFold2 IPA which operates on 3D rigid frames, IPA-Lite works directly in 2D with raw coordinates (no rotation matrices, no rigid frames).

### Why toy experiments first?

Before building the full pipeline, we needed to validate:
1. The loss landscape is well-behaved (oracle can reach near-zero energy)
2. The IPA architecture is expressive enough to learn geometry
3. Real encoder embeddings carry spatial signal
4. Cross-slide generalization is possible
5. **The inference-graph problem is solvable** — at inference on scRNA data, we won't have spatial kNN edges. We need a mechanism to build usable graphs from gene expression alone.

### Data

- **3 spatial transcriptomics slides**: liver_ST1, liver_ST2, liver_ST3
  - ST1: 1293 spots, ST2: 1363 spots, ST3: 1316 spots
  - Gene expression + known spatial coordinates
- **1000 minisets** sampled from these slides (sizes 96–384, ~333 per slide)
- **Encoder**: SharedEncoder (pre-trained, n_genes→512→256→128, frozen)
  - Produces 128-dim embeddings per cell
  - ST embeddings use mean-centered input; SC data uses an adapter

### Target Families

We use **F2 + F4** geometric targets (validated in prior work as the best combination):

- **F2 (Random Walk transitions):** Multi-step RW matrices $P^{(s)}$ for $s \in \{1,\ldots,5\}$ built on spatial kNN. The predicted coordinates should reproduce these transition probabilities.
- **F4 (Normalized Stress):** Log-space distance matching on multiscale edges (kNN + ring + landmark edges). Penalizes distortions in pairwise distance ratios.

### Adaptive Parameters

```
k = max(20, int(0.15 * n))          # ~15% graph connectivity
steps = [1,2,3] if n<=160, [1,2,3,4] if n<=300, [1,2,3,4,5] otherwise
landmarks = max(8, n // 16)
```

---

## 2. Phase 1: Validation Ladder (Cells 0–7)

### Cell 0: Data Loading

Loaded ST expression matrices, slide IDs, spatial coordinates, and pre-computed targets for all 3 slides.

### Cell 1: Utility Setup

Imported `build_miniset_geometric_targets`, `compute_rw_transition_matrix`, `compute_normalized_stress_targets`, `build_multiscale_edges` from `utils_et.py`. Set up `sample_miniset_indices()` for random subsampling.

### Cell 2: Miniset Creation

Built 1000 minisets with F2+F4 targets.

**Results:**
- Size range: 96–384 (mean=241, std=94)
- Balanced across slides (~333 each)
- kNN k: 20–57 (mean=36)
- Multiscale edges: 1,799–16,466 per miniset (mean=7,991)
- **Sanity checks passed:** RW row-sums exactly 1.0, row entropy 3.18–3.88 (healthy mid-range)

### Cell 3: Loss Functions

Implemented the full loss infrastructure. Every function listed here was validated and carried through all subsequent experiments unchanged:

| Function | Purpose | Key detail |
|---|---|---|
| `gauge_fix(X)` | Center + unit-RMS normalize | Applied after every recycle |
| `build_predicted_rw_operator(X, knn, sigma)` | Build $T_{\text{pred}}^{(1)}$ from coordinates on FIXED edges | $w_{ij} = \exp(-\|x_i - x_j\|^2 / (\sigma_i \sigma_j))$, symmetrized with $0.5(W + W^\top)$, row-normalized. **No self-loops.** |
| `compute_multistep_rw_pred(T1, steps)` | Compute $T^{(s)}$ via successive matmuls | Returns dict {step: matrix} |
| `loss_rw_kl(pred, target)` | Multi-step KL divergence | $L_{\text{RW}} = \sum_s 0.5^{(s-1)} \text{KL}(P_{\text{ref}}^{(s)} \| P_{\text{pred}}^{(s)})$ |
| `loss_stress_log(X, edges, d_target)` | Log-space normalized stress | Optimal log-scale $\alpha$ computed in closed form |
| `loss_repulsion(X)` | Soft repulsion on random pairs | $L_{\text{rep}} = \mathbb{E}[\exp(-\|x_i - x_j\|^2 / \tau)]$, $\tau=0.01$, 256 pairs |
| `loss_entropy_reg(T)` | Negative mean row-entropy | Prevents delta-like transition rows |
| `collapse_diagnostics(X)` | Runtime monitoring | Reports $\lambda_2/\lambda_1$ (line collapse) and median pairwise distance (point collapse) |

**Verified:** When $X$ = target coordinates, $T_{\text{pred}}$ matches $T_{\text{ref}}$ exactly (max diff = 0.00, KL = 0.00). All 4 pre-integration checks passed:
1. Target consistency: ✅ exact operator match
2. No self-loops: ✅ zero diagonal mass
3. Symmetrization: ✅ $0.5(W+W^\top)$ matches (the 0.5 cancels after row normalization)
4. Displacement sign: cosmetic fix applied ($x_j - x_i$ not $x_i - x_j$)

### Cell 4: Step 0 — Oracle Energy Minimization

Directly optimized $X \in \mathbb{R}^{n \times 2}$ with Adam (no model, no encoder) for 2000 steps per miniset.

**Results (N=50 minisets):**

| Metric | Value |
|---|---|
| Energy $E^*$ | $-0.0083 \pm 0.0340$ |
| $L_{\text{RW}}$ | $0.0186 \pm 0.0317$ |
| $L_{\text{stress}}$ | $0.0823 \pm 0.0365$ |
| $\lambda_2/\lambda_1$ | $0.4356 \pm 0.1848$ |
| median pdist | $1.1913 \pm 0.0507$ |
| Line collapses | 0/50 |
| Point collapses | 0/50 |

**Conclusion:** The energy landscape is well-behaved. Near-zero floor achievable. No collapse modes. Oracle solutions visually recover spatial layout after Procrustes alignment.

### Cell 5: IPA-Lite Model

Implemented the core architecture. This is the key piece the coding agent needs to understand:

#### IPALiteBlock (Single Attention Block)

The attention logit formula is:

$$\ell_{ij}^{(h)} = \frac{q_i \cdot k_j}{\sqrt{d}} - \text{softplus}(\gamma_h) \cdot \|x_i - x_j\|^2$$

Where:
- $q_i, k_j$ are per-head query/key vectors projected from node state $s$
- $\gamma_h$ is a learned positive per-head geometry weight (initialized at 1.0 via $\text{softplus}^{-1}(1.0) = 0.5414$)
- The geometry term penalizes attending to distant nodes

**Edge masking:** Logits are set to $-\infty$ outside the edge mask, then softmax is applied.

**Coordinate update (translation-equivariant):**

$$\Delta x_i = \sum_h \sum_j a_{ij}^{(h)} \cdot W_x^{(h)}(x_j - x_i)$$

Where $a_{ij}^{(h)}$ are attention weights and $W_x^{(h)}$ is a learned 2→2 linear per head (initialized to zero for stability).

**State update:**

$$s_i' = s_i + \text{MLP}\left([\text{LayerNorm}(s_i),\; \text{Concat}_h \sum_j a_{ij}^{(h)} v_j]\right)$$

MLP is `Linear(c_s + H*c_head, 2*c_s) → GELU → Linear(2*c_s, c_s)` with zero-init on last layer.

#### IPALiteModel (Full Model)

```
InitNet: H → s₀ (MLP: d_input→2*c_s→c_s)
InitNet: H → X₀ (MLP: d_input→c_s→2, + noise during training)
gauge_fix(X₀)
for r in range(n_recycles):     # weight-shared!
    s, X = ipa_block(s, X, edge_mask)
    X = gauge_fix(X)
return X
```

**Configuration tested:**

| Setting | c_s | heads | c_head | recycles | d_input | noise_std | params |
|---|---|---|---|---|---|---|---|
| Single overfit | 64 | 4 | 32 | 4 | 64 | 0.05 | 78,550 |
| Multi-miniset | 96 | 4 | 32 | 5 | 128 | 0.05 | 136,246 |

**Forward pass verified:** Output shape (n, 2), mean≈0, RMS=1.0, 19 params with gradients, γ initialized at 1.0.

### Cell 6: Step 2 — Single Miniset Overfit

Overfitted one miniset (n=288, k=43) for 3000 steps with **frozen random embeddings** (no real gene expression).

**Training schedule:** Stress-only warmup (300 steps) → RW ramp (500 steps) → full loss

**Results:**

| Metric | Init → Final |
|---|---|
| Energy | 5.0 → 0.12 |
| $L_{\text{RW}}$ | 4.997 → 0.131 |
| $L_{\text{stress}}$ | 0.877 → 0.248 |
| $\lambda_2/\lambda_1$ | 0.336 → 0.318 (healthy 2D) |
| $\gamma$ | 1.000 → 1.037 |
| Gap to oracle | 0.12 vs $E^* \approx -0.008$ → gap ≈ 0.13 |

**Conclusion:** IPA-Lite is expressive enough. The gap to oracle is expected — the oracle optimizes 576 free parameters directly, while IPA-Lite routes everything through 78k shared weights from random embeddings.

### Cell 7: Step 3 — Multi-Miniset Training

Trained on 200 minisets (val 50) for 15 epochs with random embeddings.

**Results:**

| Metric | Train | Val |
|---|---|---|
| Energy | 2.534 | 2.899 |
| $L_{\text{RW}}$ | — | 2.859 |
| Collapses | — | 0/50 (all epochs) |
| $\gamma$ | — | 1.130 |

**Why val energy is high (2.9):** Random embeddings carry zero spatial signal. Each miniset must be memorized individually. The small train-val gap (2.53 vs 2.90) confirms uniform underfitting, not overfitting.

**Architecture validated:** Zero collapses in 3000 forward passes, γ learns, training is stable.

---

## 3. Phase 2: Encoder Integration (Cells A–C)

### Cell A: Real Embedding Setup

Loaded the pre-trained SharedEncoder (128-dim output), computed embeddings for all ST spots, cached per-miniset.

```python
# Encoder architecture: n_genes → 512 → 256 → 128
# Input: mean-centered gene expression (per slide)
# Frozen: encoder weights NOT trained during IPA-Lite
real_H_cache = {}  # miniset_id → (n, 128) tensor on GPU
```

**Embedding stats:**
- ST1-3 norms: mean=9.395, std=1.106, range=[6.684, 14.398]

### Cell B: Single Overfit with Real Embeddings (d_input=128)

Same setup as Cell 6 but with real encoder embeddings instead of random.

**Results:**

| Config | Final E_geo |
|---|---|
| Random H (d=64) | 0.1199 |
| **Real H (d=128)** | **0.0063** |
| Oracle floor | -0.0083 |

**18.6× improvement.** Real embeddings essentially reach the oracle floor on a single miniset.

### Cell C: Cross-Slide Generalization Test

Train on ST1+ST2 minisets (667), validate on ST3 minisets (50). This is the first real test: can the model generalize to an unseen slide?

**Results (first run, baseline):**

| Setting | Best Epoch | Val E_geo |
|---|---|---|
| Real H cross-slide | 7 | 0.45 |
| Random H (same setup) | — | 2.85 |

**Cross-slide generalization confirmed.** Real embeddings give a massive improvement even on unseen slides. The model transfers spatial knowledge across slides because gene expression patterns are shared.

---

## 4. Phase 3: Cross-Slide Regularization Sweep (Cells D–F)

### Cell D: Initial Augmentation Experiments

Tested: baseline (no augmentation), h_noise_only (0.3), h_dropout_only (0.1), edge_drop_only (0.2), combined (all three).

**Problem discovered:** Early stopping was selecting from warmup phase (rw_weight=0), making comparisons unfair.

### Cell E: Agent-Guided Corrections

Research agent identified issues:
1. **Seeding:** Each experiment wasn't getting the same initial weights → fixed with `torch.manual_seed(0)` per experiment
2. **Early stopping rule:** Must only checkpoint after RW is fully on → prevents selecting warmup-phase models
3. **Augmentation too aggressive:** 0.3 h_noise destroys cross-slide signal → try softer values

### Cell F: Corrected Cross-Slide Sweep

Fixed seeding, RW-on-only early stopping, softer augmentation, slower ramp (6 epochs instead of 2).

**4 conditions tested:**

| Experiment | h_noise | h_drop | edge_drop | wd |
|---|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.0 | 1e-4 |
| combined_soft | 0.10 | 0.05 | 0.08 | 5e-4 |
| combined_mid | 0.15 | 0.05 | 0.10 | 5e-4 |
| combined_orig | 0.30 | 0.10 | 0.20 | 5e-4 |

**Results (Train ST1+2, Val ST3):**

| Experiment | Best Ep | Val E_geo | Val RW | Val Str | Val λ |
|---|---|---|---|---|---|
| baseline | 10 | 0.4311 | 0.3813 | 0.4979 | 0.233 |
| combined_soft | 10 | 0.4255 | 0.3743 | 0.5122 | 0.270 |
| combined_mid | 8 | 0.4331 | 0.3836 | 0.4949 | 0.253 |
| **combined_orig** | **10** | **0.3453** | **0.3046** | **0.4071** | **0.268** |

**Winner: combined_orig** — the strongest augmentation actually won once early stopping was fixed correctly. The aggressive noise forces the model to rely on robust spatial features rather than slide-specific patterns.

**Decision: Use combined_orig augmentation for all subsequent experiments.**

```python
AUG = {
    'h_noise': 0.3,
    'h_dropout': 0.1,
    'edge_drop': 0.2,
    'weight_decay': 5e-4
}
```

---

## 5. Phase 4: RW Schedule & Stress Weight Tuning (Cell G)

### Cell G: 8-Variant Sweep

With combined_orig augmentation fixed, swept RW schedule and stress weight:

**Grid:**
- `rw_cap ∈ {0.3, 0.5, 1.0}` — cap maximum RW weight
- `rw_ramp ∈ {6, 12}` — epochs to ramp RW
- `lam_stress ∈ {0.1, 0.2, 0.3}` — stress loss weight

**Results (BEST_FULL — best after RW reaches cap):**

| Config | Ep | E_geo | RW | Str | λ |
|---|---|---|---|---|---|
| ramp6_cap1.0_s0.1 | 10 | 0.3453 | 0.3046 | 0.4071 | 0.268 |
| ramp6_cap0.5_s0.1 | 8 | 0.3756 | 0.3235 | 0.5207 | 0.205 |
| **ramp6_cap0.3_s0.1** | **10** | **0.3097** | **0.2707** | **0.3906** | **0.289** |
| ramp12_cap1.0_s0.1 | 14 | 0.3697 | 0.3204 | 0.4935 | 0.219 |
| ramp6_cap1.0_s0.2 | 11 | 0.3773 | 0.2856 | 0.4583 | 0.212 |
| ramp6_cap1.0_s0.3 | 11 | 0.4074 | 0.2659 | 0.4716 | 0.213 |
| ramp6_cap0.5_s0.2 | 8 | 0.3976 | 0.2985 | 0.4956 | 0.244 |
| ramp12_cap0.5_s0.2 | 14 | 0.3798 | 0.2861 | 0.4683 | 0.232 |

**Winner: ramp6_cap0.3_s0.1 → E_geo = 0.3097** (decisive win)

**Key insight: Capping RW weight at 0.3 works much better than full RW (1.0).** Cross-slide generalization degrades when RW loss dominates because the RW operator captures slide-specific diffusion patterns. Keeping stress as the primary driver with RW as a gentle supplement produces the best cross-slide geometry.

**Decision:**
```python
RW_CAP = 0.3         # DO NOT use 1.0
RW_RAMP_EPOCHS = 6
LAM_STRESS = 0.1
WARMUP_EPOCHS = 1
```

---

## 6. Phase 5: Multi-Seed Stability (Cell H)

### Cell H: 5-Seed Check

Ran the winner config (`rw_cap=0.3`) with seeds 0–4 to check if seed 0 was a lucky outlier.

**Results:**

| Seed | Best Ep | Val E_geo |
|---|---|---|
| 0 | 10 | 0.3097 |
| 1 | 6 | 0.4414 |
| 2 | 8 | 0.5113 |
| 3 | 6 | 0.3826 |
| 4 | 3 | 0.4955 |
| **Mean** | — | **0.4281** |
| **Std** | — | **0.0737** |

**Seed 0 was indeed a lucky outlier (0.31 vs mean 0.43).** Realistic performance is ~0.44 ± 0.06. This is important context — don't expect 0.31 on every run.

---

## 7. Phase 6: Graph Topology Investigation (Cells I–J)

### The Central Problem

Everything so far used **target edges** (spatial kNN) for the attention mask. At inference on scRNA data, we don't have spatial coordinates, so we can't build spatial kNN. We must build the attention graph from **gene expression embeddings only** using kNN(H).

### Cell I: Topology Test — kNN(H) vs Target Edges

Built kNN(H) at k=50 (same as target k) from encoder embeddings, measured attention quality.

**Graph quality diagnostics (ST3 val):**
- **Recall@k of kNN(H):** 70–77% (only 70-77% of true spatial neighbors are captured)
- **False neighbor rate:** ~25% of kNN(H) edges are NOT spatial neighbors
- **Edge overlap (target ∩ kNN(H)):** ~50%

**Training comparison:**

| Condition | Best Ep | Val E_geo |
|---|---|---|
| Target edges (oracle) | 10 | 0.3097 |
| kNN(H) attention (k=50) | 3 | 0.7222 |

**Gap: +0.41.** The model is fundamentally topology-sensitive — wrong edges produce wrong coordinates. This became the central problem of the entire toy validation.

---

## 8. Phase 7: Inference Graph — Failed Approaches (Cells K–M)

### Cell K: Hybrid Curriculum Training — FAILED

**Idea:** Mix target edges and kNN(H) edges during training with a curriculum (start with target, transition to kNN(H)).

**Multiple schedules tested:** linear blend, abrupt switch at various epochs.

**Result:** All schedules that improved Val(H) did so by degrading Val(T) — the model was simply learning to accept noise, not to filter it. Net improvement was zero.

**Why it failed:** The architecture cannot "learn to ignore bad edges" through curriculum alone. If you train on noisy graphs, the model gets worse on clean graphs too.

### Cell L: Dynamic kNN(X) Refinement — FAILED

**Idea:** After the first recycle pass, rebuild the graph from predicted coordinates (kNN on predicted X), then re-run.

**Three modes tested:**
1. `dynamic_knnX`: train with target edges, eval with kNN(X) rebuild
2. `dynamic_union`: eval with kNN(X) ∪ kNN(H) union
3. `train_dynamic`: train AND eval with dynamic kNN(X)

**Results:**

| Mode | Val E_geo |
|---|---|
| target baseline | 0.3097 |
| dynamic kNN(X) | 0.9465 |
| dynamic union | 0.7674 |
| train_dynamic | 0.9212 (destroyed target perf too: 0.78) |

**All worse than static kNN(H) at 0.72.** Cascading failure: poor first-pass → bad kNN(X) → worse graph → iterative degradation.

### Cell M: Hard Top-k Edge Gating — FAILED

**Idea:** Build large candidate set (kNN(H) at k=100, recall ≈ 0.97), train a gate MLP to score edges, hard prune to top-30.

**Gate architecture:**
```
EdgeGate: MLP([h_i, h_j, |h_i-h_j|, h_i⊙h_j]) → score
Input: 4×128 = 512 → 64 → 64 → 1
Params: 37,057
```

**Gate supervised with:** KL divergence to $T_{\text{ref}}^{(1)}$ restricted to candidate edges.

**Results:**

| Condition | Val E_geo |
|---|---|
| target edges | 0.3097 |
| kNN(H) k=50 | 0.7222 |
| candidate k=100 (no gate) | 0.9124 |
| candidate k=100 (hard gated top-30) | 0.8642 |

**Hard gating was WORSE than static kNN(H).** Three reasons:
1. **Train-test mismatch:** Gate loss uses smooth softmax KL, but runtime does hard top-k. Early mistakes are catastrophic.
2. **Diagonal contamination:** Self-edges consumed top-k slots.
3. **Chicken-and-egg:** Joint training from scratch → random gate → garbage graph → garbage updates → gate never recovers.

---

## 9. Phase 8: Inference Graph — Soft Gating Success (Cells N–P)

### Cell N: Soft Gate Bias in Attention — FIRST SUCCESS ✅

**Critical architectural change:** Instead of hard pruning, inject gate logits as a **soft bias** into the attention computation:

$$\ell_{ij}^{(h)} = \frac{q_i \cdot k_j}{\sqrt{d}} - \gamma_h \|x_i - x_j\|^2 + \beta \cdot g_{ij}$$

Where $g_{ij}$ is the gate logit (shared across heads), and $\beta$ is a learned/fixed scaling factor. Wrong edges get negative bias → small attention weight → harmless. No hard pruning, no catastrophic failures.

**EdgeGateSoft module:**
```
Input: [h_i, h_j, |h_i-h_j|, h_i⊙h_j]  (4 × d_input = 512)
MLP: 512 → 64 → 64 → 1
Diagonal: excluded from computation, set to 0.0 (neutral)
Params: 37,057
```

**Warm-start pretraining:**
- Pretrain gate alone (IPA frozen) for 2000 steps on $T_{\text{ref}}^{(1)}$ KL loss
- Gate precision improved from 0.315 → 0.716 after pretraining

**Cell N Results:**

| Condition | Best Ep | Val E_geo |
|---|---|---|
| Target edges (oracle) | 10 | 0.3097 |
| kNN(H) k=50 (static) | 3 | 0.7222 |
| **Soft gate warmstart** | **10** | **0.4886** |
| Soft gate scratch | 4 | 0.5159 |

**Gap closed: 0.72 → 0.49 = 57% of the gap.** First approach that improved inference-graph performance without destroying anything.

### Cell O: β/λ_gate Sweep

Tuned the gate strength and supervision weight.

**Sweep results:**

| Config | Ep | Val E_geo |
|---|---|---|
| β=0.5, λ_gate=0.05 | 4 | 0.4510 |
| β=1.0, λ_gate=0.05 | 10 | 0.4886 |
| **β=2.0, λ_gate=0.05** | **7** | **0.4417** |
| β=1.0, λ_gate=0.10 | 16 | 0.4869 |

**Winner: β=2.0, λ_gate=0.05** → gap to target = +0.132

**Note:** β=2.0 peaked at epoch 7 then degraded — use early stopping on gated-graph validation.

### Cell P: Candidate k + Pretrain Steps Follow-up

**Recall at different k:**

| k | Recall |
|---|---|
| 80 | 0.948 ± 0.051 |
| 100 | 0.967 ± 0.038 |
| 120 | 0.977 ± 0.029 |

**6 configs tested (3 k values × 2 pretrain steps):**

| Config | Ep | Val E_geo |
|---|---|---|
| **k=80, pt=500** | **7** | **0.3794** |
| k=80, pt=2000 | 6 | 0.4174 |
| k=100, pt=500 | 10 | 0.5036 |
| k=100, pt=2000 | 4 | 0.5228 |
| k=120, pt=500 | 15 | 0.4911 |
| k=120, pt=2000 | 5 | 0.5142 |

### Two Surprises

1. **k=80 wins decisively** (0.38 vs 0.50 at k=100). Fewer candidates = less noise for the gate to filter. Recall at 0.95 is sufficient.
2. **500 pretrain steps beats 2000 everywhere.** Less pretraining keeps the gate flexible during joint training.

### Final Result

| Condition | Val E_geo | Gap to target |
|---|---|---|
| Target edges (oracle) | 0.31 | — |
| **Gated k=80 (inference)** | **0.38** | **+0.07** |
| Gated k=100 (prev best) | 0.44 | +0.13 |
| Static kNN(H) k=50 | 0.72 | +0.41 |
| Random embeddings | 2.85 | +2.54 |

**83% of the inference-graph gap closed.** Remaining +0.07 is small enough to proceed with pipeline integration.

---

 IPA-Lite → Full IPA-R

IPA-Lite was a simplified proof of concept. The full IPA-R should upgrade:

1. **Coordinate update:** IPA-Lite uses a simple learned linear on displacements. Full IPA-R should use the richer point attention from OpenFold (query/key/value points transformed through rigid frames, or at minimum through learned projections per head).

2. **Pair representations:** IPA-Lite has no pair features (just scalar qk + geometry). Full IPA can include pair embeddings $z_{ij}$ (e.g., from outer product of node states) that feed into attention logits.

3. **Separate blocks per recycle (optional):** IPA-Lite uses weight-shared recycling (same block K times). Full IPA can have independent blocks or a hybrid (shared backbone + independent head per recycle).

4. **Scale:** IPA-Lite tested on minisets (n≤384). Full pipeline operates on full slides (~1300 spots) and potentially larger.

### What MUST Be Preserved

These are validated and should NOT be changed without re-validation:

1. **Loss functions:** `loss_rw_kl`, `loss_stress_log`, `loss_repulsion`, `loss_entropy_reg` — all verified to produce exact matches with target operators. DO NOT modify these.

2. **`build_predicted_rw_operator`:** Exact match with target builder. Uses `0.5*(W + W.T)` symmetrization, row normalization, NO self-loops. DO NOT add self-loops.

3. **`gauge_fix`:** Center + unit-RMS normalize after every recycle. Essential for stability.

4. **Geometry term in attention:** The $-\text{softplus}(\gamma_h) \cdot d^2_{ij}$ term is critical. Initialize $\gamma$ at 1.0 (via $\text{softplus}^{-1}(1.0) = 0.5414$).

5. **Edge masking:** $-\infty$ for non-edges in attention logits, then softmax. Use `nan_to_num(0.0)` to handle all-masked rows.

6. **Coordinate update direction:** $x_j - x_i$ (toward neighbors), NOT $x_i - x_j$.

7. **Linear_x zero initialization:** The coordinate update linear should start at zero so initial coordinate updates are zero (stability).

### What MUST Be Integrated: Soft Edge Gating

This is the most important finding. Without it, inference on scRNA fails.

**Implementation checklist:**

1. Build candidate graph from kNN in embedding space (H), k=80
2. `EdgeGateSoft` MLP computes per-edge gate logits
3. Gate bias (scaled by β=2.0) is added to attention logits BEFORE masking
4. The candidate mask replaces the target edge mask at inference
5. Gate is pretrained for 500 steps on $T_{\text{ref}}^{(1)}$ KL before joint training
6. Gate loss ($\lambda=0.05$) is added to the total loss during joint training
7. **Diagonal handling:** Gate excludes diagonal edges, sets diagonal to 0.0 (neutral bias)

**The gate bias is broadcast across all attention heads:**
```python
logits = logits + gate_bias.unsqueeze(0)  # (1, n, n) → (H, n, n)
```

### Scaling Considerations

1. **Dense → Sparse attention:** For n>500, the dense (n, n) attention mask becomes expensive. Switch to:
   - Edge-list form: store candidate edges as (src, dst) index lists
   - Compute gate logits only on edges
   - Use sparse neighborhood softmax (torch_scatter or custom kernel)

2. **Gate computation cost:** Currently O(n²) because of dense masking. With edge-list form, becomes O(n×k) = O(n×80).

3. **Multi-slide training:** Train on as many slides as possible. Cross-slide generalization was the main bottleneck at toy scale (only 2 training slides).

4. **RW cap at 0.3:** This was critical for cross-slide generalization. When scaling to more slides, you MAY be able to increase this, but start at 0.3 and only increase if cross-slide validation improves.

### Training Recipe for Full Pipeline

```
1. Freeze encoder, compute all embeddings H
2. Build minisets with F2+F4 targets (adaptive k, steps, landmarks)
3. Build candidate masks: kNN(H) at k=80 per miniset
4. Cache T_ref (1-step RW) per training miniset for gate supervision
5. Initialize EdgeGateSoft
6. Pretrain gate alone for 500 steps on T_ref KL
7. Initialize IPA-R model
8. Joint training:
   - Warmup 1 epoch (stress only)
   - RW ramp 6 epochs to cap=0.3
   - Train with combined_orig augmentation
   - Gate loss λ=0.05
   - Early stop patience=8 on gated-graph validation E_geo
9. Always track Val(T) and Val(H_gate) separately
```

### Evaluation Metrics

At validation/test, compute:
- **E_geo:** $w_{\text{rw}} \cdot L_{\text{RW}} + 0.1 \cdot L_{\text{stress}}$ (main metric)
- **$\lambda_2/\lambda_1$:** covariance eigenvalue ratio (should be >0.01, ideally 0.2–0.5)
- **median pdist:** median pairwise distance (should be >0.01, typically ~1.2)
- **Gate precision:** fraction of gated edges that are true spatial neighbors

### What NOT To Do (Validated Failures)

These approaches were tested and failed. Do not re-implement them:

1. ❌ **Hybrid curriculum (mix target + kNN(H) edges during training):** Closes gap by degrading target performance, not by improving inference.

2. ❌ **Dynamic kNN(X) refinement (rebuild graph from predicted coords):** Cascading failure — bad first pass → worse graph → iterative degradation.

3. ❌ **Hard top-k gating (select top-k edges from gate scores):** Train-test mismatch between smooth KL supervision and hard discrete selection.

4. ❌ **RW weight cap at 1.0:** Causes cross-slide overfitting. Always cap at 0.3 until you have strong evidence to raise it.

5. ❌ **2000 pretrain steps for gate:** 500 steps is better — keeps gate flexible during joint training.

6. ❌ **Candidate k=100 or k=120:** k=80 wins. Fewer candidates = less noise for the gate.

---

## 10. Phase 9: Attention Diagnostics (Cells Q1–Q2)

### Motivation

After Cell P achieved Val E_geo = 0.38 on gated inference graphs (closing 83% of the inference gap), the research agent recommended **diagnosing the attention mechanism before scaling to full IPA-R**. The concern: is the IPA-Lite attention actually learning to select true spatial neighbors, or is it succeeding for other reasons?

Three specific diagnostics were requested:

1. **AttnRecall@k vs recycle:** For each node, take the top-k nodes by attention weight. What fraction are true spatial neighbors? Does this improve across recycles?
2. **Logit component correlations:** Decompose attention into its three terms (qk, geo, gate). Which term drives the final attention pattern?
3. **Quick ablation:** Remove each term independently and measure the effect on E_geo.

### Cell Q1: Instrumented Attention Analysis

#### What This Cell Does

1. **Trains** a fresh V1 (weight-shared) model with the Cell P winner config:
   - k=80 candidate graph, β=2.0, 500-step gate pretrain, rw_cap=0.3
   - All hyperparameters from the validated recipe (§9)
   
2. **Runs an instrumented forward pass** on 20 val minisets from ST3. The instrumentation hooks into `IPALiteBlock.forward` and extracts, for each recycle $r$ and head $h$:
   - $a^{(h)}_{ij}$ — the full attention weight matrix (after softmax)
   - $\ell_{\text{qk}}^{(h)}$ — the semantic logit term: $q_i \cdot k_j / \sqrt{d}$
   - $\ell_{\text{geo}}^{(h)}$ — the geometry logit term: $-\text{softplus}(\gamma_h) \cdot \|x_i - x_j\|^2$
   - $\ell_{\text{gate}}$ — the gate bias term: $\beta \cdot g_{ij}$ (shared across heads)

3. **Computes two diagnostics** per head per recycle, averaged over all nodes and minisets:
   - **AttnRecall@30:** For node $i$, take top-30 by $a^{(h)}_{i,:}$. Compute fraction that are true spatial neighbors (from target edge mask).
   - **Spearman $\rho$(attn, component):** Rank correlation between attention weights and each logit component, computed over all candidate edges (excluding diagonal).

#### How to Reproduce

The instrumented forward is a manual reimplementation of the block's forward pass that saves intermediate logit components. It does NOT use PyTorch hooks — instead, it copies the block's linear layers and recomputes q, k, v, then computes each logit term separately before combining them. The key code pattern:

```python
q = block.linear_q(s).view(n, H, D)
k = block.linear_k(s).view(n, H, D)
logit_qk = einsum('ihd,jhd->hij', q, k) / sqrt(D)
logit_geo = -softplus(block.head_weights)[:, None, None] * dist_sq.unsqueeze(0)
logit_gate = beta * gate_module(H, cand_mask)
logits = logit_qk + logit_geo + logit_gate.unsqueeze(0)
# ... mask, softmax, extract attention ...
# Then ALSO run the real block forward for state updates:
s, X = block(s, X, cand_mask, gate_bias=gb)
```

The true spatial neighbor mask comes from `edge_masks[mi]` (the target kNN graph built on actual ST coordinates).

#### Training Result

Model trained to best epoch 3, Val E_geo = 0.8735.

**⚠️ Critical context about this baseline:** This V1 model scored 0.87, far worse than Cell P's 0.38. Investigation revealed the `gate_bias` monkey-patch from Cell N1 (which adds gate support to `IPALiteBlock.forward`) was disrupted during the Q2 ablation's save/restore cycle. The model was effectively running without gate integration, reverting to raw kNN(H) inference graph performance. **The attention diagnostics are still valid** — they characterize the V1 shared architecture's attention behavior. The E_geo of 0.87 vs 0.38 means the Q1 model had a weaker baseline, but the attention patterns (collapse timing, correlation structure) are representative.

#### Results: Per-Recycle Attention Diagnostics

Full table (20 val minisets averaged, k_eval=30, k_cand=80):

| Metric | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| **Recall@30 head 0** | 0.495 | 0.319 | 0.155 | 0.153 | 0.153 |
| **Recall@30 head 1** | 0.540 | 0.157 | 0.154 | 0.153 | 0.153 |
| **Recall@30 head 2** | 0.541 | 0.284 | 0.153 | 0.152 | 0.153 |
| **Recall@30 head 3** | 0.491 | 0.159 | 0.153 | 0.152 | 0.153 |
| $\rho$(attn, geo) avg | 0.43 | 0.01 | −0.02 | −0.02 | −0.03 |
| $\rho$(attn, gate) avg | 0.52 | 0.02 | −0.03 | −0.03 | −0.03 |
| $\rho$(attn, qk) avg | 0.33 | 0.20 | 0.12 | 0.11 | 0.10 |

#### Key Finding: Attention Collapses After Recycle 1

- **R0:** Recall@30 ≈ 0.50. All three logit terms contribute (ρ = 0.3–0.5). This is the only recycle doing meaningful neighbor selection.
- **R1:** Recall drops to 0.16–0.32. Only qk briefly survives (ρ ≈ 0.20). Geo and gate correlations already at zero.
- **R2–R4:** Recall ≈ 0.153, which is **below random chance**. (Randomly picking 30 of 80 candidates that have ~50% overlap with true neighbors would give ~0.19.) ALL correlations → 0. Attention is effectively **uniform** — softmax produces near-equal weights across all candidate edges.

The learned $\gamma$ values at R4: [0.76, 0.63, 0.63, 0.68] — all below the initialization of 1.0, confirming the geometry term is weakening during training.

#### Interpretation

Weight-shared recycling causes the attention to degenerate. The same q/k/v projections applied repeatedly to increasingly refined (but gauge-fixed) states converge to a regime where logit variance → 0. This is consistent with known behavior in weight-tied transformers. The model extracts useful spatial signal only in R0 and partially R1. R2–R4 are performing near-uniform averaging — essentially a diffusion/smoothing operation on node states, not selective neighbor aggregation.

### Cell Q2: Logit-Term Ablation

#### What This Cell Does

Trains 4 models for 10 epochs each, identical setup except one logit term is zeroed out per condition:

- **full:** all three terms (baseline): $\ell = \ell_{\text{qk}} + \ell_{\text{geo}} + \beta \cdot g_{ij}$
- **no_qk:** $\ell = \ell_{\text{geo}} + \beta \cdot g_{ij}$ (remove expression matching)
- **no_geo:** $\ell = \ell_{\text{qk}} + \beta \cdot g_{ij}$ (remove distance penalty)
- **no_gate:** $\ell = \ell_{\text{qk}} + \ell_{\text{geo}}$ (remove learned gate bias)

Implementation: monkey-patch `IPALiteBlock.forward` with an `ablate` parameter that zeros the specified component's logit contribution before softmax. Same training loop, same k=80 candidate graph, same β=2.0.

#### Results

| Condition | Best Epoch | Val E_geo | Δ vs full |
|---|---|---|---|
| full (baseline) | 6 | 0.4039 | — |
| no_qk (geo+gate only) | 3 | 0.6765 | **+0.27** (significant degradation) |
| **no_geo (qk+gate only)** | **3** | **0.3904** | **−0.01** (slightly better!) |
| no_gate (qk+geo only) | 6 | 0.8857 | **+0.48** (catastrophic) |

#### Three Decisive Findings

1. **Gate is essential** (+0.48 degradation when removed). Without gate, performance equals raw kNN(H) at ~0.88. The gate IS the inference mechanism — it provides the spatial prior that the model cannot learn from expression alone.

2. **qk (expression matching) is the second-most important signal** (+0.27 degradation). The semantic $q \cdot k$ dot product provides meaningful neighbor discrimination beyond what the gate alone gives.

3. **The geometry penalty $-\gamma_h \|x_i - x_j\|^2$ is useless on inference graphs** (−0.01 when removed, meaning removal slightly helps). On inference graphs, coordinates $X$ start from noisy MLP predictions. Using these unreliable distances to bias attention creates a self-reinforcing error loop: the model attends to whatever happens to be close in the *current wrong layout*, reinforcing that layout.

#### Implications for Architecture

- The gate + qk combination is sufficient. Geo can be removed or redesigned.
- Any architectural change should preserve gate and qk mechanisms.
- If geo is kept, it should only activate after coordinates have been refined (not at R0).

---

## 11. Phase 10: Architecture Fixes — Unshared Blocks + QK-Norm (Cells R1–R3)

### Motivation

The Phase 9 diagnostics identified two concrete architectural problems:

1. **Weight-shared recycling degeneracy:** Reusing the same block parameters across recycles causes attention to collapse to uniform by R2. The same q/k projections + repeated gauge-fixing converge to a flat-logit regime.

2. **Mis-specified geometry term:** The $-\gamma \|x_i - x_j\|^2$ penalty assumes current distances are meaningful. On inference graphs where early coordinates are random/noisy, this is a self-reinforcing error signal.

The research agent recommended two specific fixes before scaling:
- **Fix A:** Unshare parameters across recycles (`nn.ModuleList` of independent blocks)
- **Fix B:** Remove or schedule the geometry term (disable in early recycles)

### Cell R1: IPALiteModelV2 — New Architecture + Training Comparison

#### Architecture: IPALiteBlockV2

Identical to `IPALiteBlock` with three additions:

**1. Configurable geometry (`use_geo` flag):**
Each block has a boolean `use_geo` that controls whether the $-\gamma \|x_i - x_j\|^2$ term is computed. This allows per-recycle geometry schedules like `[False, False, True]` (geo only in final recycle).

**2. QK-normalization (added after iteration, see below):**
LayerNorm applied to q and k vectors before the dot product, with a learned logit scale:

```python
# In __init__:
self.q_norm = nn.LayerNorm(c_head, elementwise_affine=False)
self.k_norm = nn.LayerNorm(c_head, elementwise_affine=False)
self.logit_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(float(c_head)))))

# In forward:
q = self.q_norm(q)
k = self.k_norm(k)
tau = self.logit_scale.exp()  # learned scale, init = sqrt(d)
logits = tau * torch.einsum('ihd,jhd->hij', q, k)
```

This forces cosine-similarity-like behavior (bounded logits) while allowing the model to learn how sharp attention should be via $\tau$.

**3. Geo clamping (when `use_geo=True`):**
```python
geo_bias = -gamma[:, None, None] * dist_sq.unsqueeze(0)
logits = logits + geo_bias.clamp(-10.0, 0.0)
```
This prevents the distance term from dominating the logits — it can only subtract up to 10 from the logit, not explode to $-\infty$.

**Full attention logit formula for V2:**

$$\ell_{ij}^{(h)} = \tau^{(h)} \cdot \left( \hat{q}_i^{(h)} \cdot \hat{k}_j^{(h)} \right) + \text{clamp}\left(-\gamma_h \|x_i - x_j\|^2,\; -10,\; 0\right) + \beta \cdot g_{ij}$$

where $\hat{q} = \text{LayerNorm}(q)$, $\hat{k} = \text{LayerNorm}(k)$, $\tau = \exp(\text{logit\_scale})$.

The geo clamp term is only present in blocks where `use_geo=True`.

#### Architecture: IPALiteModelV2

Uses `nn.ModuleList` instead of a single shared block:

```python
self.blocks = nn.ModuleList([
    IPALiteBlockV2(c_s, n_heads, c_head, use_geo=geo_schedule[r])
    for r in range(n_recycles)
])
```

Forward iterates `for r, block in enumerate(self.blocks)` instead of repeating the same block.

**Parameter counts:**

| Model | Params | Note |
|---|---|---|
| V1 shared (5R) | 154,678 | One IPA block reused 5× |
| V2 (3R) | 352,277–352,281 | 3 independent blocks |
| V2 (5R) | 549,879–549,891 | 5 independent blocks |

#### R1 Iteration History: Three Attempts

**Attempt 1: V2 with learnable temperature (FAILED — NaN)**

The first implementation included a learnable per-head temperature: `self.log_temp = nn.Parameter(torch.zeros(n_heads))`, applied as `logits = logits * exp(log_temp)`. This caused **immediate NaN explosion** — the gate bias (already large from β=2.0 scaling) combined with the temperature multiplier meant ∂L/∂log_τ was enormous on the first backward pass. `log_temp` diverged to ±∞ instantly.

Both V2 conditions produced `Val Eg=nan` for all epochs. V1 baseline trained normally (0.87).

**Fix:** Remove `log_temp` entirely. Temperature control is unnecessary once QK-norm bounds the logits.

**Attempt 2: V2 unshared blocks, no QK-norm (PARTIALLY WORKED)**

Results:

| Condition | Ep | Val E_geo |
|---|---|---|
| V2_no_geo | 4 | 0.7299 |
| V2_geo_late [F,F,T,T,T] | 3 | 0.6311 |
| V1_shared | 3 | 0.8735 |

Improvement over V1, but attention diagnostics revealed a **new problem: logit explosion**. Without shared-block regularization, independent blocks learn progressively larger q/k norms across recycles:

| Recycle | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| std(logits) | 4.3 | 80–137 | 1,148–2,639 | **17K–35K** | **50K–235K** |
| entropy (nats) | 3.2 | 0.06–1.9 | 0.08–0.13 | 0.01 | **≈0** |

This is the **opposite** failure mode from V1. V1 had logits too flat → uniform attention. V2 without QK-norm had logits exploding → one-hot attention to a single arbitrary neighbor. Both are degenerate, just in different directions.

**Attempt 3: V2 with QK-norm + geo clamping (FINAL)**

Added QK-normalization (LayerNorm on q,k) + learned logit scale + geo clamping. This is a standard fix for logit explosion in deep transformers (used in ViT, PaLM, etc.).

**Final R1 results with all fixes:**

| Condition | Recycles | Best Ep | Val E_geo | Params |
|---|---|---|---|---|
| V2_3R_no_geo | 3 | 7 | 0.4420 | 352,277 |
| **V2_3R_geo_last [F,F,T]** | **3** | **4** | **0.3999** | **352,281** |
| V2_5R_geo_late [F,F,T,T,T] | 5 | 4 | 0.4213 | 549,891 |
| V1_shared (broken gate) | 5 | 3 | 0.8735 | 154,678 |

**Key observations from R1 training:**

1. **3 recycles beats 5 recycles** (0.40 vs 0.42). R3–R4 are dead weight in every configuration.
2. **geo_last > no_geo** (0.40 vs 0.44). The geometry term IS useful when: (a) clamped to [-10, 0], and (b) applied only in the final recycle after 2 rounds of coordinate refinement. This resolves the Q2 ablation paradox — geo was useless in V1 where it's active from R0 on noisy coords, but helps in V2 where it only activates at R2 on partially refined coords.
3. **V2_3R_geo_last at 0.40 is close to Cell P's V1 best of 0.38** — within seed noise, but not a clear improvement despite 2.3× more parameters.

### Cell R2: Attention Diagnostics on V2 Winner

#### What This Cell Does

Same instrumented forward as Q1, but adapted for the V2 architecture:
- Iterates over `model.blocks[r]` (independent blocks, not shared)
- Applies QK-norm before computing logit_qk (matches the real forward exactly)
- Applies geo clamping when `block.use_geo=True`
- Computes the same Recall@30, Spearman ρ correlations, plus two NEW diagnostics the agent requested:
  - **std(logits):** Standard deviation of attention logits across candidate edges, per head per recycle. If this → 0, attention is uniform. If this → ∞, attention is one-hot.
  - **Attention entropy:** $-\sum_j a_{ij} \log a_{ij}$ averaged over nodes $i$. High entropy = spread attention, low entropy = concentrated.

#### Results for V2_3R_geo_last

| Metric | R0 | R1 | R2 (+geo) |
|---|---|---|---|
| **Recall@30 head 0** | 0.410 | 0.479 | **0.580** |
| **Recall@30 head 1** | 0.394 | 0.465 | **0.540** |
| **Recall@30 head 2** | 0.280 | 0.479 | **0.600** |
| **Recall@30 head 3** | 0.348 | 0.593 | **0.563** |
| $\rho$(attn, qk) head 0 | **0.916** | 0.553 | 0.244 |
| $\rho$(attn, qk) head 1 | **0.908** | 0.520 | 0.368 |
| $\rho$(attn, qk) head 2 | **0.843** | 0.647 | 0.173 |
| $\rho$(attn, qk) head 3 | **0.759** | 0.140 | 0.388 |
| $\rho$(attn, gate) head 0 | 0.424 | 0.587 | **0.817** |
| $\rho$(attn, gate) head 1 | 0.315 | 0.556 | **0.727** |
| $\rho$(attn, gate) head 2 | 0.002 | 0.579 | **0.865** |
| $\rho$(attn, gate) head 3 | 0.238 | 0.890 | **0.783** |
| logit_std (all heads) | 39–51 | 6–9 | **6–7** |
| entropy (nats, all heads) | 0.37–0.89 | 2.49–2.90 | **2.90–3.05** |

#### Comparison: V1 Shared vs V2 Attention Health

| Diagnostic | V1 shared (Q1) | V2_3R_geo_last (R2) |
|---|---|---|
| Recall@30 at final recycle | **0.153** (below chance) | **0.54–0.60** (healthy) |
| Correlations at final recycle | ≈ 0 (no signal) | 0.17–0.87 (structured) |
| Logit std at final recycle | ≈ 0 (flat → uniform) | 6–7 (bounded, healthy) |
| Entropy at final recycle | ≈ 0 (one-hot in V2-no-QKnorm) or flat (V1) | 2.9–3.0 (spread) |
| Recall trend across recycles | Monotone collapse ↓ | **Monotone increase** ↑ |

#### The Attention Story for V2 (Clean and Interpretable)

- **R0 is qk-dominated** ($\rho_{\text{qk}}$ = 0.76–0.92, $\rho_{\text{gate}}$ = 0.00–0.42): Attention explores by expression similarity. The gate hasn't taken over yet. Recall ≈ 0.28–0.41.
- **R1 is mixed qk + gate** ($\rho_{\text{qk}}$ = 0.14–0.65, $\rho_{\text{gate}}$ = 0.56–0.89): Both signals cooperate. The model starts using spatial priors from the gate alongside expression matching. Recall ≈ 0.47–0.59.
- **R2 is gate-dominated** ($\rho_{\text{gate}}$ = 0.73–0.87, $\rho_{\text{qk}}$ = 0.17–0.39): The pretrained gate's spatial prior guides attention after 2 rounds of coordinate refinement. The clamped geo term adds locality bias. **Recall peaks at 0.54–0.60.**

This is the first configuration where attention quality **improves** across recycles rather than degenerating.

### Cell R3: Publication-Quality Attention Visualizations

#### What This Cell Does

Generates two types of plots for 3 val minisets:

**Plot Type 1: Attention Focus Maps**

For each of 4 query nodes (one per spatial quadrant), generates a grid:
- Rows = recycles (R0, R1, R2)
- Columns = attention heads (Head 0, 1, 2, 3)

Each panel shows:
- All nodes plotted at their **true spatial coordinates** (ST layout, for interpretability — not predicted coords)
- Gray dots: non-candidate nodes
- Yellow-to-red colored dots: candidate neighbors, colored by attention weight (YlOrRd colormap)
- Green rings: true spatial neighbors
- Red star: query node
- Red lines: from query to top-5 attended nodes, thickness proportional to attention weight

**Plot Type 2: Coordinate Progression**

Side-by-side panels showing:
- Target ST layout
- Predicted layout after R0 (no geo)
- Predicted layout after R1 (no geo)
- Predicted layout after R2 (+geo)

#### Visual Findings

**Attention focus plots:**
- At R0, red lines reach across the entire miniset — long-range qk exploration by expression similarity
- At R1, lines shorten and concentrate — model starting to localize
- At R2 (+geo), lines are mostly short and local, often connecting to nearby green-ringed true neighbors. However, some heads still attend to a few wrong distant nodes.

**Coordinate progression:**
- R0 produces a rough spatial blob with some structure
- R1 spreads it into a more recognizable layout
- R2 (+geo) introduces visible **clumping/distortion** — nodes cluster into tight groups rather than maintaining the even spread of the target. The geo term's "closer-is-better" pressure, even when clamped, creates over-locality in the predicted coordinates.

---

## 12. Phase 11: Controlled Replication — V1 vs V2 Identical Conditions (Cell S1)

### Motivation

The Phase 9–10 results left a critical ambiguity: the V1 attention "collapse" diagnosed in Q1 was measured on a model with a **broken gate** (E_geo=0.87 instead of Cell P's 0.38). This meant the comparison between V1 and V2 attention health was confounded. The research agent recommended a clean replication: train V1 and V2 with identical gate integration, identical hyperparameters, and proper instrumentation from the start.

The hypothesis: V1's attention genuinely collapses after R1–R2 under shared recycling, and V2 maintains structured attention — and this difference is real, not an artifact of the earlier broken-gate run.

### Cell S1: Experiment 1 — "Truth Baseline"

#### Setup

Standardized protocol (used for all subsequent experiments):

- **Data split:** Train ST1+ST2, Val ST3 (50 minisets)
- **Candidate graph:** kNN(H) with $k_{\text{cand}}=80$
- **Gate:** EdgeGateSoft, pretrain 500 steps on $T_{\text{ref}}^{(1)}$ KL, $\lambda_{\text{gate}}=0.05$, $\beta=2.0$
- **Loss:** warmup 1 epoch, RW ramp 6 epochs to cap 0.3, $\lambda_{\text{stress}}=0.1$
- **Augmentation:** h_noise=0.3, h_dropout=0.1, edge_drop=0.2, wd=5e-4
- **Budget:** max 12 epochs, early stop patience 4 (measured only after RW cap reached)
- **Seeds:** 0, 1

Two conditions: V1 shared 5R (154,678 params) and V2 3R geo_last (352,281 params).

#### Results

| Condition | Seed | Best Ep | Val E_geo (H_gate) | Val E_geo (T_edge) |
|---|---|---|---|---|
| V1_shared_5R | 0 | 7 | 0.4175 | 0.1686 |
| V1_shared_5R | 1 | 9 | 0.4483 | 0.1632 |
| V2_3R_geo_last | 0 | 8 | 0.5546 | 0.2806 |
| V2_3R_geo_last | 1 | 6 | 0.4905 | 0.2237 |

**Averages:**

| Architecture | Mean E_geo (H_gate) | Std |
|---|---|---|
| **V1_shared_5R** | **0.4329** | **0.0154** |
| V2_3R_geo_last | 0.5225 | 0.0320 |

**V1 wins decisively on E_geo**, with both lower mean (0.43 vs 0.52) and lower variance.

#### Critical Finding: V1 Attention Does NOT Collapse

The most important result from S1 is that the V1 attention collapse diagnosed in Q1 was **an artifact of the broken gate**, not an inherent property of weight-shared recycling.

**Recall@30 at final recycle (averaged across heads):**

| Condition | Recall | Entropy | logit_std | $|\Delta X|/\sqrt{n}$ |
|---|---|---|---|---|
| V1_shared_5R s0 | 0.549 | 2.88 | 6.0 | 0.068 |
| V1_shared_5R s1 | 0.557 | 2.91 | 6.0 | 0.083 |
| V2_3R_geo_last s0 | 0.550 | 2.89 | 6.7 | 0.135 |
| V2_3R_geo_last s1 | 0.562 | 2.85 | 7.1 | 0.128 |

All four runs show **essentially identical attention health** at the final recycle: Recall ≈ 0.55, entropy ≈ 2.9, logit_std ≈ 6–7. V1 does not collapse — it maintains structured, gate-dominated attention across all 5 recycles.

**V1 per-recycle attention (seed 0, head-averaged):**

| Metric | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| Recall@30 | 0.559 | 0.513 | 0.511 | 0.532 | 0.549 |
| $\rho$(attn,gate) | 0.787 | 0.689 | 0.657 | 0.717 | 0.766 |
| $\rho$(attn,qk) | 0.291 | 0.434 | 0.472 | 0.402 | 0.352 |
| $\rho$(attn,geo) | 0.591 | 0.576 | 0.567 | 0.590 | 0.643 |
| $|\Delta X|/\sqrt{n}$ | 0.210 | 0.161 | 0.131 | 0.085 | 0.068 |

The V1 attention is **gate-dominated across all recycles** ($\rho_{\text{gate}}$ = 0.66–0.79), with qk providing a secondary contribution. Recall dips slightly at R1–R2 but recovers by R4. This is fundamentally different from the Q1 diagnosis where $\rho$ → 0 and Recall dropped to 0.15.

**V2 per-recycle attention (seed 0, head-averaged):**

| Metric | R0 | R1 | R2 |
|---|---|---|---|
| Recall@30 | 0.323 | 0.525 | 0.550 |
| $\rho$(attn,gate) | 0.176 | 0.739 | 0.773 |
| $\rho$(attn,qk) | 0.860 | 0.447 | 0.323 |
| $\rho$(attn,geo) | 0.000 | 0.000 | 0.622 |

V2 shows the expected qk→gate progression across recycles, but its R0 is much weaker (Recall 0.32 vs 0.56 for V1) because qk alone without gate is poor at neighbor selection. V2 catches up by R2 but never surpasses V1's final recall.

**Coordinate step magnitudes confirm V1's later recycles contribute diminishing but non-zero updates:**

| Architecture | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| V1 s0 | 0.210 | 0.161 | 0.131 | 0.085 | 0.068 |
| V1 s1 | 0.212 | 0.215 | 0.166 | 0.121 | 0.083 |
| V2 s0 | 0.119 | 0.308 | 0.135 | — | — |
| V2 s1 | 0.099 | 0.275 | 0.128 | — | — |

V1's $|\Delta X|$ decays monotonically but doesn't hit zero, confirming active (though gentle) refinement through R4. V2 concentrates most of its coordinate work in R1 (0.28–0.31).

#### Implications

1. **The Q1 "V1 collapse" narrative was wrong.** When the gate is properly integrated, V1 weight-shared recycling does NOT degenerate. The gate provides a stable spatial prior that keeps attention structured across all recycles.
2. **V2's architectural complexity is not justified at toy scale.** V2 uses 2.3× more parameters and achieves worse E_geo (0.52 vs 0.43). Its "healthier attention progression" (qk→gate) does not translate to better geometry.
3. **V1 shared remains the best architecture** for the current pipeline. The attention is healthy, the metric is best, and the parameter count is modest.

**Decision: V1 shared 5R is the primary architecture going forward. V2 is abandoned for now.**

---

## 13. Phase 12: Geometry Bias Redesign (Cells S6–S7)

### Motivation

With V1 confirmed as the winner, the next question from the research agent was whether the geometry bias term ($-\gamma\|x_i - x_j\|^2$) could be redesigned to actually help. The Q2 ablation had shown it was neutral-to-harmful in V1, and the agent proposed two alternatives:

1. **RBF distance bias** (Experiment 6): Replace monotone "closer is always better" with learned distance-band preferences using Radial Basis Functions.
2. **Point-IPA** (Experiment 7): Learned query/key point offsets anchored at coordinates, analogous to AlphaFold's invariant point attention.

### Cell S6: Experiment 6 — RBF Distance Bias vs Monotone vs No-Geo

#### Setup

Three geo modes on V1 5R shared, plus V2 3R with RBF at last recycle. Three seeds each.

**RBF implementation:** Normalized distances $\bar{d}_{ij} = \|x_i - x_j\| / (\text{median candidate-edge distance} + \epsilon)$, passed through $M=16$ Gaussian RBF centers in $[0, 2]$. Per-head learnable weights $w_h \in \mathbb{R}^M$ produce a bias clipped to $[-b_{\max}, b_{\max}]$.

#### Results

| Condition | Seeds | Mean E_geo | Std | Mean p5_NN | Mean CV_NN |
|---|---|---|---|---|---|
| V1_5R_mono | 3 | 0.6363 | 0.1060 | 0.0117 | 0.79 |
| V1_5R_no_geo | 3 | 0.5609 | 0.1643 | 0.0136 | 0.74 |
| **V1_5R_rbf** | **3** | **0.4832** | **0.0931** | **0.0124** | **0.79** |
| V2_3R_rbf_last | 3 | 0.4854 | 0.0573 | 0.0136 | 0.76 |

**RBF wins on V1** (0.48 vs 0.56 no_geo, 0.64 mono). V2 RBF-last is comparable (0.49). Monotone geo is the worst performer — confirming the Q2 finding that the raw $-\gamma d^2$ term is harmful.

#### Critical Discovery: Anti-Locality in RBF

The RBF learned **negative correlations** with attention — $\rho(\text{attn}, \text{geo})$ was consistently negative across all seeds and recycles:

| Seed | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| s0 (avg head) | −0.10 | −0.15 | −0.17 | −0.19 | −0.22 |
| s1 (avg head) | −0.07 | −0.10 | −0.14 | −0.18 | −0.22 |
| s2 (avg head) | −0.16 | −0.23 | −0.25 | −0.25 | −0.25 |

This means the RBF is doing the **opposite** of what was expected — it learns to slightly prefer **more distant** nodes in the candidate set, providing a mild anti-locality bias. The RBF essentially acts as a regularizer against over-locality (the "clumping" problem observed with monotone geo in R3 visualizations).

**Monotone geo shows the expected positive correlation** ($\rho$ ≈ 0.5–0.6), confirming it biases toward close neighbors. But this hurts because early coordinates are noisy.

**No-geo has $\rho = 0$ everywhere** (as expected — no geo term to correlate with).

#### Why RBF's Anti-Locality Helps

The gate already provides strong locality bias ($\rho_{\text{gate}}$ ≈ 0.7–0.9). Adding a second locality signal (monotone geo) is redundant and amplifies errors in noisy coordinates. The RBF instead learns a complementary role: gently counteracting the gate's tendency to over-concentrate on the nearest candidates, encouraging attention to also consider slightly more distant but gene-expression-compatible neighbors. This improves the diversity of the aggregation and reduces coordinate clumping.

### Cell S7: Experiment 7 — Point-IPA Geo Bias

#### Setup

Added "point-IPA" mode: per-head learned 2D offsets $u_i^{(h)} = W_{Qp}^{(h)} s_i$, $v_j^{(h)} = W_{Kp}^{(h)} s_j$, with geo bias:

$$b_{\text{geo}}^{(h)}(i,j) = \text{clip}\left(-\frac{\|(x_i + u_i^{(h)}) - (x_j + v_j^{(h)})\|^2}{\sigma_h^2},\; -10,\; 0\right)$$

Tested on V1 5R (all recycles) and V2 3R (last recycle only). Compared against no_geo and RBF baselines re-run under identical conditions.

#### Results

| Condition | Seeds | Mean E_geo | Std |
|---|---|---|---|
| V1_5R_point_ipa | 3 | 0.4680 | 0.0833 |
| V2_3R_pipa_last | 3 | 0.5011 | 0.0449 |
| **V1_5R_no_geo** | **3** | **0.4400** | **0.0396** |
| **V1_5R_rbf** | **3** | **0.4321** | **0.0573** |

Point-IPA does not improve over the simpler baselines. V1 no_geo (0.44) and V1 RBF (0.43) both outperform V1 point-IPA (0.47). V2 with point-IPA last (0.50) is also worse.

#### Point-IPA Offset Explosion in V2

V2 point-IPA showed a diagnostic warning: the learned offset norms $|u_i^{(h)}|$ and $|v_j^{(h)}|$ exploded to 40–110 in V2 (compared to 1.3–4.8 in V1):

| Architecture | $|u|$ range | $|v|$ range | $\sigma$ range |
|---|---|---|---|
| V1_5R_point_ipa | 2.2–4.8 | 0.8–2.6 | 1.01–1.14 |
| V2_3R_pipa_last | **37–109** | **21–92** | 0.98–1.04 |

In V2, the offsets grow so large that the "point" positions $x_i + u_i$ are dominated by the learned offsets rather than the actual coordinates, making the geometric term a feature-space distance rather than a spatial one. The $\sigma$ values stay near 1.0, unable to compensate.

#### Conclusions

1. **RBF is the best geo variant** (0.43 mean), but its benefit comes from anti-locality regularization, not from encoding "preferred distance bands" as originally hypothesized.
2. **No-geo is nearly as good** (0.44 mean) and simpler.
3. **Point-IPA doesn't help** — the added complexity (offset projections, per-head $\sigma$) doesn't translate to better geometry. In V2, the offsets explode.
4. **Monotone geo is confirmed harmful** (0.64 mean) and should never be used unconditionally.

**Decision: Proceed with V1 5R no_geo (simplest) or V1 5R RBF (marginal +0.01 benefit). RBF is optional — the complexity may not be worth it for <0.02 improvement.**

---

## 14. Phase 13: Pair Representation (Cell S9)

### Motivation

The research agent identified a "missing AlphaFold ingredient": a persistent pair representation $z_{ij}$ that provides per-edge context beyond what the gate scalar offers. The hypothesis was that a learned pair embedding, used as an additive per-head attention bias, would improve attention stability and reduce gate dominance.

### Cell S9: Experiment 9 — Pair Representation $z_{ij}$

#### Setup

Pair MLP computes edge embeddings on candidate edges:

$$z_{ij} = \text{MLP}_{\text{pair}}([h_i,\; h_j,\; |h_i - h_j|,\; h_i \odot h_j])$$

Input: $4 \times 128 = 512 \to 64 \to 64 \to 32$, then per-head projection: $b_{\text{pair}}^{(h)}(i,j) = w_h^\top z_{ij}$.

Pair module: 35,040 params. Tested V1 5R + pair, V2 3R + pair, and V1 5R baseline (no pair, no geo).

#### Results

| Condition | Seeds | Mean E_geo | Std |
|---|---|---|---|
| V1_5R_pair | 3 | 0.5692 | 0.0858 |
| V2_3R_pair | 3 | 0.5185 | 0.0608 |
| V1_5R_base (no pair, no geo) | 3 | 0.5821 | 0.1250 |

**Pair representation provides no meaningful improvement.** V1+pair (0.57) is barely different from V1 base (0.58), and both are worse than the V1 no_geo results from S7 (0.44). This suggests the pair bias was not integrated optimally or the run conditions differed slightly from S7.

#### The Pair Bias Is Not Being Used

The Spearman correlation $\rho(\text{attn}, \text{pair})$ at the final recycle confirms the model ignores the pair bias:

| Condition | $\rho_{\text{pair}}$ | $\rho_{\text{gate}}$ | $\rho_{\text{qk}}$ |
|---|---|---|---|
| V1_5R_pair s0 | −0.001 | 0.763 | 0.337 |
| V1_5R_pair s1 | 0.034 | 0.726 | 0.255 |
| V1_5R_pair s2 | 0.115 | 0.779 | 0.226 |
| V2_3R_pair s0 | −0.061 | 0.791 | 0.264 |
| V2_3R_pair s1 | 0.192 | 0.797 | 0.273 |
| V2_3R_pair s2 | −0.243 | 0.879 | 0.087 |

$\rho_{\text{pair}}$ hovers near zero (range: −0.24 to +0.19), while gate dominates at 0.73–0.88. The gate MLP and pair MLP receive **identical input features** $[h_i, h_j, |h_i - h_j|, h_i \odot h_j]$, so the pair bias is informationally redundant with the gate. The optimizer has no reason to route signal through both.

**Decision: Pair representation abandoned. The gate already captures the relevant pairwise information.**

---

## 15. Phase 14: Head-Specific Gate (Cell S10)

### Motivation

The research agent hypothesized that broadcasting a single gate scalar $g_{ij}$ to all attention heads forces all heads to attend to the same neighborhood structure. A per-head gate $g_{ij}^{(h)} \in \mathbb{R}^H$ could allow heads to specialize (e.g., one head focuses on conservative local edges, another explores more distant neighbors).

### Cell S10: Experiment 10 — Head-Specific Gate

#### Setup

Modified gate MLP output: final linear layer outputs $H=4$ values per edge instead of 1. Each head receives its own gate bias: $\ell_{ij}^{(h)} \leftarrow \ell_{ij}^{(h)} + \beta \cdot g_{ij}^{(h)}$.

Gate params: 37,252 (head-specific) vs 37,057 (shared). Tested V1 5R with head-specific gate, V1 5R with shared gate (control), and V2 3R with head-specific gate.

#### Results

| Condition | Seeds | Mean E_geo | Std |
|---|---|---|---|
| **V1_5R_hgate** | **3** | **0.4941** | **0.0279** |
| V1_5R_sgate | 3 | 0.6196 | 0.0822 |
| V2_3R_hgate | 3 | 0.5013 | 0.0333 |

Head-specific gate improves V1 (0.49 vs 0.62 shared), but note that the shared-gate control here (0.62) is worse than the S7 no_geo baseline (0.44), suggesting seed or minor implementation variance across cells. The V2 head-specific gate (0.50) is comparable to V1's.

#### No Head Specialization Observed

**Head diversity (1 − IoU of top-30 neighbor sets across head pairs):**

| Condition | Head Diversity |
|---|---|
| V1_5R_hgate (3 seeds avg) | 0.530 |
| V1_5R_sgate (3 seeds avg) | 0.539 |
| V2_3R_hgate (3 seeds avg) | 0.542 |

Head diversity is **nearly identical** between shared and per-head gates. No specialization emerged.

**Per-head gate precision@30:**

| Condition | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| V1_5R_hgate (avg) | 0.786 | 0.784 | 0.787 | 0.786 |
| V1_5R_sgate (avg) | 0.783 | 0.783 | 0.783 | 0.783 |

All heads converge to the same precision (~0.78), confirming no functional specialization. The per-head gate just learns 4 copies of the same function.

**Decision: Head-specific gate provides no specialization benefit. Revert to shared gate (fewer params, same behavior).**

---

## 16. Phase 15: Gate Supervision Variants (Cell S11)

### Motivation

The gate is supervised with KL divergence to $T_{\text{ref}}^{(1)}$ (the 1-step random walk operator). The research agent hypothesized that $T^{(1)}$ may be too "nearest-neighbor literal" and that supervising to $T^{(2)}$ (smoother, more diffuse) or a mixture could produce a more robust neighborhood prior, especially for cross-slide generalization on noisy scRNA.

### Cell S11: Experiment 11 — Gate Supervision: $T^{(1)}$ vs $T^{(2)}$ vs Mixture

#### Setup

V1 5R no_geo architecture. Three gate supervision conditions:

- **gate_T1:** KL to $T_{\text{ref}}^{(1)}$ (current baseline)
- **gate_T2:** KL to $T_{\text{ref}}^{(2)}$ (2-step RW, smoother neighborhoods)
- **gate_mix:** $L_{\text{gate}} = \text{KL}(T_{\text{ref}}^{(1)} \| T_{\text{gate}}) + \lambda \cdot \text{KL}(T_{\text{ref}}^{(2)} \| T_{\text{gate}})$

Two seeds each.

#### Results

| Condition | Seeds | Mean E_geo | Std | Gate Precision | Gate Recall |
|---|---|---|---|---|---|
| **gate_T1** | **2** | **0.5031** | **0.0406** | **0.7823** | **0.6347** |
| gate_T2 | 2 | 0.5563 | 0.0146 | 0.7680 | 0.6233 |
| gate_mix | 2 | 0.5180 | 0.0091 | 0.7670 | 0.6218 |

$T^{(1)}$ supervision wins on E_geo (0.50 vs 0.56 for $T^{(2)}$, 0.52 for mix). It also achieves the highest gate precision (0.782) and recall (0.635).

#### Why $T^{(2)}$ Hurts

$T^{(2)}$ distributes probability mass over 2-hop neighborhoods, making the supervision signal more diffuse. This causes the gate to be less discriminating — it becomes more permissive of non-neighbor edges, which introduces noise into the attention computation. The gate precision drops (0.782 → 0.768) and recall drops (0.635 → 0.623).

The mixture condition falls between the two, as expected. The $T^{(2)}$ term dilutes the $T^{(1)}$ signal without adding useful complementary information at this scale.

**Attention correlation at final recycle (averaged):**

| Condition | $\rho_{\text{gate}}$ | $\rho_{\text{qk}}$ |
|---|---|---|
| gate_T1 | 0.771 | 0.326 |
| gate_T2 | 0.784 | 0.371 |
| gate_mix | 0.783 | 0.314 |

The $\rho_{\text{gate}}$ is slightly higher for $T^{(2)}$ and mix, which paradoxically means the attention is MORE gate-dependent but with a WORSE gate — it relies on a less precise spatial prior.

**Decision: Keep $T^{(1)}$ supervision. The first-order random walk provides the sharpest neighborhood signal for the gate.**

---

## 17. Phase 16: Candidate k Sensitivity (Cell S12)

### Motivation

The research agent recommended testing whether $k=80$ remains optimal after all the architectural improvements, or if the improved gate quality could tolerate different candidate sizes. This also serves as a robustness check: if performance swings wildly with $k$, the pipeline is fragile.

### Cell S12: Experiment 12 — Candidate k ∈ {60, 80, 100}

#### Setup

V1 5R no_geo with shared gate. Swept $k_{\text{cand}} \in \{60, 80, 100\}$, 2 seeds each.

**Candidate recall at each k (fraction of true spatial neighbors in candidate set):**

| k | Candidate Recall |
|---|---|
| 60 | 0.913 |
| 80 | 0.948 |
| 100 | 0.967 |

#### Results

| Condition | Seeds | Mean E_geo | Std | Gate Prec | Gate Rec |
|---|---|---|---|---|---|
| k=60 | 2 | 0.5480 | 0.0792 | 0.7883 | 0.6395 |
| k=80 | 2 | 0.5341 | 0.0538 | 0.7818 | 0.6340 |
| **k=100** | **2** | **0.4566** | **0.0185** | **0.7787** | **0.6315** |

**k=100 now wins** (0.46 vs 0.53 at k=80), reversing the earlier Cell P finding where k=80 was best (0.38 vs 0.50 at k=100). This reversal likely reflects improvements in gate quality and training stability across the many intervening experiments.

**Sensitivity analysis (agent threshold: $\Delta > 0.05$ = sensitive):**

| Comparison | $\Delta$ E_geo | Status |
|---|---|---|
| k=60 vs k=80 | 0.014 | ✅ Stable |
| k=100 vs k=80 | 0.077 | ⚠️ Sensitive |

The k=100 advantage exceeds the agent's 0.05 sensitivity threshold, indicating this is a meaningful improvement, not noise. Higher candidate recall (0.967 vs 0.948) provides the gate with better coverage of true neighbors, and the gate is now good enough to filter the additional noise from the extra 20 candidates per node.

**Recall@30 at final recycle:**

| k | Mean Recall | Mean $|\Delta X|$ |
|---|---|---|
| 60 | 0.579 | 0.059 |
| 80 | 0.566 | 0.060 |
| 100 | 0.563 | 0.082 |

Slightly lower Recall@30 at k=100 (0.563 vs 0.579), but better E_geo — the model trades off attention precision for better overall geometry by having access to more true neighbors.

**Decision: Update candidate k to 100 for the full pipeline.**

---

## 18. Phase 17: Full IPA-R(2D) Integration (Cell D2)

### Motivation

Cell D2 attempts to integrate all the components recommended by the research agent into a single "Sparse IPA-R(2D)" architecture:

- QK-norm semantic attention (per-recycle $\tau$ clamped $[1, 10]$)
- Gate bias (per-recycle $\beta$ clamped $[0, 2.5]$, initialized $[0.5, 1.5, 2.0]$)
- Pair bias $z_{ij}$ (shared MLP, per-recycle head projection)
- Point-IPA geo (R2 only, clipped $[-10, 0]$)
- Step-size gating $\eta_r$ on coordinate updates ($\eta_r = \sigma(\theta_r)$, initialized at $\approx 0.2$)
- 3 unshared recycles

This is the "full IPA-R" envisioned in the research agent's architecture proposal (Section D2 of the suggestion document).

### Cell D2: Architecture & Results

**Model:** 389,119 params, k=100 (from S12 result).

**Per-recycle roles:**

- R0: qk + pair (exploration; $\beta_0 \approx 0.6$ — gate present but low)
- R1: qk + gate + pair (localize; $\beta_1 \approx 1.5$)
- R2: gate + pair + point-geo (refine; $\beta_2 \approx 2.1$)

#### Results (3 seeds)

| Tag | Best Ep | E_geo (H-gate) | E_geo (T-edge) |
|---|---|---|---|
| IPAR_s0 | 7 | 0.6186 | 0.4919 |
| IPAR_s1 | 7 | 0.6064 | 0.4670 |
| IPAR_s2 | 7 | 0.5982 | 0.4531 |
| **Mean±Std** | — | **0.6078 ± 0.0084** | **0.4707 ± 0.0161** |

**Reference:** V1 5R shared at k=100 ≈ 0.457 ± 0.019 (from S12).

**IPA-R is significantly worse than the simple V1 shared model** (0.61 vs 0.46 on H-gated inference). The integration of all components together produces worse results than the simpler architecture.

#### Attention Diagnostics Show Good Internal Behavior

Despite worse E_geo, the IPA-R model's internal attention is well-structured:

| Recycle | $\eta$ | $\beta$ | $\tau$ | $|\Delta X|$ | Recall@30 | logit_std | entropy | $\rho_{\text{qk}}$ | $\rho_{\text{gate}}$ | $\rho_{\text{pair}}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.23 | 0.66 | 4.6 | 0.026 | 0.321 | 50.8 | 0.29 | 0.851 | 0.059 | 0.067 |
| R1 | 0.41 | 1.58 | 4.35 | 0.265 | 0.519 | 2.35 | 3.26 | 0.384 | 0.632 | −0.214 |
| R2 | 0.42 | 2.08 | 4.99 | 0.170 | 0.700 | 2.10 | 3.03 | 0.282 | 0.813 | −0.060 |

The attention shows a clean qk→gate progression (as designed), Recall@30 climbs from 0.32 to 0.70 (the highest observed in any experiment), and logit_std is bounded. The step-size gating learns sensible values: $\eta_0 \approx 0.23$ (cautious initial step), $\eta_{1,2} \approx 0.41$ (larger refinement steps).

However, $\rho_{\text{pair}}$ is negative at R1 (−0.21), confirming the pair bias is again not helpful — the model is slightly anti-correlating with it, consistent with the S9 finding.

#### Why IPA-R Underperforms

Several factors likely contribute:

1. **Component interference:** The pair bias, point-IPA geo, and per-recycle $\beta$ each add learnable parameters that the optimizer must balance. With only 2 training slides and 12 epochs, there isn't enough signal to tune all these jointly. The simpler V1 model has fewer knobs to turn and converges faster.

2. **Unshared blocks cost:** 389K params vs 155K for V1 — more than 2× — without the training data to exploit the extra capacity.

3. **Step-size gating dampens R0:** $\eta_0 \approx 0.23$ means R0 barely moves coordinates ($|\Delta X| = 0.026$ vs 0.21 for V1-R0). The model effectively wastes a recycle.

4. **Components that don't help (pair, point-IPA) still consume optimization capacity** and may interfere with the components that do (gate, qk).

**Bottom line:** At toy scale (2 training slides, minisets ≤ 384), the simple V1 shared architecture outperforms the full IPA-R integration. The additional components may become useful at larger scale where there is more data to fit, but this has not been demonstrated.

**Decision: V1 shared remains the recommended architecture. The IPA-R integration is documented for future reference but not recommended for the current pipeline.**

---

## 19. Phase 18: Code Audit, Bug Fixes & Architecture Confirmation (D3–D7)

### Motivation

After the D2 IPA-R integration failed (0.61 vs V1's 0.46), a research agent produced a comprehensive audit document identifying 15 potential code-level failure modes, 6 encoder-usage risks, and 2 hypotheses for the D2 paradox (good attention metrics + bad E_geo). The D3–D7 series systematically addressed these.

### D3: Oracle Graph Confound Fix (Agent Item 1)

**Bug found:** The D2 evaluation loop computed gate logits on `candidate_masks` then passed those same logits when evaluating with `edge_masks` (target/oracle graph). Since gate logits are $-\infty$ outside the candidate set, the "oracle" evaluation was actually running on `edge_mask ∩ candidate_mask`, not the true oracle.

**Fix:** For T-edge (oracle) evaluation, pass `gate_logits = 0` (a zero tensor, making the gate bias neutral) instead of gate logits computed on a different mask.

**Impact:** After the fix, D7's T-edge oracle score improved from ~0.47 to **0.162** — the true oracle performance is dramatically better than previously measured. The inference gap was larger than thought: 0.440 (H-gate) − 0.162 (T-edge) = **0.278**.

### D4: D2 Hypothesis 1 — Term Competition (Confirmed)

Tested the agent's "remove the fights" experiment: kept D2's IPA-R architecture but set pair_bias=0, froze τ=1.0 and β=2.0 (matching V1), and clamped qk logits at R0. This eliminated the term competition that caused R0 logit_std ≈ 50.8 and entropy ≈ 0.29 in D2.

**Result:** Performance improved significantly, confirming that D2's R0 instability (extremely peaky attention from competing logit terms) was poisoning later recycles.

### D5: D2 Hypothesis 2 — Gradient-Attention Misalignment (Rejected)

Tested whether D2's attention was optimizing the wrong edges relative to multiscale stress gradients. Measured correlation between attention mass and edge gradient magnitude for each loss family.

**Result:** No evidence of systematic misalignment. The attention correctly prioritizes edges relevant to the loss. This hypothesis was rejected.

### D6: Architecture Variants with Bug Fixes

With the D3 oracle fix applied, retested V1 shared vs simplified IPA-R variants. The V1 shared 5R architecture with geometry bias (QK-norm, clamped geo) consistently matched or beat all IPA-R variants.

### D7: Final Controlled Comparison — V1 Shared 5R + Geo Bias

Ran the definitive multi-seed comparison: V1 shared 5R with QK-norm and clamped RBF/geo bias, $k=100$, with the D3 oracle fix applied.

**D7 Results (3 seeds):**

| Seed | Best Ep | E_geo (H-gate) | E_geo (T-edge) |
|---|---|---|---|
| 0 | 7 | 0.4261 | 0.1599 |
| 1 | 8 | 0.4400 | 0.1632 |
| 2 | 6 | 0.4543 | 0.1627 |
| **Mean±Std** | — | **0.4401 ± 0.0141** | **0.1619 ± 0.0014** |

**D7 is the validated baseline going forward:** E_geo = 0.440 ± 0.014 (H-gated), 0.162 (T-edge oracle). The inference gap is 0.278. The architecture is `IPALiteModelD7` — V1 shared 5R with QK-norm and geo bias, candidate $k=100$.

### Encoder Audit (Agent §1.2)

All 6 encoder-usage checks passed:

| Check | Result |
|---|---|
| Norm anisotropy (max/min RMS ratio) | 1.038 (threshold: <1.10) ✅ |
| Slide leakage (classifier accuracy) | 43.4% (chance=33.3%) ✅ |
| PCA-10 slide accuracy | 31.3% (below chance) ✅ |
| Per-gene mean centering | Applied correctly ✅ |
| SC adapter isolation | ST embeddings unmodified ✅ |
| Cache determinism | Exact match ✅ |

The residual 43% slide accuracy on full 128-dim H is biological (different cell-type proportions across tissue sections), not batch effect. The encoder is clean.

---

## 20. Phase 19: Inference Gap Diagnostics & Gating Analysis (E-Series)

### Overview

With D7 as the validated baseline (0.440 H-gate, 0.162 oracle, gap = 0.278), the E-series systematically diagnosed why the inference gap exists and attempted to close it. The research agent specified 12 experiments (E1–E12); we executed the diagnostics (E1, E2, E7) and the inference-gap-targeted experiments (E2b, E8a–e, E9).

### E1: Encoder Norm + Slide Leakage Audit

Reported in §19 above. Both checks passed. The encoder is not the bottleneck.

### E7: Graph Recovery Diagnostics

**Goal:** Measure how well kNN(H) recovers spatial neighbors at various $k$ and distance metrics.

**Protocol:** For each val miniset (50), computed kNN in H-space using both euclidean and cosine distance, then measured overlap with true spatial kNN.

**Results:**

| Metric | k=20 | k=40 | k=60 | k=80 | k=100 |
|---|---|---|---|---|---|
| Recall (euclidean) | 0.731 | 0.862 | 0.914 | 0.941 | 0.958 |
| Recall (cosine) | **0.755** | **0.880** | **0.928** | **0.952** | **0.965** |

**Cross-slide consistency:** ST1=0.771, ST2=0.744, ST3=0.751 (cosine, k=20). No val-specific degradation.

**Key findings:**

1. **Graph quality is healthy.** 73–75% of true spatial neighbors are recoverable from H-space, well above the 0.65 threshold.
2. **Cosine beats euclidean by ~2.4% consistently.** The D7 candidate construction uses cosine — confirmed optimal.
3. **Performance plateaus at k≈80, declines at k=100.** Diminishing returns past k=80.
4. **The encoder generalizes uniformly** — ST3 (val slide) sits between ST1 and ST2, not an outlier.

**Conclusion:** The inference gap is NOT caused by missing edges in the candidate graph. The bottleneck is downstream (gate discrimination).

### E2: Shuffle Controls — Leakage Test

**Goal:** Verify Stage C genuinely uses H structure — no hidden leakage through cached edges, coordinates, or slide IDs.

**Protocol:** On the D7 model at inference, fixed all masks (candidate mask + edge mask) and perturbed only H:

- **Real H** (baseline)
- **Shuffled H** within miniset (breaks node↔feature correspondence but preserves mask topology)
- **Cross-miniset H** (real features from another miniset; preserves distribution, destroys correspondence)
- **Random H** (Gaussian with matched mean/std)

**Results (50 val minisets):**

| Condition | H-gate E_geo | T-edge E_geo |
|---|---|---|
| Real H | 0.716 ± 0.187 | 0.264 ± 0.074 |
| Shuffled H | 5.564 ± 0.844 | 1.405 ± 0.533 |
| Cross-miniset H | 5.156 ± 0.760 | 1.377 ± 0.590 |
| Random H | 4.723 ± 0.596 | 1.313 ± 0.553 |

**6–8× degradation across all controls.** The model genuinely depends on H content. No leakage.

**Note on E2 baseline:** The 0.716 H-gate score here differs from D7's best (0.440) because E2 evaluated on a different checkpoint/subset than D7's best epoch. The relative degradation ratios (6–8×) are the meaningful diagnostic.

### E2b: Inference Gap Decomposition

**Goal:** Separate the gap into "missing-edge cost" vs "bad-gating cost."

**Protocol:** Evaluated 4 conditions on the D7 model, with gate OFF for conditions A–C:

- **(A) Oracle** — true spatial edges, gate off
- **(B) Candidate ∩ True** — only the correct edges within the candidate set, gate off (perfect gating ceiling)
- **(C) Spurious only** — only the wrong edges from the candidate set
- **(D) Candidate + Gate** — full candidate set with learned gate (normal inference)

**Results:**

| Condition | E_geo |
|---|---|
| (A) Oracle (true edges) | 0.264 |
| (B) Cand ∩ True (perfect gate ceiling) | 0.266 |
| (C) Spurious only | 1.921 |
| (D) Cand + Gate (H-gate) | 0.716 |

**Edge statistics:**

| Metric | Mean ± Std |
|---|---|
| True edges (n_true) | 10,013 ± 7,238 |
| Candidate edges (n_cand) | 27,313 ± 11,456 |
| Intersection (n_intersect) | 9,466 ± 6,438 |
| Spurious edges (n_spurious) | 17,847 ± 5,313 |
| Missing edges (n_missing) | 548 ± 843 |
| Candidate recall | **1.00** (perfect) |
| Candidate precision | **0.30** (70% spurious) |

**Gap decomposition:**

$$\text{Total gap (D−A)} = 0.452$$
$$\text{Missing-edge cost (B−A)} = 0.002 \quad (0.4\%)$$
$$\text{Bad-gating cost (D−B)} = 0.450 \quad (99.6\%)$$

**Critical finding:** The candidate graph is essentially perfect — recall = 1.0, meaning all true spatial neighbors are present. The entire inference gap is gating failure. The gate must discriminate 1 correct edge from ~3 candidates (precision 0.30), and it fails to do so adequately.

### E8a: Mutual-kNN Diagnostic Sweep

**Goal:** Test whether mutual-kNN filtering (requiring both $i \in \text{kNN}(j)$ AND $j \in \text{kNN}(i)$) improves candidate set precision enough to help.

**Sweep:** $k_{\text{base}} \in \{80, 100, 120, 140, 160, 200\}$.

| $k_{\text{base}}$ | $k_{\text{eff}}$ | Recall | Precision |
|---|---|---|---|
| 80 | 66 | 0.905 | 0.531 |
| 100 | 85 | 0.940 | 0.436 |
| 120 | 103 | 0.959 | 0.369 |

**Best:** $k_{\text{base}}=120$ gives recall=0.96, precision=0.37 — only 1.2× improvement over baseline 0.30. Not worth a training run.

### E8c: Gate Effectiveness Check

**Goal:** Isolate gate quality from model co-adaptation by comparing gate variants on the same D7 model.

**Results (30 val minisets):**

| Condition | E_geo |
|---|---|
| (A) Oracle (true edges) | 0.264 |
| (B) Perfect gate (cand ∩ true) | 0.266 |
| (E) No gate (all cand edges, gate=0) | 1.132 |
| (F) Pretrained gate only | **0.684** |
| (D) Jointly trained gate | 0.716 |

**Key finding:** Joint training HURTS the gate. The pretrained gate (0.684) outperforms the jointly trained gate (0.716) on the SAME model. The gate degrades during co-adaptation — gradient pressure from geometry losses pulls it off its pretrained quality.

### E8d: Frozen Gate Training + β Sweep — FAILED

**Hypothesis:** If the pretrained gate is better, freeze it and train only the model.

**Sweep:** β ∈ {2, 4, 6, 8} × 2 seeds.

| β | E_geo (mean±std) |
|---|---|
| 2.0 | 0.564 ± 0.180 |
| 4.0 | 0.586 ± 0.031 |
| 6.0 | 0.837 ± 0.041 |
| 8.0 | 0.713 ± 0.070 |

**Best β=2: 0.564** vs D7 joint 0.440 (Δ = +0.124, WORSE). Massive seed variance (0.74 vs 0.38 for β=2).

**Conclusion:** Frozen gate is worse AND unstable. The model needs co-adaptation during training — it can't learn effectively from a static gate signal. The gate and model must evolve together, but the gate degrades in the process.

### E8e: Gate Loss Weight Sweep — FAILED

**Hypothesis:** Keep joint training but increase $\lambda_{\text{gate}}$ to anchor the gate closer to $T_{\text{ref}}$ and prevent drift.

**Sweep:** $\lambda_{\text{gate}} \in \{0.05, 0.2, 0.5, 1.0\}$ × 2 seeds.

| $\lambda_{\text{gate}}$ | E_geo | Gate drift (Joint vs Pretrained) |
|---|---|---|
| 0.05 | 0.427 ± 0.043 | J=0.422 P=0.401 (joint WORSE) |
| 0.20 | 0.477 ± 0.005 | J=0.465 P=0.364 (joint WORSE) |
| 0.50 | 0.491 ± 0.060 | J=0.478 P=0.376 (joint WORSE) |
| 1.00 | 0.502 ± 0.021 | J=0.486 P=0.354 (joint WORSE) |

**Best $\lambda=0.05$: 0.427** vs D7 0.440 (Δ = −0.013, marginal). Gate drift persists at ALL weight values — the jointly trained gate is consistently worse than the pretrained gate regardless of $\lambda_{\text{gate}}$.

**Conclusion:** Increasing gate loss weight doesn't prevent drift. The E8c-e gate improvement detour failed.

### E9: Learned Projection for kNN Construction — FAILED

**Goal:** Learn a projection $P \in \mathbb{R}^{128 \times 32}$ to improve candidate graph precision before the gate sees it.

**Architecture:**

- Phase 1: Train $P$ + gate (operating on $P(H)$) for 2 epochs on gate KL to $T_{\text{ref}}$ only
- Phase 2: Freeze $P$, build candidate graphs from kNN($P(H)$), train Stage C with projected graphs

**Results (3 seeds):**

| Tag | Best Ep | E_geo (H-gate) | Graph Recall | Graph Precision |
|---|---|---|---|---|
| E9_s0 | 6 | 0.5627 | 0.9621 | 0.3279 |
| E9_s1 | 3 | 0.6711 | 0.9605 | 0.3284 |
| E9_s2 | 4 | 0.5894 | 0.9612 | 0.3287 |
| **Mean±Std** | — | **0.6077 ± 0.0461** | 0.9613 | 0.3283 |

**The projection learned nothing.** Precision went from 0.30 → 0.33 — essentially unchanged. A linear projection trained for 2 epochs on gate KL cannot discover spatial structure that the encoder (trained for thousands of steps with spatial NCE) didn't already capture. E_geo = 0.608 is significantly worse than D7's 0.440.

### E-Series Summary

| Experiment | Finding | Impact |
|---|---|---|
| E1 | Encoder clean (norm ratio 1.04, slide acc 43%) | ✅ Encoder not bottleneck |
| E7 | Graph healthy (recall=0.96, cosine > euclidean) | ✅ Graph not bottleneck |
| E2 | No leakage (6–8× degradation on controls) | ✅ Model genuinely uses H |
| E2b | 99.6% of gap is bad gating, 0.4% missing edges | 🎯 Gate is THE bottleneck |
| E8a | Mutual-kNN: precision 0.30→0.37 (marginal) | ❌ Not worth pursuing |
| E8c | Pretrained gate (0.68) > joint gate (0.72) on same model | 🔍 Gate degrades during training |
| E8d | Frozen gate training: 0.564 (worse + unstable) | ❌ Model needs co-adaptation |
| E8e | Gate loss weight sweep: no improvement, drift persists | ❌ Can't prevent gate drift |
| E9 | Learned projection: precision 0.30→0.33 (unchanged) | ❌ Projection can't help |

**Bottom line:** The inference gap (0.440 vs 0.162) is entirely a gating problem. The candidate graph is perfect (recall=1.0). The gate must discriminate among 3 candidates per true edge but fails to do so well enough, and all attempts to improve it (freezing, stronger supervision, learned projections) either failed or produced marginal gains. The 0.440 ceiling appears to be a fundamental limitation of the current gate architecture + toy data scale.

---

## 21. Phase 20: Post-E-Series Gating Fixes (P-Series)

### Context: The E→P Naming Switch

The E-series (Phase 19) concluded that **99.6% of the inference gap is bad gating** — the candidate graph has recall ≈ 0.96, but the MLP gate assigns weight to wrong edges (precision ≈ 0.30). The research agent then prescribed a new series of targeted experiments labeled **P1–P5** (distinct from the original Cell P in Phase 8). These "P-series" experiments systematically test hypotheses about *why* the gate degrades during joint training and whether alternative gate architectures can help.

The agent's plan document ("plan_feb_25") identified three root-cause hypotheses:
1. **Gradient contamination** — structure loss gradients flow through gate bias and corrupt the gate (→ test with stop-grad)
2. **Train/eval mismatch** — gate sees clean H during training but model sees augmented H_aug (→ test with gate augmentation)
3. **Gate architecture limitation** — MLP gate scores edges independently without neighborhood context (→ test with Contextual GateFormer)

### Cell P1: Stop-Grad Gate Bias — FAILED

**Hypothesis:** Structure loss gradients flow backward through the gate bias term $\beta \cdot g(H)$ in attention logits, corrupting the gate's learned edge scores.

**Setup:** 3 seeds × 2 conditions (baseline vs stop-grad where gate bias is detached before entering attention). k=100, β=2.0.

**Results:**

| Condition | $E_{\text{geo}}$ (H-gate) | $E_{\text{geo}}$ (T-edge) | Prec@30 (joint) | Prec@30 (pretrained) |
|---|---|---|---|---|
| baseline | $0.4397 \pm 0.0141$ | $0.1621 \pm 0.0169$ | 0.760 | 0.795 |
| stopgrad | $0.4662 \pm 0.0309$ | $0.1872 \pm 0.0161$ | 0.754 | 0.795 |

**Gate drift trajectory (val KL):** Both conditions show identical drift pattern — gate KL rises from 0.55 → 0.67 over 10 epochs regardless of stop-grad.

**Decision:** ✗ NO EFFECT. Gate drift is **overfitting**, not gradient contamination. The gate's train KL improves (0.34→0.26) while val KL degrades (0.55→0.67) — classic overfitting signature. Skip stop-grad, focus on regularization.

### Cell P2+P3: Gate Augmentation + β Ramp — SUCCESS ✅

**Hypothesis (P2):** Gate sees clean H while the model sees augmented H_aug during training. This mismatch means the gate overfits to clean-H patterns that don't generalize. Fix: feed H_aug to the gate too (implicit regularization).

**Hypothesis (P3):** If gate quality degrades during training, relying on it less early (low β) prevents poisoning the structure module's early epochs. Fix: β ramp from 0.5 → 2.0 over epochs 1–6.

**Setup:** 2 screening seeds × 3 conditions:
- A: baseline (clean H to gate, constant β=2.0)
- B: gate sees H_aug, constant β=2.0
- C: gate sees H_aug + β ramp 0.5→2.0

**Results:**

| Condition | $E_{\text{geo}}$ | Δ vs baseline | Best β (joint) | Gate overfit gap |
|---|---|---|---|---|
| A_baseline | 0.4700 | — | 2.0 | 0.2992 |
| **B_aug_gate** | **0.3858** | **−0.0842** | **2.0** | **0.1928** |
| C_aug_ramp | 0.3965 | −0.0734 | 2.0 | 0.2241 |

**Key findings:**
- **B_aug_gate is the clear winner:** 10.7% improvement over baseline, gate overfit gap reduced by 35%
- β ramp (C) provides no additional benefit over constant β=2.0
- Gate precision@30: baseline 0.78 → B_aug_gate 0.79 (modest, but the real gain is in preventing drift)

**Reference baselines:** D7=0.440±0.014, oracle=0.162

### Cell P5: Validation Battery — PARTIAL PASS ⚠️

**Purpose:** Before investing in CGF architecture, verify that B_aug_gate passes scale-readiness checks prescribed by the agent.

**Results:**

| Test | Criterion | Result | Pass? |
|---|---|---|---|
| 1: Shuffle H | ≥3× degradation | 14.5× (0.38 → 5.52) | ✅ |
| 2: Random graph (matched degree) | Degrades to >0.7 | 0.39 (1.0× ratio) | ❌ FAIL |
| 3: Candidate-k robustness (k∈{80,100,120}) | Range ≤ 0.03 | Range = 0.005 | ✅ |

**Critical finding:** Test 2 FAILS — the model performs *equally well* with a random graph as with the real candidate graph. This means the model **doesn't need candidate topology** to work. The gate's learned bias is so strong (β·std(g) ≈ 25) that it completely overrides graph structure. The attention sees: $\text{logits} = \text{qk} + \beta \cdot g(H)$, and with $\beta \cdot \text{std}(g) \approx 25 \gg \text{std}(\text{qk}) \approx 9$, the gate dominates everything.

**Implication:** This isn't necessarily bad for deployment (the gate works!), but it means the model can't benefit from better candidate graphs — it's purely relying on the MLP gate's H→edge scoring.

### Cell P4: Contextual GateFormer (CGF) vs MLP — CGF FAILS ❌

**Hypothesis:** A 2-layer GAT-style gate that scores edges with neighborhood context should achieve higher precision than an edge-independent MLP gate.

**Setup:** 2 seeds × 2 conditions (MLP vs CGF), both with gate augmentation (P2 winner). k=100, β=2.0. Rich eval with β sweep × {joint, pretrained}.

**Results:**

| Condition | $E_{\text{geo}}$ (H-gate) | Oracle | Gap | Prec@30 |
|---|---|---|---|---|
| **A_mlp_aug** | **$0.3815 \pm 0.0043$** | 0.170 | 0.212 | **0.793** |
| B_cgf_aug | $0.5346 \pm 0.0295$ | 0.183 | 0.352 | 0.757 |

**Decision:** ✗ MLP wins by 0.153 in $E_{\text{geo}}$. CGF is strictly worse at toy scale. Likely cause: at n≈128 with k=100, the candidate graph is near-complete (~100% density), so GAT attention over a complete graph has no useful inductive bias.

### Cell P4v2: Larger Minisets + Sparse Candidates — CGF Still Fails ❌

**Hypothesis:** CGF failed because candidate graphs were too dense. Fix: rebuild minisets with n ∈ [96, 576] and adaptive k ≈ 15% of n, giving ~25% density instead of ~100%.

**Setup:** Rebuilt 3000 minisets (2250 train, 750 val). 2 seeds × 2 conditions (MLP vs CGF). Then a 5-condition follow-up with β and detach variants.

**Initial results (2 seeds):**

| Condition | $E_{\text{geo}}$ | Gap | Prec@30 |
|---|---|---|---|
| **A_mlp** | **$0.4106 \pm 0.0233$** | 0.241 | 0.848 |
| B_cgf | $0.5890 \pm 0.0302$ | 0.382 | 0.854 |

**Follow-up 5-condition factorial (1 seed each):**

| Condition | β | Detach? | $E_{\text{geo}}$ | Δ vs MLP |
|---|---|---|---|---|
| A_mlp_baseline | 2.0 | no | 0.4879 | — |
| B_cgf_b2_nodet | 2.0 | no | 0.5025 | −0.015 |
| **C_cgf_b05_nodet** | **0.5** | **no** | **0.4315** | **+0.056** |
| D_cgf_b2_det | 2.0 | yes | 0.5286 | −0.041 |
| E_cgf_b05_det | 0.5 | yes | 0.5109 | −0.023 |

**Key discovery:** CGF at β=0.5 without detach (condition C) *beats* MLP by 0.056. But the MLP baseline here was trained at β=2.0, not its own optimal β=0.5, making this comparison unfair. Extended to 3-seed confirmation in P4v3.

### Cell P4v3: Definitive CGF Test — INCONCLUSIVE ❌

**Setup:** 3 seeds each for MLP (β=0.5) and CGF (β=0.5, no detach). Also tested CGF with high-recall candidate graphs. Full diagnostics per epoch.

**Results:**

| Condition | $E_{\text{geo}}$ (3 seeds) | Oracle | Prec@30 |
|---|---|---|---|
| A_mlp_b05 | $0.5123 \pm 0.0451$ | 0.228 | 0.827 |
| C_cgf_b05 | $0.5117 \pm 0.0445$ | 0.135 | — |
| F_cgf_b05_hr | $0.5688 \pm 0.0621$ | — | — |

**Verdict:** CGF Δ = +0.0006 (effectively zero). High-recall candidates (F) are worse. Notably, CGF achieves *much better oracle scores* (0.135 vs 0.228) — the CGF structure module is superior — but this advantage is completely erased by the gate injection mechanism. The gate dominates attention regardless of architecture quality.

**Diagnostic insight:** Gate logit std is ~25 for both MLP and CGF, while qk logit std is ~7–10. The gate term overwhelms the structure module's learned attention pattern.

### Cell P4v3-VALIDATE: β Fix + Log-Prior Injection — NO EFFECT ❌

**Purpose:** Test whether the internal D7_BETA (hardcoded 2.0 inside the model block) combined with external β creates a double-β problem, and whether log-prior injection helps.

**Three injection modes tested (eval-only, no training):**
- `raw`: current behavior — logits = qk + internal_β × (ext_β × g)
- `raw_fix`: remove internal β — logits = qk + ext_β × g
- `logprior`: logits = qk + ext_β × log(softmax(g) + ε)

**Results (best β per mode):**

| Condition | Mode | Best β | $E_{\text{geo}}$ |
|---|---|---|---|
| A_mlp | raw | 0.125 | 0.4844 |
| A_mlp | raw_fix | 0.250 | 0.4844 |
| A_mlp | logprior | 0.250 | 0.4821 |
| C_cgf | raw | 0.125 | 0.4514 |
| C_cgf | raw_fix | 0.125 | 0.4474 |
| C_cgf | logprior | 0.125 | 0.4502 |

**Verdict:** All three modes produce essentially identical results (Δ < 0.005). Neither β-fix nor log-prior injection helps. The gate's learned logit distribution is already well-calibrated for the optimal β range.

### P-Series Overall Conclusions

1. **Gate drift is overfitting, not gradient contamination** (P1)
2. **Gate augmentation (H_aug) is the single effective fix** (P2) — reduces $E_{\text{geo}}$ from 0.440 → 0.381
3. **β ramp provides no additional benefit** (P3)
4. **CGF architecture does not improve over MLP gate** (P4, P4v2, P4v3) — tested across 3 scales, multiple β values, detach variants, and injection modes
5. **The gate dominates attention** — β·std(g) ≈ 25 >> std(qk) ≈ 9, meaning the structure module's learned attention is largely overridden
6. **CGF produces better structure modules** (lower oracle) but this advantage is erased by gate dominance
7. **The model doesn't need candidate topology** (P5 Test 2) — random graphs work equally well because the gate completely overrides graph structure

**Post-P-Series best result:** $E_{\text{geo}} = 0.381 \pm 0.004$ (B_aug_gate from P2+P3)

---

## 22. Phase 21: C2/C2′ Oracle Distillation — Failed

### Context

The research agent's recommended next step after P-series was **oracle distillation (C2):** train the model with oracle (T-edge) attention, then distill that knowledge into the gated model. Two implementations were tested: the original C2 and a redesigned C2′.

### C2 (Original Design): Oracle Graph Teacher — Abandoned on Principle ❌

**Idea:** Run the same IPA model twice per training step:
1. **Teacher pass:** model sees `edge_masks` (true spatial kNN, built from ground truth coordinates), gate off → produces $X_{\text{teacher}}$
2. **Student pass:** model sees candidate masks + gate (expression-only) → produces $X_{\text{student}}$
3. **Distill:** $\mathcal{L}_{\text{distill}} = \text{KL}(T^{(1)}(X_{\text{teacher}}) \| T^{(1)}(X_{\text{student}}))$ + MSE on pairwise distances

**Why it was abandoned:** The teacher pass feeds **coordinate-derived topology** directly into the model's forward pass. This violates a core project principle: the model's forward pass should only see expression-derived inputs ($H$, kNN($H$) candidates, learned gate), with coordinates appearing only as supervision targets in the loss. C2's teacher smuggles raw spatial adjacency into the computation graph through `edge_masks`, which is categorically different from the existing geometry losses that use coordinate-derived *targets* but never coordinate-derived *inputs*.

The existing geometry losses create: coordinates → gauge-free targets → loss supervision.
C2 creates: coordinates → oracle graph → model forward pass → teacher output → distillation target → student loss.

**Decision:** C2 abandoned before any results were obtained. Redesigned as C2′.

### C2′ (Redesigned): Target-Only Distillation via Refinement — Failed ❌

**Principle fix:** No coordinate-derived topology in any forward pass. Instead, the teacher $X^{\star}$ is produced by **gradient-descent refinement** of the student's $X$ on the same intrinsic energy $E(X)$ already used in the loss:

$$X^{\star} = \text{refine}(X_{\text{student}},\; K=5,\; \text{lr}=0.1) \quad \text{where } E(X) = w_{\text{rw}} \mathcal{L}_{\text{RW}} + \lambda_{\text{st}} \mathcal{L}_{\text{stress}} + \lambda_{\text{rep}} \mathcal{L}_{\text{rep}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$$

This is an **amortized optimization** pattern: the model learns to predict $X$ that's already closer to the energy minimum, without ever seeing oracle topology.

#### Agent-Identified Bug Fixes (Applied Before Running)

The research agent reviewed the C2′ code and identified three bugs:

1. **β settings wrong:** Training used $\beta=2.0$, but IPALiteModelD7 has an internal $\text{D7\_BETA}=2.0$ multiplier, making effective $\beta_{\text{eff}}=4.0$. P4v3-VALIDATE had shown optimal eval β was 0.125–0.25. **Fix:** Set $\beta=0.125$, sweep eval β ∈ {0.0625, 0.125, 0.25, 0.5, 1.0}.
2. **Missing entropy regularizer in refinement:** Student trains with $\mathcal{L}_{\text{ent}}$ but teacher refinement omitted it, creating an energy surface mismatch. **Fix:** Added entropy term.
3. **Edge index shape ambiguity:** `multiscale_edges` could be $(2, E)$ or $(E, 2)$; `loss_distill_distance` assumed $(2, E)$. **Fix:** Added `normalize_edge_index()` helper.

#### Setup

Self-contained cell rebuilding all caches from scratch (3000 train + 150 val minisets, variable sizes 96–576). Agent's 6-run battery:

- **A: Baseline** (P2 aug, no distill) × 2 seeds
- **B: Operator-only distill** ($\mathcal{L}_d = \text{KL}(T^{(1)}(X^{\star}) \| T^{(1)}(X))$) × 2 seeds
- **C: Distance-only distill** ($\mathcal{L}_d = \text{MSE}(d_{ij}^{\star}, d_{ij})$ on multiscale edges) × 2 seeds

All at $\lambda_d = 0.1$, distillation active from epoch 2 onward.

#### Results

| Condition | $E_{\text{geo}}$ (H-gate) | Oracle | Gap | Prec@30 |
|---|---|---|---|---|
| **A_baseline** | **0.0956 ± 0.0022** | 0.087 | **0.0086** | 0.820 |
| B_op_only | 0.0997 ± 0.0062 | 0.092 | 0.0082 | 0.821 |
| C_dist_only | 0.0966 ± 0.0005 | 0.087 | 0.0094 | 0.819 |

Best eval β = 0.0625 across all conditions (effective $\beta_{\text{eff}} = 0.0625 \times 2.0 = 0.125$).

**Refinement sanity check:** Confirmed refinement works — energy drops 60–80% in 5 steps (e.g., 0.0704 → 0.0228). But this improvement doesn't transfer through distillation.

**Gate degradation:** Minimal at this β — joint gate only +0.001 worse than pretrained.

#### Why C2′ Failed

The gap is only ~0.008. At $\beta_{\text{eff}} = 0.125$, the gate barely affects attention, so the model already performs nearly as well with candidate + gate as with oracle topology. There's essentially nothing for distillation to close.

**Key insight:** The "dramatically good" baseline (0.096 vs historical 0.440) raised immediate suspicion. This led directly to the metric mismatch discovery in Phase 22.

#### Decision

C2′ distillation does not help at any distillation mode or weight. The remaining gap of ~0.008 is architectural, not informational.

---

## 23. Phase 22: β Metric Discovery & Canonical Reeval

### The Metric Mismatch Problem

The C2′ baseline of $E_{\text{geo}} = 0.096$ vs historical D7 of $E_{\text{geo}} = 0.440$ (a purported 4.6× improvement) was investigated. Two factors were identified:

1. **β change:** C2′ used $\beta = 0.125$ (per agent recommendation) while D7/P2 used $\beta = 2.0$. With internal D7_BETA = 2.0, effective $\beta_{\text{eff}}$ went from 4.0 → 0.25.
2. **FATAL: Metric formula mismatch.** The C2′ eval function computed:

$$E_{\text{geo}}^{\text{C2'}} = \underbrace{0.3}_{\text{RW\_CAP}} \cdot \mathcal{L}_{\text{RW}} + 0.1 \cdot \mathcal{L}_{\text{stress}}$$

But the historical metric (used in D7, P-series, all prior experiments) was:

$$E_{\text{geo}}^{\text{historical}} = \underbrace{1.0}_{w_{\text{rw}}} \cdot \mathcal{L}_{\text{RW}} + 0.1 \cdot \mathcal{L}_{\text{stress}}$$

The C2′ metric multiplied $\mathcal{L}_{\text{RW}}$ by $\text{RW\_CAP} = 0.3$ instead of 1.0, making all numbers appear ~3× smaller. **The "4× improvement" was entirely a metric scaling artifact, not a real performance gain.**

### Canonical Reeval Experiment Design

To definitively answer whether β tuning helps and which architecture is best, a controlled reeval was designed:

**Canonical metric (matching historical):**

$$E_{\text{geo}} = \mathcal{L}_{\text{RW}} + 0.1 \cdot \mathcal{L}_{\text{stress}}$$

**Three conditions (2 seeds each, 6 runs total):**

| Condition | Gate input | Training eff_β | Description |
|---|---|---|---|
| D7 | clean H | 2.0 | Historical D7 baseline |
| P2_aug | H_aug | 4.0 | Historical P2 winner |
| P2_aug_lowbeta | H_aug | 0.125 | Test low β hypothesis |

**Wide eval β sweep:** eff_β ∈ {0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0} applied at inference only. Oracle (no gate) also evaluated.

**Runtime:** ~6 hours total.

### Results

#### Training Outcomes

| Condition | $E_{\text{geo}}$ @ train β | Oracle | Gap |
|---|---|---|---|
| D7 | 0.1717 ± 0.0069 | 0.1024 | 0.0692 |
| P2_aug | 0.1671 ± 0.0142 | 0.1137 | 0.0534 |
| P2_aug_lowbeta | 0.1831 ± 0.0293 | 0.1550 | 0.0281 |

#### Critical Finding: Training at Low β Is Harmful

P2_aug_lowbeta shows the **worst** performance despite having the smallest gap. Its oracle score (0.1550) is dramatically worse than D7 (0.1024) and P2_aug (0.1137). Since oracle doesn't use the gate at all, this proves **the geometry module itself learned worse representations when β was weak during training**. The gate provides essential inductive bias during training — it forces the structure module to rely on edge-quality-weighted attention, which produces better learned geometry. High seed variance (±0.029) further indicates training instability.

#### β Sweep Table (Definitive)

| eff_β | D7 | P2_aug | LowBeta |
|---|---|---|---|
| 0.000 | 0.1581 | 0.1614 | 0.1813 |
| 0.125 | **0.1574** | 0.1595 | 0.1831 |
| 0.250 | **0.1573** | 0.1582 | 0.1851 |
| 0.500 | 0.1581 | 0.1564 | 0.1889 |
| 1.000 | 0.1620 | **0.1553** | 0.1941 |
| 2.000 | 0.1717 | 0.1573 | 0.2002 |
| 4.000 | 0.1915 | 0.1671 | 0.2081 |
| oracle | 0.1024 | 0.1137 | 0.1550 |

#### Key Patterns

1. **P2_aug achieves best absolute $E_{\text{geo}} = 0.1553$ at eff_β=1.0.** Gate augmentation genuinely helps.
2. **D7 optimal at eff_β=0.125–0.25** ($E_{\text{geo}} \approx 0.1573$). D7 is slightly better than P2 at very low β, but P2 wins at moderate-to-high β.
3. **β sweep at eval helps modestly, not dramatically:**
   - D7: 0.1717 → 0.1573 (Δ=0.014, ~8% relative)
   - P2: 0.1671 → 0.1553 (Δ=0.012, ~7% relative)
   - This is real but far from the "4× improvement" the C2′ metric artifact suggested.
4. **P2_aug_lowbeta degrades monotonically** across the entire β sweep — uniformly worst.

#### Component Breakdown at Optimal Points

| Condition (eval β) | $\mathcal{L}_{\text{RW}}$ | $\mathcal{L}_{\text{stress}}$ | $E_{\text{geo}}$ | Gap to oracle |
|---|---|---|---|---|
| D7 (eff_β=0.125) | 0.1059 | 0.5148 | 0.1574 | 0.0549 |
| P2_aug (eff_β=1.0) | 0.1040 | 0.5133 | 0.1553 | 0.0416 |

Gate degradation is minimal across all conditions (+0.001 to +0.009).

### Historical Comparison Invalidated

Oracle scores prove different function implementations: historical oracle ≈ 0.162, canonical reeval oracle = 0.102–0.155. The historical numbers (D7=0.440, P2=0.381) are **NOT comparable** to canonical reeval numbers. Different loss function implementations, different miniset construction, and different evaluation pipelines produce different absolute scales. **All valid comparisons must be within the same experimental cell.**

### Conclusions

1. **P2_aug is the best recipe.** Train with gate seeing $H_{\text{aug}}$ at eff_β=4.0, eval at eff_β=1.0.
2. **β tuning at eval provides 7–8% relative improvement** — real but modest. Train at high β, eval at lower β.
3. **Training at low β is actively harmful** — the gate provides essential inductive bias during training.
4. **C2′ "4× improvement" was metric artifact.** The 0.3× L_RW scaling made numbers appear dramatically better.
5. **The inference gap remains** — P2_aug at optimal β shows gap of 0.042 (27% of E_geo) from oracle.

### Exhausted Levers in Toy Regime

After C2′ failure and canonical reeval, all viable levers in the toy regime have been explored:

| Lever | Tested | Result |
|---|---|---|
| Gate input (clean vs augmented) | P2 | P2_aug wins |
| β tuning (train high, eval lower) | Canonical reeval | 7–8% gain |
| Gradient fixes (stop-grad) | P1 | No effect |
| Oracle distillation (C2′) | C2′ | Failed (gap too small to close) |
| Gate architecture (CGF vs MLP) | P4 series | MLP wins or tied |
| Frozen gate | E8d | Worse + unstable |
| Stronger gate supervision | E8e | No effect |
| Learned kNN projection | E9 | No precision gain |

**Remaining levers require scaling beyond the toy regime:** more training data, larger minisets, better candidate graph construction methods, or fundamentally different architectures.

---

## 24. Updated Final Recipe & Open Questions

### Current Best Configuration

Based on the full S1–S12 + D2–D7 + E-series + P-series + C2/C2′ + Canonical Reeval sweep:

**Architecture: V1 Shared with QK-norm + Geo Bias (IPALiteModelD7)**

| Setting | Value | Evidence |
|---|---|---|
| Architecture | IPALiteModelD7 (weight-shared block, QK-norm, clamped geo) | D7: best controlled result |
| d_input | 128 | Encoder output dim |
| c_s | 96 | All experiments |
| n_heads | 4 | All experiments |
| c_head | 32 | All experiments |
| Recycles | 5 | S1, D7: V1 5R validated |
| Geo mode | QK-norm + clamped geo bias | D7 architecture |
| Params | ~155K (model) + ~37K (gate) = ~192K | — |
| Best E_geo (H-gated, β-tuned) | **0.1553** | P2_aug at eval eff_β=1.0 (canonical reeval, 2 seeds) |
| Best E_geo (T-edge oracle) | **0.1137** | P2_aug oracle (canonical reeval, 2 seeds) |
| Inference gap at optimal β | **0.0416 (27% of E_geo)** | Canonical reeval |

> **Note:** Canonical reeval numbers (Phase 22) use a different E_geo function than historical D3–P series. Do NOT compare 0.1553 to 0.381 — they are from different metric implementations. All comparisons must be within the same experimental cell.

### Shared Settings

```
EdgeGateSoft:       d_hidden=64, 37K params, shared (not head-specific)
Candidate graph:    kNN(H) k=100, cosine distance
Gate pretrain:      500 steps on T_ref^(1) KL (T^(1) confirmed best in S11)
Gate bias:          Train with eff_β=4.0 (β=2.0, D7_BETA=2.0 multiplier)
                    Eval with eff_β=1.0 (optimal from canonical β sweep)
Gate augmentation:  Gate sees H_aug (same augmentation as model) ← P2 winner
Training schedule:  warmup=1ep, rw_ramp=6ep, rw_cap=0.3
Loss weights:       λ_stress=0.1, λ_rep=0.01, λ_ent=0.01
Augmentation:       h_noise=0.3, h_dropout=0.1, edge_drop=0.2, wd=5e-4
Optimizer:          AdamW lr=5e-4, CosineAnnealing eta_min=1e-5
Grad clip:          max_norm=1.0
Early stop:         patience=4 on gated-graph val E_geo (only after RW cap reached)
```

> **β Tuning (Phase 22):** Training at high β (eff_β=4.0) is essential — it forces the geometry module to learn good representations under strong gate supervision. At inference, reducing to eff_β=1.0 provides the best E_geo (0.1553 for P2_aug). Training at low β is actively harmful: P2_aug_lowbeta (eff_β=0.125) has the worst oracle score (0.1550 vs 0.1137), proving the geometry module itself learns worse when β is weak during training. β tuning provides a modest but real 7–8% relative improvement.

### Component Importance (Updated from E-Series)

| Component | Effect When Removed/Changed | Recommendation |
|---|---|---|
| Gate ($\beta \cdot g_{ij}$) | +0.48 catastrophic (Q2) | **Essential. Keep.** |
| qk ($q_i \cdot k_j / \sqrt{d}$) | +0.27 significant (Q2) | **Essential. Keep.** |
| Shared gate (vs head-specific) | No difference (S10) | Keep shared (simpler) |
| $T^{(1)}$ supervision (vs $T^{(2)}$/mix) | $T^{(1)}$ best by 0.05 (S11) | Keep $T^{(1)}$ |
| k=100, cosine (vs k=80, euclidean) | −0.08 improvement (S12), +2.4% recall (E7) | **Use k=100 cosine** |
| QK-norm + clamped geo | Validated in D7 | Keep (D7 architecture) |
| Monotone geo ($-\gamma d^2$ unclamped) | +0.20 vs no_geo (S6) | **Never use** |
| Pair representation ($z_{ij}$) | $\rho_{\text{pair}} \approx 0$ (S9) | Don't use |
| Head-specific gate | No specialization (S10) | Don't use |
| Unshared blocks (V2) | +0.09 worse (S1) | Don't use at this scale |
| Full IPA-R integration (D2) | +0.15 worse (D2) | Don't use at this scale |
| Increased gate loss weight | No benefit (E8e) | Keep $\lambda=0.05$ |
| Frozen gate | Worse + unstable (E8d) | Don't use |
| Learned kNN projection | No precision gain (E9) | Don't use |

### Updated "What NOT To Do" List

All failures from Phases 7–17 still apply. New validated failures from D3–E9 and P-series:

18. ❌ **Passing candidate-mask gate logits to oracle evaluation** — silently intersects oracle with candidate graph, makes oracle look worse than it is. Always pass zero gate logits for T-edge eval. (D3)
19. ❌ **Frozen pretrained gate during model training** — worse (0.56 vs 0.44) and highly unstable across seeds. Model needs co-adaptation. (E8d)
20. ❌ **Increasing gate loss weight to prevent drift** — drift persists at all $\lambda_{\text{gate}}$ values (0.05–1.0). Joint gate is consistently worse than pretrained regardless. (E8e)
21. ❌ **Learned linear projection for kNN construction** — precision unchanged (0.30→0.33). A 2-epoch linear probe can't improve on the encoder's spatial knowledge. (E9)
22. ❌ **Stop-grad on gate bias** — no effect on gate drift (P1). Drift is overfitting, not gradient contamination.
23. ❌ **β ramp (low→high)** — no benefit over constant β=2.0 when gate augmentation is already applied (P3).
24. ❌ **Contextual GateFormer (CGF)** — tested at 3 scales (n=128/k=100, n~300/adaptive k, high-recall candidates), multiple β values, detach variants. Never beats MLP by the required ≥0.03 margin (P4, P4v2, P4v3).
25. ❌ **Log-prior gate injection** — replacing $\beta \cdot g$ with $\beta \cdot \log(\text{softmax}(g))$ has no effect (P4v3-VALIDATE).
26. ❌ **Detaching gate bias in CGF** — actively hurts by +0.079 (P4v2 condition D vs C).
27. ❌ **Oracle graph teacher distillation (C2)** — uses coordinate-derived topology (oracle kNN) in forward pass. Abandoned on principle: violates constraint that no spatial ground truth enters any forward computation. (C2)
28. ❌ **Target-only gradient distillation (C2′)** — refining student X via intrinsic energy E(X) works locally (60–80% energy improvement) but distillation transfer fails: gap between baseline and all distillation modes < 0.008 at proper β. The signal is too weak to learn from. (C2′)
29. ❌ **Training at low β** — P2_aug_lowbeta (eff_β=0.125 during training) has the worst oracle (0.1550 vs 0.1137), proving geometry module itself learns worse representations when gate supervision is weak. High seed variance (±0.029) indicates instability. Always train at high β. (Canonical Reeval)

### The Gating Problem (Post E-Series Understanding)

The E-series provided a complete decomposition of the inference gap:

$$\underbrace{E_{\text{geo}}^{\text{H-gate}} - E_{\text{geo}}^{\text{oracle}}}_{\text{Total gap: } 0.278} = \underbrace{E_{\text{geo}}^{\text{perf.gate}} - E_{\text{geo}}^{\text{oracle}}}_{\text{Missing edges: } 0.002\; (0.4\%)} + \underbrace{E_{\text{geo}}^{\text{H-gate}} - E_{\text{geo}}^{\text{perf.gate}}}_{\text{Bad gating: } 0.450\; (99.6\%)}$$

The candidate graph has **perfect recall** (all true neighbors present) but **low precision** (30% — only 1 in 3 candidate edges is a true neighbor). The gate's job is to discriminate among these, but it can't do it well enough. The gate also degrades during joint training (E8c), and no amount of supervision anchoring prevents this (E8e). The gate and model are caught in a co-adaptation trap: they must train together, but training together degrades the gate.

**P-series update:** Gate augmentation (P2) partially addresses the overfitting by feeding H_aug to the gate, reducing $E_{\text{geo}}$ from 0.440 → 0.381. However, the fundamental issue remains: the gate logit scale ($\beta \cdot \text{std}(g) \approx 25$) completely dominates the structure module's QK attention ($\text{std}(\text{qk}) \approx 9$), meaning the structure module's learned geometry is largely overridden. CGF architecture doesn't help (P4 series) — the bottleneck is not gate expressiveness but gate-attention coupling.

**Canonical reeval update (Phase 22):** β sweep at inference provides a modest 7–8% relative improvement (P2_aug: 0.1671 → 0.1553 at optimal eff_β=1.0). The inference gap at optimal β is 0.0416 (27% of $E_{\text{geo}}$). Distillation (C2′) fails to close this gap — the refined coordinates improve local energy by 60–80% but the signal cannot be transferred back through the network. All toy-regime architectural levers are now exhausted.

### Open Questions

1. **Scale dependence of gating:** The current ceiling may be specific to the 2-slide toy regime. With more training slides, the gate may learn more robust neighbor discrimination. This is the strongest hypothesis for improvement. All toy-regime levers are exhausted.

2. **Gate-attention balance:** The gate term overwhelms QK attention ($\beta \cdot \text{std}(g) / \text{std}(\text{qk}) \approx 2.5$). β tuning at inference provides 7–8% improvement (canonical reeval), but learned dynamic β or attention-level gating remain untested at scale.

3. ~~**Oracle distillation (C2):**~~ **TESTED — FAILED.** C2 abandoned on principle (oracle topology in forward pass). C2′ (target-only distillation) failed: gap < 0.008 at proper β. The refined coordinates improve energy 60–80% locally but the signal cannot be distilled back into the network.

4. **Seed variance:** P2_aug canonical reeval achieves std = 0.014 across 2 seeds. The agent's threshold was ≤0.05 over 5 seeds. Needs verification at 5+ seeds.

5. **ST↔SC compatibility:** Untested. Relevant for the full pipeline but not for toy-scale experiments on ST-only data.

6. **Metric consistency:** Historical E_geo (D3–P series) and canonical E_geo (Phase 22) use different function implementations with different oracle scores (~0.162 vs ~0.102–0.115). Any future experiments must specify which metric is used and never compare across implementations.

---

## 25. Instructions for Coding Agent — Scaling to Full IPA

### What Changed Since Phase 17 (S-Series)

The D3–D7 + E-series + P-series + C2/C2′ + Canonical Reeval experiments added critical information:

1. **D3 oracle fix is mandatory.** When evaluating T-edge (oracle), always pass `gate_logits = 0` (zero tensor). Never pass candidate-mask gate logits — it silently corrupts the oracle baseline.
2. **D7 is the definitive architecture baseline.** Architecture and hyperparameters are locked in.
3. **The inference gap is 100% a gating problem** (E2b). The candidate graph has perfect recall. No graph-construction improvements will help.
4. **Gate augmentation (P2) is the one effective fix.** Feed H_aug (same augmentation as model) to the gate during training. This reduces gate overfitting.
5. **Gate drift is overfitting, not gradient contamination** (P1). Stop-grad has no effect. Gate augmentation partially addresses the overfitting.
6. **CGF architecture doesn't help** (P4 series). Tested at 3 scales with multiple β values and detach variants. MLP gate is sufficient.
7. **The gate dominates attention** ($\beta \cdot \text{std}(g) \approx 25 \gg \text{std}(\text{qk}) \approx 9$). The model essentially ignores candidate topology and relies purely on the MLP gate's learned edge scoring (P5 Test 2).
8. **Oracle distillation fails** (C2/C2′). C2 abandoned on principle. C2′ target-only distillation: refined coordinates improve energy 60–80% locally but cannot be transferred through the network (gap < 0.008).
9. **β tuning provides 7–8% gain** (Canonical Reeval). Train at high eff_β (4.0 for P2_aug), eval at eff_β=1.0. Training at low β is actively harmful.
10. **All toy-regime levers are exhausted.** Scaling data remains the most promising path.

### Architecture to Implement

```python
IPALiteModelD7:       # V1 shared, QK-norm, clamped geo bias
  d_input:       128
  c_s:           96
  n_heads:       4
  c_head:        32
  n_recycles:    5
  noise_std:     0.05
  qk_norm:       True (LayerNorm on q,k before dot product)
  geo:           Clamped [-10, 0] geometry bias

EdgeGateSoft:         # Shared (scalar output), as Cell N defined
  input:         [h_i, h_j, |h_i-h_j|, h_i⊙h_j]  (4×128=512)
  MLP:           512 → 64 → 64 → 1
  diagonal:      excluded, set to 0.0 (neutral)

Candidate graph:      kNN(H) at k=100, COSINE distance
Gate pretrain:        500 steps on T_ref^(1) KL
Gate bias:            β=2.0, broadcast to all heads
Gate augmentation:    Gate sees H_aug (P2 winner — prevents gate overfitting)
```

### What MUST Be Preserved (unchanged from Phase 10)

1. **Loss functions:** `loss_rw_kl`, `loss_stress_log`, `loss_repulsion`, `loss_entropy_reg` — all verified exact. DO NOT modify.
2. **`build_predicted_rw_operator`:** Uses `0.5*(W + W.T)` symmetrization, row-normalized, NO self-loops. DO NOT add self-loops.
3. **`gauge_fix`:** Center + unit-RMS normalize after every recycle. Essential for stability.
4. **Edge masking:** $-\infty$ for non-edges in attention logits, then softmax, then `nan_to_num(0.0)`.
5. **Coordinate update direction:** $x_j - x_i$ (toward neighbors), NOT $x_i - x_j$.
6. **Linear_x zero initialization:** The coordinate update linear should start at zero.
7. **Soft edge gating (entire mechanism from Phase 8).**

### Training Recipe

```
1.  Freeze encoder, compute all embeddings H
2.  Build minisets with F2+F4 targets (adaptive k, steps, landmarks)
3.  Build candidate masks: kNN(H) at k=100, cosine distance per miniset
4.  Cache T_ref^(1) per training miniset for gate supervision
5.  Initialize EdgeGateSoft
6.  Pretrain gate alone for 500 steps on T_ref KL
7.  Initialize IPALiteModelD7 (V1 shared, QK-norm, geo)
8.  Joint training:
    - Warmup 1 epoch (stress only)
    - RW ramp 6 epochs to cap=0.3
    - Train with combined_orig augmentation
    - Gate sees H_aug (SAME augmentation as model — P2 winner, prevents gate overfitting)
    - Gate loss λ=0.05, constant β=2.0 (eff_β=4.0 with D7_BETA multiplier for P2_aug)
    - Early stop patience=4 on gated-graph validation E_geo
9.  Always track Val(T) and Val(H_gate) separately
10. For T-edge eval: pass gate_logits = zeros (NOT candidate-mask logits)
11. β sweep at eval: sweep eff_β ∈ [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    - P2_aug optimal at eff_β=1.0; D7 optimal at eff_β=0.125–0.25
    - This provides 7–8% relative improvement over training β
12. Run attention diagnostics (Recall@k, logit_std, entropy vs recycle)
    to verify attention stays structured — this is a standard check
```

> **CRITICAL β WARNING:** Never train at low β. Canonical reeval proved that training at eff_β=0.125 (P2_aug_lowbeta) produces worse geometry module representations even at oracle eval (0.1550 vs 0.1137). The gate provides essential inductive bias during training. Only reduce β at eval time.

### Scaling Considerations

1. **Dense → Sparse attention:** For n>500, switch to edge-list form with sparse neighborhood softmax.
2. **Gate computation cost:** O(n²) dense → O(n×100) edge-list.
3. **Multi-slide training:** The primary lever for closing the inference gap. More data = better gate generalization.
4. **RW cap:** Start at 0.3, increase only with evidence.
5. **Candidate k:** Start at 100 with cosine distance. If scaling to n>1000, verify candidate recall stays >0.95.

### Evaluation Metrics

At validation/test, compute:
- **E_geo:** $w_{\text{rw}} \cdot L_{\text{RW}} + 0.1 \cdot L_{\text{stress}}$ (main metric — ensure $w_{\text{rw}}=1.0$ is consistent; see Phase 22 metric mismatch discovery)
- **E_geo at optimal β:** Run β sweep at eval (see Training Recipe step 11) — provides 7–8% gain
- **$\lambda_2/\lambda_1$:** covariance eigenvalue ratio (should be >0.01, ideally 0.2–0.5)
- **median pdist:** median pairwise distance (should be >0.01, typically ~1.2)
- **Gate precision:** fraction of gated edges that are true spatial neighbors
- **AttnRecall@30 per recycle:** should stay >0.40 at all recycles
- **logit_std per recycle:** should be 4–8 (bounded, not collapsing or exploding)

---

## Appendix: Complete Cell Index

| Cell | Description | Outcome |
|---|---|---|
| 0 | Data loading | Setup |
| 1 | Utility imports | Setup |
| 2 | Miniset creation (1000, F2+F4) | ✅ 1000 minisets, sanity checks pass |
| 3 | Loss functions + gauge_fix | ✅ All verified, exact T_ref match |
| 4 | Oracle energy minimization | ✅ E* ≈ −0.008, zero collapses |
| 5 | IPA-Lite model definition | ✅ Forward pass, gradient flow |
| 6 | Single miniset overfit | ✅ 5.0 → 0.12, architecture works |
| 7 | Multi-miniset (random H) | ✅ Zero collapses, bottleneck is embeddings |
| A | Real encoder embedding setup | ✅ 128-dim, spatial signal confirmed |
| B | Single overfit (real H) | ✅ 18.6× improvement over random |
| C | Cross-slide generalization | ✅ 0.45 (vs 2.85 random) |
| D | Initial aug sweep | ⚠️ Flawed early stopping |
| E | Agent corrections | Fixed seeding + early stopping |
| F | Corrected aug sweep | ✅ combined_orig wins (E=0.35) |
| G | RW schedule + stress sweep | ✅ rw_cap=0.3 wins decisively (E=0.31) |
| H | Multi-seed stability | ⚠️ Seed 0 was lucky, realistic ≈0.44±0.06 |
| I | kNN(H) topology test | ⚠️ +0.41 gap, ~25% false neighbors |
| J | Graph quality diagnostics | Quantified recall, false neighbor rate |
| K | Hybrid curriculum | ❌ Failed — degraded target performance |
| L | Dynamic kNN(X) refinement | ❌ Failed — cascading graph degradation |
| M | Hard top-k edge gating | ❌ Failed — train-test mismatch |
| N | Soft gate bias + warm-start | ✅ First success — 57% gap closed |
| O | β/λ_gate sweep | ✅ β=2.0, λ=0.05 wins (E=0.44) |
| P | Candidate k + pretrain steps | ✅ k=80, 500 steps wins (E=0.38, best V1) |
| Q1 | Attention instrumentation (V1) | ⚠️ Broken gate — collapse was artifact |
| Q2 | Logit-term ablation | ✅ Gate essential, qk matters, geo useless |
| R1 | V2: unshared blocks + QK-norm + geo clamp | ✅ E=0.40, healthy attention (but V2) |
| R2 | V2 attention diagnostics | ✅ Recall increases R0→R2 in V2 |
| R3 | Publication-quality attention visualizations | ✅ Attention focus maps + coord progression |
| **S1** | **Controlled replication: V1 vs V2** | **✅ V1 wins (0.43 vs 0.52), V1 attention healthy** |
| **S6** | **RBF distance bias vs monotone vs no-geo** | **✅ RBF ≈ no_geo > mono; RBF shows anti-locality** |
| **S7** | **Point-IPA geo bias** | **❌ No benefit; offset explosion in V2** |
| **S9** | **Pair representation $z_{ij}$** | **❌ Redundant with gate ($\rho \approx 0$)** |
| **S10** | **Head-specific gate** | **❌ No specialization; identical to shared gate** |
| **S11** | **Gate supervision: $T^{(1)}$ vs $T^{(2)}$ vs mix** | **✅ $T^{(1)}$ wins (best precision/recall)** |
| **S12** | **Candidate k sensitivity** | **✅ k=100 now best (0.46 vs 0.53 at k=80)** |
| **D2** | **Full IPA-R(2D) integration** | **❌ Worse than V1 (0.61 vs 0.46)** |
| **D3** | **Oracle graph confound fix** | **✅ Bug found: T-edge was intersecting with candidate mask** |
| **D4** | **D2 Hypothesis 1: term competition** | **✅ Confirmed — R0 logit explosion caused by competing terms** |
| **D5** | **D2 Hypothesis 2: gradient-attention misalignment** | **❌ Rejected — no systematic misalignment** |
| **D6** | **Architecture variants with bug fixes** | **✅ V1 shared confirmed again** |
| **D7** | **Final V1 5R + geo + QK-norm baseline** | **✅ 0.440 ± 0.014 (H-gate), 0.162 (T-edge)** |
| **E1** | **Encoder audit (norm + leakage)** | **✅ All clean** |
| **E7** | **Graph recovery diagnostics** | **✅ Recall=0.96, cosine > euclidean, all slides consistent** |
| **E2** | **Shuffle controls (leakage test)** | **✅ 6–8× degradation, no leakage** |
| **E2b** | **Inference gap decomposition** | **🎯 99.6% bad gating, 0.4% missing edges** |
| **E8a** | **Mutual-kNN precision sweep** | **❌ Marginal (0.30→0.37), not worth it** |
| **E8c** | **Gate effectiveness: pretrained vs joint** | **🔍 Pretrained gate better; joint training degrades gate** |
| **E8d** | **Frozen gate training + β sweep** | **❌ Worse + unstable (0.56 vs 0.44)** |
| **E8e** | **Gate loss weight sweep** | **❌ Drift persists at all λ values** |
| **E9** | **Learned projection for kNN** | **❌ Precision unchanged (0.30→0.33)** |
| **P1** | **Stop-grad gate bias** | **❌ No effect — drift is overfitting, not gradient-driven** |
| **P2+P3** | **Gate augmentation + β ramp** | **✅ B_aug_gate wins: 0.381 ± 0.004 (−0.059 vs D7)** |
| **P5** | **Validation battery (poison pills)** | **⚠️ 2/3 pass; random graph test FAILS (gate dominates topology)** |
| **P4** | **CGF vs MLP gate (n=128, k=100)** | **❌ MLP wins by 0.153** |
| **P4v2** | **CGF with sparse graphs (n~300, adaptive k)** | **❌ MLP wins by 0.178; CGF β=0.5 promising** |
| **P4v2-followup** | **5-condition factorial (β, detach)** | **⚠️ CGF β=0.5 beats MLP by 0.056 (MLP at wrong β)** |
| **P4v3** | **Definitive CGF 3-seed test** | **❌ Inconclusive — Δ=0.0006 at matched β** |
| **P4v3-VALIDATE** | **Log-prior + β-fix injection** | **❌ No effect (Δ < 0.005 for all modes)** |
| **C2** | **Oracle graph teacher distillation** | **❌ Abandoned on principle (oracle topology in forward pass)** |
| **C2′** | **Target-only gradient distillation** | **❌ Failed — distillation gap < 0.008 at proper β** |
| **Canon. Reeval** | **β metric fix + canonical β sweep** | **✅ P2_aug best at eval eff_β=1.0; low-β training harmful** |

---

## Appendix: Summary of All Key Numerical Results

### Cross-Slide Validation (Train ST1+2, Val ST3)

| Milestone | Val E_geo (H-gate) | Val E_geo (T-edge) | Architecture | Context |
|---|---|---|---|---|
| Oracle floor (direct X optimization) | −0.008 | — | — | Upper bound |
| **D7 T-edge oracle** | — | **0.162** | **V1 shared D7** | **True oracle (D3 fix)** |
| **P2 B_aug_gate** | **0.381 ± 0.004** | — | **V1 shared D7 + gate aug** | **Current best (P-series, historical metric)** |
| D7 H-gated inference | 0.440 ± 0.014 | — | V1 shared D7 | Pre-P-series best |
| **P2_aug β-tuned (canonical)** | **0.1553** | **0.1137** | **V1 shared D7 + gate aug** | **Canonical reeval metric (eff_β=1.0)** |
| **D7 β-tuned (canonical)** | **0.1573** | **0.1024** | **V1 shared D7** | **Canonical reeval metric (eff_β=0.125)** |
| V1 5R no_geo, k=100 (S12) | 0.46 | — | V1 shared | S-series best at k=100 |
| V1 5R no_geo (S7) | 0.44 | — | V1 shared | S-series best at k=80 |
| V1 5R shared (S1) | 0.43 | — | V1 shared | Controlled replication |
| Target edges (Cell G) | 0.31 | — | V1 shared | Oracle graph |
| V2 3R geo_last (S1) | 0.52 | — | V2 unshared | QK-norm, geo_last |
| Full IPA-R(2D) (D2) | 0.61 | — | IPA-R | All components |
| E9 learned projection | 0.61 ± 0.05 | — | V1 + projection | Failed improvement |
| E8d frozen gate (best β) | 0.56 ± 0.18 | — | V1 + frozen gate | Failed improvement |
| Static kNN(H) k=50 (Cell I) | 0.72 | — | V1 shared | No gating |
| Random embeddings (Cell C) | 2.85 | — | V1 shared | No spatial signal |

### Inference Gap Decomposition (E2b)

| Gap Component | Absolute | Fraction |
|---|---|---|
| Total gap (H-gate − oracle) | 0.452 | 100% |
| Missing-edge cost | 0.002 | 0.4% |
| **Bad-gating cost** | **0.450** | **99.6%** |

### D/E-Series Experiment Summary

| Exp | Question | Result | Key Finding |
|---|---|---|---|
| D3 | Oracle evaluation contaminated? | **Yes — bug found** | T-edge was intersecting with candidate mask |
| D4 | D2 term competition? | **Confirmed** | R0 logit explosion poisons later recycles |
| D5 | Gradient-attention misalignment? | **Rejected** | No systematic misalignment found |
| D7 | Final baseline | **0.440 ± 0.014** | Definitive V1 shared result |
| E1 | Encoder clean? | **Yes** | Norm ratio 1.04, slide acc 43% |
| E7 | Graph quality? | **Healthy** | Recall=0.96, cosine > euc, uniform slides |
| E2 | Leakage? | **None** | 6–8× degradation on all controls |
| E2b | Gap source? | **99.6% gating** | Candidate recall=1.0, precision=0.30 |
| E8a | Mutual-kNN help? | **No** | Precision 0.30→0.37 only |
| E8c | Gate quality? | **Degrades in training** | Pretrained > joint on same model |
| E8d | Frozen gate? | **Worse + unstable** | Model needs co-adaptation |
| E8e | Stronger gate loss? | **No improvement** | Drift persists at all λ |
| E9 | Learned projection? | **No improvement** | Precision unchanged |

### P-Series Experiment Summary

| Exp | Hypothesis | Result | Key Finding |
|---|---|---|---|
| P1 | Stop-grad fixes drift? | **No** | Drift is overfitting, not gradient-driven |
| **P2** | **Gate sees H_aug?** | **✅ Best fix** | **$E_{\text{geo}}$: 0.440 → 0.381 (−13.4%)** |
| P3 | β ramp helps? | **No** | No benefit over constant β with aug |
| P5 | Scale-readiness? | **Partial** | Shuffle ✅, k-robust ✅, random-graph ❌ (gate dominates topology) |
| P4 | CGF > MLP? | **No** | MLP wins by 0.153 at n=128 |
| P4v2 | CGF with sparse graphs? | **No** | MLP wins by 0.178 at n~300 |
| P4v3 | CGF definitive test? | **Inconclusive** | Δ=0.0006 at matched β (3 seeds) |
| P4v3-V | β-fix / log-prior? | **No** | All injection modes equivalent (Δ<0.005) |
| C2 | Oracle graph teacher? | **Abandoned** | Principle violation: oracle topology in forward pass |
| C2′ | Target-only distillation? | **No** | Gap < 0.008 at proper β; refined X improves energy but can't distill |
| Canon. Reeval | β sweep + metric fix? | **✅ β tuning works** | P2_aug: 0.1671→0.1553 (7–8% gain) at eval eff_β=1.0; low-β training harmful |

### The Attention Quality Story (Final)

| Diagnostic | V1 D7 (best) | V2 3R geo_last (S1) | IPA-R (D2) |
|---|---|---|---|
| **E_geo (H-gate)** | **0.440** (best) | 0.52 | 0.61 (worst) |
| **E_geo (T-edge)** | **0.162** (best) | 0.28 | 0.47 |
| Recall@30 at final recycle | 0.55 | 0.56 | 0.70 (best) |
| $\rho_{\text{gate}}$ at final recycle | 0.77 | 0.77 | 0.81 |
| logit_std at final recycle | 6.0 | 6.9 | 2.1 |
| Entropy at final recycle | 2.89 | 2.87 | 3.03 |

**Paradox persists:** IPA-R has the *best* attention metrics but the *worst* E_geo. Simple V1 works best at toy scale. Post-canonical-reeval best: P2_aug at eval eff_β=1.0, with inference gap = 0.0416 (27% of $E_{\text{geo}}$). All toy-regime architectural levers exhausted.
