# IPA-Lite Toy Validation — Complete Experiment History & Integration Guide

**Author:** Ehtesam (experiments) + Claude (code & analysis)  
**Dates:** Feb 19–22, 2026  
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
19. [Updated Final Recipe & Open Questions](#19-updated-final-recipe--open-questions)
20. [Instructions for Coding Agent — Scaling to Full IPA](#20-instructions-for-coding-agent--scaling-to-full-ipa)

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

## 19. Updated Final Recipe & Open Questions

### Current Best Configuration

Based on the full S1–S12 + D2 experimental sweep, the recommended configuration is:

**Architecture: V1 Shared (IPALiteModel)**

| Setting | Value | Evidence |
|---|---|---|
| Architecture | IPALiteModel (weight-shared block) | S1: V1 beats V2 (0.43 vs 0.52) |
| d_input | 128 | Encoder output dim |
| c_s | 96 | Cell 7 onwards |
| n_heads | 4 | All experiments |
| c_head | 32 | All experiments |
| Recycles | 5 | S1: V1 5R validated |
| Geo mode | **no_geo** (or optional RBF) | S6/S7: no_geo ≈ RBF > mono > point_ipa |
| Params | ~155K (model) + ~37K (gate) = ~192K | — |
| Best E_geo (H-gated) | **0.43–0.46** | S1, S7, S12 |

### Shared Settings

```
EdgeGateSoft:       d_hidden=64, 37K params, shared (not head-specific)
Candidate graph:    kNN(H) k=100 (updated from 80 based on S12)
Gate pretrain:      500 steps on T_ref^(1) KL (T^(1) confirmed best in S11)
Gate bias:          β=2.0, λ_gate=0.05
Training schedule:  warmup=1ep, rw_ramp=6ep, rw_cap=0.3
Loss weights:       λ_stress=0.1, λ_rep=0.01, λ_ent=0.01
Augmentation:       h_noise=0.3, h_dropout=0.1, edge_drop=0.2, wd=5e-4
Optimizer:          AdamW lr=5e-4, CosineAnnealing eta_min=1e-5
Grad clip:          max_norm=1.0
Early stop:         patience=4 on gated-graph val E_geo (only after RW cap reached)
```

### Component Importance (Updated from S6–S12)

| Component | Effect When Removed/Changed | Recommendation |
|---|---|---|
| Gate ($\beta \cdot g_{ij}$) | +0.48 catastrophic (Q2) | **Essential. Keep.** |
| qk ($q_i \cdot k_j / \sqrt{d}$) | +0.27 significant (Q2) | **Essential. Keep.** |
| Shared gate (vs head-specific) | No difference in specialization (S10) | Keep shared (simpler) |
| $T^{(1)}$ supervision (vs $T^{(2)}$/mix) | $T^{(1)}$ best by 0.05 (S11) | Keep $T^{(1)}$ |
| k=100 (vs k=80) | −0.08 improvement (S12) | **Update to k=100** |
| RBF geo bias | −0.01 vs no_geo (S6/S7) | Optional (marginal) |
| Monotone geo ($-\gamma d^2$) | +0.20 vs no_geo (S6) | **Never use** |
| Point-IPA geo | +0.03 vs no_geo (S7) | Don't use |
| Pair representation ($z_{ij}$) | $\rho_{\text{pair}} \approx 0$, no E_geo benefit (S9) | Don't use |
| Unshared blocks (V2) | +0.09 worse E_geo (S1) | Don't use at this scale |
| Full IPA-R integration (D2) | +0.15 worse than V1 (D2) | Don't use at this scale |

### What Changed Since Phase 10

| Belief (Phase 10) | Updated Belief (Phase 17) | Evidence |
|---|---|---|
| V1 attention collapses at R2+ | **V1 attention is healthy across all recycles** when gate is properly integrated | S1 |
| V2 has "genuinely superior attention" | V2 attention is not superior — V1 matches V2 Recall/entropy | S1 |
| V2 recommended for paper | **V1 recommended** — simpler, better E_geo, healthy attention | S1 |
| geo_last helps in V2 (0.40 vs 0.44) | Monotone geo is harmful; RBF provides marginal anti-locality benefit | S6, S7 |
| Pair representation is a "missing ingredient" | **Pair bias is informationally redundant** with gate | S9 |
| Head-specific gate enables specialization | **No specialization observed** — all heads learn identical functions | S10 |
| k=80 is optimal | **k=100 is now optimal** (0.46 vs 0.53 at k=80) | S12 |
| Full IPA-R should be the target architecture | **Simple V1 outperforms full IPA-R** at toy scale | D2 |

### Updated "What NOT To Do" List

All failures from Phases 7–10 still apply. New validated failures from S1–D2:

11. ❌ **Unshared blocks (V2) at toy scale** — worse E_geo (0.52 vs 0.43), 2.3× params, no attention benefit. (S1)
12. ❌ **Monotone geo bias ($-\gamma d^2$)** — worst performer across all geo modes (0.64 mean). Creates self-reinforcing locality errors on noisy coordinates. (S6)
13. ❌ **Point-IPA geo offsets** — no benefit over no_geo; offset norms explode in V2. (S7)
14. ❌ **Pair representation $z_{ij}$ as attention bias** — informationally redundant with gate ($\rho_{\text{pair}} \approx 0$), no E_geo improvement. (S9)
15. ❌ **Head-specific gate** — no specialization emerges; all heads converge to same function. (S10)
16. ❌ **$T^{(2)}$ gate supervision** — dilutes neighborhood signal, reduces gate precision. (S11)
17. ❌ **Full IPA-R integration (all components together)** — component interference at toy scale produces worse results (0.61) than simple V1 (0.46). (D2)

### Open Questions

1. **Scale dependence:** Nearly all "advanced" components (V2 unshared, pair bias, IPA-R) performed worse at 2-slide toy scale. Would they help with 10+ training slides, larger minisets, and more complex spatial layouts? The current experiments cannot answer this.

2. **RBF anti-locality:** The RBF learns to counter the gate's locality bias. Is this a stable phenomenon across datasets, or dataset-specific? Would it persist at scale?

3. **k=100 reversal:** Cell P found k=80 best; S12 finds k=100 best. This suggests the optimal k depends on gate quality and training conditions. At scale, $k$ should be treated as a tunable hyperparameter, not fixed.

4. **Seed variance:** Many experiments show 0.05–0.10 std across seeds (e.g., V1 no_geo S7: 0.44±0.04). The agent's scale-readiness threshold was std ≤ 0.05 over 5 seeds. This has not been met for any configuration.

5. **The 0.43 vs 0.31 gap:** The best H-gated inference E_geo (0.43) is still +0.12 above the target-edge oracle (0.31). This remaining gap is the gate's imperfection — the candidate set misses ~5% of true neighbors and the gate doesn't perfectly rank the rest. Is this gap compressible at scale, or is it a fundamental limit?

---

## 20. Instructions for Coding Agent — Scaling to Full IPA

### What Changed Since Phase 10

The S1–D2 experiments fundamentally changed the architectural recommendation:

1. **V1 shared is the winner.** Drop V2 unshared blocks, QK-norm, and all V2-specific machinery.
2. **No geo term needed.** The geometry bias is harmful (mono) or negligible (RBF/point-IPA) at this scale. Ship without it.
3. **No pair representation.** The gate already provides all relevant pairwise information.
4. **Shared gate, not head-specific.** No specialization benefit from per-head gating.
5. **$T^{(1)}$ gate supervision confirmed.** Don't experiment with $T^{(2)}$ or mixtures.
6. **Update k from 80 to 100.** Higher candidate recall helps now that the gate is mature.

### Architecture to Implement

```python
IPALiteModel:         # V1 shared, exactly as Cell P/S1 validated
  d_input:       128
  c_s:           96
  n_heads:       4
  c_head:        32
  n_recycles:    5
  noise_std:     0.05
  geo:           DISABLED (no γ term in attention logits)

EdgeGateSoft:         # Shared (scalar output), as Cell N defined
  input:         [h_i, h_j, |h_i-h_j|, h_i⊙h_j]  (4×128=512)
  MLP:           512 → 64 → 64 → 1
  diagonal:      excluded, set to 0.0 (neutral)

Candidate graph:      kNN(H) at k=100
Gate pretrain:        500 steps on T_ref^(1) KL
Gate bias:            β=2.0, broadcast to all heads
```

### What MUST Be Preserved (unchanged)

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
3.  Build candidate masks: kNN(H) at k=100 per miniset
4.  Cache T_ref^(1) per training miniset for gate supervision
5.  Initialize EdgeGateSoft
6.  Pretrain gate alone for 500 steps on T_ref KL
7.  Initialize IPALiteModel (V1 shared, no geo)
8.  Joint training:
    - Warmup 1 epoch (stress only)
    - RW ramp 6 epochs to cap=0.3
    - Train with combined_orig augmentation
    - Gate loss λ=0.05
    - Early stop patience=4 on gated-graph validation E_geo
9.  Always track Val(T) and Val(H_gate) separately
10. Run attention diagnostics (Recall@k, logit_std, entropy vs recycle)
    to verify attention stays structured — this is a standard check
```

### Scaling Considerations

1. **Dense → Sparse attention:** For n>500, switch to edge-list form with sparse neighborhood softmax.
2. **Gate computation cost:** O(n²) dense → O(n×100) edge-list.
3. **Multi-slide training:** Main bottleneck at toy scale. More training data is expected to reduce seed variance and may enable more complex architectures.
4. **RW cap:** Start at 0.3, increase only with evidence.
5. **Candidate k:** Start at 100. If scaling to n>1000, verify candidate recall stays >0.95 and adjust k if needed.

### Evaluation Metrics

At validation/test, compute:
- **E_geo:** $w_{\text{rw}} \cdot L_{\text{RW}} + 0.1 \cdot L_{\text{stress}}$ (main metric)
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

---

## Appendix: Summary of All Key Numerical Results

### Cross-Slide Validation (Train ST1+2, Val ST3)

| Milestone | Val E_geo (H-gate) | Architecture | Context |
|---|---|---|---|
| Oracle floor (direct X optimization) | −0.008 | — | Upper bound |
| Target edges (Cell G) | 0.31 | V1 shared | Best case (oracle graph) |
| Gated inference, V1 shared (Cell P) | 0.38 | V1 shared | Early best (k=80) |
| **V1 5R no_geo, k=100 (Cell S12)** | **0.46** | **V1 shared** | **Current best at k=100** |
| V1 5R no_geo (Cell S7) | 0.44 | V1 shared | Best at k=80 |
| V1 5R shared (Cell S1) | 0.43 | V1 shared | Controlled replication |
| V1 5R rbf (Cell S7) | 0.43 | V1 shared | RBF geo |
| V1 5R point_ipa (Cell S7) | 0.47 | V1 shared | Point-IPA geo |
| V1 5R hgate (Cell S10) | 0.49 | V1 shared | Head-specific gate |
| gate_T1 (Cell S11) | 0.50 | V1 shared | $T^{(1)}$ supervision |
| V2 3R geo_last (Cell S1) | 0.52 | V2 unshared | QK-norm, geo_last |
| gate_mix (Cell S11) | 0.52 | V1 shared | Mixture supervision |
| gate_T2 (Cell S11) | 0.56 | V1 shared | $T^{(2)}$ supervision |
| V1 5R pair (Cell S9) | 0.57 | V1 shared | Pair representation |
| **Full IPA-R(2D) (Cell D2)** | **0.61** | **IPA-R** | **All components integrated** |
| V1 5R mono geo (Cell S6) | 0.64 | V1 shared | Monotone geo |
| Static kNN(H) k=50 (Cell I) | 0.72 | V1 shared | No gating |
| Random embeddings (Cell C) | 2.85 | V1 shared | No spatial signal |

### S-Series Experiment Summary (Agent-Recommended Experiments)

| Exp | Question | Winner | Key Finding |
|---|---|---|---|
| S1 (Exp 1) | V1 vs V2 under identical conditions? | **V1** (0.43 vs 0.52) | V1 attention healthy; Q1 collapse was artifact |
| S6 (Exp 6) | RBF vs monotone vs no-geo? | **RBF ≈ no_geo** (0.48 vs 0.56 vs 0.64) | RBF learns anti-locality; mono always harmful |
| S7 (Exp 7) | Point-IPA geo? | **no_geo** (0.44) | Point-IPA no benefit (0.47); offsets explode in V2 |
| S9 (Exp 9) | Pair representation $z_{ij}$? | **No benefit** (0.57 vs 0.58 base) | $\rho_{\text{pair}} \approx 0$; redundant with gate |
| S10 (Exp 10) | Head-specific gate? | **No specialization** (diversity identical) | All heads learn same function |
| S11 (Exp 11) | $T^{(1)}$ vs $T^{(2)}$ supervision? | **$T^{(1)}$** (0.50 vs 0.56) | $T^{(2)}$ dilutes neighborhood signal |
| S12 (Exp 12) | Candidate k sensitivity? | **k=100** (0.46 vs 0.53) | Reverses Cell P finding; higher recall helps |
| D2 | Full IPA-R integration? | **V1 still better** (0.46 vs 0.61) | Component interference at toy scale |

### The Attention Quality Story (Revised)

| Diagnostic | V1 Shared (S1, proper gate) | V2 3R geo_last (S1) | IPA-R (D2) |
|---|---|---|---|
| **E_geo** | **0.43** (best) | 0.52 | 0.61 (worst) |
| Recall@30 at final recycle | 0.55 | 0.56 | 0.70 (best) |
| $\rho_{\text{gate}}$ at final recycle | 0.77 | 0.77 | 0.81 |
| logit_std at final recycle | 6.0 | 6.9 | 2.1 |
| Entropy at final recycle | 2.89 | 2.87 | 3.03 |
| Attention structured? | ✅ All recycles | ✅ All recycles | ✅ All recycles |

**Paradox:** IPA-R has the *best* attention metrics (Recall 0.70, clean progression) but the *worst* E_geo (0.61). V1 has moderate attention metrics but the best E_geo. This suggests that at toy scale, having "perfect" attention selection is less important than having a simple, well-regularized architecture that converges reliably with limited training data.
