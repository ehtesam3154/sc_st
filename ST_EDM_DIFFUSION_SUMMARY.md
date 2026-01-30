# ST EDM Diffusion Training - Technical Summary for Research Agent

**Purpose**: This document describes the ST (Spatial Transcriptomics) EDM diffusion training system. We need help diagnosing whether there's a real problem and what to fix.

---

## 1. CRITICAL CONTEXT: THE REAL GOAL

### 1.1 What We're Actually Trying to Do

**Goal**: Single-cell spatial reconstruction from gene expression. Given single cells (no spatial info), predict their relative spatial positions.

**Training Data**: Spatial transcriptomics (ST) spots with known coordinates. We use these to learn a mapping from gene expression → spatial geometry.

**Key Insight**: We don't care about absolute scale. We care about **intrinsic geometry** - the relative positions of cells. If the model outputs coordinates at 0.5x scale but the neighborhood structure is correct, that's acceptable.

### 1.2 Inference Pipeline: Generator + Refiner

We do NOT do full denoising from pure noise. Our pipeline is:

```
Gene Expression → Generator → V_init (initial structure estimate)
                                 ↓
                              Add moderate noise (σ ≈ 1-2× σ_data)
                                 ↓
                              Refiner (few denoising steps)
                                 ↓
                              V_final (refined structure)
```

**σ_data for this dataset ≈ 0.157**

So we only need the diffusion model to work well for moderate noise levels, NOT from extremely high noise (σ >> σ_data).

### 1.3 What Actually Works

| Component | Performance | Notes |
|-----------|-------------|-------|
| Generator | scale_r ≈ 0.96, Jacc@10 ≈ 0.34 | Excellent |
| Low-σ denoise (σ < 0.1) | scale_r ≈ 0.97, Jacc ≈ 0.5-0.8 | Good |
| One-step denoise | Beats baseline (ratio ~0.09) | Good |

---

## 2. THE OBSERVED "PROBLEM" - IS IT REALLY A PROBLEM?

### 2.1 Scale Collapse at High σ

We observe that at high noise levels, the model outputs under-scaled predictions:

| σ | scale_r | What's Happening |
|---|---------|------------------|
| 0.05 | 0.97 | Good - c_skip ≈ 0.9, skip connection dominates |
| 0.15 | 0.74 | Degrading |
| 0.40 | 0.37 | Poor |
| 0.70 | 0.22 | Severe collapse |
| 1.20 | 0.14 | Very severe |
| 2.40 | 0.08 | Extreme collapse |

**But here's the question**: If we only need to refine from σ ≈ 0.15-0.30 (1-2× σ_data), does scale collapse at σ > 0.5 even matter for our use case?

### 2.2 The Curriculum is Stuck

The curriculum is at stage 4/6 with stall count 57-61/6. It can't advance because:
- Open-loop sampling from high σ fails (kNN@10 ≈ 0.04)
- Structure degrades through sampling chain instead of improving

But again: **we don't plan to do open-loop sampling from high noise in production**.

### 2.3 A/B Test Results (Recent Logs)

Recent diagnostic tests show:
- **H (conditioning) IS being used** at high σ (rel_diff > 0.10)
- F_x magnitude changes when H is ablated (Fx_ratio ≈ 0.73)
- Conclusion: "likely objective/weighting, not conditioning pipeline"

The conditioning pathway is working. The model IS using the gene expression info. The issue is in how the objective handles scale at high σ.

---

## 3. INTRINSIC GEOMETRY: WHY SCALE MAY NOT MATTER

### 3.1 V_target is Already Intrinsic Geometry

**Critical code** (core_models_et_p2.py:4572-4580):
```python
# --- PATCH 3: Canonicalize ST target coordinates from Gram ---
V_target = torch.zeros_like(V_target_raw)
for i in range(batch_size_real):
    n_valid = int(mask[i].sum().item())
    if n_valid <= 1:
        continue
    G_i = G_target[i, :n_valid, :n_valid].float()
    V_i = uet.factor_from_gram(G_i, D_latent).to(V_target_raw.dtype)
    V_target[i, :n_valid] = V_i
```

V_target is **factored from the Gram matrix** - it's the intrinsic geometry extracted via eigendecomposition. We're NOT memorizing raw ST coordinates.

### 3.2 What Intrinsic Geometry Cares About

For relative positions / neighborhood structure:
- **kNN rankings** - scale-invariant (if you scale all by 0.3x, rankings don't change)
- **Jaccard overlap** - scale-invariant
- **Angles between vectors** - scale-invariant
- **Proportional distances** - scale-invariant (if A-B is 2x A-C, that ratio is preserved)

For absolute structure:
- **Gram matrix values** - scale-dependent (G scales with V²)
- **Trace ratios** - scale-dependent
- **RMS of coordinates** - scale-dependent

**Question**: Should we even be measuring/penalizing absolute scale, or should everything be scale-invariant?

---

## 4. ARCHITECTURE OVERVIEW

### 4.1 Data Flow

```
Gene Expression (per spot)
       ↓
Shared Encoder (frozen, Stage A)
       ↓
Z embeddings (B, N, h_dim)
       ↓
SetEncoderContext (context_encoder)
       ↓
Context H (B, N, c_dim=256)
       ↓           ↓
  Generator    Score Network (Denoiser)
       ↓           ↓
   V_init      ε̂ prediction → V_hat
```

### 4.2 Score Network / Denoiser (`DiffusionScoreNet`)

**Location**: `core_models_et_p2.py:936`

- Input: `[H, t_emb, V_in, angle_features, self_cond_features]`
- Stack of 4 ISAB blocks (Induced Set Attention) with FiLM modulation
- Output: ε̂ (predicted noise) → converted to V_hat via EDM formula

### 4.3 Generator (`MetricSetGenerator`)

**Location**: `core_models_et_p2.py:774`

- 2 ISAB blocks + MLP head
- Direct structure prediction from H
- Works excellently (scale_r ≈ 0.96)

---

## 5. EDM PRECONDITIONING

### 5.1 The EDM Formula

```python
# Preconditioning (σ_data ≈ 0.157)
c_skip = σ_data² / (σ² + σ_data²)        # Skip connection weight
c_out = σ · σ_data / √(σ² + σ_data²)     # Output scaling
c_in = 1 / √(σ_data² + σ²)               # Input scaling

# Network prediction
F_x = score_net(c_in · V_t, c_noise, H, mask)

# Final denoised output
V_hat = c_skip · V_t + c_out · F_x
```

**At high σ**: c_skip → 0, so V_hat ≈ c_out · F_x
**At low σ**: c_skip → 1, so V_hat ≈ V_t (noisy input passed through)

### 5.2 The F_x Magnitude Issue

We observe F_x is severely under-scaled at high σ:

```
[DEBUG-OUT-SCALE-FX] F_x space scale analysis:
  σ∈[0.00, 0.10): Fx_pred/Fx_tgt = 0.065
  σ∈[0.30, 0.50): Fx_pred/Fx_tgt = 0.034
  σ∈[0.70, 1.20): Fx_pred/Fx_tgt = 0.027  ← 37x too small!
```

When c_skip → 0 at high σ, the under-scaled F_x directly causes output collapse.

**But**: This only affects denoising from high σ, which we don't plan to do.

---

## 6. LOSS FUNCTIONS (NON-ZERO WEIGHTS)

### 6.1 Loss Weights

```python
WEIGHTS = {
    'score': 16.0,      # Primary denoising MSE
    'gram': 2.0,        # Gram matrix matching (global structure)
    'gram_scale': 2.0,  # Gram trace matching (scale)
    'out_scale': 2.0,   # RMS scale matching
    'gram_learn': 2.0,  # Gram loss on learned branch at high σ
    'knn_scale': 0.2,   # kNN distance scale
    'knn_nca': 2.0,     # Neighborhood consistency
    'edge': 4.0,        # Local edge lengths
    'gen_align': 10.0,  # Generator Procrustes alignment
    'gen_scale': 10.0,  # Generator scale matching
    'subspace': 0.5,    # Low-rank structure
    'ctx_edge': 0.05,   # Context invariance
}
```

### 6.2 Score Loss (weight=16.0)

The primary denoising objective:
```python
err2_sample = masked_mean((V_hat - V_target)², mask)
w = λ(σ) · g(σ) · boost(σ)  # EDM weight + high-σ compensation
L_score = (w · err2_sample).sum() / w.sum()
```

### 6.3 Gram Loss (weight=2.0)

**Scale-sensitive** - compares G_pred to G_target:
```python
per_set_relative_loss = |G_pred - G_target|² / |G_target|²
```

**Important**: Gated to low-σ samples only (σ < ~0.4). At high σ, this loss doesn't apply.

### 6.4 Edge Loss (weight=4.0)

Local metric preservation - compares kNN edge lengths:
```python
L_edge = mean((D_pred[edges] - D_target[edges])²)
```

### 6.5 kNN NCA Loss (weight=2.0)

Neighborhood consistency:
```python
# Maximize probability of true neighbors under predicted distances
L_nca = -log(softmax(-d²/τ)[true_neighbors])
```

### 6.6 Generator Losses (weights=10.0 each)

- `gen_align`: Procrustes alignment MSE
- `gen_scale`: RMS scale matching

---

## 7. V_GEOM CLAMPING

The geometry losses operate on clamped predictions:

```python
# Compute scale ratio
ratio = RMS(V_hat) / RMS(V_target)

# Clamp correction factor
max_log_corr = 0.50  # Max 1.65x correction
s_corr = exp(clamp(log(1/ratio), -max_log_corr, max_log_corr))

# Apply correction
V_geom = V_hat_centered * s_corr
```

**Effect**: Geometry losses see normalized/bounded predictions, NOT the raw (possibly under-scaled) output. This masks the scale problem from geometry losses.

---

## 8. RECENT TRAINING LOGS (Epoch 216-220)

### 8.1 Per-Sigma Breakdown

```
[SCALE-BY-SIGMA] @ step 86600:
  σ∈[0.00,0.10): n=5  | scale_r=0.972 | scale_loss=0.00
  σ∈[0.10,0.30): n=10 | scale_r=0.733 | scale_loss=0.08
  σ∈[0.30,0.50): n=9  | scale_r=0.427 | scale_loss=0.35
  σ∈[0.50,1.00): n=16 | scale_r=0.275 | scale_loss=0.55
  σ∈[1.00,∞):   n=24 | scale_r=0.148 | scale_loss=0.73
```

### 8.2 Generator Performance

```
[GENERATOR-HEALTH] Jacc@10=0.345 | scale_r=0.959 | scale_loss=0.00
```

Generator is healthy. If we use Generator + light refinement, this may be fine.

### 8.3 Open-Loop Sampling (Which We Don't Plan to Use)

```
[PROBE-SAMPLE-END] knn10=0.039, nearmiss_p50=5.50
    → ⚠️ Open-loop sampling fails, structure degrades instead of improving
```

### 8.4 A/B Conditioning Test

```
[ABLATION_FX-A/B] @ step 86800
  (High-σ samples, c_skip < 0.05)
  H IS being used: rel_diff=0.135 > 0.10 threshold  ✓
  Fx_ratio = 0.728 (how F_x changes when H ablated)
  Issue is likely objective/weighting, not conditioning pipeline.
```

---

## 9. THE REAL QUESTION FOR THE RESEARCH AGENT

Given that:
1. **We use Generator + Refiner**, not full denoising from high noise
2. **Generator works great** (scale_r ≈ 0.96, Jacc@10 ≈ 0.34)
3. **Low-σ denoising works** (scale_r ≈ 0.97 at σ < 0.1)
4. **We care about intrinsic geometry** (relative positions), not absolute scale
5. **V_target is already intrinsic** (factored from Gram matrix)

**Questions**:

1. **Is "scale collapse at high σ" even a problem for us?** If we only refine from σ ≈ 0.15-0.30, we might be fine.

2. **If it IS a problem**: Why does F_x learn to output 20-40x smaller values at high σ? The score loss should penalize this, but it's not learning correct magnitudes.

3. **Should the training objective be scale-invariant?** Since we care about relative positions, maybe we should:
   - Normalize predictions and targets before computing losses
   - Use only rank-based metrics (Jaccard, kNN overlap)
   - Remove gram_scale, out_scale losses

4. **Is there a simpler approach?** Maybe we're over-engineering with all these losses. The generator works great with just gen_align + gen_scale. Why is the denoiser so much harder?

5. **Architecture question**: The generator (2 ISAB blocks, direct prediction) works perfectly. The score net (4 ISAB blocks, FiLM conditioning, noise prediction) fails at high σ. Is the denoiser architecture appropriate for this task?

6. **EDM preconditioning**: The standard EDM formulas assume image-like data. For point cloud geometry, is this the right formulation? Should we use a different preconditioning?

---

## 10. CODE LOCATIONS

| Component | File | Lines |
|-----------|------|-------|
| V_target from Gram | core_models_et_p2.py | 4572-4580 |
| DiffusionScoreNet | core_models_et_p2.py | 936-1250 |
| MetricSetGenerator | core_models_et_p2.py | 774-850 |
| Score loss | core_models_et_p2.py | 5316-5600 |
| Gram/geometry losses | core_models_et_p2.py | 6640-7500 |
| V_geom clamping | core_models_et_p2.py | 6050-6250 |
| EDM preconditioning | utils_et.py | 366-427 |
| Loss weights | core_models_et_p2.py | 3549-3580 |

---

## 11. SUMMARY

**What works**: Generator, low-σ denoising, conditioning pipeline

**What "fails"**: Scale at high σ, open-loop sampling

**The question**: For Generator + Refiner with intrinsic geometry, is the high-σ failure actually a problem? Or should we just accept that we don't need full denoising and focus on making low/moderate-σ refinement excellent?

If it IS a problem, the issue appears to be in the training objective - the model learns to output under-scaled F_x at high σ, and nothing in the current loss setup effectively penalizes this (geometry losses are gated to low σ, V_geom is clamped).
