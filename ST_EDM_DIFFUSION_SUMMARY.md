# ST EDM Diffusion Training - Comprehensive Technical Summary

**Purpose**: This document describes the ST (Spatial Transcriptomics) branch of the EDM diffusion training system. It is intended for a research agent to diagnose the scale collapse problem observed in training logs.

---

## 1. PROBLEM STATEMENT

The diffusion model exhibits **scale collapse at high noise levels (σ)**:

| σ | scale_r | trace_r | Target |
|---|---------|---------|--------|
| 0.05 | 0.98 | 0.98 | ~1.0 |
| 0.15 | 0.78 | 0.64 | ~1.0 |
| 0.40 | 0.40 | 0.18 | ~1.0 |
| 0.70 | 0.24 | 0.07 | ~1.0 |
| 1.20 | 0.15 | 0.02 | ~1.0 |
| 2.40 | 0.08 | 0.01 | ~1.0 |

**Consequences**:
- Open-loop sampling fails: kNN@10 ≈ 0.04-0.05 (should improve through chain)
- Curriculum stalls at stage 4/6 with 0/6 structure passes
- Model outputs predictions 3-10x smaller than targets at high σ
- Blob formation: `dist_ratio ≈ 0.3-0.5` (should be ~1.0)

**What works**:
- Generator: scale_r ≈ 0.96, Jacc@10 ≈ 0.31 (good)
- Low-σ diffusion: σ < 0.1 has Jaccard 0.5-0.8
- One-step denoise beats baseline (ratio ~0.09)

---

## 2. ARCHITECTURE OVERVIEW

### 2.1 Data Flow

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

### 2.2 Score Network / Denoiser (`DiffusionScoreNet`)

**Location**: `core_models_et_p2.py:936`

**Architecture**:
- Input projection: Concatenates `[H, t_emb, V_in, angle_features, self_cond_features]`
- Stack of 4 ISAB blocks (Induced Set Attention) with FiLM modulation
- FiLM layers: Per-block `[H, t_emb] → γ, β` for adaptive modulation
- Output head: MLP → ε̂ (predicted noise)

**Key Features**:
- **Canonical centering**: Centers V_t to zero mean before processing
- **Distance bias**: RBF kernels from pairwise distances in attention
- **Self-conditioning**: Can use prior prediction as auxiliary input
- **Angle features**: Local geometry orientation (8 bins)

**Forward signature**:
```python
def forward(self, V_t, t, H, mask, self_cond=None, x_raw=None, c_in=None):
    # V_t: (B, N, D_latent) noisy coordinates
    # t: (B,) time embedding input
    # H: (B, N, c_dim) context
    # Returns: ε̂ (B, N, D_latent) predicted noise
```

### 2.3 Generator (`MetricSetGenerator`)

**Location**: `core_models_et_p2.py:774`

**Architecture**:
- 2 ISAB blocks
- MLP head: c_dim → c_dim → D_latent
- Row-mean centering (translation invariance)

**Purpose**: Provides initial structure estimate. Works well (scale_r ≈ 0.96).

### 2.4 Context Encoder (`SetEncoderContext`)

**Location**: `core_models_et_p2.py:579`

- 3 ISAB blocks
- Input: Z embeddings from frozen encoder
- Output: H context features for denoiser and generator

---

## 3. EDM PRECONDITIONING

### 3.1 EDM Formulation (Karras et al. 2022)

The score network operates in a preconditioned space:

```python
# Preconditioning scalars (σ_data ≈ 0.16 for this dataset)
c_skip = σ_data² / (σ² + σ_data²)        # Skip connection weight
c_out = σ · σ_data / √(σ² + σ_data²)     # Output scaling
c_in = 1 / √(σ_data² + σ²)               # Input scaling
c_noise = log(σ) / 4                      # Time embedding

# Network prediction
F_x = score_net(c_in · V_t, c_noise, H, mask)

# Final denoised output
V_hat = c_skip · V_t + c_out · F_x
```

**Critical insight**: At high σ, `c_skip → 0`, so `V_hat ≈ c_out · F_x`. The learned branch F_x must carry correct geometry AND scale.

### 3.2 EDM Loss Weight

```python
λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
```

This weight emphasizes low-noise samples. Additional compensation is applied:
- High-σ multiplier: 3x for σ > 0.5
- g(σ) boost: `(σ/σ₀)^2` for σ > 0.5

---

## 4. SIGMA SAMPLING (CURRICULUM)

### 4.1 Curriculum Stages

| Stage | σ_cap multiplier | σ_cap (σ_data=0.16) |
|-------|------------------|---------------------|
| 0 | 0.3x | 0.05 |
| 1 | 0.9x | 0.15 |
| 2 | 2.3x | 0.37 |
| 3 | 4.0x | 0.64 |
| 4 | 7.0x | 1.12 |
| 5 | 14.0x | 2.24 |
| 6 | 17.0x | 2.72 |

**Current state**: Stuck at stage 4, cannot pass promotion thresholds.

### 4.2 Cap-Band Sampling

Each batch samples σ from a mixture:
- **Cap-band** (70% at late stages): Uniform in log-space over [0.7·σ_cap, σ_cap]
- **Full-range** (30%): Log-normal clamped to [σ_min, σ_cap]

```python
# Log-normal parameters
P_mean = -1.2
P_std = 1.2
ln(σ) ~ N(P_mean, P_std²), clamped to [σ_min, σ_cap]
```

### 4.3 Promotion Thresholds

For each stage, promotion requires 5/6 consecutive passes:
- `scale_r ≥ threshold` (0.72-0.88 depending on stage)
- `trace_r ≥ threshold` (0.49-0.74 depending on stage)
- `Jacc@10 ≥ threshold` (0.07-0.15 depending on stage)

---

## 5. TRAINING LOOP

### 5.1 Batch Construction

```python
# ST miniset sampling
- Sample random patches from ST slides (n_min to n_max points)
- Stochastic kNN-based spatial locality
- Returns: Z_set, V_target, G_target, D_target, mask, n
```

**Key tensors**:
- `V_target`: (B, N, D_latent) ground truth coordinates, **factored from Gram matrix** and canonicalized
- `G_target`: (B, N, N) Gram matrix (V·V^T / N)
- `mask`: (B, N) validity mask

### 5.2 Forward Pass

```python
# 1. Encode context
H = context_encoder(Z_set, mask)  # (B, N, c_dim)

# 2. Sample sigma from curriculum
σ ~ curriculum_mixture(σ_cap)

# 3. Add noise
V_t = V_target + σ · ε,  where ε ~ N(0, I)

# 4. Optional: Two-pass self-conditioning
with torch.no_grad():
    x0_pred_0 = score_net.forward_edm(V_t, σ, H, mask, self_cond=None)
x0_pred = score_net.forward_edm(V_t, σ, H, mask, self_cond=x0_pred_0)

# 5. Compute losses
V_hat = x0_pred  # Denoised prediction
```

### 5.3 V_geom Clamping

A critical mechanism to prevent geometry losses from seeing unrealistic predictions:

```python
# Compute scale ratio from kNN edge lengths
ratio_raw = RMS(V_hat_centered) / RMS(V_target)

# Clamp correction factor
max_log_corr = 0.50  # Maximum 1.65x scale correction
s_corr = exp(clamp(log(1/ratio), -max_log_corr, max_log_corr))

# Apply correction
V_geom = V_hat_centered * s_corr
```

**Problem observed**: Clamp pin-rate was 93.8% at stage 3 (now 15.6% at stage 4), meaning model relied heavily on clamping.

---

## 6. LOSS FUNCTIONS (NON-ZERO WEIGHTS)

### 6.1 Score Loss (weight=16.0)

**Primary denoising objective**:

```python
# Per-sample MSE
err2_sample = masked_mean((x0_pred - V_target)², mask)  # (B,)

# EDM weighting with high-σ boost
w = λ(σ) · g(σ) · boost(σ)

# Weighted average (WNORM enabled)
L_score = (w · err2_sample).sum() / w.sum()
```

**High-σ compensation**:
- `g(σ) = (σ/0.5)²` for σ > 0.5
- `HI_SIGMA_MULT = 3.0` for σ > 0.5
- Adaptive boost gate (target 25% highest-noise samples)

### 6.2 Gram Loss (weight=2.0)

**Global structure preservation**:

```python
# Gram matrices
G_pred = V_geom @ V_geom.T / N
G_target = V_target @ V_target.T / N

# Frobenius norm (masked)
L_gram = ||G_pred - G_target||_F²
```

**Gating**: Adaptive quantile gate, target 50% of samples (low-noise preference)

### 6.3 Gram Scale Loss (weight=2.0)

```python
# Trace ratio
trace_pred = trace(G_pred)
trace_target = trace(G_target)
L_gram_scale = (trace_pred - trace_target)² / trace_target²
```

**Gating**: Adaptive quantile gate, target 60%

### 6.4 Out Scale Loss (weight=2.0)

**Direct scale supervision**:

```python
# RMS matching
rms_pred = sqrt(mean(V_hat²))
rms_target = sqrt(mean(V_target²))
L_out_scale = (rms_pred - rms_target)² / rms_target²
```

**Gating**: Applied when c_skip < 0.25 (high-σ samples)

### 6.5 Edge Loss (weight=4.0)

**Local metric preservation**:

```python
# Build kNN graph from V_target
edges = knn_graph(V_target, k=12)

# Edge length MSE
D_pred = pairwise_distances(V_geom)[edges]
D_target = pairwise_distances(V_target)[edges]
L_edge = mean((D_pred - D_target)²)
```

**Gating**: Adaptive quantile gate, target 30%

### 6.6 kNN NCA Loss (weight=2.0)

**Neighborhood consistency via Noise Contrastive Approximation**:

```python
# For each point i:
#   - Get k=10 spatial nearest neighbors from V_target
#   - Compute softmax over predicted distances
#   - Maximize probability of true neighbors

τ = (15th-NN distance)²  # Temperature from data statistics

# NCA objective
L_nca = -log(sum_j∈N(i) exp(-d(i,j)² / τ))
```

**Gating**: Adaptive quantile gate, target 40%

### 6.7 kNN Scale Loss (weight=0.2)

**Scale calibration via neighbor distances**:

```python
# Compare median kNN distances
D_knn_pred = knn_distances(V_hat_centered, k=10)
D_knn_target = knn_distances(V_target, k=10)
L_knn_scale = (median(D_knn_pred) - median(D_knn_target))²
```

### 6.8 Generator Losses

**Gen Align (weight=10.0)**: Procrustes alignment between generator output and target

```python
# Optimal alignment
W, b = procrustes(V_gen, V_target)
V_aligned = V_gen @ W + b
L_gen_align = ||V_aligned - V_target||²
```

**Gen Scale (weight=10.0)**: Scale matching

```python
L_gen_scale = (RMS(V_gen) - RMS(V_target))²
```

### 6.9 Subspace Loss (weight=0.5)

**Low-rank structure preservation**:

```python
# Penalize variance outside top-k singular values
L_subspace = variance_outside_topk(V_struct, mask, k=2)
```

### 6.10 Context Edge Loss (weight=0.05)

**Invariance to context perturbation**:

```python
# Replace extra tokens with donors from another sample
Z_rep = replace_extras(Z_set, donor_idx)
H_rep = context_encoder(Z_rep, mask)

# Forward with replacement context
x0_pred_rep = score_net(V_t, σ, H_rep, mask)

# Edge-based invariance (core points only)
L_ctx_edge = edge_invariance(x0_pred, x0_pred_rep, core_mask)
```

---

## 7. ADAPTIVE GATING SYSTEM

Each geometry loss uses an adaptive quantile gate:

```python
class AdaptiveQuantileGate:
    # Maintains running quantile of noise_score = -log(c_skip)
    # Passes samples below (mode="low") or above (mode="high") threshold

    # Loss-specific configurations:
    "gram":       target_rate=0.50, mode="low"   # Half of samples
    "gram_scale": target_rate=0.60, mode="low"
    "edge":       target_rate=0.30, mode="low"
    "nca":        target_rate=0.40, mode="low"
    "learn_hi":   target_rate=0.50, mode="high"  # High-noise samples
```

---

## 8. KEY DIAGNOSTIC METRICS FROM LOGS

### 8.1 Per-Sigma Scale Analysis

```
[DEBUG-PRECOND] Step 4100 - Per-sigma preconditioning analysis:
  σ∈[0.0,0.1): n=1, scale_ratio=0.960   ← Good
  σ∈[0.1,0.3): n=2, scale_ratio=0.683   ← Degrading
  σ∈[0.3,0.5): n=2, scale_ratio=0.610
  σ∈[0.5,1.0): n=14, scale_ratio=0.509  ← Severe
  σ∈[1.0,inf): n=13, scale_ratio=0.510  ← Severe
```

### 8.2 F_x Space Analysis

```
[DEBUG-OUT-SCALE-FX] Step 4100 - F_x space scale analysis:
  σ∈[0.00, 0.10): Fx_pred/Fx_tgt = 0.0646   ← Learned branch too small
  σ∈[0.30, 0.50): Fx_pred/Fx_tgt = 0.0343
  σ∈[0.70, 1.20): Fx_pred/Fx_tgt = 0.0265   ← 40x too small!
```

**Critical**: At high σ where `c_skip ≈ 0`, the output is `V_hat ≈ c_out · F_x`. If `F_x` is 40x too small, the output will be severely under-scaled.

### 8.3 Hidden Error Analysis

```
[CLAMP-HIDDEN-ERROR] step=4100
  σ-range    | raw_ratio | geom_ratio | hidden_err
  low[0.0,0.3)  |     7.024 |      0.440 |      2.770
  high[0.7,1.5) |     1.003 |      0.173 |      1.756
```

The clamp is masking 1.756 units of log-scale error at high σ.

### 8.4 Sampling Trajectory

```
[PROBE-SAMPLE-TRACE] epoch=181 sigma=3.000 knn10=0.057
[PROBE-SAMPLE-TRACE] epoch=181 sigma=0.200 knn10=0.048 ← Gets WORSE

[PROBE-SAMPLE-END] knn10=0.047, nearmiss_p50=5.52
⚠️ Open-loop sampling fails! Structure not recovered from noise.
```

The sampling chain degrades structure rather than improving it.

---

## 9. HYPOTHESIZED ROOT CAUSES

1. **Learned branch (F_x) under-scaled at high σ**: The network outputs F_x that is 20-40x too small when c_skip ≈ 0. Since V_hat = c_skip·V_t + c_out·F_x, and c_skip→0 at high σ, the output is dominated by an under-scaled F_x.

2. **Loss signal insufficient at high σ**: Despite high-σ boosting (3x multiplier, additional compensation), the gradient signal may still be dominated by low-σ samples where prediction is easy.

3. **Geometry losses masked by clamping**: The V_geom clamp hides the true scale error from geometry losses. While L_out_scale sees raw error, it may be insufficient.

4. **Missing direct F_x supervision**: There's EXP_SCORE_FX_HI but it may not be strong enough. The learned branch needs explicit scale supervision independent of the skip connection.

5. **EDM preconditioning mismatch**: The c_out scaling may be inappropriate for this geometry task. Standard EDM assumes image-like data.

---

## 10. RELEVANT CODE LOCATIONS

| Component | File | Lines |
|-----------|------|-------|
| Training loop | core_models_et_p2.py | 2672-7500+ |
| DiffusionScoreNet | core_models_et_p2.py | 936-1250 |
| MetricSetGenerator | core_models_et_p2.py | 774-850 |
| SetEncoderContext | core_models_et_p2.py | 579-678 |
| Score loss computation | core_models_et_p2.py | 5316-5600 |
| Geometry losses | core_models_et_p2.py | 6640-7500 |
| V_geom clamping | core_models_et_p2.py | 6050-6250 |
| EDM preconditioning | utils_et.py | 366-427 |
| Loss weights | core_models_et_p2.py | 3549-3580 |

---

## 11. CURRENT LOSS WEIGHTS

```python
WEIGHTS = {
    'score': 16.0,
    'gram': 2.0,
    'gram_scale': 2.0,
    'out_scale': 2.0,
    'gram_learn': 2.0,
    'knn_scale': 0.2,
    'knn_nca': 2.0,
    'edge': 4.0,
    'gen_align': 10.0,
    'gen_scale': 10.0,
    'subspace': 0.5,
    'ctx_edge': 0.05,
    # All others = 0.0
}
```

---

## 12. QUESTIONS FOR RESEARCH AGENT

1. Why does the learned branch F_x predict 20-40x smaller outputs at high σ?
2. How can we directly supervise F_x to have correct scale independent of c_skip?
3. Should we modify EDM preconditioning for this non-image geometry task?
4. Is the loss weighting/gating strategy appropriate for high-σ training?
5. Would a different noise schedule help (e.g., variance-preserving instead of variance-exploding)?
6. Should geometry losses operate on F_x directly rather than the final V_hat?
