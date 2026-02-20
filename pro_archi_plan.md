# GEMS Stage C Architecture Design Document

## 1. Executive Recommendation

According to a document from 2026-02-19, Stage C must consume frozen shared-encoder embeddings ($H$) (Stage A), train only on precomputed *intrinsic* geometric targets (no raw ST coordinates), and be supervised primarily by multi-step random-walk/diffusion operator matching plus a secondary multiscale stress term with fixed weighting $L = L_{\text{rw}} + 0.1\,L_{\text{stress}}$.

### Context from GEARS v2 (KDD-era) that matters for Stage C design

- GEARS v2's earlier geometry stage was "Gram / ground-matrix era" with a generator + diffusion refiner and patchwise inference + stitching; the *procedural* insight that still transfers is: **miniset-sized prediction + distance-first stitching** is robust and scalable for large scRNA.
- Stage A has changed: the new plan emphasizes mean-centering and VICReg + spatial NCE to preserve locality while removing slide identity, plus a post-hoc scRNA adapter (whitening/coloring OT initialization + CORAL/MMD fine-tuning). Stage C should therefore assume $H$ already encodes local spatial structure but is gauge-free.
- The targets have changed: we now supervise geometry intrinsically via **random-walk transition matrices** (multi-step) and **multiscale stress edges/distances**, with adaptive $k$, adaptive RW steps, and landmark edges baked into the target builder.

### Recommended Trio for the NeurIPS Submission

**Flagship (novelty + strongest NeurIPS "story"):**
**Family 1 — IPA-style Geometric Attention with Stochastic Recycling (Bucket 1)**
*Why this is the flagship:* it gives a clean narrative: *"structure module for intrinsic geometry under gauge ambiguity"*. AlphaFold's IPA is the canonical "learn a geometry by iterated attention + coordinate refinement," but we adapt it to unordered 2D point sets with Stage A embeddings and *intrinsic* supervision (RW+stress). Recycling gives a principled way to learn multiscale structure with miniset sizes and provides a natural mechanism for test-time refinement on scRNA patches.

**Workhorse baseline (lowest risk, easiest to make work):**
**Family 2 — EGNN Coordinate Generator + Distance Uncertainty Head (Bucket 2)**
*Why this is the baseline:* EGNN is stable, permutation-equivariant, and enforces the right SE(2) behavior. The auxiliary distance head directly uses the multiscale stress targets (and improves stitching), while the universal RW loss pushes diffusion structure. This will almost certainly train and gives a strong ablation anchor.

**Optional energy-based variant (for an additional "principled modeling" angle):**
**Family 3 — Amortized Optimization / Learned Gradient Flow (Neural ODE/SDE) (Bucket 3)**
*Why keep this as an optional variant:* it explicitly ties Stage C to minimizing the fixed intrinsic energy (RW+stress) during training, then learns an amortized dynamical system that reproduces that behavior from $H$ alone at test time (no targets). This is the cleanest way to tell an "energy-based intrinsic geometry learner" story without requiring coordinate regression to pixels.

### Shared Inference Strategy Across the Trio (and Recommended for the Paper): Distance-First Stitching

- Even if the network outputs coordinates for each patch, we recommend **stitching using distances** (robustly aggregated across overlapping patches), then solving a global distance-geometry embedding. This avoids gauge alignment issues and mirrors what GEARS v2 already validated procedurally.

### Paper Positioning

- Core claim: we learn an **intrinsic geometry** $\mathcal{G}$ (neighborhoods / diffusion / relative distances) from expression-only inputs, where coordinates are merely a *latent gauge*. Training never sees ST pixel coordinates; it sees only intrinsic targets derived from them.
- Evaluation: use gauge-invariant metrics (RW operator error, stress error, kNN overlap, diffusion-distance correlations), and for scRNA report downstream neighborhood/diffusion analyses using the predicted operator/graph/coordinates.

---

## 2. "Do We Have to Generate Coordinates?"

Let $H \in \mathbb{R}^{n \times d}$ be the frozen Stage A embeddings for a miniset of $n$ spots/cells. Stage C must output a geometry object that is identifiable only up to **gauge** (at least similarity transforms).

### Option 1: Output Coordinates $X \in \mathbb{R}^{n \times 2}$

We treat coordinates as a *chart* on an intrinsic manifold, defined only up to similarity. A standard gauge fix is

$$
\tilde{X} = \frac{X - \frac{1}{n}\mathbf{1}\mathbf{1}^\top X}{\mathrm{RMS}(X - \bar{X}) + \epsilon}, \qquad \mathrm{RMS}(Y) = \sqrt{\frac{1}{n}\sum_i |y_i|^2}.
$$

This is exactly the normalization used in the universal loss pipeline.

From $\tilde{X}$, you can compute:

- distances $d_{ij} = |\tilde{x}_i - \tilde{x}_j|$,
- affinities $W_{ij} = \exp\!\big(-d_{ij}^2 / (\sigma_i \sigma_j + \epsilon)\big)$ on target kNN edges,
- diffusion operator $T = D^{-1}W$.

**Pros:** simplest downstream use (visualization, kNN, diffusion); easy to plug into universal RW+stress losses.

**Cons:** coordinates are non-identifiable; patch outputs need alignment or distance-first stitching.

### Option 2: Output Sparse Distances (Edge Distances + Uncertainty)

Let $E$ be an edge set (e.g., kNN + multiscale edges). Output

$$
\{(\hat{d}_{ij}, \hat{s}_{ij}) : (i,j) \in E\},
$$

where $\hat{s}_{ij}$ is an uncertainty (e.g., log-variance).

**Pros:** distances are gauge-invariant; stitching becomes natural (aggregate edge distances across patches); directly leverages stress targets (which are distances on multiscale edges).

**Cons:** partial distances may not correspond to a valid Euclidean distance matrix (EDM); need an embedding solver (SMACOF / stress majorization) or keep it as a graph distance object.

### Option 3: Output an Operator $T$ (or $W/L$)

Output a row-stochastic operator $T$ on a graph:

$$
T_{ij} = \frac{\exp(\ell_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(\ell_{ik})}, \quad j \in \mathcal{N}(i).
$$

**Pros:** matches the primary supervision object directly (multi-step RW transitions).

**Cons:** downstream "distance" is not unique. Choices include diffusion distance or commute time:

$$
d_{\text{diff}}^{(s)}(i,j) = \|T^s(i,:) - T^s(j,:)\|_2,
$$

or an embedding from Laplacian eigenvectors. Stitching across patches is easy for sparse $T$, but coordinates (if needed) require spectral methods or a learned decoder.

### Recommended Stance for a NeurIPS Story

**Yes, it is acceptable to generate coordinates as a latent gauge**, as long as:

- supervision is intrinsic (RW+stress), never pixel-coordinate regression,
- evaluation is intrinsic (operator/distance metrics),
- large-$n$ inference uses **distance-first stitching** (gauge-free) even if the model internally predicts coordinates.

The paper framing can be: **"intrinsic geometry learning under gauge ambiguity."** Coordinates are a convenient representation; the learned object is the equivalence class $[X]$ under similarity transforms, or more abstractly the induced diffusion operator / neighborhood structure.

---

## 3. Architecture Family 1 (Bucket 1) — IPA-style Geometric Attention with Stochastic Recycling (IPA-R)

### 3.1 Modeling Object + Probabilistic Formulation

- **Random variable:** latent coordinates $X \in \mathbb{R}^{n \times 2}$ (gauge-free up to similarity), optionally with per-edge uncertainty.
- **Model family (implicit conditional distribution via recycling):**

$$
z \sim \mathcal{N}(0, I), \quad X^{(0)} = g_{\theta}^{\text{init}}(H, z), \quad X^{(r)} = f_{\theta}^{\text{rec}}(H, X^{(r-1)};\,\eta_r),\; r = 1, \dots, R,
$$

where $\eta_r$ are dropout / stochastic depth masks ("stochastic recycling"). The induced distribution is

$$
p_\theta(X \mid H) = \int \delta\!\big(X - f^{(R)}_\theta(H, z)\big)\,p(z)\,dz.
$$

- **Invariances/equivariances**
  - **Permutation equivariant:** the model is a set/graph attention module.
  - **SE(2)-equivariant coordinate updates:** coordinate updates use only relative vectors $(x_j - x_i)$ and distances $|x_i - x_j|$, so translation equivariance and rotation equivariance hold.
  - **Scale/gauge handling:** final losses are computed on $\tilde{X}$ after centering + unit-RMS normalization, making the objective similarity-invariant.

### 3.2 Full Math Specification

**Inputs**

- $H = [h_1, \dots, h_n]^\top$, $h_i \in \mathbb{R}^d$ from frozen Stage A encoder.
- A graph $\mathcal{E}$ for attention/message passing during Stage C. Use $\mathcal{E} = \text{kNN}(H)$ (available at inference), not GT spatial edges, to reduce train–test mismatch.

**Geometric Attention Block (2D IPA-inspired)**

- Define attention logits with a geometric bias:

$$
\ell_{ij} = \frac{(W_Q f_i)^\top (W_K f_j)}{\sqrt{d_f}} - \gamma |x_i - x_j|^2 + b_{ij}, \qquad a_{ij} = \text{softmax}_{j \in \mathcal{N}(i)}(\ell_{ij}),
$$

where $f_i$ is the current feature state, $b_{ij}$ optional edge bias from $\text{MLP}([h_i, h_j, |x_i - x_j|])$.

- Feature update:

$$
m_i = \sum_{j \in \mathcal{N}(i)} a_{ij}\, W_V f_j, \quad f_i \leftarrow \mathrm{LN}\big(f_i + m_i\big), \quad f_i \leftarrow \mathrm{LN}\big(f_i + \mathrm{FFN}(f_i)\big).
$$

- Coordinate update (EGNN-style, attention-weighted):

$$
x_i \leftarrow x_i + \sum_{j \in \mathcal{N}(i)} a_{ij}\;\phi_\theta(f_i, f_j, |x_i - x_j|)\,(x_j - x_i).
$$

**Losses (≥4 terms)**

**(i) Fixed geometric supervision (universal pipeline):**

Let $\tilde{X}^{(r)} = \text{center+unitRMS}(X^{(r)})$. Compute

$$
L_{\text{geo}}(X^{(r)};\text{targets}) = L_{\text{rw}}(\tilde{X}^{(r)}) + 0.1\,L_{\text{stress}}(\tilde{X}^{(r)}),
$$

where $L_{\text{rw}}$ matches multi-step transitions via KL and $L_{\text{stress}}$ matches multiscale distances, using the target-provided $\sigma_{\text{local}}$, kNN edges, multiscale edges, and reference distances.

**(ii) Recycling deep supervision (stability):**

$$
L_{\text{recycle}} = \sum_{r=1}^{R} w_r\,L_{\text{geo}}(X^{(r)};\text{targets}),
$$

with $w_r$ increasing (e.g., $w_r \propto r$) so early recycles learn coarse structure.

**(iii) Step-size/kinetic regularizer (prevents exploding updates):**

$$
L_{\text{step}} = \sum_{r=1}^{R} \frac{1}{n} \|X^{(r)} - X^{(r-1)}\|_F^2.
$$

**(iv) Overlap-consistency regularizer (improves patch stitching):**

Sample two overlapping minisets $A, B$ from the same slide during training (as in GEARS-style overlap training). For the overlap index set $O = A \cap B$,

$$
L_{\text{ov}} = \frac{1}{|O|^2} \sum_{i,j \in O} \left(\frac{|\tilde{x}^A_i - \tilde{x}^A_j|}{|\tilde{x}^B_i - \tilde{x}^B_j| + \epsilon} - 1\right)^2.
$$

(Distance-ratio makes it similarity-invariant.)

**(v) Anti-collapse / diversity:** repulsive potential on short distances

$$
L_{\text{rep}} = \frac{2}{n(n-1)} \sum_{i < j} \exp\!\left(-\frac{|\tilde{x}_i - \tilde{x}_j|^2}{\tau^2}\right).
$$

**Total**

$$
L = L_{\text{recycle}} + \lambda_{\text{step}} L_{\text{step}} + \lambda_{\text{ov}} L_{\text{ov}} + \lambda_{\text{rep}} L_{\text{rep}}.
$$

### 3.3 Training Algorithm

```
for each training step:
    sample miniset (or overlapping pair of minisets) with:
        H, geo_targets = build_miniset_geometric_targets(...)
    build graph E_msg = kNN(H)  # inference-available topology

    z ~ N(0, I)
    X0, F0 = InitNet(H, z)

    for r in 1..R:
        (Xr, Fr) = IPA_BlockStack(H, X_{r-1}, F_{r-1}, E_msg, dropout_masks η_r)
        compute L_geo(Xr; geo_targets) via universal pipeline
        accumulate weighted deep supervision

    if overlap pair:
        compute L_ov using overlap indices

    compute L_step, L_rep
    backprop through (optionally checkpointed) recycling stack
    optimizer.step()
```

**Backprop details / feasibility**

- For $n \le 384$, computing dense $T^s$ (after sparse-to-dense propagation) is feasible; otherwise checkpoint and evaluate $L_{\text{geo}}$ only at selected recycles.
- Use mixed precision; detach intermediate $X^{(r)}$ between recycles if memory is tight (still works because deep supervision remains on later steps).
- RW loss requires **fixed $\sigma_{\text{local}}$ from targets** (do not recompute from $X$).

### 3.4 scRNA Inference Algorithm (Large $n_{\text{sc}}$)

**Recommended: patch inference + distance-first stitching (GEARS-style)**

1. Compute $H_{\text{sc}}$ for all cells with Stage A (including the SC adapter).
2. Build a locality graph $G_H$ (mutual kNN + optional Jaccard filtering) on $H_{\text{sc}}$ for patch sampling (not a supervision target). This mirrors the patch-sampling strategy used in GEARS v2.
3. Sample overlapping patches $P_1, \dots, P_M$ via random walks on $G_H$, patch size $n_p \approx 256$, enforce min overlap (e.g., 64).
4. For each patch $P_m$, run IPA-R to produce local coordinates $X^{(m)}$ (gauge-fixed internally by centering+RMS when extracting distances).
5. Extract distance measurements $\hat{d}^{(m)}_{ij} = |\tilde{x}^{(m)}_i - \tilde{x}^{(m)}_j|$ for edges within the patch (use a fixed small edge set, e.g., kNN within patch).
6. Aggregate across patches into a global sparse distance graph:

$$
\hat{d}_{ij} = \mathrm{median}\{\hat{d}^{(m)}_{ij} : (i,j) \in P_m\}, \quad w_{ij} = \mathrm{IQR}^{-1} \text{ or overlap-count}.
$$

7. Solve global coordinates $\hat{X}$ by weighted stress majorization:

$$
\min_{X \in \mathbb{R}^{n_{\text{sc}} \times 2}} \sum_{(i,j) \in E_{\text{glob}}} w_{ij}\big(|x_i - x_j| - \hat{d}_{ij}\big)^2,
$$

then center+unitRMS for gauge.

**Avoiding overlap-shortcut failures**

- Do **not** stitch by forcing coordinates of overlaps to match directly (that can overfit overlaps and distort global geometry); stitch by distances (gauge-invariant).
- During training, keep $\lambda_{\text{ov}}$ small and gate it to late recycles to avoid dominating intrinsic losses.

### 3.5 Implementation Pointers with Official GitHub Repos

- **Invariant Point Attention (IPA) reference (official):**
  - Repo: `google-deepmind/alphafold` — [GitHub](https://github.com/google-deepmind/alphafold)
  - Files to study: `alphafold/model/folding.py` (contains `InvariantPointAttention`, structure module loop / iterative updates).
  - What to reuse vs rewrite:
    - Reuse the *attention-with-geometry-bias* pattern and the *iterative refinement / recycling* training style.
    - Rewrite everything data-structure-related (we have unordered 2D sets, no residue frames), and implement 2D coordinate updates (EGNN-style) rather than rigid-frame updates.

- **Authoritative PyTorch reimplementation (not DeepMind-official; closest authoritative alternative):**
  - Repo: `aqlaboratory/openfold` — [GitHub](https://github.com/aqlaboratory/openfold)
  - File: `openfold/model/structure_module.py` (PyTorch IPA).

- **Set Transformer building blocks (official):**
  - Repo: `juho-lee/set_transformer` — [GitHub](https://github.com/juho-lee/set_transformer)
  - File: `modules.py` (MAB/ISAB/PMA), for permutation-equivariant attention scaffolding.

### 3.6 Expected Strengths/Risks + Diagnostics

**Strengths**

- Strong inductive bias for "learn a geometry by iterated refinement," aligning with intrinsic-gauge narrative.
- Recycling naturally supports multiscale geometry without increasing patch size.
- Outputs coordinates for universal pipeline directly; also supports distance-first stitching.
- Attention can use $H$-space neighborhood graphs (inference-available topology).

**Failure modes (and diagnostics/mitigations)**

1. **Mode collapse to near-regular layouts** (model ignores $H$, outputs generic shapes).
   - Diagnostic: mutual information proxy—correlation between predicted distances and $H$-space similarities; also evaluate RW loss on held-out ST.
   - Mitigation: increase RW weight vs repulsion; add feature-to-geometry conditional loss (e.g., predict local rank order of distances from $H$).

2. **Overfitting to patch size / poor scaling**
   - Diagnostic: train on $n=128$, test on $n=384$ minisets; monitor RW-step errors across steps.
   - Mitigation: randomize $n$ and adaptive $k$/rw_steps (already in targets).

3. **Overlap shortcut during training** (overlap global structure)
   - Diagnostic: compare intrinsic losses with/without overlap regularizer; plot energy vs overlap error.
   - Mitigation: gate overlap to late recycles; cap $\lambda_{\text{ov}}$.

4. **Sensitivity to $H$-graph quality on scRNA**
   - Diagnostic: on ST, replace spatial kNN with $H$-kNN at inference and measure degradation.
   - Mitigation: enlarge patch overlaps and use robust median aggregation; optionally use hierarchical landmarks.

---

## 4. Architecture Family 2 (Bucket 2) — EGNN + Distance Uncertainty Head (EGNN-Dist)

### 4.1 Modeling Object + Probabilistic Formulation

- **Random variables:**
  - Coordinates $X \in \mathbb{R}^{n \times 2}$.
  - Edge distances on multiscale edges: $\hat{d}_e$ and uncertainty $\hat{s}_e$ for $e \in E_{\text{ms}}$.

- **Conditional model:**

$$
(X, \{\hat{d}_e, \hat{s}_e\}) = f_\theta(H, \mathcal{E}_H),
$$

where $f_\theta$ is an EGNN stack (SE(2)-equivariant coordinate updates) and $\mathcal{E}_H$ is a kNN graph in embedding space (available at inference).

- **Invariances:**
  - Permutation equivariant by graph message passing.
  - Translation equivariant + rotation equivariant due to EGNN coordinate update form.
  - Similarity invariance in the loss via unit-RMS normalization inside universal pipeline.

### 4.2 Full Math Specification

**EGNN Layer Updates**

- For edge $(i,j) \in \mathcal{E}_H$, define radial feature $r_{ij} = |x_i - x_j|^2$.
- Message:

$$
m_{ij} = \phi_e([h_i, h_j, r_{ij}]), \quad \alpha_{ij} = \sigma(\phi_a(m_{ij})) \text{ (optional attention)}.
$$

- Coordinate update:

$$
x_i \leftarrow x_i + \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\,\phi_x(m_{ij})\,(x_i - x_j).
$$

- Feature update:

$$
h_i \leftarrow \phi_h\Big(h_i, \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, m_{ij}\Big).
$$

(This matches the standard EGNN formulation in the authors' code.) ([GitHub](https://github.com/vgsatorras/egnn))

**Distance Uncertainty Head**

- For each target multiscale edge $e = (i,j) \in E_{\text{ms}}$ (provided only for loss), predict:

$$
\hat{d}_{ij} = \psi_d([h_i, h_j]), \quad \hat{s}_{ij} = \psi_s([h_i, h_j]),
$$

where $\hat{s}$ parameterizes log-variance.

**Losses (≥4 terms)**

**(i) Fixed geometric supervision:**

$$
L_{\text{geo}}(X) = L_{\text{rw}}(\tilde{X}) + 0.1\,L_{\text{stress}}(\tilde{X})
$$

via universal pipeline.

**(ii) Distance NLL on multiscale edges:**

$$
L_{\text{dist}} = \frac{1}{|E_{\text{ms}}|} \sum_{(i,j) \in E_{\text{ms}}} \left[\frac{(\hat{d}_{ij} - \delta_{ij})^2}{\exp(\hat{s}_{ij}) + \epsilon} + \hat{s}_{ij}\right],
$$

where $\delta_{ij}$ is the target reference distance.

**(iii) Triangle inequality / metric regularizer (sampled triples):**

$$
L_\triangle = \mathbb{E}_{(i,j,k) \sim \mathcal{T}} \left[\max(0, \hat{d}_{ij} - \hat{d}_{ik} - \hat{d}_{kj})\right],
$$

sampling triples from within-patch kNN neighborhoods.

**(iv) Coordinate–distance consistency (ties coordinate chart to distance head):**

$$
L_{\text{cycle}} = \frac{1}{|E_{\text{ms}}|} \sum_{(i,j) \in E_{\text{ms}}} \left(|\tilde{x}_i - \tilde{x}_j| - \hat{d}_{ij}\right)^2.
$$

**(v) Anti-collapse / diversity:** repulsion on $\tilde{X}$ (same as Family 1) or simply enforce spread:

$$
L_{\text{var}} = \left(\mathrm{Var}_i |\tilde{x}_i| - v_0\right)^2.
$$

**Total**

$$
L = L_{\text{geo}} + \lambda_{\text{dist}} L_{\text{dist}} + \lambda_\triangle L_\triangle + \lambda_{\text{cycle}} L_{\text{cycle}} + \lambda_{\text{rep}} L_{\text{rep}}.
$$

### 4.3 Training Algorithm

```
for each step:
    sample miniset with H and geo_targets
    build E_H = kNN(H)  # used by EGNN, available at inference

    initialize X0 ~ N(0, I) or X0 = MLP(H)
    run EGNN layers -> X_pred, node features
    run distance heads -> d_hat, s_hat on E_ms (only for loss)

    compute L_geo(X_pred) via universal pipeline (uses geo_targets edges/σ/transition matrices)
    compute L_dist, L_triangle, L_cycle, L_rep
    backprop and step
```

**Complexity**

- EGNN forward: $O(|\mathcal{E}_H| \cdot d)$ with small $k$ (e.g., 16–32).
- Universal RW loss: manageable for $n \le 384$; use sparse–dense multiplies for $T^s$.

### 4.4 scRNA Inference Algorithm (Large $n_{\text{sc}}$)

- Same as Family 1 (recommended). The distance head provides a **natural per-edge uncertainty**; use $w_{ij} = \exp(-\hat{s}_{ij})$ in global stitching.
- If you need a purely local geometry object (no global solve), output a sparse kNN graph with edge weights $w_{ij} = \exp(-\hat{d}_{ij}^2 / \tau^2)$ and use graph diffusion directly.

### 4.5 Implementation Pointers with Official GitHub Repos

- **EGNN (official authors' repo):**
  - Repo: `vgsatorras/egnn` — [GitHub](https://github.com/vgsatorras/egnn)
  - File: `models/egnn_clean/egnn_clean.py` (core coordinate update + message passing).
  - Reuse: coordinate-update form; attention option; normalize option.
  - Rewrite: replace molecular features with Stage A embeddings; graph construction; add distance uncertainty heads; integrate universal loss pipeline.

- **Graph tooling (authoritative library):**
  - Repo: `pyg-team/pytorch_geometric` — [GitHub](https://github.com/pyg-team/pytorch_geometric)
  - Reuse: kNN graph construction, sparse message passing utilities.

### 4.6 Expected Strengths/Risks + Diagnostics

**Strengths**

- Low implementation risk; stable training.
- Distance head improves stitching and provides uncertainty weighting.
- Strong baseline for ablations (swap EGNN ↔ attention, with/without distance head).

**Failure modes**

1. **Distance head learns but coordinates drift** (cycle inconsistency).
   - Diagnostic: $L_{\text{cycle}}$ and correlation between $\hat{d}_{ij}$ and $|\tilde{x}_i - \tilde{x}_j|$.
   - Mitigation: increase $\lambda_{\text{cycle}}$; add mild coordinate smoothness.

2. **RW loss dominates and hurts metricity**
   - Diagnostic: triangle-inequality violation rate on sampled triples.
   - Mitigation: increase $\lambda_\triangle$; reduce RW weight for early epochs.

3. **Train–test mismatch in graph topology** (spatial kNN vs $H$-kNN).
   - Diagnostic: on ST, compare performance when EGNN uses $H$-kNN vs spatial kNN.
   - Mitigation: always train EGNN on $H$-kNN; keep targets only inside loss.

---

## 5. Architecture Family 3 (Bucket 3) — Learned Gradient Flow (Neural ODE/SDE) Tied to RW+Stress Energy

### 5.1 Modeling Object + Probabilistic Formulation

- **Random variable:** a trajectory $\{X(t)\}_{t \in [0,1]}$, with final $X(1)$ used as geometry.

- **Continuous-time conditional dynamics**
  - Deterministic ODE:

$$
\frac{dX}{dt} = \mu_\theta(X, H, t).
$$

  - Or SDE (for diversity):

$$
dX = \mu_\theta(X, H, t)\,dt + \sigma(t)\,dW_t.
$$

- **Energy tie-in (training-time "teacher"):**

Define the target intrinsic energy on ST minisets:

$$
E_{\text{gt}}(X;\text{targets}) \equiv L_{\text{rw}}(\tilde{X}) + 0.1\,L_{\text{stress}}(\tilde{X}).
$$

The *training objective* pushes $\mu_\theta$ to behave like a descent direction for $E_{\text{gt}}$ even though $\mu_\theta$ depends only on $(X, H, t)$.

### 5.2 Full Math Specification

**Drift Parameterization**

- Use a permutation-equivariant network:

$$
\mu_\theta(X, H, t) = \text{EGNN}_\theta\big([H;\,\text{TimeEmbed}(t)], X, \mathcal{E}_H\big).
$$

**Losses (≥4 terms)**

**(i) Final intrinsic energy (fixed supervision via universal pipeline):**

$$
L_{\text{final}} = E_{\text{gt}}(X(1);\text{targets}).
$$

**(ii) Lyapunov / monotone decrease regularizer (stability):**

For discretized times $t_k$,

$$
L_{\text{mono}} = \sum_k \max\big(0, E_{\text{gt}}(X_{k+1}) - E_{\text{gt}}(X_k) + \gamma\big).
$$

**(iii) Gradient-alignment distillation (amortizes optimization):**

$$
g_k = \nabla_X E_{\text{gt}}(X_k;\text{targets}), \quad L_{\text{align}} = \sum_k \left(1 - \cos\big(\mu_\theta(X_k, H, t_k),\; -g_k\big)\right).
$$

(Compute $g_k$ with autodiff through the universal pipeline; evaluate at a few checkpoints to control cost.)

**(iv) Velocity regularizer (prevents stiff/unstable ODE):**

$$
L_{\text{vel}} = \sum_k \frac{1}{n} \|\mu_\theta(X_k, H, t_k)\|_F^2.
$$

**(v) Anti-collapse / entropy (for SDE variant):**

If using noise, encourage non-degenerate distance distribution:

$$
L_{\text{spread}} = -\mathrm{Var}_{i < j} |\tilde{x}_i - \tilde{x}_j|.
$$

**Total**

$$
L = L_{\text{final}} + \lambda_{\text{mono}} L_{\text{mono}} + \lambda_{\text{align}} L_{\text{align}} + \lambda_{\text{vel}} L_{\text{vel}} + \lambda_{\text{spread}} L_{\text{spread}}.
$$

### 5.3 Training Algorithm

```
for each step:
    sample ST miniset with H and geo_targets
    build E_H = kNN(H)

    initialize X(0) ~ N(0, I) or InitNet(H)
    solve ODE/SDE with K steps (or adaptive solver) to obtain {X_k}
        X_{k+1} = X_k + Δt * μθ(X_k, H, t_k) + (if SDE) sqrt(Δt)*σ(t_k)*ε_k

    compute E_gt(X_k) at a few checkpoints k ∈ K_check
    L_final = E_gt(X_K)
    L_mono  = sum hinge(E_gt(X_{k+1}) - E_gt(X_k) + γ)
    L_align = sum(1 - cos(μθ, -∇E_gt)) at checkpoints
    L_vel, L_spread

    backprop through solver (use checkpointing / adjoint if needed)
    optimizer.step()
```

**Compute/memory**

- Prefer fixed-step solvers with checkpointing ($K = 8$–$16$) to keep memory predictable.
- Gradient-alignment requires computing $\nabla E_{\text{gt}}$; do it at 2–4 checkpoints, not every step.
- If training a diffusion-style model, the tracker suggests applying losses to the denoised $x_{\hat{0}}$ across noise levels; you can implement the same idea by sampling initial noise scales and weighting by $w(\sigma)$.

### 5.4 scRNA Inference Algorithm (Large $n_{\text{sc}}$)

- Inference uses only the learned drift $\mu_\theta(X, H, t)$ for $K$ steps on patches.
- Use the same distance-first stitching pipeline as Family 1.
- If you want an operator output: after global embedding $\hat{X}$, build $T$ via the same kernel form used in training (with a fixed bandwidth heuristic or learned bandwidth predictor; do **not** use GT $\sigma_{\text{local}}$ on scRNA).

### 5.5 Implementation Pointers with Official GitHub Repos

- **Neural ODE reference (official):**
  - Repo: `rtqichen/torchdiffeq` — [GitHub](https://github.com/rtqichen/torchdiffeq)
  - Files: `torchdiffeq/_impl/odeint.py` (solver interface), adjoint method if needed.

- **Score/SDE diffusion reference (official):**
  - Repo: `yang-song/score_sde_pytorch` — [GitHub](https://github.com/yang-song/score_sde_pytorch)
  - Reuse: SDE training scaffolding, noise schedules; rewrite model/backbone for set/graph and plug in our intrinsic energy losses.

- **Energy-based cooperative sampling (official paper code):**
  - Repo: `XingXie/CoopNets` — [GitHub](https://github.com/jianwen-xie/CoopNets)
  - Reuse: Langevin sampling patterns if you want an explicit EBM sampler variant.

### 5.6 Expected Strengths/Risks + Diagnostics

**Strengths**

- Clear "amortized optimization of intrinsic energy" story.
- Natural way to incorporate iterative refinement without coordinate regression.
- Stochastic variant can represent uncertainty/multi-modality in geometry.

**Failure modes**

1. **ODE learns a trivial drift (near zero)**
   - Diagnostic: $|\mu_\theta|$ distribution; energy decrease per step on validation minisets.
   - Mitigation: increase $\lambda_{\text{align}}$; enforce monotone decrease.

2. **Stiff dynamics / exploding steps**
   - Diagnostic: step-to-step $|X_{k+1} - X_k|$ and NaNs.
   - Mitigation: stronger $L_{\text{vel}}$, smaller $K$, clamp updates, use normalize=True EGNN.

3. **Generalization gap to scRNA (no targets)**
   - Diagnostic: on ST, run inference without targets and compare to training-time trajectories.
   - Mitigation: rely more on $H$-graph topology; add dropout/noise in training to simulate scRNA noise.

---

## 6. Architecture Family 4 (Bucket 4) — Operator-first Transition Predictor + Coordinate Decoder (OpNet)

### 6.1 Modeling Object + Probabilistic Formulation

- **Primary random variable:** diffusion operator $T$ (row-stochastic) supported on a sparse neighbor graph.

- **Conditional row-wise logistic-normal model:**

$$
\ell_{ij} = s_\theta(h_i, h_j), \quad T_{ij} = \text{softmax}_{j \in \mathcal{N}(i)}(\ell_{ij}), \quad (i,j) \in \mathcal{E},
$$

where $\mathcal{E}$ is a kNN graph (in training you may use the target-provided knn_edges; in inference use kNN in $H$).

- **Optional coordinate decoder:** $X = g_\omega(H, T)$ (a learned set/graph decoder), used for stress loss and for coordinate outputs.

- **Invariances:** permutation equivariant by construction; no coordinate gauge at primary output.

### 6.2 Full Math Specification

**Operator Prediction**

- Use a GAT/Graphormer-style edge scoring network:

$$
s_\theta(h_i, h_j) = \text{MLP}\left([h_i, h_j, h_i \odot h_j]\right),
$$

or attention score from a transformer block.

**Decoder (optional)**

- Learned decoder that maps node features plus operator rows to coordinates:

$$
u_i = \text{MLP}\left([h_i, T_{i,:} \Pi]\right), \quad X = g_\omega(\{u_i\}),
$$

where $\Pi$ is a learned projection of sparse row into a fixed dimension (e.g., aggregate neighbor embeddings). (Alternatively use a Set Transformer decoder.)

**Losses (≥4 terms)**

**(i) RW operator matching (primary, direct):**

For target multi-step matrices $T_{\text{ref}}^{(s)}$ and predicted powers $T^{(s)}$,

$$
L_{\text{rw-op}} = \sum_{s \in \mathcal{S}} \alpha_s \sum_{i=1}^n \mathrm{KL}\!\left(T_{\text{ref}}^{(s)}[i,:]\;\|\;T^{(s)}[i,:]\right),
$$

matching the same object as the universal pipeline but without going through coordinates.

**(ii) Stress via decoded coordinates (secondary):**

$$
L_{\text{stress}}(X) = \frac{1}{|E_{\text{ms}}|} \sum_{(i,j) \in E_{\text{ms}}} \left(\frac{|\tilde{x}_i - \tilde{x}_j|}{\delta_{ij} + \epsilon} - 1\right)^2,
$$

and required.

**(iii) Row-entropy matching (prevents degenerate transitions):**

$$
L_{\text{ent}} = \frac{1}{n} \sum_i \left(H(T_{i,:}) - H(T_{\text{ref}}^{(1)}[i,:])\right)^2.
$$

**(iv) Detailed balance / symmetry proxy (encourages diffusion-like behavior):**

Let $\pi$ be the stationary distribution estimate (power iteration on $T$, detached). Penalize

$$
L_{\text{db}} = \sum_{(i,j) \in \mathcal{E}} (\pi_i T_{ij} - \pi_j T_{ji})^2.
$$

**(v) Cycle consistency (operator ↔ coordinates):**

Re-encode operator from decoded coordinates with the same kernel form used in universal pipeline but using a learned bandwidth head $\hat{\sigma}(H)$:

$$
T^{\text{coord}} = \text{RowNorm}\Big(\exp(-|x_i - x_j|^2 / (\hat{\sigma}_i \hat{\sigma}_j))\Big), \quad L_{\text{cyc}} = \sum_{(i,j) \in \mathcal{E}} (T_{ij} - T^{\text{coord}}_{ij})^2.
$$

**Total**

$$
L = L_{\text{rw-op}} + 0.1\,L_{\text{stress}} + \lambda_{\text{ent}} L_{\text{ent}} + \lambda_{\text{db}} L_{\text{db}} + \lambda_{\text{cyc}} L_{\text{cyc}}.
$$

### 6.3 Training Algorithm

```
for each step:
    sample ST miniset with H and geo_targets
    choose operator edge set E:
        either geo_targets.knn_edges (matches RW target support)
        or kNN(H) (reduces train-test mismatch)

    compute logits l_ij = score_net(h_i, h_j) on E
    row-softmax -> sparse T_pred
    compute dense T_pred^s for s in rw_steps (small n => OK)

    if using decoder:
        X = decode(H, T_pred)
        compute stress loss on geo_targets multiscale edges

    compute L_rw-op, L_ent, L_db, L_cyc
    backprop and step
```

**Feasibility**

- Training-time dense $T^s$ is fine for $n \le 384$.
- Inference-time on large scRNA: keep $T$ sparse and compute diffusion quantities via sparse multiplications/eigensolvers.

### 6.4 scRNA Inference Algorithm (Large $n_{\text{sc}}$)

- **Option A (operator-only, no coordinates):**
  1. Build kNN graph on $H_{\text{sc}}$.
  2. Predict sparse $T$ on edges.
  3. Downstream: use diffusion distances / diffusion maps / random-walk neighborhoods directly from $T$.

- **Option B (need coordinates): landmark/hierarchical decode**
  1. Choose $L$ landmarks (e.g., k-center in $H$).
  2. Predict $T$ restricted to landmarks; decode landmark coordinates.
  3. Place remaining nodes by barycentric/Nyström extension using transition weights to landmarks.

- **Patch+stitch variant:** predict $T$ (and optionally $X$) on patches and merge sparse operators by averaging logits or probabilities on shared edges; no Procrustes alignment needed for operator merging.

### 6.5 Implementation Pointers with Official GitHub Repos

- **Graph Attention Networks (official):**
  - Repo: `PetarV-/GAT` — [GitHub](https://github.com/PetarV-/GAT)
  - Files: `models/gat.py`, `utils/layers.py` (attention head implementation).
  - Reuse: attention scoring patterns; rewrite in PyTorch (or use PyG GATConv) for integration.

- **Set Transformer (official) for decoder:**
  - Repo: `juho-lee/set_transformer` — [GitHub](https://github.com/juho-lee/set_transformer)
  - File: `modules.py` for ISAB/PMA blocks.

- **PyTorch Geometric (authoritative) for sparse ops:**
  - Repo: `pyg-team/pytorch_geometric` — [GitHub](https://github.com/pyg-team/pytorch_geometric)

### 6.6 Expected Strengths/Risks + Diagnostics

**Strengths**

- Directly predicts the primary target object (RW transitions).
- Operator merging across patches is simpler than coordinate alignment.
- Natural downstream diffusion analyses on scRNA without requiring 2D embedding.

**Failure modes**

1. **Row-stochastic but non-diffusive operators** (bad global geometry even if local KL is good).
   - Diagnostic: check detailed balance error $L_{\text{db}}$; compare diffusion-map embeddings vs ST intrinsic distances.
   - Mitigation: stronger $L_{\text{db}}$ and cycle consistency.

2. **Decoder instability / arbitrary coordinates**
   - Diagnostic: stress loss vs operator loss; correlation between decoded distances and operator-based diffusion distances.
   - Mitigation: emphasize $L_{\text{cyc}}$ and decode via landmarks.

3. **Train–test support mismatch**
   - Diagnostic: train using $H$-kNN vs target knn_edges and measure RW KL.
   - Mitigation: train on $H$-kNN edges for operator support and compute RW loss on that support via masking/renormalization (if allowed); otherwise rely on patching.

---

## 7. Comparison Table

| Family | Modeling Object | Novelty | Implementation Risk | Training Cost (miniset) | Inference Cost (large scRNA) | Stitching Complexity | Expected scRNA Generalization | Exploits RW Targets | Exploits Stress Targets |
|---|---|---|---|---|---|---|---|---|---|
| 1) IPA-R (Bucket 1) | $X$ via geometric attention + recycling | High (structure-module story) | Medium | Medium–High (recycles + RW powers) | Medium (patches) | Medium (distance-first recommended) | High if $H$ locality holds | Indirect (via pipeline) | Direct (via pipeline) |
| 2) EGNN-Dist (Bucket 2) | $X$ + $\hat{d}, \hat{s}$ | Medium | Low | Medium | Medium (patches; uncertainty helps) | Low–Medium | Medium–High | Indirect (via pipeline) | Very direct (distance NLL + pipeline) |
| 3) Learned Gradient Flow (Bucket 3) | Trajectory $X(t)$ | High (energy-amortization) | Medium–High | High (rollout + $\nabla E$ checkpoints) | Medium (patches) | Medium | Medium (depends on drift learning) | Direct (energy) | Direct (energy) |
| 4) OpNet (Bucket 4) | $T$ (operator-first) + optional decoder | Medium–High | Medium | Medium (dense $T^s$ at train) | Low–Medium (sparse $T$ global possible) | Low (operator merge easy) | Medium | Very direct | Indirect unless decoding |

---

## 8. Implementation Blueprint for the Recommended Trio

### Minimal Module Diagram

```
Stage A (frozen):
    expression -> mean-centering -> encoder -> (SC adapter) -> H  (n × d)

Stage C (trainable):
    [Flagship] IPA-R:
        InitNet(H, z) -> (X0, F0)
        R× recycling blocks: geometric attention + coord updates -> X_R
        universal_loss_pipeline(X_R, geo_targets) -> L_rw + 0.1 L_stress

    [Baseline] EGNN-Dist:
        EGNN(H, X0, kNN(H)) -> X
        DistHead(H) -> (d_hat, s_hat) on E_ms (for loss)
        universal_loss_pipeline(X, geo_targets) + L_dist + L_cycle + L_triangle

    [Energy variant] ODE/SDE Flow:
        InitNet(H) -> X0
        rollout: X_{k+1} = X_k + dt * μθ(X_k, H, t_k) (+ noise)
        compute E_gt(X_k) checkpoints via universal pipeline
```

### Step-by-Step Build Order

1. **Lock the universal loss pipeline as the single source of truth**
   Implement `compute_losses_from_coords(X_pred, geo_targets)` exactly as in the design doc: unit-RMS gauge fix, fixed $\sigma_{\text{local}}$, KL on RW transitions, and multiscale stress with weight 0.1.

2. **Implement the target builder plumbing in the Stage C dataloader**
   Ensure each miniset yields `geo_targets` with `knn_edges`, `sigma_local`, `rw_steps`, `rw_transitions`, `multiscale_edges`, `ref_distances`, using adaptive $k$/steps/landmarks as specified.

3. **Build EGNN-Dist baseline first**
   - Build kNN($H$) graph constructor and patch sampler for scRNA inference.
   - Train on ST minisets; verify $L_{\text{rw}}$ and $L_{\text{stress}}$ decrease on validation.
   - Add distance head and verify distance NLL correlates with stress term.

4. **Flagship IPA-R**, 2–4 attention blocks.
   - Add deep supervision across recycles.
   - Add overlap training only after intrinsic losses are stable.

5. **Energy variant**
   - Start with discrete $K$-step unrolled dynamics ($K = 8$) without gradient-alignment.
   - Add monotonicity regularizer.
   - Add gradient-alignment at 2 checkpoints if stable.

6. **Inference pipeline**
   - Implement patch sampler (random-walk on $H$-graph).
   - Implement distance aggregation + global stress solve.
   - Validate end-to-end on held-out ST by dropping coordinates and comparing intrinsic metrics.

### 10 Most Important Hyperparameters (and What to Sweep First)

1. Patch size $n_p$ (128, 256, 384)
2. Patch overlap minimum (32, 64, 96)
3. Message-passing kNN in $H$ ($k$ = 16, 24, 32)
4. RW steps weights $\alpha_s$ schedule (fixed $0.5^{s-1}$ vs learned scalar)
5. $\lambda_{\text{dist}}$ (distance head weight)
6. $\lambda_{\text{cycle}}$ (coord–distance consistency)
7. Repulsion temperature $\tau$ and $\lambda_{\text{rep}}$
8. IPA-R recycles $R$ (2, 4, 6) and blocks per recycle
9. ODE steps $K$ and step size $\Delta t$ (for Family 3)
10. Overlap loss weight $\lambda_{\text{ov}}$ and gating (late-only vs always)

**Sweep order**

- First: kNN($H$), $\lambda_{\text{dist}}$, $\lambda_{\text{cycle}}$, patch size/overlap.
- Second: IPA-R recycles and repulsion.
- Third: energy-variant alignment and monotonicity.

### Debugging Checklist

1. **Target sanity**
   - Check `rw_transitions[s]` rows sum to 1; verify `sigma_local > 0`; verify `ref_distances > 0`.

2. **Gauge sanity**
   - After normalization, verify $\sum_i \tilde{x}_i \approx 0$ and RMS $\approx 1$.

3. **RW loss sanity**
   - On a toy grid of points, verify RW loss decreases when predicted distances match GT.
   - Confirm you are using **target** `sigma_local` (not recomputed from $X$).

4. **Stress loss sanity**
   - On GT coordinates (held internally), check stress near 0; on random coords, stress large.

5. **No-coordinate leakage**
   - Assert Stage C forward never receives raw ST coords; only $H$ (+ optionally $H$-graph).

6. **Train–test graph mismatch test**
   - Run inference using $H$-kNN edges (not spatial) and quantify degradation.

7. **Stitching stability**
   - Plot distribution of aggregated distances' IQR; large IQR indicates unreliable patches.

8. **scRNA scaling**
   - Profile patch inference throughput; verify it is bounded by patch size.

9. **Ablations**
   - Train baseline with only $L_{\text{rw}}$, only stress, and combined; verify combined is best.

10. **Failure visualization**
    - For ST held-out slides, compare diffusion-distance Spearman between predicted geometry and GT geometry (computed offline) as a primary diagnostic.
