# IPA-Lite Architecture Analysis & Next Directions
> Phase 25+ Post-Mortem: Operator-First Geometry Learning for Spatial Transcriptomics

---

## 1. Delta Review: `ipa_lite_mar_01_full.md` → `ipa_lite_mar_03.md`

`ipa_lite_mar_03.md` adds a new **Phase 25** block — *"Inference Quality Investigation, Gradient Competition, and Gate Overfitting"* — and updates downstream conclusions accordingly. Key new findings:

1. **Phase 25 motivation:** Even with oracle edges, qualitative reconstructions and Pearson can still be "mediocre," motivating a deeper investigation beyond $E_\text{geo}$.
2. **25a (Gradient competition):** Gradients of $L_\text{RW}$ and $L_\text{stress}$ are near-orthogonal (cosine $\approx 0.035$), so the inference gap is **not** a Pareto/competition issue.
3. **25b (Stress weight):** Increasing $\lambda_\text{stress}$ (e.g., 0.3) improves *both* $L_\text{stress}$ and $L_\text{RW}$ (regularization effect rather than tradeoff).
4. **25c (Data limitation diagnostic):** Train gap $\approx 0.002$ vs val gap $\approx 0.034$ ("18×") indicates a generalization/memorization regime.
5. **25d (Embedding distribution / "geometry style"):** After global alignment, **local slide clustering persists**: self-kNN purity ~**89.3%**, and ST3 remains a local outlier, consistent with slide-specific embedding geometry.
6. **25e (Task/feature signal is present on val):** Geometric targets and gate input features look similar across slides; feature→target "separation" on val slide matches train slides ($\approx 0.131$ vs $0.132$), implying the **discriminative signal exists everywhere**.
7. **25e conclusion (gate memorization):** The 18× gap is attributed to **gate memorization** (lookup-table behavior) rather than missing signal.
8. **25f (Gate feature ablation):** Raw cosine similarity alone gets AUC $\approx 0.80$ across slides; absolute features add little AUC but are implicated as a memorization source.
9. **25g–25h (Regularizing gate fine-tune):** Dropout-based joint fine-tuning and sweeps were explored; best dropout around 0.2 noted, but this doesn't resolve the core gap.
10. **Open-questions update:** Phase 25 explicitly elevates SC-adapter/slide-style issues and remaining generalization concerns.

---

## 2. Integrating Phase 25 with New Inference-Time Experiments

Phase 25 establishes:

- The remaining gap is **not** $L_\text{RW}$ vs $L_\text{stress}$ competition (near-orthogonal gradients).
- Increasing $\lambda_\text{stress}$ tends to help *both* objectives (regularization).
- Generalization is fragile (18× train→val gap).
- Embeddings exhibit slide-specific "geometry style" locally (89.3% self-kNN purity) even after global alignment.

The new inference-time experiments add the missing mechanistic piece:

- The *learned attention/edge weights* already contain a strong intrinsic geometry signal: diffusion map from the attention matrix gives Pearson $\approx 0.80$ vs $\approx 0.51$ for IPA ($R{=}5$) from the same checkpoint.
- Increasing recycles or $\eta$ improves rank metrics while causing **spectral oversmoothing / boundary-ring collapse**, which new collapse metrics detect (Pearson/Spearman/kNN@10 are "blind").

### Combined Conclusions

1. **Phase 25's observation that oracle edges can still look bad** is consistent with the finding that the *coordinate decoder dynamics themselves* can collapse (oracle collapses even harder past $R \approx 10$).
2. **Because gradients aren't competing**, tuning $\lambda_\text{stress}$ can only regularize within the expressivity limits of the decoder; it cannot change the fact that the current IPA update behaves like repeated low-pass filtering once attention stabilizes.
3. **Phase 25's "memorization regime"** (18× gap; local slide clustering) is compatible with the inference results: if the model can form a good operator on a held-out miniset, but the coordinate decoder systematically projects into a low-frequency embedding (ring/oversmooth), apparent gains in rank metrics will not translate into true geometric fidelity and will be unstable across slides.

> **Net directive:** Treat the **operator** (Markov/diffusion structure) as the primary learned object, and replace the IPA coordinate update with a **non-diffusive coordinate extraction** (spectral-with-calibration or stress-optimization), or go fully **coordinate-free**.

---

## 3. Bottleneck Diagnosis: Encoder vs Architecture vs Both

### 3.1 What the Evidence Forces

#### (A) Architecture Bottleneck — Real and Immediate

Inference-time experiments show:

- The attention operator contains strong geometry (diffusion map from attention gives high Pearson/Spearman), **but**
- The IPA coordinate update cannot reliably convert that operator to faithful 2D geometry and tends toward oversmoothing collapse with more recycles / larger $\eta$.

This is a **decoder expressivity/dynamics bottleneck**: the update is a (learned) Laplacian-like smoothing step once attention stabilizes. This also explains Phase 25's motivation — "oracle edges still look mediocre" is exactly what you see when the decoder collapses even with perfect support.

**Conclusion:** The IPA coordinate update is a **primary bottleneck** for coordinates, independent of gate quality.

#### (B) Encoder (Embedding "Style") — Secondary, Cross-Slide Bottleneck

Phase 25 documents that even after global alignment, local neighborhoods remain slide-clustered (self-kNN purity ~89.3%) and ST3 is locally outlying. The encoder design explicitly anticipates this: Spatial InfoNCE introduces slide-specific anisotropy and subspace divergence ("geometry style").

**Conclusion:** Even if operator-learning is viable, cross-slide generalization can still be limited by slide-style in $H$ and by how the inference graph is constructed (kNN in an anisotropic, slide-clustered space).

### 3.2 Is 89% self-kNN Expected or Pathological?

Given the encoder design, it is **not surprising** that local neighborhoods remain slide-biased — NCE can cause slide-specific covariance geometry ("subspace divergence"). Whether it is pathological depends on deployment:

- **Within-batch inference:** self-kNN across multiple slides is not directly relevant.
- **Cross-patient/slide/batch inference:** a kNN graph built in a slide-clustered embedding can bias the candidate edge distribution and encourage memorization/shortcut solutions (consistent with Phase 25's generalization findings).

### 3.3 Minimal, Frozen-Encoder–Compatible Intervention

1. **Verify per-gene mean centering** is applied exactly as intended (claimed to be the "complete fix" at the input level).
2. If local style remains: apply **post-hoc linear alignment** (CORAL/whitening) *outside* the encoder (no gradients through encoder).
3. In Stage C, allow a **tiny domain adapter** $A$ on $H$ (LayerNorm + learned diagonal scale/bias, or a single linear map initialized to identity) trained only with Stage C losses.

### 3.4 Final Diagnosis

| Bottleneck | Scope | Priority |
|---|---|---|
| **IPA coordinate update** (decoder dynamics / oversmoothing / collapse) | Coordinates | **Primary** |
| **Embedding "geometry style"** and slide-local clustering in $H$ | Cross-slide robustness | Secondary |

---

## 4. Architecture Directions

### Shared Notation

- Miniset size: $n \in [128, 384]$
- Frozen encoder outputs: $H \in \mathbb{R}^{n \times d}$ (typically $d = 128$)
- Candidate graph $E_c$: directed kNN in $H$ (or adapted $\tilde{H}$), with $|N_c(i)| = k$ (e.g., 80–120)
- Target supervision per miniset:
  - **F2:** multi-step random-walk transitions $\{P_\text{ref}^{(s)}\}_{s \in S}$ (row-stochastic)
  - **F4:** multiscale edge set $E_\text{ms}$ with reference intrinsic distances $d^\text{ref}_{ij}$ and weights $w_{ij}$

Core losses (fixed across all options):

$$L_\text{RW} = \frac{1}{|S|}\sum_{s \in S} \frac{1}{n} \sum_{i=1}^n \text{KL}\!\left(P_\text{ref}^{(s)}(i, \cdot) \;\big|\; P_\text{pred}^{(s)}(i, \cdot)\right)$$

$$L_\text{stress} = \frac{1}{|E_\text{ms}|} \sum_{(i,j) \in E_\text{ms}} w_{ij} \left(\log(d^\text{pred}_{ij} + \epsilon) - \log(d^\text{ref}_{ij} + \epsilon)\right)^2$$

---

### Option 1 — OperatorNet (Coordinate-Free / Coordinate-Light)

> **One-sentence idea:** Predict the sparse Markov operator $T_\text{pred}$ directly from $H$ on the inference graph, supervise with F2, predict distances on multiscale edges for F4, and only *derive* coordinates for visualization via diffusion map or stress embedding.

#### E1. Full Mathematical Specification

**E1.1 Input processing (frozen encoder + optional adapter)**

$$\tilde{H} = \text{LN}(H)W_A + b_A, \quad W_A \approx I$$

where $W_A \in \mathbb{R}^{d \times d}$ is constrained (e.g., diagonal or low-rank) for sample-efficiency.

**E1.2 Candidate graph construction**

Build $E_c$ from kNN in $\tilde{H}$: $N_c(i) = \text{kNN}_k(\tilde{H}_i)$.

**E1.3 Edge features**

For each candidate directed edge $(i,j) \in E_c$, define relative-only features to reduce memorization:

$$f_{ij} = \left[|\tilde{H}_i - \tilde{H}_j|,\; \tilde{H}_i \odot \tilde{H}_j,\; \cos(\tilde{H}_i, \tilde{H}_j)\right]$$

**E1.4 Graph transformer backbone → operator logits**

$$S, Z = \text{SparseGraphormer}(\tilde{H}, E_c, f)$$

$$u_{ij} = \text{MLP}_T\!\left([S_i, S_j, Z_{ij}, f_{ij}]\right)$$

$$T_\text{pred}(i,j) = \begin{cases} \dfrac{\exp(u_{ij}/\tau)}{\sum_{k \in N_c(i)} \exp(u_{ik}/\tau)} & j \in N_c(i) \\ 0 & \text{otherwise} \end{cases}$$

Multi-step predictions:

$$P_\text{pred}^{(s)} = T_\text{pred}^s$$

computed by sparse matrix multiplication on the candidate support.

**E1.5 Distance head for stress edges (coordinate-free F4)**

$$\ell_{ij} = \text{MLP}_d\!\left([S_i, S_j, Z_{ij}, f_{ij}]\right), \qquad d^\text{pred}_{ij} = \exp(\ell_{ij})$$

**E1.6 Complete loss**

Regularizers:

1. **Row-entropy calibration** (anti-collapse to one-hot or uniform):

$$L_\text{ent} = \frac{1}{n}\sum_i \left(H_i(T_\text{pred}) - H_i(T_\text{ref})\right)^2$$

where $H_i(T) = -\sum_j T(i,j)\log(T(i,j)+\epsilon)$.

2. **Reversibility / detailed balance regularizer** (stabilizes spectrum):

$$L_\text{rev} = \frac{1}{|E_c|}\sum_{(i,j) \in E_c} \left(\pi_i T_\text{pred}(i,j) - \pi_j T_\text{pred}(j,i)\right)^2$$

3. **Triangle / metric consistency**:

$$L_\triangle = \mathbb{E}_{(i,j,k)}\!\left[\text{ReLU}\!\left(d^\text{pred}_{ij} - d^\text{pred}_{ik} - d^\text{pred}_{kj}\right)^2\right]$$

4. **Operator–distance compatibility** (near in operator ⇒ near in distance):

$$L_\text{od} = \frac{1}{|E_c|}\sum_{(i,j) \in E_c} \left(\log(d^\text{pred}_{ij}+\epsilon) + \alpha\log(T_\text{pred}(i,j)+\epsilon)\right)^2$$

**Total loss:**

$$L_\text{total} = L_\text{RW} + \lambda_\text{stress} L_\text{stress} + \lambda_\text{ent} L_\text{ent} + \lambda_\text{rev} L_\text{rev} + \lambda_\triangle L_\triangle + \lambda_\text{od} L_\text{od}$$

*Schedule:* start with $\lambda_\text{rev}, \lambda_\triangle, \lambda_\text{od}$ small and ramp after $L_\text{RW}$ stabilizes.

**E1.7 Training algorithm**

```python
for step in range(num_steps):
    mi = sample_miniset()
    H = encoder(mi)                # frozen
    Ht = adapter(H)                # optional tiny adapter
    E_c = knn_graph(Ht, k)         # inference-style candidate edges

    S, Z = SparseGraphormer(Ht, E_c, edge_feats(Ht, E_c))
    u = T_head(S, Z, E_c)
    T = row_softmax(u, E_c, tau)

    # Multi-step transitions (sparse powers)
    P_pred = {s: sparse_matpow(T, s) for s in S_steps}

    # Distance head on E_ms (train-time available)
    d_pred = dist_head(S, Z, E_ms)

    # Losses
    L_rw    = KL_multistep(P_ref, P_pred)
    L_stress = log_stress(d_ref, d_pred, E_ms)
    L_ent   = entropy_match(T, T_ref)
    L_rev   = detailed_balance(T)
    L_tri   = triangle_penalty(d_pred)
    L_od    = operator_distance_compat(T, d_pred)

    loss = L_rw + lam_stress*L_stress + ...
    loss.backward()
    opt.step()
```

**Complexity:** $O(nkc)$ per layer for sparse attention; powers $T^s$ cost $O(|E_c|s)$ via sparse multiplies.

#### E2. Why It Avoids Documented Failure Modes

- **Avoids IPA oversmoothing by construction:** no iterative coordinate diffusion; the model directly represents geometry as a Markov operator.
- **Coordinates are optional and derived:** post-hoc embedding (diffusion maps / stress) is collapse-proof via the new metrics.
- **Handles 89% self-kNN:** relative-only edge features (absolute features drive memorization per Phase 25) and entropy/detailed-balance regularizers reduce capacity for slide-specific lookup tables.

#### E3. scRNA Inference Pipeline

1. Compute $H$ for all scRNA cells (encoder frozen), then $\tilde{H}$ via the tiny adapter.
2. Build approximate kNN graph $E_c$ (FAISS/Annoy; mutual-kNN optional).
3. Run OperatorNet **patchwise** (farthest point sampling landmarks, $r$-hop neighborhoods capped at 384 nodes; stitch by averaging logits $u_{ij}$ across patches, then global row-softmax).
4. Downstream: diffusion distances, pseudotime, etc. directly from $T_\text{pred}$; coordinates via diffusion map (Nyström on landmarks) + optional stress refinement.

#### E4. Official GitHub Anchors

| Component | Repo | File |
|---|---|---|
| Graphormer | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) | `graphormer/modules/graphormer_graph_encoder.py` |
| PyG sparse utilities | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `torch_geometric/nn/conv/gatv2_conv.py` |
| OpenFold (init/norm conventions) | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) | `openfold/model/primitives.py` |

#### E5. Implementation Blueprint

**Build order:**
1. $T_\text{pred}$ head + F2 ($L_\text{RW}$) only (no distances).
2. Multi-step powers + verify oracle-consistency.
3. Distance head + F4 on $E_\text{ms}$.
4. Regularizers one by one: $L_\text{ent}$ → $L_\text{rev}$ → $L_\triangle$ → $L_\text{od}$.
5. Optional adapter; evaluate slide-style effects.

**Key hyperparameters:** candidate $k$, transformer depth (2–6), hidden size $c$, temperature $\tau$ schedule, $\lambda_\text{stress}$, $\lambda_\text{ent}$, $\lambda_\text{rev}$, patch size / overlap policy.

**Go/no-go criteria:**
- On ST val: $L_\text{RW}$ within 10–15% of oracle $L_\text{RW}$ on reference operator support.
- $L_\text{stress}$ decreases without inducing operator degeneration.
- Collapse-aware composite score improves vs IPA ($R{=}5$) baseline.

#### E6. NeurIPS Paper Story

> *"Learn intrinsic spatial geometry by predicting a diffusion operator from expression embeddings under gauge ambiguity, supervised only by intrinsic diffusion and multiscale metric targets."*

Key novelty: operator-first learning + inference-graph constraint + collapse-aware evaluation + optional coordinate derivation.

---

### Option 2 — SpectralCoords: Operator-First + Differentiable Spectral Coordinate Decoder

> **One-sentence idea:** Predict $T_\text{pred}$ as in OperatorNet, then decode coordinates via a spectral embedding layer (learned diffusion coordinates), avoiding iterative smoothing recycles.

#### E1. Full Mathematical Specification

**E1.1 Operator prediction** — same as Option 1.

**E1.2 Make operator reversible (for stable eigenvectors)**

Compute stationary distribution $\pi$ (power iteration). Form symmetric diffusion operator:

$$S = \Pi^{1/2} T_\text{pred} \Pi^{-1/2}$$

**E1.3 Differentiable spectral embedding**

Compute top $m$ nontrivial eigenvectors of $S$:

$$S v_k = \lambda_k v_k, \quad k = 1, \dots, m$$

Drop the trivial constant eigenvector. Define spectral features per node:

$$\phi_i = [v_2(i), v_3(i), \dots, v_{m+1}(i)] \in \mathbb{R}^m$$

Decode 2D coordinates:

$$x_i = \text{MLP}_\text{spec}(\phi_i) \in \mathbb{R}^2, \qquad X = \text{gauge\_fix}(x)$$

$$d^\text{pred}_{ij} = |x_i - x_j|$$

**E1.4 Loss (F2 + F4 + spectral anti-collapse)**

1. **Eigenvalue spread / effective-rank regularizer:**

Let $\tilde{\lambda}_k = 1 - \lambda_k$ for $k \ge 2$ and $p_k = \tilde{\lambda}_k / \sum_{j \ge 2} \tilde{\lambda}_j$:

$$L_\text{spec} = -\sum_{k=2}^{m+1} p_k \log(p_k + \epsilon)$$

2. **Coordinate effective-dimension regularizer:**

Let $C = \frac{1}{n}\sum_i (x_i - \bar{x})(x_i - \bar{x})^\top$ with eigenvalues $\sigma_1 \ge \sigma_2$:

$$d_\text{eff}(X) = \frac{(\sigma_1 + \sigma_2)^2}{\sigma_1^2 + \sigma_2^2}$$

$$L_\text{dim} = \text{ReLU}(d_\text{min} - d_\text{eff}(X))^2$$

3. **Row-entropy calibration** as in Option 1.

**Total loss:**

$$L_\text{total} = L_\text{RW} + \lambda_\text{stress} L_\text{stress} + \lambda_\text{spec}(-L_\text{spec}) + \lambda_\text{dim} L_\text{dim} + \lambda_\text{ent} L_\text{ent}$$

**E1.5 Training algorithm**

```python
H = encoder(mi)             # frozen
Ht = adapter(H)             # optional
E_c = knn_graph(Ht, k)

S_node, Z_edge = SparseGraphormer(Ht, E_c, edge_feats)
u = T_head(S_node, Z_edge, E_c)
T = row_softmax(u, E_c, tau)

# Spectral decode (m small, n<=384 => torch.linalg.eigh tractable)
pi = stationary(T)                      # few power iterations
S = symmetrize(pi, T)                   # Π^{1/2} T Π^{-1/2}
eigvals, eigvecs = top_eigs(S, m+1)
phi = eigvecs[:, 1:m+1]                 # drop trivial
X = gauge_fix(MLP_spec(phi))

# Losses
L_rw    = KL_multistep(P_ref, matpow(T))
L_stress = log_stress(d_ref, pair_dists(X), E_ms)
L_spec  = spectral_entropy(eigvals)
L_dim   = effdim_penalty(X)
L_ent   = entropy_match(T, T_ref)

loss = L_rw + lam_stress*L_stress + ...
backprop(loss)
```

#### E2. Why It Avoids Failure Modes

- Replaces "iterate a smoothing update" with a **global spectral solve** — directly aligned with the empirical discovery that diffusion-map coordinates from the attention matrix outperform the IPA update.
- Ring collapse addressed explicitly via: $m > 2$ eigenvectors + learned decoder, stress (F4) on decoded coordinates, spectral/eff-dimension regularizers.

#### E3. scRNA Inference Pipeline

1. Compute $T_\text{pred}$ on global kNN($\tilde{H}$).
2. Compute diffusion coordinates via **landmark Nyström** diffusion maps (pick $L \ll n$ landmarks, compute eigensystem on $L \times L$ submatrix, extend to all nodes).
3. Optionally refine coordinates by a few steps of stress minimization on predicted distances.

#### E4. Official GitHub Anchors

| Component | Repo | File |
|---|---|---|
| Graphormer | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) | `graphormer/modules/graphormer_graph_encoder.py` |
| Spectral embedding reference | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | `sklearn/manifold/_spectral_embedding.py` |
| OpenFold norms/init | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) | `openfold/model/primitives.py` |

#### E5. Implementation Blueprint

1. Start from Option 1 operator prediction.
2. Add spectral decode with $m = 8$ first; validate eigenvector sign/ordering consistency.
3. Add $L_\text{dim}$ hinge and $L_\text{spec}$ entropy.
4. Evaluate with collapse metrics.

**Key sweeps:** $m$, $\lambda_\text{spec}$, $\lambda_\text{dim}$, $\lambda_\text{stress}$, temperature $\tau$.

**Go/no-go:** spectral decode must improve **collapse-aware composite** over IPA ($R{=}5$) without inflating $L_\text{RW}$.

#### E6. NeurIPS Story

> *"Predict diffusion operators and extract gauge-free coordinates via differentiable spectral decoding; avoids iterative oversmoothing failure in coordinate generators."*

---

### Option 3 — DistNet + Unrolled Stress Embedding

> **One-sentence idea:** Predict intrinsic distances on multiscale edges, then obtain coordinates by unrolling a stress-minimization solver (SMACOF-like), avoiding Laplacian smoothing dynamics.

#### E1. Full Mathematical Specification

**E1.1 Distance prediction (primary) + operator (secondary)**

Backbone: sparse graph transformer on $E_c$ producing $S, Z$.

$$\ell_{ij} = \text{MLP}_d([S_i, S_j, Z_{ij}, f_{ij}]), \qquad \hat{d}_{ij} = \exp(\ell_{ij})$$

**E1.2 Coordinate decoder: unrolled stress minimization (not diffusion)**

Define the predicted-distance stress energy:

$$E_\text{pred}(X) = \sum_{(i,j) \in E_d} w_{ij}\left(|x_i - x_j| - \hat{d}_{ij}\right)^2$$

Run $R$ unrolled gradient steps:

$$X^{r+1} = \text{gauge\_fix}\!\left(X^r - \eta_r \nabla_X E_\text{pred}(X^r)\right)$$

Initialize $X^0$ from a small MLP on $H$ or from 2D PCA of $H$.

**E1.3 Loss**

1. **Direct distance supervision** (stabilizes unrolling):

$$L_{\text{stress},d} = \frac{1}{|E_\text{ms}|}\sum_{(i,j) \in E_\text{ms}} w_{ij}\left(\log(\hat{d}_{ij}+\epsilon) - \log(d^\text{ref}_{ij}+\epsilon)\right)^2$$

2. **Triangle penalty on $\hat{d}$:** $L_\triangle$ (as Option 1).

3. **Step-size stabilization:**

$$L_\text{step} = \frac{1}{R}\sum_{r=0}^{R-1} \frac{1}{n}\sum_i |\Delta x_i^{(r)}|^2$$

**Total loss:**

$$L_\text{total} = L_\text{RW} + \lambda_\text{stress} L_\text{stress}(X^R) + \lambda_{\text{stress},d} L_{\text{stress},d} + \lambda_\triangle L_\triangle + \lambda_\text{step} L_\text{step}$$

**E1.4 Training algorithm**

```python
Ht = adapter(encoder(mi))
E_c = knn_graph(Ht, k)
S, Z = SparseGraphormer(Ht, E_c, edge_feats)

# Predict distances on E_d (contains E_ms during training)
d_hat = exp(dist_head(S, Z, E_d))

# Optional operator head
T = row_softmax(T_head(S, Z, E_c))

# Unrolled stress embedding
X = init_X(Ht)  # MLP or PCA
for r in range(R):
    grad = grad_stress(X, d_hat, E_d, w)
    X = gauge_fix(X - eta[r] * grad)

# Losses
L_rw      = KL_multistep(P_ref, matpow(T))
L_stress  = log_stress(d_ref, pair_dists(X), E_ms)
L_stress_d = log_stress(d_ref, d_hat, E_ms)
L_tri     = triangle_penalty(d_hat)
L_step    = step_penalty(X_updates)

loss = L_rw + lam_stress*L_stress + ...
backprop(loss)
```

#### E2. Why It Avoids Oversmoothing/Ring Collapse

The coordinate update is **not** "move toward neighbor centroid"; it is the gradient of a rest-length objective, which has both attractive and repulsive components depending on whether current distances are too big or too small. Collapse becomes detectable and preventable via $L_\text{step}$ and triangle constraints.

#### E3. scRNA Inference Pipeline

1. Build $E_c$ from $\tilde{H}$.
2. Construct inference distance edge set $E_d^\text{inf}$: kNN edges + longer edges via random-walk sampling on $T$ (or k-bank).
3. Predict $\hat{d}$ on $E_d^\text{inf}$.
4. Embed via unrolled stress solver patchwise; stitch using patch alignment on distance constraints.

#### E4. Official GitHub Anchors

| Component | Repo | File |
|---|---|---|
| SMACOF/MDS reference | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | `sklearn/manifold/_mds.py` |
| Graph transformer | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) | `graphormer/modules/graphormer_graph_encoder.py` |
| OpenFold (gauge-fix conventions) | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) | `openfold/model/structure_module.py` |

#### E5. Implementation Blueprint

1. Implement distance head $\hat{d}$ + direct $L_{\text{stress},d}$ first (no unrolling).
2. Add unrolled solver with small $R$ (e.g., 5–10) and check gradients.
3. Add operator head for F2 (or reuse existing RW pipeline from $X^R$).
4. Track collapse metrics every epoch.

**Go/no-go:** improves collapse-aware composite score and qualitative structure relative to IPA ($R{=}5$) without regressing $L_\text{RW}$.

#### E6. NeurIPS Story

> *"Learn intrinsic metric and diffusion structure from expression; recover coordinates by differentiable stress embedding rather than diffusion-like coordinate updates, preventing spectral oversmoothing collapse."*

---

## 5. Collapse-Aware Evaluation Framework

### 5.1 Proposed Metrics Assessment

You proposed six metrics:

| # | Metric | Assessment |
|---|---|---|
| 1 | $d_\text{eff}$ — participation ratio of singular values | Keep — smoother collapse alarm |
| 2 | $s_2 / s_1$ — singular value ratio | Keep — quick alarm (partially redundant with $d_\text{eff}$) |
| 3 | boundary_mass | Keep — catches convex-hull migration failure |
| 4 | grid_entropy | Keep — coarse global spread (more stable with gauge-fix) |
| 5 | density_corr (log kNN radius correlation) | Keep |
| 6 | W1_knn (Wasserstein-1 between log-radius distributions) | Keep — complementary to density_corr |

**Two missing metrics (add both):**

**7. Area / alpha-shape area ratio** — strong collapse detector for boundary/ring artifacts:

$$m_\text{area} = \frac{\text{Area}(\text{Hull}(X_\text{pred}))}{\text{Area}(\text{Hull}(X_\text{gt}))}$$

Report log-ratio; large deviations indicate collapse or overexpansion.

**8. Topological ring detector (persistent H1 mass)** — ring collapse is fundamentally a topological artifact. Compute a lightweight persistence summary (maximum H1 persistence / total H1 persistence) on the point cloud via Vietoris–Rips on a subsample. This directly detects "everything moved to a loop."

*Cheaper proxy:* **radial variance** (variance of $|x_i|$ after gauge-fix) + **angle uniformity**. Rings show low radial variance and near-uniform angles.

### 5.2 A Composite Score That Cannot Be "Won" By Collapsing

For each metric $m$, estimate $\mu_m, \sigma_m$ on GT coords over the val set. For a prediction, compute $z_m = (m - \mu_m) / \sigma_m$.

Define bad deviation functions:
- Higher-is-better (effdim, sv_ratio, entropy, density_corr): $p_m = \text{ReLU}(-z_m)$
- Lower-is-better (boundary_mass, W1_knn): $p_m = \text{ReLU}(z_m)$

$$\boxed{E_\text{safe} = E_\text{geo} + \lambda_c \left( p_{d_\text{eff}} + p_{s_2/s_1} + p_\text{entropy} + p_\text{density} + p_\text{bnd} + p_\text{W1} + p_\text{area} \right)}$$

Choose $\lambda_c$ so that a **2σ collapse** adds a penalty comparable to (or larger than) typical improvements in $E_\text{geo}$.

### 5.3 How to Report in the Paper

- **Main table:** $E_\text{geo}$ and $E_\text{safe}$ (collapse-safe composite), plus 2–3 representative collapse metrics (e.g., $d_\text{eff}$, boundary_mass, density_corr).
- **Supplement:** full metric panel and scatter plots showing Pearson/Spearman vs collapse penalty to demonstrate "rank metrics are blind."

---

## 6. Prioritized Implementation Plan

| Step | Action | Dependencies |
|---|---|---|
| **1** | Lock in collapse-aware evaluation as a gating criterion. Log $E_\text{geo}$, $E_\text{safe}$, and full collapse panel (including area) on every run. | None |
| **2** | Implement **Option 1 (OperatorNet)** first. | Existing gate/operator KL code. Validate: show $P_\text{ref}^{(s)}$ matching without coordinates; check operator entropy and spectral gap. |
| **3** | Add "coordinate derivation" as a separate, non-train-critical step. Start with diffusion maps from $T_\text{pred}$ for visualization only. Do **not** select models by Pearson. | Option 1 stable. |
| **4** | If coordinates required as primary output: implement **Option 3 (stress unrolling)**. | Option 1 operator head stable. |
| **5** | **Option 2 (SpectralCoords)** as the "fancy paper" add-on once OperatorNet is stable cross-slide. | Options 1+3 validated. |
| **6** | Address encoder "geometry style" only if it blocks operator generalization: (1) verify per-gene mean centering, (2) post-hoc CORAL/whitening, (3) tiny adapter $A$ if needed. | Post-hoc; never fine-tune the encoder. |

---

## 7. Comparison Table

| Direction | Modeling object | Novelty | Impl. complexity | Expected collapse behavior | Inference clarity (scRNA) | F2 exploitation | F4 exploitation | Risk |
|---|---|---|---|---|---|---|---|---|
| **Option 1: OperatorNet** | $T_\text{pred}$ + $d^\text{pred}$ (sparse) | High | Medium | Low — no coordinate diffusion during training; collapse only in optional visualization | Very clear: build kNN($H$) → predict $T$ | Directly (KL on $T^s$) | Directly (stress on predicted distances) | Low–Med |
| **Option 2: SpectralCoords** | $T_\text{pred}$ → spectral coords $X$ | High | Med–High | Moderate — mitigated via $m>2$ + regularizers | Clear but needs landmark/Nyström for large $n$ | Directly | Via decoded $X$ | Med |
| **Option 3: DistNet + unrolled stress** | $\hat{d}_{ij}$ + unrolled $X$ | High | High | Lower risk than IPA smoothing; collapse controlled by solver/penalties | Clear patchwise; embedding step adds compute | Either direct head or via $X$ | Directly (stress is the decoder objective) | Med–High |

---

> **Decisive recommendation:** The combined evidence strongly favors **operator-first** (Option 1 as the backbone), with **Option 3** as the coordinate head if high-fidelity 2D coordinates are required beyond visualization.
