## 0) Current state that the plan must address (no re-arguing)

* The **validated baseline** is **D7 / V1 shared 5R + QK-norm + clamped geo**, with **$E_{\text{geo}}(\text{H-gate})=0.440\pm0.014$** and **$E_{\text{geo}}(\text{T-edge oracle})=0.1619\pm0.0014$**, so the **post–oracle-fix inference gap is $0.278$**.
* The **oracle evaluation contamination bug** was real: "oracle" had been running on `edge_mask ∩ candidate_mask` due to passing candidate-derived gate logits into oracle eval; fix is **gate_logits = 0** for T-edge eval.
* The **remaining gap is gating**, not missing edges: candidate graph is healthy (recall ≈ 0.96) and the E-series concludes **bad gating accounts for ~99.6% of the gap**.
* "Full integrated IPA‑R(2D)" (D2) is worse (≈0.61) despite "nice attention metrics," and the log shows why.

Everything below is aimed at: (i) removing the remaining evaluation/implementation foot‑guns, and (ii) **shipping one "fancy IPA‑R" that is actually scale-ready by fixing gating without exploding knobs**.

---

# Deliverable D — Code-level audit checklist (remaining plausible failure modes)

These are prioritized by: "can still plausibly exist after D3" + "would produce exactly your observed pathologies (gate drift, precision failure, weird stability)." Each item has **symptom → minimal test → expected outcome → what bug it implies**.

### D0. Never again: oracle contamination via mismatched masks

**Symptom:** T-edge oracle score changes if you change candidate k, candidate mask, or gate module.

**Minimal test (unit):**

1. Pick one val miniset $m_i$.
2. Compute `gl_cand = gate(H, cand_mask)`.
3. Run model with `edge_mask` three ways:
   * A: `gate_logits = 0`
   * B: `gate_logits = gl_cand`  *(this should be forbidden)*
   * C: `gate_logits = gate(H, edge_mask)` *(if you ever allow it)*
4. Assert A == C (within tolerance), and B != A (should get worse / differ).

**Expected:** "Gate=0" gives the true oracle. Using candidate logits must not be allowed for oracle eval (this was D3).

**Bug implied:** Any path that reuses gate logits from a different support is reintroducing the D3 confound.

**Hard guardrail to add (recommended):**

```python
if edge_mask is not cand_mask:
    assert gate_logits is None or torch.allclose(gate_logits, torch.zeros_like(gate_logits))
```

---

### D1. Masking/softmax order: "mask after adding all logits, before softmax"

**Symptom:** attention mass leaks onto non-edges or becomes NaN under fp16; gate seems ineffective.

**Minimal test:** Construct a tiny $n=5$ example where:
* qk logits are random,
* gate logits are large for one forbidden edge (mask false),
* check whether that forbidden edge gets nonzero attn.

**Expected:** masked edges must be exactly zero attention.

**Bug implied:** applying mask after softmax, or masking only one term (e.g., qk but not gate/geo), or using a finite "-30" instead of $-\infty$ in a way that breaks at fp16.

---

### D2. Diagonal handling in the gate (self-edge must be neutral, not trained as positive)

**Symptom:** gate KL looks "good" but gate precision on true spatial neighbors stays low; or gate collapses to self-loops.

**Minimal test:** For a batch of minisets:
* compute $G = \text{softmax}(\text{gate\_logits})$ and measure $\text{mean}_i\, G[i,i]$.

**Expected:** $G[i,i] \sim 0$ (or exactly excluded) unless you *explicitly* want self-loops in the gate objective.

**Bug implied:** candidate masks include diagonal and gate loss is inadvertently training diagonal as a "free" sink. Your notebook shows you explicitly patched to exclude diagonal and set diag=0; keep that invariant everywhere.

---

### D3. Candidate-mask determinism (top‑k tie-break + dtype drift)

**Symptom:** seed variance or "same seed" variance across GPUs; attention metrics shift without code changes.

**Minimal test:** Build candidate masks twice for the same $H$ with deterministic flags on; assert equality.

**Expected:** exact match.

**Bug implied:** `topk` instability due to ties; dtype differences (fp16 vs fp32) in cosine; non-deterministic ops.

**Fix:** build kNN in fp32; optionally add tiny deterministic jitter $1\mathrm{e}{-6} \cdot \text{arange}(n)$ to break ties.

---

### D4. Symmetrization and degree: candidate graph must match what you think you evaluate

**Symptom:** measured candidate recall/precision doesn't match "effective edges" used in attention.

**Minimal test:** After building `cand`, ensure:
* `cand == cand.T`,
* `cand.diagonal().all()` (if you rely on diag in attention),
* node degrees are in expected range ($\approx 2k$ if symm union).

**Expected:** stable degree stats.

**Bug implied:** you evaluate recall on symm graph but attention uses directed, or vice versa.

---

### D5. Gate logits computed on one mask but loss computed on another

**Symptom:** gate KL improves but precision@k doesn't; or drift patterns look weird.

**Minimal test:** assert `gate_loss_kl` uses the *same* `candidate_mask` that was used to place logits.

**Expected:** same support.

**Bug implied:** mixing `cand_mask` and `attn_mask` (edge-drop) or using a different mask in loss vs forward.

---

### D6. Train/eval mismatch: gate uses clean H, model uses augmented H

Your loop currently computes `gb = gate(H_clean, cand)` but runs structure on `H_aug`.

**Symptom:** co-adaptation trap gets worse; freezing gate hurts more than expected; instability.

**Minimal test:** run two short trainings:
* gate computed on `H_clean`,
* gate computed on `H_aug` (same augmentation as structure),

keeping everything else fixed.

**Expected:** one dominates clearly; if `H_aug` helps, the old pipeline was mismatched.

**Bug implied:** not a bug, but a likely optimization pathology source.

---

### D7. Gate gradient leakage into structure loss (the "drift injector")

**Symptom:** pretrained gate is better than jointly trained gate, yet freezing makes model worse (your E8c/E8d pattern).

**Minimal test:** in joint training, compare:
* `gb = gate(...); out = model(..., gate_bias=gb)`  (baseline)
* `gb = gate(...).detach(); out = model(..., gate_bias=gb)`

**Expected:** detaching should prevent gate loss degradation (drift) if drift is driven by geometry gradients.

**Bug implied:** gate is being updated to satisfy the structure module's current geometry error rather than $T_{\text{ref}}$ supervision.

(This is the single highest-ROI audit item because it is both a potential "real" bug and also a direct fix.)

---

### D8. Softmax temperature / scaling: ensure logit scales are bounded at R0

**Symptom:** D2-style `logit_std ≈ 50` and attention entropy near 0 at R0, poisoning later recycles.

**Minimal test:** log per-recycle:
* `logit_std(h)`, `entropy(h)` for qk term alone and full logits.

**Expected:** qk-only logit_std in single digits with QK-norm; full logits also bounded.

**Bug implied:** missing QK-norm on some path, or $\tau$/$\beta$ not actually frozen/scheduled as intended.

---

### D9. Broadcasting bugs in head dims (gate bias applied per-head incorrectly)

**Symptom:** head-specific plots are identical or nonsensical; changing n_heads changes results drastically.

**Minimal test:** create synthetic gate bias that differs per head and check it affects logits per head.

**Expected:** additive per-head bias changes attention per head.

**Bug implied:** gate bias accidentally broadcast across wrong dim or summed incorrectly.

---

### D10. Edge-drop symmetry and diagonal preservation

In your training, you do `drop = drop & drop.t()` but diagonal preservation depends on candidate having diagonal.

**Symptom:** occasional NaNs or collapsed coordinates due to disconnected rows.

**Minimal test:** after edge-drop, verify every row has $\geq 1$ True edge (at least self).

**Expected:** always connected.

**Bug implied:** diagonal absent and edge-drop can zero out a row.

---

### D11. `build_predicted_rw_operator` exactness and diagonal conventions

**Symptom:** "oracle-consistent" check passes for some minisets but not all; or F2 doesn't decrease even on oracle graph.

**Minimal test:** feed target coords, compute $T_{\text{pred}}$, compare to cached $T_{\text{ref}}$ exactly.

**Expected:** match within tolerance. Your summary says this is verified; still keep as a regression test.

**Bug implied:** mismatch in $\sigma$, knn indices, symmetrization, or diagonal handling.

---

### D12. Gauge-fix gradient path (accidental detach)

**Symptom:** coordinates barely move, $\Delta X$ tiny across all recycles, training plateaus.

**Minimal test:** compute scalar loss from $X_{\text{gf}}$ and call `.backward()`; check `X.grad` nonzero.

**Expected:** nonzero.

**Bug implied:** detach inside gauge_fix or use of `.data`.

---

### D13. Stress edge family mismatch (F4 gradients not on the same edges you visualize)

**Symptom:** attention "looks good" on local kNN edges but $E_{\text{geo}}$ doesn't improve; confusion about interpretability.

**Minimal test:** confirm F4 uses the declared sparse multiscale edge set, not candidate edges.

**Expected:** F4 unaffected by candidate k changes (except via model output).

**Bug implied:** accidentally reusing candidate adjacency for stress evaluation.

---

### D14. Gate KL implementation direction and masking

Your `gate_loss_kl` is $\text{KL}(p \| q)$ with $p$ = masked/renormalized $T_{\text{ref}}$ and $q = \text{softmax}(g)$.

**Symptom:** gate KL decreases but precision/recall stays flat.

**Minimal test:** sanity check on a fabricated example where $q$ matches $p$ exactly → KL ≈ 0; swap direction should differ.

**Expected:** KL ≈ 0 only when distributions match.

**Bug implied:** wrong KL direction, missing renorm, or wrong masking.

---

# Deliverable A — Mechanistic diagnosis: why "fancier IPA‑R" lost to simpler baseline

You need 1–2 concrete hypotheses consistent with D2/D4/D7 and the notebook.

## Hypothesis A1 (primary): R0 term-competition creates an irreversible "poisoned refinement trajectory"

**Claim:** In D2, multiple logit sources (qk + gate + pair + geo + learned $\tau$/$\beta$ schedules) created a **massive logit scale at R0** (`logit_std ≈ 50.8`, `entropy ≈ 0.29`), yielding near-delta attention early. That causes a bad initial coordinate move / representation update that later recycles cannot recover from, even though later attention diagnostics look "structured."

**Evidence (directly from your logs):**
* D2 shows R0 `logit_std=50.8`, `entropy=0.29`.
* D4 "remove the fights" (pair_bias=0, freeze $\tau=1$, $\beta=2$, clamp R0 qk) is explicitly marked **confirmed** and improved performance, implying term competition was causal.

**One decisive falsification experiment (cheap):**
* Take the D2 model **unchanged**, but force at R0:
  * qk_norm on,
  * $\tau$ fixed=1,
  * $\beta$ fixed=2,
  * geo off,
  * pair off,
  * clamp logits to $[-10,10]$.
* If $E_{\text{geo}}$ remains ~0.61, then term competition was *not* the cause.

You already did a close version (D4); the "confirmed" label implies you saw meaningful improvement, so this hypothesis is not only plausible—it's already supported.

**Fix if confirmed (implementation change):**
* Make **logit staging an architectural constraint**, not a training-time patch:
  * start all non-qk terms at 0 contribution (zero-init last layer),
  * clamp geo and any bias terms,
  * fix $\tau$ to a safe range,
  * only activate geo after coordinates stabilize ($r \geq 2$), consistent with your own ablation guidance.

---

## Hypothesis A2 (secondary): D2 wastes the most valuable recycle because step-size gating suppresses R0 motion

**Claim:** D2's step-size gate makes R0 almost a no-op ($\eta_0 \approx 0.23$, $|\Delta X| = 0.026$) compared to V1's strong initial move (~0.21). That means the "qk exploration" recycle in D2 does not actually create a usable coarse layout, so later gate/geo refinement operates on junk.

**Evidence:**
* D2: $|\Delta X|$ at R0 = 0.026 with $\eta_0 = 0.23$.
* V1: R0 step magnitudes $\approx 0.21$.

**One decisive falsification experiment:**
* In D2, **force $\eta_0 = 1$** (or initialize $\eta_0$ to 0.6–0.8 and freeze it for first epoch).
  Success criterion: $|\Delta X|$ at R0 should jump into ~0.1–0.25 range and $E_{\text{geo}}$ should improve significantly (>0.03).
  If $E_{\text{geo}}$ does not improve, then wasted R0 is not the limiting factor.

**Fix if confirmed:**
* Hard-code $\eta$ schedule:
  * $\eta_0$ fixed high (0.6–1.0),
  * later $\eta$ trainable or decayed,
  * or remove $\eta$ gate entirely and rely on gradient clipping + gauge-fix.

---

# Deliverable B — Pick ONE "fancy" scale path (commit)

## Decision: Ship **Path 3 (Hybrid)**

**Minimal IPA backbone (D7-quality) + genuinely stronger contextual topology/gating module.**

Rationale: your own E-series conclusion is that **the candidate graph is not the problem; gating is**. The fancy work should go where the bottleneck is. "Full IPA‑R integration" increased knobs and hurt toy performance without addressing gating.

This hybrid is still "fancy/NeurIPS-grade" because:
* it retains a structure module with **SE(2)-equivariant coordinate refinement** (IPA-Lite style),
* and introduces a **contextual learned topology** module that is more expressive than an MLP on $(H_i, H_j)$ and is the critical missing piece.

---

## B1. Module definitions (implementation-grade)

### Inputs (fixed by your setup)

* Node embeddings: $H \in \mathbb{R}^{n \times d}$ (encoder frozen).
* Candidate adjacency: $M \in \{0,1\}^{n \times n}$ from cosine kNN($H$), symmetrized, $k=100$ (your latest best).
* Targets: $\{T^{(s)}_{\text{ref}}\}_s$ and multiscale stress edges for F4 (oracle-consistent).

---

## B2. Contextual GateFormer (CGF): topology/gating module

Goal: produce **edge logits** $g_{ij}$ on the candidate support that can separate true spatial neighbors from confounders.

### CGF design (two-layer, sparse, contextual)

Let $N(i) = \{j : M_{ij} = 1\}$.

**Node context encoder (graph attention with edge features):**

* Initialize node context:

$$c_i^{(0)} = \text{LN}(W_c H_i)$$

* Initialize edge features for candidate edges:

$$e_{ij}^{(0)} = \text{LN}\bigl(W_e [H_i,\; H_j,\; |H_i - H_j|,\; H_i \odot H_j,\; \cos(H_i, H_j)]\bigr)$$

(all computed only for $(i,j) \in M$; store as edge-list).

For layer $\ell = 0, 1$, multi-head attention:

$$\alpha_{ij}^{(\ell,h)} = \text{softmax}_{j \in N(i)}\left(\frac{\langle W_q^{(h)} c_i^{(\ell)},\, W_k^{(h)} c_j^{(\ell)} \rangle}{\sqrt{d_h}} + (w_b^{(h)})^\top e_{ij}^{(\ell)}\right)$$

$$m_i^{(\ell)} = \text{concat}_h \sum_{j \in N(i)} \alpha_{ij}^{(\ell,h)}\, W_v^{(h)} c_j^{(\ell)}$$

$$c_i^{(\ell+1)} = \text{LN}\left(c_i^{(\ell)} + \text{Drop}(m_i^{(\ell)})\right);\quad c_i^{(\ell+1)} \leftarrow c_i^{(\ell+1)} + \text{MLP}_c(c_i^{(\ell+1)})$$

**Edge update (optional but cheap and helpful):**

$$e_{ij}^{(\ell+1)} = \text{LN}\left(e_{ij}^{(\ell)} + \text{MLP}_e\bigl([e_{ij}^{(\ell)},\, c_i^{(\ell+1)},\, c_j^{(\ell+1)},\, c_i^{(\ell+1)} - c_j^{(\ell+1)}]\bigr)\right)$$

**Final gate logits:**

$$g_{ij} = \text{clip}\left(w_g^\top \text{MLP}_g\bigl([e_{ij}^{(2)},\, c_i^{(2)},\, c_j^{(2)}]\bigr),\; [-g_{\max}, g_{\max}]\right)$$

* For non-candidate edges: $g_{ij} = -\infty$.
* For diagonal: force $g_{ii} = 0$ (neutral) and exclude from gate loss.

This is the minimal "contextual gate" that you did **not** yet test (your own open questions list calls out deeper/contextual gate as missing).

**Capacity control:**
* $d_c = 64$, heads=4, layers=2, dropout=0.1, weight decay 5e-4.
* Total params is small (order 100–300k), appropriate for low-data minisets.

---

## B3. IPA backbone: stable D7-style shared block (5 recycles)

This is essentially D7, with one critical modification: **gate integration and geo scheduling are "safe-by-construction"**.

### State + coordinate init

$$s_i^{(0)} = \text{LN}(\text{MLP}_s(H_i)), \qquad x_i^{(0)} = \text{gauge\_fix}(\text{MLP}_x(H_i))$$

### Per-recycle attention logits (exact formula)

For recycle $r = 0, \dots, R-1$, heads $h = 1, \dots, H$:

Compute q/k/v projections from $s^{(r)}$ (standard transformer style), with **QK-norm**:

$$q_i^{(r,h)} = W_q^{(h)} \text{LN}(s_i^{(r)}), \quad k_i^{(r,h)} = W_k^{(h)} \text{LN}(s_i^{(r)}), \quad v_i^{(r,h)} = W_v^{(h)} \text{LN}(s_i^{(r)})$$

$$\hat{q} = \text{LN}(q), \quad \hat{k} = \text{LN}(k)$$

**Gate bias term:**

$$b^{\text{gate}}_{ij} = \beta_r \cdot g_{ij}$$

**Geo bias term (RBF, only after the layout has moved):**

Let $d_{ij}^{(r)} = \|x_i^{(r)} - x_j^{(r)}\|$, normalized by median candidate distance:

$$\bar{d}_{ij}^{(r)} = \frac{d_{ij}^{(r)}}{\text{median}_{(u,v) \in M}\, d_{uv}^{(r)} + \epsilon}$$

RBF features: $\phi_m(\bar{d}) = \exp\bigl(-(\bar{d} - \mu_m)^2 / (2\sigma^2)\bigr)$, $m = 1, \dots, M$.

Per head:

$$b^{\text{geo}}_{ij,h} = \begin{cases} \text{clip}\left((w_h)^\top \phi(\bar{d}_{ij}^{(r)}),\; [-b_{\max}, b_{\max}]\right), & r \geq r_{\text{geo}} \\ 0, & r < r_{\text{geo}} \end{cases}$$

with $r_{\text{geo}} = 2$. This matches your own conclusion that geo should not act at R0.

**Final attention logits:**

$$\ell^{(r,h)}_{ij} = \tau \cdot \langle \hat{q}_i^{(r,h)},\, \hat{k}_j^{(r,h)} \rangle + b^{\text{gate}}_{ij} + b^{\text{geo}}_{ij,h}$$

Mask:

$$\ell^{(r,h)}_{ij} \leftarrow -\infty \quad \text{if } M_{ij} = 0$$

Attention:

$$a^{(r,h)}_{ij} = \text{softmax}_j\bigl(\ell^{(r,h)}_{ij}\bigr)$$

**Stability conventions:**
* Fix $\tau = 1$ (or learn but clamp to $[0.5, 4]$).
* $\beta_r$ scheduled (see training).

This is consistent with your D7 baseline and your S6 geo redesign (RBF).

### Coordinate update (D7-style, minimal and stable)

Use per-head linear map $A_h \in \mathbb{R}^{2 \times 2}$ (implemented as `linear_x` producing $2H$ dims), **zero-initialized**.

For each edge:

$$\Delta x_i^{(r)} = \eta_r \sum_{h=1}^{H} \sum_{j} a^{(r,h)}_{ij}\; A_h\, (x_j^{(r)} - x_i^{(r)})$$

Update with gauge-fix:

$$x_i^{(r+1)} = \text{gauge\_fix}(x_i^{(r)} + \Delta x_i^{(r)})$$

Step size:

$$\eta_r = \sigma(\theta_r) \quad \text{with } \theta_r \text{ learned, but initialized so } \eta_0 \approx 0.6$$

This is explicitly to avoid D2's "wasted R0" issue.

### State update

$$m_i^{(r)} = \text{concat}_h \sum_j a^{(r,h)}_{ij}\, v_j^{(r,h)}$$

$$s_i^{(r+1)} = s_i^{(r)} + \text{MLP}\bigl([\text{LN}(s_i^{(r)}),\, m_i^{(r)}]\bigr)$$

(standard transformer residual block, with last layer zero-init).

---

## B4. Full loss (explicit)

Your fixed objectives:
* $F2$: multi-step RW transition matching via KL on $T^{(s)}_{\text{pred}}$.
* $F4$: multiscale log-stress on sparse multiscale edge set.

Total loss:

$$\mathcal{L} = \mathcal{L}_{\text{RW}} + \lambda_{\text{stress}}\mathcal{L}_{\text{stress}} + \lambda_{\text{rep}}\mathcal{L}_{\text{rep}} + \lambda_{\text{ent}}\mathcal{L}_{\text{ent}} + \lambda_{\text{gate}}\mathcal{L}_{\text{gate}} + \lambda_{\text{trust}}\mathcal{L}_{\text{trust}} + \lambda_{\text{rank}}\mathcal{L}_{\text{rank}}$$

Where:

* $\mathcal{L}_{\text{RW}}$ and $\mathcal{L}_{\text{stress}}$ are exactly your F2/F4 pipeline.
* $\mathcal{L}_{\text{gate}}$ is your existing KL gate supervision:

$$\mathcal{L}_{\text{gate}} = \sum_i \text{KL}(p_i \,\|\, q_i)$$

with $p_i$ from masked/renormalized $T^{(1)}_{\text{ref}}$ on candidate set and $q_i = \text{softmax}(g_{i\cdot})$.

* $\mathcal{L}_{\text{trust}}$: trust-region against pretrained gate (prevents drift without requiring huge $\lambda_{\text{gate}}$):

$$\mathcal{L}_{\text{trust}} = \sum_i \text{KL}\bigl(\text{softmax}(g^{\text{pre}}_{i\cdot}) \,\|\, \text{softmax}(g_{i\cdot})\bigr)$$

* $\mathcal{L}_{\text{rank}}$: ranking auxiliary that directly targets **precision**:

For each $i$, sample positives $j^+$ from top mass of $T^{(1)}_{\text{ref}}(i, \cdot)$, negatives $j^-$ from candidate non-edges:

$$\mathcal{L}_{\text{rank}} = \sum_{(i, j^+, j^-)} \max\bigl(0,\; m - g_{ij^+} + g_{ij^-}\bigr)$$

This is the simplest way to push "pick the right edge among 3 candidates."

**Suggested starting weights (toy):**
* $\lambda_{\text{stress}} = 0.1$ (as in D7 recipe).
* $\lambda_{\text{gate}} = 0.05$ (as in D7).
* $\lambda_{\text{trust}} = 0.02$
* $\lambda_{\text{rank}} = 0.02$
* keep your existing $\lambda_{\text{rep}}, \lambda_{\text{ent}}$ as in baseline.

---

## B5. What is frozen vs trained (explicit)

* **Frozen**: SharedEncoder (non-negotiable; already your protocol).
* **Trained**:
  * IPA backbone parameters (D7-style)
  * CGF gate parameters
* **Important constraint**: in the recommended training schedule below, the **structure loss gradients must not flow into the gate bias path** (to prevent drift). This is the key co-adaptation fix.

---

## B6. Why this is sample-efficient under minisets

1. **Weight sharing in the structure module** (D7 is ~155k params) matches the best empirical regime you already validated.
2. The extra capacity is focused on the **identified bottleneck** (gate precision), not on extra geo knobs that empirically don't help on inference graphs.
3. Strong capacity control in gate:
   * small hidden size, few layers,
   * explicit trust region to pretrained gate,
   * ranking loss to improve precision without requiring more recycles.

---

# Deliverable C — Fix the gating bottleneck (two strategies + pick one)

You asked for ≥2 strategies and one recommendation. Here are two.

## Strategy C1 (recommended): Contextual gate + two-timescale training + stop-grad gate bias

This directly addresses:
* **bad gating = 99.6% gap**
* **pretrained gate better than joint gate** (gate degrades during joint training)

### C1 training schedule (implementation-ready)

Use two optimizers (or two parameter groups with different lrs).

**Phase 0 — Cache targets**
* Cache $T^{(1)}_{\text{ref}}$ per training miniset for gate supervision (you already do).

**Phase 1 — Gate pretraining (no structure module)**
* Train CGF on $\mathcal{L}_{\text{gate}} + \lambda_{\text{rank}}\mathcal{L}_{\text{rank}} + \lambda_{\text{trust}}\mathcal{L}_{\text{trust}}$
* Use the *same H augmentation distribution* you will use later (test both H_clean and H_aug; see Experiment E2).
* Early stop on **val gate precision@K** and val gate KL (not train KL).
* Output: frozen snapshot `gate_pre.pt`.

**Phase 2 — Structure warmup with frozen gate, but low $\beta$**
* Freeze gate params.
* Train IPA backbone on F4 only for 1 epoch (your existing warmup), then start RW ramp.
* Set $\beta_r$ small initially:
  * $\beta_r(t) = \beta_{\max} \cdot \text{sigmoid}(\kappa(t - t_0))$ over epochs
  * start effectively near 0.3–0.6, ramp to 2.0 by epoch ~4–6.

This "β ramp" is explicitly listed as not-yet-tested and is high ROI given your co-adaptation trap.

**Phase 3 — Joint training, but *gate bias detached***

On each batch:
1. Compute gate logits $g$ with CGF.
2. **Detach** before feeding to structure:
   * `gb = (beta_r * g).detach()`
3. Forward IPA using `gate_bias=gb`.
4. Backprop:
   * structure optimizer gets gradients from F2/F4/regularizers,
   * gate optimizer gets gradients only from gate losses.

This prevents geometry gradients from "rewriting" the gate into a model-dependent shortcut, which is the simplest mechanistic explanation for "pretrained gate > joint gate" plus "frozen gate hurts."

**Optional alternation (two-timescale):**
* Every step: update gate once (gate loss only), update structure once (geo loss only).
* Or gate:structure update ratio 2:1 early, then 1:1.

### What to log to detect drift early

Log these on *val minisets* every epoch:
* gate_KL (val), gate_precision@30, gate_recall@30
* KL(current gate || pretrained gate)  (drift metric)
* entropy of gate distribution (too low = collapse, too high = useless)
* correlation of gate logits with $T_{\text{ref}}$ weights

You already have "attention diag" utilities; extend them to "gate diag."

### Ensure oracle eval never contaminated again

* Implement the D0 guardrail and assert the gate tensor is zero for oracle eval.
* Keep two separate eval functions:
  * `eval_hgate(model, gate, cand_mask)`
  * `eval_oracle(model, edge_mask)` with `gate_bias=None` or zeros

This is already documented as mandatory.

---

## Strategy C2 (alternative): Teacher-distillation from oracle pass to inference pass

Two-pass training on ST minisets:
1. Teacher: run model on oracle edges (T-edge), gate disabled.
2. Student: run on candidate edges + learned gate.
3. Distill a gauge-free object:
   * match $T^{(1)}_{\text{pred}}$ teacher vs student (operator distillation), and/or
   * match pairwise distance matrices up to monotone transform.

**Pros:**
* directly trains "inference path" to behave like oracle path.

**Cons:**
* more compute; can hide gate failures by pushing structure to compensate.

I would only do this if C1 fails to move $E_{\text{geo}}$ substantially.

---

# Deliverable B+C — The decisive "fancy IPA‑R" plan (one coherent story)

**Name it**: **Topo‑IPA‑R**

**Story** (paper framing):
* Learn a **gauge-free intrinsic geometry** from expression embeddings using only operator/stress supervision (F2/F4).
* Use a **structure module** (IPA-style equivariant refinement) that produces latent coordinates $X$ (gauge) solely to represent intrinsic geometry.
* Crucially, learn the **topology** needed for inference from embeddings alone via a contextual gate network trained to match the reference diffusion transitions $T^{(1)}_{\text{ref}}$.

This directly satisfies: ST→intrinsic geometry, scRNA inference graph from $H$ only, no coordinate regression to pixel frame.

---

# Deliverable E — 12 high-ROI toy experiments with decision logic

All experiments use:
* Split: Train ST1+ST2, Val ST3 (50 minisets) unless otherwise stated (and include slide-swap test later).
* Seeds: at least 3 seeds; stability experiments use 5.
* Budget: 12 epochs, warmup 1 epoch stress-only, RW ramp 6 epochs cap 0.3, patience=4 (your recipe).
* Always report both:
  * Val(H-gate) $E_{\text{geo}}$
  * Val(T-edge) $E_{\text{geo}}$ with gate=0 (oracle)

I'll format each as: Goal → change → protocol → success → decision.

---

## E1. Stop-grad gate bias (drift killer) — *must-run*

**Goal:** test whether gate drift is caused by structure gradients.

**Change:** `gb = beta * gate(H, cand)` vs `gb = (beta * gate(H, cand)).detach()`.

**Protocol:** D7 backbone + current EdgeGateSoft (no CGF yet), 3 seeds. Log gate_KL(train/val), gate_precision@30, $E_{\text{geo}}$.

**Success:** gate_KL(val) no longer degrades after joint starts; $E_{\text{geo}}$ improves $\geq 0.03$ or stays same but gate metrics improve.

**Decision:**
* If success → adopt stop-grad in the final system and proceed to contextual gate.
* If no change → drift is not gradient-driven; focus on objective/mismatch (E2/E3).

---

## E2. Gate sees H_aug vs H_clean (train/eval mismatch check)

**Goal:** diagnose co-adaptation trap due to gate/structure seeing different inputs.

**Change:** compute gate on `H_aug` (same augmentation as structure) vs `H_clean`.

**Protocol:** D7 + stop-grad from E1 on, 3 seeds.

**Success:** improved gate precision@30 and/or lower seed variance; $E_{\text{geo}}$ improves $\geq 0.02$.

**Decision:** pick the better variant as default.

---

## E3. β ramp schedule (stability + better refinement)

**Goal:** prevent early poisoning by inaccurate gate; match "β ramp not tested" open question.

**Change:** $\beta(t)$ ramp $0.5 \to 2.0$ over epochs 1–6 vs constant $\beta = 2.0$.

**Protocol:** D7 + stop-grad, 5 seeds (stability requirement).

**Success:** $\text{std}(E_{\text{geo}}) \leq 0.02$ and mean improves $\geq 0.02$.

**Decision:** keep ramp if it improves stability without hurting oracle.

---

## E4. CGF vs MLP gate (main bottleneck experiment)

**Goal:** improve gate precision inside high-recall candidate set.

**Change:** replace EdgeGateSoft with Contextual GateFormer (2 layers).

**Protocol:**
* Pretrain gate 2k steps on gate losses.
* Joint with stop-grad gate bias, β ramp.
* 3 seeds.

**Success:** gate precision@30 increases by $\geq 0.07$ absolute AND $E_{\text{geo}}(\text{H-gate})$ drops to $\leq 0.38$ (closing $\geq 22\%$ of current gap).

**Decision:**
* If success → CGF is the final shipped gate.
* If gate improves but $E_{\text{geo}}$ doesn't → structure module not using gate (check attention $\rho_{\text{gate}}$).
* If neither improves → need different gate objective (E8).

---

## E5. Gate pretraining length sweep (capacity vs data)

**Goal:** see if gate is undertrained.

**Change:** pretrain steps $\{500, 2000, 8000\}$.

**Protocol:** CGF + same joint schedule, 3 seeds.

**Success:** monotone improvement in gate precision and $E_{\text{geo}}$ up to some saturation.

**Decision:** pick the knee; if longer pretrain doesn't help, gate needs objective/architecture changes.

---

## E6. Two-timescale alternation (gate update frequency)

**Goal:** reduce co-adaptation instability without freezing.

**Change:** per batch:
* A: update both each step,
* B: update gate every step, structure every step (baseline),
* C: update gate twice per step (2:1) for first 3 epochs.

**Protocol:** CGF + stop-grad, 5 seeds.

**Success:** fewer collapses + lower std + improved gate precision.

**Decision:** choose the most stable that doesn't hurt mean.

---

## E7. Gate trust-region strength sweep

**Goal:** stop "gate gets worse than pretrained" (E8c) without freezing.

**Change:** $\lambda_{\text{trust}} \in \{0, 0.01, 0.02, 0.05\}$.

**Protocol:** CGF + stop-grad, 3 seeds.

**Success:** gate drift metric $\text{KL}(\text{current} \| \text{pre})$ stays small while $E_{\text{geo}}$ improves.

**Decision:** choose smallest $\lambda$ that prevents drift.

---

## E8. Ranking auxiliary loss on gate (precision-focused)

**Goal:** directly attack "must discriminate among ~3 candidates per true edge."

**Change:** add $\lambda_{\text{rank}} \in \{0, 0.01, 0.02, 0.05\}$.

**Protocol:** CGF, 3 seeds.

**Success:** precision@30 increases; $E_{\text{geo}}$ improves.

**Decision:** keep if it improves precision without increasing variance.

---

## E9. $\eta_0$ floor (avoid wasting R0)

**Goal:** test A2 hypothesis in the *final* backbone.

**Change:** enforce $\eta_0 \geq 0.5$ (clamp sigmoid output) vs fully learned.

**Protocol:** D7 + CGF gate, 3 seeds.

**Success:** $|\Delta X|$ at R0 increases and $E_{\text{geo}}$ improves $\geq 0.02$.

**Decision:** keep $\eta_0$ floor if stable.

---

## E10. Candidate-k robustness (must pass before scaling)

**Goal:** ensure the gate solution is not brittle to k.

**Change:** evaluate $k \in \{80, 100, 120\}$ at inference; training fixed at 100.

**Protocol:** 3 seeds on best model.

**Success:** $E_{\text{geo}}$ variation $\leq 0.03$ across ks; gate precision decreases smoothly, not catastrophically.

**Decision:** if brittle → gate is overfitting to candidate construction; add candidate-drop augmentation or train with mixed k.

---

## E11. Cross-slide swap robustness

**Goal:** ensure not learning slide shortcuts (low-data reality).

**Change:** three folds:
* train (ST1,ST2) val ST3,
* train (ST1,ST3) val ST2,
* train (ST2,ST3) val ST1.

**Protocol:** 3 seeds each, same hyperparams.

**Success:** mean $E_{\text{geo}}$ across folds within $\pm 0.03$.

**Decision:** if one fold collapses, gate may be learning slide-specific cues; increase augmentation or add explicit slide-invariance regularizer to $H$ (encoder-side) or gate.

---

## E12. Two poison-pill leakage tests (must always fail "in the right direction")

### Poison A: shuffle H within each miniset

**Goal:** confirm model uses $H$ and cannot succeed via graph artifacts.

**Change:** permute $H$ rows but keep targets the same.

**Protocol:** eval-only on trained checkpoint.

**Success:** $E_{\text{geo}}$ degrades $\geq 6\times$ (consistent with your prior control).

**Decision:** if it doesn't degrade, there is leakage or target reuse.

### Poison B: random candidate graph with same degree

**Goal:** ensure candidate topology matters and you're not accidentally using hidden spatial edges.

**Change:** replace candidate mask with random symmetric mask matching degrees.

**Success:** $E_{\text{geo}}$ approaches "no_gate" catastrophic regime (~0.88) and attention recall collapses.

**Decision:** if it doesn't, there is contamination in graph construction.

---

# Deliverable F — Scale-readiness thresholds (hard gates)

These are **go/no-go** gates before you spend on large-scale training.

## F1. Inference gap closure

* Define gap: $\Delta = E_{\text{H-gate}} - E_{\text{T-edge}}$.
* Current: $0.440 - 0.162 = 0.278$.

**Scale-ready threshold:**
* $E_{\text{H-gate}} \leq 0.35$ **and** $\Delta \leq 0.18$.

(That's ~35% gap reduction vs current, without needing oracle improvements.)

## F2. Multi-seed stability

* 5 seeds on the full toy protocol.
* **$\text{Std}(E_{\text{geo}}) \leq 0.02$** and **no NaNs**.
* Early stopping epoch variance $\leq 3$ epochs.

## F3. No silent geometric collapse

Compute on val minisets:
* covariance eigenvalue ratio: $\lambda_2 / \lambda_1 \geq 0.15$ (2D non-line collapse)
* clumping proxy $p_5(\text{nn\_dist}) \geq 0.05$ (your clumping helper computes this)
* median pairwise distance stable (not near-zero)

Any run violating these is "fail," even if $E_{\text{geo}}$ looks acceptable (because collapse can hide in operator matching).

## F4. Candidate-k robustness

* Evaluate $k \in \{80, 100, 120\}$.
* **Max–min $E_{\text{geo}} \leq 0.03$** (same model, no retraining).

## F5. Cross-slide robustness

* 3-fold slide swap (E11).
* Worst fold $E_{\text{geo}}$ no worse than best fold by $> 0.05$.

## F6. Attention-health metrics (only if you claim interpretability)

Report mean±std over val minisets:
* AttnRecall@30 per recycle stays $\geq 0.40$ (no "collapse" narrative). Your D7 recipe already tracks this.
* logit_std per recycle stays in a bounded range (e.g., 2–10; no D2-like explosion).

---

# Deliverable G — "Paper-grade" attention visualizations (non-cherry-picked)

You want transformer-style attention figures that reviewers can't dismiss as cherry-picked. The key is **deterministic selection + aggregate stats + tying attention to F2/F4.**

## G1. Three aggregate plots (mean±std, many minisets, deterministic)

All computed on a deterministic fixed list: e.g., the first 50 val minisets after sorting by miniset_id.

### Plot 1 — Attention localization across recycles

For each recycle $r$:
* AttnRecall@K ($K=30$) against true spatial neighbors
* mean±std across minisets and heads

This is already consistent with your diagnostics tables.

### Plot 2 — Logit term scale + entropy per recycle

For each recycle $r$:
* logit_std of the full logits (per head, mean across edges)
* entropy of attention distribution (per head)

Mean±std across minisets.

This directly demonstrates your "no explosion, no collapse" claim and highlights the D2 failure mode vs the shipped model.

### Plot 3 — Gradient–attention alignment (ties to F2/F4)

For each loss family separately (RW edges vs stress edges):
* compute per-edge gradient magnitude $|\partial \mathcal{L} / \partial d_{ij}|$ or $|\partial \mathcal{L} / \partial x|$ projected onto edges
* correlate with attention mass on the same edges (Spearman $\rho$)

Plot $\rho$ distributions (box/violin) across minisets.

This defends interpretability: attention is not arbitrary; it aligns with edges that matter.

(Your D5 infrastructure already does this analysis.)

---

## G2. Two qualitative templates (deterministic query selection)

### Template 1 — Attention focus maps on ST layout

Use a deterministic query node selection per miniset:
* choose the node with **median gate entropy** (neither trivial nor extreme),
  or if you want one per class: pick top-1 by degree then median.

Plot:
* true spatial coords for visualization only (explicitly label as such),
* candidate neighbors colored by attention weight,
* true spatial neighbors ringed.

This is already aligned with your existing visualization schema.

### Template 2 — Coordinate progression across recycles (gauge-fixed)

For the same deterministic miniset + query selection:
* show $X$ after each recycle (gauge-fixed),
* overlay a small subset of highest-attended edges (top-5),
* show how refinement stabilizes.

Again, consistent with your existing "coordinate progression" figure style.

**Rule to prevent cherry-picking:** pre-register the selection: "for each fold, we visualize minisets #0, #17, #34 and query node = median gate-entropy node," fixed before training.

---

# Explicit answer to your "viability" question (under your low-data constraints)

A fully integrated "AlphaFold-ish" IPA‑R with many per-recycle knobs (pair bias + point IPA + per-recycle $\tau$/$\beta$ + unshared blocks) is **not sample-efficient in your current miniset + few-slide regime**; D2 already shows it underperforms and the log points to component interference and wasted R0 dynamics.

A "fancy IPA‑R" **is viable** if you:
1. keep the structure module close to D7 (shared, QK-norm, safe geo schedule),
2. invest sophistication where the gap actually is: **contextual topology/gating**, and
3. enforce training dynamics that prevent gate drift (stop-grad + two-timescale).

That's the Topo‑IPA‑R plan above.

---

# Reference implementations to ground "fancy IPA‑R" components + stability conventions

Use these as the "official/public" anchors in the paper and as implementation sanity references:

| Component | Reference |
|---|---|
| IPA + structure module conventions | [OpenFold](https://github.com/aqlaboratory/openfold) |
| Recycling / IPA reference | [AlphaFold](https://github.com/deepmind/alphafold) |
| Graph transformer with attention bias | [Graphormer](https://github.com/microsoft/Graphormer) |
| GAT/GNN building blocks | [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) |
| Fallback equivariant GNN baseline | [EGNN](https://github.com/vgsatorras/egnn) |

**Stability conventions you should explicitly enforce (some you already do):**

* **Zero-init** for coordinate update projection (`linear_x`) and last MLP layers (prevents early explosions).
* **QK normalization + bounded $\tau$** (prevents D2 R0 blow-up).
* **Clamped geo terms** and **geo delayed** until $r \geq 2$ (prevents self-reinforcing wrong layouts).
* **Oracle eval must use gate=0** (D3 fix as invariant).
* **Separate optimizers / stop-grad on gate bias** to prevent gate drift (your E8c pattern).

---

# One decision tree (what to do next, no branching into 3 papers)

1. **Add stop-grad gate bias** (E1) to current D7 + EdgeGateSoft.
   * If gate drift disappears and $E_{\text{geo}}$ improves → keep and proceed.

2. **Add β ramp** (E3).
   * If stability improves → keep.

3. **Swap EdgeGateSoft → Contextual GateFormer** (E4) with pretrain + trust + ranking.
   * If $E_{\text{geo}} \leq 0.38$ and stable → this is the shipped model.

4. If CGF improves gate metrics but not $E_{\text{geo}}$ → run attention $\rho_{\text{gate}}$ and ensure gate bias actually influences logits; if it does, re-check structure module uses it (masking/broadcast).

5. If still stuck at ~0.44 ceiling → only then consider distillation (C2).
