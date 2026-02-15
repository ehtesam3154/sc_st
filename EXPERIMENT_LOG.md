# Experiment Log: Cross-Slide Alignment (Mouse Liver)

## Setup
- **Dataset**: Mouse liver ST, 4 slides (ST1-ST3 training, ST4 inference)
- **Encoder**: SharedEncoder (n_genes → 512 → 256 → 128), VICReg + adversarial + Spatial NCE
- **Preprocessing**: `normalize_total` + `log1p`, common gene intersection across all slides
- **Notebook**: `liver_encoder_v2.ipynb`

---

## Hypothesis H1: QC Covariate Leakage

**Claim**: Library size, sparsity, detected gene counts differ systematically across slides. Even after normalization, these QC features leak through and let a classifier trivially separate slides.

**Tests run**:
1. Computed 7 per-cell QC features (n_detected, total_expr, mean_expr, zero_frac, var_expr, max_expr, mean_nonzero) — all from expression only, no coordinates
2. Trained LogisticRegression (5-fold CV) on QC features alone → slide prediction
3. Trained LogisticRegression on Raw-PCA(50) → slide prediction
4. Trained LogisticRegression on QC + PCA combined

**Results** (PENDING — fill in after running):
- QC-only 5-fold accuracy: ___
- Raw-PCA(50) 5-fold accuracy: ___
- QC+PCA combined accuracy: ___
- Chance level: 0.333

**Interpretation**: (fill after running)

---

## Hypothesis H2: Per-gene Mean/Variance Shifts (Batch Effect)

**Claim**: Different slides have systematic gene-wise expression shifts due to chemistry, section thickness, permeability — classic batch effect pattern.

**Tests run**:
1. **2a**: Kruskal-Wallis DE test for every gene across 3 slides, FDR-corrected. Classified top DE genes as Mito/Ribo/Stress-IEG/Ambient-Hepato/Other.
2. **2b**: Ablation — removed top {50, 100, 200, 500, 1000} DE genes, recomputed PCA(50), re-tested slide classifier accuracy. Checked Moran's I on ablated PCs to verify spatial structure preserved.
3. **2c**: Batch correction — applied Harmony, per-slide mean-centering, per-slide z-scoring. Re-tested PCA slide separability after each.

**Results** (PENDING — fill in after running):
- Significant DE genes (FDR<0.05): ___/___ (___%)
- Top 100 DE gene categories: Mito=___, Ribo=___, Stress/IEG=___, Ambient/Hepato=___, Other=___
- Ablation curve:
  - Remove 0: ___
  - Remove 50: ___
  - Remove 100: ___
  - Remove 200: ___
  - Remove 500: ___
  - Remove 1000: ___
- Batch correction:
  - Raw PCA(50): ___
  - Mean-centered: ___
  - Z-scored: ___
  - Harmony: ___
- Spatial sanity (Moran's I after remove-500): ___

**Interpretation**: (fill after running)

---

## Key Findings So Far

| Finding | Status | Implication |
|---------|--------|-------------|
| Raw PCA(50) already ~94.6% slide-separable (H1 prediction) | Predicted | Slide signal exists in raw expression before encoder |
| Encoder is expression-only (no coords enter the network) | Confirmed | Spatial NCE uses coords externally for +/- pair selection only |
| `normalize_total` removes gross library size but not zero-patterns | Known | QC leakage possible through sparsity structure |

---

## Hypotheses Queue

- [x] H1: QC covariate leakage (TESTED)
- [x] H2: Per-gene mean/variance shifts (TESTED)
- [ ] H3: (next — to be defined)
- [ ] H4: (to be defined)

---

## Design Decisions & Notes

- All H1/H2 tests use **expression only** (no spatial coordinates) because the encoder input is expression-only
- Kruskal-Wallis chosen over t-test for DE because it's non-parametric and handles >2 groups
- Gene classification uses mouse naming conventions (mt- for mito, Rpl/Rps for ribo)
- Ambient/hepatocyte gene list based on known highly-abundant liver markers that dominate ambient RNA
