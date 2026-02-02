# Multi-Patient Multi-Source Embedding Alignment Problem

## Overview

You are working on GEMS (Gene Expression Mapping System), a system that maps single-cell RNA sequencing (scRNA-seq) data to spatial coordinates. The system has two main components:

1. **Encoder**: Maps gene expression → embedding space (trained with self-supervised learning)
2. **Diffusion Model**: Maps embedding → spatial coordinates (trained on ST data with known geometry)

**Critical Insight**: The diffusion model is trained ONLY on ST1/ST2 data (which has spatial geometry). At inference, it must work on ST3 and SC data from ANY patient. This ONLY works if ALL sources produce indistinguishable embeddings.

## The Problem

Despite extensive efforts, embeddings from different sources remain separated:

```
Current Centroid Distance Matrix (after all alignment attempts):
           P2_ST1   P2_ST2   P2_ST3    P2_SC  P10_ST3   P10_SC
P2_ST1      0.000    0.421    0.685    2.492    0.863    0.576
P2_ST2      0.421    0.000    0.201    2.505    0.790    0.440
P2_ST3      0.685    0.201    0.000    2.623    0.745    0.597
P2_SC       2.492    2.505    2.623    0.000    3.183    2.102
P10_ST3     0.863    0.790    0.745    3.183    0.000    1.119
P10_SC      0.576    0.440    0.597    2.102    1.119    0.000
```

**Observations**:
- ST sources cluster well together (distances 0.2-0.8) ✓
- SC sources have some alignment (P2_SC↔P10_SC = 2.102)
- **PROBLEM**: ST↔SC gap persists (~2.5-3.5 distance)
  - P2_ST3 ↔ P2_SC = 2.623 (same patient, should be ~0)
  - P10_ST3 ↔ P10_SC = 3.183 (same patient, should be ~0)

## Data Structure

```
Data sources:
├── Patient P2
│   ├── P2_ST1: ~1700 spots (Spatial Transcriptomics, has geometry)
│   ├── P2_ST2: ~1500 spots (Spatial Transcriptomics, has geometry)
│   ├── P2_ST3: ~1200 spots (Spatial Transcriptomics, NO geometry for training)
│   └── P2_SC:  ~2700 cells (Single-cell, NO geometry)
├── Patient P10
│   ├── P10_ST3: ~460 spots (Spatial Transcriptomics, NO geometry for training)
│   └── P10_SC:  ~950 cells (Single-cell, NO geometry)
```

**Training data split**:
- **ST data** (for diffusion): ST1 + ST2 spots with geometry labels
- **All data** (for encoder): ST1 + ST2 + ST3 + SC from all patients

**Goal**: Encoder must make ALL 6 sources indistinguishable so diffusion model trained on ST1/ST2 geometry generalizes to ST3 and SC.

## Current Architecture

### Encoder Training (train_encoder function)

```python
def train_encoder(
    st_expr,           # (n_st, n_genes) - Combined ST1+ST2+ST3 expression
    sc_expr,           # (n_sc, n_genes) - SC expression
    st_slide_ids,      # (n_st,) - Which slide each ST spot belongs to
    sc_slide_ids,      # (n_sc,) - Which "slide" each SC cell belongs to
    st_source_ids,     # (n_st,) - Source ID (0=P2_ST1, 1=P2_ST2, 2=P2_ST3, 4=P10_ST3)
    sc_source_ids,     # (n_sc,) - Source ID (3=P2_SC, 5=P10_SC)
    ...
):
```

### Loss Components Currently Used

1. **VICReg Loss** (self-supervised):
   - Invariance: embeddings of augmented views should match
   - Variance: maintain variance in embedding dimensions
   - Covariance: decorrelate embedding dimensions

2. **Domain Adversary with GRL** (binary ST=0, SC=1):
   - Discriminator tries to predict ST vs SC
   - Gradient Reversal Layer makes encoder confuse discriminator
   - Currently achieving ~50% accuracy (good, means confused)

3. **Domain CORAL** (ST vs SC):
   - Aligns mean and covariance of ST and SC distributions
   - `coral_loss(z_st, z_sc)`

4. **Patient CORAL** (within SC only):
   - Aligns SC distributions across patients
   - `coral_loss(z_sc_p2, z_sc_p10)`

5. **Pairwise Source CORAL** (all 15 pairs):
   - Computes CORAL between every pair of sources
   - Should align all sources explicitly

6. **Local Alignment Loss**:
   - Uses expression similarity as teacher
   - Aligns embeddings based on gene expression neighbors

### Key Code Locations

- **`/home/user/sc_st/model/core_models_et_p1.py`**: `train_encoder` function
- **`/home/user/sc_st/model/ssl_utils.py`**: Sampling utilities, CORAL loss
- **`/home/user/sc_st/model/adversarial_utils.py`**: Domain discriminator, GRL

## What Has Been Tried and Why It Failed

### Attempt 1: Binary Domain Adversary (ST=0, SC=1)
- **Approach**: Discriminator distinguishes ST vs SC, GRL confuses it
- **Result**: Discriminator accuracy ~50% (good), BUT ST-SC gap remains
- **Why it failed**: Adversary can be confused while distributions are still different. The discriminator being confused doesn't mean distributions are aligned - it might just mean the discriminator is weak or the signal is hard to learn.

### Attempt 2: N-class Source Adversary (6 classes)
- **Approach**: Discriminator distinguishes all 6 sources
- **Result**: CRASHED - discriminator collapsed to always predicting majority class (P2_SC with 2688 samples)
- **Why it failed**: Severe class imbalance. With 2688 samples in one class vs 462 in another, the network just learns to always predict the majority class.

### Attempt 3: Domain CORAL (mean + covariance alignment)
- **Approach**: Explicitly align first and second moments of ST and SC
- **Result**: Minimal improvement
- **Why it failed**: CORAL aligns global statistics but doesn't guarantee point-wise alignment. Two distributions can have same mean/covariance but be in different locations.

### Attempt 4: Pairwise Source CORAL (all 15 pairs)
- **Approach**: Compute CORAL between every pair of sources
- **Result**: Minimal improvement even with weight=500
- **Why it failed**: Same as above - moment matching doesn't ensure actual overlap

### Attempt 5: More Genes (28 → 88)
- **Approach**: Relaxed gene filtering to include more shared genes
- **Result**: No improvement
- **Why it failed**: The problem isn't lack of genes, it's the alignment mechanism

## Fundamental Issue

The ST↔SC gap likely stems from **biological differences** between:
- **Spatial Transcriptomics**: Measures spots (multiple cells), has spatial context
- **Single-cell RNA-seq**: Measures individual cells, dissociated from tissue

These create systematic batch effects that our current alignment methods don't fully address.

## Potential Solutions to Explore

### 1. Optimal Transport Based Alignment
Instead of moment matching (CORAL), use optimal transport to find actual correspondences:
```python
# Sinkhorn distance / Wasserstein distance
from geomloss import SamplesLoss
ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
loss = ot_loss(z_st, z_sc)
```

### 2. Contrastive Domain Adaptation
Use contrastive learning across domains:
- Positive pairs: similar expression, different domains
- Negative pairs: different expression
```python
# Find expression-similar pairs across ST and SC
# Pull them together in embedding space
```

### 3. Stronger Adversarial Training
- Use Wasserstein GAN-style discriminator (more stable)
- Multi-scale discriminator
- Feature matching loss instead of just adversarial
```python
# Feature matching: match intermediate discriminator features
disc_features_st = discriminator.get_features(z_st)
disc_features_sc = discriminator.get_features(z_sc)
loss_fm = F.mse_loss(disc_features_st.mean(0), disc_features_sc.mean(0))
```

### 4. Maximum Mean Discrepancy (MMD)
Alternative to CORAL that uses kernel embedding:
```python
def mmd_loss(x, y, kernel='rbf'):
    xx = kernel_matrix(x, x)
    yy = kernel_matrix(y, y)
    xy = kernel_matrix(x, y)
    return xx.mean() + yy.mean() - 2 * xy.mean()
```

### 5. Domain-Specific Batch Normalization + Shared Encoder
- Separate batch norm statistics for ST and SC
- Shared encoder weights
- Forces encoder to learn domain-invariant features

### 6. Cycle Consistency
If you can "translate" ST embeddings to SC space and back:
```python
# z_st -> translated_sc -> reconstructed_st
# loss = |z_st - reconstructed_st|
```

### 7. Pseudo-Labeling / Self-Training
1. Train encoder on ST data with geometry
2. Predict geometry for SC data
3. Use confident predictions as pseudo-labels
4. Retrain including pseudo-labeled SC

### 8. Expression-Guided Contrastive Loss
Create positive pairs based on expression similarity across domains:
```python
# For each SC cell, find most similar ST spot by expression
# These become positive pairs in contrastive loss
exp_sim = cosine_similarity(sc_expr, st_expr)
for i, sc_cell in enumerate(sc_embeddings):
    closest_st = st_embeddings[exp_sim[i].argmax()]
    positive_pairs.append((sc_cell, closest_st))
```

## Key Hyperparameters

Current settings:
```python
vicreg_weight = 1.0
adversarial_weight = 1.0
coral_weight = 10.0
patient_coral_weight = 100.0
source_coral_weight = 500.0
local_align_weight = 1.0
batch_size = 256
learning_rate = 1e-4
```

## Success Criteria

The alignment is successful when:
1. All pairwise centroid distances < 1.0
2. UMAP shows complete overlap of all 6 sources
3. Diffusion model trained on ST1/ST2 produces accurate reconstructions for ST3 and SC

## Files You Need to Modify

1. **`/home/user/sc_st/model/core_models_et_p1.py`** - `train_encoder` function
2. **`/home/user/sc_st/model/ssl_utils.py`** - Add new loss functions
3. **`/home/user/sc_st/model/adversarial_utils.py`** - Modify discriminator if needed

## Current Branch

Work on branch: `claude/setup-patient-training-stage-SxmKc`

## Data Loading Code

The multi-patient data is loaded as:
```python
# Load all sources
p2_st1_expr, p2_st2_expr, p2_st3_expr, p2_sc_expr = load_patient_data('P2')
p10_st3_expr, p10_sc_expr = load_patient_data('P10')

# Concatenate for encoder training
st_expr = torch.cat([p2_st1_expr, p2_st2_expr, p2_st3_expr, p10_st3_expr])
sc_expr = torch.cat([p2_sc_expr, p10_sc_expr])

# Source IDs
st_source_ids = torch.cat([
    torch.full((len(p2_st1),), 0),  # P2_ST1
    torch.full((len(p2_st2),), 1),  # P2_ST2
    torch.full((len(p2_st3),), 2),  # P2_ST3
    torch.full((len(p10_st3),), 4), # P10_ST3
])
sc_source_ids = torch.cat([
    torch.full((len(p2_sc),), 3),   # P2_SC
    torch.full((len(p10_sc),), 5),  # P10_SC
])
```

## Summary

The core problem is that despite:
- Binary domain adversary (achieving 50% confusion)
- Domain CORAL
- Patient CORAL
- Pairwise source CORAL (weight=500)
- Local alignment loss

ST and SC embeddings remain ~2.5-3.5 apart in embedding space. This breaks the diffusion model's ability to generalize from ST1/ST2 (training) to SC (inference).

The solution likely requires going beyond moment-matching (CORAL) to methods that enforce actual distributional overlap (optimal transport, strong contrastive learning, or feature matching).
