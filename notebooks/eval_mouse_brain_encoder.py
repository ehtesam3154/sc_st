#!/usr/bin/env python3
"""
Evaluate Mouse Brain Shared Encoder Quality

Data: 1 ST slide (~1300 spots) + 1 SC dataset (~10.5k cells)
Checkpoint: mouse_brain_res_curr_v2/ckpt_latest.pt
"""

# ===================================================================
# CELL 1: IMPORTS AND PATHS
# ===================================================================
import sys
sys.path.insert(0, '/home/ehtesamul/sc_st/model')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from core_models_et_p1 import SharedEncoder
from core_models_et_p3 import GEMSModel, infer_anchor_train_from_checkpoint
from ssl_utils import set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Data paths
ST_COUNTS = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st3_counts_et.csv'
ST_META   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st3_metadata_et.csv'
ST_CT     = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st3_celltype_et.csv'

SC_COUNTS = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_sc_counts.csv'
SC_META   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/metadata.csv'

CHECKPOINT_PATH = "/home/ehtesamul/sc_st/model/mouse_brain_res_curr_v2/ckpt_latest.pt"

SEED = 42
set_seed(SEED)


# ===================================================================
# CELL 2: LOAD DATA
# ===================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

# Load ST data
st_counts = pd.read_csv(ST_COUNTS, index_col=0)
st_meta = pd.read_csv(ST_META, index_col=0)
st_ct = pd.read_csv(ST_CT, index_col=0)

# Load SC data
sc_counts = pd.read_csv(SC_COUNTS, index_col=0)
sc_meta = pd.read_csv(SC_META, index_col=0)

print(f"ST counts shape: {st_counts.shape}")
print(f"SC counts shape: {sc_counts.shape}")

# Get common genes
common_genes = sorted(list(set(st_counts.columns) & set(sc_counts.columns)))
print(f"Common genes: {len(common_genes)}")

# Subset to common genes
st_counts = st_counts[common_genes]
sc_counts = sc_counts[common_genes]

# Normalize (total count + log1p)
def normalize_counts(df):
    """Normalize counts: total count normalization + log1p"""
    X = df.values.astype(np.float32)
    # Total count normalize to 10000
    X = X / X.sum(axis=1, keepdims=True) * 10000
    # Log1p
    X = np.log1p(X)
    return X

X_st = normalize_counts(st_counts)
X_sc = normalize_counts(sc_counts)

print(f"\nAfter normalization:")
print(f"  X_st shape: {X_st.shape} (ST spots)")
print(f"  X_sc shape: {X_sc.shape} (SC cells)")

n_genes = X_st.shape[1]


# ===================================================================
# CELL 3: LOAD ENCODER FROM CHECKPOINT
# ===================================================================
print("\n" + "="*70)
print("LOADING ENCODER FROM CHECKPOINT")
print("="*70)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Create encoder
encoder = SharedEncoder(
    n_genes=n_genes,
    n_embedding=[512, 256, 128],
    dropout=0.1
).to(device)

# Load encoder weights
if 'encoder' in checkpoint:
    encoder.load_state_dict(checkpoint['encoder'])
    print("✓ Loaded encoder from checkpoint['encoder']")
else:
    print("⚠️ 'encoder' key not found in checkpoint")
    print("  Available keys:", list(checkpoint.keys()))

encoder.eval()
print(f"Encoder output dim: 128")


# ===================================================================
# CELL 4: COMPUTE EMBEDDINGS
# ===================================================================
print("\n" + "="*70)
print("COMPUTING EMBEDDINGS")
print("="*70)

X_st_tensor = torch.tensor(X_st, dtype=torch.float32, device=device)
X_sc_tensor = torch.tensor(X_sc, dtype=torch.float32, device=device)

with torch.no_grad():
    Z_st = encoder(X_st_tensor)
    Z_sc = encoder(X_sc_tensor)

print(f"Z_st shape: {Z_st.shape}")
print(f"Z_sc shape: {Z_sc.shape}")

# Compute normalized versions
Z_st_ln = F.layer_norm(Z_st, (Z_st.shape[1],))
Z_sc_ln = F.layer_norm(Z_sc, (Z_sc.shape[1],))

Z_st_norm = F.normalize(Z_st, dim=1)
Z_sc_norm = F.normalize(Z_sc, dim=1)

print("✓ Computed Z_raw, Z_ln, Z_norm for ST and SC")


# ===================================================================
# CELL 5: VISUALIZATION 1 - X vs Z_raw vs Z_ln vs Z_norm (4 panels)
# ===================================================================
print("\n" + "="*70)
print("VIS 1: Side-by-side embeddings - X vs Z_raw vs Z_ln vs Z_norm")
print("="*70)

N_VIS = 2000  # samples per domain for visualization

def subsample_matched(X, Z, n_vis):
    """Subsample with matched indices."""
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Z, torch.Tensor):
        Z = torch.tensor(Z, dtype=torch.float32)

    X = X.cpu()
    Z = Z.cpu()

    n = min(X.shape[0], Z.shape[0])
    X = X[:n]
    Z = Z[:n]

    if n <= n_vis:
        return X, Z
    idx = torch.randperm(n)[:n_vis]
    return X[idx], Z[idx]

# Subsample each domain
X_st_vis, Z_st_vis = subsample_matched(X_st, Z_st, N_VIS)
X_sc_vis, Z_sc_vis = subsample_matched(X_sc, Z_sc, N_VIS)

n_st_vis = X_st_vis.shape[0]
n_sc_vis = X_sc_vis.shape[0]

print(f"Visualization samples: ST={n_st_vis}, SC={n_sc_vis}")

# Concatenate
X_all_vis = torch.cat([X_st_vis, X_sc_vis], dim=0).numpy()
Z_all_vis = torch.cat([Z_st_vis, Z_sc_vis], dim=0)

labels_vis = np.array(['ST'] * n_st_vis + ['SC'] * n_sc_vis)

# Compute normalizations
Z_all_vis_raw = Z_all_vis.numpy()
Z_all_vis_ln = F.layer_norm(Z_all_vis, (Z_all_vis.shape[1],)).numpy()
Z_all_vis_norm = F.normalize(Z_all_vis, dim=1).numpy()

# Color scheme
colors_map = {
    'ST': '#FF6B35',  # orange
    'SC': '#A82FFC'   # purple
}

# PCA projections
print("Computing PCA projections...")

pca_x = PCA(n_components=2, random_state=42)
X_pca = pca_x.fit_transform(X_all_vis)
var_x = pca_x.explained_variance_ratio_

pca_z_raw = PCA(n_components=2, random_state=42)
Z_pca_raw = pca_z_raw.fit_transform(Z_all_vis_raw)
var_z_raw = pca_z_raw.explained_variance_ratio_

pca_z_ln = PCA(n_components=2, random_state=42)
Z_pca_ln = pca_z_ln.fit_transform(Z_all_vis_ln)
var_z_ln = pca_z_ln.explained_variance_ratio_

pca_z_norm = PCA(n_components=2, random_state=42)
Z_pca_norm = pca_z_norm.fit_transform(Z_all_vis_norm)
var_z_norm = pca_z_norm.explained_variance_ratio_

# Plot 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('white')

# Panel A: PCA on X
ax = axes[0, 0]
for label in ['ST', 'SC']:
    mask = labels_vis == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors_map[label], label=label, alpha=0.6, s=25, edgecolors='none')
ax.set_title(f'(A) PCA on X (log1p expr)\nVar: PC1={var_x[0]:.2%}, PC2={var_x[1]:.2%}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# Panel B: PCA on Z_raw
ax = axes[0, 1]
for label in ['ST', 'SC']:
    mask = labels_vis == label
    ax.scatter(Z_pca_raw[mask, 0], Z_pca_raw[mask, 1],
               c=colors_map[label], label=label, alpha=0.6, s=25, edgecolors='none')
ax.set_title(f'(B) PCA on Z_raw (encoder output)\nVar: PC1={var_z_raw[0]:.2%}, PC2={var_z_raw[1]:.2%}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# Panel C: PCA on Z_ln (LayerNorm only)
ax = axes[1, 0]
for label in ['ST', 'SC']:
    mask = labels_vis == label
    ax.scatter(Z_pca_ln[mask, 0], Z_pca_ln[mask, 1],
               c=colors_map[label], label=label, alpha=0.6, s=25, edgecolors='none')
ax.set_title(f'(C) PCA on Z_ln (LayerNorm only)\nVar: PC1={var_z_ln[0]:.2%}, PC2={var_z_ln[1]:.2%}\n⚠️ Adversary trained on this',
             fontsize=13, fontweight='bold', color='darkred')
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# Panel D: PCA on Z_norm (L2 only)
ax = axes[1, 1]
for label in ['ST', 'SC']:
    mask = labels_vis == label
    ax.scatter(Z_pca_norm[mask, 0], Z_pca_norm[mask, 1],
               c=colors_map[label], label=label, alpha=0.6, s=25, edgecolors='none')
ax.set_title(f'(D) PCA on Z_norm (L2-normalized)\nVar: PC1={var_z_norm[0]:.2%}, PC2={var_z_norm[1]:.2%}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/home/ehtesamul/sc_st/model/mouse_brain_res_curr_v2/embedding_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("HOW TO INTERPRET:")
print("="*70)
print("✓ PASS if:")
print("  - In (C) Z_ln: SC (purple) is well mixed with ST (orange)")
print("  - Z_ln should show best mixing since adversary optimized for this")
print("")
print("❌ FAIL if:")
print("  - Z_ln (panel C) still shows SC island")
print("="*70)


# ===================================================================
# CELL 6: DOMAIN MIXING EVALUATION (ST vs SC)
# ===================================================================
print("\n" + "="*70)
print("EVALUATION: Domain Mixing (ST vs SC)")
print("="*70)

# Use L2-normalized embeddings for kNN
Z_all = torch.cat([Z_st_norm, Z_sc_norm], dim=0)

n_st_total = Z_st_norm.shape[0]
n_sc_total = Z_sc_norm.shape[0]
n_total = n_st_total + n_sc_total

# Domain labels: 0=ST, 1=SC
domain_labels = torch.cat([
    torch.zeros(n_st_total, dtype=torch.long, device=device),
    torch.ones(n_sc_total, dtype=torch.long, device=device)
])

K = 20

# SC neighbors
sc_start = n_st_total
D_sc = torch.cdist(Z_all[sc_start:], Z_all)
for i in range(n_sc_total):
    D_sc[i, sc_start + i] = float('inf')

_, knn_sc = torch.topk(D_sc, k=K, dim=1, largest=False)
frac_same_sc = (domain_labels[knn_sc] == 1).float().mean().item()
base_rate_sc = n_sc_total / n_total

print(f"\n[SC NEIGHBORS] K={K}:")
print(f"  Same-domain (SC) fraction: {frac_same_sc:.4f}")
print(f"  Base rate (chance):        {base_rate_sc:.4f}")
print(f"  Mixing score:              {1 - (frac_same_sc - base_rate_sc):.4f}")

if frac_same_sc < base_rate_sc + 0.1:
    print("  ✓ Good mixing (SC not clustering)")
else:
    print("  ⚠️ SC may be clustering")

# ST neighbors
D_st = torch.cdist(Z_all[:sc_start], Z_all)
for i in range(n_st_total):
    D_st[i, i] = float('inf')

_, knn_st = torch.topk(D_st, k=K, dim=1, largest=False)
frac_same_st = (domain_labels[knn_st] == 0).float().mean().item()
base_rate_st = n_st_total / n_total

print(f"\n[ST NEIGHBORS] K={K}:")
print(f"  Same-domain (ST) fraction: {frac_same_st:.4f}")
print(f"  Base rate (chance):        {base_rate_st:.4f}")


# ===================================================================
# CELL 7: LINEAR PROBE (2-class)
# ===================================================================
print("\n" + "="*70)
print("LINEAR PROBE: Domain Separability")
print("="*70)

Z_np = Z_all.cpu().numpy()
y_np = domain_labels.cpu().numpy()

probe = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
probe.fit(Z_np, y_np)
pred = probe.predict(Z_np)
bal_acc = balanced_accuracy_score(y_np, pred)

print(f"  2-class balanced accuracy: {bal_acc:.4f} (chance=0.50)")

if bal_acc < 0.60:
    print("  ✓ Excellent: Domains are well-mixed (hard to separate)")
elif bal_acc < 0.70:
    print("  ✓ Good: Moderate mixing")
elif bal_acc < 0.80:
    print("  ⚠️ Domains partially separable")
else:
    print("  ❌ Domains highly separable - encoder may need more training")

# Confusion matrix
cm = confusion_matrix(y_np, pred, labels=[0, 1])
cmn = cm / cm.sum(axis=1, keepdims=True)
print("\nConfusion Matrix (row-normalized) [ST, SC]:")
print(np.round(cmn, 3))


# ===================================================================
# CELL 8: CENTROID DISTANCES
# ===================================================================
print("\n" + "="*70)
print("CENTROID ANALYSIS")
print("="*70)

centroid_st = Z_st_norm.mean(dim=0)
centroid_sc = Z_sc_norm.mean(dim=0)

dist_st_sc = (centroid_st - centroid_sc).norm().item()

print(f"  ST centroid norm:  {centroid_st.norm().item():.4f}")
print(f"  SC centroid norm:  {centroid_sc.norm().item():.4f}")
print(f"  ST-SC distance:    {dist_st_sc:.4f}")

# Within-domain variance
var_st = Z_st_norm.var(dim=0).mean().item()
var_sc = Z_sc_norm.var(dim=0).mean().item()

print(f"\n  ST within-domain variance: {var_st:.6f}")
print(f"  SC within-domain variance: {var_sc:.6f}")

# Distance-to-variance ratio
ratio = dist_st_sc / np.sqrt(var_st + var_sc)
print(f"\n  Centroid distance / sqrt(total var): {ratio:.4f}")
print(f"  (Lower is better - means domains overlap more)")


# ===================================================================
# CELL 9: H-PROBE (Context Encoder Output)
# ===================================================================
print("\n" + "="*70)
print("H-PROBE: Context Encoder Domain Mixing")
print("="*70)

# Check if context_encoder is in checkpoint
if 'context_encoder' not in checkpoint:
    print("⚠️ 'context_encoder' not found in checkpoint. Skipping H-probe.")
else:
    # Detect anchor_train mode
    base_h_dim = 128
    anchor_train_detected = infer_anchor_train_from_checkpoint(checkpoint, base_h_dim)
    print(f"Detected anchor_train={anchor_train_detected}")

    # Build model
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=32,
        c_dim=256,
        n_heads=4,
        isab_m=128,
        dist_bins=24,
        device=device,
        anchor_train=anchor_train_detected,
    )

    # Load weights
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.context_encoder.load_state_dict(checkpoint['context_encoder'])

    model.encoder.eval()
    model.context_encoder.eval()

    # Compute H for ST and SC
    @torch.no_grad()
    def get_H_for_domain(X, batch_size=256):
        """Compute context encoder output H."""
        H_list = []
        n = X.shape[0]
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size].to(device)
            zb = model.encoder(xb)
            zb_ln = F.layer_norm(zb, (zb.shape[1],))
            zb_ln = zb_ln.unsqueeze(1)  # (B, 1, D)
            mask = torch.ones(zb_ln.shape[0], 1, dtype=torch.bool, device=device)
            Hb = model.context_encoder(zb_ln, mask)
            H_list.append(Hb.squeeze(1).cpu())
        return torch.cat(H_list, dim=0)

    print("Computing H for ST...")
    H_st = get_H_for_domain(X_st_tensor)
    print("Computing H for SC...")
    H_sc = get_H_for_domain(X_sc_tensor)

    print(f"H shapes: ST={H_st.shape}, SC={H_sc.shape}")

    # H-level kNN mixing
    H_all = torch.cat([H_st, H_sc], dim=0)
    H_all_norm = F.normalize(H_all, dim=1)

    n_st_h = H_st.shape[0]
    n_sc_h = H_sc.shape[0]

    labels_h = torch.cat([
        torch.zeros(n_st_h, dtype=torch.long),
        torch.ones(n_sc_h, dtype=torch.long)
    ])

    # SC→all kNN
    sc_start_h = n_st_h
    D_h = torch.cdist(H_all_norm[sc_start_h:], H_all_norm)
    for i in range(n_sc_h):
        D_h[i, sc_start_h + i] = float('inf')

    _, knn_h = torch.topk(D_h, k=K, largest=False)
    frac_same_sc_h = (labels_h[knn_h] == 1).float().mean().item()
    base_rate_sc_h = n_sc_h / (n_st_h + n_sc_h)

    print(f"\n[H-MIXING] SC neighbors (K={K}):")
    print(f"  Same-domain fraction: {frac_same_sc_h:.4f}")
    print(f"  Base rate (chance):   {base_rate_sc_h:.4f}")

    # H-Probe (2-class)
    clf = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
    H_np = H_all_norm.numpy()
    y_h_np = labels_h.numpy()
    clf.fit(H_np, y_h_np)
    pred_h = clf.predict(H_np)
    bal_acc_h = balanced_accuracy_score(y_h_np, pred_h)

    print(f"\n[H-PROBE] 2-class balanced accuracy:")
    print(f"  Balanced Acc: {bal_acc_h:.4f} (chance=0.50)")

    # PCA visualization of H
    print("\n[H-PCA] Plotting...")
    pca_h = PCA(n_components=2, random_state=42)
    H_pca = pca_h.fit_transform(H_all_norm)

    plt.figure(figsize=(10, 8))
    for i, (label, color) in enumerate([('ST', '#FF6B35'), ('SC', '#A82FFC')]):
        if label == 'ST':
            mask = np.arange(n_st_h)
        else:
            mask = np.arange(n_st_h, n_st_h + n_sc_h)
        plt.scatter(H_pca[mask, 0], H_pca[mask, 1],
                    c=color, label=label, alpha=0.4, s=10, edgecolors='none')

    plt.title(f"H (Context Encoder Output) PCA\nPC1={pca_h.explained_variance_ratio_[0]*100:.1f}%, "
              f"PC2={pca_h.explained_variance_ratio_[1]*100:.1f}%", fontsize=14, fontweight='bold')
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/ehtesamul/sc_st/model/mouse_brain_res_curr_v2/context_encoder_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ H-probe complete.")


# ===================================================================
# CELL 10: SUMMARY
# ===================================================================
print("\n" + "="*70)
print("MOUSE BRAIN ENCODER EVALUATION SUMMARY")
print("="*70)
print(f"Data: ST={X_st.shape[0]} spots, SC={X_sc.shape[0]} cells, Genes={n_genes}")
print(f"\n[Z-SPACE METRICS]")
print(f"  ST-SC kNN mixing (SC same-domain frac): {frac_same_sc:.4f} (base={base_rate_sc:.4f})")
print(f"  ST-SC kNN mixing (ST same-domain frac): {frac_same_st:.4f} (base={base_rate_st:.4f})")
print(f"  2-class probe accuracy: {bal_acc:.4f} (chance=0.50)")
print(f"  Centroid distance: {dist_st_sc:.4f}")

if 'context_encoder' in checkpoint:
    print(f"\n[H-SPACE METRICS]")
    print(f"  H kNN mixing (SC same-domain frac): {frac_same_sc_h:.4f}")
    print(f"  H 2-class probe accuracy: {bal_acc_h:.4f}")

print("\n" + "="*70)
print("INTERPRETATION:")
if bal_acc < 0.65:
    print("✓ Good encoder - domains are well-mixed")
elif bal_acc < 0.75:
    print("⚠️ Moderate mixing - encoder may benefit from more training")
else:
    print("❌ Poor mixing - domains are separable, encoder needs work")
print("="*70)
