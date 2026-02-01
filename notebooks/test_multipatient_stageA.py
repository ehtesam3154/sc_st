# ===================================================================
# TEST: Multi-Patient Stage A Training with SC Slide Balancing
# ===================================================================
#
# Setup for Model 1 (patient invariance test):
# - ST domain (diffusion training): P2_ST1, P2_ST2
# - SC domain (inference targets): P2_ST3, P2_SC, P10_ST3, P10_SC
#
# This tests the new sc_slide_ids parameter for per-SC-slide balancing.
# ===================================================================

import torch
import torch.nn.functional as F
import scanpy as sc
import numpy as np
import sys
sys.path.insert(0, '/home/ehtesamul/sc_st/model')

from core_models_et_p1 import SharedEncoder, train_encoder
from ssl_utils import set_seed
import utils_et as uet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================================================================
# SET SEED FOR REPRODUCIBILITY
# ===================================================================
# Must be set BEFORE creating the encoder to seed weight initialization
SEED = 42
set_seed(SEED)

print("="*70)
print("MULTI-PATIENT STAGE A: VICReg + Domain Adversary")
print("ST = P2_ST1, P2_ST2 (training)")
print("SC = P2_ST3, P2_SC, P10_ST3, P10_SC (inference)")
print("="*70)


# ===================================================================
# 1) LOAD ALL DATA
# ===================================================================
print("\n--- Loading HSCC data (2 patients) ---")

# Patient 2 data
stP2_1 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2.h5ad')       # ST1
stP2_2 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep2.h5ad')   # ST2
stP2_3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep3.h5ad')   # ST3 (as SC)
scP2 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP2.h5ad')         # SC

# Patient 10 data
stP10_3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP10rep3.h5ad') # ST3 (as SC)
scP10 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP10.h5ad')       # SC

# Normalize all
all_data = [stP2_1, stP2_2, stP2_3, scP2, stP10_3, scP10]
for adata in all_data:
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

# Get common genes across ALL datasets
common = sorted(list(
    set(stP2_1.var_names) & set(stP2_2.var_names) & set(stP2_3.var_names) &
    set(scP2.var_names) & set(stP10_3.var_names) & set(scP10.var_names)
))
n_genes = len(common)
print(f"✓ Common genes across all datasets: {n_genes}")


# ===================================================================
# 2) PREPARE ST DOMAIN (training slides: P2_ST1, P2_ST2)
# ===================================================================
print("\n--- Preparing ST domain (training) ---")

def extract_expr(adata, genes):
    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X

X_st1 = extract_expr(stP2_1, common)  # P2_ST1
X_st2 = extract_expr(stP2_2, common)  # P2_ST2

st_expr = torch.tensor(np.vstack([X_st1, X_st2]), dtype=torch.float32, device=device)

# ST coordinates (required for function signature, not used by VICReg)
st_coords1 = stP2_1.obsm['spatial']
st_coords2 = stP2_2.obsm['spatial']
st_coords_raw = torch.tensor(np.vstack([st_coords1, st_coords2]), dtype=torch.float32, device=device)

# ST slide IDs: 0 for P2_ST1, 1 for P2_ST2
slide_ids = torch.tensor(
    np.concatenate([
        np.zeros(X_st1.shape[0], dtype=int),   # slide 0: P2_ST1
        np.ones(X_st2.shape[0], dtype=int),    # slide 1: P2_ST2
    ]),
    dtype=torch.long, device=device
)

# Canonicalize coordinates
st_coords, st_mu, st_scale = uet.canonicalize_st_coords_per_slide(st_coords_raw, slide_ids)

print(f"✓ ST expr: {st_expr.shape}")
print(f"  - P2_ST1: {X_st1.shape[0]} spots (slide 0)")
print(f"  - P2_ST2: {X_st2.shape[0]} spots (slide 1)")


# ===================================================================
# 3) PREPARE SC DOMAIN (inference targets: P2_ST3, P2_SC, P10_ST3, P10_SC)
# ===================================================================
print("\n--- Preparing SC domain (inference targets) ---")

X_p2_st3 = extract_expr(stP2_3, common)   # P2_ST3 (treated as SC)
X_p2_sc = extract_expr(scP2, common)       # P2_SC
X_p10_st3 = extract_expr(stP10_3, common)  # P10_ST3 (treated as SC)
X_p10_sc = extract_expr(scP10, common)     # P10_SC

# Concatenate all SC sources
sc_expr_list = [X_p2_st3, X_p2_sc, X_p10_st3, X_p10_sc]
sc_expr = torch.tensor(np.vstack(sc_expr_list), dtype=torch.float32, device=device)

# SC slide IDs for balanced sampling
sc_slide_ids = torch.tensor(
    np.concatenate([
        np.full(X_p2_st3.shape[0], 0, dtype=int),   # SC slide 0: P2_ST3
        np.full(X_p2_sc.shape[0], 1, dtype=int),    # SC slide 1: P2_SC
        np.full(X_p10_st3.shape[0], 2, dtype=int),  # SC slide 2: P10_ST3
        np.full(X_p10_sc.shape[0], 3, dtype=int),   # SC slide 3: P10_SC
    ]),
    dtype=torch.long, device=device
)

# SC patient IDs for patient-level CORAL (cross-patient alignment within SC)
# Patient 0 = P2, Patient 1 = P10
sc_patient_ids = torch.tensor(
    np.concatenate([
        np.full(X_p2_st3.shape[0], 0, dtype=int),   # Patient 0: P2_ST3
        np.full(X_p2_sc.shape[0], 0, dtype=int),    # Patient 0: P2_SC
        np.full(X_p10_st3.shape[0], 1, dtype=int),  # Patient 1: P10_ST3
        np.full(X_p10_sc.shape[0], 1, dtype=int),   # Patient 1: P10_SC
    ]),
    dtype=torch.long, device=device
)

print(f"✓ SC expr: {sc_expr.shape}")
print(f"  - P2_ST3:  {X_p2_st3.shape[0]} cells (SC slide 0)")
print(f"  - P2_SC:   {X_p2_sc.shape[0]} cells (SC slide 1)")
print(f"  - P10_ST3: {X_p10_st3.shape[0]} cells (SC slide 2)")
print(f"  - P10_SC:  {X_p10_sc.shape[0]} cells (SC slide 3)")
print(f"✓ SC slide IDs: {sc_slide_ids.shape} (slides: {torch.unique(sc_slide_ids).tolist()})")


# ===================================================================
# 4) SUMMARY
# ===================================================================
print("\n" + "="*70)
print("DATA SUMMARY")
print("="*70)
print(f"ST domain (training):    {st_expr.shape[0]} spots from 2 slides")
print(f"SC domain (inference):   {sc_expr.shape[0]} cells from 4 slides")
print(f"Total cells for Stage A: {st_expr.shape[0] + sc_expr.shape[0]}")
print(f"Genes:                   {n_genes}")
print("="*70)


# ===================================================================
# 5) CREATE AND TRAIN ENCODER WITH VICREG + DOMAIN ADVERSARY
# ===================================================================
print("\n" + "="*70)
print("TRAINING STAGE A ENCODER (VICReg + Domain Adversary)")
print("With per-SC-slide balancing enabled")
print("="*70)

encoder = SharedEncoder(
    n_genes=n_genes,
    n_embedding=[512, 256, 128],
    dropout=0.1
)

import os
outdir = '/home/ehtesamul/sc_st/model/gems_multipatient_test'
os.makedirs(outdir, exist_ok=True)

encoder, projector, discriminator, hist = train_encoder(
    model=encoder,
    st_gene_expr=st_expr,
    st_coords=st_coords,
    sc_gene_expr=sc_expr,
    slide_ids=slide_ids,
    sc_slide_ids=sc_slide_ids,  # Enable per-SC-slide balancing
    sc_patient_ids=sc_patient_ids,  # Enable patient-level CORAL for P2↔P10 alignment
    n_epochs=500,  # Shorter for testing
    batch_size=256,
    lr=1e-3,
    device=device,
    outf=outdir,
    # ========== VICReg Mode ==========
    stageA_obj='vicreg_adv',
    vicreg_lambda_inv=25.0,
    vicreg_lambda_var=25.0,
    vicreg_lambda_cov=1.0,
    vicreg_gamma=1.0,
    vicreg_eps=1e-4,
    vicreg_project_dim=256,
    vicreg_use_projector=False,
    vicreg_float32_stats=True,
    vicreg_ddp_gather=False,
    # Expression augmentations
    aug_gene_dropout=0.2,
    aug_gauss_std=0.01,
    aug_scale_jitter=0.1,
    # Domain adversary
    adv_slide_weight=50.0,
    patient_coral_weight=10.0,  # Patient alignment within SC domain
    adv_warmup_epochs=50,
    adv_ramp_epochs=200,
    grl_alpha_max=1.0,
    disc_hidden=256,
    disc_dropout=0.1,
    # Balanced domain sampling
    stageA_balanced_slides=True,
    # Representation mode
    adv_representation_mode='clean',
    adv_use_layernorm=True,
    adv_log_diagnostics=True,
    adv_log_grad_norms=False,
    # Local alignment (optional)
    use_local_align=True,
    return_aux=True,
    local_align_bidirectional=True,
    local_align_weight=4.0,
    local_align_tau_z=0.07,
    # Reproducibility
    seed=SEED,
)

print("\n✓ VICReg Stage A training complete!")

# Save encoder
torch.save(encoder.state_dict(), f'{outdir}/encoder_multipatient.pt')
print(f"✓ Encoder saved to: {outdir}/encoder_multipatient.pt")


# ===================================================================
# 6) EVALUATION: Domain Mixing (ST vs SC, and within SC)
# ===================================================================
print("\n" + "="*70)
print("EVALUATION: Domain Mixing")
print("="*70)

N_MAX = 2000

# Reset seed for reproducible evaluation
set_seed(SEED)

def subsample(X, n_max, device):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)
    if X.shape[0] <= n_max:
        return X
    idx = torch.randperm(X.shape[0], device=device)[:n_max]
    return X[idx]

# Subsample all 6 sources
X_p2st1_sub = subsample(X_st1, N_MAX, device)
X_p2st2_sub = subsample(X_st2, N_MAX, device)
X_p2st3_sub = subsample(X_p2_st3, N_MAX, device)
X_p2sc_sub = subsample(X_p2_sc, N_MAX, device)
X_p10st3_sub = subsample(X_p10_st3, N_MAX, device)
X_p10sc_sub = subsample(X_p10_sc, N_MAX, device)

# Compute embeddings
encoder.eval()
with torch.no_grad():
    Z_p2st1 = encoder(X_p2st1_sub)
    Z_p2st2 = encoder(X_p2st2_sub)
    Z_p2st3 = encoder(X_p2st3_sub)
    Z_p2sc = encoder(X_p2sc_sub)
    Z_p10st3 = encoder(X_p10st3_sub)
    Z_p10sc = encoder(X_p10sc_sub)

print(f"Embeddings computed:")
print(f"  P2_ST1:  {Z_p2st1.shape}")
print(f"  P2_ST2:  {Z_p2st2.shape}")
print(f"  P2_ST3:  {Z_p2st3.shape}")
print(f"  P2_SC:   {Z_p2sc.shape}")
print(f"  P10_ST3: {Z_p10st3.shape}")
print(f"  P10_SC:  {Z_p10sc.shape}")


# ===================================================================
# TEST 1: ST vs SC kNN mixing
# ===================================================================
print("\n[ST-vs-SC MIXING] kNN domain distribution:")

Z_st_all = torch.cat([Z_p2st1, Z_p2st2], dim=0)
Z_sc_all = torch.cat([Z_p2st3, Z_p2sc, Z_p10st3, Z_p10sc], dim=0)
Z_all = torch.cat([Z_st_all, Z_sc_all], dim=0)

n_st = Z_st_all.shape[0]
n_sc = Z_sc_all.shape[0]
n_total = n_st + n_sc

# Domain labels: 0=ST, 1=SC
domain_labels = torch.cat([
    torch.zeros(n_st, dtype=torch.long, device=device),
    torch.ones(n_sc, dtype=torch.long, device=device)
])

Z_all_norm = F.normalize(Z_all, dim=1)

K = 20
sc_start = n_st

# SC neighbors
D_sc = torch.cdist(Z_all_norm[sc_start:], Z_all_norm)
for i in range(n_sc):
    D_sc[i, sc_start + i] = float('inf')

_, knn_sc = torch.topk(D_sc, k=K, dim=1, largest=False)
frac_same_sc = (domain_labels[knn_sc] == 1).float().mean().item()
base_rate_sc = n_sc / n_total

print(f"  SC neighbors (K={K}):")
print(f"    Same-domain (SC) fraction: {frac_same_sc:.4f}")
print(f"    Base rate (chance):        {base_rate_sc:.4f}")

if frac_same_sc < base_rate_sc + 0.1:
    print("    ✓ Good mixing (SC not clustering)")
else:
    print("    ⚠️ SC may be clustering")


# ===================================================================
# TEST 2: Patient-level mixing (P2 vs P10)
# ===================================================================
print("\n[PATIENT MIXING] P2 vs P10 kNN distribution:")

# Label by patient: P2=0, P10=1
patient_labels = torch.cat([
    torch.zeros(Z_p2st1.shape[0], dtype=torch.long, device=device),  # P2_ST1
    torch.zeros(Z_p2st2.shape[0], dtype=torch.long, device=device),  # P2_ST2
    torch.zeros(Z_p2st3.shape[0], dtype=torch.long, device=device),  # P2_ST3
    torch.zeros(Z_p2sc.shape[0], dtype=torch.long, device=device),   # P2_SC
    torch.ones(Z_p10st3.shape[0], dtype=torch.long, device=device),  # P10_ST3
    torch.ones(Z_p10sc.shape[0], dtype=torch.long, device=device),   # P10_SC
])

n_p2 = Z_p2st1.shape[0] + Z_p2st2.shape[0] + Z_p2st3.shape[0] + Z_p2sc.shape[0]
n_p10 = Z_p10st3.shape[0] + Z_p10sc.shape[0]
base_rate_p10 = n_p10 / n_total

# P10 kNN neighbors
p10_start = n_p2
D_p10 = torch.cdist(Z_all_norm[p10_start:], Z_all_norm)
for i in range(n_p10):
    D_p10[i, p10_start + i] = float('inf')

_, knn_p10 = torch.topk(D_p10, k=K, dim=1, largest=False)
frac_same_p10 = (patient_labels[knn_p10] == 1).float().mean().item()

print(f"  P10 neighbors (K={K}):")
print(f"    Same-patient (P10) fraction: {frac_same_p10:.4f}")
print(f"    Base rate (chance):          {base_rate_p10:.4f}")

if frac_same_p10 < base_rate_p10 + 0.15:
    print("    ✓ Good patient mixing")
else:
    print("    ⚠️ Patients may be separating")


# ===================================================================
# TEST 3: Linear Probe (6-class)
# ===================================================================
print("\n[6-CLASS PROBE] Linear separability test:")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# 6-class labels
class_labels = torch.cat([
    torch.full((Z_p2st1.shape[0],), 0, dtype=torch.long, device=device),   # P2_ST1
    torch.full((Z_p2st2.shape[0],), 1, dtype=torch.long, device=device),   # P2_ST2
    torch.full((Z_p2st3.shape[0],), 2, dtype=torch.long, device=device),   # P2_ST3
    torch.full((Z_p2sc.shape[0],), 3, dtype=torch.long, device=device),    # P2_SC
    torch.full((Z_p10st3.shape[0],), 4, dtype=torch.long, device=device),  # P10_ST3
    torch.full((Z_p10sc.shape[0],), 5, dtype=torch.long, device=device),   # P10_SC
])

Z_np = Z_all_norm.cpu().numpy()
y_np = class_labels.cpu().numpy()

probe = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
probe.fit(Z_np, y_np)
pred = probe.predict(Z_np)
bal_acc = balanced_accuracy_score(y_np, pred)
chance = 1.0 / 6.0

print(f"  Balanced accuracy: {bal_acc:.4f} (chance={chance:.3f})")

if bal_acc < 0.30:
    print("  ✓ Excellent: Sources are well-mixed")
elif bal_acc < 0.40:
    print("  ✓ Good: Moderate mixing")
else:
    print("  ⚠️ Sources may be separable")


# ===================================================================
# TEST 4: Centroid distances
# ===================================================================
print("\n[CENTROID ANALYSIS] Cross-source distances:")

centroids = {
    'P2_ST1': Z_p2st1.mean(dim=0),
    'P2_ST2': Z_p2st2.mean(dim=0),
    'P2_ST3': Z_p2st3.mean(dim=0),
    'P2_SC': Z_p2sc.mean(dim=0),
    'P10_ST3': Z_p10st3.mean(dim=0),
    'P10_SC': Z_p10sc.mean(dim=0),
}

names = list(centroids.keys())
print("\nCentroid distance matrix:")
print("         ", "  ".join([f"{n:>7}" for n in names]))

for i, n1 in enumerate(names):
    row = f"{n1:8s}"
    for j, n2 in enumerate(names):
        dist = (centroids[n1] - centroids[n2]).norm().item()
        row += f"  {dist:7.3f}"
    print(row)


# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*70)
print("MULTI-PATIENT STAGE A TEST COMPLETE")
print("="*70)
print(f"✓ Encoder trained with SC slide balancing (4 SC slides)")
print(f"✓ ST-SC mixing:     SC same-domain frac = {frac_same_sc:.4f} (base = {base_rate_sc:.4f})")
print(f"✓ Patient mixing:   P10 same-patient frac = {frac_same_p10:.4f} (base = {base_rate_p10:.4f})")
print(f"✓ 6-class probe:    balanced acc = {bal_acc:.4f} (chance = {chance:.3f})")
print("="*70)
