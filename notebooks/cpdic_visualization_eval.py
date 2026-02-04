"""
CP-DIC Encoder Visualization and Evaluation
============================================
This script runs comprehensive visualization and evaluation for all 4 training configs.

Configs:
- Config 1: ST=P2, SC=P2 (same patient)
- Config 2: ST=P2, SC=P10 (different patients - challenging)
- Config 3: ST=P2+P10, SC=P2+P10 (both patients both domains)
- Config 4: ST=P2, SC=P2+P10 (partial overlap)

Run this script and share the output with me.
"""

import sys
sys.path.insert(0, '/home/user/sc_st/model')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_VIS = 2000  # samples per source for visualization
N_EVAL = 5000  # samples for evaluation
N_EPOCHS = 500  # training epochs (shorter for this test)

# Data paths - UPDATE THESE TO YOUR PATHS
DATA_DIR = '/home/user/sc_st/data/hSCC'  # Update this path

print("="*70)
print("CP-DIC VISUALIZATION AND EVALUATION")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ===================================================================
# DATA LOADING FUNCTIONS
# ===================================================================
def load_adata(path):
    """Load and normalize AnnData object."""
    adata = sc.read_h5ad(path)
    # Check if already normalized (skip if max > 100 suggests raw counts)
    if adata.X.max() > 100:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    return adata

def extract_expr(adata, genes):
    """Extract expression matrix for given genes."""
    X = adata[:, genes].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X.astype(np.float32)

def get_common_genes(*adatas):
    """Get intersection of genes across all datasets."""
    gene_sets = [set(adata.var_names) for adata in adatas]
    common = sorted(list(set.intersection(*gene_sets)))
    return common

# ===================================================================
# LOAD ALL DATA SOURCES
# ===================================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# You need to update these paths to match your data location
# Try to load the data - if paths are wrong, print instructions
try:
    # Patient 2 data
    st_p2_1 = load_adata(f'{DATA_DIR}/stP2.h5ad')
    st_p2_2 = load_adata(f'{DATA_DIR}/stP2rep2.h5ad')
    sc_p2 = load_adata(f'{DATA_DIR}/scP2.h5ad')

    # Patient 10 data
    st_p10_1 = load_adata(f'{DATA_DIR}/stP10rep1.h5ad')
    st_p10_2 = load_adata(f'{DATA_DIR}/stP10rep2.h5ad')
    sc_p10 = load_adata(f'{DATA_DIR}/scP10.h5ad')

    print(f"Loaded P2 ST1: {st_p2_1.shape}")
    print(f"Loaded P2 ST2: {st_p2_2.shape}")
    print(f"Loaded P2 SC:  {sc_p2.shape}")
    print(f"Loaded P10 ST1: {st_p10_1.shape}")
    print(f"Loaded P10 ST2: {st_p10_2.shape}")
    print(f"Loaded P10 SC:  {sc_p10.shape}")

    DATA_LOADED = True

except FileNotFoundError as e:
    print(f"\n*** DATA NOT FOUND ***")
    print(f"Error: {e}")
    print(f"\nPlease update DATA_DIR in this script to point to your data.")
    print(f"Expected files:")
    print(f"  {DATA_DIR}/stP2.h5ad")
    print(f"  {DATA_DIR}/stP2rep2.h5ad")
    print(f"  {DATA_DIR}/scP2.h5ad")
    print(f"  {DATA_DIR}/stP10rep1.h5ad")
    print(f"  {DATA_DIR}/stP10rep2.h5ad")
    print(f"  {DATA_DIR}/scP10.h5ad")
    print(f"\n*** PLEASE SHARE THE CORRECT DATA PATH ***")
    DATA_LOADED = False

if DATA_LOADED:
    # Get common genes
    common = get_common_genes(st_p2_1, st_p2_2, sc_p2, st_p10_1, st_p10_2, sc_p10)
    print(f"\nCommon genes: {len(common)}")

    # Extract expression matrices
    X_p2_st1 = extract_expr(st_p2_1, common)
    X_p2_st2 = extract_expr(st_p2_2, common)
    X_p2_sc = extract_expr(sc_p2, common)
    X_p10_st1 = extract_expr(st_p10_1, common)
    X_p10_st2 = extract_expr(st_p10_2, common)
    X_p10_sc = extract_expr(sc_p10, common)

    print(f"\nExtracted expression matrices:")
    print(f"  P2_ST1:  {X_p2_st1.shape}")
    print(f"  P2_ST2:  {X_p2_st2.shape}")
    print(f"  P2_SC:   {X_p2_sc.shape}")
    print(f"  P10_ST1: {X_p10_st1.shape}")
    print(f"  P10_ST2: {X_p10_st2.shape}")
    print(f"  P10_SC:  {X_p10_sc.shape}")

# ===================================================================
# DEFINE CONFIGURATIONS
# ===================================================================
if DATA_LOADED:
    n_genes = len(common)

    # Config definitions
    CONFIGS = {
        'Config1': {
            'name': 'Config 1: ST=P2, SC=P2 (same patient)',
            'st_data': np.vstack([X_p2_st1, X_p2_st2]),
            'sc_data': X_p2_sc,
            'st_sources': ['P2_ST1', 'P2_ST2'],
            'sc_sources': ['P2_SC'],
            'slide_ids': np.concatenate([
                np.zeros(X_p2_st1.shape[0], dtype=int),
                np.ones(X_p2_st2.shape[0], dtype=int)
            ]),
        },
        'Config2': {
            'name': 'Config 2: ST=P2, SC=P10 (different patients)',
            'st_data': np.vstack([X_p2_st1, X_p2_st2]),
            'sc_data': X_p10_sc,
            'st_sources': ['P2_ST1', 'P2_ST2'],
            'sc_sources': ['P10_SC'],
            'slide_ids': np.concatenate([
                np.zeros(X_p2_st1.shape[0], dtype=int),
                np.ones(X_p2_st2.shape[0], dtype=int)
            ]),
        },
        'Config3': {
            'name': 'Config 3: ST=P2+P10, SC=P2+P10 (both patients)',
            'st_data': np.vstack([X_p2_st1, X_p2_st2, X_p10_st1, X_p10_st2]),
            'sc_data': np.vstack([X_p2_sc, X_p10_sc]),
            'st_sources': ['P2_ST1', 'P2_ST2', 'P10_ST1', 'P10_ST2'],
            'sc_sources': ['P2_SC', 'P10_SC'],
            'slide_ids': np.concatenate([
                np.zeros(X_p2_st1.shape[0], dtype=int),
                np.ones(X_p2_st2.shape[0], dtype=int),
                np.full(X_p10_st1.shape[0], 2, dtype=int),
                np.full(X_p10_st2.shape[0], 3, dtype=int),
            ]),
            'sc_slide_ids': np.concatenate([
                np.zeros(X_p2_sc.shape[0], dtype=int),
                np.ones(X_p10_sc.shape[0], dtype=int),
            ]),
        },
        'Config4': {
            'name': 'Config 4: ST=P2, SC=P2+P10 (partial overlap)',
            'st_data': np.vstack([X_p2_st1, X_p2_st2]),
            'sc_data': np.vstack([X_p2_sc, X_p10_sc]),
            'st_sources': ['P2_ST1', 'P2_ST2'],
            'sc_sources': ['P2_SC', 'P10_SC'],
            'slide_ids': np.concatenate([
                np.zeros(X_p2_st1.shape[0], dtype=int),
                np.ones(X_p2_st2.shape[0], dtype=int)
            ]),
            'sc_slide_ids': np.concatenate([
                np.zeros(X_p2_sc.shape[0], dtype=int),
                np.ones(X_p10_sc.shape[0], dtype=int),
            ]),
        },
    }

    print("\n" + "="*70)
    print("CONFIGURATIONS DEFINED")
    print("="*70)
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{cfg['name']}")
        print(f"  ST: {cfg['st_data'].shape} from {cfg['st_sources']}")
        print(f"  SC: {cfg['sc_data'].shape} from {cfg['sc_sources']}")

# ===================================================================
# IMPORT MODEL AND TRAINING
# ===================================================================
if DATA_LOADED:
    from core_models_et_p1 import SharedEncoder, train_encoder_cpdic

    print("\n" + "="*70)
    print("TRAINING ENCODERS (CP-DIC)")
    print("="*70)

    trained_encoders = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Training {cfg_name}: {cfg['name']}")
        print(f"{'='*70}")

        # Create encoder
        encoder = SharedEncoder(
            n_genes=n_genes,
            n_embedding=[512, 256, 128]
        )

        # Prepare data
        st_expr = torch.tensor(cfg['st_data'], dtype=torch.float32)
        sc_expr = torch.tensor(cfg['sc_data'], dtype=torch.float32)
        slide_ids = torch.tensor(cfg['slide_ids'], dtype=torch.long)

        sc_slide_ids = None
        if 'sc_slide_ids' in cfg:
            sc_slide_ids = torch.tensor(cfg['sc_slide_ids'], dtype=torch.long)

        # Train with CP-DIC
        encoder = train_encoder_cpdic(
            model=encoder,
            st_gene_expr=st_expr,
            sc_gene_expr=sc_expr,
            slide_ids=slide_ids,
            sc_slide_ids=sc_slide_ids,
            n_epochs=N_EPOCHS,
            batch_size=256,
            lr=1e-3,
            device=DEVICE,
            outf=f'/home/user/sc_st/output/cpdic_{cfg_name}',
            n_prototypes=50,
            phase1_epochs=100,
            phase2_epochs=100,
            seed=SEED,
        )

        trained_encoders[cfg_name] = encoder
        print(f"[{cfg_name}] Training complete!")

# ===================================================================
# VISUALIZATION FUNCTION
# ===================================================================
def visualize_embeddings(encoder, config_name, config, all_sources, device, n_vis=2000):
    """
    Comprehensive embedding visualization for a single config.

    Returns dict of metrics for summary.
    """
    print(f"\n{'='*70}")
    print(f"VISUALIZING: {config['name']}")
    print(f"{'='*70}")

    encoder.eval()

    # Prepare all source data
    source_data = {}
    source_data['P2_ST1'] = torch.tensor(all_sources['P2_ST1'], dtype=torch.float32)
    source_data['P2_ST2'] = torch.tensor(all_sources['P2_ST2'], dtype=torch.float32)
    source_data['P2_SC'] = torch.tensor(all_sources['P2_SC'], dtype=torch.float32)
    source_data['P10_ST1'] = torch.tensor(all_sources['P10_ST1'], dtype=torch.float32)
    source_data['P10_ST2'] = torch.tensor(all_sources['P10_ST2'], dtype=torch.float32)
    source_data['P10_SC'] = torch.tensor(all_sources['P10_SC'], dtype=torch.float32)

    # Compute embeddings for all sources
    embeddings = {}
    with torch.no_grad():
        for name, X in source_data.items():
            X_dev = X.to(device)
            Z = encoder(X_dev).cpu()
            embeddings[name] = Z

    # Subsample for visualization
    def subsample(X, Z, n):
        if X.shape[0] <= n:
            return X, Z
        idx = torch.randperm(X.shape[0])[:n]
        return X[idx], Z[idx]

    X_vis = {}
    Z_vis = {}
    for name in source_data.keys():
        X_vis[name], Z_vis[name] = subsample(source_data[name], embeddings[name], n_vis)

    # Concatenate all
    all_names = list(source_data.keys())
    X_all = torch.cat([X_vis[name] for name in all_names], dim=0)
    Z_all = torch.cat([Z_vis[name] for name in all_names], dim=0)

    # Normalizations
    Z_ln = F.layer_norm(Z_all, (Z_all.shape[1],))
    Z_norm = F.normalize(Z_all, dim=1)

    # Labels
    sizes = [X_vis[name].shape[0] for name in all_names]
    labels = np.concatenate([np.array([name] * s) for name, s in zip(all_names, sizes)])

    print(f"Visualization samples: {dict(zip(all_names, sizes))}")

    # ===================================================================
    # COLLAPSE DETECTION
    # ===================================================================
    print("\n[COLLAPSE DETECTION]")

    # Variance across embedding dimensions
    var_per_dim = Z_norm.var(dim=0)
    mean_var = var_per_dim.mean().item()
    min_var = var_per_dim.min().item()

    # Cosine similarity (collapse = all vectors same direction)
    Z_norm_np = Z_norm.numpy()
    n_samples = min(1000, Z_norm_np.shape[0])
    idx = np.random.choice(Z_norm_np.shape[0], n_samples, replace=False)
    Z_sample = Z_norm_np[idx]
    cos_sim_matrix = Z_sample @ Z_sample.T
    np.fill_diagonal(cos_sim_matrix, 0)
    mean_cos_sim = cos_sim_matrix.mean()

    collapsed = mean_var < 0.25 or mean_cos_sim > 0.95

    print(f"  Mean variance: {mean_var:.4f}")
    print(f"  Min variance:  {min_var:.4f}")
    print(f"  Mean cos sim:  {mean_cos_sim:.4f}")
    print(f"  Status: {'*** COLLAPSED ***' if collapsed else 'OK (not collapsed)'}")

    # ===================================================================
    # PCA PROJECTIONS
    # ===================================================================
    print("\n[PCA PROJECTIONS]")

    pca_x = PCA(n_components=2, random_state=42)
    X_pca = pca_x.fit_transform(X_all.numpy())
    var_x = pca_x.explained_variance_ratio_

    pca_z = PCA(n_components=2, random_state=42)
    Z_pca = pca_z.fit_transform(Z_all.numpy())
    var_z = pca_z.explained_variance_ratio_

    pca_z_norm = PCA(n_components=2, random_state=42)
    Z_norm_pca = pca_z_norm.fit_transform(Z_norm.numpy())
    var_z_norm = pca_z_norm.explained_variance_ratio_

    print(f"  X PCA variance: PC1={var_x[0]:.2%}, PC2={var_x[1]:.2%}")
    print(f"  Z PCA variance: PC1={var_z[0]:.2%}, PC2={var_z[1]:.2%}")
    print(f"  Z_norm PCA variance: PC1={var_z_norm[0]:.2%}, PC2={var_z_norm[1]:.2%}")

    # ===================================================================
    # PLOT 2x2 GRID
    # ===================================================================
    colors_map = {
        'P2_ST1':  '#e74c3c',
        'P2_ST2':  '#3498db',
        'P2_SC':   '#f39c12',
        'P10_ST1': '#9b59b6',
        'P10_ST2': '#2ecc71',
        'P10_SC':  '#1abc9c',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{config_name}: {config["name"]}', fontsize=14, fontweight='bold')

    def plot_panel(ax, data, title, var_ratio):
        for name in all_names:
            mask = labels == name
            ax.scatter(data[mask, 0], data[mask, 1],
                      c=colors_map[name], label=name, alpha=0.5, s=15, edgecolors='none')
        ax.set_title(f'{title}\nVar: PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%}', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    plot_panel(axes[0, 0], X_pca, '(A) PCA on X (expression)', var_x)
    plot_panel(axes[0, 1], Z_pca, '(B) PCA on Z (raw embeddings)', var_z)
    plot_panel(axes[1, 0], Z_norm_pca, '(C) PCA on Z_norm (L2-normalized)', var_z_norm)

    # ST vs SC binary view
    is_st = np.isin(labels, ['P2_ST1', 'P2_ST2', 'P10_ST1', 'P10_ST2'])
    axes[1, 1].scatter(Z_norm_pca[is_st, 0], Z_norm_pca[is_st, 1],
                       c='#3498db', label='ST', alpha=0.4, s=15, edgecolors='none')
    axes[1, 1].scatter(Z_norm_pca[~is_st, 0], Z_norm_pca[~is_st, 1],
                       c='#e74c3c', label='SC', alpha=0.4, s=15, edgecolors='none')
    axes[1, 1].set_title('(D) ST vs SC (binary view)', fontsize=11)
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/home/user/sc_st/output/{config_name}_pca.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: /home/user/sc_st/output/{config_name}_pca.png")

    # ===================================================================
    # EVALUATION METRICS
    # ===================================================================
    print("\n[EVALUATION METRICS]")

    # kNN domain mixing (ST vs SC)
    K = 20
    Z_norm_t = torch.tensor(Z_norm.numpy())

    n_st = is_st.sum()
    n_sc = (~is_st).sum()
    n_total = len(is_st)

    domain_labels = torch.tensor(is_st.astype(int))  # 1=ST, 0=SC

    # For SC samples, check fraction of SC neighbors
    sc_indices = np.where(~is_st)[0]
    if len(sc_indices) > 0:
        D_sc = torch.cdist(Z_norm_t[sc_indices], Z_norm_t)
        for i, idx in enumerate(sc_indices):
            D_sc[i, idx] = float('inf')

        _, knn_sc = torch.topk(D_sc, k=K, dim=1, largest=False)
        frac_same_sc = (domain_labels[knn_sc] == 0).float().mean().item()
        base_rate_sc = n_sc / n_total

        print(f"  [kNN ST-SC mixing] K={K}")
        print(f"    SC neighbors same-domain frac: {frac_same_sc:.4f}")
        print(f"    Base rate (chance):            {base_rate_sc:.4f}")
        if frac_same_sc < base_rate_sc + 0.1:
            print(f"    Status: GOOD mixing")
        else:
            print(f"    Status: SC may be clustering")
    else:
        frac_same_sc = None
        base_rate_sc = None

    # Patient mixing (P2 vs P10)
    is_p2 = np.isin(labels, ['P2_ST1', 'P2_ST2', 'P2_SC'])
    n_p2 = is_p2.sum()
    n_p10 = (~is_p2).sum()

    if n_p10 > 0:
        patient_labels = torch.tensor(is_p2.astype(int))  # 1=P2, 0=P10

        p10_indices = np.where(~is_p2)[0]
        D_p10 = torch.cdist(Z_norm_t[p10_indices], Z_norm_t)
        for i, idx in enumerate(p10_indices):
            D_p10[i, idx] = float('inf')

        _, knn_p10 = torch.topk(D_p10, k=K, dim=1, largest=False)
        frac_same_p10 = (patient_labels[knn_p10] == 0).float().mean().item()
        base_rate_p10 = n_p10 / n_total

        print(f"\n  [kNN Patient mixing] K={K}")
        print(f"    P10 neighbors same-patient frac: {frac_same_p10:.4f}")
        print(f"    Base rate (chance):              {base_rate_p10:.4f}")
        if frac_same_p10 < base_rate_p10 + 0.15:
            print(f"    Status: GOOD patient mixing")
        else:
            print(f"    Status: Patients may be separating")
    else:
        frac_same_p10 = None
        base_rate_p10 = None

    # Linear probe (6-class)
    print(f"\n  [6-class Linear Probe]")

    class_labels = np.zeros(len(labels), dtype=int)
    for i, name in enumerate(all_names):
        class_labels[labels == name] = i

    probe = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
    probe.fit(Z_norm.numpy(), class_labels)
    pred = probe.predict(Z_norm.numpy())
    bal_acc = balanced_accuracy_score(class_labels, pred)
    chance = 1.0 / len(all_names)

    print(f"    Balanced accuracy: {bal_acc:.4f} (chance={chance:.3f})")
    if bal_acc < 0.30:
        print(f"    Status: EXCELLENT - sources well-mixed")
    elif bal_acc < 0.40:
        print(f"    Status: GOOD - moderate mixing")
    else:
        print(f"    Status: Sources may be separable")

    # Centroid distances
    print(f"\n  [Centroid Distances]")
    centroids = {}
    for name in all_names:
        mask = labels == name
        centroids[name] = Z_norm[mask].mean(dim=0)

    print("    Distance matrix:")
    header = "         " + "  ".join([f"{n:>8}" for n in all_names])
    print(header)
    for n1 in all_names:
        row = f"{n1:8s}"
        for n2 in all_names:
            dist = (centroids[n1] - centroids[n2]).norm().item()
            row += f"  {dist:8.3f}"
        print(row)

    # Return metrics
    metrics = {
        'config': config_name,
        'mean_var': mean_var,
        'min_var': min_var,
        'mean_cos_sim': mean_cos_sim,
        'collapsed': collapsed,
        'knn_sc_frac': frac_same_sc,
        'knn_sc_base': base_rate_sc,
        'knn_p10_frac': frac_same_p10,
        'knn_p10_base': base_rate_p10,
        'probe_acc': bal_acc,
        'probe_chance': chance,
    }

    return metrics

# ===================================================================
# RUN VISUALIZATION FOR ALL CONFIGS
# ===================================================================
if DATA_LOADED:
    print("\n" + "="*70)
    print("RUNNING VISUALIZATION FOR ALL CONFIGS")
    print("="*70)

    # Prepare all source data
    all_sources = {
        'P2_ST1': X_p2_st1,
        'P2_ST2': X_p2_st2,
        'P2_SC': X_p2_sc,
        'P10_ST1': X_p10_st1,
        'P10_ST2': X_p10_st2,
        'P10_SC': X_p10_sc,
    }

    all_metrics = []

    for cfg_name in ['Config1', 'Config2', 'Config3', 'Config4']:
        encoder = trained_encoders[cfg_name]
        config = CONFIGS[cfg_name]

        metrics = visualize_embeddings(
            encoder=encoder,
            config_name=cfg_name,
            config=config,
            all_sources=all_sources,
            device=DEVICE,
            n_vis=N_VIS
        )
        all_metrics.append(metrics)

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\n[COLLAPSE DETECTION]")
    print(f"{'Config':<10} {'Mean Var':>10} {'Min Var':>10} {'Cos Sim':>10} {'Status':>15}")
    print("-"*60)
    for m in all_metrics:
        status = "COLLAPSED" if m['collapsed'] else "OK"
        print(f"{m['config']:<10} {m['mean_var']:>10.4f} {m['min_var']:>10.4f} {m['mean_cos_sim']:>10.4f} {status:>15}")

    print("\n[DOMAIN MIXING (kNN)]")
    print(f"{'Config':<10} {'SC Frac':>10} {'SC Base':>10} {'P10 Frac':>10} {'P10 Base':>10}")
    print("-"*60)
    for m in all_metrics:
        sc_frac = f"{m['knn_sc_frac']:.4f}" if m['knn_sc_frac'] else "N/A"
        sc_base = f"{m['knn_sc_base']:.4f}" if m['knn_sc_base'] else "N/A"
        p10_frac = f"{m['knn_p10_frac']:.4f}" if m['knn_p10_frac'] else "N/A"
        p10_base = f"{m['knn_p10_base']:.4f}" if m['knn_p10_base'] else "N/A"
        print(f"{m['config']:<10} {sc_frac:>10} {sc_base:>10} {p10_frac:>10} {p10_base:>10}")

    print("\n[LINEAR PROBE (6-class)]")
    print(f"{'Config':<10} {'Accuracy':>10} {'Chance':>10} {'Status':>15}")
    print("-"*50)
    for m in all_metrics:
        if m['probe_acc'] < 0.30:
            status = "EXCELLENT"
        elif m['probe_acc'] < 0.40:
            status = "GOOD"
        else:
            status = "SEPARABLE"
        print(f"{m['config']:<10} {m['probe_acc']:>10.4f} {m['probe_chance']:>10.3f} {status:>15}")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
COLLAPSE DETECTION:
  - Mean Variance > 0.35: Good (diverse representations)
  - Mean Variance < 0.25: COLLAPSED (all embeddings similar)
  - Cosine Similarity > 0.95: COLLAPSED (all same direction)

DOMAIN MIXING (kNN):
  - SC Frac close to SC Base: Good ST-SC mixing
  - SC Frac >> SC Base: SC clustering (ST-SC separation)
  - P10 Frac close to P10 Base: Good patient mixing
  - P10 Frac >> P10 Base: Patient batch effect

LINEAR PROBE:
  - Accuracy < 0.30: Excellent mixing (sources indistinguishable)
  - Accuracy 0.30-0.40: Good mixing
  - Accuracy > 0.40: Sources separable (batch effects)

PCA PLOTS:
  - Good: All colors mixed/overlapping in Z_norm plot
  - Bad: Clear separation by source/patient in Z_norm plot
""")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nPlease share the output above and the generated PNG files.")

else:
    print("\n*** INSTRUCTIONS ***")
    print("1. Update DATA_DIR variable at the top of this script")
    print("2. Make sure you have these files:")
    print("   - stP2.h5ad, stP2rep2.h5ad, scP2.h5ad")
    print("   - stP10rep1.h5ad, stP10rep2.h5ad, scP10.h5ad")
    print("3. Re-run this script")
    print("\nAlternatively, share the correct data path with me!")
