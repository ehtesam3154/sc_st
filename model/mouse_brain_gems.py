import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from threadpoolctl import threadpool_limits
threadpool_limits(limits=1, user_api='blas')

import scanpy as sc
import pandas as pd
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from core_models_et_p1 import SharedEncoder, STStageBPrecomputer, STSetDataset
from core_models_et_p2 import SetEncoderContext, DiffusionScoreNet
from core_models_et_p3 import GEMSModel

print("Loading SC data...")
scdata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_sc_counts.csv', index_col=0)
scdata = scdata.T
scmetadata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/metadata.csv', index_col=0)
print("Loading ST data...")
stdata = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_counts.csv', index_col=0)

stdata = stdata.T
spcoor = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_metadata.csv', index_col=0)
stgtcelltype = pd.read_csv('/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_celltype.csv', index_col=0)

print(f"SC data shape: {scdata.shape}")
print(f"ST data shape: {stdata.shape}")
print(f"ST coords shape: {spcoor.shape}")
print(f"ST celltype shape: {stgtcelltype.shape}")

scadata = sc.AnnData(scdata, obs=scmetadata)
sc.pp.normalize_total(scadata)
sc.pp.log1p(scadata)

scadata.obsm['spatial'] = scmetadata[['x_global', 'y_global']].values

print(f"SC AnnData: {scadata}")

stadata = sc.AnnData(stdata)
sc.pp.normalize_total(stadata)
sc.pp.log1p(stadata)

stadata.obsm['spatial'] = spcoor[['coord_x', 'coord_y']].values

cell_type_columns = stgtcelltype.columns
dominant_celltypes = []

for i in range(stgtcelltype.shape[0]):
    cell_types_present = [col for col, val in zip(cell_type_columns, stgtcelltype.iloc[i]) if val > 0]
    dominant_celltype = cell_types_present[0] if cell_types_present else 'Unknown'
    dominant_celltypes.append(dominant_celltype)

stadata.obs['celltype'] = dominant_celltypes

print(f"ST AnnData: {stadata}")
print(f"ST cell types: {stadata.obs['celltype'].value_counts()}")

def train_gems_mousebrain(scadata, stadata, output_dir='gems_mousebrain_output', device='cuda'):
    """
    Train GEMS model for mouse brain data with mixed ST/SC training.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GEMS MOUSE BRAIN TRAINING (MIXED ST/SC)")
    print("="*70)
    print(f"Device: {device}")
    
    sc_genes = set(scadata.var_names)
    st_genes = set(stadata.var_names)
    common_genes = sorted(list(sc_genes & st_genes))
    n_genes = len(common_genes)
    
    print(f"Common genes: {n_genes}")
    
    sc_expr = scadata[:, common_genes].X
    st_expr = stadata[:, common_genes].X
    
    if hasattr(sc_expr, 'toarray'):
        sc_expr = sc_expr.toarray()
    if hasattr(st_expr, 'toarray'):
        st_expr = st_expr.toarray()
    
    st_coords = stadata.obsm['spatial']
    
    print(f"SC expression shape: {sc_expr.shape}")
    print(f"ST expression shape: {st_expr.shape}")
    print(f"ST coords shape: {st_coords.shape}")
    
    sc_expr_tensor = torch.tensor(sc_expr, dtype=torch.float32)
    st_expr_tensor = torch.tensor(st_expr, dtype=torch.float32)
    st_coords_tensor = torch.tensor(st_coords, dtype=torch.float32)
    
    slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long)
    
    slides_dict = {
        0: (st_coords_tensor, st_expr_tensor)
    }
    
    st_gene_expr_dict = {
        0: st_expr_tensor
    }
    
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=16,
        c_dim=256,
        n_heads=4,
        isab_m=64,
        device=device
    )
    
    print("\n" + "="*70)
    print("STAGE A: Training Shared Encoder")
    print("="*70)
    
    model.train_stageA(
        st_gene_expr=st_expr_tensor,
        st_coords=st_coords_tensor,
        sc_gene_expr=sc_expr_tensor,
        slide_ids=slide_ids,
        n_epochs=100,
        batch_size=256,
        lr=0.0001,
        sigma=None,
        alpha=0.8,
        ratio_start=0.0,
        ratio_end=1.0,
        mmdbatch=1.0,
        outf=output_dir
    )
    
    print("\n" + "="*70)
    print("STAGE B: Precomputing Geometric Targets")
    print("="*70)
    
    slides_dict_device = {
        sid: (coords.to(device), expr.to(device))
        for sid, (coords, expr) in slides_dict.items()
    }
    
    model.train_stageB(
        slides=slides_dict_device,
        outdir=str(Path(output_dir) / 'stage_b_cache')
    )
    
    print("\n" + "="*70)
    print("STAGE C: Training Diffusion Generator (Mixed ST/SC)")
    print("="*70)
    
    st_gene_expr_dict_device = {
        sid: expr.to(device)
        for sid, expr in st_gene_expr_dict.items()
    }
    
    model.train_stageC(
        st_gene_expr_dict=st_gene_expr_dict_device,
        sc_gene_expr=sc_expr_tensor,
        n_min=64,
        n_max=192,
        num_st_samples=6000,
        num_sc_samples=9000,
        n_epochs=600,
        batch_size=4,
        lr=1e-4,
        n_timesteps=600,
        sigma_min=0.01,
        sigma_max=5.0,
        outf=output_dir
    )
    
    model.save(str(Path(output_dir) / 'gems_model_mousebrain.pt'))
    
    return model, common_genes

print("Starting GEMS training with mixed ST/SC regimen...")
model, common_genes = train_gems_mousebrain(scadata, stadata, device='cuda')
print("\nTraining complete! Model saved.")

print("\n" + "="*70)
print("SC COORDINATE INFERENCE (ANCHOR-CONDITIONED)")
print("="*70)

sc_expr = scadata[:, common_genes].X
if hasattr(sc_expr, 'toarray'):
    sc_expr = sc_expr.toarray()
sc_expr_tensor = torch.tensor(sc_expr, dtype=torch.float32)

print(f"SC data shape: {sc_expr_tensor.shape}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
import gc
gc.collect()

results = model.infer_sc_anchored(
    sc_gene_expr=sc_expr_tensor,
    n_timesteps_sample=200,
    return_coords=True,
    anchor_size=350,
    batch_size=384,
    eta=0.0
)

print(f"\nInference complete:")
print(f"  D_edm shape: {results['D_edm'].shape}")
if 'coords_canon' in results:
    print(f"  Coordinates shape: {results['coords_canon'].shape}")

coords_canon = results['coords_canon'].numpy()
scadata.obsm['gems_coords'] = coords_canon

print(f"\nGenerated coordinates added to scadata.obsm['gems_coords']")
print(f"Shape: {scadata.obsm['gems_coords'].shape}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

if 'cell_type' in scadata.obs.columns:
    cell_types = scadata.obs['cell_type']
    unique_types = cell_types.unique()
    
    for ct in unique_types:
        mask = cell_types == ct
        axes[0].scatter(
            coords_canon[mask, 0], 
            coords_canon[mask, 1],
            s=1, 
            alpha=0.6, 
            label=ct
        )
    
    axes[0].set_title('GEMS Coordinates (by cell type)', fontsize=14)
    axes[0].set_xlabel('GEMS Dim 1')
    axes[0].set_ylabel('GEMS Dim 2')
    axes[0].legend(markerscale=5, fontsize=8, loc='best')
else:
    axes[0].scatter(coords_canon[:, 0], coords_canon[:, 1], s=1, alpha=0.6)
    axes[0].set_title('GEMS Coordinates', fontsize=14)
    axes[0].set_xlabel('GEMS Dim 1')
    axes[0].set_ylabel('GEMS Dim 2')

axes[0].axis('equal')

D_edm = results['D_edm'].numpy()
upper_tri_idx = np.triu_indices_from(D_edm, k=1)
distances = D_edm[upper_tri_idx]

axes[1].hist(distances, bins=100, alpha=0.7, edgecolor='black')
axes[1].set_title('Distance Distribution (EDM)', fontsize=14)
axes[1].set_xlabel('Distance')
axes[1].set_ylabel('Count')
axes[1].axvline(distances.mean(), color='r', linestyle='--', label=f'Mean: {distances.mean():.2f}')
axes[1].axvline(np.median(distances), color='g', linestyle='--', label=f'Median: {np.median(distances):.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(str(Path(output_dir) / 'gems_inference_results.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to {output_dir}/gems_inference_results.png")

results_save = {
    'coords': coords_canon,
    'D_edm': D_edm,
    'common_genes': common_genes
}
torch.save(results_save, str(Path(output_dir) / 'gems_inference_results.pt'))
print(f"Results saved to {output_dir}/gems_inference_results.pt")

D = results['D_edm'].cpu().numpy()
n = D.shape[0]

H = np.eye(n) - np.ones((n, n)) / n

D_squared = D ** 2
G = -0.5 * H @ D_squared @ H

eigenvalues = np.linalg.eigvalsh(G)
eigenvalues = np.sort(eigenvalues)[::-1]
top_20 = eigenvalues[:20]

print("Top 20 eigenvalues:")
print(top_20)
print(f"\nTop 10 eigenvalues: {top_20[:10]}")
print(f"Ratio λ1/λ10: {top_20[0] / top_20[9]:.2f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, 21), top_20, 'o-')
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue')
plt.title('Top 20 Eigenvalues')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), top_20[:10], 'o-', color='red')
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue')
plt.title('Top 10 Eigenvalues (Blob Check)')
plt.grid(True)

plt.tight_layout()
plt.show()

if top_20[0] / top_20[9] < 3.0:
    print("\n⚠️ BLOB PATTERN: Top 10 eigenvalues are similar → coordinates may collapse to a blob")
else:
    print("\n✓ Good eigenvalue spread → coordinates should be well-separated")

plt.rcParams['figure.figsize'] = (6,5)

if 'celltype' not in scadata.obs.columns:
    if 'celltype_mapped_refined' in scmetadata.columns:
        scadata.obs['celltype'] = scmetadata['celltype_mapped_refined'].values
    else:
        print("No celltype column found. Creating dummy column.")
        scadata.obs['celltype'] = 'Unknown'

n_celltypes = scadata.obs['celltype'].nunique()
my_tab20 = sns.color_palette("tab20", n_colors=n_celltypes).as_hex()

if 'x_global' in scadata.obs.columns:
    fig = plt.figure(figsize=(6, 3))
    sc.pl.embedding(
        scadata, 
        basis='spatial', 
        color='celltype',
        title='Original SC Coordinates',
        size=60,
        palette=my_tab20,
        legend_loc='right margin',
        show=True
    )

fig = plt.figure(figsize=(6, 3))
sc.pl.embedding(
    scadata,
    basis='gems_coords_avg',
    color='celltype',
    title='Generated GEMS Coordinates - Mouse Brain',
    size=60,
    palette=my_tab20,
    legend_loc='right margin',
    show=True
)