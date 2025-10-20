"""
hSCC Training Script - Patient 2 (P2)
Follows the exact structure from hSCC.ipynb for GEMS multi-slide training
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from threadpoolctl import threadpool_limits
threadpool_limits(limits=1, user_api='blas')

import torch
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import GEMS components
from core_models_et_p1 import SharedEncoder, STStageBPrecomputer, STSetDataset
from core_models_et_p2 import SetEncoderContext, DiffusionScoreNet, sample_sc_edm
from core_models_et_p3 import GEMSModel


# ==============================================================================
# DATA LOADING - Exact replica of hSCC.ipynb
# ==============================================================================

def load_and_process_cscc_data():
    """
    Load and process the cSCC dataset with multiple ST replicates.
    Follows exact structure from hSCC.ipynb Cell 6
    """
    print("Loading cSCC data...")
    
    # Load SC data
    scadata = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/scP2.h5ad')
    
    # Load all 3 ST datasets
    stadata1 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2.h5ad')
    stadata2 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep2.h5ad')
    stadata3 = sc.read_h5ad('/home/ehtesamul/sc_st/data/cSCC/processed/stP2rep3.h5ad')
    
    # Normalize and log transform
    for adata in [scadata, stadata1, stadata2, stadata3]:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    
    # Create rough cell types for SC data
    scadata.obs['rough_celltype'] = scadata.obs['level1_celltype'].astype(str)
    scadata.obs.loc[scadata.obs['level1_celltype']=='CLEC9A', 'rough_celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='CD1C', 'rough_celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='ASDC', 'rough_celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='PDC', 'rough_celltype'] = 'PDC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='MDSC', 'rough_celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='LC', 'rough_celltype'] = 'DC'
    scadata.obs.loc[scadata.obs['level1_celltype']=='Mac', 'rough_celltype'] = 'Myeloid cell'
    scadata.obs.loc[scadata.obs['level1_celltype']=='Tcell', 'rough_celltype'] = 'T cell'
    scadata.obs.loc[scadata.obs['level2_celltype']=='TSK', 'rough_celltype'] = 'TSK'
    scadata.obs.loc[scadata.obs['level2_celltype'].isin(['Tumor_KC_Basal', 'Tumor_KC_Diff', 'Tumor_KC_Cyc']), 'rough_celltype'] = 'NonTSK'
    
    return scadata, stadata1, stadata2, stadata3


def prepare_combined_st_for_diffusion(stadata1, stadata2, stadata3, scadata):
    """
    Combine all ST datasets for diffusion training while maintaining gene alignment.
    Follows exact structure from hSCC.ipynb
    """
    print("Preparing combined ST data for diffusion training...")
    
    # Get common genes between SC and all ST datasets
    sc_genes = set(scadata.var_names)
    st1_genes = set(stadata1.var_names)
    st2_genes = set(stadata2.var_names)
    st3_genes = set(stadata3.var_names)
    
    common_genes = sorted(list(sc_genes & st1_genes & st2_genes & st3_genes))
    print(f"Common genes across all datasets: {len(common_genes)}")
    
    # Extract aligned expression data
    sc_expr = scadata[:, common_genes].X
    st1_expr = stadata1[:, common_genes].X
    st2_expr = stadata2[:, common_genes].X
    st3_expr = stadata3[:, common_genes].X
    
    # Convert to dense if sparse
    if hasattr(sc_expr, 'toarray'):
        sc_expr = sc_expr.toarray()
    if hasattr(st1_expr, 'toarray'):
        st1_expr = st1_expr.toarray()
    if hasattr(st2_expr, 'toarray'):
        st2_expr = st2_expr.toarray()
    if hasattr(st3_expr, 'toarray'):
        st3_expr = st3_expr.toarray()
    
    # Get spatial coordinates
    st1_coords = stadata1.obsm['spatial']
    st2_coords = stadata2.obsm['spatial']
    st3_coords = stadata3.obsm['spatial']
    
    # Combine ST data
    X_st_combined = np.vstack([st1_expr, st2_expr, st3_expr])
    Y_st_combined = np.vstack([st1_coords, st2_coords, st3_coords])
    
    # Create dataset labels
    dataset_labels = np.concatenate([
        np.zeros(st1_expr.shape[0], dtype=int),
        np.ones(st2_expr.shape[0], dtype=int),
        np.full(st3_expr.shape[0], 2, dtype=int)
    ])
    
    st_coords_list = [st1_coords, st2_coords, st3_coords]
    
    return sc_expr, X_st_combined, Y_st_combined, dataset_labels, common_genes, st_coords_list


def prepare_tensors_for_gems(scadata, stadata1, stadata2, stadata3, common_genes):
    """
    Convert AnnData to PyTorch tensors with slide IDs for GEMS training.
    """
    # SC expression
    sc_expr = scadata[:, common_genes].X
    if hasattr(sc_expr, 'toarray'):
        sc_expr = sc_expr.toarray()
    
    # ST expression and coordinates (combined)
    st_expr_list = []
    st_coords_list = []
    slide_ids_list = []
    slides_dict = {}
    st_gene_expr_dict = {}
    
    stadatas = [stadata1, stadata2, stadata3]
    
    for slide_id, stadata in enumerate(stadatas):
        st_expr = stadata[:, common_genes].X
        if hasattr(st_expr, 'toarray'):
            st_expr = st_expr.toarray()
        
        st_coords = stadata.obsm['spatial']
        n_spots = st_expr.shape[0]
        
        st_expr_list.append(st_expr)
        st_coords_list.append(st_coords)
        slide_ids_list.append(np.full(n_spots, slide_id))
        
        # For Stage B
        slides_dict[slide_id] = (
            torch.tensor(st_coords, dtype=torch.float32),
            torch.tensor(st_expr, dtype=torch.float32)
        )
        st_gene_expr_dict[slide_id] = torch.tensor(st_expr, dtype=torch.float32)
    
    # Combine
    sc_expr = torch.tensor(sc_expr, dtype=torch.float32)
    st_expr_combined = torch.tensor(np.vstack(st_expr_list), dtype=torch.float32)
    st_coords_combined = torch.tensor(np.vstack(st_coords_list), dtype=torch.float32)
    slide_ids = torch.tensor(np.concatenate(slide_ids_list), dtype=torch.long)
    
    return sc_expr, st_expr_combined, st_coords_combined, slide_ids, slides_dict, st_gene_expr_dict


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train_gems_hscc(
    sc_expr,
    st_expr_combined,
    st_coords_combined,
    slide_ids,
    slides_dict,
    st_gene_expr_dict,
    n_genes,
    output_dir='gems_hscc_p2_output',
    device='cuda'
):
    """
    Complete GEMS training pipeline for hSCC P2 data.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GEMS hSCC P2 TRAINING PIPELINE")
    print("="*70)
    print(f"Device: {device}")
    print(f"SC cells: {sc_expr.shape[0]}")
    print(f"ST spots (total): {st_expr_combined.shape[0]}")
    print(f"Number of slides: {len(slides_dict)}")
    print(f"Genes: {n_genes}")
    
    # Initialize model
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=16,
        c_dim=256,
        n_heads=4,
        isab_m=64,
        device=device
    )
    
    # ========================================================================
    # STAGE A: Train Shared Encoder
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE A: Training Shared Encoder (Multi-Slide)")
    print("="*70)
    
    model.train_stageA(
        st_gene_expr=st_expr_combined,
        st_coords=st_coords_combined,
        sc_gene_expr=sc_expr,
        slide_ids=slide_ids,
        n_epochs=1000,
        batch_size=256,
        lr=0.002,
        sigma=None,
        alpha=0.8,
        mmdbatch=1.0,
        ratio_start=0.0,
        ratio_end=1.0,
        outf=output_dir
    )
    
    # ========================================================================
    # STAGE B: Precompute Geometric Targets
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE B: Precomputing Geometric Targets")
    print("="*70)
    
    slides_dict_device = {
        sid: (coords.to(device), expr.to(device))
        for sid, (coords, expr) in slides_dict.items()
    }
    
    model.train_stageB(
        slides=slides_dict_device,
        outdir=Path(output_dir) / 'stage_b_cache'
    )
    
    # ========================================================================
    # STAGE C: Train Diffusion Generator
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE C: Training Diffusion Generator")
    print("="*70)
    
    st_gene_expr_dict_device = {
        sid: expr.to(device)
        for sid, expr in st_gene_expr_dict.items()
    }
    
    model.train_stageC(
        st_gene_expr_dict=st_gene_expr_dict_device,
        n_min=128,
        n_max=512,
        num_samples=300,
        n_epochs=10,
        batch_size=8,
        lr=1e-4,
        n_timesteps=600,
        sigma_min=0.01,
        sigma_max=10.0,
        loss_weights={'alpha': 0.1, 'beta': 1.0, 'gamma': 0.5, 'eta': 0.5},
        outf=output_dir
    )
    
    # Save model
    model.save(Path(output_dir) / 'gems_model_hscc_p2.pt')
    
    return model


# ==============================================================================
# INFERENCE
# ==============================================================================

def infer_sc_coordinates(model, sc_expr, device='cuda'):
    """
    Infer SC coordinates using trained GEMS model.
    """
    print("\n" + "="*70)
    print("SC COORDINATE INFERENCE")
    print("="*70)
    
    results = model.infer_sc(
        sc_gene_expr=sc_expr.to(device),
        n_samples=1,
        n_timesteps_sample=250,
        return_coords=True
    )
    
    return results


# ==============================================================================
# VISUALIZATION - Following hSCC.ipynb style
# ==============================================================================

def visualize_results_hscc(scadata, save_dir='figures'):
    """
    Visualize inferred SC coordinates following hSCC.ipynb style.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    sc.settings.set_figure_params(format='svg')
    
    n_groups = scadata.obs["rough_celltype"].nunique()
    my_tab20 = sns.color_palette("tab20", n_colors=n_groups).as_hex()
    
    # Plot averaged coordinates
    plt.figure(figsize=(8, 6))
    sc.pl.embedding(
        scadata, 
        basis='gems_coords_avg', 
        color='rough_celltype',
        size=85, 
        title='SC GEMS Coordinates (Averaged)',
        palette=my_tab20, 
        legend_loc='right margin', 
        legend_fontsize=10,
        save='_hscc_p2_gems_avg.svg'
    )
    
    print(f"Figures saved to {save_dir}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    """
    Main execution for hSCC Patient 2 training.
    """
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'gems_hscc_p2_output'
    
    print("\n" + "="*70)
    print("hSCC P2 GEMS TRAINING - START")
    print("="*70)
    
    # ========================================================================
    # Load and process data
    # ========================================================================
    print("\nStep 1: Loading data...")
    scadata, stadata1, stadata2, stadata3 = load_and_process_cscc_data()
    
    print(f"\nData loaded successfully:")
    print(f"  SC cells: {scadata.shape[0]}")
    print(f"  ST slide 1: {stadata1.shape[0]} spots")
    print(f"  ST slide 2: {stadata2.shape[0]} spots")
    print(f"  ST slide 3: {stadata3.shape[0]} spots")
    
    # ========================================================================
    # Prepare data for GEMS
    # ========================================================================
    print("\nStep 2: Preparing data for GEMS...")
    sc_expr, X_st_combined, Y_st_combined, dataset_labels, common_genes, st_coords_list = \
        prepare_combined_st_for_diffusion(stadata1, stadata2, stadata3, scadata)
    
    sc_expr_tensor, st_expr_combined, st_coords_combined, slide_ids, slides_dict, st_gene_expr_dict = \
        prepare_tensors_for_gems(scadata, stadata1, stadata2, stadata3, common_genes)
    
    n_genes = len(common_genes)
    
    print(f"\nData preparation complete:")
    print(f"  SC expression: {sc_expr_tensor.shape}")
    print(f"  ST expression (combined): {st_expr_combined.shape}")
    print(f"  ST coordinates (combined): {st_coords_combined.shape}")
    print(f"  Slide IDs: {slide_ids.shape}")
    print(f"  Common genes: {n_genes}")
    
    # ========================================================================
    # Train GEMS
    # ========================================================================
    print("\nStep 3: Training GEMS model...")
    model = train_gems_hscc(
        sc_expr=sc_expr_tensor,
        st_expr_combined=st_expr_combined,
        st_coords_combined=st_coords_combined,
        slide_ids=slide_ids,
        slides_dict=slides_dict,
        st_gene_expr_dict=st_gene_expr_dict,
        n_genes=n_genes,
        output_dir=output_dir,
        device=device
    )
    
    # ========================================================================
    # Inference
    # ========================================================================
    print("\nStep 4: Running inference on SC data...")

    # Clear cache before inference
    if device == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Use batched inference
    results = model.infer_sc_batched(
        sc_gene_expr=sc_expr_tensor.to(device),
        n_timesteps_sample=250,
        return_coords=True,
        batch_size=512
    )

print(f"\nInference complete:")
print(f"  D_edm shape: {results['D_edm'].shape}")
if 'coords_canon' in results:
    print(f"  Coordinates shape: {results['coords_canon'].shape}")
    
    # Add coordinates to scadata
    coords_avg = results['coords_canon'].cpu().numpy()
    scadata.obsm['gems_coords_avg'] = coords_avg
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\nStep 5: Visualizing results...")
    visualize_results_hscc(scadata, save_dir='figures')
    
    # Save scadata with GEMS coordinates
    scadata.write_h5ad(Path(output_dir) / 'scadata_with_gems_coords.h5ad')
    
    print("\n" + "="*70)
    print("hSCC P2 GEMS TRAINING - COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Model: {output_dir}/gems_model_hscc_p2.pt")
    print(f"  Cached targets: {output_dir}/stage_b_cache/")
    print(f"  SC data with coords: {output_dir}/scadata_with_gems_coords.h5ad")
    print(f"  Figures: figures/")
    print("\nCoordinates stored in scadata.obsm['gems_coords_avg']")