# run_mouse_brain_lightning.py
import os
import torch
from lightning.fabric import Fabric
import scanpy as sc
from core_models_et_p3 import GEMSModel
# from mouse_brain import train_gems_mousebrain  # optional, if you want its helpers

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='GEMS Training with Lightning Fabric')
    
    # Training config
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--stageA_epochs', type=int, default=1000)
    parser.add_argument('--stageC_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='gems_mousebrain_output')
    parser.add_argument('--num_st_samples', type=int, default=4000)  # ADD THIS
    parser.add_argument('--num_sc_samples', type=int, default=9000)  # ADD THIS
    
    # Geometry-aware diffusion parameters
    parser.add_argument('--use_canonicalize', action='store_true', default=True)
    parser.add_argument('--use_dist_bias', action='store_true', default=True)
    parser.add_argument('--dist_bins', type=int, default=24)
    parser.add_argument('--dist_head_shared', action='store_true', default=True)
    parser.add_argument('--use_angle_features', action='store_true', default=True)
    parser.add_argument('--angle_bins', type=int, default=8)
    parser.add_argument('--knn_k', type=int, default=12)
    parser.add_argument('--self_conditioning', action='store_true', default=True)
    parser.add_argument('--sc_feat_mode', type=str, default='concat', choices=['concat', 'mlp'])
    parser.add_argument('--landmarks_L', type=int, default=32)
    
    return parser.parse_args()

def load_mouse_data():
    # Replace these with your actual file paths
    sc_counts = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_sc_counts.csv'
    sc_meta   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/metadata.csv'
    st_counts = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_counts.csv'
    st_meta   = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_metadata.csv'
    st_ct     = '/home/ehtesamul/sc_st/data/mousedata_2020/E1z2/simu_st_celltype.csv'

    import pandas as pd
    import anndata as ad
    
    # Load data
    print("Loading SC data...")
    scdata = pd.read_csv(sc_counts, index_col=0).T
    scmetadata = pd.read_csv(sc_meta, index_col=0)
    
    print("Loading ST data...")
    stdata = pd.read_csv(st_counts, index_col=0).T
    spcoor = pd.read_csv(st_meta, index_col=0)
    stct = pd.read_csv(st_ct, index_col=0)
    
    print(f"SC data shape: {scdata.shape}")
    print(f"ST data shape: {stdata.shape}")
    print(f"ST coords shape: {spcoor.shape}")
    print(f"ST celltype shape: {stct.shape}")
    
    # Print column names to verify
    print(f"SC metadata columns: {scmetadata.columns.tolist()}")
    print(f"ST coords columns: {spcoor.columns.tolist()}")
    print(f"ST celltype columns: {stct.columns.tolist()}")
    
    # Create SC AnnData
    scadata = ad.AnnData(scdata, obs=scmetadata)
    
    # Use correct column names for SC spatial coords
    if 'x_global' in scmetadata.columns and 'y_global' in scmetadata.columns:
        scadata.obsm['spatial'] = scmetadata[['x_global', 'y_global']].values
    elif 'x' in scmetadata.columns and 'y' in scmetadata.columns:
        scadata.obsm['spatial'] = scmetadata[['x', 'y']].values
    else:
        print(f"Warning: Could not find spatial columns in SC metadata. Available: {scmetadata.columns.tolist()}")
        scadata.obsm['spatial'] = None
    
    # Create ST AnnData
    stadata = ad.AnnData(stdata)
    
    # Use correct column names for ST spatial coords
    if 'coord_x' in spcoor.columns and 'coord_y' in spcoor.columns:
        stadata.obsm['spatial'] = spcoor[['coord_x', 'coord_y']].values
    elif 'x' in spcoor.columns and 'y' in spcoor.columns:
        stadata.obsm['spatial'] = spcoor[['x', 'y']].values
    else:
        raise KeyError(f"Could not find spatial columns in ST metadata. Available: {spcoor.columns.tolist()}")
    
    # Process cell type information
    # Exclude spatial coordinate columns from cell type columns
    cell_type_columns = [c for c in stct.columns if c not in ('x', 'y', 'coord_x', 'coord_y', 'x_global', 'y_global')]
    
    dominant_celltypes = []
    for i in range(stct.shape[0]):
        cell_types_present = [col for col, val in zip(cell_type_columns, stct.iloc[i][cell_type_columns]) if val > 0]
        dominant_celltype = cell_types_present[0] if cell_types_present else 'Unknown'
        dominant_celltypes.append(dominant_celltype)
    
    stadata.obs['celltype'] = dominant_celltypes
    
    print(f"SC AnnData: {scadata}")
    print(f"ST AnnData: {stadata}")
    print(f"ST cell types: {stadata.obs['celltype'].value_counts()}")
    
    return scadata, stadata

def main(args=None):
    # Parse args if not provided
    if args is None:
        args = parse_args()
    
    # Extract parameters
    devices = args.devices
    precision = args.precision
    stageA_epochs = args.stageA_epochs
    stageB_outdir = "gems_stageB_cache"
    stageC_epochs = args.stageC_epochs
    stageC_batch = args.batch_size
    lr = args.lr
    outdir = args.outdir

    fabric = Fabric(accelerator="gpu", devices=devices, strategy="ddp", precision=precision)
    fabric.launch()

    # ---------- Load data on all ranks (OK) ----------
    scadata, stadata = load_mouse_data()

    # ---------- Build model (same across ranks) ----------
    n_genes = len(sorted(list(set(scadata.var_names) & set(stadata.var_names))))
    # model = GEMSModel(
    #     n_genes=n_genes,
    #     n_embedding=[512, 256, 128],
    #     D_latent=16,
    #     c_dim=256,
    #     n_heads=4,
    #     isab_m=64,
    #     device=str(fabric.device),
    # )
    model = GEMSModel(
        n_genes=n_genes,
        n_embedding=[512, 256, 128],
        D_latent=16,
        c_dim=256,
        n_heads=4,
        isab_m=64,
        device=str(fabric.device),
        # New geometry-aware parameters
        use_canonicalize=args.use_canonicalize,
        use_dist_bias=args.use_dist_bias,
        dist_bins=args.dist_bins,
        dist_head_shared=args.dist_head_shared,
        use_angle_features=args.use_angle_features,
        angle_bins=args.angle_bins,
        knn_k=args.knn_k,
        self_conditioning=args.self_conditioning,
        sc_feat_mode=args.sc_feat_mode,
        landmarks_L=args.landmarks_L,
    )

    # ---------- Stage A & B on rank-0 only ----------
    if fabric.is_global_zero:
        # Extract tensors like mouse_brain.py does
        import numpy as np
        common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
        X_sc = scadata[:, common].X
        X_st = stadata[:, common].X
        if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
        if hasattr(X_st, "toarray"): X_st = X_st.toarray()
        sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
        st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)
        st_coords = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
        slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)

        print("\n=== Stage A (single GPU, rank-0) ===")
        model.train_stageA(
            st_gene_expr=st_expr,
            st_coords=st_coords,
            sc_gene_expr=sc_expr,
            slide_ids=slide_ids,
            n_epochs=stageA_epochs,
            batch_size=256,
            lr=1e-4,
            sigma=None,
            alpha=0.8,
            ratio_start=0.0,
            ratio_end=1.0,
            mmdbatch=1.0,
            outf=outdir,
        )

        print("\n=== Stage B (single GPU, rank-0) ===")
        slides_dict = {0: (st_coords, st_expr)}
        model.train_stageB(
            slides=slides_dict,
            outdir=stageB_outdir,
        )

        # Save a checkpoint after Stage A/B for other ranks to load
        ckpt_ab = {
            "encoder": model.encoder.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "generator": model.generator.state_dict(),
            "score_net": model.score_net.state_dict(),
        }
        os.makedirs(outdir, exist_ok=True)
        torch.save(ckpt_ab, os.path.join(outdir, "ab_init.pt"))

    # Sync and make sure all ranks see the Stage A/B weights
    fabric.barrier()
    if not fabric.is_global_zero:
        # Load A/B weights
        path = os.path.join(outdir, "ab_init.pt")
        ck = torch.load(path, map_location=fabric.device)
        model.encoder.load_state_dict(ck["encoder"])
        model.context_encoder.load_state_dict(ck["context_encoder"])
        model.generator.load_state_dict(ck["generator"])
        model.score_net.load_state_dict(ck["score_net"])

        # CRITICAL: Non-rank-0 processes need to recompute Stage B
        # because targets_dict is not serializable (has sparse tensors, etc.)
        print(f"[Rank {fabric.global_rank}] Recomputing Stage B targets...")
        common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
        X_st = stadata[:, common].X
        if hasattr(X_st, "toarray"): X_st = X_st.toarray()
        st_expr_rank = torch.tensor(X_st, dtype=torch.float32)
        st_coords_rank = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32)
        
        slides_dict_rank = {0: (st_coords_rank, st_expr_rank)}
        model.train_stageB(
            slides=slides_dict_rank,
            outdir=stageB_outdir,
        )

    # ---------- Stage C (multi-GPU with Fabric) ----------
    print("\n=== Stage C (DDP across GPUs) ===")
    # Rebuild tensors on each rank (cheap)
    common = sorted(list(set(scadata.var_names) & set(stadata.var_names)))
    X_sc = scadata[:, common].X
    X_st = stadata[:, common].X
    if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
    if hasattr(X_st, "toarray"): X_st = X_st.toarray()
    sc_expr = torch.tensor(X_sc, dtype=torch.float32, device=fabric.device)
    st_expr = torch.tensor(X_st, dtype=torch.float32, device=fabric.device)
    st_coords = torch.tensor(stadata.obsm["spatial"], dtype=torch.float32, device=fabric.device)
    slide_ids = torch.zeros(st_expr.shape[0], dtype=torch.long, device=fabric.device)

    st_gene_expr_dict = {0: st_expr}

    training_history = model.train_stageC(
        st_gene_expr_dict=st_gene_expr_dict,
        sc_gene_expr=sc_expr,
        n_min=128, n_max=256,
        num_st_samples=args.num_st_samples,
        num_sc_samples=args.num_sc_samples,
        n_epochs=stageC_epochs,
        batch_size=stageC_batch,   # per-GPU batch; global batch = this * #GPUs
        lr=lr,
        n_timesteps=600,
        sigma_min=0.01,
        sigma_max=5.0,
        outf=outdir,
        fabric=fabric,
        precision=precision,
    )

    fabric.barrier()

    fabric.barrier()
    
    # ---------- Inference (rank-0 only, single GPU) ----------
    if fabric.is_global_zero:
        from datetime import datetime
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create timestamp for all outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        print("\n=== Inference (rank-0) with Multi-Sample Ranking ===")

        if True:

            # Multi-sample inference with quality ranking
            K_samples = 1  # Number of samples to generate
            all_results = []
            all_scores = []

            import utils_et as uet

            for k in range(K_samples):
                print(f"\n  Generating sample {k+1}/{K_samples}...")
                
                # Set different seed for each sample
                torch.manual_seed(42 + k)
                
                sample_results = model.infer_sc_anchored(
                    sc_gene_expr=sc_expr,
                    n_timesteps_sample=300,
                    return_coords=True,
                    anchor_size=640,
                    batch_size=512,
                    eta=0.0,
                    guidance_scale=7.0,
                )
                
                # Compute EDM cone penalty as quality score (lower is better)
                coords_sample = sample_results['coords_canon']
                mask = torch.ones(coords_sample.shape[0], dtype=torch.bool, device=coords_sample.device)
                cone_penalty = uet.edm_cone_penalty_from_V(
                    coords_sample.unsqueeze(0), 
                    mask.unsqueeze(0)
                ).item()
                # cone_penalty = uet.edm_cone_penalty_from_V(coords_sample.unsqueeze(0), mask.unsqueeze(0))
                
                all_results.append(sample_results)
                all_scores.append(cone_penalty)
                
                print(f"  Sample {k+1} EDM cone penalty: {cone_penalty:.6f}")

            # Select best sample (lowest penalty)
            best_idx = int(np.argmin(all_scores))
            results = all_results[best_idx]

            print(f"\n✓ Selected sample {best_idx+1} with lowest cone penalty: {all_scores[best_idx]:.6f}")
            print(f"  Mean penalty: {np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")

            # Extract results from best sample
            D_edm = results['D_edm'].cpu().numpy()
            coords = results['coords'].cpu().numpy()
            coords_canon = results['coords_canon'].cpu().numpy()
            
            # Extract results
            D_edm = results['D_edm'].cpu().numpy()
            coords = results['coords'].cpu().numpy()
            coords_canon = results['coords_canon'].cpu().numpy()
            
            print(f"\nInference complete!")
            print(f"  D_edm shape: {D_edm.shape}")
            print(f"  Coordinates shape: {coords_canon.shape}")

        if False:
            print("\n=== Inference (rank-0) with Multi-Sample Ranking [RANDOM MODE] ===")

            # Get number of cells from sc_expr
            n_cells = sc_expr.shape[0]
            D_latent = 16  # matches your model config

            # Multi-sample inference with quality ranking
            K_samples = 1  # Number of samples to generate
            all_results = []
            all_scores = []
            import utils_et as uet

            for k in range(K_samples):
                print(f"\n  Generating sample {k+1}/{K_samples}...")
                
                # Set different seed for each sample
                torch.manual_seed(42 + k)
                
                # ===== RANDOM RESULTS INSTEAD OF INFERENCE =====
                # Generate random coordinates in latent space
                coords_random = torch.randn(n_cells, D_latent, device=fabric.device)
                
                # Generate random canonicalized coordinates
                coords_canon_random = torch.randn(n_cells, D_latent, device=fabric.device)
                
                # Generate random distance matrix
                D_edm_random = torch.cdist(coords_canon_random.unsqueeze(0), 
                                        coords_canon_random.unsqueeze(0)).squeeze(0)
                
                # Create fake results dictionary matching expected format
                sample_results = {
                    'D_edm': D_edm_random,
                    'coords': coords_random,
                    'coords_canon': coords_canon_random
                }
                # ===== END RANDOM RESULTS =====
                
                # Compute EDM cone penalty as quality score (lower is better)
                coords_sample = sample_results['coords_canon']
                mask = torch.ones(coords_sample.shape[0], dtype=torch.bool, device=coords_sample.device)
                cone_penalty = uet.edm_cone_penalty_from_V(
                    coords_sample.unsqueeze(0), 
                    mask.unsqueeze(0)
                ).item()
                
                all_results.append(sample_results)
                all_scores.append(cone_penalty)
                print(f"  Sample {k+1} EDM cone penalty: {cone_penalty:.6f}")

            # Select best sample (lowest penalty)
            best_idx = int(np.argmin(all_scores))
            results = all_results[best_idx]
            print(f"\n✓ Selected sample {best_idx+1} with lowest cone penalty: {all_scores[best_idx]:.6f}")
            print(f"  Mean penalty: {np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")

            # Extract results from best sample
            D_edm = results['D_edm'].cpu().numpy()
            coords = results['coords'].cpu().numpy()
            coords_canon = results['coords_canon'].cpu().numpy()

            print(f"\nInference complete!")
            print(f"  D_edm shape: {D_edm.shape}")
            print(f"  Coordinates shape: {coords_canon.shape}")
        
        
        # ============================================================================
        # SAVE RESULTS WITH DATETIME
        # ============================================================================
        
        # Save raw results
        results_filename = f"sc_inference_{timestamp}.pt"
        results_path = os.path.join(outdir, results_filename)
        torch.save(results, results_path)
        print(f"\n✓ Saved raw results: {results_path}")
        
        # Save processed results with metadata
        results_processed = {
            'D_edm': D_edm,
            'coords': coords,
            'coords_canon': coords_canon,
            'n_cells': coords_canon.shape[0],
            'timestamp': timestamp,
            'model_config': {
                'n_genes': model.n_genes,
                'D_latent': model.D_latent,
                'c_dim': model.c_dim,
            }
        }
        processed_filename = f"sc_inference_processed_{timestamp}.pt"
        processed_path = os.path.join(outdir, processed_filename)
        torch.save(results_processed, processed_path)
        print(f"✓ Saved processed results: {processed_path}")
        
        # ============================================================================
        # ADD COORDINATES TO ANNDATA
        # ============================================================================
        
        # Add GEMS coordinates to scadata
        scadata.obsm['X_gems'] = coords_canon
        
        # Save AnnData with GEMS coordinates
        adata_filename = f"scadata_with_gems_{timestamp}.h5ad"
        adata_path = os.path.join(outdir, adata_filename)
        scadata.write_h5ad(adata_path)
        print(f"✓ Saved AnnData with GEMS coords: {adata_path}")
        
        # ============================================================================
        # VISUALIZATION
        # ============================================================================
        
        print("\n=== Creating visualizations ===")
        
        # Figure 1: Cell type colored scatter (matplotlib)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: GEMS coordinates colored by cell type
        if 'celltype' in scadata.obs.columns:
            cell_types = scadata.obs['celltype']
            unique_types = cell_types.unique()
            
            # Use a nice color palette
            colors = sns.color_palette("husl", len(unique_types))
            color_map = dict(zip(unique_types, colors))
            
            for ct in unique_types:
                mask = (cell_types == ct).values
                axes[0].scatter(
                    coords_canon[mask, 0], 
                    coords_canon[mask, 1],
                    s=3, 
                    alpha=0.7, 
                    label=ct,
                    c=[color_map[ct]]
                )
            
            axes[0].set_title('GEMS Coordinates (by cell type)', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('GEMS Dim 1', fontsize=12)
            axes[0].set_ylabel('GEMS Dim 2', fontsize=12)
            axes[0].legend(markerscale=3, fontsize=10, loc='best', framealpha=0.9)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].scatter(coords_canon[:, 0], coords_canon[:, 1], s=3, alpha=0.7, c='steelblue')
            axes[0].set_title('GEMS Coordinates', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('GEMS Dim 1', fontsize=12)
            axes[0].set_ylabel('GEMS Dim 2', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        
        axes[0].axis('equal')
        
        # Plot 2: Distance distribution
        upper_tri_idx = np.triu_indices_from(D_edm, k=1)
        distances = D_edm[upper_tri_idx]
        
        axes[1].hist(distances, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        axes[1].set_title('Pairwise Distance Distribution (EDM)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Distance', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].axvline(distances.mean(), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {distances.mean():.2f}')
        axes[1].axvline(np.median(distances), color='g', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(distances):.2f}')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save matplotlib figure
        plot_filename = f"gems_inference_results_{timestamp}.png"
        plot_path = os.path.join(outdir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved matplotlib plot: {plot_path}")
        plt.close()
        
        # ============================================================================
        # SCANPY VISUALIZATION
        # ============================================================================
        
        if 'celltype' in scadata.obs.columns:
            # Create scanpy-style embedding plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            import scanpy as sc
            sc.settings.set_figure_params(dpi=150, frameon=False)
            
            # Use scanpy's embedding plot
            sc.pl.embedding(
                scadata, 
                basis='gems',  # This uses obsm['X_gems']
                color='celltype',
                title='GEMS Embedding (Scanpy)',
                save=False,
                show=False,
                ax=ax,
                size=20,
                alpha=0.8,
                legend_loc='on data',
                legend_fontsize=8
            )
            
            # Save scanpy plot
            scanpy_plot_filename = f"gems_embedding_scanpy_{timestamp}.png"
            scanpy_plot_path = os.path.join(outdir, scanpy_plot_filename)
            plt.savefig(scanpy_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved scanpy plot: {scanpy_plot_path}")
            plt.close()
            
            # Also create a high-quality version with better styling
            fig, ax = plt.subplots(figsize=(12, 10))
            sc.pl.embedding(
                scadata, 
                basis='gems',
                color='celltype',
                title='',
                save=False,
                show=False,
                ax=ax,
                size=30,
                alpha=0.7,
                frameon=True,
                legend_loc='right margin',
                legend_fontsize=10
            )
            ax.set_xlabel('GEMS-1', fontsize=14, fontweight='bold')
            ax.set_ylabel('GEMS-2', fontsize=14, fontweight='bold')
            
            hq_plot_filename = f"gems_embedding_hq_{timestamp}.png"
            hq_plot_path = os.path.join(outdir, hq_plot_filename)
            plt.savefig(hq_plot_path, dpi=600, bbox_inches='tight')
            print(f"✓ Saved high-quality plot: {hq_plot_path}")
            plt.close()
        
        # ============================================================================
        # SUMMARY STATISTICS
        # ============================================================================
        
        print("\n" + "="*70)
        print("INFERENCE SUMMARY")
        print("="*70)
        print(f"Timestamp: {timestamp}")
        print(f"Number of cells: {coords_canon.shape[0]}")
        print(f"Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]")
        print(f"Distance statistics:")
        print(f"  Mean: {distances.mean():.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        print(f"  Std: {distances.std():.4f}")
        print(f"  Min: {distances.min():.4f}")
        print(f"  Max: {distances.max():.4f}")
        
        if 'celltype' in scadata.obs.columns:
            print(f"\nCell type distribution:")
            for ct, count in scadata.obs['celltype'].value_counts().items():
                print(f"  {ct}: {count} cells ({count/len(scadata)*100:.1f}%)")
        
        print("\n" + "="*70)
        print("All outputs saved to:", outdir)
        print("="*70)
        
        # Create a summary file
        summary_filename = f"inference_summary_{timestamp}.txt"
        summary_path = os.path.join(outdir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write("GEMS Inference Summary\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Number of cells: {coords_canon.shape[0]}\n")
            f.write(f"Coordinate range: [{coords_canon.min():.2f}, {coords_canon.max():.2f}]\n")
            f.write(f"\nDistance statistics:\n")
            f.write(f"  Mean: {distances.mean():.4f}\n")
            f.write(f"  Median: {np.median(distances):.4f}\n")
            f.write(f"  Std: {distances.std():.4f}\n")
            f.write(f"  Min: {distances.min():.4f}\n")
            f.write(f"  Max: {distances.max():.4f}\n")
            if 'celltype' in scadata.obs.columns:
                f.write(f"\nCell type distribution:\n")
                for ct, count in scadata.obs['celltype'].value_counts().items():
                    f.write(f"  {ct}: {count} cells ({count/len(scadata)*100:.1f}%)\n")
        
        print(f"✓ Saved summary: {summary_path}")


        # ============================================================================
        # PLOT STAGE C TRAINING LOSSES
        # ============================================================================
        
        print("\n=== Loading and plotting Stage C training losses ===")
        
        # ============================================================================
        # PLOT STAGE C TRAINING LOSSES (NO CHECKPOINT NEEDED)
        # ============================================================================

        if fabric.is_global_zero:
            print("\n=== Plotting Stage C training losses ===")
            
            if training_history is not None and len(training_history['epoch']) > 0:
                epochs = training_history['epoch']
                losses = training_history['epoch_avg']
                
                print(f"Found {len(epochs)} epochs of training history")
                
                # ============================================================================
                # CREATE COMPREHENSIVE LOSS PLOTS
                # ============================================================================
                
                # Plot 1: All losses on separate subplots (detailed view)
                fig, axes = plt.subplots(3, 3, figsize=(20, 15))
                fig.suptitle('Stage C Training Losses (All Components)', fontsize=18, fontweight='bold', y=0.995)
                
                loss_names = ['total', 'score', 'cone', 'gram', 'heat', 'sw_st', 'sw_sc', 'overlap', 'ordinal_sc']
                colors = ['black', 'blue', 'cyan', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
                
                for idx, (name, color) in enumerate(zip(loss_names, colors)):
                    if name in losses and len(losses[name]) > 0:
                        ax = axes[idx // 3, idx % 3]
                        ax.plot(epochs, losses[name], color=color, linewidth=2, alpha=0.7, marker='o', markersize=4)
                        ax.set_xlabel('Epoch', fontsize=12)
                        ax.set_ylabel('Loss', fontsize=12)
                        ax.set_title(f'{name.upper()} Loss', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        # Add smoothed trend line if enough data
                        if len(epochs) > 10:
                            from scipy.ndimage import gaussian_filter1d
                            smoothed = gaussian_filter1d(losses[name], sigma=2)
                            ax.plot(epochs, smoothed, '--', color=color, linewidth=2.5, alpha=0.5, label='Trend')
                            ax.legend(fontsize=10)
                
                # Hide empty subplot
                if len(loss_names) < 9:
                    axes[2, 2].axis('off')
                
                plt.tight_layout()
                detailed_plot_filename = f"stageC_losses_detailed_{timestamp}.png"
                detailed_plot_path = os.path.join(outdir, detailed_plot_filename)
                plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved detailed loss plot: {detailed_plot_path}")
                plt.close()
                
                # ============================================================================
                # Plot 2: All losses on ONE plot (log scale for comparison)
                # ============================================================================
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                loss_components = ['score', 'cone', 'gram', 'heat', 'sw_st', 'sw_sc', 'overlap', 'ordinal_sc']
                colors_comp = ['blue', 'cyan', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
                
                for name, color in zip(loss_components, colors_comp):
                    if name in losses and len(losses[name]) > 0:
                        ax.plot(epochs, losses[name], color=color, linewidth=2.5, 
                            label=name.upper(), marker='o', markersize=5, 
                            markevery=max(1, len(epochs)//20), alpha=0.8)
                
                ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
                ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
                ax.set_title('Stage C Training - All Loss Components', fontsize=16, fontweight='bold')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, which='both')
                ax.legend(fontsize=12, loc='best', framealpha=0.9)
                
                plt.tight_layout()
                combined_plot_filename = f"stageC_losses_combined_{timestamp}.png"
                combined_plot_path = os.path.join(outdir, combined_plot_filename)
                plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved combined loss plot: {combined_plot_path}")
                plt.close()
                
                # ============================================================================
                # Plot 3: Total loss only (clean view)
                # ============================================================================
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if 'total' in losses and len(losses['total']) > 0:
                    ax.plot(epochs, losses['total'], color='black', linewidth=3, 
                        marker='o', markersize=6, markevery=max(1, len(epochs)//20), 
                        alpha=0.8, label='Total Loss')
                    
                    # Add smoothed trend
                    if len(epochs) > 10:
                        from scipy.ndimage import gaussian_filter1d
                        smoothed = gaussian_filter1d(losses['total'], sigma=3)
                        ax.plot(epochs, smoothed, '--', color='red', linewidth=2.5, 
                            alpha=0.6, label='Trend (smoothed)')
                    
                    # Add min/max annotations
                    min_loss = min(losses['total'])
                    min_epoch = epochs[losses['total'].index(min_loss)]
                    ax.axhline(y=min_loss, color='green', linestyle=':', linewidth=2, alpha=0.5)
                    ax.text(0.02, 0.98, f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})', 
                        transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
                ax.set_ylabel('Total Loss', fontsize=14, fontweight='bold')
                ax.set_title('Stage C Training - Total Loss', fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12, loc='best')
                
                plt.tight_layout()
                total_plot_filename = f"stageC_loss_total_{timestamp}.png"
                total_plot_path = os.path.join(outdir, total_plot_filename)
                plt.savefig(total_plot_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved total loss plot: {total_plot_path}")
                plt.close()
                
                # ============================================================================
                # Save loss history as JSON for easy access
                # ============================================================================
                
                history_json = {
                    'epochs': epochs,
                    'losses': {k: [float(x) for x in v] for k, v in losses.items()},
                    'n_epochs': len(epochs),
                    'final_total_loss': float(losses['total'][-1]) if 'total' in losses else None,
                    'min_total_loss': float(min(losses['total'])) if 'total' in losses else None,
                    'timestamp': timestamp
                }
                
                json_filename = f"stageC_training_history_{timestamp}.json"
                json_path = os.path.join(outdir, json_filename)
                
                import json
                with open(json_path, 'w') as f:
                    json.dump(history_json, f, indent=2)
                
                print(f"✓ Saved training history JSON: {json_path}")
                
                # ============================================================================
                # Print loss summary
                # ============================================================================
                
                print("\n" + "="*70)
                print("STAGE C TRAINING SUMMARY")
                print("="*70)
                print(f"Total epochs trained: {len(epochs)}")
                print(f"Final total loss: {losses['total'][-1]:.4f}")
                print(f"Minimum total loss: {min(losses['total']):.4f} (Epoch {epochs[losses['total'].index(min(losses['total']))]})") 
                
                print("\nFinal loss components:")
                for name in loss_components:
                    if name in losses and len(losses[name]) > 0:
                        print(f"  {name}: {losses[name][-1]:.4f}")
                
                print("="*70)
                
            else:
                print("⚠ No training history available (training may have failed)")


# if __name__ == "__main__":
#     # Example: 2 GPUs, fp16
#     main(devices=2, precision="16-mixed",
#          stageA_epochs=1000, stageC_epochs=600, stageC_batch=64,
#          lr=1e-4, outdir="gems_mousebrain_output")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)