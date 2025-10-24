"""
GEMS (Generative Euclidean Metric Synthesis) - Orchestrator
Part 3: Main GEMS model class with complete training pipeline
"""

import torch
import torch.nn as nn
import os
from typing import Dict, List, Tuple, Optional

from core_models_et_p1 import (
    SharedEncoder, train_encoder, STStageBPrecomputer, 
    STSetDataset, STTargets
)
from core_models_et_p2 import (
    SetEncoderContext, MetricSetGenerator, DiffusionScoreNet,
    train_stageC_diffusion_generator
)
import utils_et as uet
import numpy as np

# ==============================================================================
# GEMS MODEL - MAIN ORCHESTRATOR
# ==============================================================================

class GEMSModel:
    """
    GEMS (Generative Euclidean Metric Synthesis) - Main Model
    
    Complete pipeline:
    1. Stage A: Shared encoder (align ST & SC)
    2. Stage B: Precompute pose-free targets
    3. Stage C: Train set-equivariant diffusion generator
    4. Stage D: Inference (sample EDM for SC)
    
    Usage:
        model = GEMSModel(n_genes=2000, device='cuda')
        model.train_stageA(st_gene_expr, st_coords, sc_gene_expr, ...)
        model.train_stageB(slides_dict)
        model.train_stageC(...)
        results = model.infer_sc(sc_gene_expr)
    """
    
    def __init__(
        self,
        n_genes: int,
        n_embedding: List[int] = [512, 256, 128],
        D_latent: int = 16,
        c_dim: int = 256,
        n_heads: int = 4,
        isab_m: int = 64,
        device: str = 'cuda'
    ):
        """
        Args:
            n_genes: number of genes
            n_embedding: encoder MLP dimensions
            D_latent: latent dimension for generator
            c_dim: context dimension
            n_heads: attention heads
            isab_m: ISAB inducing points
            device: torch device
        """
        self.n_genes = n_genes
        self.n_embedding = n_embedding
        self.D_latent = D_latent
        self.c_dim = c_dim
        self.n_heads = n_heads
        self.isab_m = isab_m
        self.device = device
        
        # Stage A: Shared encoder
        self.encoder = SharedEncoder(n_genes, n_embedding).to(device)
        
        # Stage B: Precomputer
        self.precomputer = STStageBPrecomputer(device=device)
        self.targets_dict = {}
        
        # Stage C: Generator components
        h_dim = n_embedding[-1]
        self.context_encoder = SetEncoderContext(
            h_dim=h_dim, c_dim=c_dim, n_heads=n_heads, 
            n_blocks=3, isab_m=isab_m
        ).to(device)
        
        self.generator = MetricSetGenerator(
            c_dim=c_dim, D_latent=D_latent, n_heads=n_heads,
            n_blocks=2, isab_m=isab_m
        ).to(device)
        
        self.score_net = DiffusionScoreNet(
            D_latent=D_latent, c_dim=c_dim, n_heads=n_heads,
            n_blocks=4, isab_m=isab_m
        ).to(device)
        
        print(f"GEMS Model initialized:")
        print(f"  Encoder: {n_genes} → {n_embedding}")
        print(f"  D_latent: {D_latent}")
        print(f"  Context dim: {c_dim}")
        print(f"  ISAB inducing points: {isab_m}")
    
    # ==========================================================================
    # STAGE A: ENCODER TRAINING
    # ==========================================================================
    def train_stageA(
        self,
        st_gene_expr: torch.Tensor,
        st_coords: torch.Tensor,
        sc_gene_expr: torch.Tensor,
        slide_ids: Optional[torch.Tensor] = None,
        n_epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        sigma: Optional[float] = None,
        alpha: float = 0.9,
        ratio_start: float = 0.0,      # ADD THIS
        ratio_end: float = 1.0,        # ADD THIS
        mmdbatch: float = 0.1,
        outf: str = 'output'
    ):
        """
        Train shared encoder (Stage A).
        
        Args:
            st_gene_expr: (n_st, n_genes)
            st_coords: (n_st, 2)
            sc_gene_expr: (n_sc, n_genes)
            slide_ids: (n_st,) optional slide identifiers
            n_epochs: training epochs
            batch_size: batch size
            lr: learning rate
            sigma: RBF bandwidth (auto if None)
            alpha: MMD loss weight (domain alignment)           # UPDATE THIS
            ratio_start: circle loss warmup start value         # ADD THIS
            ratio_end: circle loss warmup end value             # ADD THIS
            mmdbatch: MMD batch fraction
            outf: output directory
        """
        print("\n" + "="*60)
        print("STAGE A: Training Shared Encoder")
        print("="*60)
        
        self.encoder = train_encoder(
            model=self.encoder,
            st_gene_expr=st_gene_expr,
            st_coords=st_coords,
            sc_gene_expr=sc_gene_expr,
            slide_ids=slide_ids,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            sigma=sigma,
            alpha=alpha,
            ratio_start=ratio_start,      # ADD THIS
            ratio_end=ratio_end,          # ADD THIS
            mmdbatch=mmdbatch,
            device=self.device,
            outf=outf
        )
        
        # Freeze encoder for subsequent stages
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        print("Stage A complete. Encoder frozen.")
    
    # ==========================================================================
    # STAGE B: PRECOMPUTE GEOMETRIC TARGETS
    # ==========================================================================
    
    def train_stageB(
        self,
        slides: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        outdir: str = 'stage_b_cache'
    ):
        """
        Precompute pose-free geometric targets for ST slides.
        """
        print("\n" + "="*60)
        print("STAGE B: Precomputing Geometric Targets")
        print("="*60)
        
        # Auto-clear old cache if it exists
        import shutil
        if os.path.exists(outdir):
            print(f"Clearing old cache at {outdir}")
            shutil.rmtree(outdir)
        
        self.targets_dict = self.precomputer.precompute(
            slides=slides,
            encoder=self.encoder,
            outdir=outdir
        )
        
        print("Stage B complete. Targets computed.")
    
    # ==========================================================================
    # STAGE C: TRAIN DIFFUSION GENERATOR
    # ==========================================================================
    
    def train_stageC(
        self,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        sc_gene_expr: torch.Tensor,
        n_min: int = 64,
        n_max: int = 256,
        num_st_samples: int = 10000,
        num_sc_samples: int = 5000,
        n_epochs: int = 1000,
        batch_size: int = 4,
        lr: float = 1e-4,
        n_timesteps: int = 600,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        outf: str = 'output'
    ):
        """
        Train diffusion generator with mixed ST/SC regimen.
        """
        print("\n" + "="*60)
        print("STAGE C: Training Diffusion Generator (Mixed ST/SC)")
        print("="*60)
        
        # ST dataset
        from core_models_et_p1 import STSetDataset, SCSetDataset
        st_dataset = STSetDataset(
            targets_dict=self.targets_dict,
            encoder=self.encoder,
            st_gene_expr_dict=st_gene_expr_dict,
            n_min=n_min,
            n_max=n_max,
            D_latent=self.D_latent,
            num_samples=num_st_samples,
            knn_k=12,  # NEW: add knn_k parameter
            device=self.device
        )
                
        # SC dataset
        sc_dataset = SCSetDataset(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            n_min=n_min,
            n_max=n_max,
            num_samples=num_sc_samples,
            device=self.device
        )
        
        # Precompute ST prototypes
        from core_models_et_p2 import precompute_st_prototypes
        prototype_bank = precompute_st_prototypes(
            targets_dict=self.targets_dict,
            encoder=self.encoder,
            st_gene_expr_dict=st_gene_expr_dict,
            n_prototypes=3000,
            device=self.device
        )
        
        # Train
        from core_models_et_p2 import train_stageC_diffusion_generator
        train_stageC_diffusion_generator(
            context_encoder=self.context_encoder,
            generator=self.generator,
            score_net=self.score_net,
            st_dataset=st_dataset,
            sc_dataset=sc_dataset,
            prototype_bank=prototype_bank,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_timesteps=n_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=self.device,
            outf=outf
        )
        
        print("Stage C complete.")
    
    # ==========================================================================
    # STAGE D: SC INFERENCE
    # ==========================================================================
    
    def infer_sc(
        self,
        sc_gene_expr: torch.Tensor,
        n_samples: int = 1,
        n_timesteps_sample: int = 250,
        return_coords: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Infer EDM (and coordinates) for SC data.
        
        Args:
            sc_gene_expr: (n_sc, n_genes)
            n_samples: number of samples (currently only 1 supported)
            n_timesteps_sample: reverse diffusion steps
            return_coords: whether to compute coordinates
            
        Returns:
            dict with 'D_edm', optionally 'coords', 'coords_canon'
        """
        print("\n" + "="*60)
        print("STAGE D: SC Inference")
        print("="*60)
        
        results = sample_sc_edm(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            context_encoder=self.context_encoder,
            score_net=self.score_net,
            n_samples=n_samples,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=0.01,
            sigma_max=50.0,
            return_coords=return_coords,
            device=self.device
        )
        
        return results
    
    # ==========================================================================
    # SAVE / LOAD
    # ==========================================================================
    
    def save(self, path: str):
        """Save all model components."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'score_net': self.score_net.state_dict(),
            'config': {
                'n_genes': self.n_genes,
                'n_embedding': self.n_embedding,
                'D_latent': self.D_latent,
                'c_dim': self.c_dim,
                'n_heads': self.n_heads,
                'isab_m': self.isab_m
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load all model components."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.score_net.load_state_dict(checkpoint['score_net'])
        print(f"Model loaded from {path}")
    
    # ==========================================================================
    # COMPLETE TRAINING PIPELINE
    # ==========================================================================
    
    def train(
        self,
        st_gene_expr: torch.Tensor,
        st_coords: torch.Tensor,
        sc_gene_expr: torch.Tensor,
        slide_ids: Optional[torch.Tensor] = None,
        encoder_epochs: int = 1000,
        stageC_epochs: int = 1000,
        outf: str = 'output'
    ):
        """
        Complete training pipeline: A → B → C.
        
        Args:
            st_gene_expr: (n_st, n_genes)
            st_coords: (n_st, 2)
            sc_gene_expr: (n_sc, n_genes)
            slide_ids: (n_st,) optional
            encoder_epochs: Stage A epochs
            stageC_epochs: Stage C epochs
            outf: output directory
        """
        # Stage A
        self.train_stageA(
            st_gene_expr, st_coords, sc_gene_expr,
            slide_ids=slide_ids,
            n_epochs=encoder_epochs,
            outf=outf
        )
        
        # Stage B
        if slide_ids is None:
            slides = {0: (st_coords, st_gene_expr)}
            st_gene_expr_dict = {0: st_gene_expr}
        else:
            # Group by slide
            unique_slides = torch.unique(slide_ids).cpu().numpy()
            slides = {}
            st_gene_expr_dict = {}
            for sid in unique_slides:
                mask = (slide_ids == sid).cpu().numpy()
                slides[int(sid)] = (st_coords[mask], st_gene_expr[mask])
                st_gene_expr_dict[int(sid)] = st_gene_expr[mask]
        
        self.train_stageB(slides, outdir=os.path.join(outf, 'stage_b_cache'))
        
        # Stage C
        self.train_stageC(
            st_gene_expr_dict,
            n_epochs=stageC_epochs,
            outf=outf
        )
        
        print("\n" + "="*60)
        print("GEMS TRAINING COMPLETE!")
        print("="*60)


    def infer_sc_batched(
        self,
        sc_gene_expr: torch.Tensor,
        n_timesteps_sample: int = 250,
        return_coords: bool = True,
        batch_size: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        MEMORY-OPTIMIZED batched inference for SC data.
        
        Processes cells in batches to avoid OOM errors.
        
        Args:
            sc_gene_expr: (n_sc, n_genes)
            n_timesteps_sample: reverse diffusion steps
            return_coords: whether to compute coordinates
            batch_size: cells per batch (reduce if OOM)
            
        Returns:
            dict with 'D_edm', optionally 'coords', 'coords_canon'
        """
        print("\n" + "="*60)
        print("STAGE D: SC Inference (BATCHED)")
        print("="*60)
        
        self.encoder.eval()
        self.context_encoder.eval()
        self.score_net.eval()
        
        n_sc = sc_gene_expr.shape[0]
        D_latent = self.D_latent
        
        print(f"Processing {n_sc} cells in batches of {batch_size}...")
        
        n_batches = (n_sc + batch_size - 1) // batch_size
        all_V_0 = []
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx+1}/{n_batches}...")
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_sc)
                batch_size_actual = end_idx - start_idx
                
                sc_batch = sc_gene_expr[start_idx:end_idx].to(self.device)
                
                # Encode
                Z_sc_batch = self.encoder(sc_batch)
                Z_sc_input = Z_sc_batch.unsqueeze(0)
                mask = torch.ones(1, batch_size_actual, dtype=torch.bool, device=self.device)
                
                # Context
                H = self.context_encoder(Z_sc_input, mask)
                
                # Initialize noise
                V_t = torch.randn(1, batch_size_actual, D_latent, device=self.device) * 50.0
                V_t = V_t - V_t.mean(dim=1, keepdim=True)
                
                if not torch.isfinite(V_t).all():
                    print(f"WARNING: NaNs in batch {batch_idx+1}/{n_batches}")
                    V_t = torch.nan_to_num(V_t, nan=0.0, posinf=0.0, neginf=0.0)

                # Reverse diffusion
                sigmas = torch.exp(torch.linspace(
                    np.log(50.0), np.log(0.01), n_timesteps_sample, device=self.device
                ))
                
                for t_idx in reversed(range(n_timesteps_sample)):
                    t_norm = torch.tensor([[t_idx / (n_timesteps_sample - 1)]], device=self.device)
                    sigma_t = sigmas[t_idx]
                    
                    eps_pred = self.score_net(V_t, t_norm, H, mask)
                    s_hat = -eps_pred / (sigma_t + 1e-8)  # Convert eps to score

                    if t_idx > 0:
                        sigma_prev = sigmas[t_idx - 1]
                        d_sigma = sigma_prev - sigma_t
                        V_t = V_t + d_sigma * s_hat
                        
                        # Optional stochasticity
                        eta = 0.2  # Start deterministic
                        if eta > 0:
                            noise_std = ((sigma_prev**2 - sigma_t**2).clamp(min=0)).sqrt()
                            V_t = V_t + eta * noise_std * torch.randn_like(V_t)
                    else:
                        V_t = V_t - sigma_t * s_hat
                    
                    V_t = V_t - V_t.mean(dim=1, keepdim=True)
                
                # Store on CPU
                all_V_0.append(V_t.squeeze(0).cpu())
                
                # Clear cache
                del Z_sc_batch, Z_sc_input, H, V_t, eps_pred, sc_batch, mask
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate
        V_0_full = torch.cat(all_V_0, dim=0)
        print(f"Concatenated V_0: {V_0_full.shape}")
        
        # Compute distances on CPU
        print("Computing Gram and distances...")
        G = V_0_full @ V_0_full.t()
        diag = torch.diag(G).unsqueeze(1)
        D = torch.sqrt(torch.clamp(diag + diag.t() - 2 * G, min=0))
        
        # EDM projection
        print("EDM projection...")
        try:
            D_gpu = D.to(self.device)
            D_edm = uet.edm_project(D_gpu).cpu()
            del D_gpu
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        except:
            D_edm = uet.edm_project(D)

        if not torch.isfinite(D).all():
            print("WARNING: Non-finite distances before EDM projection!")
            D = torch.nan_to_num(D, nan=1e-3, posinf=1e3, neginf=0.0)
        
        result = {'D_edm': D_edm}
        
        if return_coords:
            print("Computing MDS coordinates...")
            try:
                D_edm_gpu = D_edm.to(self.device)
                n = D_edm_gpu.shape[0]
                J = torch.eye(n, device=self.device) - torch.ones(n, n, device=self.device) / n
                B = -0.5 * J @ (D_edm_gpu ** 2) @ J
                coords = uet.classical_mds(B, d_out=2)
                coords_canon = uet.canonicalize_coords(coords)
                result['coords'] = coords.cpu()
                result['coords_canon'] = coords_canon.cpu()
                del D_edm_gpu, J, B, coords, coords_canon
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            except:
                print("  (using CPU for MDS)")
                n = D_edm.shape[0]
                J = torch.eye(n) - torch.ones(n, n) / n
                B = -0.5 * J @ (D_edm ** 2) @ J
                coords = uet.classical_mds(B, d_out=2)
                coords_canon = uet.canonicalize_coords(coords)
                result['coords'] = coords
                result['coords_canon'] = coords_canon
        
        print("SC inference complete!")
        return result
    
    def infer_sc_anchored(
        self,
        sc_gene_expr: torch.Tensor,
        n_timesteps_sample: int = 160,
        return_coords: bool = True,
        anchor_size: int = 384,
        batch_size: int = 512,
        eta: float = 0.0,
        guidance_scale: float = 7.0
    ) -> Dict[str, torch.Tensor]:
        """
        ANCHOR-CONDITIONED inference for SC data (FIXES GLOBAL GEOMETRY).
        
        This is the RECOMMENDED inference method. It ensures all cells
        are placed in a globally consistent coordinate system.
        
        Args:
            sc_gene_expr: (n_sc, n_genes)
            n_timesteps_sample: diffusion steps (120-200 recommended for 20GB GPU)
            return_coords: compute coordinates
            anchor_size: number of anchor cells (256-384)
            batch_size: cells per batch excluding anchors (384-512)
            eta: stochasticity (0.0 = deterministic, safer)
            
        Returns:
            dict with 'D_edm', 'coords', 'coords_canon'
        """
        from core_models_et_p2 import sample_sc_edm_anchored
        
        return sample_sc_edm_anchored(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            context_encoder=self.context_encoder,
            score_net=self.score_net,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=0.01,
            sigma_max=50.0,
            return_coords=return_coords,
            anchor_size=anchor_size,
            batch_size=batch_size,
            eta=eta,
            guidance_scale=guidance_scale,  # <- ADD THIS
            device=self.device
        )

