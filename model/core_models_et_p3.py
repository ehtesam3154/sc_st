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
    train_stageC_diffusion_generator, sample_sc_edm
)
import utils_et as uet

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
            alpha: circular coupling weight
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
        
        Args:
            slides: {slide_id: (st_coords, st_gene_expr)}
            outdir: cache directory
        """
        print("\n" + "="*60)
        print("STAGE B: Precomputing Geometric Targets")
        print("="*60)
        
        self.targets_dict = self.precomputer.precompute(
            slides=slides,
            encoder=self.encoder,
            outdir=outdir
        )
        
        print("Stage B complete. Targets cached.")
    
    # ==========================================================================
    # STAGE C: TRAIN DIFFUSION GENERATOR
    # ==========================================================================
    
    def train_stageC(
        self,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_min: int = 256,
        n_max: int = 1024,
        num_samples: int = 10000,
        n_epochs: int = 1000,
        batch_size: int = 4,
        lr: float = 1e-4,
        n_timesteps: int = 1000,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        loss_weights: Optional[Dict[str, float]] = None,
        outf: str = 'output'
    ):
        """
        Train set-equivariant diffusion generator (Stage C).
        
        Args:
            st_gene_expr_dict: {slide_id: st_gene_expr}
            n_min, n_max: mini-set size range
            num_samples: number of mini-sets
            n_epochs: training epochs
            batch_size: batch size
            lr: learning rate
            n_timesteps: diffusion steps
            sigma_min, sigma_max: VE SDE noise levels
            loss_weights: {'alpha', 'beta', 'gamma', 'eta'}
            outf: output directory
        """
        print("\n" + "="*60)
        print("STAGE C: Training Diffusion Generator")
        print("="*60)
        
        if loss_weights is None:
            loss_weights = {'alpha': 0.1, 'beta': 1.0, 'gamma': 0.25, 'eta': 0.5}
        
        # Create mini-set dataset
        dataset = STSetDataset(
            targets_dict=self.targets_dict,
            encoder=self.encoder,
            st_gene_expr_dict=st_gene_expr_dict,
            n_min=n_min,
            n_max=n_max,
            D_latent=self.D_latent,
            num_samples=num_samples,
            device=self.device
        )
        
        print(f"Created dataset with {len(dataset)} mini-sets")
        
        # Train
        train_stageC_diffusion_generator(
            context_encoder=self.context_encoder,
            generator=self.generator,
            score_net=self.score_net,
            dataset=dataset,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_timesteps=n_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            loss_weights=loss_weights,
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
