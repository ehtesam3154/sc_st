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

# from core_models_et_p2 import (
#     SetEncoderContext, MetricSetGenerator, DiffusionScoreNet,
#     train_stageC_diffusion_generator
# )

from core_models_et_p2_v2 import (
    SetEncoderContext, MetricSetGenerator, DiffusionScoreNet
)
from core_models_et_p2 import train_stageC_diffusion_generator

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
        device: str = 'cuda',
        #new params
        use_canonicalize: bool = True,
        use_dist_bias: bool = True,
        dist_bins: int = 16,
        dist_head_shared: bool = True,
        use_angle_features: bool = True,
        angle_bins: int = 8,
        knn_k: int = 12,
        self_conditioning: bool = True,
        sc_feat_mode: str = "concat",
        lambda_cone: float = 1e-3,
        landmarks_L: int = 32

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

        # Store all configuration
        self.cfg = {
            'denoiser': {
                'use_canonicalize': use_canonicalize,
                'use_dist_bias': use_dist_bias,
                'dist_bins': dist_bins,
                'dist_head_shared': dist_head_shared,
                'use_angle_features': use_angle_features,
                'angle_bins': angle_bins,
                'knn_k': knn_k,
                'self_conditioning': self_conditioning,
                'sc_feat_mode': sc_feat_mode,
            },
            'dataset': {
                'landmarks_L': landmarks_L,
            }
        }
        
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
        
        # self.score_net = DiffusionScoreNet(
        #     D_latent=D_latent, c_dim=c_dim, n_heads=n_heads,
        #     n_blocks=4, isab_m=isab_m
        # ).to(device)

        self.score_net = DiffusionScoreNet(
            D_latent=D_latent, 
            c_dim=c_dim, 
            n_heads=n_heads,
            n_blocks=4, 
            isab_m=isab_m,
            use_canonicalize=use_canonicalize,
            use_dist_bias=use_dist_bias,
            dist_bins=dist_bins,
            dist_head_shared=dist_head_shared,
            use_angle_features=use_angle_features,
            angle_bins=angle_bins,
            knn_k=knn_k,
            self_conditioning=self_conditioning,
            sc_feat_mode=sc_feat_mode,
            use_st_dist_head=True
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
        outf: str = 'output',
        fabric: Optional['Fabric'] = None,
        precision: str = "16-mixed",
        logger = None,
        log_interval: int = 10,
        # Early stopping
        enable_early_stop: bool = True,
        early_stop_min_epochs: int = 12,
        early_stop_patience: int = 6,
        early_stop_threshold: float = 0.01,
        phase_name: str = "Mixed",  # "ST-only" or "Fine-tune" or "Mixed"
):
        """
        Train diffusion generator with mixed ST/SC regimen.
        """
        print("\n" + "="*60)
        # print("STAGE C: Training Diffusion Generator (Mixed ST/SC)")
        print(f"STAGE C: Training Diffusion Generator ({phase_name})")

        print("="*60)
        
        # ST dataset
        from core_models_et_p1 import STSetDataset, SCSetDataset

        # Ensure gene expression is on CPU for datasets
        st_gene_expr_dict_cpu = {
            k: v.cpu() if torch.is_tensor(v) else v 
            for k, v in st_gene_expr_dict.items()
        }
        sc_gene_expr_cpu = sc_gene_expr.cpu() if torch.is_tensor(sc_gene_expr) else sc_gene_expr

        # ST dataset (conditional - can be None if num_st_samples=0)
        if num_st_samples > 0:
            st_dataset = STSetDataset(
                targets_dict=self.targets_dict,
                encoder=self.encoder,
                st_gene_expr_dict=st_gene_expr_dict_cpu,
                n_min=n_min,
                n_max=n_max,
                D_latent=self.D_latent,
                num_samples=num_st_samples,
                knn_k=12,
                device=self.device,
                landmarks_L=self.cfg['dataset']['landmarks_L']
            )
        else:
            st_dataset = None
                
        # SC dataset (optional - can be None if num_sc_samples=0)
        if num_sc_samples > 0:
            sc_dataset = SCSetDataset(
                sc_gene_expr=sc_gene_expr_cpu,
                encoder=self.encoder,
                n_min=n_min,
                n_max=n_max,
                num_samples=num_sc_samples,
                device=self.device,
                landmarks_L=self.cfg['dataset']['landmarks_L']
            )
            print(f"[SC Dataset] Created with {num_sc_samples} samples")
        else:
            sc_dataset = None
            print("[SC Dataset] DISABLED (num_sc_samples=0)")

        
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
        history = train_stageC_diffusion_generator(
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
            outf=outf,
            fabric=fabric,
            precision=precision,
            logger=logger,
            log_interval=log_interval,
            # Early stopping
            enable_early_stop=enable_early_stop,
            early_stop_min_epochs=early_stop_min_epochs,
            early_stop_patience=early_stop_patience,
            early_stop_threshold=early_stop_threshold,
        )
        
        print("Stage C complete.")

        return history
    

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'score_net': self.score_net.state_dict(),
            'cfg': self.cfg,  # ADD THIS
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load all model components."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.score_net.load_state_dict(checkpoint['score_net'])
        
        # Load config if available
        if 'cfg' in checkpoint:
            self.cfg = checkpoint['cfg']
            
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
    


    def infer_sc_anchored(
        self,
        sc_gene_expr: torch.Tensor,
        n_timesteps_sample: int = 160,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        return_coords: bool = True,
        anchor_size: int = 384,
        batch_size: int = 512,
        eta: float = 0.0,
        guidance_scale: float = 8.0,
    ) -> Dict[str, torch.Tensor]:
        
        from core_models_et_p2 import sample_sc_edm_anchored
        
        return sample_sc_edm_anchored(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            context_encoder=self.context_encoder,
            score_net=self.score_net,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            return_coords=return_coords,
            anchor_size=anchor_size,
            batch_size=batch_size,
            eta=eta,
            device=self.device,
            guidance_scale=guidance_scale,
            DEBUG_FLAG=True,
            DEBUG_EVERY=10,
        )
    
    def infer_sc_patchwise(
        self,
        sc_gene_expr: torch.Tensor,
        n_timesteps_sample: int = 160,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        return_coords: bool = True,
        patch_size: int = 384,
        coverage_per_cell: float = 4.0,
        n_align_iters: int = 10,
        eta: float = 0.0,
        guidance_scale: float = 3.0,
        debug_flag: bool = True,
        debug_every: int = 10,
        fixed_patch_graph: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        SC inference using patch-based global alignment (no masked/frozen points).
        """
        from core_models_et_p2 import sample_sc_edm_patchwise

        # If you have a stored ST scale, you can pass it; otherwise None
        target_st_p95 = getattr(self, "target_st_p95", None)

        res = sample_sc_edm_patchwise(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            context_encoder=self.context_encoder,
            score_net=self.score_net,
            target_st_p95=target_st_p95,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            guidance_scale=guidance_scale,
            eta=eta,
            device=self.device,
            patch_size=patch_size,
            coverage_per_cell=coverage_per_cell,
            n_align_iters=n_align_iters,
            return_coords=return_coords,
            DEBUG_FLAG=debug_flag,
            DEBUG_EVERY=debug_every,
            fixed_patch_graph=fixed_patch_graph,
        )


        return res
    

    def infer_sc_single_patch(
        self,
        sc_gene_expr: torch.Tensor,
        n_timesteps_sample: int = 500,
        sigma_min: float = 0.01,
        sigma_max: float = 3.0,
        guidance_scale: float = 2.0,
        eta: float = 0.0,
        target_st_p95: Optional[float] = None,
        return_coords: bool = True,
        DEBUG_FLAG: bool = True,
    ):
        """
        Single-patch SC inference (no patchwise stitching).
        Treats all SC cells as one batch - useful for small datasets and debugging.
        """
        from core_models_et_p2 import sample_sc_edm_single_patch
        
        result = sample_sc_edm_single_patch(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            context_encoder=self.context_encoder,
            score_net=self.score_net,
            target_st_p95=target_st_p95,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            guidance_scale=guidance_scale,
            eta=eta,
            device=self.device,
            DEBUG_FLAG=DEBUG_FLAG,
        )
        
        return result

    # ==========================================================================
    # STAGE C v2: ST-ONLY TRAINING WITH GRAPH-AWARE LOSSES
    # ==========================================================================
    
    def train_stageC_v2(
        self,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_min: int = 96,
        n_max: int = 384,
        num_st_samples: int = 4000,
        n_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-4,
        n_timesteps: int = 500,
        sigma_min: float = 0.01,
        sigma_max: float = 3.0,
        outf: str = 'output',
        fabric: Optional['Fabric'] = None,
        precision: str = '16-mixed',
        w_score: float = 1.0,
        w_edge: float = 1.0,
        w_repel: float = 0.1,
        enable_early_stop: bool = False,
        early_stop_patience: int = 10,
        early_stop_min_epochs: int = 20,
        early_stop_threshold: float = 0.01,
    ):
        """
        Stage C v2: ST-only training with graph-aware losses.
        
        Losses:
            1. Score loss (EDM-weighted denoising)
            2. kNN edge distance loss (index-aware geometry)
            3. Repulsion loss (prevent collapse)
        """
        from core_models_et_p2_v2 import train_stageC_st_only
        from core_models_et_p1 import STSetDataset
        
        # Create ST dataset
        st_dataset = STSetDataset(
            targets_dict=self.targets_dict,
            encoder=self.encoder,
            st_gene_expr_dict=st_gene_expr_dict,
            n_min=n_min,
            n_max=n_max,
            D_latent=self.D_latent,
            num_samples=num_st_samples,
            # knn_k=self.cfg.get('knn_k', 12),
            knn_k=self.cfg['denoiser']['knn_k'],
            device=self.device,
            landmarks_L=self.cfg.get('landmarks_L', 0),
        )
        
        history = train_stageC_st_only(
            context_encoder=self.context_encoder,
            generator=self.generator,
            score_net=self.score_net,
            st_dataset=st_dataset,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_timesteps=n_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=self.device,
            outf=outf,
            fabric=fabric,
            precision=precision,
            w_score=w_score,
            w_edge=w_edge,
            w_repel=w_repel,
            enable_early_stop=enable_early_stop,
            early_stop_patience=early_stop_patience,
            early_stop_min_epochs=early_stop_min_epochs,
            early_stop_threshold=early_stop_threshold,
        )
        
        return history
    
    # ==========================================================================
    # SC ENCODER FINE-TUNING v2
    # ==========================================================================
    
    def finetune_encoder_on_sc_v2(
        self,
        sc_gene_expr: torch.Tensor,
        D_st_reference: torch.Tensor,
        n_min: int = 96,
        n_max: int = 384,
        num_sc_samples: int = 5000,
        n_epochs: int = 50,
        batch_size: int = 8,
        lr: float = 3e-5,
        outf: str = 'output',
        fabric: Optional['Fabric'] = None,
        precision: str = '16-mixed',
        w_overlap: float = 1.0,
        w_geom: float = 1.0,
        w_dim: float = 0.1,
    ):
        """
        SC Encoder Fine-tuning with Frozen Geometry Prior.
        
        Generator and score_net weights are FROZEN but used as differentiable 
        functions so gradients flow through them into encoder/context_encoder.
        """
        from core_models_et_p2_v2 import finetune_encoder_on_sc
        from core_models_et_p1 import SCSetDataset
        
        # Create SC dataset
        sc_dataset = SCSetDataset(
            sc_gene_expr=sc_gene_expr.cpu(),
            encoder=self.encoder,
            n_min=n_min,
            n_max=n_max,
            overlap_min=20,
            overlap_max=128,
            num_samples=num_sc_samples,
            K_nbrs=2048,
            device=self.device,
            landmarks_L=0,
        )
        
        history = finetune_encoder_on_sc(
            encoder=self.encoder,
            context_encoder=self.context_encoder,
            generator=self.generator,
            score_net=self.score_net,
            sc_gene_expr=sc_gene_expr,
            sc_dataset=sc_dataset,
            D_st_reference=D_st_reference,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=self.device,
            outf=outf,
            fabric=fabric,
            precision=precision,
            w_overlap=w_overlap,
            w_geom=w_geom,
            w_dim=w_dim,
        )
        
        return history




