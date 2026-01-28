"""
GEMS (Generative Euclidean Metric Synthesis) - Orchestrator
Part 3: Main GEMS model class with complete training pipeline
"""

import torch
import torch.nn as nn
import os
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

from core_models_et_p1 import (
    SharedEncoder, train_encoder, STStageBPrecomputer, 
    STSetDataset, STTargets
)

# from core_models_et_p2 import (
#     SetEncoderContext, MetricSetGenerator, DiffusionScoreNet,
#     train_stageC_diffusion_generator
# )

from core_models_et_p2 import (
    SetEncoderContext, MetricSetGenerator, DiffusionScoreNet
)
from core_models_et_p2 import train_stageC_diffusion_generator

import utils_et as uet
import numpy as np


# ==============================================================================
# HELPER: AUTO-DETECT ANCHOR MODE FROM CHECKPOINT
# ==============================================================================

def infer_anchor_train_from_checkpoint(checkpoint: dict, base_h_dim: int = 128) -> bool:
    """
    Auto-detect whether a checkpoint was trained with anchor_train=True.
    
    Inspects the context_encoder's input_proj.weight shape:
    - If shape[1] == base_h_dim + 1: anchor_train=True
    - If shape[1] == base_h_dim: anchor_train=False
    
    Args:
        checkpoint: Loaded checkpoint dict (must have 'context_encoder' key)
        base_h_dim: Base embedding dimension (default 128 for [512,256,128] encoder)
    
    Returns:
        bool: True if checkpoint was trained with anchored mode
    
    Raises:
        ValueError: If input dimension doesn't match expected values
    """
    if 'context_encoder' not in checkpoint:
        print("[ANCHOR-DETECT] No context_encoder in checkpoint, assuming anchor_train=False")
        return False
    
    ctx_sd = checkpoint['context_encoder']
    
    if 'input_proj.weight' not in ctx_sd:
        print("[ANCHOR-DETECT] No input_proj.weight in context_encoder, assuming anchor_train=False")
        return False
    
    input_proj_weight = ctx_sd['input_proj.weight']
    ckpt_in_features = input_proj_weight.shape[1]
    
    if ckpt_in_features == base_h_dim + 1:
        print(f"[ANCHOR-DETECT] context_encoder.input_proj.in_features={ckpt_in_features} "
              f"(base_h_dim={base_h_dim} + 1) => anchor_train=True")
        return True
    elif ckpt_in_features == base_h_dim:
        print(f"[ANCHOR-DETECT] context_encoder.input_proj.in_features={ckpt_in_features} "
              f"=> anchor_train=False")
        return False
    else:
        raise ValueError(
            f"[ANCHOR-DETECT] Unexpected input_proj.in_features={ckpt_in_features}. "
            f"Expected {base_h_dim} (unanchored) or {base_h_dim + 1} (anchored). "
            f"Check your encoder dimensions."
        )

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
        landmarks_L: int = 32,
        # ========== NEW: Anchored training params ==========
        anchor_train: bool = False,
        # ========== ANCHOR GEOMETRY LOSSES ==========
        anchor_geom_losses: bool = True,
        anchor_geom_mode: str = "clamp_only",
        anchor_geom_min_unknown: int = 8,
        anchor_geom_debug_every: int = 200,
        # ---- Resume Stage C ----
        resume_ckpt_path: Optional[str] = None,
        resume_reset_optimizer: bool = False,
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
            },
            'anchor': {
                'anchor_train': anchor_train,
                'anchor_geom_losses': anchor_geom_losses,
                'anchor_geom_mode': anchor_geom_mode,
                'anchor_geom_min_unknown': anchor_geom_min_unknown,
                'anchor_geom_debug_every': anchor_geom_debug_every,
            }

        }
        
        # Store anchor_train for later use
        self.anchor_train = anchor_train
        self.anchor_geom_losses = anchor_geom_losses
        self.anchor_geom_mode = anchor_geom_mode
        self.anchor_geom_min_unknown = anchor_geom_min_unknown
        self.anchor_geom_debug_every = anchor_geom_debug_every



        # EMA copies (will be populated if loaded from checkpoint or after training)
        self.score_net_ema = None
        self.context_encoder_ema = None

        # Stage A: Shared encoder
        self.encoder = SharedEncoder(n_genes, n_embedding).to(device)
        
        # Stage B: Precomputer
        self.precomputer = STStageBPrecomputer(device=device)
        self.targets_dict = {}
        
        # Stage C: Generator components
        h_dim = n_embedding[-1]
        self.context_encoder = SetEncoderContext(
            h_dim=h_dim, c_dim=c_dim, n_heads=n_heads, 
            n_blocks=3, isab_m=isab_m,
            anchor_train=anchor_train,  # NEW: pass anchor flag
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
    # ==========================================================================
    # STAGE A: ENCODER TRAINING (FIXED VICReg+GRL CONFIG)
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
        outf: str = 'output',
    ):
        """
        Train shared encoder (Stage A) using fixed VICReg + adversary config.
        This is now the only supported Stage A path.
        """
        print("\n" + "="*60)
        print("STAGE A: Training Shared Encoder (VICReg + GRL)")
        print("="*60)

        encoder_vicreg = SharedEncoder(
            n_genes=st_gene_expr.shape[1],
            n_embedding=[512, 256, 128],
            dropout=0.1
        ).to(self.device)

        encoder_vicreg, projector, discriminator, hist = train_encoder(
            model=encoder_vicreg,
            st_gene_expr=st_gene_expr,
            st_coords=st_coords,
            sc_gene_expr=sc_gene_expr,
            slide_ids=slide_ids,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=self.device,
            outf=outf,
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
            adv_warmup_epochs=50,
            adv_ramp_epochs=200,
            grl_alpha_max=1.0,
            disc_hidden=256,
            disc_dropout=0.1,
            # Balanced domain sampling
            stageA_balanced_slides=True,
            # Adversary representation
            adv_representation_mode='clean',
            adv_use_layernorm=True,
            adv_log_diagnostics=True,
            adv_log_grad_norms=False,
            # Local alignment
            use_local_align=True,
            local_align_bidirectional=True,
            local_align_weight=4.0,
            local_align_tau_z=0.07,
            return_aux=True
        )

        self.encoder = encoder_vicreg

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
        # Early stopping (legacy, non-curriculum)
        enable_early_stop: bool = True,
        early_stop_min_epochs: int = 12,
        early_stop_patience: int = 6,
        early_stop_threshold: float = 0.01,
        # Curriculum early stopping (three-gate)
        curriculum_target_stage: int = 6,
        curriculum_min_epochs: int = 100,
        curriculum_early_stop: bool = True,
        phase_name: str = "Mixed",  # "ST-only" or "Fine-tune" or "Mixed"
        # NEW: Stochastic kNN sampling parameters (for STSetDataset)
        # NOTE: For small slides (<1000 spots), use 1.5-2.0 to maintain spatial locality
        pool_mult: float = 2.0,
        stochastic_tau: float = 1.0,
        # NEW: Context augmentation parameters
        z_noise_std: float = 0.02,
        z_dropout_rate: float = 0.1,
        aug_prob: float = 0.5,
        # ========== COMPETITOR TRAINING (ChatGPT hypothesis test) ==========
        compete_train: bool = False,
        compete_n_extra: int = 128,
        compete_n_rand: int = 64,
        compete_n_hard: int = 64,
        compete_use_pos_closure: bool = True,
        compete_k_pos: int = 10,
        compete_expr_knn_k: int = 50,
        compete_anchor_only: bool = True,
        compete_diag_every: int = 200,
        # ========== NEW: Anchored training params ==========
        anchor_train: bool = None,  # None = use model default
        anchor_p_uncond: float = 0.50,
        anchor_frac_min: float = 0.10,
        anchor_frac_max: float = 0.30,
        anchor_min: int = 8,
        anchor_max: int = 96,
        anchor_mode: str = "ball",
        anchor_exclude_landmarks: bool = True,
        anchor_clamp_clean: bool = True,
        anchor_mask_score_loss: bool = True,
        anchor_pointweight_nca: bool = True,
        anchor_debug_every: int = 200,
        anchor_warmup_steps: int = 0,
        # ========== ANCHOR GEOMETRY LOSSES ==========
        anchor_geom_losses: bool = None,  # None = use model default
        anchor_geom_mode: str = None,
        anchor_geom_min_unknown: int = None,
        anchor_geom_debug_every: int = None,
        # ---- Resume Stage C ----
        resume_stageC_ckpt: str = None,
        resume_reset_optimizer: bool = False,
        # ========== CONTEXT REPLACEMENT INVARIANCE ==========
        ctx_replace_variant: str = 'permute',
        ctx_loss_weight: float = 0.0,
        ctx_replace_p: float = 0.5,
        ctx_snr_thresh: float = 0.3,
        ctx_warmup_steps: int = 1000,
        ctx_debug_every: int = 100,
        # ========== SELF-CONDITIONING MODE ==========
        self_cond_mode: str = 'standard',
        # ========== PAIRED OVERLAP TRAINING (Candidate 1) ==========
        train_pair_overlap: bool = False,
        pair_overlap_alpha: float = 0.5,
        pair_overlap_min_I: int = 16,
        overlap_loss_weight_shape: float = 1.0,
        overlap_loss_weight_scale: float = 0.5,
        overlap_loss_weight_kl: float = 1.0,
        overlap_kl_tau: float = 0.5,
        overlap_sigma_thresh: float = 0.5,
        disable_ctx_loss_when_overlap: bool = True,
        overlap_debug_every: int = 100,
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
        # Use model's anchor_train if not explicitly provided
        effective_anchor_train = anchor_train if anchor_train is not None else self.anchor_train

        # Use model defaults for anchor_geom params if not explicitly provided
        effective_anchor_geom_losses = anchor_geom_losses if anchor_geom_losses is not None else self.anchor_geom_losses
        effective_anchor_geom_mode = anchor_geom_mode if anchor_geom_mode is not None else self.anchor_geom_mode
        effective_anchor_geom_min_unknown = anchor_geom_min_unknown if anchor_geom_min_unknown is not None else self.anchor_geom_min_unknown
        effective_anchor_geom_debug_every = anchor_geom_debug_every if anchor_geom_debug_every is not None else self.anchor_geom_debug_every


        
        # ========== VALIDATION: Ensure anchor_train matches model configuration ==========
        if effective_anchor_train and not self.anchor_train:
            raise ValueError(
                f"Cannot enable anchor_train in train_stageC when model was constructed with "
                f"anchor_train=False. The context encoder expects input dim={self.context_encoder.input_dim} "
                f"but anchored training requires input dim={self.context_encoder.input_dim + 1}. "
                f"Reconstruct the model with anchor_train=True."
            )

        
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
                landmarks_L=self.cfg['dataset']['landmarks_L'],
                pool_mult=pool_mult,
                stochastic_tau=stochastic_tau,
                # ========== COMPETITOR TRAINING PARAMS ==========
                compete_train=compete_train,
                compete_n_extra=compete_n_extra,
                compete_n_rand=compete_n_rand,
                compete_n_hard=compete_n_hard,
                compete_use_pos_closure=compete_use_pos_closure,
                compete_k_pos=compete_k_pos,
                compete_expr_knn_k=compete_expr_knn_k,
                compete_anchor_only=compete_anchor_only,
                # ========== NEW: Anchored training params ==========
                anchor_train=effective_anchor_train,
                anchor_p_uncond=anchor_p_uncond,
                anchor_frac_min=anchor_frac_min,
                anchor_frac_max=anchor_frac_max,
                anchor_min=anchor_min,
                anchor_max=anchor_max,
                anchor_mode=anchor_mode,
                anchor_exclude_landmarks=anchor_exclude_landmarks,
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
            encoder=self.encoder,
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
            curriculum_target_stage=curriculum_target_stage,
            curriculum_min_epochs=curriculum_min_epochs,
            curriculum_early_stop=curriculum_early_stop,
            # NEW: Context augmentation
            z_noise_std=z_noise_std,
            z_dropout_rate=z_dropout_rate,
            aug_prob=aug_prob,
            # ========== COMPETITOR TRAINING PARAMS ==========
            compete_train=compete_train,
            compete_n_extra=compete_n_extra,
            compete_n_rand=compete_n_rand,
            compete_n_hard=compete_n_hard,
            compete_use_pos_closure=compete_use_pos_closure,
            compete_k_pos=compete_k_pos,
            compete_expr_knn_k=compete_expr_knn_k,
            compete_anchor_only=compete_anchor_only,
            compete_diag_every=compete_diag_every,
            # ========== NEW: Anchored training params ==========
            anchor_train=effective_anchor_train,
            anchor_p_uncond=anchor_p_uncond,
            anchor_clamp_clean=anchor_clamp_clean,
            anchor_mask_score_loss=anchor_mask_score_loss,
            anchor_pointweight_nca=anchor_pointweight_nca,
            anchor_debug_every=anchor_debug_every,
            anchor_warmup_steps=anchor_warmup_steps,
            # ========== ANCHOR GEOMETRY LOSSES ==========
            anchor_geom_losses=effective_anchor_geom_losses,
            anchor_geom_mode=effective_anchor_geom_mode,
            anchor_geom_min_unknown=anchor_geom_min_unknown,
            anchor_geom_debug_every=anchor_geom_debug_every,
            resume_ckpt_path=resume_stageC_ckpt,
            resume_reset_optimizer=resume_reset_optimizer,
            # ========== CONTEXT REPLACEMENT INVARIANCE ==========
            ctx_replace_variant=ctx_replace_variant,
            ctx_loss_weight=ctx_loss_weight,
            ctx_replace_p=ctx_replace_p,
            ctx_snr_thresh=ctx_snr_thresh,
            ctx_warmup_steps=ctx_warmup_steps,
            ctx_debug_every=ctx_debug_every,
            # ========== SELF-CONDITIONING MODE ==========
            self_cond_mode=self_cond_mode,
            # ========== PAIRED OVERLAP TRAINING (Candidate 1) ==========
            train_pair_overlap=train_pair_overlap,
            pair_overlap_alpha=pair_overlap_alpha,
            pair_overlap_min_I=pair_overlap_min_I,
            overlap_loss_weight_shape=overlap_loss_weight_shape,
            overlap_loss_weight_scale=overlap_loss_weight_scale,
            overlap_loss_weight_kl=overlap_loss_weight_kl,
            overlap_kl_tau=overlap_kl_tau,
            overlap_sigma_thresh=overlap_sigma_thresh,
            disable_ctx_loss_when_overlap=disable_ctx_loss_when_overlap,
            overlap_debug_every=overlap_debug_every,
        )


        # Store sigma_data for inference
        # if 'sigma_data' in history:
        #     self.sigma_data = history['sigma_data']

        # Store EDM parameters for inference
        self.sigma_data = history.get('sigma_data', 1.0)
        self.sigma_min = history.get('sigma_min', 0.002)
        self.sigma_max = history.get('sigma_max', 80.0)

        print(f"[EDM] Stored: sigma_data={self.sigma_data:.4f}, sigma_min={self.sigma_min:.6f}, sigma_max={self.sigma_max:.2f}")

        # Restore EMA models from training
        import copy
        if 'score_net_ema_state' in history:
            self.score_net_ema = copy.deepcopy(self.score_net).eval()
            for p in self.score_net_ema.parameters():
                p.requires_grad_(False)
            self.score_net_ema.load_state_dict(history['score_net_ema_state'])
            print("[EMA] Restored score_net_ema from training")
        
        if 'context_encoder_ema_state' in history:
            self.context_encoder_ema = copy.deepcopy(self.context_encoder).eval()
            for p in self.context_encoder_ema.parameters():
                p.requires_grad_(False)
            self.context_encoder_ema.load_state_dict(history['context_encoder_ema_state'])
            print("[EMA] Restored context_encoder_ema from training")

        
        print("Stage C complete.")

        return history
    

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'score_net': self.score_net.state_dict(),
            'cfg': self.cfg,
            'sigma_data': getattr(self, 'sigma_data', None),
            'sigma_min': getattr(self, 'sigma_min', None), 
            'sigma_max': getattr(self, 'sigma_max', None),
        }
        # Save EMA if available
        if self.score_net_ema is not None:
            checkpoint['score_net_ema'] = self.score_net_ema.state_dict()
        if self.context_encoder_ema is not None:
            checkpoint['context_encoder_ema'] = self.context_encoder_ema.state_dict()
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    
    def load(self, path: str, auto_detect_anchor: bool = True):
        """
        Load all model components with auto-detection of anchor mode.
        
        Args:
            path: Path to checkpoint file
            auto_detect_anchor: If True, auto-detect anchor_train from checkpoint
                               and rebuild context_encoder if needed
        """
        import copy
        checkpoint = torch.load(path, map_location=self.device)
        
        # ========== AUTO-DETECT ANCHOR MODE ==========
        if auto_detect_anchor:
            base_h_dim = self.n_embedding[-1]  # typically 128
            ckpt_anchor_train = infer_anchor_train_from_checkpoint(checkpoint, base_h_dim)
            
            # Check if we need to rebuild context_encoder
            if ckpt_anchor_train != self.anchor_train:
                print(f"[ANCHOR-LOAD] Checkpoint anchor_train={ckpt_anchor_train} != "
                      f"model anchor_train={self.anchor_train}")
                print(f"[ANCHOR-LOAD] Rebuilding context_encoder with anchor_train={ckpt_anchor_train}")
                
                # Rebuild context_encoder with correct anchor_train
                self.anchor_train = ckpt_anchor_train
                self.cfg['anchor']['anchor_train'] = ckpt_anchor_train
                
                h_dim = self.n_embedding[-1]
                self.context_encoder = SetEncoderContext(
                    h_dim=h_dim,
                    c_dim=self.c_dim,
                    n_heads=self.n_heads,
                    n_blocks=3,
                    isab_m=self.isab_m,
                    anchor_train=ckpt_anchor_train,
                ).to(self.device)
                
                print(f"[ANCHOR-LOAD] Rebuilt context_encoder: input_dim={self.context_encoder.input_dim}")
        
        # Now load state dicts
        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
        else:
            print("[LOAD] WARNING: encoder not found in checkpoint; load it separately.")
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.score_net.load_state_dict(checkpoint['score_net'])
        
        if 'sigma_data' in checkpoint:
            self.sigma_data = checkpoint['sigma_data']
        if 'sigma_min' in checkpoint:
            self.sigma_min = checkpoint['sigma_min']
        if 'sigma_max' in checkpoint:
            self.sigma_max = checkpoint['sigma_max']
        
        # Load config if available
        if 'cfg' in checkpoint:
            self.cfg = checkpoint['cfg']
        
        # Load EMA weights if available
        if 'score_net_ema' in checkpoint:
            self.score_net_ema = copy.deepcopy(self.score_net).eval()
            for p in self.score_net_ema.parameters():
                p.requires_grad_(False)
            self.score_net_ema.load_state_dict(checkpoint['score_net_ema'])
            print("  Loaded score_net_ema")
        
        if 'context_encoder_ema' in checkpoint:
            # Rebuild EMA context encoder with same anchor_train
            self.context_encoder_ema = copy.deepcopy(self.context_encoder).eval()
            for p in self.context_encoder_ema.parameters():
                p.requires_grad_(False)
            self.context_encoder_ema.load_state_dict(checkpoint['context_encoder_ema'])
            print("  Loaded context_encoder_ema")
            
        print(f"Model loaded from {path}")
        print(f"  anchor_train={self.anchor_train}")


    
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
    
    
    # def infer_sc_patchwise(
    #     self,
    #     sc_gene_expr: torch.Tensor,
    #     n_timesteps_sample: int = 160,
    #     sigma_min: float = None,
    #     sigma_max: float = None,
    #     return_coords: bool = True,
    #     patch_size: int = 384,
    #     coverage_per_cell: float = 4.0,
    #     n_align_iters: int = 10,
    #     eta: float = 0.0,
    #     guidance_scale: float = 3.0,
    #     debug_flag: bool = True,
    #     debug_every: int = 10,
    #     fixed_patch_graph: Optional[dict] = None,
    #     # --- DEBUG KNN ARGS ---
    #     gt_coords: Optional[torch.Tensor] = None,
    #     debug_knn: bool = False,
    #     debug_max_patches: int = 20,
    #     debug_k_list: Tuple[int, int] = (10, 20),
    #     debug_global_subset: int = 4096,
    #     debug_gap_k: int = 10,
    #     two_pass: bool = False,
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     SC inference using patch-based global alignment (no masked/frozen points).
    #     """
    #     from core_models_et_p2 import sample_sc_edm_patchwise

    #     # Prepare CORAL params if enabled
    #     coral_params = None
    #     if hasattr(self, 'coral_enabled') and self.coral_enabled:
    #         coral_params = {
    #             'mu_sc': self.coral_mu_sc,
    #             'mu_st': self.coral_mu_st,
    #             'A': self.coral_A,
    #             'B': self.coral_B,
    #         }
    #         print("[CORAL] Applying CORAL transformation during SC inference")

    #     # If you have a stored ST scale, you can pass it; otherwise None
    #     target_st_p95 = getattr(self, "target_st_p95", None)

    #     sigma_data = getattr(self, 'sigma_data', None)

    #     # Use model's EDM parameters if not provided
    #     if sigma_min is None:
    #         sigma_min = getattr(self, 'sigma_min', 0.002)
    #     if sigma_max is None:
    #         sigma_max = getattr(self, 'sigma_max', 80.0)
        
    #     print(f"[Inference] Using sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.2f}, sigma_data={sigma_data:.4f}")

    #     if sigma_data is None:
    #         raise ValueError("sigma_data not set - load from checkpoint or compute from data")



    #     # Use EMA weights for inference if available
    #     ctx_enc = self.context_encoder_ema if self.context_encoder_ema is not None else self.context_encoder
    #     sc_net = self.score_net_ema if self.score_net_ema is not None else self.score_net
        
    #     # Ensure EMA models are in eval mode
    #     ctx_enc.eval()
    #     sc_net.eval()
        
    #     res = sample_sc_edm_patchwise(
    #         sc_gene_expr=sc_gene_expr,
    #         encoder=self.encoder,
    #         context_encoder=ctx_enc,
    #         score_net=sc_net,
    #         generator=self.generator,  # NEW: pass generator for refinement mode
    #         target_st_p95=target_st_p95,
    #         n_timesteps_sample=n_timesteps_sample,
    #         sigma_min=sigma_min,
    #         sigma_max=sigma_max,
    #         sigma_data=sigma_data,
    #         guidance_scale=guidance_scale,
    #         eta=eta,
    #         device=self.device,
    #         patch_size=patch_size,
    #         coverage_per_cell=coverage_per_cell,
    #         n_align_iters=n_align_iters,
    #         return_coords=return_coords,
    #         DEBUG_FLAG=debug_flag,
    #         DEBUG_EVERY=debug_every,
    #         fixed_patch_graph=fixed_patch_graph,
    #         coral_params=coral_params,
    #         gt_coords=gt_coords,
    #         debug_knn=debug_knn,
    #         debug_max_patches=debug_max_patches,
    #         debug_k_list=debug_k_list,
    #         debug_global_subset=debug_global_subset,
    #         debug_gap_k=debug_gap_k,
    #         two_pass=two_pass,
    #     )
    #     return res

    def infer_sc_patchwise(
        self,
        sc_gene_expr: torch.Tensor,
        n_timesteps_sample: int = 160,
        sigma_min: float = None,
        sigma_max: float = None,
        return_coords: bool = True,
        patch_size: int = 384,
        coverage_per_cell: float = 4.0,
        n_align_iters: int = 10,
        eta: float = 0.0,
        guidance_scale: float = 3.0,
        debug_flag: bool = True,
        debug_every: int = 10,
        fixed_patch_graph: Optional[dict] = None,
        # --- DEBUG KNN ARGS ---
        gt_coords: Optional[torch.Tensor] = None,
        debug_knn: bool = False,
        debug_max_patches: int = 20,
        debug_k_list: Tuple[int, int] = (10, 20),
        debug_global_subset: int = 4096,
        debug_gap_k: int = 10,
        anchor_bit_only_diag: bool = False,
        anchor_bit_scale: float = 1.0,
        two_pass: bool = False,
        # --- ST-STYLE STOCHASTIC PATCH SAMPLING ---
        pool_mult: float = 2.0,
        stochastic_tau: float = 0.8,
        tau_mode: str = "adaptive_kth",
        ensure_connected: bool = True,
        # --- MERGE MODE ---
        # merge_mode: str = "mean",
        # --- ALIGNMENT CONSTRAINTS ---
        align_freeze_scale: bool = True,
        align_scale_clamp: Tuple[float, float] = (1.0, 1.0),
        # --- LOCAL REFINEMENT ---
        local_refine: bool = False,
        local_refine_steps: int = 100,
        local_refine_lr: float = 0.01,
        local_refine_anchor_weight: float = 0.1,
        # --- DGSO: Distance-Graph Stitch Optimization ---
        # --- DGSO-v2: Distance-Graph Stitch Optimization ---
        enable_dgso: bool = False,
        dgso_k_edge: int = 15,
        dgso_iters: int = 1000,
        dgso_lr: float = 1e-2,
        dgso_batch_size: int = 100000,
        dgso_huber_delta: float = 0.1,
        dgso_anchor_lambda: float = 1.0,  # CHANGED: full trust region
        dgso_log_every: int = 100,
        # --- DGSO-v2 NEW PARAMS ---
        dgso_m_min: int = 3,              # min measurements per edge to keep
        dgso_tau_spread: float = 0.30,    # max rel_spread to keep edge
        dgso_spread_penalty_alpha: float = 10.0,  # penalize high-spread edges
        dgso_dist_band: Tuple[float, float] = (0.05, 0.95),  # distance quantile band
        dgso_radius_lambda: float = 0.1,  # anti-collapse radius term weight
        dgso_two_phase: bool = True,
        dgso_phase1_iters: int = 200,
        dgso_phase1_anchor_mult: float = 10.0,
        # --- NEW DEBUG FLAGS (pass-through) ---
        debug_oracle_gt_stitch: bool = False,
        debug_incremental_stitch_curve: bool = False,
        debug_overlap_postcheck: bool = False,
        debug_cycle_closure: bool = False,
        debug_scale_compression: bool = False,
        # --- DEBUG/ABLATION FLAGS (pass-through) ---
        debug_gen_vs_noise: bool = False,
        ablate_use_generator_init: bool = False,
        ablate_use_pure_noise_init: bool = False,

    ) -> Dict[str, torch.Tensor]:


        """
        SC inference using patch-based global alignment (no masked/frozen points).
        """
        from core_models_et_p2 import sample_sc_edm_patchwise

        # Prepare CORAL params if enabled
        coral_params = None
        # if hasattr(self, 'coral_enabled') and self.coral_enabled:
        #     coral_params = {
        #         'mu_sc': self.coral_mu_sc,
        #         'mu_st': self.coral_mu_st,
        #         'A': self.coral_A,
        #         'B': self.coral_B,
        #     }
        #     print("[CORAL] Applying CORAL transformation during SC inference")

        # If you have a stored ST scale, you can pass it; otherwise None
        target_st_p95 = getattr(self, "target_st_p95", None)

        sigma_data = getattr(self, 'sigma_data', None)

        # Use model's EDM parameters if not provided
        if sigma_min is None:
            sigma_min = getattr(self, 'sigma_min', 0.002)
        if sigma_max is None:
            sigma_max = getattr(self, 'sigma_max', 80.0)
        
        print(f"[Inference] Using sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.2f}, sigma_data={sigma_data:.4f}")
        print(f"[DEBUG_TAG][INFER-SIGMA] sigma_min={sigma_min:.6f} sigma_max={sigma_max:.6f} sigma_data={sigma_data:.6f}")

        if sigma_data is None:
            raise ValueError("sigma_data not set - load from checkpoint or compute from data")



        # Use EMA weights for inference if available
        ctx_enc = self.context_encoder_ema if self.context_encoder_ema is not None else self.context_encoder
        sc_net = self.score_net_ema if self.score_net_ema is not None else self.score_net
        
        # Ensure EMA models are in eval mode
        ctx_enc.eval()
        sc_net.eval()
        
        res = sample_sc_edm_patchwise(
            sc_gene_expr=sc_gene_expr,
            encoder=self.encoder,
            context_encoder=ctx_enc,
            score_net=sc_net,
            generator=self.generator,  # NEW: pass generator for refinement mode
            target_st_p95=target_st_p95,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
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
            coral_params=coral_params,
            gt_coords=gt_coords,
            debug_knn=debug_knn,
            debug_max_patches=debug_max_patches,
            debug_k_list=debug_k_list,
            debug_global_subset=debug_global_subset,
            debug_gap_k=debug_gap_k,
            two_pass=two_pass,
            # --- ST-STYLE STOCHASTIC PATCH SAMPLING ---
            pool_mult=pool_mult,
            stochastic_tau=stochastic_tau,
            tau_mode=tau_mode,
            ensure_connected=ensure_connected,
            # --- MERGE MODE ---
            # merge_mode=merge_mode,
            # --- ALIGNMENT CONSTRAINTS ---
            align_freeze_scale=align_freeze_scale,
            align_scale_clamp=align_scale_clamp,
            # --- LOCAL REFINEMENT ---
            local_refine=local_refine,
            local_refine_steps=local_refine_steps,
            local_refine_lr=local_refine_lr,
            local_refine_anchor_weight=local_refine_anchor_weight,
            # --- DGSO params ---
            # --- DGSO-v2 params ---
            enable_dgso=enable_dgso,
            dgso_k_edge=dgso_k_edge,
            dgso_iters=dgso_iters,
            dgso_lr=dgso_lr,
            dgso_batch_size=dgso_batch_size,
            dgso_huber_delta=dgso_huber_delta,
            dgso_anchor_lambda=dgso_anchor_lambda,
            dgso_log_every=dgso_log_every,
            dgso_m_min=dgso_m_min,
            dgso_tau_spread=dgso_tau_spread,
            dgso_spread_penalty_alpha=dgso_spread_penalty_alpha,
            dgso_dist_band=dgso_dist_band,
            dgso_radius_lambda=dgso_radius_lambda,
            dgso_two_phase=dgso_two_phase,
            dgso_phase1_iters=dgso_phase1_iters,
            dgso_phase1_anchor_mult=dgso_phase1_anchor_mult,
            # --- ANCHORED SEQUENTIAL SAMPLING ---
            # --- NEW DEBUG FLAGS (pass-through) ---
            debug_oracle_gt_stitch=debug_oracle_gt_stitch,
            debug_incremental_stitch_curve=debug_incremental_stitch_curve,
            debug_overlap_postcheck=debug_overlap_postcheck,
            debug_cycle_closure=debug_cycle_closure,
            debug_scale_compression=debug_scale_compression,
            # --- DEBUG/ABLATION FLAGS (pass-through) ---
            debug_gen_vs_noise=debug_gen_vs_noise,
            ablate_use_generator_init=ablate_use_generator_init,
            ablate_use_pure_noise_init=ablate_use_pure_noise_init,
        )
        return res

    # ==========================================================================
    # CORAL TRANSFORMATION FOR SC INFERENCE
    # ==========================================================================

    def compute_coral_params_from_st(
        self,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_samples: int = 2000,
        n_min: int = 96,
        n_max: int = 384,
    ):
        """
        Compute ST context distribution statistics for CORAL.
        Sample ST mini-sets and encode them to get context distribution.
        
        Args:
            st_gene_expr_dict: Dict of ST gene expression per slide
            n_samples: Number of ST mini-sets to sample
            n_min, n_max: Mini-set size range
        """
        print("\n" + "="*70)
        print("COMPUTING CORAL PARAMETERS FROM ST DATA")
        print("="*70)
        
        from core_models_et_p1 import STSetDataset
        
        # Ensure on CPU
        st_gene_expr_dict_cpu = {
            k: v.cpu() if torch.is_tensor(v) else v 
            for k, v in st_gene_expr_dict.items()
        }
        
        # Create ST dataset
        st_dataset = STSetDataset(
            targets_dict=self.targets_dict,
            encoder=self.encoder,
            st_gene_expr_dict=st_gene_expr_dict_cpu,
            n_min=n_min,
            n_max=n_max,
            D_latent=self.D_latent,
            num_samples=n_samples,
            knn_k=12,
            device=self.device,
            landmarks_L=0,
        )
        
        # Collect ST context tokens
        st_contexts = []
        self.encoder.eval()
        self.context_encoder.eval()
        
        with torch.no_grad():
            for idx in range(len(st_dataset)):
                miniset = st_dataset[idx]
                n = miniset['n']
                
                # Get gene expression
                slide_id = miniset['overlap_info']['slide_id']
                indices = miniset['overlap_info']['indices'][:n]
                gene_expr = st_gene_expr_dict_cpu[slide_id][indices].to(self.device)
                
                # Encode
                Z = self.encoder(gene_expr)
                Z = F.layer_norm(Z, (Z.shape[1],))

                mask = torch.ones(n, dtype=torch.bool, device=self.device)
                
                # Context
                Z_batch = Z.unsqueeze(0)
                mask_batch = mask.unsqueeze(0)
                H = self.context_encoder(Z_batch, mask_batch)
                
                st_contexts.append(H.squeeze(0).cpu())
        
        # Concatenate all contexts
        st_H = torch.cat(st_contexts, dim=0)  # (N_st, c_dim)
        
        print(f"✓ Collected {st_H.shape[0]} ST context tokens")
        
        # Compute statistics
        mu_st = st_H.mean(dim=0).to(self.device)
        st_H_centered = st_H - mu_st.cpu()
        cov_st = (st_H_centered.T @ st_H_centered) / (st_H.shape[0] - 1)
        cov_st = cov_st.to(self.device)
        
        print(f"✓ mu_st shape: {mu_st.shape}")
        print(f"✓ cov_st shape: {cov_st.shape}")
        
        # Store
        self.coral_mu_st = mu_st
        self.coral_cov_st = cov_st
        
        print("✓ CORAL ST parameters computed and stored")
        print("="*70)

    def build_coral_transform(
        self,
        sc_gene_expr: torch.Tensor,
        n_samples: int = 2000,
        n_min: int = 96,
        n_max: int = 384,
        shrink: float = 0.01,
        eps: float = 1e-5,
    ):
        """
        Build CORAL transformation matrices from SC distribution.
        Must call compute_coral_params_from_st() first.
        
        Args:
            sc_gene_expr: SC gene expression tensor
            n_samples: Number of SC mini-sets to sample
            shrink: Covariance shrinkage factor
            eps: Numerical stability epsilon
        """
        if not hasattr(self, 'coral_mu_st') or not hasattr(self, 'coral_cov_st'):
            raise RuntimeError("Must call compute_coral_params_from_st() first")
        
        print("\n" + "="*70)
        print("BUILDING CORAL TRANSFORMATION FROM SC DATA")
        print("="*70)
        
        from core_models_et_p1 import SCSetDataset
        
        sc_gene_expr_cpu = sc_gene_expr.cpu() if torch.is_tensor(sc_gene_expr) else sc_gene_expr
        
        # Create SC dataset
        sc_dataset = SCSetDataset(
            sc_gene_expr=sc_gene_expr_cpu,
            encoder=self.encoder,
            n_min=n_min,
            n_max=n_max,
            num_samples=n_samples,
            device=self.device,
            landmarks_L=0,
        )
        
        # Collect SC context tokens
        sc_contexts = []
        self.encoder.eval()
        self.context_encoder.eval()
        
        with torch.no_grad():
            for idx in range(len(sc_dataset)):
                # miniset = sc_dataset[idx]
                # n_A = miniset['n_A']
                
                # # Get gene expression
                # indices_A = miniset['global_indices_A'][:n_A]
                # gene_expr = sc_gene_expr_cpu[indices_A].to(self.device)

                miniset = sc_dataset[idx]
                n = miniset['n']

                # Get gene expression
                indices = miniset['global_indices'][:n]
                gene_expr = sc_gene_expr[indices].to(self.device)

                
                # Encode
                Z = self.encoder(gene_expr)
                Z = F.layer_norm(Z, (Z.shape[1],))

                mask = torch.ones(n, dtype=torch.bool, device=self.device)
                
                # Context
                Z_batch = Z.unsqueeze(0)
                mask_batch = mask.unsqueeze(0)
                H = self.context_encoder(Z_batch, mask_batch)
                
                sc_contexts.append(H.squeeze(0).cpu())
        
        # Concatenate all contexts
        sc_H = torch.cat(sc_contexts, dim=0)  # (N_sc, c_dim)
        
        print(f"✓ Collected {sc_H.shape[0]} SC context tokens")
        
        # Compute SC statistics
        mu_sc = sc_H.mean(dim=0).to(self.device)
        sc_H_centered = sc_H - mu_sc.cpu()
        cov_sc = (sc_H_centered.T @ sc_H_centered) / (sc_H.shape[0] - 1)
        cov_sc = cov_sc.to(self.device)
        
        print(f"✓ mu_sc shape: {mu_sc.shape}")
        print(f"✓ cov_sc shape: {cov_sc.shape}")
        
        # Build transform with shrinkage
        D = cov_sc.shape[0]
        I = torch.eye(D, device=self.device, dtype=torch.float32)
        
        cov_sc_shrunk = (1 - shrink) * cov_sc + shrink * I
        cov_st_shrunk = (1 - shrink) * self.coral_cov_st + shrink * I
        
        # Compute A = C_sc^{-1/2} and B = C_st^{1/2}
        def sqrtm_psd(C, eps):
            evals, evecs = torch.linalg.eigh(C)
            evals = torch.clamp(evals, min=eps)
            return (evecs * torch.sqrt(evals)) @ evecs.T
        
        def invsqrtm_psd(C, eps):
            evals, evecs = torch.linalg.eigh(C)
            evals = torch.clamp(evals, min=eps)
            return (evecs * (1.0 / torch.sqrt(evals))) @ evecs.T
        
        A = invsqrtm_psd(cov_sc_shrunk, eps=eps)  # C_sc^{-1/2}
        B = sqrtm_psd(cov_st_shrunk, eps=eps)      # C_st^{1/2}
        
        # Store transform
        self.coral_mu_sc = mu_sc
        self.coral_A = A
        self.coral_B = B
        self.coral_enabled = True
        
        print(f"✓ A (C_sc^{{-1/2}}) shape: {A.shape}")
        print(f"✓ B (C_st^{{1/2}}) shape: {B.shape}")
        print("✓ CORAL transformation matrices computed")
        print("="*70)

    @staticmethod
    def apply_coral_transform(H, mu_sc, A, B, mu_st):
        """
        Apply CORAL transformation: (H - mu_sc) @ A @ B + mu_st
        
        Args:
            H: (B, N, D) or (B, D) context tensor
            mu_sc, A, B, mu_st: CORAL parameters
        
        Returns:
            H_transformed: Same shape as H
        """
        if H.dim() == 2:
            # (B, D)
            H_centered = H - mu_sc
            return H_centered @ A @ B + mu_st
        elif H.dim() == 3:
            # (B, N, D)
            B_size, N, D = H.shape
            H_flat = H.reshape(-1, D)
            H_centered = H_flat - mu_sc
            H_transformed_flat = H_centered @ A @ B + mu_st
            return H_transformed_flat.reshape(B_size, N, D)
        else:
            raise ValueError(f"Unsupported H shape: {H.shape}")


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
