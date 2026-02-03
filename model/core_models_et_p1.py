"""
GEMS (Generative Euclidean Metric Synthesis) - Core Model Classes
Coordinate-free supervision for spatial transcriptomics reconstruction

Part 1: Encoder, Stage B Precomputation, Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from ssl_utils import (
    FullGatherLayer, batch_all_gather, GradientReversalFunction, grad_reverse,
    off_diagonal, VICRegLoss, SlideDiscriminator, augment_expression,
    sample_balanced_slide_indices, sample_balanced_domain_and_slide_indices,
    grl_alpha_schedule, coral_loss,
    # NEW: diagnostic utilities (GPT 5.2 Pro fix)
    compute_confusion_matrix, compute_entropy, compute_gradient_norms,
    augment_expression_mild, get_adversary_representation, log_adversary_diagnostics
)
import math


# Import from project knowledge Set-Transformer components
from modules import MAB, SAB, ISAB, PMA
import utils_et as uet


def _cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


# ==============================================================================
# STAGE A: SHARED ENCODER
# ==============================================================================

class SharedEncoder(nn.Module):
    """
    Shared encoder for ST and SC gene expression.
    Aligns both modalities into a joint embedding space.
    
    Architecture: MLP [n_genes] → [512, 256, 128]
    Losses: RBF adjacency (with block-diag mask for multi-slide),
            circular coupling, MMD
    
    DO NOT MODIFY - Critical for continuity with past runs.
    """
    
    def __init__(self, n_genes: int, n_embedding: List[int] = [512, 256, 128], dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.n_embedding = n_embedding
        
        # Build MLP encoder
        layers = []
        prev_dim = n_genes
        for i, dim in enumerate(n_embedding):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(n_embedding) - 1:  # No activation on last layer
                layers.append(nn.LayerNorm(dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, n_genes) gene expression
            
        Returns:
            Z: (batch, n_embedding[-1]) embeddings
        """
        return self.encoder(X)


def build_vicreg_projector(
    input_dim: int,
    project_dim: int = 256,
    hidden_dim: int = 512
) -> nn.Module:
    """
    Build projector for VICReg (MLP with BN).
    
    Args:
        input_dim: backbone output dimension
        project_dim: final projection dimension
        hidden_dim: hidden layer dimension
        
    Returns:
        projector: nn.Sequential module
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, project_dim, bias=False)
    )


def train_encoder(
    model: SharedEncoder,
    st_gene_expr: torch.Tensor,
    st_coords: torch.Tensor,
    sc_gene_expr: torch.Tensor,
    slide_ids: Optional[torch.Tensor] = None,
    sc_slide_ids: Optional[torch.Tensor] = None,  # NEW: Per-SC-slide IDs for balanced sampling
    sc_patient_ids: Optional[torch.Tensor] = None,  # NEW: Per-SC-sample patient IDs for patient CORAL
    # ========== NEW: N-class Source Adversary ==========
    st_source_ids: Optional[torch.Tensor] = None,  # (n_st,) Source IDs for ST samples (e.g., 0=ST1, 1=ST2)
    sc_source_ids: Optional[torch.Tensor] = None,  # (n_sc,) Source IDs for SC samples (e.g., 2=ST3, 3=SC_P2, 4=ST3_P5, 5=SC_P5)
    use_source_adversary: bool = False,  # True = N-class source adversary, False = 2-class domain adversary
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    sigma: Optional[float] = None,
    alpha: float = 0.9,
    ratio_start: float = 0.0,
    ratio_end: float = 1.0,
    mmdbatch: float = 0.1,
    device: str = 'cuda',
    outf: str = 'output',
    # NEW: Local miniset sampling (Stage A/C consistency)
    local_miniset_mode: bool = False,
    n_min: int = 128,
    n_max: int = 384,
    pool_mult: float = 4.0,
    stochastic_tau: float = 1.0,
    # NEW: Slide-invariance regularization
    slide_align_mode: str = 'none',  # 'none', 'coral', 'mmd', 'infonce'
    slide_align_weight: float = 1.0,
    # NEW: InfoNCE parameters
    infonce_tau: float = 0.07,
    infonce_match: str = 'expr',  # 'expr' or 'embed'
    infonce_topk: int = 1,
    infonce_sym: bool = True,
    # NEW: MNN + alignment scheduling
    infonce_smin: float = 0.75,   # cosine sim threshold for MNN pairs
    align_warmup: int = 200,      # epochs before alignment starts
    align_ramp: int = 200,        # epochs to ramp alignment weight
    # NEW: Optional SC-related losses
    use_circle: bool = True,
    use_mmd_sc: bool = True,
    # ========== NEW: VICReg + Adversary mode ==========
    stageA_obj: str = 'geom',  # 'geom' (default) or 'vicreg_adv'
    # VICReg parameters
    vicreg_lambda_inv: float = 25.0,
    vicreg_lambda_var: float = 25.0,
    vicreg_lambda_cov: float = 1.0,
    vicreg_gamma: float = 1.0,
    vicreg_eps: float = 1e-4,
    vicreg_project_dim: int = 256,
    vicreg_use_projector: bool = True,
    vicreg_float32_stats: bool = True,
    vicreg_ddp_gather: bool = True,
    # Expression augmentation parameters
    aug_gene_dropout: float = 0.2,
    aug_gauss_std: float = 0.01,
    aug_scale_jitter: float = 0.2,
    # Slide adversary parameters
    adv_slide_weight: float = 50.0,
    patient_coral_weight: float = 10.0,  # Weight for patient-level CORAL (cross-patient alignment)
    source_coral_weight: float = 50.0,  # Weight for pairwise source CORAL (align ALL sources)
    mmd_weight: float = 20.0,  # ST↔SC MMD weight (you used 20)
    mmd_use_l2norm: bool = True,  # L2-normalize only for MMD
    mmd_ramp: bool = True,  # ramp MMD with CORAL schedule
    adv_warmup_epochs: int = 50,
    adv_ramp_epochs: int = 200,
    grl_alpha_max: float = 1.0,
    disc_hidden: int = 256,
    disc_dropout: float = 0.1,
    # Balanced slide batching
    # Balanced slide batching
    stageA_balanced_slides: bool = True,
    inference_dropout_prob: float = 0.5,  # NEW: randomly drop 1 inference source each batch
    return_aux: bool = False,
    # ========== NEW: Adversary representation control (GPT 5.2 Pro fix) ==========
    # These parameters control what representation the discriminator sees
    adv_representation_mode: str = 'clean',  # 'clean', 'mild_aug', 'ln_avg_aug'
    adv_use_layernorm: bool = False,  # Set False for Run 1+ (remove LN mismatch)
    adv_mild_dropout: float = 0.05,   # For 'mild_aug' mode
    adv_mild_noise: float = 0.005,    # For 'mild_aug' mode
    adv_mild_jitter: float = 0.05,    # For 'mild_aug' mode
    adv_log_diagnostics: bool = True, # Enable comprehensive discriminator logging
    adv_log_grad_norms: bool = False, # Log gradient norms (expensive, for debugging)
    # ========== NEW: Local Alignment Loss (Run 4) ==========
    use_local_align: bool = False,
    local_align_weight: float = 1.0,
    local_align_tau_x: float = 0.1,
    local_align_tau_z: float = 0.1,
    local_align_bidirectional: bool = True,
    local_align_warmup: int = 100,  # Start after discriminator warmup
    # ========== Reproducibility ==========
    seed: Optional[int] = None,  # Random seed for reproducibility
    # ========== Best checkpoint ==========
    use_best_checkpoint: bool = True,  # Return best model (by alignment) instead of final
):

    """
    Train the shared encoder (Stage A).

    Losses:
    1. RBF adjacency prediction (with block-diagonal mask for multi-slide)
    2. Circular coupling (ST→SC→ST)
    3. MMD between ST and SC embeddings

    Args:
        model: SharedEncoder instance
        st_gene_expr: (n_st, n_genes) ST expression
        st_coords: (n_st, 2) ST coordinates
        sc_gene_expr: (n_sc, n_genes) SC expression
        slide_ids: (n_st,) slide identifiers (default: all zeros)
        sc_slide_ids: (n_sc,) SC slide identifiers for per-SC-slide balancing (default: None = all pooled)
        n_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        sigma: RBF bandwidth (auto-computed if None)
        alpha: circular coupling weight
        mmdbatch: MMD batch fraction
        device: torch device
        outf: output directory
        seed: random seed for reproducibility (sets torch, numpy, python random)
    """
    # ========== Set random seeds for reproducibility ==========
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For full determinism (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Stage A] Random seed set to {seed} for reproducibility")
        print(f"  NOTE: For full reproducibility, also call ssl_utils.set_seed({seed})")
        print(f"        BEFORE creating the encoder to seed weight initialization.")

    model = model.to(device)
    model.train()

    st_gene_expr = st_gene_expr.to(device)
    st_coords = st_coords.to(device)
    sc_gene_expr = sc_gene_expr.to(device)
    if slide_ids is not None:
        slide_ids = slide_ids.to(device)
    if sc_slide_ids is not None:
        sc_slide_ids = sc_slide_ids.to(device)
    if sc_patient_ids is not None:
        sc_patient_ids = sc_patient_ids.to(device)
    # NEW: Source adversary setup
    if st_source_ids is not None:
        st_source_ids = st_source_ids.to(device)
    if sc_source_ids is not None:
        sc_source_ids = sc_source_ids.to(device)

    # Normalize ST coordinates (pose-invariant)
    # st_coords_norm, center, radius = uet.normalize_coordinates_isotropic(st_coords)
    # NEW: Coords are already normalized from run_mouse_brain_2.py
    st_coords_norm = st_coords
    print("[Stage A] Using pre-normalized coordinates")

    # Auto-compute sigma if not provided
    # if stageA_obj != 'vicreg_adv':
    if sigma is None:
        with torch.no_grad():
            sigmas_per_slide = []
            unique_slides = torch.unique(slide_ids)

            for sid in unique_slides:
                mask_slide = (slide_ids == sid)
                coords_slide = st_coords_norm[mask_slide]
                n_spots = coords_slide.shape[0]

                # Within-slide distances only
                D_slide = torch.cdist(coords_slide, coords_slide)
                k = min(15, n_spots - 1)
                kth_dists = torch.kthvalue(D_slide, k+1, dim=1).values
                sigma_slide = torch.median(kth_dists).item()
                sigmas_per_slide.append(sigma_slide)

            # Aggregate across slides
            sigma = float(np.median(sigmas_per_slide)) * 0.8
            print(f"Auto-computed sigma (per-slide median) = {sigma:.4f}")
            print(f"  Per-slide sigmas: {sigmas_per_slide}")

    # Compute RBF adjacency target from normalized coordinates
    with torch.no_grad():
        D_norm = torch.cdist(st_coords_norm, st_coords_norm)
        A_target = torch.exp(-D_norm**2 / (2 * sigma**2))
        A_target = F.normalize(A_target, p=1, dim=1)  # Row-normalize

    # Create slide mask (block-diagonal for multi-slide)
    if slide_ids is None:
        slide_ids = torch.zeros(st_gene_expr.shape[0], dtype=torch.long, device=device)
    else:
        slide_ids = slide_ids.to(device)

    slide_mask = uet.block_diag_mask(slide_ids)  # (n_st, n_st)

    # Optimizer
    # ========== VICReg Mode Setup ==========
    if stageA_obj == 'vicreg_adv':
        from ssl_utils import (
            VICRegLoss, SlideDiscriminator, grad_reverse,
            augment_expression, sample_balanced_slide_indices,
            sample_balanced_source_indices,  # NEW: for N-class source adversary
            grl_alpha_schedule, coral_loss, compute_local_alignment_loss,
            mmd_rbf_loss
        )


        print("\n" + "="*70)
        print("STAGE A MODE: VICReg + Domain Adversary (ST vs SC)")
        print("="*70 + "\n")

        # ---- (optional) keep your ST slide-id remap for future use ----
        unique_st_slides = torch.unique(slide_ids)
        n_st_slides = len(unique_st_slides)
        if not torch.equal(unique_st_slides, torch.arange(n_st_slides, device=device)):
            print(f"[VICReg] Remapping ST slide_ids from {unique_st_slides.tolist()} to 0..{n_st_slides-1}")
            slide_id_map = {int(s.item()): i for i, s in enumerate(unique_st_slides)}
            slide_ids = torch.tensor([slide_id_map[int(s.item())] for s in slide_ids],
                                    dtype=torch.long, device=device)
        else:
            print(f"[VICReg] ST slide_ids already contiguous: 0..{n_st_slides-1}")

        # ---- Domain/Source labels setup ----
        n_st = st_gene_expr.shape[0]
        n_sc = sc_gene_expr.shape[0]

        X_ssl = torch.cat([st_gene_expr, sc_gene_expr], dim=0)

        # ========== NEW: N-class Source Adversary ==========
        if use_source_adversary and st_source_ids is not None and sc_source_ids is not None:
            # Combine source IDs from ST and SC
            all_source_ids = torch.cat([st_source_ids, sc_source_ids], dim=0).to(device)
            unique_sources = torch.unique(all_source_ids)
            n_sources = len(unique_sources)

            # Remap to contiguous 0..n_sources-1 if needed
            if not torch.equal(unique_sources.sort().values, torch.arange(n_sources, device=device)):
                print(f"[VICReg] Remapping source_ids from {unique_sources.tolist()} to 0..{n_sources-1}")
                source_id_map = {int(s.item()): i for i, s in enumerate(unique_sources.sort().values)}
                all_source_ids = torch.tensor([source_id_map[int(s.item())] for s in all_source_ids],
                                              dtype=torch.long, device=device)

            # Use source_ids for adversary instead of domain_ids
            domain_ids = all_source_ids  # Reuse variable name for minimal code changes
            n_domains = n_sources

            print(f"\n[VICReg] SOURCE ADVERSARY MODE: {n_sources} sources")
            # Count samples per source
            for src_id in range(n_sources):
                n_src = (domain_ids == src_id).sum().item()
                print(f"  Source {src_id}: {n_src} samples")
        else:
            # Original: Binary domain labels (ST=0, SC=1)
            domain_ids = torch.cat([
                torch.zeros(n_st, device=device, dtype=torch.long),
                torch.ones(n_sc, device=device, dtype=torch.long),
            ], dim=0)
            n_domains = 2
            print(f"[VICReg] Domain setup: ST(n={n_st}) vs SC(n={n_sc}) => {n_domains} domains")

        ST_LABEL = 0  # Keep for CORAL (first n_st samples are always "ST-like")
        SC_LABEL = 1  # Keep for CORAL

        if batch_size % n_domains != 0:
            print(f"[WARNING] batch_size={batch_size} not divisible by {n_domains} "
                f"(recommend even batch_size for perfectly balanced sampling)")

        # Report slide balancing status
        n_st_slides = len(torch.unique(slide_ids)) if slide_ids is not None else 1
        print(f"[VICReg] ST slide balancing: {n_st_slides} slides")
        if sc_slide_ids is not None:
            n_sc_slides = len(torch.unique(sc_slide_ids))
            print(f"[VICReg] SC slide balancing: {n_sc_slides} slides (hierarchical sampling enabled)")
        else:
            print(f"[VICReg] SC slide balancing: disabled (all SC pooled)")

        # Report patient CORAL status
        if sc_patient_ids is not None:
            n_patients = len(torch.unique(sc_patient_ids))
            if n_patients > 1:
                print(f"[VICReg] Patient CORAL: enabled ({n_patients} patients in SC)")
            else:
                print(f"[VICReg] Patient CORAL: disabled (only 1 patient in SC)")
        else:
            print(f"[VICReg] Patient CORAL: disabled (sc_patient_ids not provided)")

        # Build projector
        h_dim = model.n_embedding[-1]
        if vicreg_use_projector:
            projector = build_vicreg_projector(h_dim, vicreg_project_dim).to(device)
            print(f"  Projector: {h_dim} → {vicreg_project_dim}")
        else:
            projector = None
            print("  No projector (VICReg on backbone)")

        # Build DOMAIN/SOURCE discriminator
        discriminator = SlideDiscriminator(
            h_dim, n_domains, disc_hidden, disc_dropout
        ).to(device)
        if use_source_adversary and st_source_ids is not None:
            print(f"  Discriminator: {h_dim} → {n_domains} (source adversary)")
        else:
            print(f"  Discriminator: {h_dim} → {n_domains} (ST vs SC)")

        # VICReg loss
        vicreg_loss_fn = VICRegLoss(
            vicreg_lambda_inv, vicreg_lambda_var, vicreg_lambda_cov,
            vicreg_gamma, vicreg_eps, vicreg_ddp_gather, vicreg_float32_stats
        )
        print(f"  VICReg: λ_inv={vicreg_lambda_inv}, λ_var={vicreg_lambda_var}, λ_cov={vicreg_lambda_cov}")

        # Optimizers
        enc_params = list(model.parameters())
        if projector is not None:
            enc_params += list(projector.parameters())

        opt_enc = torch.optim.Adam(enc_params, lr=lr)
        opt_disc = torch.optim.Adam(discriminator.parameters(), lr=3.0 * lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=n_epochs)
        disc_steps = 10

        print(f"  Augmentations: dropout={aug_gene_dropout}, noise={aug_gauss_std}, jitter={aug_scale_jitter}")
        print(f"  Adversary: weight={adv_slide_weight}, warmup={adv_warmup_epochs}, ramp={adv_ramp_epochs}")
        print("  CORAL: ST↔SC alignment enabled (on same z_cond used by adversary)")
        if sc_patient_ids is not None:
            print(f"  Patient CORAL: weight={patient_coral_weight} (aligns patients within SC domain)")
        print(f"  [GPT5.2-FIX] adv_representation_mode='{adv_representation_mode}', adv_use_layernorm={adv_use_layernorm}")
        if adv_representation_mode == 'ln_avg_aug':
            print("  [WARNING] Using legacy ln_avg_aug mode - this causes train/inference mismatch!")
        print("="*70 + "\n")


        history_vicreg = {
            'epoch': [], 'loss_total': [], 'loss_vicreg': [], 'loss_adv': [],
            'loss_coral': [], 'loss_mmd': [], 'mmd_sigma': [],
            'loss_patient_coral': [],  # Patient-level CORAL for cross-patient alignment
            'loss_source_coral': [],  # Pairwise source CORAL for all-source alignment
            'loss_local': [],  # NEW: Local alignment loss
            'vicreg_inv': [], 'vicreg_var': [], 'vicreg_cov': [],
            'std_mean': [], 'std_min': [], 'disc_acc': [], 'disc_acc_clean': [], 'grl_alpha': []
        }



        # ========== BEST CHECKPOINT TRACKING ==========
        # Track best alignment based on combined CORAL losses
        # Lower = better alignment between domains and patients
        best_alignment_score = float('inf')
        best_epoch = 0
        best_encoder_state = None
        best_projector_state = None
        best_discriminator_state = None

        # Print local alignment config
        if use_local_align:
            print(f"  LOCAL ALIGN: weight={local_align_weight}, tau_x={local_align_tau_x}, "
                  f"tau_z={local_align_tau_z}, bidir={local_align_bidirectional}, warmup={local_align_warmup}")


    else:
        # Geometry mode: original optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    nettrue = A_target

    # Training loop
    # Precompute per-slide index lists for local sampling
    if local_miniset_mode:
        unique_slides = torch.unique(slide_ids)
        slide_index_lists = {}
        for s in unique_slides:
            slide_index_lists[int(s.item())] = torch.where(slide_ids == s)[0]
        print(f"[LOCAL-MINISET] Enabled with {len(unique_slides)} slides")


    # ---------------------------------------------------------------
    # [INFO-NCE PRECOMP] Mutual NN matches across the first 2 slides
    # ---------------------------------------------------------------
    mnn_pairs = None  # list of (i0_global, j1_global)

    mnn_slides = None
    mnn_map0 = None  # global0 -> global1
    mnn_map1 = None  # global1 -> global0

    if slide_align_mode == 'infonce' and slide_ids is not None:
        all_slides = list(slide_index_lists.keys()) if local_miniset_mode else sorted(torch.unique(slide_ids).tolist())
        if len(all_slides) >= 2:
            s0, s1 = all_slides[0], all_slides[1]
            mnn_slides = (s0, s1)
            I0 = slide_index_lists[s0] if local_miniset_mode else torch.where(slide_ids == s0)[0]
            I1 = slide_index_lists[s1] if local_miniset_mode else torch.where(slide_ids == s1)[0]

            with torch.no_grad():
                X0 = F.normalize(st_gene_expr[I0].to(device), dim=1)
                X1 = F.normalize(st_gene_expr[I1].to(device), dim=1)
                S = X0 @ X1.T  # (n0, n1)

                nn01 = S.argmax(dim=1)            # (n0,)
                nn10 = S.argmax(dim=0)            # (n1,)
                mnn_mask = (nn10[nn01] == torch.arange(nn01.shape[0], device=device))

                conf = S[torch.arange(S.shape[0], device=device), nn01]
                mnn_mask = mnn_mask & (conf >= infonce_smin)

                i0_local = torch.where(mnn_mask)[0]
                j1_local = nn01[i0_local]

                if i0_local.numel() > 0:
                    i0_global = I0[i0_local].detach().cpu()
                    j1_global = I1[j1_local].detach().cpu()
                    mnn_pairs = torch.stack([i0_global, j1_global], dim=1)  # (n_pairs, 2)

                    # Build maps once
                    pairs_list = mnn_pairs.tolist()
                    mnn_map0 = {a: b for a, b in pairs_list}
                    mnn_map1 = {b: a for a, b in pairs_list}

                    print(f"[STAGEA-INFO] MNN pairs: {mnn_pairs.shape[0]} (smin={infonce_smin})")
                else:
                    print(f"[STAGEA-INFO] MNN pairs: 0 (smin={infonce_smin})")


    # Training loop
    print(f"Training encoder for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # ========== BRANCH 1: VICReg Mode ==========
        if stageA_obj == 'vicreg_adv':
            model.train()
            if projector is not None:
                projector.train()
            discriminator.train()

            # ========== Balanced domain sampling ==========
            # Sample from all domains (ST slides + SC)
            # ========== Balanced domain sampling ==========
            if use_source_adversary and st_source_ids is not None:
                # N-class source adversary: balance across ALL sources
                idx = sample_balanced_source_indices(domain_ids, batch_size, device)
            elif stageA_balanced_slides:
                if sc_slide_ids is not None:
                    # Hierarchical balancing: domain -> slides within domain
                    idx = sample_balanced_domain_and_slide_indices(
                        domain_ids, slide_ids, sc_slide_ids, batch_size, device
                    )
                else:
                    # Legacy: just domain balancing (ST vs SC)
                    idx = sample_balanced_slide_indices(domain_ids, batch_size, device)
            else:
                idx = torch.randperm(X_ssl.shape[0], device=device)[:batch_size]

            X_batch = X_ssl[idx]
            s_batch = domain_ids[idx]  # 0=ST, 1=SC

            # === NEW: inference-domain dropout (stabilizes across inference subsets)
            if sc_slide_ids is not None and torch.rand(1).item() < inference_dropout_prob:
                sc_mask = (idx >= n_st)                # full-batch mask for SC slots
                sc_local_idx = idx[sc_mask] - n_st     # SC-local indices
                if sc_local_idx.numel() > 0:
                    unique_sc = torch.unique(sc_slide_ids[sc_local_idx])
                    if len(unique_sc) > 1:
                        drop_sc = unique_sc[torch.randint(0, len(unique_sc), (1,)).item()]
                        keep_mask = torch.ones_like(s_batch, dtype=torch.bool)
                        # drop only the SC slots that match the selected slide
                        keep_mask[sc_mask] = (sc_slide_ids[sc_local_idx] != drop_sc)
                        idx = idx[keep_mask]
                        X_batch = X_ssl[idx]
                        s_batch = domain_ids[idx]



            # Two augmented views
            X1 = augment_expression(X_batch, aug_gene_dropout, aug_gauss_std, aug_scale_jitter)
            X2 = augment_expression(X_batch, aug_gene_dropout, aug_gauss_std, aug_scale_jitter)

            # Forward
            z1 = model(X1)
            z2 = model(X2)

            # Projector (VICReg uses y*)
            if projector is not None:
                y1 = projector(z1)
                y2 = projector(z2)
            else:
                y1 = z1
                y2 = z2

            loss_vicreg, vicreg_stats = vicreg_loss_fn(y1, y2)

            alpha = grl_alpha_schedule(epoch, adv_warmup_epochs, adv_ramp_epochs, grl_alpha_max)

            # ------------------------------------------------------------
            # FIX (Run 4.1): Compute conditioning representation from CLEAN input
            # CRITICAL: z1/z2 are augmented (for VICReg invariance only)
            # But adversary, CORAL, local_align must use CLEAN embeddings
            # This matches what evaluation and diffusion will see!
            # ------------------------------------------------------------
            z_clean = model(X_batch)  # Clean input, NO augmentation

            if adv_use_layernorm:
                z_cond = F.layer_norm(z_clean, (z_clean.shape[1],))
            else:
                z_cond = z_clean  # Use raw clean embedding

            # (A) Train discriminator on DETACHED z_cond
            with torch.no_grad():
                z_det = z_cond.detach()

            disc_loss_val = 0.0
            disc_acc_val = 0.0

            # Always train disc a bit (even during warmup) so it doesn't start from scratch later
            disc_grad_norms = {}
            for _ in range(disc_steps):
                logits_d = discriminator(z_det)
                loss_disc = F.cross_entropy(logits_d, s_batch)

                opt_disc.zero_grad(set_to_none=True)
                loss_disc.backward()

                # Optionally log gradient norms before step
                if adv_log_grad_norms and epoch % 50 == 0:
                    disc_grad_norms = compute_gradient_norms(
                        discriminator.named_parameters(), prefix="disc_"
                    )

                opt_disc.step()

                with torch.no_grad():
                    pred = logits_d.argmax(dim=1)
                    disc_acc_val = (pred == s_batch).float().mean().item()
                    disc_loss_val = loss_disc.item()

            # Comprehensive discriminator diagnostics (GPT 5.2 Pro recommendation)
            disc_diag = {}
            if adv_log_diagnostics:
                disc_diag = log_adversary_diagnostics(
                    logits=logits_d,
                    targets=s_batch,
                    n_classes=n_domains,
                    prefix="disc_",
                    epoch=epoch,
                    verbose=(epoch % 100 == 0)
                )

            # ========== DISCRIMINATOR HEALTH CHECK & REVIVAL ==========
            # If discriminator collapses (always predicts one class), revive it
            if disc_diag:
                acc_st = disc_diag.get('disc_acc_class0', 0.5)
                acc_sc = disc_diag.get('disc_acc_class1', 0.5)
                min_class_acc = min(acc_st, acc_sc)

                # Discriminator is "dying" if one class accuracy < 0.1
                if min_class_acc < 0.1 and epoch > adv_warmup_epochs:
                    # Revival: extra training steps with fresh samples
                    revival_steps = 20
                    for _ in range(revival_steps):
                        # Sample fresh batch
                        revival_idx = torch.randperm(X_ssl.shape[0], device=device)[:batch_size]
                        X_revival = X_ssl[revival_idx]
                        s_revival = domain_ids[revival_idx]
                        z_revival = model(X_revival).detach()
                        if adv_use_layernorm:
                            z_revival = F.layer_norm(z_revival, (z_revival.shape[1],))

                        logits_revival = discriminator(z_revival)
                        loss_revival = F.cross_entropy(logits_revival, s_revival)
                        opt_disc.zero_grad(set_to_none=True)
                        loss_revival.backward()
                        opt_disc.step()

                    if epoch % 50 == 0:
                        print(f"  [REVIVAL] Disc collapsed (min_class_acc={min_class_acc:.3f}), did {revival_steps} extra steps")


            # (B) Train encoder to confuse discriminator via GRL
            for p in discriminator.parameters():
                p.requires_grad_(False)

            # CORAL between ST and SC on the SAME z_cond space
            # For source adversary: ST samples are idx < n_st, SC samples are idx >= n_st
            # CORAL + MMD between ST and SC on the SAME z_cond space
            loss_coral = torch.tensor(0.0, device=device)
            loss_mmd = torch.tensor(0.0, device=device)
            mmd_sigma = 0.0
            is_sc = (idx >= n_st)  # FIXED: use position, not source_id
            z_st = z_cond[~is_sc]
            z_sc = z_cond[is_sc]
            if z_st.shape[0] > 8 and z_sc.shape[0] > 8:
                loss_coral = coral_loss(z_st, z_sc)

                # MMD on optional L2-normalized embeddings (more stable)
                z_st_mmd = F.normalize(z_st, dim=1) if mmd_use_l2norm else z_st
                z_sc_mmd = F.normalize(z_sc, dim=1) if mmd_use_l2norm else z_sc

                loss_mmd, mmd_sigma = mmd_rbf_loss(
                    z_st_mmd, z_sc_mmd, return_sigma=True
                )




            # ========== NEW: Patient-level CORAL within SC ==========
            loss_patient_coral = torch.tensor(0.0, device=device)
            if sc_patient_ids is not None and is_sc.sum() > 16:
                # Get SC indices in batch and their global positions
                sc_batch_mask = is_sc
                sc_global_idx = idx[sc_batch_mask] - n_st  # Convert to SC-local indices

                # Get patient IDs for SC samples in this batch
                batch_patient_ids = sc_patient_ids[sc_global_idx]
                unique_patients = torch.unique(batch_patient_ids)

                # Only compute if we have multiple patients in this batch
                if len(unique_patients) >= 2:
                    z_sc_batch = z_cond[sc_batch_mask]

                    # Split by patient and compute pairwise CORAL
                    patient_corals = []
                    for i in range(len(unique_patients)):
                        for j in range(i + 1, len(unique_patients)):
                            p_i, p_j = unique_patients[i], unique_patients[j]
                            mask_i = (batch_patient_ids == p_i)
                            mask_j = (batch_patient_ids == p_j)

                            if mask_i.sum() > 4 and mask_j.sum() > 4:
                                z_pi = z_sc_batch[mask_i]
                                z_pj = z_sc_batch[mask_j]
                                patient_corals.append(coral_loss(z_pi, z_pj))

                    if len(patient_corals) > 0:
                        loss_patient_coral = torch.stack(patient_corals).mean()

            logits_adv = discriminator(grad_reverse(z_cond, alpha))
            loss_adv_enc = F.cross_entropy(logits_adv, s_batch)

            # ========== NEW: Local Alignment Loss (Run 4.1 - FIXED) ==========
            loss_local = torch.tensor(0.0, device=device)
            local_w = 0.0
            if use_local_align and epoch >= local_align_warmup:
                # Get raw expressions for this batch (teacher signal)
                x_st_batch = X_batch[~is_sc]  # ST expression in batch
                x_sc_batch = X_batch[is_sc]   # SC expression in batch

                # FIXED: Use CLEAN embeddings (matches adversary and evaluation)
                z_st_raw = z_clean[~is_sc]  # Clean ST embeddings
                z_sc_raw = z_clean[is_sc]   # Clean SC embeddings


                if z_st_raw.shape[0] > 8 and z_sc_raw.shape[0] > 8:
                    loss_local = compute_local_alignment_loss(
                        z_sc=z_sc_raw,
                        z_st=z_st_raw,
                        x_sc=x_sc_batch,
                        x_st=x_st_batch,
                        tau_x=local_align_tau_x,
                        tau_z=local_align_tau_z,
                        bidirectional=local_align_bidirectional
                    )

                    # Ramp up local align weight
                    local_w = local_align_weight * float(np.clip(
                        (epoch - local_align_warmup) / max(1, adv_ramp_epochs), 0.0, 1.0
                    ))

            # ========== NEW: Pairwise Source CORAL (align ALL sources) ==========
            # This is more stable than N-class adversary - directly minimizes distribution differences
            loss_source_coral = torch.tensor(0.0, device=device)
            if st_source_ids is not None and sc_source_ids is not None:
                # Get source IDs for samples in this batch
                all_source_ids_batch = torch.cat([st_source_ids, sc_source_ids], dim=0).to(device)
                batch_source_ids = all_source_ids_batch[idx]
                unique_sources_in_batch = torch.unique(batch_source_ids)

                # Compute pairwise CORAL between all source pairs present in batch
                source_corals = []
                for i in range(len(unique_sources_in_batch)):
                    for j in range(i + 1, len(unique_sources_in_batch)):
                        src_i, src_j = unique_sources_in_batch[i], unique_sources_in_batch[j]
                        mask_i = (batch_source_ids == src_i)
                        mask_j = (batch_source_ids == src_j)

                        # Need enough samples for stable covariance estimation
                        if mask_i.sum() >= 8 and mask_j.sum() >= 8:
                            z_i = z_cond[mask_i]
                            z_j = z_cond[mask_j]
                            source_corals.append(coral_loss(z_i, z_j))

                if len(source_corals) > 0:
                    loss_source_coral = torch.stack(source_corals).mean()

            coral_w = float(np.clip((epoch - adv_warmup_epochs) / max(1, adv_ramp_epochs), 0.0, 1.0))
            # Patient CORAL uses same ramp as domain CORAL
            patient_coral_w = coral_w * patient_coral_weight if sc_patient_ids is not None else 0.0
            # Source CORAL uses same ramp
            source_coral_w = coral_w * source_coral_weight if (st_source_ids is not None and sc_source_ids is not None) else 0.0
            mmd_w = (coral_w * mmd_weight) if mmd_ramp else mmd_weight
            loss_total = (loss_vicreg + adv_slide_weight * loss_adv_enc + coral_w * loss_coral +
                         local_w * loss_local + patient_coral_w * loss_patient_coral +
                         source_coral_w * loss_source_coral + mmd_w * loss_mmd)


            # Optionally log encoder gradient norms before step
            # Backprop for encoder (MISSING BEFORE!)
            opt_enc.zero_grad(set_to_none=True)
            loss_total.backward()

            # Optionally log encoder gradient norms AFTER backward
            enc_grad_norms = {}
            if adv_log_grad_norms and epoch % 50 == 0:
                enc_grad_norms = compute_gradient_norms(
                    model.named_parameters(), prefix="enc_"
                )
                if projector is not None:
                    proj_grad_norms = compute_gradient_norms(
                        projector.named_parameters(), prefix="proj_"
                    )
                    enc_grad_norms.update(proj_grad_norms)

            opt_enc.step()
            scheduler.step()



            for p in discriminator.parameters():
                p.requires_grad_(True)

            # Logging accuracy (chance=0.5 is what you want to approach)
            with torch.no_grad():
                logits_det = discriminator(z_det)
                disc_acc_det = (logits_det.argmax(dim=1) == s_batch).float().mean().item()

                # Post-update: use CLEAN representation (matches training now)
                z_clean_post = model(X_batch)
                if adv_use_layernorm:
                    z_post = F.layer_norm(z_clean_post, (z_clean_post.shape[1],))
                else:
                    z_post = z_clean_post

                logits_post = discriminator(z_post)
                disc_acc_post = (logits_post.argmax(dim=1) == s_batch).float().mean().item()

                # Now disc_post and disc_CLEAN should be identical (same representation)
                disc_acc_clean = disc_acc_post  # They're the same now!

                # DIAGNOSTIC: Also check augmented for comparison
                z_aug_check = model(X1)
                logits_aug = discriminator(z_aug_check)
                disc_acc_aug = (logits_aug.argmax(dim=1) == s_batch).float().mean().item()


                # Post-update diagnostics on clean representation
                clean_diag = {}
                if adv_log_diagnostics:
                    clean_diag = log_adversary_diagnostics(
                        logits=logits_post,
                        targets=s_batch,
                        n_classes=n_domains,
                        prefix="clean_",
                        epoch=epoch,
                        verbose=(epoch % 100 == 0)
                    )

                loss_adv = loss_adv_enc.detach()


            # Logging
            if epoch % 50 == 0 or (epoch < 5 or epoch > (n_epochs - 25)):
                local_str = f", Local={loss_local.item():.4f}" if use_local_align and epoch >= local_align_warmup else ""
                patient_coral_str = f", PatCORAL={loss_patient_coral.item():.6f}" if sc_patient_ids is not None else ""
                source_coral_str = f", SrcCORAL={loss_source_coral.item():.6f}" if (st_source_ids is not None and sc_source_ids is not None) else ""
                aug_str = f", disc_AUG={disc_acc_aug:.3f}" if epoch % 100 == 0 else ""
                print(f"Epoch {epoch}/{n_epochs} | "
                    f"Loss={loss_total.item():.4f} "
                    f"(VIC={loss_vicreg.item():.3f}, Adv={loss_adv.item():.3f}, "
                    f"CORAL={loss_coral.item():.6f}, MMD={loss_mmd.item():.6f}, "
                    f"MMDw={mmd_w:.2f}, σ={mmd_sigma:.3f}"
                    f"{patient_coral_str}{source_coral_str}{local_str}) | "
                    f"inv={vicreg_stats['inv']:.3f}, var={vicreg_stats['var']:.3f}, cov={vicreg_stats['cov']:.3f} | "
                    f"std: μ={vicreg_stats['std_mean']:.3f}, min={vicreg_stats['std_min']:.3f} | "
                    f"disc_det={disc_acc_det:.3f}, disc_post={disc_acc_post:.3f}{aug_str}, α={alpha:.3f}")



                # Extra diagnostic logging every 100 epochs
                if epoch % 100 == 0 and adv_log_diagnostics:
                    print(f"  [DIAG] mode='{adv_representation_mode}', LN={adv_use_layernorm}")
                    if disc_diag:
                        print(f"  [DIAG] Disc entropy ratio: {disc_diag.get('disc_ent_ratio', 0):.3f}")
                    if clean_diag:
                        print(f"  [DIAG] CLEAN entropy ratio: {clean_diag.get('clean_ent_ratio', 0):.3f}")
                        print(f"  [DIAG] CLEAN per-class: ST={clean_diag.get('clean_acc_class0', 0):.3f}, "
                              f"SC={clean_diag.get('clean_acc_class1', 0):.3f}")


            # Save history
            history_vicreg['epoch'].append(epoch)
            history_vicreg['loss_total'].append(loss_total.item())
            history_vicreg['loss_vicreg'].append(loss_vicreg.item())
            history_vicreg['loss_adv'].append(loss_adv.item())
            history_vicreg['loss_coral'].append(loss_coral.item())
            history_vicreg['loss_mmd'].append(loss_mmd.item())
            history_vicreg['mmd_sigma'].append(float(mmd_sigma))
            history_vicreg['loss_patient_coral'].append(loss_patient_coral.item() if sc_patient_ids is not None else 0.0)
            history_vicreg['loss_source_coral'].append(loss_source_coral.item() if (st_source_ids is not None and sc_source_ids is not None) else 0.0)
            history_vicreg['loss_local'].append(loss_local.item() if use_local_align else 0.0)
            history_vicreg['vicreg_inv'].append(vicreg_stats['inv'])
            history_vicreg['vicreg_var'].append(vicreg_stats['var'])
            history_vicreg['vicreg_cov'].append(vicreg_stats['cov'])
            history_vicreg['std_mean'].append(vicreg_stats['std_mean'])
            history_vicreg['std_min'].append(vicreg_stats['std_min'])
            history_vicreg['disc_acc'].append(disc_acc_det)
            history_vicreg['disc_acc_clean'].append(disc_acc_clean)  # NEW: track clean accuracy
            history_vicreg['grl_alpha'].append(alpha)

            # ========== BEST CHECKPOINT SAVING ==========
            # Only check after warmup, and evaluate every 50 epochs on FULL dataset for stability
            if epoch >= adv_warmup_epochs + adv_ramp_epochs and epoch % 50 == 0:
                # Evaluate on full dataset (or subsample) for stable alignment metric
                with torch.no_grad():
                    # Use up to 2000 samples per domain for speed
                    max_eval = 2000
                    idx_st_eval = torch.arange(n_st, device=device)
                    idx_sc_eval = torch.arange(n_st, n_st + n_sc, device=device)
                    if n_st > max_eval:
                        idx_st_eval = idx_st_eval[torch.randperm(n_st, device=device)[:max_eval]]
                    if n_sc > max_eval:
                        idx_sc_eval = idx_sc_eval[torch.randperm(n_sc, device=device)[:max_eval]]

                    # Get embeddings for ST and SC
                    z_st_eval = model(X_ssl[idx_st_eval])
                    z_sc_eval = model(X_ssl[idx_sc_eval])
                    if adv_use_layernorm:
                        z_st_eval = F.layer_norm(z_st_eval, (z_st_eval.shape[1],))
                        z_sc_eval = F.layer_norm(z_sc_eval, (z_sc_eval.shape[1],))

                    # Get discriminator predictions
                    logits_st = discriminator(z_st_eval)
                    logits_sc = discriminator(z_sc_eval)
                    pred_st = logits_st.argmax(dim=1)
                    pred_sc = logits_sc.argmax(dim=1)

                    # Per-class accuracy (ST should be 0, SC should be 1)
                    acc_class0 = (pred_st == 0).float().mean().item()  # ST correctly predicted as ST
                    acc_class1 = (pred_sc == 1).float().mean().item()  # SC correctly predicted as SC

                    # Alignment score: max deviation from 0.5 (lower = better, means confusion)
                    alignment_score = max(abs(acc_class0 - 0.5), abs(acc_class1 - 0.5))

                if alignment_score < best_alignment_score:
                    best_alignment_score = alignment_score
                    best_epoch = epoch
                    best_encoder_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    if projector is not None:
                        best_projector_state = {k: v.cpu().clone() for k, v in projector.state_dict().items()}
                    best_discriminator_state = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
                    print(f"  [BEST] New best at epoch {epoch}: acc_ST={acc_class0:.3f}, acc_SC={acc_class1:.3f}, score={alignment_score:.4f}")
                elif epoch % 100 == 0:
                    print(f"  [EVAL] Epoch {epoch}: acc_ST={acc_class0:.3f}, acc_SC={acc_class1:.3f}, score={alignment_score:.4f} (best={best_alignment_score:.4f})")

            continue  # Skip geometry code, go to next epoch


        # ========== BRANCH 2: Geometry Mode (rest of your code stays the same) ==========
        # ========== BRANCH 2: Geometry Mode (existing code below) ==========
        # ===================================================================
        # SAMPLE ST BATCH (local miniset mode or global random)
        # ===================================================================
        if local_miniset_mode:
            # Initialize idx_st_list to avoid accidental reference
            idx_st_list = None
            
            # Sample patches from 2 slides for slide alignment
            if slide_align_mode != 'none' and len(slide_index_lists) >= 2:
                # Half batch from each slide
                half_batch = batch_size // 2
                
                # Randomly choose 2 distinct slides each epoch
                all_slides = list(slide_index_lists.keys())

                if slide_align_mode == 'infonce':
                    # MUST use the same two slides used to build mnn_pairs
                    if mnn_slides is None:
                        slides_sampled = all_slides[:2]
                    else:
                        slides_sampled = [mnn_slides[0], mnn_slides[1]]
                else:
                    if len(all_slides) > 2:
                        slides_sampled = np.random.choice(all_slides, size=2, replace=False).tolist()
                    else:
                        slides_sampled = all_slides
                
                idx_st_list = []


                # If InfoNCE: pick a matched anchor pair and force centers
                forced_centers = None
                if slide_align_mode == 'infonce' and mnn_pairs is not None and mnn_pairs.shape[0] > 0 and mnn_slides is not None:
                    pair = mnn_pairs[torch.randint(0, mnn_pairs.shape[0], (1,)).item()]
                    forced_centers = {
                        mnn_slides[0]: int(pair[0].item()),
                        mnn_slides[1]: int(pair[1].item()),
                    }


                
                for s in slides_sampled:
                    I_s = slide_index_lists[s]
                    n_s = len(I_s)
                    
                    # Sample CORE size (leave room for distractors)
                    n_rand = min(32, half_batch // 4)  # Reserve ~25% for distractors
                    n_core_max = min(n_max, half_batch - n_rand, n_s)
                    n_core = np.random.randint(n_min, n_core_max + 1)
                    
                    # Sample center
                    if forced_centers is not None and s in forced_centers:
                        # find local index of forced global center
                        center_global = forced_centers[s]
                        center_local = (I_s == center_global).nonzero(as_tuple=False)
                        if center_local.numel() == 0:
                            center_local = np.random.randint(0, n_s)
                            center_global = I_s[center_local].item()
                        else:
                            center_local = int(center_local[0].item())
                    else:
                        center_local = np.random.randint(0, n_s)
                        center_global = I_s[center_local].item()

                    
                    # Get distances from center to all in slide
                    coords_slide = st_coords_norm[I_s]
                    center_coord = st_coords_norm[center_global]
                    dists = torch.norm(coords_slide - center_coord, dim=1)
                    
                    # Build pool
                    K_pool = min(int(pool_mult * n_core), n_s - 1)
                    K_pool = max(K_pool, n_core - 1)
                    
                    # Sort by distance (exclude self)
                    dists_no_self = dists.clone()
                    dists_no_self[center_local] = float('inf')
                    sorted_idx = torch.argsort(dists_no_self)
                    pool_local = sorted_idx[:K_pool]
                    pool_dists = dists_no_self[pool_local]
                    
                    # Sample neighbors from FULL POOL (FIX 1)
                    n_neighbors = min(n_core - 1, len(pool_local))
                    if n_neighbors > 0:
                        weights = torch.softmax(-pool_dists / stochastic_tau, dim=0)  # Full pool!
                        sampled_local = torch.multinomial(weights, n_neighbors, replacement=False)
                        neighbors_local = pool_local[sampled_local]
                        core_local = torch.cat([torch.tensor([center_local], device=device), neighbors_local])
                    else:
                        core_local = torch.tensor([center_local], device=device)
                    
                    core_global = I_s[core_local]
                    
                    # Add DISTRACTORS (FIX 2)
                    core_set = set(core_local.cpu().tolist())
                    available = [i for i in range(n_s) if i not in core_set]
                    n_distract = min(n_rand, len(available))
                    
                    if n_distract > 0:
                        distract_idx = np.random.choice(available, size=n_distract, replace=False)
                        distract_global = I_s[torch.tensor(distract_idx, device=device)]
                        patch_global = torch.cat([core_global, distract_global])
                    else:
                        patch_global = core_global
                    
                    # Pad to half_batch if still short
                    if len(patch_global) < half_batch:
                        remaining = half_batch - len(patch_global)
                        extra_available = [i for i in range(n_s) if i not in set(patch_global.cpu().tolist())]
                        if len(extra_available) >= remaining:
                            extra_idx = np.random.choice(extra_available, size=remaining, replace=False)
                            extra_global = I_s[torch.tensor(extra_idx, device=device)]
                            patch_global = torch.cat([patch_global, extra_global])
                    
                    idx_st_list.append(patch_global[:half_batch])

                
                idx_st = torch.cat(idx_st_list)
                
            else:
                # Single slide or no alignment: sample one patch
                s = np.random.choice(list(slide_index_lists.keys()))
                I_s = slide_index_lists[s]
                n_s = len(I_s)
                
                # Sample CORE size (leave room for distractors)
                n_rand = min(64, batch_size // 4)
                n_core_max = min(n_max, batch_size - n_rand, n_s)
                n_core = np.random.randint(n_min, n_core_max + 1)
                
                center_local = np.random.randint(0, n_s)
                center_global = I_s[center_local].item()
                
                coords_slide = st_coords_norm[I_s]
                center_coord = st_coords_norm[center_global]
                dists = torch.norm(coords_slide - center_coord, dim=1)
                
                K_pool = min(int(pool_mult * n_core), n_s - 1)
                K_pool = max(K_pool, n_core - 1)
                
                dists_no_self = dists.clone()
                dists_no_self[center_local] = float('inf')
                sorted_idx = torch.argsort(dists_no_self)
                pool_local = sorted_idx[:K_pool]
                pool_dists = dists_no_self[pool_local]
                
                # Sample neighbors from FULL POOL
                n_neighbors = min(n_core - 1, len(pool_local))
                if n_neighbors > 0:
                    weights = torch.softmax(-pool_dists / stochastic_tau, dim=0)
                    sampled_local = torch.multinomial(weights, n_neighbors, replacement=False)
                    neighbors_local = pool_local[sampled_local]
                    core_local = torch.cat([torch.tensor([center_local], device=device), neighbors_local])
                else:
                    core_local = torch.tensor([center_local], device=device)
                
                core_global = I_s[core_local]
                
                # Add DISTRACTORS
                core_set = set(core_local.cpu().tolist())
                available = [i for i in range(n_s) if i not in core_set]
                n_distract = min(n_rand, len(available))
                
                if n_distract > 0:
                    distract_idx = np.random.choice(available, size=n_distract, replace=False)
                    distract_global = I_s[torch.tensor(distract_idx, device=device)]
                    patch_global = torch.cat([core_global, distract_global])
                else:
                    patch_global = core_global
                
                # Pad to batch_size if still short
                if len(patch_global) < batch_size:
                    remaining = batch_size - len(patch_global)
                    extra_available = [i for i in range(n_s) if i not in set(patch_global.cpu().tolist())]
                    if len(extra_available) >= remaining:
                        extra_idx = np.random.choice(extra_available, size=remaining, replace=False)
                        extra_global = I_s[torch.tensor(extra_idx, device=device)]
                        patch_global = torch.cat([patch_global, extra_global])
                
                idx_st = patch_global[:batch_size]

        else:
            # Original: global random sampling
            idx_st = torch.randperm(st_gene_expr.shape[0], device=device)[:batch_size]
        
        idx_sc = torch.randperm(sc_gene_expr.shape[0], device=device)[:batch_size]
        
        X_st_batch = st_gene_expr[idx_st]
        X_sc_batch = sc_gene_expr[idx_sc]
        
        # Forward pass
        Z_st = model(X_st_batch)
        Z_sc = model(X_sc_batch)
               
        #loss 1: rbf adjacency prediction (with slide mask)
        # Loss 1: RBF adjacency prediction (on-the-fly for local patches)
        if local_miniset_mode:
            # Compute target q from patch coordinates on-the-fly
            coords_batch = st_coords_norm[idx_st]
            D_batch = torch.cdist(coords_batch, coords_batch)
            q_target = torch.exp(-D_batch**2 / (2 * sigma**2))
            
            # Mask diagonal and normalize
            slide_mask_batch = slide_mask[idx_st][:, idx_st]
            q_target = q_target * slide_mask_batch.float()
            q_target.fill_diagonal_(0.0)
            row_sums = q_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
            q_target = q_target / row_sums
        else:
            # Use precomputed A_target
            A_target_batch = A_target[idx_st][:, idx_st]
            slide_mask_batch = slide_mask[idx_st][:, idx_st]
            
            q_target = A_target_batch * slide_mask_batch.float()
            row_sums = q_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
            q_target = q_target / row_sums
        
        # Predict p from embeddings
        temp = 0.1
        netpred = Z_st @ Z_st.t()
        logits = netpred / temp
        logits = logits.masked_fill(~slide_mask_batch, -1e9)
        
        log_probs = F.log_softmax(logits, dim=1)
        loss_pred = F.kl_div(log_probs, q_target, reduction='batchmean')

        
        # Loss 2: circular coupling (with slide mask)
        # Get all embeddings
        # Loss 2: circular coupling (optional)
        if use_circle:
            Z_st_all = model(st_gene_expr)
            Z_sc_all = model(sc_gene_expr)

            st2sc = F.softmax(Z_st @ Z_sc_all.t(), dim=1)
            sc2st = F.softmax(Z_sc_all @ Z_st.t(), dim=1)
            st2st_unnorm = st2sc @ sc2st
            
            st2st_unnorm = st2st_unnorm * slide_mask_batch.float()
            st2st_unnorm = st2st_unnorm / (st2st_unnorm.sum(dim=1, keepdim=True) + 1e-8)
            st2st = torch.log(st2st_unnorm + 1e-7)

            if local_miniset_mode:
                nettrue_batch = q_target
            else:
                nettrue_batch = nettrue[idx_st][:, idx_st]
                nettrue_batch = nettrue_batch * slide_mask_batch.float()
                nettrue_batch = nettrue_batch / (nettrue_batch.sum(dim=1, keepdim=True) + 1e-8)

            loss_circle = F.kl_div(st2st, nettrue_batch, reduction='none').sum(1).mean()
        else:
            loss_circle = torch.tensor(0.0, device=device)


        
        # Loss 3: MMD
        # Loss 3: MMD (ST-SC alignment, optional)
        if use_mmd_sc:
            n_mmd = int(mmdbatch * min(Z_st.shape[0], Z_sc.shape[0]))
            if n_mmd > 1:
                idx_mmd_st = torch.randperm(Z_st.shape[0], device=device)[:n_mmd]
                idx_mmd_sc = torch.randperm(Z_sc.shape[0], device=device)[:n_mmd]
                
                K_st = torch.exp(-torch.cdist(Z_st[idx_mmd_st], Z_st[idx_mmd_st])**2 / 0.1)
                K_sc = torch.exp(-torch.cdist(Z_sc[idx_mmd_sc], Z_sc[idx_mmd_sc])**2 / 0.1)
                K_st_sc = torch.exp(-torch.cdist(Z_st[idx_mmd_st], Z_sc[idx_mmd_sc])**2 / 0.1)
                
                mmd = K_st.mean() + K_sc.mean() - 2 * K_st_sc.mean()
                loss_mmd = torch.clamp(mmd, min=0)
            else:
                loss_mmd = torch.tensor(0.0, device=device)
        else:
            loss_mmd = torch.tensor(0.0, device=device)
        
        # Loss 4: Slide-invariance alignment (InfoNCE / CORAL / MMD between slides)
        if slide_align_mode != 'none' and local_miniset_mode and idx_st_list is not None and len(idx_st_list) == 2:
            # Split embeddings and expressions by slide
            half_batch = batch_size // 2
            X0 = X_st_batch[:half_batch]
            X1 = X_st_batch[half_batch:]
            Z0 = Z_st[:half_batch]
            Z1 = Z_st[half_batch:]
            
            if slide_align_mode == 'infonce':
                loss_align = torch.tensor(0.0, device=device)

                if mnn_map0 is None or mnn_map1 is None:
                    loss_align = torch.tensor(0.0, device=device)
                else:
                    # Build in-batch targets using precomputed MNN maps
                    idx0_global = idx_st[:half_batch]
                    idx1_global = idx_st[half_batch:]

                    targets01 = []
                    rows0 = []
                    for i in range(half_batch):
                        g0 = int(idx0_global[i].item())
                        g1 = mnn_map0.get(g0, None)
                        if g1 is None:
                            continue
                        pos = (idx1_global == g1).nonzero(as_tuple=False)
                        if pos.numel() == 0:
                            continue
                        rows0.append(i)
                        targets01.append(int(pos[0].item()))

                    if len(rows0) >= 8:
                        valid0 = torch.tensor(rows0, device=device, dtype=torch.long)
                        pos01 = torch.tensor(targets01, device=device, dtype=torch.long)

                        Z0n = F.normalize(Z0, dim=1)
                        Z1n = F.normalize(Z1, dim=1)

                        L01_full = (Z0n @ Z1n.T) / infonce_tau          # (half, half)
                        L01 = L01_full[valid0]                          # (n_valid0, half)
                        loss01 = F.cross_entropy(L01, pos01)

                        loss_align = loss01

                        # Optional symmetric direction using reverse MNN map
                        acc10 = None
                        if infonce_sym:
                            targets10 = []
                            rows1 = []
                            for j in range(half_batch):
                                g1 = int(idx1_global[j].item())
                                g0 = mnn_map1.get(g1, None)
                                if g0 is None:
                                    continue
                                pos = (idx0_global == g0).nonzero(as_tuple=False)
                                if pos.numel() == 0:
                                    continue
                                rows1.append(j)
                                targets10.append(int(pos[0].item()))

                            if len(rows1) >= 8:
                                valid1 = torch.tensor(rows1, device=device, dtype=torch.long)
                                pos10 = torch.tensor(targets10, device=device, dtype=torch.long)

                                L10_full = (Z1n @ Z0n.T) / infonce_tau      # (half, half)
                                L10 = L10_full[valid1]                      # (n_valid1, half)
                                loss10 = F.cross_entropy(L10, pos10)

                                loss_align = 0.5 * (loss01 + loss10)

                        # Debug every 100 epochs (no undefined Sx)
                        if epoch % 100 == 0:
                            with torch.no_grad():
                                pred01 = L01.argmax(dim=1)
                                acc01 = (pred01 == pos01).float().mean().item()

                                # expression similarity for matched pairs (in-batch)
                                X0n_dbg = F.normalize(X0, dim=1)
                                X1n_dbg = F.normalize(X1, dim=1)
                                Sx = X0n_dbg @ X1n_dbg.T  # (half, half)

                                mean_cos_x = Sx[valid0, pos01].mean().item()
                                mean_cos_z = (Z0n[valid0] * Z1n[pos01]).sum(dim=1).mean().item()

                                if infonce_sym and 'L10' in locals():
                                    pred10 = L10.argmax(dim=1)
                                    acc10 = (pred10 == pos10).float().mean().item()
                                    acc_mean = 0.5 * (acc01 + acc10)
                                else:
                                    acc_mean = acc01

                                print(
                                    f"[STAGEA-INFO] InfoNCE: loss={loss_align.item():.4f} "
                                    f"acc={acc_mean:.3f} npos={valid0.numel()} "
                                    f"cos_expr={mean_cos_x:.3f} cos_embed={mean_cos_z:.3f}"
                                )
                    else:
                        loss_align = torch.tensor(0.0, device=device)
            
            elif slide_align_mode == 'coral':
                # CORAL: match mean and covariance
                mu0 = Z0.mean(dim=0)
                mu1 = Z1.mean(dim=0)
                
                Z0_c = Z0 - mu0
                Z1_c = Z1 - mu1
                
                cov0 = (Z0_c.T @ Z0_c) / (Z0_c.shape[0] - 1)
                cov1 = (Z1_c.T @ Z1_c) / (Z1_c.shape[0] - 1)
                
                loss_align = (mu0 - mu1).pow(2).sum() + (cov0 - cov1).pow(2).sum()
                
            elif slide_align_mode == 'mmd':
                # MMD between two slides
                K_00 = torch.exp(-torch.cdist(Z0, Z0)**2 / 0.1)
                K_11 = torch.exp(-torch.cdist(Z1, Z1)**2 / 0.1)
                K_01 = torch.exp(-torch.cdist(Z0, Z1)**2 / 0.1)
                
                mmd_align = K_00.mean() + K_11.mean() - 2 * K_01.mean()
                loss_align = torch.clamp(mmd_align, min=0)
            else:
                loss_align = torch.tensor(0.0, device=device)
        else:
            loss_align = torch.tensor(0.0, device=device)

        
        # Total loss
        # Total loss
        ratio = ratio_start + (ratio_end - ratio_start) * min(epoch / (n_epochs * 0.8), 1.0)
        
        # Warmup alignment so adjacency learns first
        if slide_align_mode != 'none' and loss_align.item() > 0:
            align_w = slide_align_weight * float(np.clip((epoch - align_warmup) / max(1, align_ramp), 0.0, 1.0))
        else:
            align_w = 0.0

        loss = loss_pred + alpha * loss_mmd + ratio * loss_circle + align_w * loss_align


        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 100 == 0:
            log_str = f"Epoch {epoch}/{n_epochs} | Loss: {loss.item():.4f} | Pred: {loss_pred.item():.4f}"
            if use_circle:
                log_str += f" | Circle: {loss_circle.item():.4f}"
            if use_mmd_sc:
                log_str += f" | MMD: {loss_mmd.item():.4f}"
            if slide_align_mode != 'none' and loss_align.item() > 0:
                log_str += f" | Align({slide_align_mode}): {loss_align.item():.4f}"
            if local_miniset_mode:
                log_str += f" | n_patch={len(idx_st)}"
            print(log_str)


        
        # Debug: sanity check slide masking (print rarely)
        if epoch % 200 == 0:
            with torch.no_grad():
                # Compute unmasked probability distribution for adjacency loss
                logits_raw = (Z_st @ Z_st.t()) / temp
                probs_raw = F.softmax(logits_raw, dim=1)
                
                # Compute masked probability distribution
                logits_masked = logits_raw.masked_fill(~slide_mask_batch, -1e9)
                probs_masked = F.softmax(logits_masked, dim=1)
                
                # Probability mass on invalid (cross-slide) entries
                invalid_mask = ~slide_mask_batch
                mass_invalid_before = (probs_raw * invalid_mask.float()).sum(dim=1).mean().item()
                mass_invalid_after = (probs_masked * invalid_mask.float()).sum(dim=1).mean().item()
                
                print(f"[STAGEA-MASKDBG] Epoch {epoch}: invalid prob mass before={mass_invalid_before:.6f} after={mass_invalid_after:.6f}")

    
    # Save checkpoint
    os.makedirs(outf, exist_ok=True)
    
    if stageA_obj == 'vicreg_adv':
        # ========== RESTORE BEST CHECKPOINT IF AVAILABLE ==========
        if use_best_checkpoint and best_encoder_state is not None:
            print(f"\n[BEST CHECKPOINT] Restoring best model from epoch {best_epoch}")
            print(f"  Best alignment score: {best_alignment_score:.6f}")
            model.load_state_dict(best_encoder_state)
            if projector is not None and best_projector_state is not None:
                projector.load_state_dict(best_projector_state)
            if best_discriminator_state is not None:
                discriminator.load_state_dict(best_discriminator_state)
        elif use_best_checkpoint:
            print("\n[BEST CHECKPOINT] No checkpoint saved (training too short or warmup not reached)")
            print("  Using final model instead")

        # Save VICReg history
        import json
        history_path = os.path.join(outf, 'stageA_vicreg_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_vicreg, f, indent=2)

        print("\n" + "="*70)
        print("VICReg Training Complete")
        print("="*70)
        if use_best_checkpoint and best_encoder_state is not None:
            print(f"Using BEST checkpoint from epoch {best_epoch}")
            print(f"Best alignment score: {best_alignment_score:.6f}")
        else:
            print(f"Using FINAL model from epoch {n_epochs-1}")
        print(f"Final Loss: {history_vicreg['loss_total'][-1]:.4f}")
        print(f"Final VICReg: {history_vicreg['loss_vicreg'][-1]:.3f}")
        print(f"Final std mean: {history_vicreg['std_mean'][-1]:.3f}")
        print(f"Final disc acc: {history_vicreg['disc_acc'][-1]:.3f}")
        print("="*70 + "\n")
        print(f"History saved: {history_path}")
    
    torch.save(model.state_dict(), os.path.join(outf, 'encoder_final_new.pt'))

    if stageA_obj == 'vicreg_adv':
        aux_path = os.path.join(outf, 'stageA_vicreg_aux.pt')
        aux = {
            "vicreg_use_projector": vicreg_use_projector,
            "h_dim": model.n_embedding[-1],
            "vicreg_project_dim": vicreg_project_dim,
            "disc_hidden": disc_hidden,
            "disc_dropout": disc_dropout,
            "n_slides": int(torch.unique(slide_ids).numel()),
            "projector_state_dict": (projector.state_dict() if projector is not None else None),
            "discriminator_state_dict": discriminator.state_dict(),
        }
        torch.save(aux, aux_path)
        print(f"[StageA] Saved aux modules to: {aux_path}")

    print("Encoder training complete!")

    if return_aux and stageA_obj == 'vicreg_adv':
        return model, projector, discriminator, history_vicreg
    return model



# ==============================================================================
# STAGE B: GEOMETRIC TARGET PRECOMPUTATION
# ==============================================================================

@dataclass
@dataclass
class STTargets:
    """Precomputed geometric targets for an ST slide."""
    y_hat: torch.Tensor      # (n, 2) normalized coords
    G: torch.Tensor          # (n, n) Gram matrix
    D: torch.Tensor          # (n, n) distance matrix
    H: torch.Tensor          # (num_bins,) distance histogram
    H_bins: torch.Tensor     # (num_bins,) histogram bin edges
    L: torch.Tensor          # (n, n) graph Laplacian
    t_list: List[float]      # heat kernel time points
    triplets: torch.Tensor   # (T, 3) ordinal triplets
    k: int                   # kNN parameter
    scale: float             # normalization scale factor
    knn_indices: torch.Tensor  # (n, k) k-NN indices for each spot
    knn_spatial: torch.Tensor   # (n, k) k-NN indices from SPATIAL coordinates (not embeddings)
    # NEW: Persistent homology topology info (for full slide)
    topo_pairs_0: torch.Tensor  # (m0, 2) index pairs for 0D features (connected components)
    topo_dists_0: torch.Tensor  # (m0,) birth distances for 0D features
    topo_pairs_1: torch.Tensor  # (m1, 2) index pairs for 1D features (loops)
    topo_dists_1: torch.Tensor  # (m1,) birth distances for 1D features



class STStageBPrecomputer:
    """
    Precompute pose-free geometric targets for each ST slide.
    
    Outputs:
    - Per-slide: center, scale, y_hat, G, D, H, triplets, metadata
    - Saved as .pt files for fast loading
    """
    
    def __init__(
        self,
        k: int = 20,
        sigma_policy: str = 'median_knn',
        t_list: List[float] = [0.25, 1.0, 4.0],
        num_bins: int = 64,
        n_triplets: int = 1000,
        margin_ratio: float = 0.05,
        device: str = 'cuda'
    ):
        self.k = k
        self.sigma_policy = sigma_policy
        self.t_list = t_list
        self.num_bins = num_bins
        self.n_triplets = n_triplets
        self.margin_ratio = margin_ratio
        self.device = device
    
    def precompute_slide(
        self,
        slide_id: int,
        st_coords: torch.Tensor,
        encoder: SharedEncoder,
        st_gene_expr: torch.Tensor
    ) -> STTargets:
        """
        Precompute targets for a single slide.
        
        Args:
            slide_id: slide identifier
            st_coords: (m, 2) raw spatial coordinates
            encoder: frozen encoder (for Z indices)
            st_gene_expr: (m, n_genes) gene expression
            
        Returns:
            STTargets object
        """
        device = self.device
        st_coords = st_coords.to(device)
        
        # 1. Pose normalization
        # y_hat, center, scale = uet.normalize_pose_scale(st_coords)
        # 1. Coords already normalized from run_mouse_brain_2.py
        y_hat = st_coords
        center = torch.zeros(2, device=device)
        scale = 1.0
        print(f"[Stage B] Using pre-normalized coordinates for slide {slide_id}")
        
        # 2. Pairwise distances & Gram
        D = torch.cdist(y_hat, y_hat, p=2)
        G = uet.gram_from_coords(y_hat)

        # Compute k-NN indices from EMBEDDINGS (not coordinates)
        # This makes the loss biologically meaningful (expression similarity preservation)
        print(f"[Stage B] Computing kNN from embeddings (not coordinates)")
        Z = encoder(st_gene_expr.to(device))  # (n, 128) embeddings
        D_Z = torch.cdist(Z, Z)  # Distance in embedding space

        knn_k = self.k
        n_spots = y_hat.shape[0]
        knn_indices = torch.zeros(n_spots, knn_k, dtype=torch.long, device=device)

        for i in range(n_spots):
            # Get distances from spot i to all others IN EMBEDDING SPACE
            dists_from_i = D_Z[i]  # (n,)
            # Set self-distance to infinity to exclude
            dists_from_i_copy = dists_from_i.clone()
            dists_from_i_copy[i] = float('inf')
            # Get indices of k smallest distances
            _, indices = torch.topk(dists_from_i_copy, k=min(knn_k, n_spots-1), largest=False)
            knn_indices[i, :len(indices)] = indices
            # If we have fewer than k neighbors, pad with -1
            if len(indices) < knn_k:
                knn_indices[i, len(indices):] = -1

        print(f"[Stage B] kNN computed from embeddings: {knn_indices.shape}")
        
        # 3. Distance histogram
        d_95 = torch.quantile(D[torch.triu(torch.ones_like(D), diagonal=1).bool()], 0.95)
        bins = torch.linspace(0, d_95, self.num_bins + 1, device=device)
        H = uet.compute_distance_hist(D, bins)
        
        # 4. Ordinal triplets
        triplets = uet.sample_ordinal_triplets(D, self.n_triplets, self.margin_ratio)
        
        # 5. Z indices (for linking to encoder)
        Z_indices = torch.arange(st_coords.shape[0], device=device)
        
        
        targets = STTargets(
            slide_id=slide_id,
            center=center,
            scale=scale,
            y_hat=y_hat,
            G=G,
            D=D,
            H=H,
            t_list=self.t_list,
            k=self.k,
            sigma_policy=self.sigma_policy,
            triplets=triplets,
            Z_indices=Z_indices,
            knn_indices=knn_indices
        )
        
        return targets
    
    def precompute(
        self,
        slides: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        encoder: SharedEncoder,
        outdir: str = 'stage_b_cache',
        use_affine_whitening: bool = True,
        use_geodesic_targets: bool = True,
        geodesic_k: int = 15
    ) -> Dict[int, STTargets]:
        """
        Precompute pose-free targets for ST slides (NO CACHING - always recompute).
        """
        targets_dict = {}
        
        encoder.eval()
        for slide_id, (st_coords, st_gene_expr) in slides.items():
            print(f"Computing targets for slide {slide_id} (n={st_coords.shape[0]})")
            
            # Normalize coordinates
            # if use_affine_whitening:
            #     y_hat, scale = uet.affine_whitening(st_coords, eps=1e-6)
            # else:
            #     y_hat = uet.normalize_coordinates_isotropic(st_coords)
            #     scale = 1.0

            # NEW: Coords already normalized from run_mouse_brain_2.py
            y_hat = st_coords
            scale = 1.0
            print(f"[Stage B] Using pre-normalized coordinates for slide {slide_id}")
            
            n = y_hat.shape[0]
            
            # Compute distances
            if use_geodesic_targets:
                D = uet.compute_geodesic_distances(y_hat, k=geodesic_k, device=self.device)
            else:
                D = torch.cdist(y_hat, y_hat)
            
            # Gram from geodesic or coords
            if use_geodesic_targets:
                G = uet.gram_from_geodesic(D)
            else:
                G = uet.gram_from_coords(y_hat)
            
            # Distance histogram
            d_95 = torch.quantile(D[torch.triu(torch.ones_like(D), diagonal=1).bool()], 0.95)
            bins = torch.linspace(0, d_95, self.num_bins, device=self.device)
            H = uet.compute_distance_hist(D, bins)
            
            # Ordinal triplets from geodesic distances
            triplets = uet.sample_ordinal_triplets(D, n_triplets=self.n_triplets, margin_ratio=self.margin_ratio)
            
            # Graph Laplacian
            edge_index, edge_weight = uet.build_knn_graph(y_hat, k=self.k, device=self.device)
            L = uet.compute_graph_laplacian(edge_index, edge_weight, n)

            # Compute k-NN indices for each spot
            knn_k = self.k
            n_spots = y_hat.shape[0]
            knn_indices = torch.zeros(n_spots, knn_k, dtype=torch.long, device=self.device)

            for i in range(n_spots):
                # Get distances from spot i to all others
                dists_from_i = D[i]  # (n,)
                # Set self-distance to infinity to exclude
                dists_from_i_copy = dists_from_i.clone()
                dists_from_i_copy[i] = float('inf')
                # Get indices of k smallest distances
                _, indices = torch.topk(dists_from_i_copy, k=min(knn_k, n_spots-1), largest=False)
                knn_indices[i, :len(indices)] = indices
                # If we have fewer than k neighbors, pad with -1
                if len(indices) < knn_k:
                    knn_indices[i, len(indices):] = -1

            # NEW: Compute SPATIAL kNN (from coordinates, not embeddings)
            print(f"[Stage B] Computing spatial kNN for slide {slide_id}...")
            D_spatial = torch.cdist(y_hat, y_hat)  # (n, n)
            D_spatial_noself = D_spatial + torch.eye(n_spots, device=y_hat.device) * 1e10
            _, knn_spatial = torch.topk(D_spatial_noself, k=knn_k, dim=1, largest=False)
            # knn_spatial is (n, k) indices in global slide indexing

            # NEW: Compute persistent homology for topology preservation
            print(f"[Stage B] Computing persistent homology for slide {slide_id}...")
            y_hat_np = y_hat.cpu().numpy()
            topo_info = uet.compute_persistent_pairs(y_hat_np, max_pairs_0d=10, max_pairs_1d=5)
            print(f"[Stage B]   0D pairs: {topo_info['pairs_0'].shape[0]}, 1D pairs: {topo_info['pairs_1'].shape[0]}")

            targets = STTargets(
                y_hat=y_hat.cpu(),
                G=G.cpu(),
                D=D.cpu(),
                H=H.cpu(),
                H_bins=bins.cpu(),
                L=L.to_dense().cpu() if L.is_sparse else L.cpu(),  # Convert sparse to dense + CPU
                t_list=self.t_list,
                triplets=triplets.cpu(),
                k=self.k,
                scale=scale,
                knn_indices=knn_indices.cpu(),
                knn_spatial=knn_spatial.cpu(),
                topo_pairs_0=torch.from_numpy(topo_info['pairs_0']).long(),
                topo_dists_0=torch.from_numpy(topo_info['dists_0']).float(),
                topo_pairs_1=torch.from_numpy(topo_info['pairs_1']).long(),
                topo_dists_1=torch.from_numpy(topo_info['dists_1']).float()
            )
            
            targets_dict[slide_id] = targets
            print(f"  Targets computed for slide {slide_id}")
        
        return targets_dict
    

# ==============================================================================
# STAGE B: MINI-SET DATASET
# ==============================================================================

class STSetDataset(Dataset):
    def __init__(
        self,
        targets_dict: Dict[int, STTargets],
        encoder: SharedEncoder,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_min: int = 64,
        n_max: int = 256,
        D_latent: int = 16,
        num_samples: int = 10000,
        knn_k: int = 12,
        device: str = 'cuda',
        landmarks_L: int = 32,
        pool_mult: float = 4.0,
        stochastic_tau: float = 1.0,
        # ========== NEW: Competitor training params ==========
        compete_train: bool = False,
        compete_n_extra: int = 128,
        compete_n_rand: int = 64,
        compete_n_hard: int = 64,
        compete_use_pos_closure: bool = True,
        compete_k_pos: int = 10,
        compete_expr_knn_k: int = 50,
        compete_anchor_only: bool = True,
        # ========== NEW: Anchored training params ==========
        anchor_train: bool = False,
        anchor_p_uncond: float = 0.50,
        anchor_frac_min: float = 0.10,
        anchor_frac_max: float = 0.30,
        anchor_min: int = 8,
        anchor_max: int = 96,
        anchor_mode: str = "ball",
        anchor_exclude_landmarks: bool = True,
        debug_anchor_dataset: bool = False,
    ):

        self.targets_dict = targets_dict
        self.encoder = encoder
        self.st_gene_expr_dict = st_gene_expr_dict
        self.n_min = n_min
        self.n_max = n_max
        self.D_latent = D_latent
        self.num_samples = num_samples
        self.knn_k = knn_k
        self.device = device
        self.slide_ids = list(targets_dict.keys())
        self.landmarks_L = 0

        # Store stochastic sampling params
        self.pool_mult = pool_mult
        self.stochastic_tau = stochastic_tau
        
        # ========== NEW: Store competitor training params ==========
        self.compete_train = compete_train
        self.compete_n_extra = compete_n_extra
        self.compete_n_rand = compete_n_rand
        self.compete_n_hard = compete_n_hard
        self.compete_use_pos_closure = compete_use_pos_closure
        self.compete_k_pos = compete_k_pos
        self.compete_expr_knn_k = compete_expr_knn_k
        self.compete_anchor_only = compete_anchor_only
        
        # ========== NEW: Store anchored training params ==========
        self.anchor_train = anchor_train
        self.anchor_p_uncond = anchor_p_uncond
        self.anchor_frac_min = anchor_frac_min
        self.anchor_frac_max = anchor_frac_max
        self.anchor_min = anchor_min
        self.anchor_max = anchor_max
        self.anchor_mode = anchor_mode
        self.anchor_exclude_landmarks = anchor_exclude_landmarks
        self.debug_anchor_dataset = debug_anchor_dataset
        
        # Precompute encoder embeddings for all slides
        self.Z_dict = {}

        self.expr_knn_indices = {}  # NEW: expression-based kNN for hard negatives
        
        with torch.no_grad():
            for slide_id, st_expr in st_gene_expr_dict.items():
                Z = self.encoder(st_expr.to(device))
                self.Z_dict[slide_id] = Z.cpu()
                
                # ========== NEW: Precompute expression kNN for hard negatives ==========
                if compete_train and compete_n_hard > 0:
                    n_spots = Z.shape[0]
                    # Normalize embeddings for cosine similarity
                    Z_norm = F.normalize(Z, p=2, dim=1)  # (n_spots, h_dim)
                    # Compute similarity matrix
                    sim_matrix = Z_norm @ Z_norm.T  # (n_spots, n_spots)
                    # Mask out self-similarity
                    sim_matrix.fill_diagonal_(-float('inf'))
                    # Get top-K most similar (expression neighbors)
                    k_expr = min(compete_expr_knn_k, n_spots - 1)
                    _, expr_knn_idx = torch.topk(sim_matrix, k=k_expr, dim=1)  # (n_spots, k_expr)
                    self.expr_knn_indices[slide_id] = expr_knn_idx.cpu()
                    print(f"[STSetDataset] Computed expression kNN for slide {slide_id}: "
                          f"{n_spots} spots, k={k_expr}")

    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Pick a slide and subset size
        slide_id = np.random.choice(self.slide_ids)
        targets = self.targets_dict[slide_id]

        n = np.random.randint(self.n_min, self.n_max + 1)
        m = targets.y_hat.shape[0]
        n = min(n, m)  # don't exceed slide size

        # DEBUG: Only print for first 3 minisets per epoch
        debug_this_sample = (idx < 3)

        # ------------------------------------------------------------------
        # CORE PATCH SAMPLING: pure local kNN patch in TRUE ST distance space
        # ------------------------------------------------------------------
        center_idx = np.random.randint(0, m)
        D_row = targets.D[center_idx]  # (m,)
        
        all_idx = torch.arange(m)
        mask_self = all_idx != center_idx
        dists = D_row[mask_self]
        idx_no_self = all_idx[mask_self]

        if idx_no_self.numel() == 0:
            indices_core = all_idx[:n]
        else:
            sort_order = torch.argsort(dists)
            sorted_neighbors = idx_no_self[sort_order]
            sorted_dists = dists[sort_order]

            n_neighbors_needed = n - 1
            K_pool = min(sorted_neighbors.numel(), int(self.pool_mult * n))
            K_pool = max(K_pool, n_neighbors_needed)

            pool = sorted_neighbors[:K_pool]
            pool_d = sorted_dists[:K_pool]

            if pool.numel() <= n_neighbors_needed:
                neighbors = pool
            else:
                weights = torch.softmax(-pool_d / self.stochastic_tau, dim=0)
                sampled_idx = torch.multinomial(weights, n_neighbors_needed, replacement=False)
                neighbors = pool[sampled_idx]

            indices_core = torch.cat([
                torch.tensor([center_idx], dtype=torch.long),
                neighbors
            ])

            if indices_core.numel() < n:
                missing = n - indices_core.numel()
                mask_extra = ~torch.isin(all_idx, indices_core)
                extra_pool = all_idx[mask_extra]
                if extra_pool.numel() > 0:
                    perm = torch.randperm(extra_pool.numel())
                    add = extra_pool[perm[:min(missing, extra_pool.numel())]]
                    indices_core = torch.cat([indices_core, add])

        # Shuffle core to avoid positional bias
        indices_core = indices_core[torch.randperm(indices_core.numel())]
        n_core = indices_core.numel()

        # ------------------------------------------------------------------
        # COMPETITOR EXPANSION (ChatGPT hypothesis test)
        # ------------------------------------------------------------------
        indices_extra = []
        n_pos_added = 0
        n_rand_added = 0
        n_hard_added = 0
        
        if self.compete_train:
            already_included = set(indices_core.tolist())
            
            # 1) POSITIVE CLOSURE: Include GT spatial neighbors not in core
            if self.compete_use_pos_closure and self.compete_k_pos > 0:
                pos_closure_candidates = []
                for local_i in range(n_core):
                    global_i = indices_core[local_i].item()
                    # Get GT spatial neighbors from targets.knn_spatial
                    gt_neighbors = targets.knn_spatial[global_i]  # (k,)
                    for neighbor in gt_neighbors.tolist():
                        if neighbor >= 0 and neighbor not in already_included:
                            pos_closure_candidates.append(neighbor)
                
                # Deduplicate and cap
                pos_closure_candidates = list(set(pos_closure_candidates))
                n_pos_max = self.compete_n_extra // 2  # Reserve half for pos closure
                if len(pos_closure_candidates) > n_pos_max:
                    perm = torch.randperm(len(pos_closure_candidates))[:n_pos_max]
                    pos_closure_candidates = [pos_closure_candidates[i] for i in perm.tolist()]
                
                if pos_closure_candidates:
                    pos_tensor = torch.tensor(pos_closure_candidates, dtype=torch.long)
                    indices_extra.append(pos_tensor)
                    n_pos_added = len(pos_closure_candidates)
                    already_included.update(pos_closure_candidates)
            
            # 2) RANDOM DISTRACTORS
            if self.compete_n_rand > 0:
                available = [i for i in range(m) if i not in already_included]
                n_rand = min(self.compete_n_rand, len(available))
                if n_rand > 0:
                    rand_idx = torch.randperm(len(available))[:n_rand]
                    rand_distractors = torch.tensor([available[i] for i in rand_idx.tolist()], 
                                                    dtype=torch.long)
                    indices_extra.append(rand_distractors)
                    n_rand_added = n_rand
                    already_included.update(rand_distractors.tolist())
            
            # 3) HARD NEGATIVES (expression-similar distractors)
            if self.compete_n_hard > 0 and slide_id in self.expr_knn_indices:
                expr_knn = self.expr_knn_indices[slide_id]  # (m, k_expr)
                hard_candidates = []
                
                # Sample hard negatives from expression neighbors of core points
                anchor_indices = indices_core[:min(10, n_core)]  # Use subset of anchors
                for anchor in anchor_indices.tolist():
                    expr_neighbors = expr_knn[anchor].tolist()
                    for neighbor in expr_neighbors:
                        if neighbor not in already_included:
                            hard_candidates.append(neighbor)
                
                hard_candidates = list(set(hard_candidates))
                n_hard = min(self.compete_n_hard, len(hard_candidates))
                if n_hard > 0:
                    perm = torch.randperm(len(hard_candidates))[:n_hard]
                    hard_distractors = torch.tensor([hard_candidates[i] for i in perm.tolist()],
                                                    dtype=torch.long)
                    indices_extra.append(hard_distractors)
                    n_hard_added = n_hard
        
        # Combine core + extra
        if indices_extra:
            indices_extra_cat = torch.cat(indices_extra)
            indices_all = torch.cat([indices_core, indices_extra_cat])
            indices_all = torch.unique(indices_all)  # Remove any duplicates
        else:
            indices_all = indices_core
        
        # Cap total size for memory
        n_total_max = n_core + self.compete_n_extra if self.compete_train else n_core
        if indices_all.numel() > n_total_max:
            # Keep all core points, subsample extras
            core_set = set(indices_core.tolist())
            extras_in_all = [i.item() for i in indices_all if i.item() not in core_set]
            n_extras_to_keep = n_total_max - n_core
            if n_extras_to_keep > 0 and len(extras_in_all) > n_extras_to_keep:
                perm = torch.randperm(len(extras_in_all))[:n_extras_to_keep]
                extras_to_keep = [extras_in_all[i] for i in perm.tolist()]
            else:
                extras_to_keep = extras_in_all[:n_extras_to_keep]
            indices_all = torch.cat([indices_core, torch.tensor(extras_to_keep, dtype=torch.long)])
        
        # Create anchor mask (True for core points, False for extras)
        anchor_mask = torch.zeros(indices_all.numel(), dtype=torch.bool)
        core_set = set(indices_core.tolist())
        for i, idx in enumerate(indices_all.tolist()):
            if idx in core_set:
                anchor_mask[i] = True
        
        # Use indices_all for all downstream processing
        indices = indices_all
        n_total = indices.numel()

        # Store actual base size BEFORE adding landmarks
        base_n = indices.shape[0]

        # ------------------------------------------------------------------
        # kNN mapping (same as before, but using indices which may include extras)
        # ------------------------------------------------------------------
        global_to_local = torch.full((m,), -1, dtype=torch.long)
        global_to_local[indices] = torch.arange(n_total, dtype=torch.long)

        knn_global = targets.knn_indices[indices]  # (n_total, k)
        knn_local = global_to_local[knn_global]

        knn_spatial_global = targets.knn_spatial[indices]
        knn_spatial_local = global_to_local[knn_spatial_global]

        # ------------------------------------------------------------------
        # Landmarks (unchanged)
        # ------------------------------------------------------------------
        if self.landmarks_L > 0:
            Z_subset = self.Z_dict[slide_id][indices]
            n_landmarks = min(self.landmarks_L, base_n)
            landmark_indices = uet.farthest_point_sampling(Z_subset, n_landmarks)
            landmark_global = indices[landmark_indices]
            indices = torch.cat([indices, landmark_global])

            is_landmark = torch.zeros(indices.shape[0], dtype=torch.bool)
            is_landmark[base_n:] = True
            
            # Extend anchor_mask for landmarks (landmarks are NOT anchors)
            anchor_mask = torch.cat([anchor_mask, torch.zeros(n_landmarks, dtype=torch.bool)])
        else:
            is_landmark = torch.zeros(base_n, dtype=torch.bool)

        overlap_info = {
            'slide_id': slide_id,
            'indices': indices,
        }

        # ------------------------------------------------------------------
        # Extract embeddings + coords for this FINAL index set
        # (needed for anchor sampling below and for downstream targets)
        # ------------------------------------------------------------------
        Z_set = self.Z_dict[slide_id][indices]          # (n_total_final, h_dim)
        y_hat_subset = targets.y_hat[indices]           # (n_total_final, d)

        # ========== NEW: Generate anchor_cond_mask for anchored training ==========
        n_total_final = indices.numel()
        anchor_cond_mask = torch.zeros(n_total_final, dtype=torch.bool)
        anchor_cond_debug = {
            'mode': 'none',
            'n_anchor': 0,
            'n_total': n_total_final,
            'n_landmarks': int(is_landmark.sum().item()) if is_landmark is not None else 0,
            'seed_idx': -1,
        }

        if self.anchor_train:
            # Decide: anchored vs unanchored for this sample
            u = torch.rand(1).item()

            if u < self.anchor_p_uncond:
                # Unanchored: all points unknown
                anchor_cond_debug['mode'] = 'uncond'
            else:
                # Anchored: sample some points as anchors
                eligible = torch.ones(n_total_final, dtype=torch.bool)
                if self.anchor_exclude_landmarks and is_landmark is not None:
                    eligible &= (~is_landmark)

                n_eligible = int(eligible.sum().item())
                if n_eligible > 0:
                    frac = torch.empty(1).uniform_(self.anchor_frac_min, self.anchor_frac_max).item()
                    n_anchor_raw = int(round(frac * n_eligible))
                    n_anchor = max(self.anchor_min, min(self.anchor_max, n_anchor_raw, n_eligible))

                    # Ball mode uses 2D coords (or full coords if <2 dims)
                    y_xy = y_hat_subset[:, :2] if y_hat_subset.shape[1] >= 2 else y_hat_subset

                    # kNN BFS mode uses knn_spatial_local which is already defined above
                    knn_for_bfs = knn_spatial_local

                    if self.anchor_mode == "ball":
                        anchor_cond_mask, seed_idx = uet.sample_anchor_mask_ball(y_xy, eligible, n_anchor)
                        anchor_cond_debug['seed_idx'] = int(seed_idx)
                    elif self.anchor_mode == "uniform":
                        anchor_cond_mask = uet.sample_anchor_mask_uniform(eligible, n_anchor)
                    elif self.anchor_mode == "knn_bfs":
                        # If knn_for_bfs is not usable, fall back to uniform
                        if knn_for_bfs is not None and knn_for_bfs.numel() > 0:
                            anchor_cond_mask, seed_idx = uet.sample_anchor_mask_knn_bfs(
                                knn_for_bfs, eligible, n_anchor
                            )
                            anchor_cond_debug['seed_idx'] = int(seed_idx)
                        else:
                            anchor_cond_mask = uet.sample_anchor_mask_uniform(eligible, n_anchor)
                    else:
                        # Fallback
                        anchor_cond_mask = uet.sample_anchor_mask_uniform(eligible, n_anchor)

                    anchor_cond_debug['mode'] = self.anchor_mode
                    anchor_cond_debug['n_anchor'] = int(anchor_cond_mask.sum().item())

            if self.debug_anchor_dataset and idx < 3:
                print(f"[ANCHOR-DATASET] idx={idx} mode={anchor_cond_debug['mode']} "
                    f"n_anchor={anchor_cond_debug['n_anchor']}/{n_total_final} "
                    f"n_landmarks={anchor_cond_debug['n_landmarks']}")


        y_hat_centered = y_hat_subset - y_hat_subset.mean(dim=0, keepdim=True)
        G_subset = y_hat_centered @ y_hat_centered.t()

        D_subset = targets.D[indices][:, indices]

        V_target = uet.factor_from_gram(G_subset, self.D_latent)

        edge_index, edge_weight = uet.build_knn_graph(y_hat_subset, k=self.knn_k)
        L_subset = uet.compute_graph_laplacian(edge_index, edge_weight, n_nodes=len(indices))

        triu_mask = torch.triu(torch.ones_like(D_subset, dtype=torch.bool), diagonal=1)
        d_95 = torch.quantile(D_subset[triu_mask], 0.95)
        bins = torch.linspace(0, float(d_95), 64)
        H_subset = uet.compute_distance_hist(D_subset, bins)

        n_nodes = len(indices)
        triplets_subset = uet.sample_ordinal_triplets(D_subset, n_triplets=min(500, n_nodes), margin_ratio=0.05)

        L_info = {
            'L': L_subset,
            't_list': targets.t_list
        }

        # Compute PH for topology
        y_hat_miniset = targets.y_hat[indices].cpu().numpy()
        topo_result = uet.compute_persistent_pairs(
            y_hat_miniset,
            max_pairs_0d=min(20, n_nodes // 4),
            max_pairs_1d=min(10, n_nodes // 8)
        )
        
        topo_info = {
            'pairs_0': torch.from_numpy(topo_result['pairs_0']).long(),
            'dists_0': torch.from_numpy(topo_result['dists_0']).float(),
            'pairs_1': torch.from_numpy(topo_result['pairs_1']).long(),
            'dists_1': torch.from_numpy(topo_result['dists_1']).float(),
        }

        # ------------------------------------------------------------------
        # COMPETITOR DEBUG COUNTERS
        # ------------------------------------------------------------------
        compete_debug = {
            'n_core': n_core,
            'n_pos': n_pos_added,
            'n_rand': n_rand_added,
            'n_hard': n_hard_added,
            'n_total': n_total,
        }

        # ========== SPOT IDENTITY HARDENING: Pack slide_id into global UID ==========
        # uid = (slide_id << 32) + within_slide_idx (int64)
        # This guarantees cross-slide uniqueness for any future intersection/cache operations.
        # Decode: slide_id = uid >> 32, spot_idx = uid & 0xFFFFFFFF
        
        global_uid = (slide_id << 32) + indices.long()  # (n,) int64 UIDs
        spot_indices = indices  # Keep within-slide indices for targets_dict lookup
        
        overlap_info = {
            'slide_id': slide_id,
            'indices': indices,  # Original within-slide indices
            'global_uid': global_uid,  # NEW: Globally unique IDs
        }

        return {
            'Z_set': Z_set,
            'is_landmark': is_landmark,
            'V_target': V_target,
            'G_target': G_subset,
            'D_target': D_subset,
            'H_target': H_subset,
            'H_bins': bins,
            'L_info': L_info,
            'triplets': triplets_subset,
            'n': len(indices),
            'overlap_info': overlap_info,
            'knn_indices': knn_local,
            'topo_info': topo_info,
            'knn_spatial': knn_spatial_local,
            # ========== NEW: Competitor training fields ==========
            'anchor_mask': anchor_mask,
            'global_indices': global_uid,  # ✅ NOW TRULY GLOBAL (int64 UIDs)
            'spot_indices': spot_indices,  # NEW: Within-slide indices [0, m) for targets_dict
            'compete_debug': compete_debug,
            # ========== NEW: Anchored training fields ==========
            'anchor_cond_mask': anchor_cond_mask,
            'anchor_cond_debug': anchor_cond_debug,
        }

            
def collate_minisets(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate mini-sets with padding.
    """
    device = batch[0]['Z_set'].device
    n_max = max(item['n'] for item in batch)
    batch_size = len(batch)
    h_dim = batch[0]['Z_set'].shape[1]
    D_latent = batch[0]['V_target'].shape[1]
    num_bins = batch[0]['H_target'].shape[0]
        
    L_info_batch = []
    triplets_batch = []
    H_bins_batch = []
    overlap_info_batch = []
    topo_info_batch = [] 

    #init padded tensors
    Z_batch = torch.zeros(batch_size, n_max, h_dim, device=device)
    V_batch = torch.zeros(batch_size, n_max, D_latent, device=device)
    G_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    D_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    H_batch = torch.zeros(batch_size, num_bins, device=device)
    mask_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    n_batch = torch.zeros(batch_size, dtype=torch.long)
    is_landmark_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)  # ADD THIS

    knn_k = batch[0]['knn_indices'].shape[1]  # Get k from first item
    knn_batch = torch.full((batch_size, n_max, knn_k), -1, dtype=torch.long, device=device)
    knn_spatial_batch = torch.full((batch_size, n_max, knn_k), -1, dtype=torch.long, device=device)

    # ========== NEW: Competitor training batch tensors ==========
    anchor_mask_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    # ✅ HARDENING: global_indices now int64 (UIDs), spot_indices int64 (within-slide)
    global_indices_batch = torch.full((batch_size, n_max), -1, dtype=torch.int64, device=device)
    spot_indices_batch = torch.full((batch_size, n_max), -1, dtype=torch.int64, device=device)
    compete_debug_batch = []
    
    # ========== NEW: Anchored training batch tensors ==========
    anchor_cond_mask_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    anchor_cond_debug_batch = []


    for i, item in enumerate(batch):
        n = item['n']
        Z_batch[i, :n] = item['Z_set']
        V_batch[i, :n] = item['V_target']
        G_batch[i, :n, :n] = item['G_target']
        D_batch[i, :n, :n] = item['D_target']
        H_batch[i] = item['H_target']
        mask_batch[i, :n] = True
        n_batch[i] = n
        is_landmark_batch[i, :n] = item['is_landmark']  # ADD THIS
        L_info_batch.append(item['L_info'])
        triplets_batch.append(item['triplets'])
        H_bins_batch.append(item['H_bins'])
        overlap_info_batch.append(item['overlap_info'])
        topo_info_batch.append(item.get('topo_info', None))  # NEW!
        knn_batch[i, :n] = item['knn_indices']

        if 'knn_spatial' in item:
            knn_spatial_batch[i, :n, :] = item['knn_spatial'][:n, :]

        # ========== NEW: Competitor fields ==========
        # if 'anchor_mask' in item:
        #     anchor_mask_batch[i, :n] = item['anchor_mask'][:n]
        # else:
        #     anchor_mask_batch[i, :n] = True  # Default: all points are anchors
        
        # if 'global_indices' in item:
        #     global_indices_batch[i, :n] = item['global_indices'][:n]
        
        # if 'compete_debug' in item:
        #     compete_debug_batch.append(item['compete_debug'])

        # ========== NEW: Competitor fields ==========
        if 'anchor_mask' in item:
            anchor_mask_batch[i, :n] = item['anchor_mask'][:n]
        else:
            anchor_mask_batch[i, :n] = True  # Default: all points are anchors
        
        # ✅ HARDENING: Collate both UIDs and within-slide indices
        if 'global_indices' in item:
            global_indices_batch[i, :n] = item['global_indices'][:n].long()  # int64 UIDs
        
        if 'spot_indices' in item:
            spot_indices_batch[i, :n] = item['spot_indices'][:n].long()  # int64 within-slide
        
        if 'compete_debug' in item:
            compete_debug_batch.append(item['compete_debug'])

        
        # ========== NEW: Anchored training fields ==========
        if 'anchor_cond_mask' in item:
            anchor_cond_mask_batch[i, :n] = item['anchor_cond_mask'][:n]
        
        if 'anchor_cond_debug' in item:
            anchor_cond_debug_batch.append(item['anchor_cond_debug'])


    return {
        'Z_set': Z_batch,
        'V_target': V_batch,
        'G_target': G_batch,
        'D_target': D_batch,
        'H_target': H_batch,
        'L_info': L_info_batch,
        'triplets': triplets_batch,
        'H_bins': H_bins_batch,
        'mask': mask_batch,
        'n': n_batch,
        'overlap_info': overlap_info_batch,
        'is_landmark': is_landmark_batch,  # ADD THIS
        'is_sc': False,
        'knn_indices': knn_batch,
        'topo_info': topo_info_batch,
        'knn_spatial': knn_spatial_batch,
        # ========== NEW: Competitor training fields ==========
        'anchor_mask': anchor_mask_batch,
        'global_indices': global_indices_batch,  # ✅ int64 UIDs (slide-aware)
        'spot_indices': spot_indices_batch,      # NEW: Within-slide indices for targets_dict
        'compete_debug': compete_debug_batch,
        # ========== NEW: Anchored training fields ==========
        'anchor_cond_mask': anchor_cond_mask_batch,
        'anchor_cond_debug': anchor_cond_debug_batch,
    }


# ==============================================================================
# PAIRED OVERLAPPING MINISETS DATASET (Candidate 1)
# ==============================================================================

class STPairSetDataset(Dataset):
    """
    ST mini-sets with PAIRED overlapping cores for overlap-consistency training.

    Each __getitem__ returns TWO minisets from the SAME slide with a controlled
    overlap in their core points. This enables overlap-consistency losses that
    enforce the model produces consistent geometry for shared cells across
    different context sets.

    Key differences from STSetDataset:
    1. Returns a pair (view1, view2) instead of a single miniset
    2. Core points have controlled overlap (alpha fraction shared)
    3. Overlap mapping (idx1_I, idx2_I) tracks which indices correspond
    4. Extras are sampled independently for each view
    """

    def __init__(
        self,
        targets_dict: Dict[int, STTargets],
        encoder: 'SharedEncoder',
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_min: int = 64,
        n_max: int = 256,
        D_latent: int = 16,
        num_samples: int = 10000,
        knn_k: int = 12,
        device: str = 'cuda',
        landmarks_L: int = 32,
        pool_mult: float = 4.0,
        stochastic_tau: float = 1.0,
        # ========== Paired overlap params ==========
        pair_overlap_alpha: float = 0.5,
        pair_overlap_min_I: int = 16,
        # ========== FIX #4: Robust pairing params ==========
        pair_difficulty_probs: tuple = (0.5, 0.3, 0.2),  # (easy, medium, hard)
        pair_hard_alpha_range: tuple = (0.15, 0.25),     # alpha range for hard pairs
        pair_medium_center_dist_mult: float = 0.3,       # how far center2 is from center1 (fraction of pool radius)
        # ========== Competitor training params ==========
        compete_train: bool = False,
        compete_n_extra: int = 128,
        compete_n_rand: int = 64,
        compete_n_hard: int = 64,
        compete_use_pos_closure: bool = True,
        compete_k_pos: int = 10,
        compete_expr_knn_k: int = 50,
        compete_anchor_only: bool = True,
    ):
        self.targets_dict = targets_dict
        self.encoder = encoder
        self.st_gene_expr_dict = st_gene_expr_dict
        self.n_min = n_min
        self.n_max = n_max
        self.D_latent = D_latent
        self.num_samples = num_samples
        self.knn_k = knn_k
        self.device = device
        self.slide_ids = list(targets_dict.keys())
        self.landmarks_L = 0  # Disable landmarks for pair mode

        # Paired overlap params
        self.pair_overlap_alpha = pair_overlap_alpha
        self.pair_overlap_min_I = pair_overlap_min_I

        # FIX #4: Robust pairing params
        self.pair_difficulty_probs = pair_difficulty_probs
        self.pair_hard_alpha_range = pair_hard_alpha_range
        self.pair_medium_center_dist_mult = pair_medium_center_dist_mult

        # Stochastic sampling params
        self.pool_mult = pool_mult
        self.stochastic_tau = stochastic_tau

        # Competitor training params
        self.compete_train = compete_train
        self.compete_n_extra = compete_n_extra
        self.compete_n_rand = compete_n_rand
        self.compete_n_hard = compete_n_hard
        self.compete_use_pos_closure = compete_use_pos_closure
        self.compete_k_pos = compete_k_pos
        self.compete_expr_knn_k = compete_expr_knn_k
        self.compete_anchor_only = compete_anchor_only

        # FIX #4: Tracking stats for debug
        self.pair_difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}

        # Precompute encoder embeddings for all slides
        self.Z_dict = {}
        self.expr_knn_indices = {}

        with torch.no_grad():
            for slide_id, st_expr in st_gene_expr_dict.items():
                Z = encoder(st_expr.to(device))
                self.Z_dict[slide_id] = Z.cpu()

                # Precompute expression kNN for hard negatives
                if compete_train and compete_n_hard > 0:
                    n_spots = Z.shape[0]
                    Z_norm = F.normalize(Z, p=2, dim=1)
                    sim_matrix = Z_norm @ Z_norm.T
                    sim_matrix.fill_diagonal_(-float('inf'))
                    k_expr = min(compete_expr_knn_k, n_spots - 1)
                    _, expr_knn_idx = torch.topk(sim_matrix, k=k_expr, dim=1)
                    self.expr_knn_indices[slide_id] = expr_knn_idx.cpu()

        print(f"[STPairSetDataset] Created with alpha={pair_overlap_alpha}, "
              f"min_I={pair_overlap_min_I}, n_range=[{n_min},{n_max}]")
        print(f"[STPairSetDataset] FIX #4: Difficulty probs={pair_difficulty_probs} "
              f"(easy/medium/hard), hard_alpha={pair_hard_alpha_range}")

    def __len__(self):
        return self.num_samples

    def _sample_core_patch(self, targets, m, n_core, center_idx=None):
        """
        Sample a core patch of n_core points from a slide with m total points.
        Uses stochastic kNN sampling from a pool around the center.

        Returns: indices_core (tensor of indices), center_idx (int)
        """
        if center_idx is None:
            center_idx = np.random.randint(0, m)

        D_row = targets.D[center_idx]
        all_idx = torch.arange(m)
        mask_self = all_idx != center_idx
        dists = D_row[mask_self]
        idx_no_self = all_idx[mask_self]

        if idx_no_self.numel() == 0:
            return all_idx[:n_core], center_idx

        sort_order = torch.argsort(dists)
        sorted_neighbors = idx_no_self[sort_order]
        sorted_dists = dists[sort_order]

        n_neighbors_needed = n_core - 1
        K_pool = min(sorted_neighbors.numel(), int(self.pool_mult * n_core))
        K_pool = max(K_pool, n_neighbors_needed)

        pool = sorted_neighbors[:K_pool]
        pool_d = sorted_dists[:K_pool]

        if pool.numel() <= n_neighbors_needed:
            neighbors = pool
        else:
            weights = torch.softmax(-pool_d / self.stochastic_tau, dim=0)
            sampled_idx = torch.multinomial(weights, n_neighbors_needed, replacement=False)
            neighbors = pool[sampled_idx]

        indices_core = torch.cat([
            torch.tensor([center_idx], dtype=torch.long),
            neighbors
        ])

        return indices_core, center_idx

    def _add_competitor_extras(self, targets, indices_core, slide_id, m):
        """
        Add competitor extras (positive closure, random, hard negatives) to a core.
        Returns indices_all, anchor_mask
        """
        n_core = indices_core.numel()
        indices_extra = []
        already_included = set(indices_core.tolist())

        if not self.compete_train:
            anchor_mask = torch.ones(n_core, dtype=torch.bool)
            return indices_core, anchor_mask

        # 1) Positive closure
        if self.compete_use_pos_closure and self.compete_k_pos > 0:
            pos_closure_candidates = []
            for local_i in range(n_core):
                global_i = indices_core[local_i].item()
                gt_neighbors = targets.knn_spatial[global_i]
                for neighbor in gt_neighbors.tolist():
                    if neighbor >= 0 and neighbor not in already_included:
                        pos_closure_candidates.append(neighbor)

            pos_closure_candidates = list(set(pos_closure_candidates))
            n_pos_max = self.compete_n_extra // 2
            if len(pos_closure_candidates) > n_pos_max:
                perm = torch.randperm(len(pos_closure_candidates))[:n_pos_max]
                pos_closure_candidates = [pos_closure_candidates[i] for i in perm.tolist()]

            if pos_closure_candidates:
                pos_tensor = torch.tensor(pos_closure_candidates, dtype=torch.long)
                indices_extra.append(pos_tensor)
                already_included.update(pos_closure_candidates)

        # 2) Random distractors
        if self.compete_n_rand > 0:
            available = [i for i in range(m) if i not in already_included]
            n_rand = min(self.compete_n_rand, len(available))
            if n_rand > 0:
                rand_idx = torch.randperm(len(available))[:n_rand]
                rand_distractors = torch.tensor([available[i] for i in rand_idx.tolist()],
                                                dtype=torch.long)
                indices_extra.append(rand_distractors)
                already_included.update(rand_distractors.tolist())

        # 3) Hard negatives
        if self.compete_n_hard > 0 and slide_id in self.expr_knn_indices:
            expr_knn = self.expr_knn_indices[slide_id]
            hard_candidates = []
            anchor_indices = indices_core[:min(10, n_core)]
            for anchor in anchor_indices.tolist():
                expr_neighbors = expr_knn[anchor].tolist()
                for neighbor in expr_neighbors:
                    if neighbor not in already_included:
                        hard_candidates.append(neighbor)

            hard_candidates = list(set(hard_candidates))
            n_hard = min(self.compete_n_hard, len(hard_candidates))
            if n_hard > 0:
                perm = torch.randperm(len(hard_candidates))[:n_hard]
                hard_distractors = torch.tensor([hard_candidates[i] for i in perm.tolist()],
                                                dtype=torch.long)
                indices_extra.append(hard_distractors)

        # Combine core + extra
        if indices_extra:
            indices_extra_cat = torch.cat(indices_extra)
            indices_all = torch.cat([indices_core, indices_extra_cat])
            indices_all = torch.unique(indices_all)
        else:
            indices_all = indices_core

        # Cap total size
        n_total_max = n_core + self.compete_n_extra
        if indices_all.numel() > n_total_max:
            core_set = set(indices_core.tolist())
            extras_in_all = [i.item() for i in indices_all if i.item() not in core_set]
            n_extras_to_keep = n_total_max - n_core
            if n_extras_to_keep > 0 and len(extras_in_all) > n_extras_to_keep:
                perm = torch.randperm(len(extras_in_all))[:n_extras_to_keep]
                extras_to_keep = [extras_in_all[i] for i in perm.tolist()]
            else:
                extras_to_keep = extras_in_all[:n_extras_to_keep]
            indices_all = torch.cat([indices_core, torch.tensor(extras_to_keep, dtype=torch.long)])

        # Create anchor mask
        anchor_mask = torch.zeros(indices_all.numel(), dtype=torch.bool)
        core_set = set(indices_core.tolist())
        for i, idx in enumerate(indices_all.tolist()):
            if idx in core_set:
                anchor_mask[i] = True

        return indices_all, anchor_mask

    def _build_miniset_dict(self, targets, indices, anchor_mask, slide_id):
        """
        Build the miniset dictionary from indices (same structure as STSetDataset).
        """
        n_total = indices.numel()
        m = targets.y_hat.shape[0]

        # Build kNN mappings
        global_to_local = torch.full((m,), -1, dtype=torch.long)
        global_to_local[indices] = torch.arange(n_total, dtype=torch.long)

        knn_global = targets.knn_indices[indices]
        knn_local = global_to_local[knn_global]

        knn_spatial_global = targets.knn_spatial[indices]
        knn_spatial_local = global_to_local[knn_spatial_global]

        # Extract embeddings and coords
        Z_set = self.Z_dict[slide_id][indices]
        y_hat_subset = targets.y_hat[indices]

        # Compute targets
        y_hat_centered = y_hat_subset - y_hat_subset.mean(dim=0, keepdim=True)
        G_subset = y_hat_centered @ y_hat_centered.t()
        D_subset = targets.D[indices][:, indices]
        V_target = uet.factor_from_gram(G_subset, self.D_latent)

        # Histogram
        triu_mask = torch.triu(torch.ones_like(D_subset, dtype=torch.bool), diagonal=1)
        d_95 = torch.quantile(D_subset[triu_mask], 0.95) if triu_mask.any() else 1.0
        bins = torch.linspace(0, float(d_95), 64)
        H_subset = uet.compute_distance_hist(D_subset, bins)

        # Triplets
        triplets_subset = uet.sample_ordinal_triplets(D_subset, n_triplets=min(500, n_total), margin_ratio=0.05)

        # L_info
        edge_index, edge_weight = uet.build_knn_graph(y_hat_subset, k=self.knn_k)
        L_subset = uet.compute_graph_laplacian(edge_index, edge_weight, n_nodes=n_total)
        L_info = {'L': L_subset, 't_list': targets.t_list}

        # Topo info
        y_hat_miniset = targets.y_hat[indices].cpu().numpy()
        topo_result = uet.compute_persistent_pairs(
            y_hat_miniset,
            max_pairs_0d=min(20, n_total // 4),
            max_pairs_1d=min(10, n_total // 8)
        )
        topo_info = {
            'pairs_0': torch.from_numpy(topo_result['pairs_0']).long(),
            'dists_0': torch.from_numpy(topo_result['dists_0']).float(),
            'pairs_1': torch.from_numpy(topo_result['pairs_1']).long(),
            'dists_1': torch.from_numpy(topo_result['dists_1']).float(),
        }

        # Global UIDs
        global_uid = (slide_id << 32) + indices.long()

        # No landmarks in pair mode
        is_landmark = torch.zeros(n_total, dtype=torch.bool)

        # No anchor_cond_mask in pair mode (not using anchored training)
        anchor_cond_mask = torch.zeros(n_total, dtype=torch.bool)

        return {
            'Z_set': Z_set,
            'is_landmark': is_landmark,
            'V_target': V_target,
            'G_target': G_subset,
            'D_target': D_subset,
            'H_target': H_subset,
            'H_bins': bins,
            'L_info': L_info,
            'triplets': triplets_subset,
            'n': n_total,
            'overlap_info': {'slide_id': slide_id, 'indices': indices, 'global_uid': global_uid},
            'knn_indices': knn_local,
            'topo_info': topo_info,
            'knn_spatial': knn_spatial_local,
            'anchor_mask': anchor_mask,
            'global_indices': global_uid,
            'spot_indices': indices,
            'compete_debug': {},
            'anchor_cond_mask': anchor_cond_mask,
            'anchor_cond_debug': {},
        }

    def __getitem__(self, idx):
        """
        Return a PAIR of minisets from the same slide with controlled core overlap.

        FIX #4: Three-tier difficulty system for robust context invariance learning:
        - EASY (50%): Same center for both views, standard overlap (current behavior)
        - MEDIUM (30%): Same overlap set I, but exclusive points from different nearby center
        - HARD (20%): Smaller overlap alpha (0.15-0.25) to prevent overlap-only behavior
        """
        # Pick a slide
        slide_id = np.random.choice(self.slide_ids)
        targets = self.targets_dict[slide_id]
        m = targets.y_hat.shape[0]

        # Sample miniset size for both views
        n = np.random.randint(self.n_min, self.n_max + 1)
        n = min(n, m)

        # FIX #4: Select difficulty level
        p_easy, p_medium, p_hard = self.pair_difficulty_probs
        rand_val = np.random.random()
        if rand_val < p_easy:
            difficulty = 'easy'
        elif rand_val < p_easy + p_medium:
            difficulty = 'medium'
        else:
            difficulty = 'hard'

        # FIX #4: Adjust overlap alpha based on difficulty
        if difficulty == 'hard':
            # Use smaller alpha for hard pairs
            alpha_lo, alpha_hi = self.pair_hard_alpha_range
            effective_alpha = np.random.uniform(alpha_lo, alpha_hi)
        else:
            effective_alpha = self.pair_overlap_alpha

        # Compute overlap size
        n_overlap = max(self.pair_overlap_min_I, int(effective_alpha * n))

        # If n is too small to satisfy min overlap, bump n up
        if n - 4 < self.pair_overlap_min_I:
            n = min(m, self.pair_overlap_min_I + 4)

        n_overlap = min(n_overlap, n - 4)  # Leave room for non-overlapping points

        # Sample a center point for view1 (and view2 if easy mode)
        center_idx_1 = np.random.randint(0, m)

        # Build a shared pool of candidates near center1
        D_row_1 = targets.D[center_idx_1]
        all_idx = torch.arange(m)
        mask_self_1 = all_idx != center_idx_1
        dists_1 = D_row_1[mask_self_1]
        idx_no_self_1 = all_idx[mask_self_1]

        sort_order_1 = torch.argsort(dists_1)
        sorted_neighbors_1 = idx_no_self_1[sort_order_1]
        sorted_dists_1 = dists_1[sort_order_1]

        # Pool size: enough for 2 views with overlap
        n_unique_needed = 2 * n - n_overlap  # Total unique points across both views
        K_pool = min(sorted_neighbors_1.numel(), int(self.pool_mult * n_unique_needed))
        K_pool = max(K_pool, n_unique_needed)

        pool_1 = sorted_neighbors_1[:K_pool]
        pool_d_1 = sorted_dists_1[:K_pool]

        # Sample OVERLAP set (I) - shared core points from center1's neighborhood
        if pool_1.numel() >= n_overlap:
            # Stochastic sampling for overlap
            weights_I = torch.softmax(-pool_d_1 / self.stochastic_tau, dim=0)
            if pool_1.numel() >= n_overlap:
                sampled_I = torch.multinomial(weights_I, n_overlap, replacement=False)
                I_indices = pool_1[sampled_I]
            else:
                I_indices = pool_1[:n_overlap]
        else:
            I_indices = pool_1
            n_overlap = I_indices.numel()

        # Points available after overlap (from center1's pool)
        I_set = set(I_indices.tolist())
        remaining_pool_1 = torch.tensor([p.item() for p in pool_1 if p.item() not in I_set], dtype=torch.long)

        # FIX #4: For MEDIUM difficulty, find a second center nearby but different
        if difficulty == 'medium':
            # Find center2: a point near center1 but far enough to provide context shift
            # Use a neighbor at medium distance (30% of pool radius by default)
            pool_radius = sorted_dists_1[min(K_pool - 1, sorted_dists_1.numel() - 1)].item() if sorted_dists_1.numel() > 0 else 1.0
            target_dist = self.pair_medium_center_dist_mult * pool_radius

            # Find candidates at roughly the target distance
            dist_diffs = (sorted_dists_1 - target_dist).abs()
            # Pick from the 10 closest to target distance, excluding overlap points
            candidate_mask = torch.ones(sorted_neighbors_1.numel(), dtype=torch.bool)
            for i, neighbor in enumerate(sorted_neighbors_1.tolist()):
                if neighbor in I_set or neighbor == center_idx_1:
                    candidate_mask[i] = False

            valid_candidates = sorted_neighbors_1[candidate_mask]
            valid_dist_diffs = dist_diffs[candidate_mask[:dist_diffs.numel()]] if dist_diffs.numel() > 0 else torch.tensor([])

            if valid_candidates.numel() > 0:
                # Pick one of the closest candidates to target distance
                n_top = min(10, valid_candidates.numel())
                _, top_indices = valid_dist_diffs[:valid_candidates.numel()].topk(n_top, largest=False)
                chosen_idx = top_indices[np.random.randint(0, n_top)]
                center_idx_2 = valid_candidates[chosen_idx].item()
            else:
                # Fallback: just use a random point not in overlap
                available = [i for i in range(m) if i not in I_set and i != center_idx_1]
                if available:
                    center_idx_2 = np.random.choice(available)
                else:
                    center_idx_2 = center_idx_1  # Fallback to same center

            # Build pool around center2 for view2's exclusive points
            D_row_2 = targets.D[center_idx_2]
            mask_self_2 = all_idx != center_idx_2
            dists_2 = D_row_2[mask_self_2]
            idx_no_self_2 = all_idx[mask_self_2]

            sort_order_2 = torch.argsort(dists_2)
            sorted_neighbors_2 = idx_no_self_2[sort_order_2]

            # Remaining pool for view2: neighbors of center2, excluding overlap
            remaining_pool_2 = torch.tensor(
                [p.item() for p in sorted_neighbors_2[:K_pool] if p.item() not in I_set and p.item() != center_idx_2],
                dtype=torch.long
            )
        else:
            # EASY or HARD: same center for both views
            center_idx_2 = center_idx_1
            remaining_pool_2 = remaining_pool_1

        # Sample non-overlapping points for view1 (A) and view2 (B)
        n_A = n - n_overlap - 1  # -1 for center
        n_B = n - n_overlap - 1

        # View1 exclusive: from center1's pool
        if remaining_pool_1.numel() >= n_A:
            perm_1 = torch.randperm(remaining_pool_1.numel())
            A_exclusive = remaining_pool_1[perm_1[:n_A]]
        else:
            A_exclusive = remaining_pool_1

        # View2 exclusive: from center2's pool (different in MEDIUM mode)
        if difficulty == 'medium':
            # Use center2's pool, ensuring no overlap with A_exclusive
            A_set = set(A_exclusive.tolist())
            available_for_B = torch.tensor(
                [p.item() for p in remaining_pool_2 if p.item() not in A_set],
                dtype=torch.long
            )
            if available_for_B.numel() >= n_B:
                perm_2 = torch.randperm(available_for_B.numel())
                B_exclusive = available_for_B[perm_2[:n_B]]
            else:
                B_exclusive = available_for_B
        else:
            # EASY or HARD: split remaining_pool_1 between A and B
            if remaining_pool_1.numel() >= n_A + n_B:
                perm = torch.randperm(remaining_pool_1.numel())
                A_exclusive = remaining_pool_1[perm[:n_A]]
                B_exclusive = remaining_pool_1[perm[n_A:n_A + n_B]]
            else:
                half = remaining_pool_1.numel() // 2
                A_exclusive = remaining_pool_1[:half]
                B_exclusive = remaining_pool_1[half:]

        # Build core indices for each view
        center_tensor_1 = torch.tensor([center_idx_1], dtype=torch.long)
        center_tensor_2 = torch.tensor([center_idx_2], dtype=torch.long)
        indices_core_1 = torch.cat([center_tensor_1, I_indices, A_exclusive])
        indices_core_2 = torch.cat([center_tensor_2, I_indices, B_exclusive])

        # Shuffle cores
        indices_core_1 = indices_core_1[torch.randperm(indices_core_1.numel())]
        indices_core_2 = indices_core_2[torch.randperm(indices_core_2.numel())]

        # Add competitor extras independently to each view
        indices_all_1, anchor_mask_1 = self._add_competitor_extras(
            targets, indices_core_1, slide_id, m
        )
        indices_all_2, anchor_mask_2 = self._add_competitor_extras(
            targets, indices_core_2, slide_id, m
        )

        # Build miniset dicts
        view1 = self._build_miniset_dict(targets, indices_all_1, anchor_mask_1, slide_id)
        view2 = self._build_miniset_dict(targets, indices_all_2, anchor_mask_2, slide_id)

        # Compute overlap mapping (using global_uid for safety)
        # For EASY/HARD: Include center_idx as shared overlap point
        # For MEDIUM: centers are different, so only I_indices are overlap
        if difficulty == 'medium':
            I_indices_for_overlap = I_indices  # Centers are different, not shared
        else:
            I_indices_for_overlap = torch.cat([I_indices, torch.tensor([center_idx_1], dtype=torch.long)])

        I_global_uids = (slide_id << 32) + I_indices_for_overlap.long()

        # Build index mapping: for each point in I, find its position in view1 and view2
        idx1_I = []
        idx2_I = []

        global_uid_1 = view1['global_indices']
        global_uid_2 = view2['global_indices']

        uid_to_pos_1 = {uid.item(): pos for pos, uid in enumerate(global_uid_1)}
        uid_to_pos_2 = {uid.item(): pos for pos, uid in enumerate(global_uid_2)}

        for uid in I_global_uids.tolist():
            if uid in uid_to_pos_1 and uid in uid_to_pos_2:
                idx1_I.append(uid_to_pos_1[uid])
                idx2_I.append(uid_to_pos_2[uid])

        idx1_I = torch.tensor(idx1_I, dtype=torch.long)
        idx2_I = torch.tensor(idx2_I, dtype=torch.long)

        # FIX #4: Track difficulty counts (thread-safe increment)
        self.pair_difficulty_counts[difficulty] += 1

        # Overlap info
        overlap_mapping = {
            'idx1_I': idx1_I,  # Positions in view1 of shared points
            'idx2_I': idx2_I,  # Positions in view2 of shared points
            'I_size': len(idx1_I),
            'slide_id': slide_id,
            'difficulty': difficulty,  # FIX #4: Track difficulty for debugging
            'effective_alpha': effective_alpha,  # FIX #4: Track actual alpha used
        }

        return {
            'view1': view1,
            'view2': view2,
            'overlap_mapping': overlap_mapping,
        }


def collate_pair_minisets(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate paired minisets with padding.

    Returns a batch dict with:
    - 'view1': collated batch for first view (same structure as collate_minisets output)
    - 'view2': collated batch for second view
    - 'idx1_I': (batch_size, max_I) positions in view1 of overlap points
    - 'idx2_I': (batch_size, max_I) positions in view2 of overlap points
    - 'I_mask': (batch_size, max_I) validity mask for overlap points
    - 'I_sizes': (batch_size,) number of overlap points per sample
    - 'difficulty_counts': FIX #4 - dict with counts of easy/medium/hard pairs in batch
    - 'effective_alphas': FIX #4 - list of effective alpha values used
    """
    batch_size = len(batch)

    # Collect views
    view1_list = [item['view1'] for item in batch]
    view2_list = [item['view2'] for item in batch]
    overlap_list = [item['overlap_mapping'] for item in batch]

    # Collate each view using standard collate
    view1_batch = collate_minisets(view1_list)
    view2_batch = collate_minisets(view2_list)

    # Collate overlap mappings
    device = view1_batch['Z_set'].device
    max_I = max(om['I_size'] for om in overlap_list)
    max_I = max(max_I, 1)  # At least 1 to avoid empty tensors

    idx1_I_batch = torch.full((batch_size, max_I), -1, dtype=torch.long, device=device)
    idx2_I_batch = torch.full((batch_size, max_I), -1, dtype=torch.long, device=device)
    I_mask_batch = torch.zeros(batch_size, max_I, dtype=torch.bool, device=device)
    I_sizes_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

    # FIX #4: Track difficulty distribution in batch
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    effective_alphas = []

    for i, om in enumerate(overlap_list):
        I_size = om['I_size']
        if I_size > 0:
            idx1_I_batch[i, :I_size] = om['idx1_I']
            idx2_I_batch[i, :I_size] = om['idx2_I']
            I_mask_batch[i, :I_size] = True
        I_sizes_batch[i] = I_size

        # FIX #4: Count difficulties
        difficulty = om.get('difficulty', 'easy')
        difficulty_counts[difficulty] += 1
        effective_alphas.append(om.get('effective_alpha', 0.5))

    return {
        'view1': view1_batch,
        'view2': view2_batch,
        'idx1_I': idx1_I_batch,
        'idx2_I': idx2_I_batch,
        'I_mask': I_mask_batch,
        'I_sizes': I_sizes_batch,
        'is_pair_batch': True,
        # FIX #4: Include difficulty info
        'difficulty_counts': difficulty_counts,
        'effective_alphas': effective_alphas,
    }


# class SCSetDataset(Dataset):
#     """
#     SC mini-sets with intentional overlap, using a prebuilt K-NN index.
#     Returns CPU-pinned tensors; collate moves to CUDA in batch.
#     """
#     def __init__(
#         self,
#         sc_gene_expr: torch.Tensor,
#         encoder: "SharedEncoder",
#         n_min: int = 64,
#         n_max: int = 256,
#         n_large_min: int = 384,
#         n_large_max: int = 512,
#         large_fraction: float = 0.15,
#         overlap_min: int = 20,
#         overlap_max: int = 512,
#         num_samples: int = 5000,
#         K_nbrs: int = 2048,
#         device: str = "cuda",
#         landmarks_L: int = 32
#     ):
#         self.n_min = n_min
#         self.n_max = n_max
#         self.n_large_min = n_min
#         self.n_large_max = n_max
#         self.large_fraction = large_fraction
#         self.overlap_min = overlap_min
#         self.overlap_max = overlap_max
#         self.num_samples = num_samples
#         self.device = device
#         self.landmarks_L = 0

#         # Encode all SC cells once (CUDA), then keep a CPU pinned copy
#         print("encoding SC cells....")
#         encoder.eval()
#         with torch.no_grad():
#             chunks, bs = [], 1024
#             for s in range(0, sc_gene_expr.shape[0], bs):
#                 z = encoder(sc_gene_expr[s:s+bs].to(device))
#                 chunks.append(z)
#             Z_all = torch.cat(chunks, 0).contiguous()
#         print(f"SC embeddings computed: {Z_all.shape}")

#         self.h_dim = Z_all.shape[1]
#         self.Z_cpu = Z_all.detach().to("cpu", non_blocking=True).pin_memory()

#         # Build neighbor index once (CPU pinned)
#         self.nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs, block=min(4096, Z_all.shape[0]))

#         # Compute embedding-based kNN indices (for NCA loss)
#         print(f"[SC Dataset] Computing embedding-based kNN indices...")
#         n_sc = self.Z_cpu.shape[0]
#         knn_k = min(K_nbrs, 20)  # Use smaller k for NCA loss

#         # Compute distance matrix in chunks to save memory
#         self.knn_indices = torch.zeros(n_sc, knn_k, dtype=torch.long)

#         chunk_size = 2048
#         for i in range(0, n_sc, chunk_size):
#             end_i = min(i + chunk_size, n_sc)
#             D_chunk = torch.cdist(self.Z_cpu[i:end_i], self.Z_cpu)
            
#             for local_i, global_i in enumerate(range(i, end_i)):
#                 dists_from_i = D_chunk[local_i]
#                 dists_from_i_copy = dists_from_i.clone()
#                 dists_from_i_copy[global_i] = float('inf')  # Exclude self
#                 _, indices = torch.topk(dists_from_i_copy, k=min(knn_k, n_sc-1), largest=False)
#                 self.knn_indices[global_i, :len(indices)] = indices
#                 if len(indices) < knn_k:
#                     self.knn_indices[global_i, len(indices):] = -1

#         print(f"[SC Dataset] kNN indices computed: {self.knn_indices.shape}")

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         N = self.Z_cpu.shape[0]
#         # Decide set size
#         if torch.rand(1).item() < self.large_fraction:
#             n_set = int(torch.randint(self.n_large_min, self.n_large_max + 1, (1,)).item())
#         else:
#             n_set = int(torch.randint(self.n_min, self.n_max + 1, (1,)).item())
#         n_set = min(n_set, N)

#         # Overlap size
#         n_ov_max = min(self.overlap_max, n_set // 2 if n_set >= 2 else 0)
#         n_overlap = int(torch.randint(self.overlap_min, max(self.overlap_min + 1, n_ov_max + 1), (1,)).item()) if n_ov_max >= self.overlap_min else 0

#         # Seed & neighbor-based sets (CPU index)
#         i = int(torch.randint(N, (1,)).item())
#         A = self.nbr_idx[i, :n_set]        # (n_set,) cpu long

#         if n_overlap > 0:
#             ov_pos = torch.randperm(n_set)[:n_overlap]         # positions in A
#             shared_global = A[ov_pos]
#         else:
#             ov_pos = torch.tensor([], dtype=torch.long)
#             shared_global = ov_pos

#         # pick nearby seed j
#         k_nn = min(50, self.nbr_idx.shape[1] - 1) if self.nbr_idx.shape[1] > 1 else 0
#         if k_nn > 0:
#             near = self.nbr_idx[i, 1:k_nn+1]
#             j = int(near[torch.randint(len(near), (1,))].item())
#         else:
#             j = (i + 1) % N

#         cand = self.nbr_idx[j]             # (K,)
#         # remove A \ overlap
#         non_sharedA_pos = torch.arange(n_set)
#         if n_overlap > 0:
#             non_sharedA_pos = non_sharedA_pos[~torch.isin(non_sharedA_pos, ov_pos)]
#         # non_shared_A = A[non_sharedA_pos]
#         # mask = ~torch.isin(cand, non_shared_A)
#         # cand = cand[mask]

#         # n_new = n_set - n_overlap
#         # new_B = cand[:n_new]
#         # B = torch.cat([shared_global, new_B])[:n_set]
#         mask = ~torch.isin(cand, A)
#         cand = cand[mask]

#         n_new = n_set - n_overlap
#         new_B = cand[:n_new]
#         B = torch.cat([shared_global, new_B])[:n_set]

#         # positions of shared in A and B
#         mapB = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
#         mapB[B] = torch.arange(B.numel(), dtype=torch.long)
#         shared_A_pos = ov_pos
#         shared_B_pos = mapB[shared_global]

#         # Add landmarks from union (but cap total size at n_set)
#         if self.landmarks_L > 0:
#             # Get union of A and B
#             union = torch.unique(torch.cat([A, B]))
#             Z_union = self.Z_cpu[union]

#             # FPS on union to select landmarks
#             landmark_local = uet.farthest_point_sampling(
#                 Z_union, 
#                 min(self.landmarks_L, len(union)),
#                 device='cpu'
#             )

#             landmark_global = union[landmark_local]
            
#             # Remove any landmarks already in A or B (prevent duplicates)
#             new_landmarks_A = landmark_global[~torch.isin(landmark_global, A)]
#             new_landmarks_B = landmark_global[~torch.isin(landmark_global, B)]

#             # Cap landmarks to ensure final size doesn't exceed n_set
#             space_left_A = n_set - A.numel()
#             space_left_B = n_set - B.numel()
            
#             new_landmarks_A = new_landmarks_A[:space_left_A]
#             new_landmarks_B = new_landmarks_B[:space_left_B]

#             # Append only NEW landmarks to both sets
#             A = torch.cat([A, new_landmarks_A])
#             B = torch.cat([B, new_landmarks_B])

#             # Update shared indices to include landmarks
#             n_landmarks = len(landmark_global)
#             landmark_pos_A = torch.arange(len(A) - len(new_landmarks_A), len(A))
#             landmark_pos_B = torch.arange(len(B) - len(new_landmarks_B), len(B))

#             shared_A_pos = torch.cat([shared_A_pos, landmark_pos_A])
#             shared_B_pos = torch.cat([shared_B_pos, landmark_pos_B])

#             # Create is_landmark masks
#             is_landmark_A = torch.zeros(len(A), dtype=torch.bool)
#             is_landmark_A[-len(new_landmarks_A):] = True
#             is_landmark_B = torch.zeros(len(B), dtype=torch.bool)
#             is_landmark_B[-len(new_landmarks_B):] = True
#         else:
#             is_landmark_A = torch.zeros(len(A), dtype=torch.bool)
#             is_landmark_B = torch.zeros(len(B), dtype=torch.bool)

#         # Pull CPU pinned slices (do NOT move to CUDA here)
#         Z_A = self.Z_cpu[A]
#         Z_B = self.Z_cpu[B]

#         # Duplicate guards
#         assert A.numel() == A.unique().numel(), f"Set A has duplicates: {A.numel()} vs {A.unique().numel()}"
#         assert B.numel() == B.unique().numel(), f"Set B has duplicates: {B.numel()} vs {B.unique().numel()}"

#         # Extract kNN indices for minisets A and B, map to local indexing
#         global_to_local_A = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
#         global_to_local_A[A] = torch.arange(len(A), dtype=torch.long)
#         knn_global_A = self.knn_indices[A]  # (n_A, k)
#         knn_local_A = global_to_local_A[knn_global_A]  # Map to local indices

#         global_to_local_B = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
#         global_to_local_B[B] = torch.arange(len(B), dtype=torch.long)
#         knn_global_B = self.knn_indices[B]  # (n_B, k)
#         knn_local_B = global_to_local_B[knn_global_B]  # Map to local indices

#         return {
#             "Z_A": Z_A, "Z_B": Z_B,
#             "n_A": int(A.numel()), "n_B": int(B.numel()),
#             "shared_A": shared_A_pos,
#             "shared_B": shared_B_pos,
#             "is_sc": True,
#             "is_landmark_A": is_landmark_A,  # ADD THIS
#             "is_landmark_B": is_landmark_B,  # ADD THIS
#             "global_indices_A": A,  # ADD THIS
#             "global_indices_B": B,   # ADD THIS
#             "knn_indices_A": knn_local_A,  # ADD THIS
#             "knn_indices_B": knn_local_B 
#         }
    

class SCSetDataset(Dataset):
    """
    SC mini-sets sampled like STSetDataset:
    - pick random center
    - build local patch using embedding-space distances
    - stochastic sampling from a distance-weighted pool
    """
    def __init__(
        self,
        sc_gene_expr: torch.Tensor,
        encoder: "SharedEncoder",
        n_min: int = 64,
        n_max: int = 256,
        num_samples: int = 5000,
        knn_k: int = 12,
        device: str = "cuda",
        landmarks_L: int = 0,
        pool_mult: float = 4.0,
        stochastic_tau: float = 1.0,
        K_nbrs: int = 2048,
    ):
        self.n_min = n_min
        self.n_max = n_max
        self.num_samples = num_samples
        self.device = device
        self.landmarks_L = landmarks_L

        self.pool_mult = pool_mult
        self.stochastic_tau = stochastic_tau

        # Encode all SC cells once (CUDA), then keep a CPU pinned copy
        print("encoding SC cells....")
        encoder.eval()
        with torch.no_grad():
            chunks, bs = [], 1024
            for s in range(0, sc_gene_expr.shape[0], bs):
                z = encoder(sc_gene_expr[s:s+bs].to(device))
                chunks.append(z)
            Z_all = torch.cat(chunks, 0).contiguous()
        print(f"SC embeddings computed: {Z_all.shape}")

        self.h_dim = Z_all.shape[1]
        self.Z_cpu = Z_all.detach().to("cpu", non_blocking=True).pin_memory()

        # Optional: precompute a big neighbor index (not required for sampling logic)
        self.nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs, block=min(4096, Z_all.shape[0]))

        # Compute embedding-based kNN indices (for NCA loss or downstream graph use)
        print(f"[SC Dataset] Computing embedding-based kNN indices...")
        n_sc = self.Z_cpu.shape[0]
        knn_k = min(knn_k, n_sc - 1) if n_sc > 1 else 0
        self.knn_indices = torch.zeros(n_sc, knn_k, dtype=torch.long)

        chunk_size = 2048
        for i in range(0, n_sc, chunk_size):
            end_i = min(i + chunk_size, n_sc)
            D_chunk = torch.cdist(self.Z_cpu[i:end_i], self.Z_cpu)

            for local_i, global_i in enumerate(range(i, end_i)):
                dists_from_i = D_chunk[local_i]
                dists_from_i_copy = dists_from_i.clone()
                dists_from_i_copy[global_i] = float('inf')  # Exclude self
                _, indices = torch.topk(dists_from_i_copy, k=knn_k, largest=False)
                self.knn_indices[global_i, :len(indices)] = indices
        print(f"[SC Dataset] kNN indices computed: {self.knn_indices.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        N = self.Z_cpu.shape[0]
        n_set = int(torch.randint(self.n_min, self.n_max + 1, (1,)).item())
        n_set = min(n_set, N)

        # Pick center
        center_idx = int(torch.randint(N, (1,)).item())

        # Distance from center in embedding space
        center_vec = self.Z_cpu[center_idx].unsqueeze(0)
        dists = torch.cdist(center_vec, self.Z_cpu).squeeze(0)  # (N,)

        all_idx = torch.arange(N)
        mask_self = all_idx != center_idx
        idx_no_self = all_idx[mask_self]
        dists_no_self = dists[mask_self]

        if idx_no_self.numel() == 0:
            indices_core = all_idx[:n_set]
        else:
            sort_order = torch.argsort(dists_no_self)
            sorted_neighbors = idx_no_self[sort_order]
            sorted_dists = dists_no_self[sort_order]

            n_neighbors_needed = n_set - 1
            K_pool = min(sorted_neighbors.numel(), int(self.pool_mult * n_set))
            K_pool = max(K_pool, n_neighbors_needed)

            pool = sorted_neighbors[:K_pool]
            pool_d = sorted_dists[:K_pool]

            if pool.numel() <= n_neighbors_needed:
                neighbors = pool
            else:
                weights = torch.softmax(-pool_d / self.stochastic_tau, dim=0)
                sampled_idx = torch.multinomial(weights, n_neighbors_needed, replacement=False)
                neighbors = pool[sampled_idx]

            indices_core = torch.cat([
                torch.tensor([center_idx], dtype=torch.long),
                neighbors
            ])

            # If still short, pad with random unused points
            if indices_core.numel() < n_set:
                missing = n_set - indices_core.numel()
                mask_extra = ~torch.isin(all_idx, indices_core)
                extra_pool = all_idx[mask_extra]
                if extra_pool.numel() > 0:
                    perm = torch.randperm(extra_pool.numel())
                    add = extra_pool[perm[:min(missing, extra_pool.numel())]]
                    indices_core = torch.cat([indices_core, add])

        # Shuffle to avoid positional bias
        indices = indices_core[torch.randperm(indices_core.numel())]
        n_total = indices.numel()

        # Optional landmarks
        if self.landmarks_L > 0:
            Z_subset = self.Z_cpu[indices]
            n_landmarks = min(self.landmarks_L, n_total)
            landmark_indices = uet.farthest_point_sampling(Z_subset, n_landmarks, device='cpu')
            landmark_global = indices[landmark_indices]
            indices = torch.cat([indices, landmark_global])

            is_landmark = torch.zeros(indices.shape[0], dtype=torch.bool)
            is_landmark[n_total:] = True
        else:
            is_landmark = torch.zeros(n_total, dtype=torch.bool)

        # Extract Z_set
        Z_set = self.Z_cpu[indices]

        # Map kNN to local indices
        global_to_local = torch.full((N,), -1, dtype=torch.long)
        global_to_local[indices] = torch.arange(indices.numel(), dtype=torch.long)
        knn_global = self.knn_indices[indices]  # (n, k)
        knn_local = global_to_local[knn_global]

        return {
            "Z_set": Z_set,
            "n": int(indices.numel()),
            "is_sc": True,
            "is_landmark": is_landmark,
            "global_indices": indices,
            "knn_indices": knn_local,
        }


from typing import List, Dict


def collate_sc_minisets(batch: List[Dict]) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = len(batch)
    n_max_A = max(x["n_A"] for x in batch)
    n_max_B = max(x["n_B"] for x in batch)
    n_max = max(n_max_A, n_max_B)
    h_dim = batch[0]["Z_A"].shape[1]

    Z_batch = torch.zeros(B * 2, n_max, h_dim, device=device)
    mask = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)
    n_vec = torch.zeros(B * 2, dtype=torch.long, device=device)

    # keep your global indices tensor
    global_indices = torch.full((B * 2, n_max), -1, dtype=torch.long, device=device)

    # Overlap tensors (padded)
    Kmax = max((len(x["shared_A"]) for x in batch), default=0)
    shared_A_idx = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    shared_B_idx = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    shared_len   = torch.zeros(B, dtype=torch.long, device=device)

    # --- these two ARE the pair indices the debug trainer wants ---
    pair_idxA = torch.arange(0, 2 * B, 2, device=device, dtype=torch.long)
    pair_idxB = torch.arange(1, 2 * B, 2, device=device, dtype=torch.long)

    # we'll also build the older "shared_info" list for core_models_et_p2.py
    shared_info: List[Dict] = []

    for i, item in enumerate(batch):
        nA, nB = item["n_A"], item["n_B"]

        ZA = item["Z_A"].to(device, non_blocking=True)
        ZB = item["Z_B"].to(device, non_blocking=True)

        Z_batch[2*i,   :nA] = ZA
        Z_batch[2*i+1, :nB] = ZB
        mask[2*i,   :nA] = True
        mask[2*i+1, :nB] = True
        n_vec[2*i]   = nA
        n_vec[2*i+1] = nB

        # global ids (if you produce them; fine if you don't)
        if "global_indices_A" in item:
            global_indices[2*i,   :nA] = item["global_indices_A"].to(device, non_blocking=True)
        if "global_indices_B" in item:
            global_indices[2*i+1, :nB] = item["global_indices_B"].to(device, non_blocking=True)

        # overlaps (local positions within each set)
        k = len(item["shared_A"])
        if k > 0:
            shared_A_idx[i, :k] = item["shared_A"].to(device, non_blocking=True)
            shared_B_idx[i, :k] = item["shared_B"].to(device, non_blocking=True)
            shared_len[i] = k

        # build old-style entry too (for core_models_et_p2.py)
        shared_info.append({
            "idx_A": int(2*i),
            "idx_B": int(2*i+1),
            # keep these on CPU; the trainer .to(device)s them later
            "shared_A": item["shared_A"].detach().cpu(),
            "shared_B": item["shared_B"].detach().cpu(),
        })

    # Landmark masks (optional; keep your existing logic)
    if 'is_landmark_A' in batch[0]:
        is_landmark_batch = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)
        for i, item in enumerate(batch):
            nA, nB = item["n_A"], item["n_B"]
            is_landmark_batch[2*i,   :nA] = item["is_landmark_A"].to(device, non_blocking=True)
            is_landmark_batch[2*i+1, :nB] = item["is_landmark_B"].to(device, non_blocking=True)
    else:
        is_landmark_batch = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)

    # Collate kNN indices
    knn_k = batch[0]['knn_indices_A'].shape[1]
    knn_batch = torch.full((B * 2, n_max, knn_k), -1, dtype=torch.long, device=device)

    for i, item in enumerate(batch):
        nA, nB = item["n_A"], item["n_B"]
        knn_batch[2*i, :nA] = item["knn_indices_A"].to(device, non_blocking=True)
        knn_batch[2*i+1, :nB] = item["knn_indices_B"].to(device, non_blocking=True)

    return {
        "Z_set": Z_batch,                 # (2B, n_max, h)
        "mask": mask,                     # (2B, n_max)
        "n": n_vec,                       # (2B,)
        "pair_idxA": pair_idxA,           # (P=B,)
        "pair_idxB": pair_idxB,           # (P=B,)
        "shared_A_idx": shared_A_idx,     # (B, Kmax) padded -1
        "shared_B_idx": shared_B_idx,     # (B, Kmax) padded -1
        "shared_len": shared_len,         # (B,)
        "shared_info": shared_info,
        "sc_global_indices": global_indices,
        "is_sc": True,
        'knn_indices': knn_batch,
        "is_landmark": is_landmark_batch,
    }


@torch.no_grad()
def build_triplets_from_cache_for_set(
    set_global_idx: torch.Tensor,
    pos_idx_cpu: torch.Tensor,
    n_per_anchor: int = 10,
    triplet_cap: int = 20000
) -> torch.Tensor:
    if set_global_idx.is_cuda:
        set_global_idx_cpu = set_global_idx.detach().cpu()
    else:
        set_global_idx_cpu = set_global_idx
    
    n_valid = int(set_global_idx_cpu.numel())
    if n_valid <= 2:
        return torch.empty((0, 3), dtype=torch.long)
    
    N_sc_total = int(pos_idx_cpu.size(0))
    global_to_local = torch.full((N_sc_total,), -1, dtype=torch.int32)
    global_to_local[set_global_idx_cpu] = torch.arange(n_valid, dtype=torch.int32)
    
    pos_glob = pos_idx_cpu[set_global_idx_cpu]
    pos_loc = global_to_local[pos_glob]
    
    triplets_list = []
    
    for a_loc in range(n_valid):
        pl = pos_loc[a_loc]
        pl = pl[pl >= 0]
        if pl.numel() == 0:
            continue
        
        if pl.numel() > n_per_anchor:
            sel = torch.randint(0, pl.numel(), (n_per_anchor,))
            pl = pl[sel]
        
        neg_mask = torch.ones(n_valid, dtype=torch.bool)
        neg_mask[a_loc] = False
        neg_mask[pl.long()] = False
        neg_candidates = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
        if neg_candidates.numel() == 0:
            continue
        
        neg_sel = neg_candidates[torch.randint(0, neg_candidates.numel(), (pl.numel(),))]
        a_col = torch.full((pl.numel(),), a_loc, dtype=torch.long)
        triplets_list.append(torch.stack([a_col, pl.long(), neg_sel.long()], dim=1))
    
    if len(triplets_list) == 0:
        return torch.empty((0, 3), dtype=torch.long)
    
    triplets = torch.cat(triplets_list, dim=0)
    if triplets.size(0) > triplet_cap:
        idx = torch.randperm(triplets.size(0))[:triplet_cap]
        triplets = triplets[idx]
    return triplets