"""
Self-Supervised Learning Utilities for Stage A
- VICReg loss (Variance-Invariance-Covariance Regularization)
- Gradient Reversal Layer (GRL) for domain adversarial training
- Expression augmentations (coordinate-free)
- Slide discriminator
- DDP-safe gather utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Dict, Optional
import math
import random
import numpy as np


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Call this BEFORE creating any models or data loaders.

    Args:
        seed: Random seed to use
        deterministic: If True, also sets CUDA to deterministic mode (slower but reproducible)

    Example:
        from ssl_utils import set_seed
        set_seed(42)
        encoder = SharedEncoder(...)  # Now initialized with seed 42
        train_encoder(model=encoder, ..., seed=42)  # Training also seeded
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[set_seed] Random seed set to {seed} (deterministic={deterministic})")


# ==============================================================================
# GENE FILTERING
# ==============================================================================

def filter_informative_genes(
    expression_arrays: dict,
    gene_names: list,
    max_zero_frac: float = 0.95,
    min_variance: float = 0.01,
    verbose: bool = True
) -> list:
    """
    Filter genes to keep only informative ones across ALL data sources.

    A gene is kept if it passes BOTH criteria in ALL sources:
    1. Zero fraction < max_zero_frac (not too sparse)
    2. Variance > min_variance (has signal)

    Args:
        expression_arrays: Dict mapping source name to numpy array (n_cells, n_genes)
        gene_names: List of gene names corresponding to columns
        max_zero_frac: Maximum allowed fraction of zeros (default 0.85)
        min_variance: Minimum required variance (default 0.01)
        verbose: Print per-source statistics

    Returns:
        List of gene names that pass all filters in all sources

    Example:
        sources = {
            'P2_ST1': X_st1,
            'P2_ST2': X_st2,
            'P10_ST3': X_p10_st3,
        }
        filtered_genes = filter_informative_genes(sources, common_genes)
    """
    n_genes = len(gene_names)
    gene_mask = np.ones(n_genes, dtype=bool)

    if verbose:
        print(f"\n[Gene Filter] Filtering {n_genes} genes (max_zero={max_zero_frac}, min_var={min_variance})")

    for name, X in expression_arrays.items():
        if hasattr(X, 'toarray'):
            X = X.toarray()

        zero_frac = (X == 0).mean(axis=0)
        variance = X.var(axis=0)

        # Keep if <max_zero_frac zeros AND var > min_variance
        sparse_ok = zero_frac < max_zero_frac
        var_ok = variance > min_variance
        source_mask = sparse_ok & var_ok

        n_pass = source_mask.sum()
        if verbose:
            print(f"  {name}: {n_pass}/{n_genes} genes pass")

        gene_mask = gene_mask & source_mask

    filtered_genes = [gene_names[i] for i in range(n_genes) if gene_mask[i]]
    n_filtered = len(filtered_genes)

    if verbose:
        print(f"[Gene Filter] Final: {n_filtered}/{n_genes} genes pass ALL sources")

    return filtered_genes


# ==============================================================================
# DDP-SAFE GATHER LAYER (from Facebook VICReg)
# ==============================================================================

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes and support backward propagation.
    
    From: facebookresearch/vicreg
    """
    
    @staticmethod
    def forward(ctx, x):
        if not dist.is_initialized():
            return (x,)
        
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        if not dist.is_initialized():
            return grads[0]
        
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def batch_all_gather(x):
    """Gather x from all ranks and concatenate."""
    if not dist.is_initialized():
        return x
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


# ==============================================================================
# GRADIENT REVERSAL LAYER (from tadeephuy/GradientReversal)
# ==============================================================================

# class GradientReversalFunction(torch.autograd.Function):
#     """
#     Gradient Reversal Layer for domain adversarial training.
    
#     From: tadeephuy/GradientReversal
#     Forward: identity
#     Backward: -alpha * grad
#     """
    
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.save_for_backward(x, alpha)
#         return x
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         _, alpha = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = -alpha * grad_output
#         return grad_input, None


# def grad_reverse(x, alpha):
#     """Apply gradient reversal."""
#     return GradientReversalFunction.apply(x, alpha)

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Forward: identity
    Backward: multiply gradient by -alpha
    """

    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.alpha) * grad_output, None


def grad_reverse(x, alpha: float):
    return GradientReversalFunction.apply(x, float(alpha))



# ==============================================================================
# VICREG LOSS (from Facebook VICReg with adaptations)
# ==============================================================================

def off_diagonal(x):
    """
    Return flattened view of off-diagonal elements of square matrix.
    
    From: facebookresearch/vicreg
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization
    
    Based on: Bardes et al., "VICReg: Variance-Invariance-Covariance 
              Regularization for Self-Supervised Learning" (ICLR 2022)
    
    Implementation adapted from: facebookresearch/vicreg
    """
    
    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        use_ddp_gather: bool = True,
        compute_stats_fp32: bool = True
    ):
        """
        Args:
            lambda_inv: weight for invariance loss
            lambda_var: weight for variance loss
            lambda_cov: weight for covariance loss
            gamma: target std for variance regularization
            eps: numerical stability constant
            use_ddp_gather: gather embeddings across GPUs for var/cov
            compute_stats_fp32: compute var/cov in fp32 for stability
        """
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps
        self.use_ddp_gather = use_ddp_gather
        self.compute_stats_fp32 = compute_stats_fp32
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss.
        
        Args:
            z1: (B, D) first view embeddings
            z2: (B, D) second view embeddings
            
        Returns:
            loss: scalar VICReg loss
            stats: dict with loss components
        """
        B, D = z1.shape
        
        # ========== INVARIANCE LOSS ==========
        # MSE between views
        loss_inv = F.mse_loss(z1, z2)
        
        # ========== GATHER FOR VAR/COV (optional) ==========
        if self.use_ddp_gather and dist.is_initialized():
            z1_gathered = batch_all_gather(z1)
            z2_gathered = batch_all_gather(z2)
            effective_batch_size = z1_gathered.shape[0]
        else:
            z1_gathered = z1
            z2_gathered = z2
            effective_batch_size = B
        
        # ========== COMPUTE IN FP32 FOR STABILITY ==========
        if self.compute_stats_fp32:
            z1_gathered = z1_gathered.float()
            z2_gathered = z2_gathered.float()
        
        # Center embeddings (important for var/cov)
        z1_centered = z1_gathered - z1_gathered.mean(dim=0)
        z2_centered = z2_gathered - z2_gathered.mean(dim=0)
        
        # ========== VARIANCE LOSS ==========
        # Hinge loss: encourage std >= gamma for each dimension
        std_z1 = torch.sqrt(z1_centered.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2_centered.var(dim=0) + self.eps)
        
        loss_var = (
            torch.mean(F.relu(self.gamma - std_z1)) +
            torch.mean(F.relu(self.gamma - std_z2))
        ) / 2.0
        
        # ========== COVARIANCE LOSS ==========
        # Decorrelate dimensions: penalize off-diagonal covariance
        # Facebook VICReg uses batch_size from args, not effective
        # For single GPU, this is just B
        cov_z1 = (z1_centered.T @ z1_centered) / (z1_centered.shape[0] - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (z2_centered.shape[0] - 1)
        
        loss_cov = (
            off_diagonal(cov_z1).pow(2).sum() / D +
            off_diagonal(cov_z2).pow(2).sum() / D
        )
        
        # ========== TOTAL LOSS ==========
        loss = (
            self.lambda_inv * loss_inv +
            self.lambda_var * loss_var +
            self.lambda_cov * loss_cov
        )
        
        # Statistics for logging
        stats = {
            'inv': loss_inv.item(),
            'var': loss_var.item(),
            'cov': loss_cov.item(),
            'std_mean': (std_z1.mean().item() + std_z2.mean().item()) / 2.0,
            'std_min': min(std_z1.min().item(), std_z2.min().item()),
        }
        
        return loss, stats


# ==============================================================================
# SLIDE DISCRIMINATOR
# ==============================================================================

class SlideDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_slides: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # ← ADD THIS LINE
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # ← ADD THIS LINE
            nn.Linear(hidden_dim, n_slides)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) embeddings
            
        Returns:
            logits: (B, n_slides) classification logits
        """
        return self.net(x)


# ==============================================================================
# EXPRESSION AUGMENTATIONS (coordinate-free)
# ==============================================================================

def augment_expression(
    X: torch.Tensor,
    gene_dropout: float = 0.2,
    gauss_std: float = 0.01,
    scale_jitter: float = 0.2,
) -> torch.Tensor:
    """
    Coordinate-free augmentations for log1p expression tensors.
    """
    device = X.device
    B, G = X.shape
    X_aug = X.clone()

    # 1) Gene dropout
    if gene_dropout > 0:
        mask = (torch.rand(B, G, device=device) > gene_dropout).to(X_aug.dtype)
        X_aug = X_aug * mask

    # 2) Scale jitter (library size simulation)
    if scale_jitter > 0:
        X_lin = torch.expm1(X_aug)
        scale = torch.empty(B, 1, device=device).uniform_(1.0 - scale_jitter, 1.0 + scale_jitter)
        X_lin = X_lin * scale
        X_aug = torch.log1p(X_lin)

    # 3) Gaussian noise
    if gauss_std > 0:
        X_aug = X_aug + torch.randn_like(X_aug) * gauss_std

    return torch.clamp(X_aug, min=-10.0, max=10.0)


# ==============================================================================
# BALANCED SLIDE SAMPLING
# ==============================================================================

def sample_balanced_slide_indices(
    slide_ids: torch.Tensor,
    batch_size: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Sample indices with balanced representation across slides.
    
    Args:
        slide_ids: (N,) long tensor of slide IDs for each spot
        batch_size: target batch size
        device: device
        
    Returns:
        indices: (batch_size,) sampled indices
    """
    unique_slides = torch.unique(slide_ids)
    n_slides = len(unique_slides)
    
    # Target: equal samples per slide
    samples_per_slide = batch_size // n_slides
    remainder = batch_size % n_slides
    
    indices_list = []
    
    for i, slide_id in enumerate(unique_slides):
        # Find all indices for this slide
        slide_mask = (slide_ids == slide_id)
        slide_indices = torch.nonzero(slide_mask, as_tuple=True)[0]
        
        # Sample from this slide
        n_sample = samples_per_slide + (1 if i < remainder else 0)
        if n_sample <= 0:
            continue

        if len(slide_indices) >= n_sample:
            perm = torch.randperm(len(slide_indices), device=device)[:n_sample]
            sampled = slide_indices[perm]
        else:
            # sample with replacement if slide is small
            ridx = torch.randint(0, len(slide_indices), (n_sample,), device=device)
            sampled = slide_indices[ridx]

        indices_list.append(sampled)

    
    # Concatenate and shuffle
    if len(indices_list) == 0:
        return torch.randint(0, len(slide_ids), (batch_size,), device=device)

    indices = torch.cat(indices_list, dim=0)

    # Make length exactly batch_size
    if indices.numel() > batch_size:
        indices = indices[:batch_size]
    elif indices.numel() < batch_size:
        extra = torch.randint(0, len(slide_ids), (batch_size - indices.numel(),), device=device)
        indices = torch.cat([indices, extra], dim=0)

    # Shuffle final
    indices = indices[torch.randperm(indices.numel(), device=device)]

    # Debug: verify balance (optional)
    if torch.rand(1).item() < 0.01:
        counts = [(slide_ids[indices] == s).sum().item() for s in unique_slides]
        print(f"[BALANCE-CHECK] Batch slide counts: {counts} (target ~{batch_size//len(unique_slides)})")

    return indices


def sample_balanced_domain_and_slide_indices(
    domain_ids: torch.Tensor,
    st_slide_ids: torch.Tensor,
    sc_slide_ids: Optional[torch.Tensor],
    batch_size: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Sample indices with hierarchical balancing for multi-slide SC data.

    1. First level: 50% ST, 50% SC (domain balancing)
    2. Second level:
       - Within ST: balance across ST slides (via st_slide_ids)
       - Within SC: balance across SC slides (via sc_slide_ids)

    Args:
        domain_ids: (N,) long tensor - 0 for ST, 1 for SC (concatenated)
        st_slide_ids: (n_st,) slide IDs for ST samples only
        sc_slide_ids: (n_sc,) slide IDs for SC samples only, or None for single-pool SC
        batch_size: target batch size (should be even)
        device: device

    Returns:
        indices: (batch_size,) sampled global indices into X_ssl
    """
    st_mask = (domain_ids == 0)
    sc_mask = (domain_ids == 1)

    # Global indices for each domain
    st_global_indices = torch.where(st_mask)[0]
    sc_global_indices = torch.where(sc_mask)[0]

    n_st = st_global_indices.shape[0]
    n_sc = sc_global_indices.shape[0]

    # Target: half batch from each domain
    n_st_sample = batch_size // 2
    n_sc_sample = batch_size - n_st_sample

    # ========== Sample from ST domain (balanced across ST slides) ==========
    st_local_idx = _sample_balanced_internal(st_slide_ids, n_st_sample, device)
    st_batch_global = st_global_indices[st_local_idx]

    # ========== Sample from SC domain (balanced across SC slides if provided) ==========
    if sc_slide_ids is not None:
        sc_local_idx = _sample_balanced_internal(sc_slide_ids, n_sc_sample, device)
    else:
        # No SC slide balancing - random sample from all SC
        if n_sc >= n_sc_sample:
            sc_local_idx = torch.randperm(n_sc, device=device)[:n_sc_sample]
        else:
            sc_local_idx = torch.randint(0, n_sc, (n_sc_sample,), device=device)

    sc_batch_global = sc_global_indices[sc_local_idx]

    # Combine and shuffle
    indices = torch.cat([st_batch_global, sc_batch_global], dim=0)
    indices = indices[torch.randperm(indices.numel(), device=device)]

    # Debug: verify balance (1% of the time)
    if torch.rand(1).item() < 0.01:
        n_st_sampled = (domain_ids[indices] == 0).sum().item()
        n_sc_sampled = (domain_ids[indices] == 1).sum().item()
        print(f"[DOMAIN-BALANCE] ST={n_st_sampled}, SC={n_sc_sampled}")

        if sc_slide_ids is not None:
            sc_idx_local = indices[domain_ids[indices] == 1] - n_st
            unique_sc_slides = torch.unique(sc_slide_ids)
            sc_counts = [(sc_slide_ids[sc_idx_local] == s).sum().item() for s in unique_sc_slides]
            print(f"[SC-SLIDE-BALANCE] Per-slide counts: {sc_counts}")

    return indices


def _sample_balanced_internal(
    slide_ids: torch.Tensor,
    n_samples: int,
    device: str
) -> torch.Tensor:
    """
    Internal helper: sample n_samples indices balanced across unique slide_ids.
    Returns LOCAL indices (0 to len(slide_ids)-1).
    """
    unique_slides = torch.unique(slide_ids)
    n_slides = len(unique_slides)

    samples_per_slide = n_samples // n_slides
    remainder = n_samples % n_slides

    indices_list = []

    for i, slide_id in enumerate(unique_slides):
        slide_mask = (slide_ids == slide_id)
        slide_local_indices = torch.nonzero(slide_mask, as_tuple=True)[0]

        n_sample = samples_per_slide + (1 if i < remainder else 0)
        if n_sample <= 0:
            continue

        if len(slide_local_indices) >= n_sample:
            perm = torch.randperm(len(slide_local_indices), device=device)[:n_sample]
            sampled = slide_local_indices[perm]
        else:
            # Sample with replacement
            ridx = torch.randint(0, len(slide_local_indices), (n_sample,), device=device)
            sampled = slide_local_indices[ridx]

        indices_list.append(sampled)

    if len(indices_list) == 0:
        return torch.randint(0, len(slide_ids), (n_samples,), device=device)

    result = torch.cat(indices_list, dim=0)

    # Ensure exact count
    if result.numel() > n_samples:
        result = result[:n_samples]
    elif result.numel() < n_samples:
        extra = torch.randint(0, len(slide_ids), (n_samples - result.numel(),), device=device)
        result = torch.cat([result, extra], dim=0)

    return result



def sample_balanced_source_indices(
    source_ids: torch.Tensor,
    batch_size: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Sample indices balanced across ALL unique source IDs.
 
    For N-class source adversary where we want equal representation
    from each source (P2_ST1, P2_ST2, P2_ST3, P2_SC, P10_ST3, P10_SC, etc.)
 
    Args:
        source_ids: (N,) long tensor with source labels (0, 1, 2, ..., n_sources-1)
        batch_size: target batch size
        device: device
 
    Returns:
        indices: (batch_size,) sampled global indices
    """
    unique_sources = torch.unique(source_ids)
    n_sources = len(unique_sources)
 
    samples_per_source = batch_size // n_sources
    remainder = batch_size % n_sources
 
    indices_list = []
 
    for i, src_id in enumerate(unique_sources):
        src_mask = (source_ids == src_id)
        src_indices = torch.nonzero(src_mask, as_tuple=True)[0]
 
        n_sample = samples_per_source + (1 if i < remainder else 0)
        if n_sample <= 0:
            continue
 
        if len(src_indices) >= n_sample:
            perm = torch.randperm(len(src_indices), device=device)[:n_sample]
            sampled = src_indices[perm]
        else:
            # Sample with replacement if source is small
            ridx = torch.randint(0, len(src_indices), (n_sample,), device=device)
            sampled = src_indices[ridx]
 
        indices_list.append(sampled)
 
    if len(indices_list) == 0:
        return torch.randperm(len(source_ids), device=device)[:batch_size]
 
    result = torch.cat(indices_list, dim=0)
 
    # Shuffle so sources are interleaved
    result = result[torch.randperm(result.numel(), device=device)]
 
    return result

# ==============================================================================
# GRL ALPHA SCHEDULE
# ==============================================================================

def grl_alpha_schedule(
    epoch: int,
    warmup_epochs: int = 10,
    ramp_epochs: int = 20,
    alpha_max: float = 1.0
) -> float:
    """
    Compute GRL alpha with warmup and ramp.
    
    Args:
        epoch: current epoch (0-indexed)
        warmup_epochs: epochs before adversary starts
        ramp_epochs: epochs to ramp from 0 to alpha_max
        alpha_max: maximum alpha value
        
    Returns:
        alpha: current alpha value
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + ramp_epochs:
        progress = (epoch - warmup_epochs) / ramp_epochs
        return alpha_max * progress
    else:
        return alpha_max
    


def coral_loss(z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """
    CORAL (CORrelation ALignment) loss between two embedding distributions.
    Matches mean and covariance.
    
    Args:
        z_source: (N1, D) source embeddings
        z_target: (N2, D) target embeddings
        
    Returns:
        loss: scalar CORAL loss
    """
    # Mean matching
    mu_s = z_source.mean(dim=0)
    mu_t = z_target.mean(dim=0)
    loss_mean = (mu_s - mu_t).pow(2).mean()
    
    # Covariance matching
    z_s_centered = z_source - mu_s
    z_t_centered = z_target - mu_t
    
    cov_s = (z_s_centered.T @ z_s_centered) / max(z_s_centered.shape[0] - 1, 1)
    cov_t = (z_t_centered.T @ z_t_centered) / max(z_t_centered.shape[0] - 1, 1)
    
    loss_cov = (cov_s - cov_t).pow(2).mean()
    
    return loss_mean + loss_cov


def mmd_rbf_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[float] = None
) -> torch.Tensor:
    """
    RBF-MMD loss between two sets of embeddings.

    Args:
        x: (N, D)
        y: (M, D)
        sigma: kernel bandwidth. If None, use median heuristic.

    Returns:
        scalar MMD loss
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    with torch.no_grad():
        xy = torch.cat([x, y], dim=0)
        dists = torch.cdist(xy, xy)
        if sigma is None:
            vals = dists[dists > 0]
            sigma = torch.median(vals).item() if vals.numel() > 0 else 1.0
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0

    gamma = 1.0 / (2.0 * sigma ** 2)
    Kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    Kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
    Kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))

    mmd = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return mmd


# ==============================================================================
# DIAGNOSTIC UTILITIES (for adversary debugging - GPT 5.2 Pro recommendations)
# ==============================================================================

def compute_confusion_matrix(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    n_classes: int
) -> torch.Tensor:
    """
    Compute normalized confusion matrix.
    
    Args:
        pred: (N,) predicted class indices (long tensor)
        target: (N,) ground truth class indices (long tensor)
        n_classes: number of classes
        
    Returns:
        cm: (n_classes, n_classes) normalized confusion matrix
            cm[i,j] = P(pred=j | true=i)
    """
    if pred.numel() == 0:
        return torch.zeros(n_classes, n_classes, device=pred.device)
    
    # Ensure tensors are on same device and are long type
    pred = pred.long()
    target = target.long()
    device = pred.device
    
    # Build confusion matrix
    cm = torch.zeros(n_classes, n_classes, device=device, dtype=torch.float32)
    for i in range(n_classes):
        mask_i = (target == i)
        if mask_i.sum() > 0:
            for j in range(n_classes):
                cm[i, j] = ((pred == j) & mask_i).sum().float()
    
    # Row-normalize: cm[i,:] sums to 1 (or 0 if no samples for class i)
    row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1e-8)
    cm = cm / row_sums
    
    return cm


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute prediction entropy from logits.
    
    High entropy = uncertain predictions (good for domain confusion).
    Low entropy = confident predictions (discriminator winning).
    
    Args:
        logits: (..., n_classes) raw logits
        dim: dimension over which to compute softmax
        
    Returns:
        entropy: (...,) entropy values in nats
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    
    # H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=dim)
    
    return entropy


def compute_gradient_norms(
    named_parameters,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute gradient norms for model parameters.
    
    Call this AFTER backward() but BEFORE optimizer.step().
    
    Args:
        named_parameters: iterator of (name, param) tuples (model.named_parameters())
        prefix: optional prefix for keys
        
    Returns:
        grad_norms: dict mapping parameter names to gradient L2 norms
    """
    grad_norms = {}
    total_norm_sq = 0.0
    
    for name, param in named_parameters:
        if param.grad is not None:
            grad_norm = param.grad.detach().norm(2).item()
            key = f"{prefix}{name}" if prefix else name
            grad_norms[key] = grad_norm
            total_norm_sq += grad_norm ** 2
    
    grad_norms[f"{prefix}total"] = total_norm_sq ** 0.5
    
    return grad_norms


def augment_expression_mild(
    X: torch.Tensor,
    gene_dropout: float = 0.05,
    gauss_std: float = 0.005,
    scale_jitter: float = 0.05,
) -> torch.Tensor:
    """
    MILD augmentation for clean-ish representation.
    
    Use this for adversary input when you want the discriminator to see
    something closer to the clean representation used at inference.
    
    Args:
        X: (B, G) log1p expression
        gene_dropout: fraction of genes to zero (default: 5%)
        gauss_std: gaussian noise std (default: 0.005)
        scale_jitter: library size jitter (default: 5%)
        
    Returns:
        X_aug: mildly augmented expression
    """
    device = X.device
    B, G = X.shape
    X_aug = X.clone()
    
    # 1) Very light gene dropout
    if gene_dropout > 0:
        mask = (torch.rand(B, G, device=device) > gene_dropout).to(X_aug.dtype)
        X_aug = X_aug * mask
    
    # 2) Very light scale jitter
    if scale_jitter > 0:
        X_lin = torch.expm1(X_aug)
        scale = torch.empty(B, 1, device=device).uniform_(1.0 - scale_jitter, 1.0 + scale_jitter)
        X_lin = X_lin * scale
        X_aug = torch.log1p(X_lin)
    
    # 3) Very light gaussian noise
    if gauss_std > 0:
        X_aug = X_aug + torch.randn_like(X_aug) * gauss_std
    
    return torch.clamp(X_aug, min=-10.0, max=10.0)


def get_adversary_representation(
    model: nn.Module,
    X_batch: torch.Tensor,
    z1_aug: torch.Tensor,
    z2_aug: torch.Tensor,
    mode: str = 'clean',
    use_layernorm: bool = False,
    mild_dropout: float = 0.05,
    mild_noise: float = 0.005,
    mild_jitter: float = 0.05,
) -> torch.Tensor:
    """
    Get the representation for adversary training based on mode.
    
    This function centralizes the logic for what the discriminator sees,
    ensuring consistency between training and inference.
    
    Args:
        model: encoder model
        X_batch: (B, G) original clean expression batch
        z1_aug: (B, D) embedding of first augmented view
        z2_aug: (B, D) embedding of second augmented view
        mode: one of:
            - 'clean': use encoder(X_clean) - RECOMMENDED for inference consistency
            - 'mild_aug': use encoder(X_mildly_augmented)
            - 'ln_avg_aug': use LayerNorm(avg(z1_aug, z2_aug)) - LEGACY (buggy)
        use_layernorm: whether to apply LayerNorm to final representation
        mild_dropout, mild_noise, mild_jitter: parameters for mild augmentation
        
    Returns:
        z_cond: (B, D) representation for adversary
    """
    device = X_batch.device
    
    if mode == 'clean':
        # Run clean batch through encoder - matches inference exactly
        z_cond = model(X_batch)
        
    elif mode == 'mild_aug':
        # Apply very mild augmentation - close to clean but with some noise
        X_mild = augment_expression_mild(X_batch, mild_dropout, mild_noise, mild_jitter)
        z_cond = model(X_mild)
        
    elif mode == 'ln_avg_aug':
        # LEGACY: average of augmented views - causes train/inference mismatch!
        z_bar_raw = (z1_aug + z2_aug) / 2.0
        z_cond = z_bar_raw
        
    else:
        raise ValueError(f"Unknown adversary representation mode: {mode}. "
                        f"Valid modes: 'clean', 'mild_aug', 'ln_avg_aug'")
    
    # Optional LayerNorm (should match what Stage C diffusion uses)
    if use_layernorm:
        z_cond = F.layer_norm(z_cond, (z_cond.shape[1],))
    
    return z_cond


def log_adversary_diagnostics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    prefix: str = "",
    epoch: int = 0,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive logging for adversary diagnostics.
    
    Args:
        logits: (B, n_classes) discriminator output
        targets: (B,) true domain labels
        n_classes: number of domains
        prefix: string prefix for log keys
        epoch: current epoch (for printing)
        verbose: whether to print detailed stats
        
    Returns:
        stats: dict with accuracy, entropy, per-class accuracy, confusion matrix
    """
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        
        # Overall accuracy
        acc = (pred == targets).float().mean().item()
        
        # Per-class accuracy
        per_class_acc = {}
        for c in range(n_classes):
            mask_c = (targets == c)
            if mask_c.sum() > 0:
                per_class_acc[c] = (pred[mask_c] == c).float().mean().item()
            else:
                per_class_acc[c] = 0.0
        
        # Entropy (mean and std)
        ent = compute_entropy(logits)
        ent_mean = ent.mean().item()
        ent_std = ent.std().item() if ent.numel() > 1 else 0.0
        
        # Max entropy for reference (uniform = log(n_classes))
        max_entropy = math.log(n_classes)
        ent_ratio = ent_mean / max_entropy  # 1.0 = perfect confusion
        
        # Confusion matrix
        cm = compute_confusion_matrix(pred, targets, n_classes)
        
        stats = {
            f'{prefix}acc': acc,
            f'{prefix}ent_mean': ent_mean,
            f'{prefix}ent_std': ent_std,
            f'{prefix}ent_ratio': ent_ratio,
        }
        
        for c in range(n_classes):
            stats[f'{prefix}acc_class{c}'] = per_class_acc[c]
        
        # Flatten confusion matrix into stats
        for i in range(n_classes):
            for j in range(n_classes):
                stats[f'{prefix}cm_{i}_{j}'] = cm[i, j].item()
        
        if verbose and epoch % 100 == 0:
            print(f"  [{prefix}] Acc={acc:.3f}, Ent={ent_mean:.3f}/{max_entropy:.3f} "
                  f"(ratio={ent_ratio:.3f})")
            print(f"  [{prefix}] Per-class acc: {per_class_acc}")
            print(f"  [{prefix}] Confusion matrix:\n{cm.cpu().numpy()}")
        
        return stats


def compute_local_alignment_loss(
    z_sc: torch.Tensor,
    z_st: torch.Tensor,
    x_sc: torch.Tensor = None,
    x_st: torch.Tensor = None,
    tau_x: float = 0.1,
    tau_z: float = 0.1,
    bidirectional: bool = True,
    mnn_min_sim: float = 0.2,
) -> torch.Tensor:
    """
    Local alignment via MNN in embedding space (InfoNCE).
    Ignores x_sc/x_st (kept for backward compatibility).
    """

    # Normalize for cosine similarity
    z_sc_norm = F.normalize(z_sc, dim=1)  # (n_sc, D)
    z_st_norm = F.normalize(z_st, dim=1)  # (n_st, D)

    # Similarity matrix (SC x ST)
    S = z_sc_norm @ z_st_norm.T  # (n_sc, n_st)

    # Nearest neighbors
    nn_sc = S.argmax(dim=1)      # (n_sc,)
    nn_st = S.argmax(dim=0)      # (n_st,)

    # Mutual nearest neighbors mask
    idx_sc = torch.arange(S.shape[0], device=S.device)
    mnn_mask = (nn_st[nn_sc] == idx_sc)

    # Optional similarity threshold
    if mnn_min_sim > 0:
        mnn_mask = mnn_mask & (S[idx_sc, nn_sc] >= mnn_min_sim)

    # If no pairs, return zero
    if mnn_mask.sum() == 0:
        return torch.tensor(0.0, device=S.device)

    # InfoNCE: SC -> ST
    pos_st = nn_sc[mnn_mask]          # (n_pairs,)
    pos_sc = idx_sc[mnn_mask]         # (n_pairs,)
    logits_sc2st = S[pos_sc] / tau_z  # (n_pairs, n_st)
    loss_sc2st = F.cross_entropy(logits_sc2st, pos_st)

    if not bidirectional:
        return loss_sc2st

    # InfoNCE: ST -> SC (use the same pairs)
    logits_st2sc = S.T[pos_st] / tau_z  # (n_pairs, n_sc)
    loss_st2sc = F.cross_entropy(logits_st2sc, pos_sc)

    return 0.5 * (loss_sc2st + loss_st2sc)


def mmd_rbf_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    use_median: bool = True,
    return_sigma: bool = False
) -> torch.Tensor:
    """
    Multi-kernel RBF-MMD between two sets of embeddings.

    Args:
        x: (N, D)
        y: (M, D)
        sigmas: scale multipliers for base sigma
        use_median: if True, base sigma = median of pairwise distances
        return_sigma: if True, also return base sigma

    Returns:
        scalar MMD loss (and optionally base sigma)
    """
    if x.numel() == 0 or y.numel() == 0:
        out = torch.tensor(0.0, device=x.device)
        return (out, 1.0) if return_sigma else out

    with torch.no_grad():
        if use_median:
            xy = torch.cat([x, y], dim=0)
            dists = torch.cdist(xy, xy)
            vals = dists[dists > 0]
            base_sigma = torch.median(vals).item() if vals.numel() > 0 else 1.0
        else:
            base_sigma = 1.0
        if not np.isfinite(base_sigma) or base_sigma <= 0:
            base_sigma = 1.0

    mmd = 0.0
    for s in sigmas:
        sigma = base_sigma * s
        gamma = 1.0 / (2.0 * sigma ** 2)
        Kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
        Kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
        Kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
        mmd = mmd + (Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    mmd = mmd / len(sigmas)
    return (mmd, base_sigma) if return_sigma else mmd


# ==============================================================================
# SPATIAL InfoNCE LOSS (coordinate-supervised neighborhood)
# ==============================================================================

@torch.no_grad()
def precompute_spatial_nce_structures(
    st_coords: torch.Tensor,
    st_gene_expr: torch.Tensor,
    slide_ids: torch.Tensor,
    k_phys: int = 20,
    far_mult: float = 4.0,
    n_hard: int = 20,
    device: str = 'cuda',
) -> dict:
    """
    Precompute physical neighbors, far sets, and hard negatives per ST spot.

    Returns dict with:
        pos_idx:  (n_st, k_phys) physical neighbor indices
        far_mask: (n_st, n_st)   bool mask for physically far spots (per-slide)
        hard_neg: (n_st, n_hard) expression-similar but physically far indices
        r_pos:    per-slide median k-th neighbor distance (for logging)
    """
    n_st = st_coords.shape[0]
    unique_slides = torch.unique(slide_ids)

    # Per-slide physical kNN
    pos_idx = torch.full((n_st, k_phys), -1, dtype=torch.long, device=device)
    far_mask = torch.zeros(n_st, n_st, dtype=torch.bool, device=device)
    hard_neg = torch.full((n_st, n_hard), -1, dtype=torch.long, device=device)
    r_pos_list = []

    X_norm = F.normalize(st_gene_expr, dim=1)

    for sid in unique_slides:
        mask = (slide_ids == sid)
        idx_global = torch.where(mask)[0]
        coords_s = st_coords[idx_global]
        n_s = coords_s.shape[0]

        D_phys = torch.cdist(coords_s, coords_s)

        # Physical kNN (exclude self)
        k_eff = min(k_phys, n_s - 1)
        _, topk = D_phys.topk(k_eff + 1, dim=1, largest=False)
        topk = topk[:, 1:]  # drop self
        # pos_idx[idx_global[:, None].expand(-1, k_eff), :k_eff] = idx_global[topk]
        pos_idx[idx_global, :k_eff] = idx_global[topk]

        # r_pos = median distance to k-th physical neighbor
        kth_dists = D_phys.gather(1, topk)
        r_pos = kth_dists[:, -1].median().item()
        r_pos_list.append(r_pos)
        r_far = far_mult * r_pos

        # Far mask (within same slide only)
        far_slide = D_phys >= r_far
        far_slide.fill_diagonal_(False)
        for i_local in range(n_s):
            i_g = idx_global[i_local]
            far_globals = idx_global[far_slide[i_local]]
            far_mask[i_g, far_globals] = True

        # Hard negatives: expression-similar but physically far
        X_s = X_norm[idx_global]
        sim_expr = X_s @ X_s.T  # (n_s, n_s)

        for i_local in range(n_s):
            i_g = idx_global[i_local]
            far_local = far_slide[i_local]
            if far_local.sum() == 0:
                continue
            # Among far spots, pick top-m by expression similarity
            sim_far = sim_expr[i_local].clone()
            sim_far[~far_local] = -float('inf')
            m_eff = min(n_hard, far_local.sum().item())
            _, hard_local = sim_far.topk(m_eff)
            hard_neg[i_g, :m_eff] = idx_global[hard_local]

    print(f"[SpatialNCE] Precomputed: k_phys={k_phys}, r_far={far_mult}x, "
          f"n_hard={n_hard}, r_pos per slide={[f'{r:.2f}' for r in r_pos_list]}")

    return {
        'pos_idx': pos_idx,
        'far_mask': far_mask,
        'hard_neg': hard_neg,
        'r_pos': r_pos_list,
    }


def compute_spatial_infonce_loss(
    z: torch.Tensor,
    batch_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    far_mask: torch.Tensor,
    hard_neg: torch.Tensor,
    tau: float = 0.1,
    n_rand_neg: int = 128,
    is_st_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Spatial InfoNCE: positives = physical neighbors, negatives = far + hard.

    Args:
        z:          (B, D) embeddings for entire batch
        batch_idx:  (B,) global indices into ST arrays
        pos_idx:    (n_st, k) precomputed physical neighbor global indices
        far_mask:   (n_st, n_st) bool far mask
        hard_neg:   (n_st, n_hard) hard negative global indices
        tau:        temperature
        n_rand_neg: number of random far negatives per anchor
        is_st_mask: (B,) bool mask for ST samples in batch (only ST has coords)

    Returns:
        scalar loss
    """
    if is_st_mask is not None:
        st_idx_in_batch = torch.where(is_st_mask)[0]
    else:
        st_idx_in_batch = torch.arange(z.shape[0], device=z.device)

    if st_idx_in_batch.numel() < 4:
        return torch.tensor(0.0, device=z.device)

    z_norm = F.normalize(z, dim=1)
    losses = []

    for local_i in st_idx_in_batch:
        g_i = batch_idx[local_i].item()
        z_anchor = z_norm[local_i]  # (D,)

        # Positives: physical neighbors that are also in this batch
        pos_globals = pos_idx[g_i]
        pos_globals = pos_globals[pos_globals >= 0]
        if pos_globals.numel() == 0:
            continue

        # Map global → batch-local
        global_to_local = torch.full(
            (max(batch_idx.max().item() + 1, pos_globals.max().item() + 1),),
            -1, dtype=torch.long, device=z.device
        )
        global_to_local[batch_idx] = torch.arange(z.shape[0], device=z.device)

        pos_local = global_to_local[pos_globals]
        pos_local = pos_local[pos_local >= 0]
        if pos_local.numel() == 0:
            continue

        # Hard negatives in batch
        hard_globals = hard_neg[g_i]
        hard_globals = hard_globals[hard_globals >= 0]
        hard_local = global_to_local[hard_globals.clamp(max=global_to_local.shape[0] - 1)]
        hard_local = hard_local[hard_local >= 0]

        # Random far negatives in batch
        # FIX: far_mask is (n_st, n_st) -- only index with ST batch members
        n_st_nce = far_mask.shape[0]
        is_st_in_batch = (batch_idx < n_st_nce)
        st_local_idx = torch.where(is_st_in_batch)[0]
        if st_local_idx.numel() > 0:
            st_globals = batch_idx[st_local_idx]
            far_among_st = far_mask[g_i][st_globals]
            far_among_st = far_among_st & (st_local_idx != local_i)
            far_candidates = st_local_idx[far_among_st]
        else:
            far_candidates = torch.tensor([], dtype=torch.long, device=z.device)
        if far_candidates.numel() > n_rand_neg:
            perm = torch.randperm(far_candidates.numel(), device=z.device)[:n_rand_neg]
            rand_neg = far_candidates[perm]
        else:
            rand_neg = far_candidates

        # Combine negatives (deduplicate)
        all_neg = torch.cat([hard_local, rand_neg]).unique()
        if all_neg.numel() == 0:
            continue

        # InfoNCE
        sim_pos = (z_anchor @ z_norm[pos_local].T) / tau  # (n_pos,)
        sim_neg = (z_anchor @ z_norm[all_neg].T) / tau    # (n_neg,)

        # log-sum-exp denominator
        logits = torch.cat([sim_pos, sim_neg])  # (n_pos + n_neg,)
        log_denom = torch.logsumexp(logits, dim=0)
        loss_i = -(sim_pos.mean() - log_denom)
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=z.device)

    return torch.stack(losses).mean()


# ==============================================================================
# SUPPORT-SET Spatial InfoNCE (full pos/hard/rand per anchor)
# ==============================================================================

def compute_spatial_infonce_supportset(
    model: 'nn.Module',
    st_gene_expr: torch.Tensor,
    pos_idx: torch.Tensor,
    far_mask: torch.Tensor,
    hard_neg: torch.Tensor,
    slide_ids: torch.Tensor,
    tau: float = 0.1,
    n_rand_neg: int = 128,
    n_anchors_per_step: int = 64,
    return_diagnostics: bool = False,
):
    """
    Support-set Spatial InfoNCE: explicitly forward-pass all positives,
    hard negatives, and random far negatives for each anchor.

    Unlike the in-batch version, this guarantees every anchor sees its
    full positive set (~k_phys) and full hard-negative set (~n_hard),
    regardless of what was randomly sampled into the VICReg batch.

    Per step:
      1. Pick one random slide, sample n_anchors_per_step anchors from it
      2. Gather all pos_idx, hard_neg, and random far indices for those anchors
      3. Forward-pass the union through the encoder
      4. Compute InfoNCE per anchor using full sets

    Args:
        model:       encoder (must be in train mode, gradients will flow)
        st_gene_expr: (n_st, n_genes) full ST expression, already on device
        pos_idx:     (n_st, k_phys) precomputed physical neighbor indices
        far_mask:    (n_st, n_st) bool mask for physically far spots
        hard_neg:    (n_st, n_hard) expression-similar but physically far indices
        slide_ids:   (n_st,) per-spot slide IDs
        tau:         temperature
        n_rand_neg:  random far negatives per anchor
        n_anchors_per_step: anchors to sample per slide per step
        return_diagnostics: if True, returns (loss, stats_dict)

    Returns:
        loss (or (loss, stats) if return_diagnostics=True)
    """
    device = st_gene_expr.device
    n_st = st_gene_expr.shape[0]

    # 1. Pick a random slide, sample anchors from it
    unique_slides = torch.unique(slide_ids)
    slide_pick = unique_slides[torch.randint(len(unique_slides), (1,)).item()]
    slide_indices = torch.where(slide_ids == slide_pick)[0]

    n_avail = slide_indices.shape[0]
    n_anc = min(n_anchors_per_step, n_avail)
    perm = torch.randperm(n_avail, device=device)[:n_anc]
    anchor_globals = slide_indices[perm]

    # 2. Collect all indices needed for the support set
    all_needed = set()
    anchor_list = anchor_globals.cpu().tolist()
    all_needed.update(anchor_list)

    per_anchor_pos = []
    per_anchor_hard = []
    per_anchor_far = []

    for a in anchor_list:
        p = pos_idx[a]
        p = p[p >= 0].cpu().tolist()
        per_anchor_pos.append(p)
        all_needed.update(p)

        h = hard_neg[a]
        h = h[h >= 0].cpu().tolist()
        per_anchor_hard.append(h)
        all_needed.update(h)

        far_of_a = torch.where(far_mask[a])[0]
        if far_of_a.numel() > n_rand_neg:
            rperm = torch.randperm(far_of_a.numel(), device=device)[:n_rand_neg]
            far_of_a = far_of_a[rperm]
        f = far_of_a.cpu().tolist()
        per_anchor_far.append(f)
        all_needed.update(f)

    # 3. Forward-pass all unique spots
    all_needed_sorted = sorted(all_needed)
    all_needed_t = torch.tensor(all_needed_sorted, dtype=torch.long, device=device)

    z_support = model(st_gene_expr[all_needed_t])
    z_support_norm = F.normalize(z_support, dim=1)

    # Global -> support-local index mapping
    g2l = torch.full((n_st,), -1, dtype=torch.long, device=device)
    g2l[all_needed_t] = torch.arange(len(all_needed_sorted), device=device)

    # 4. Compute InfoNCE per anchor
    losses = []
    diag_sim_pos = []
    diag_sim_hard = []
    diag_sim_rand = []
    diag_n_pos = []
    diag_n_hard = []
    diag_n_rand = []

    for i, a in enumerate(anchor_list):
        a_local = g2l[a].item()
        z_a = z_support_norm[a_local]

        # Positives
        pos_g = per_anchor_pos[i]
        if len(pos_g) == 0:
            continue
        pos_locals = g2l[torch.tensor(pos_g, dtype=torch.long, device=device)]
        pos_locals = pos_locals[pos_locals >= 0]
        if pos_locals.numel() == 0:
            continue

        # Hard negatives
        hard_g = per_anchor_hard[i]
        if hard_g:
            hard_locals = g2l[torch.tensor(hard_g, dtype=torch.long, device=device)]
            hard_locals = hard_locals[hard_locals >= 0]
        else:
            hard_locals = torch.tensor([], dtype=torch.long, device=device)

        # Far negatives
        far_g = per_anchor_far[i]
        if far_g:
            far_locals = g2l[torch.tensor(far_g, dtype=torch.long, device=device)]
            far_locals = far_locals[far_locals >= 0]
        else:
            far_locals = torch.tensor([], dtype=torch.long, device=device)

        # Combine negatives, remove self and any positive overlap
        neg_parts = [t for t in [hard_locals, far_locals] if t.numel() > 0]
        if len(neg_parts) == 0:
            continue
        all_neg_locals = torch.cat(neg_parts).unique()
        pos_set = set(pos_locals.cpu().tolist())
        keep = torch.tensor(
            [v.item() not in pos_set and v.item() != a_local for v in all_neg_locals],
            dtype=torch.bool, device=device,
        )
        all_neg_locals = all_neg_locals[keep]
        if all_neg_locals.numel() == 0:
            continue

        # Cosine similarities
        sim_pos_raw = z_a @ z_support_norm[pos_locals].T
        sim_neg_raw = z_a @ z_support_norm[all_neg_locals].T

        if return_diagnostics:
            diag_sim_pos.append(sim_pos_raw.detach())
            if hard_locals.numel() > 0:
                diag_sim_hard.append((z_a @ z_support_norm[hard_locals].T).detach())
            if far_locals.numel() > 0:
                diag_sim_rand.append((z_a @ z_support_norm[far_locals].T).detach())
            diag_n_pos.append(pos_locals.numel())
            diag_n_hard.append(hard_locals.numel())
            diag_n_rand.append(far_locals.numel())

        # InfoNCE
        sim_pos_t = sim_pos_raw / tau
        sim_neg_t = sim_neg_raw / tau
        logits = torch.cat([sim_pos_t, sim_neg_t])
        log_denom = torch.logsumexp(logits, dim=0)
        loss_i = -(sim_pos_t.mean() - log_denom)
        losses.append(loss_i)

    if len(losses) == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss = torch.stack(losses).mean()

    if not return_diagnostics:
        return loss

    sim_pos_cat = torch.cat(diag_sim_pos) if diag_sim_pos else torch.tensor([0.0])
    sim_hard_cat = torch.cat(diag_sim_hard) if diag_sim_hard else torch.tensor([0.0])
    sim_rand_cat = torch.cat(diag_sim_rand) if diag_sim_rand else torch.tensor([0.0])

    stats = {
        'n_anchors': n_anc,
        'n_active': len(losses),
        'slide_picked': slide_pick.item(),
        'support_set_size': all_needed_t.shape[0],
        'sim_pos_mean': sim_pos_cat.mean().item(),
        'sim_pos_std': sim_pos_cat.std().item() if sim_pos_cat.numel() > 1 else 0.0,
        'sim_hard_mean': sim_hard_cat.mean().item(),
        'sim_hard_std': sim_hard_cat.std().item() if sim_hard_cat.numel() > 1 else 0.0,
        'sim_rand_mean': sim_rand_cat.mean().item(),
        'sim_rand_std': sim_rand_cat.std().item() if sim_rand_cat.numel() > 1 else 0.0,
        'n_pos_per_anchor': np.mean(diag_n_pos) if diag_n_pos else 0.0,
        'n_hard_per_anchor': np.mean(diag_n_hard) if diag_n_hard else 0.0,
        'n_rand_per_anchor': np.mean(diag_n_rand) if diag_n_rand else 0.0,
        'loss': loss.item(),
    }

    return loss, stats


# ==============================================================================
# SPATIAL InfoNCE DIAGNOSTICS
# ==============================================================================

def compute_spatial_infonce_loss_with_diagnostics(
    z: torch.Tensor,
    batch_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    far_mask: torch.Tensor,
    hard_neg: torch.Tensor,
    tau: float = 0.1,
    n_rand_neg: int = 128,
    is_st_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Same as compute_spatial_infonce_loss but returns detailed diagnostics.

    Returns:
        loss: scalar loss (differentiable)
        stats: dict with similarity gap diagnostics
    """
    if is_st_mask is not None:
        st_idx_in_batch = torch.where(is_st_mask)[0]
    else:
        st_idx_in_batch = torch.arange(z.shape[0], device=z.device)

    if st_idx_in_batch.numel() < 4:
        return torch.tensor(0.0, device=z.device, requires_grad=True), {
            'n_anchors': 0, 'n_active': 0,
            'sim_pos_mean': 0.0, 'sim_hard_mean': 0.0, 'sim_rand_mean': 0.0,
            'n_pos_per_anchor': 0.0, 'n_hard_per_anchor': 0.0, 'n_rand_per_anchor': 0.0,
            'loss': 0.0,
        }

    z_norm = F.normalize(z, dim=1)
    losses = []

    # Accumulators for diagnostics (raw cosine sim, NOT divided by tau)
    all_sim_pos = []
    all_sim_hard = []
    all_sim_rand = []
    n_pos_counts = []
    n_hard_counts = []
    n_rand_counts = []
    n_skipped_no_pos_global = 0
    n_skipped_no_pos_batch = 0
    n_skipped_no_neg = 0

    for local_i in st_idx_in_batch:
        g_i = batch_idx[local_i].item()
        z_anchor = z_norm[local_i]

        # Positives: physical neighbors
        pos_globals = pos_idx[g_i]
        pos_globals = pos_globals[pos_globals >= 0]
        if pos_globals.numel() == 0:
            n_skipped_no_pos_global += 1
            continue

        # Map global -> batch-local
        global_to_local = torch.full(
            (max(batch_idx.max().item() + 1, pos_globals.max().item() + 1),),
            -1, dtype=torch.long, device=z.device
        )
        global_to_local[batch_idx] = torch.arange(z.shape[0], device=z.device)

        pos_local = global_to_local[pos_globals]
        pos_local = pos_local[pos_local >= 0]
        if pos_local.numel() == 0:
            n_skipped_no_pos_batch += 1
            continue

        # Hard negatives in batch
        hard_globals = hard_neg[g_i]
        hard_globals = hard_globals[hard_globals >= 0]
        hard_local = global_to_local[hard_globals.clamp(max=global_to_local.shape[0] - 1)]
        hard_local = hard_local[hard_local >= 0]

        # Random far negatives in batch
        # FIX: far_mask is (n_st, n_st) -- only index with ST batch members
        n_st_nce = far_mask.shape[0]
        is_st_in_batch = (batch_idx < n_st_nce)
        st_local_idx = torch.where(is_st_in_batch)[0]
        if st_local_idx.numel() > 0:
            st_globals = batch_idx[st_local_idx]
            far_among_st = far_mask[g_i][st_globals]
            far_among_st = far_among_st & (st_local_idx != local_i)
            far_candidates = st_local_idx[far_among_st]
        else:
            far_candidates = torch.tensor([], dtype=torch.long, device=z.device)
        if far_candidates.numel() > n_rand_neg:
            perm = torch.randperm(far_candidates.numel(), device=z.device)[:n_rand_neg]
            rand_neg = far_candidates[perm]
        else:
            rand_neg = far_candidates

        # Combine negatives
        all_neg = torch.cat([hard_local, rand_neg]).unique()
        if all_neg.numel() == 0:
            n_skipped_no_neg += 1
            continue

        # Raw cosine similarities (for diagnostics, without tau)
        raw_sim_pos = (z_anchor @ z_norm[pos_local].T)
        raw_sim_hard = (z_anchor @ z_norm[hard_local].T) if hard_local.numel() > 0 else torch.tensor([], device=z.device)
        raw_sim_rand = (z_anchor @ z_norm[rand_neg].T) if rand_neg.numel() > 0 else torch.tensor([], device=z.device)

        all_sim_pos.append(raw_sim_pos.detach())
        if raw_sim_hard.numel() > 0:
            all_sim_hard.append(raw_sim_hard.detach())
        if raw_sim_rand.numel() > 0:
            all_sim_rand.append(raw_sim_rand.detach())

        n_pos_counts.append(pos_local.numel())
        n_hard_counts.append(hard_local.numel())
        n_rand_counts.append(rand_neg.numel())

        # InfoNCE (same as original)
        sim_pos = raw_sim_pos / tau
        sim_neg = (z_anchor @ z_norm[all_neg].T) / tau
        logits = torch.cat([sim_pos, sim_neg])
        log_denom = torch.logsumexp(logits, dim=0)
        loss_i = -(sim_pos.mean() - log_denom)
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True), {
            'n_anchors': st_idx_in_batch.numel(),
            'n_active': 0,
            'n_skipped_no_pos_global': n_skipped_no_pos_global,
            'n_skipped_no_pos_batch': n_skipped_no_pos_batch,
            'n_skipped_no_neg': n_skipped_no_neg,
            'sim_pos_mean': 0.0, 'sim_hard_mean': 0.0, 'sim_rand_mean': 0.0,
            'n_pos_per_anchor': 0.0, 'n_hard_per_anchor': 0.0, 'n_rand_per_anchor': 0.0,
            'loss': 0.0,
        }

    loss = torch.stack(losses).mean()

    # Aggregate similarity stats
    sim_pos_cat = torch.cat(all_sim_pos) if all_sim_pos else torch.tensor([0.0])
    sim_hard_cat = torch.cat(all_sim_hard) if all_sim_hard else torch.tensor([0.0])
    sim_rand_cat = torch.cat(all_sim_rand) if all_sim_rand else torch.tensor([0.0])

    stats = {
        'n_anchors': st_idx_in_batch.numel(),
        'n_active': len(losses),
        'n_skipped_no_pos_global': n_skipped_no_pos_global,
        'n_skipped_no_pos_batch': n_skipped_no_pos_batch,
        'n_skipped_no_neg': n_skipped_no_neg,
        'sim_pos_mean': sim_pos_cat.mean().item(),
        'sim_pos_std': sim_pos_cat.std().item() if sim_pos_cat.numel() > 1 else 0.0,
        'sim_hard_mean': sim_hard_cat.mean().item(),
        'sim_hard_std': sim_hard_cat.std().item() if sim_hard_cat.numel() > 1 else 0.0,
        'sim_rand_mean': sim_rand_cat.mean().item(),
        'sim_rand_std': sim_rand_cat.std().item() if sim_rand_cat.numel() > 1 else 0.0,
        'n_pos_per_anchor': np.mean(n_pos_counts) if n_pos_counts else 0.0,
        'n_hard_per_anchor': np.mean(n_hard_counts) if n_hard_counts else 0.0,
        'n_rand_per_anchor': np.mean(n_rand_counts) if n_rand_counts else 0.0,
        'loss': loss.item(),
    }

    return loss, stats


def diagnose_spatial_infonce(
    model: nn.Module,
    st_gene_expr: torch.Tensor,
    st_coords: torch.Tensor,
    sc_gene_expr: torch.Tensor,
    slide_ids: torch.Tensor,
    sc_slide_ids: Optional[torch.Tensor] = None,
    spatial_nce_weight: float = 3.0,
    spatial_nce_k_phys: int = 20,
    spatial_nce_far_mult: float = 4.0,
    spatial_nce_n_hard: int = 20,
    spatial_nce_tau: float = 0.1,
    spatial_nce_n_rand_neg: int = 128,
    spatial_nce_n_anchors: int = 64,
    vicreg_lambda_inv: float = 25.0,
    vicreg_lambda_var: float = 50.0,
    vicreg_lambda_cov: float = 1.0,
    vicreg_gamma: float = 1.0,
    vicreg_eps: float = 1e-4,
    aug_gene_dropout: float = 0.25,
    aug_gauss_std: float = 0.01,
    aug_scale_jitter: float = 0.1,
    local_align_weight: float = 0.0,
    local_align_tau_z: float = 0.07,
    local_align_bidirectional: bool = True,
    batch_size: int = 256,
    n_diagnostic_steps: int = 200,
    lr: float = 1e-4,
    device: str = 'cuda',
    seed: int = 42,
):
    """
    Run two diagnostic checks on spatial InfoNCE:

    A) Gradient check (single batch):
       - Backprop only L_spatialNCE -> record ||grad_theta||
       - Backprop only L_VICReg     -> record ||grad_theta||
       - Backprop only L_local_align -> record ||grad_theta||
       - Backprop full loss          -> record ||grad_theta||

    B) Loss scale check (first n_diagnostic_steps steps):
       - Mean L_spatialNCE per step
       - Logit gap: sim(z_i, pos) vs sim(z_i, hardneg) vs sim(z_i, randneg)

    Returns:
        results: dict with all diagnostics
    """
    import copy

    set_seed(seed)

    # --- Setup (mirrors train_encoder) ---
    model_diag = copy.deepcopy(model).to(device).train()

    n_st = st_gene_expr.shape[0]
    n_sc = sc_gene_expr.shape[0]
    X_ssl = torch.cat([st_gene_expr, sc_gene_expr], dim=0).to(device)

    domain_ids = torch.cat([
        torch.zeros(n_st, device=device, dtype=torch.long),
        torch.ones(n_sc, device=device, dtype=torch.long),
    ])

    # Precompute spatial NCE structures
    spatial_nce_data = precompute_spatial_nce_structures(
        st_coords=st_coords.to(device),
        st_gene_expr=st_gene_expr.to(device),
        slide_ids=slide_ids.to(device),
        k_phys=spatial_nce_k_phys,
        far_mult=spatial_nce_far_mult,
        n_hard=spatial_nce_n_hard,
        device=device,
    )

    vicreg_loss_fn = VICRegLoss(
        vicreg_lambda_inv, vicreg_lambda_var, vicreg_lambda_cov,
        vicreg_gamma, vicreg_eps, use_ddp_gather=False,
        compute_stats_fp32=True,
    )

    opt = torch.optim.Adam(model_diag.parameters(), lr=lr)

    # ================================================================
    # CHECK A: Gradient norms from each loss component (single batch)
    # ================================================================
    print("=" * 70)
    print("CHECK A: Per-component gradient norms (support-set NCE)")
    print("=" * 70)

    # Sample one balanced batch (for VICReg / local align)
    idx = sample_balanced_domain_and_slide_indices(
        domain_ids, slide_ids.to(device),
        sc_slide_ids.to(device) if sc_slide_ids is not None else None,
        batch_size, device,
    )
    X_batch = X_ssl[idx]
    s_batch = domain_ids[idx]

    # Augmented views for VICReg
    X1 = augment_expression(X_batch, aug_gene_dropout, aug_gauss_std, aug_scale_jitter)
    X2 = augment_expression(X_batch, aug_gene_dropout, aug_gauss_std, aug_scale_jitter)

    st_expr_dev = st_gene_expr.to(device)

    grad_norms = {}

    # A1: Spatial InfoNCE only (support-set — own forward pass)
    opt.zero_grad(set_to_none=True)
    loss_nce, nce_stats = compute_spatial_infonce_supportset(
        model=model_diag,
        st_gene_expr=st_expr_dev,
        pos_idx=spatial_nce_data['pos_idx'],
        far_mask=spatial_nce_data['far_mask'],
        hard_neg=spatial_nce_data['hard_neg'],
        slide_ids=slide_ids.to(device),
        tau=spatial_nce_tau,
        n_rand_neg=spatial_nce_n_rand_neg,
        n_anchors_per_step=spatial_nce_n_anchors,
        return_diagnostics=True,
    )
    if loss_nce.requires_grad and loss_nce.item() != 0.0:
        (spatial_nce_weight * loss_nce).backward(retain_graph=True)
        gnorm_nce = _total_grad_norm(model_diag)
    else:
        gnorm_nce = 0.0
    grad_norms['L_spatialNCE'] = gnorm_nce

    # A2: VICReg only
    opt.zero_grad(set_to_none=True)
    z1_ = model_diag(X1)
    z2_ = model_diag(X2)
    loss_vic, _ = vicreg_loss_fn(z1_, z2_)
    loss_vic.backward(retain_graph=True)
    gnorm_vic = _total_grad_norm(model_diag)
    grad_norms['L_VICReg'] = gnorm_vic

    # A3: Local alignment only
    opt.zero_grad(set_to_none=True)
    z_clean2 = model_diag(X_batch)
    is_sc = (idx >= n_st)
    z_st_b = z_clean2[~is_sc]
    z_sc_b = z_clean2[is_sc]
    if local_align_weight > 0 and z_st_b.shape[0] > 8 and z_sc_b.shape[0] > 8:
        loss_local = compute_local_alignment_loss(
            z_sc=z_sc_b, z_st=z_st_b,
            tau_z=local_align_tau_z,
            bidirectional=local_align_bidirectional,
        )
        (local_align_weight * loss_local).backward(retain_graph=True)
        gnorm_local = _total_grad_norm(model_diag)
    else:
        gnorm_local = 0.0
        loss_local = torch.tensor(0.0)
    grad_norms['L_local_align'] = gnorm_local

    # A4: Full loss (VICReg + support-set NCE)
    opt.zero_grad(set_to_none=True)
    z1_f = model_diag(X1)
    z2_f = model_diag(X2)
    loss_vic_f, _ = vicreg_loss_fn(z1_f, z2_f)
    loss_nce_f = compute_spatial_infonce_supportset(
        model=model_diag,
        st_gene_expr=st_expr_dev,
        pos_idx=spatial_nce_data['pos_idx'],
        far_mask=spatial_nce_data['far_mask'],
        hard_neg=spatial_nce_data['hard_neg'],
        slide_ids=slide_ids.to(device),
        tau=spatial_nce_tau,
        n_rand_neg=spatial_nce_n_rand_neg,
        n_anchors_per_step=spatial_nce_n_anchors,
    )
    loss_full = loss_vic_f + spatial_nce_weight * loss_nce_f
    loss_full.backward()
    gnorm_full = _total_grad_norm(model_diag)
    grad_norms['L_full'] = gnorm_full

    print(f"\n  Gradient norms (||grad_theta||):")
    print(f"    L_spatialNCE (w={spatial_nce_weight}): {gnorm_nce:.6f}")
    print(f"    L_VICReg:                              {gnorm_vic:.6f}")
    print(f"    L_local_align (w={local_align_weight}): {gnorm_local:.6f}")
    print(f"    L_full:                                {gnorm_full:.6f}")

    if gnorm_vic > 0:
        ratio = gnorm_nce / gnorm_vic
        print(f"\n    Ratio NCE/VICReg: {ratio:.6f}")
        if ratio < 1e-4:
            print("    ** SPATIAL NCE IS EFFECTIVELY A NO-OP (gradient ~0) **")
        elif ratio < 0.01:
            print("    ** WARNING: Spatial NCE gradient is very small relative to VICReg **")
        else:
            print("    ** Spatial NCE gradient is non-trivial -- loss IS live **")

    print(f"\n  Spatial NCE support-set stats:")
    for k, v in nce_stats.items():
        print(f"    {k}: {v}")

    # ================================================================
    # CHECK C: Index set sanity (physical distances + overlap)
    # ================================================================
    print("\n" + "=" * 70)
    print("CHECK C: Index set sanity — pos/neg distances & overlap")
    print("=" * 70)

    pos_idx_t = spatial_nce_data['pos_idx']   # (n_st, k_phys)
    far_mask_t = spatial_nce_data['far_mask']  # (n_st, n_st)
    hard_neg_t = spatial_nce_data['hard_neg']  # (n_st, n_hard)
    st_coords_dev = st_coords.to(device)

    n_sample_c = min(500, n_st)
    rng_c = np.random.RandomState(seed)
    sample_idx = rng_c.choice(n_st, n_sample_c, replace=False)

    pos_phys_dists_all = []   # physical distance to each positive
    hard_phys_dists_all = []  # physical distance to each hard neg
    far_counts = []           # how many far spots each anchor has
    pos_counts = []           # how many valid positives each anchor has
    hard_counts = []          # how many valid hard negs each anchor has
    overlap_counts = []       # |pos_set ∩ hard_set| per anchor
    overlap_any_neg = []      # |pos_set ∩ all_neg_set| per anchor

    for i in sample_idx:
        coord_i = st_coords_dev[i]

        # Positives
        p_globals = pos_idx_t[i]
        p_globals = p_globals[p_globals >= 0]
        pos_counts.append(p_globals.numel())
        if p_globals.numel() > 0:
            d_pos = torch.norm(st_coords_dev[p_globals] - coord_i, dim=1)
            pos_phys_dists_all.extend(d_pos.cpu().tolist())
        pos_set = set(p_globals.cpu().tolist())

        # Hard negatives
        h_globals = hard_neg_t[i]
        h_globals = h_globals[h_globals >= 0]
        hard_counts.append(h_globals.numel())
        if h_globals.numel() > 0:
            d_hard = torch.norm(st_coords_dev[h_globals] - coord_i, dim=1)
            hard_phys_dists_all.extend(d_hard.cpu().tolist())
        hard_set = set(h_globals.cpu().tolist())

        # Far mask count
        far_count = far_mask_t[i].sum().item()
        far_counts.append(far_count)

        # Overlap: positives that are also hard negatives
        overlap = pos_set & hard_set
        overlap_counts.append(len(overlap))

        # Overlap: positives that appear in hard_neg OR far_mask
        far_globals = set(torch.where(far_mask_t[i])[0].cpu().tolist())
        all_neg_set = hard_set | far_globals
        overlap_any = pos_set & all_neg_set
        overlap_any_neg.append(len(overlap_any))

    pos_phys_dists_all = np.array(pos_phys_dists_all)
    hard_phys_dists_all = np.array(hard_phys_dists_all)
    r_pos_values = spatial_nce_data['r_pos']
    r_far_threshold = spatial_nce_far_mult * np.mean(r_pos_values)

    print(f"\n  Sampled {n_sample_c} random anchors:")
    print(f"  r_pos per slide: {[f'{r:.4f}' for r in r_pos_values]}")
    print(f"  r_far threshold: {r_far_threshold:.4f} ({spatial_nce_far_mult}x mean r_pos)")
    print()
    print(f"  Positives (physical neighbors):")
    print(f"    Count per anchor: mean={np.mean(pos_counts):.1f}, min={np.min(pos_counts)}, max={np.max(pos_counts)}")
    if len(pos_phys_dists_all) > 0:
        print(f"    Physical dist: mean={pos_phys_dists_all.mean():.4f}, "
              f"median={np.median(pos_phys_dists_all):.4f}, "
              f"max={pos_phys_dists_all.max():.4f}")
        pct_pos_below_rfar = (pos_phys_dists_all < r_far_threshold).mean() * 100
        print(f"    % pos below r_far: {pct_pos_below_rfar:.1f}% (should be ~100%)")
    print()
    print(f"  Hard negatives (expr-similar but far):")
    print(f"    Count per anchor: mean={np.mean(hard_counts):.1f}, min={np.min(hard_counts)}, max={np.max(hard_counts)}")
    if len(hard_phys_dists_all) > 0:
        print(f"    Physical dist: mean={hard_phys_dists_all.mean():.4f}, "
              f"median={np.median(hard_phys_dists_all):.4f}, "
              f"min={hard_phys_dists_all.min():.4f}")
        pct_hard_above_rfar = (hard_phys_dists_all >= r_far_threshold).mean() * 100
        print(f"    % hard_neg above r_far: {pct_hard_above_rfar:.1f}% (should be ~100%)")
    print()
    print(f"  Far mask:")
    print(f"    Far spots per anchor: mean={np.mean(far_counts):.1f}, "
          f"min={np.min(far_counts)}, max={np.max(far_counts)}")
    print()
    print(f"  INDEX OVERLAP (must be zero!):")
    print(f"    |pos ∩ hard_neg|: mean={np.mean(overlap_counts):.2f}, "
          f"max={np.max(overlap_counts)}, nonzero={sum(1 for x in overlap_counts if x > 0)}/{n_sample_c}")
    print(f"    |pos ∩ (hard ∪ far)|: mean={np.mean(overlap_any_neg):.2f}, "
          f"max={np.max(overlap_any_neg)}, nonzero={sum(1 for x in overlap_any_neg if x > 0)}/{n_sample_c}")

    if np.max(overlap_counts) > 0:
        print("    ** LEAKAGE DETECTED: some positives are also hard negatives! **")
        print("    ** InfoNCE is contradictory — same spot is pushed close AND far **")
    elif np.max(overlap_any_neg) > 0:
        print("    ** LEAKAGE DETECTED: some positives are in the far set! **")
        print("    ** The loss may be pulling and pushing the same pairs **")
    else:
        print("    ** CLEAN: zero overlap between pos and neg index sets **")

    check_c_results = {
        'pos_phys_dists': pos_phys_dists_all,
        'hard_phys_dists': hard_phys_dists_all,
        'r_far_threshold': r_far_threshold,
        'pos_counts': np.array(pos_counts),
        'hard_counts': np.array(hard_counts),
        'far_counts': np.array(far_counts),
        'overlap_pos_hard': np.array(overlap_counts),
        'overlap_pos_any_neg': np.array(overlap_any_neg),
    }

    # ================================================================
    # CHECK B: Loss scale & logit gap over first N steps (support-set)
    # ================================================================
    print("\n" + "=" * 70)
    print(f"CHECK B: Loss scale & logit gap (first {n_diagnostic_steps} steps, support-set)")
    print("=" * 70)

    # Fresh model copy for the training run
    model_train = copy.deepcopy(model).to(device).train()
    opt_train = torch.optim.Adam(model_train.parameters(), lr=lr)

    st_expr_dev = st_gene_expr.to(device)

    step_logs = {
        'step': [], 'loss_nce': [], 'loss_vicreg': [],
        'sim_pos_mean': [], 'sim_hard_mean': [], 'sim_rand_mean': [],
        'sim_pos_std': [], 'sim_hard_std': [], 'sim_rand_std': [],
        'n_active_anchors': [], 'n_pos_per_anchor': [],
        'n_hard_per_anchor': [], 'n_rand_per_anchor': [],
    }

    for step in range(n_diagnostic_steps):
        # Sample batch for VICReg
        idx_s = sample_balanced_domain_and_slide_indices(
            domain_ids, slide_ids.to(device),
            sc_slide_ids.to(device) if sc_slide_ids is not None else None,
            batch_size, device,
        )
        X_b = X_ssl[idx_s]

        X1_s = augment_expression(X_b, aug_gene_dropout, aug_gauss_std, aug_scale_jitter)
        X2_s = augment_expression(X_b, aug_gene_dropout, aug_gauss_std, aug_scale_jitter)

        z1_s = model_train(X1_s)
        z2_s = model_train(X2_s)

        loss_vic_s, _ = vicreg_loss_fn(z1_s, z2_s)

        # Support-set spatial InfoNCE (does its own forward pass)
        loss_nce_s, nce_diag_s = compute_spatial_infonce_supportset(
            model=model_train,
            st_gene_expr=st_expr_dev,
            pos_idx=spatial_nce_data['pos_idx'],
            far_mask=spatial_nce_data['far_mask'],
            hard_neg=spatial_nce_data['hard_neg'],
            slide_ids=slide_ids.to(device),
            tau=spatial_nce_tau,
            n_rand_neg=spatial_nce_n_rand_neg,
            n_anchors_per_step=spatial_nce_n_anchors,
            return_diagnostics=True,
        )

        loss_total_s = loss_vic_s + spatial_nce_weight * loss_nce_s

        opt_train.zero_grad(set_to_none=True)
        loss_total_s.backward()
        opt_train.step()

        # Log
        step_logs['step'].append(step)
        step_logs['loss_nce'].append(loss_nce_s.item())
        step_logs['loss_vicreg'].append(loss_vic_s.item())
        step_logs['sim_pos_mean'].append(nce_diag_s['sim_pos_mean'])
        step_logs['sim_hard_mean'].append(nce_diag_s['sim_hard_mean'])
        step_logs['sim_rand_mean'].append(nce_diag_s['sim_rand_mean'])
        step_logs['sim_pos_std'].append(nce_diag_s.get('sim_pos_std', 0.0))
        step_logs['sim_hard_std'].append(nce_diag_s.get('sim_hard_std', 0.0))
        step_logs['sim_rand_std'].append(nce_diag_s.get('sim_rand_std', 0.0))
        step_logs['n_active_anchors'].append(nce_diag_s['n_active'])
        step_logs['n_pos_per_anchor'].append(nce_diag_s['n_pos_per_anchor'])
        step_logs['n_hard_per_anchor'].append(nce_diag_s['n_hard_per_anchor'])
        step_logs['n_rand_per_anchor'].append(nce_diag_s['n_rand_per_anchor'])

        if step % 50 == 0 or step < 5:
            print(f"  step {step:4d} | NCE={loss_nce_s.item():.4f} VIC={loss_vic_s.item():.4f} | "
                  f"sim(pos)={nce_diag_s['sim_pos_mean']:.4f} sim(hard)={nce_diag_s['sim_hard_mean']:.4f} "
                  f"sim(rand)={nce_diag_s['sim_rand_mean']:.4f} | "
                  f"active={nce_diag_s['n_active']}/{nce_diag_s['n_anchors']} | "
                  f"support={nce_diag_s['support_set_size']}")

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY over all steps:")

    arr = lambda k: np.array(step_logs[k])

    mean_nce = arr('loss_nce').mean()
    mean_vic = arr('loss_vicreg').mean()
    mean_pos = arr('sim_pos_mean').mean()
    mean_hard = arr('sim_hard_mean').mean()
    mean_rand = arr('sim_rand_mean').mean()
    mean_active = arr('n_active_anchors').mean()
    mean_n_pos = arr('n_pos_per_anchor').mean()
    mean_n_hard = arr('n_hard_per_anchor').mean()
    mean_n_rand = arr('n_rand_per_anchor').mean()

    print(f"  Mean L_spatialNCE:  {mean_nce:.4f}")
    print(f"  Mean L_VICReg:      {mean_vic:.4f}")
    print(f"  Weighted NCE loss:  {spatial_nce_weight * mean_nce:.4f}")
    print(f"  NCE / total ratio:  {spatial_nce_weight * mean_nce / (mean_vic + spatial_nce_weight * mean_nce + 1e-10):.4f}")
    print()
    print(f"  Mean sim(anchor, pos):      {mean_pos:.4f}")
    print(f"  Mean sim(anchor, hard_neg): {mean_hard:.4f}")
    print(f"  Mean sim(anchor, rand_neg): {mean_rand:.4f}")
    print(f"  Gap pos - hard:             {mean_pos - mean_hard:.4f}")
    print(f"  Gap pos - rand:             {mean_pos - mean_rand:.4f}")
    print()
    print(f"  Mean active anchors/batch:  {mean_active:.1f}")
    print(f"  Mean pos/anchor:            {mean_n_pos:.1f}")
    print(f"  Mean hard_neg/anchor:       {mean_n_hard:.1f}")
    print(f"  Mean rand_neg/anchor:       {mean_n_rand:.1f}")
    print()

    # Interpretation
    if mean_active < 2:
        print("  ** CRITICAL: Almost no anchors have both pos AND neg! **")
        print("     -> Check pos_idx / hard_neg precomputation.")
    elif mean_nce < 0.01:
        print("  ** WARNING: NCE loss is near zero -- already saturated or disconnected. **")
    elif mean_pos > mean_hard + 0.1:
        print("  ** Positives already more similar than hard negatives -- loss is already solved. **")
        print("     -> Spatial structure may already be captured, or positives are trivially similar.")
    elif mean_pos < mean_hard:
        print("  ** Positives LESS similar than hard negatives -- loss should be active. **")
        print("     -> This is the expected hard case. Check gradient norm to confirm it's training.")
    else:
        print("  ** sim(pos) ~ sim(hard) -- loss should be pushing gradients. **")

    return {
        'grad_norms': grad_norms,
        'nce_single_batch_stats': nce_stats,
        'step_logs': step_logs,
        'spatial_nce_data_r_pos': spatial_nce_data['r_pos'],
        'check_c': check_c_results,
    }


def _total_grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5