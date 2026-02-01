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
        
        if verbose and epoch % 50 == 0:
            print(f"  [{prefix}] Acc={acc:.3f}, Ent={ent_mean:.3f}/{max_entropy:.3f} "
                  f"(ratio={ent_ratio:.3f})")
            print(f"  [{prefix}] Per-class acc: {per_class_acc}")
            print(f"  [{prefix}] Confusion matrix:\n{cm.cpu().numpy()}")
        
        return stats



# ==============================================================================
# LOCAL ALIGNMENT LOSS (Run 4 - GPT 5.2 Pro)
# ==============================================================================

# def compute_local_alignment_loss(
#     z_sc: torch.Tensor,
#     z_st: torch.Tensor, 
#     x_sc: torch.Tensor,
#     x_st: torch.Tensor,
#     tau_x: float = 0.1,
#     tau_z: float = 0.1,
#     bidirectional: bool = True
# ) -> torch.Tensor:
#     """
#     Local alignment loss: Match neighborhood distributions between expression and embedding space.
    
#     For each SC cell, compute soft neighbors in ST based on:
#       - Teacher: expression similarity (ground truth)
#       - Student: embedding similarity (learned)
#     Then minimize KL(teacher || student).
    
#     Args:
#         z_sc: (n_sc, D) SC embeddings from encoder
#         z_st: (n_st, D) ST embeddings from encoder
#         x_sc: (n_sc, G) SC expression (raw, not encoded)
#         x_st: (n_st, G) ST expression (raw, not encoded)
#         tau_x: temperature for expression similarities (teacher)
#         tau_z: temperature for embedding similarities (student)
#         bidirectional: if True, also compute ST->SC alignment
        
#     Returns:
#         loss: scalar local alignment loss
#     """
#     # Normalize for cosine similarity
#     x_sc_norm = F.normalize(x_sc, dim=1)  # (n_sc, G)
#     x_st_norm = F.normalize(x_st, dim=1)  # (n_st, G)
#     z_sc_norm = F.normalize(z_sc, dim=1)  # (n_sc, D)
#     z_st_norm = F.normalize(z_st, dim=1)  # (n_st, D)
    
#     # ========== SC -> ST direction ==========
#     # Teacher: expression-based soft neighbors (SC queries ST)
#     sim_x_sc2st = (x_sc_norm @ x_st_norm.T) / tau_x  # (n_sc, n_st)
#     q_sc2st = F.softmax(sim_x_sc2st, dim=1)  # Teacher distribution
    
#     # Student: embedding-based soft neighbors
#     sim_z_sc2st = (z_sc_norm @ z_st_norm.T) / tau_z  # (n_sc, n_st)
#     log_p_sc2st = F.log_softmax(sim_z_sc2st, dim=1)  # Student log-distribution
    
#     # KL divergence: KL(q || p) = sum_j q_j * (log q_j - log p_j)
#     # Using F.kl_div which expects log_p and q
#     loss_sc2st = F.kl_div(log_p_sc2st, q_sc2st, reduction='batchmean')
    
#     if not bidirectional:
#         return loss_sc2st
    
#     # ========== ST -> SC direction ==========
#     # Teacher: expression-based soft neighbors (ST queries SC)
#     sim_x_st2sc = (x_st_norm @ x_sc_norm.T) / tau_x  # (n_st, n_sc)
#     q_st2sc = F.softmax(sim_x_st2sc, dim=1)
    
#     # Student: embedding-based soft neighbors
#     sim_z_st2sc = (z_st_norm @ z_sc_norm.T) / tau_z  # (n_st, n_sc)
#     log_p_st2sc = F.log_softmax(sim_z_st2sc, dim=1)
    
#     loss_st2sc = F.kl_div(log_p_st2sc, q_st2sc, reduction='batchmean')
    
#     # Average both directions
#     return 0.5 * (loss_sc2st + loss_st2sc)

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
