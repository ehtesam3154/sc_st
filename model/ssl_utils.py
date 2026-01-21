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