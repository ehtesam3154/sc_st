"""
GEMS (Generative Euclidean Metric Synthesis) - Core Model Classes
Part 2: Stage C (Generator + Diffusion), Stage D (Inference), Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional

# Import from project knowledge Set-Transformer components
from modules import MAB, SAB, ISAB, PMA
import utils_et as uet

from tqdm import tqdm

import matplotlib.pyplot as plt
import json
from datetime import datetime

from torch.utils.data import DataLoader
from core_models_et_p1 import collate_minisets, collate_sc_minisets
from tqdm.auto import tqdm
import sys

from typing import Optional
try:
    from lightning.fabric import Fabric
except Exception:
    Fabric = None  # allow running without Fabric

# ---- simple CUDA-aware timers ----
import time
from contextlib import contextmanager

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@contextmanager
def timed(section_name, bucket):
    _cuda_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _cuda_sync()
        dt = time.perf_counter() - t0
        bucket[section_name] = bucket.get(section_name, 0.0) + dt

# ==============================================================================
# STAGE C: SET-EQUIVARIANT CONTEXT ENCODER
# ==============================================================================

class SetEncoderContext(nn.Module):
    """
    Permutation-equivariant context encoder using Set Transformer.
    
    Takes set of embeddings Z_set and produces context H.
    Uses ISAB blocks for O(mn) complexity.
    """
    
    def __init__(
        self,
        h_dim: int = 128,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 3,
        isab_m: int = 64,
        ln: bool = True
    ):
        """
        Args:
            h_dim: input embedding dimension
            c_dim: output context dimension
            n_heads: number of attention heads
            n_blocks: number of ISAB blocks
            isab_m: number of inducing points in ISAB
            ln: use layer normalization
        """
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim
        
        # Input projection
        self.input_proj = nn.Linear(h_dim, c_dim)
        
        # Stack of ISAB blocks
        self.isab_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])
    
    def forward(self, Z_set: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_set: (batch, n, h_dim) set of embeddings
            mask: (batch, n) boolean mask (True = valid)
            
        Returns:
            H: (batch, n, c_dim) context features
        """
        batch_size, n, _ = Z_set.shape
        
        # Project to context dimension
        H = self.input_proj(Z_set)  # (batch, n, c_dim)

        for isab in self.isab_blocks:
            # ISAB expects (batch, n, dim); keep the set intact
            H = isab(H, mask=mask)
            H = H * mask.unsqueeze(-1).float()

        
        return H
    
# def precompute_st_prototypes(
#     targets_dict: Dict[int, 'STTargets'],
#     encoder: 'SharedEncoder',
#     st_gene_expr_dict: Dict[int, torch.Tensor],
#     n_prototypes: int = 3000,
#     n_min: int = 64,
#     n_max: int = 256,
#     device: str = 'cuda'
# ) -> Dict:
#     '''
#     precompute ST prototypes for Z-matched distogram loss
#     '''

#     print(f'processing {n_prototypes} ST protypes.....')

#     encoder.eval()
#     Z_dict = {}
#     with torch.no_grad():
#         for slide_id, st_expr in st_gene_expr_dict.items():
#             Z = encoder(st_expr.to(device))
#             Z_dict[slide_id] = Z

#     prototypes = []
#     slide_ids = list(targets_dict.keys())

#     for _ in range(n_prototypes):
#         slide_id = slide_ids[torch.randint(len(slide_ids), (1,)).item()]
#         targets = targets_dict[slide_id]
#         Z_slide = Z_dict[slide_id]

#         m = targets.D.shape[0]
#         n = torch.randint(n_min, min(n_max + 1, m+1), (1, )).item()
#         indices = torch.randperm(m)[:n]

#         Z_subset = Z_slide[indices]
#         centroid_Z = Z_subset.mean(dim=0)

#         D_subset = targets.D[indices][:, indices]
#         d_95 = torch.quantile(D_subset[torch.triu(torch.ones_like(D_subset), diagonal=1).bool()], 0.95)
#         # bins = torch.linspace(0, d_95, 64, device=device)
#         bins = torch.linspace(0, d_95, 48, device=device)  # was: 64
#         hist = uet.compute_distance_hist(D_subset, bins)

#         prototypes.append({
#             'centroid_Z': centroid_Z.cpu(),
#             'hist': hist.cpu(),
#             'bins': bins.cpu(),
#             'D': D_subset.cpu()
#         })

#     #stack centroid for fast lookup
#     centroids = torch.stack([p['centroid_Z'] for p in prototypes])

#     print(f'precomputed {len(prototypes)} protoypes')
#     return {'prototypes': prototypes, 'centroids': centroids}

# def precompute_st_prototypes(
#     targets_dict: Dict[int, 'STTargets'],
#     encoder: 'SharedEncoder',
#     st_gene_expr_dict: Dict[int, torch.Tensor],
#     n_prototypes: int = 3000,
#     n_min: int = 64,
#     n_max: int = 256,
#     device: str = 'cuda'
# ) -> Dict:
#     """
#     Precompute ST prototypes for Z-matched distogram loss.
#     Returns GPU tensors ready for batched operations.
#     """
#     print(f'Processing {n_prototypes} ST prototypes.....')
#     encoder.eval()
    
#     # Encode all slides once
#     Z_dict = {}
#     with torch.no_grad():
#         for slide_id, st_expr in st_gene_expr_dict.items():
#             Z = encoder(st_expr.to(device))
#             Z_dict[slide_id] = Z
    
#     # Create SHARED bin edges for all prototypes (faster batched operations)
#     # Use a reasonable max distance based on data
#     max_distance = 0.0
#     slide_ids = list(targets_dict.keys())
    
#     # Sample a few slides to estimate max distance range
#     for _ in range(min(10, len(slide_ids))):
#         slide_id = slide_ids[torch.randint(len(slide_ids), (1,)).item()]
#         targets = targets_dict[slide_id]
#         D_sample = targets.D[:min(100, targets.D.shape[0]), :min(100, targets.D.shape[0])]
#         max_distance = max(max_distance, D_sample.max().item())
    
#     # Shared bins across all prototypes (on GPU)
#     shared_bins = torch.linspace(0, max_distance * 1.2, 49, device=device)  # 48 histogram bins
#     nb = shared_bins.numel() - 1
    
#     # Pre-allocate tensors for batched storage
#     centroid_list = []
#     hist_list = []
    
#     with torch.no_grad():
#         for _ in range(n_prototypes):
#             slide_id = slide_ids[torch.randint(len(slide_ids), (1,)).item()]
#             targets = targets_dict[slide_id]
#             Z_slide = Z_dict[slide_id]
            
#             m = targets.D.shape[0]
#             n = torch.randint(n_min, min(n_max + 1, m + 1), (1,)).item()
#             indices = torch.randperm(m, device=device)[:n]
            
#             Z_subset = Z_slide[indices]
#             centroid_Z = Z_subset.mean(dim=0)
            
#             # D_subset = targets.D[indices][:, indices]
#             D_subset = targets.D.to(device)[indices][:, indices]
            
#             # Use SHARED bins (no per-prototype bins!)
#             hist = uet.compute_distance_hist(D_subset, shared_bins)
            
#             centroid_list.append(centroid_Z)
#             hist_list.append(hist)
    
#     # Stack everything into batched tensors (P, h_dim), (P, nb)
#     centroids = torch.stack(centroid_list, dim=0)  # (P, h_dim)
#     hists = torch.stack(hist_list, dim=0)  # (P, nb)
    
#     print(f'Precomputed {len(centroid_list)} prototypes')
    
#     # Return format optimized for batched GPU operations
#     return {
#         'centroids': centroids,  # (P, h_dim) on GPU
#         'hists': hists,          # (P, nb) on GPU
#         'bins': shared_bins      # (nb+1,) on GPU
#     }

# core_models_et_p2.py
import torch
import torch.nn.functional as F
import utils_et as uet
from typing import Dict

@torch.no_grad()
def precompute_st_prototypes(
    targets_dict: Dict[int, "STTargets"],
    encoder: "SharedEncoder",
    st_gene_expr_dict: Dict[int, torch.Tensor],
    n_prototypes: int = 3000,
    n_min: int = 64,
    n_max: int = 256,
    nbins: int = 64,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Build a CUDA prototype bank:
      - centroids: (P, h_dim) on CUDA
      - hists:     (P, nbins) CUDA (unified bin edges for all)
      - bins:      (nbins+1,) CUDA
    """
    encoder.eval()
    Z_dict = {}
    for slide_id, st_expr in st_gene_expr_dict.items():
        Z = encoder(st_expr.to(device))
        Z_dict[slide_id] = Z

    centroids = []
    d95_list = []
    proto_sets = []
    slide_ids = list(targets_dict.keys())
    for _ in range(n_prototypes):
        sid = slide_ids[torch.randint(len(slide_ids), (1,)).item()]
        targets = targets_dict[sid]
        Z = Z_dict[sid]
        m = targets.D.shape[0]
        n = torch.randint(n_min, min(n_max + 1, m + 1), (1,)).item()
        idx = torch.randperm(m, device=device)[:n]
        Zs = Z[idx]                                         # (n, h)
        centroids.append(Zs.mean(dim=0))
        D = targets.D.to(device)[idx][:, idx].float()
        iu, ju = torch.triu_indices(n, n, 1, device=device)
        d95_list.append(torch.quantile(D[iu, ju], 0.95))
        proto_sets.append(D)

    # unified bins by median 95th
    d95 = torch.stack(d95_list).median()
    bins = torch.linspace(0, d95, steps=nbins+1, device=device)

    # hists
    hists = []
    for D in proto_sets:
        h = uet.compute_distance_hist(D, bins)
        hists.append(h)
    hists = torch.stack(hists, dim=0)                       # (P, nbins)
    centroids = torch.stack(centroids, dim=0)               # (P, h)

    return {"centroids": centroids, "hists": hists, "bins": bins}

# core_models_et_p2.py
@torch.no_grad()
def match_prototypes_batched(Z_set: torch.Tensor, mask: torch.Tensor, bank: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Z_set: (B, N, h), mask: (B, N)
    bank: {'centroids': (P,h), 'hists': (P,nb), 'bins': (nb+1)}
    Returns: H_target (B, nb)
    """
    B, N, h = Z_set.shape
    Zc = (Z_set * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    sim = F.normalize(Zc, dim=1) @ F.normalize(bank["centroids"], dim=1).T    # (B,P)
    best = sim.argmax(dim=1)                                                   # (B,)
    return bank["hists"][best]                                                 # (B, nb)


def find_nearest_prototypes(Z_centroid: torch.Tensor, prototype_bank: Dict) -> int:
    '''
    find nearest ST protoype by centroid matching
    '''

    centroids = prototype_bank['centroids']
    Z_centroid_norm = Z_centroid / (Z_centroid.norm() + 1e-8)
    centroids_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)

    similarities = centroids_norm @ Z_centroid_norm
    best_idx = similarities.argmax().item()
    return best_idx 


# ==============================================================================
# STAGE C: METRIC SET GENERATOR Φθ
# ==============================================================================

class MetricSetGenerator(nn.Module):
    """
    Generator that produces V ∈ ℝ^{n×D} from context H.
    
    Architecture: Stack of SAB/ISAB blocks + MLP head
    Output: V_0 with row-mean centering (translation neutrality)
    """
    
    def __init__(
        self,
        c_dim: int = 256,
        D_latent: int = 16,
        n_heads: int = 4,
        n_blocks: int = 2,
        isab_m: int = 64,
        ln: bool = True
    ):
        """
        Args:
            c_dim: context dimension
            D_latent: latent dimension of V
            n_heads: number of attention heads
            n_blocks: number of SAB/ISAB blocks
            isab_m: number of inducing points
            ln: use layer normalization
        """
        super().__init__()
        self.c_dim = c_dim
        self.D_latent = D_latent
        
        # Stack of ISAB blocks
        self.isab_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])
        
        # MLP head to produce V
        self.head = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.ReLU(),
            nn.Linear(c_dim, D_latent)
        )
    
    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch, n, c_dim) context features
            mask: (batch, n) boolean mask
            
        Returns:
            V: (batch, n, D_latent) factor matrix (row-mean centered)
        """
        batch_size, n, _ = H.shape
        
        # Apply ISAB blocks
        X = H
        for isab in self.isab_blocks:
            X = isab(X)
            X = X * mask.unsqueeze(-1).float()
        
        # MLP head
        V = self.head(X)  # (batch, n, D_latent)
        
        # Row-mean centering (translation neutrality)
        V_centered = V - V.mean(dim=1, keepdim=True)
        
        # Apply mask
        V_centered = V_centered * mask.unsqueeze(-1).float()
        
        return V_centered
    
# ==============================================================================
# STAGE C: DIFFUSION SCORE NETWORK sψ
# ==============================================================================

class DiffusionScoreNet(nn.Module):
    """
    Conditional denoiser for V_t → ε̂.
    
    Set-equivariant architecture with time embedding.
    VE SDE: σ_min=0.01, σ_max=50
    """
    
    def __init__(
        self,
        D_latent: int = 16,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        isab_m: int = 64,
        time_emb_dim: int = 128,
        ln: bool = True
    ):
        """
        Args:
            D_latent: dimension of V
            c_dim: context dimension
            n_heads: number of attention heads
            n_blocks: number of denoising blocks
            isab_m: number of inducing points
            time_emb_dim: dimension of time embedding
            ln: layer normalization
        """
        super().__init__()
        self.D_latent = D_latent
        self.c_dim = c_dim
        self.time_emb_dim = time_emb_dim
        
        # Time embedding (Fourier features)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, c_dim)
        )
        
        # Input projection: V_t + H → combined features
        self.input_proj = nn.Linear(D_latent + c_dim, c_dim)
        
        # Denoising blocks (ISAB)
        self.denoise_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])
        
        # Output head: predict noise ε
        self.output_head = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, D_latent)
        )
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Fourier time embedding.
        
        Args:
            t: (batch, 1) normalized time in [0, 1]
            
        Returns:
            emb: (batch, time_emb_dim) time embedding
        """
        half_dim = self.time_emb_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
    
    def forward(self, V_t: torch.Tensor, t: torch.Tensor, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V_t: (batch, n, D_latent) noisy factor
            t: (batch, 1) normalized time
            H: (batch, n, c_dim) context
            mask: (batch, n) boolean mask
            
        Returns:
            eps_pred: (batch, n, D_latent) predicted noise
        """
        batch_size, n, _ = V_t.shape
        
        # Time embedding
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, n, -1)
        
        # Combine V_t and H
        V_H = torch.cat([V_t, H], dim=-1)
        X = self.input_proj(V_H)
        X = X + t_emb
        
        # Apply blocks with gradient checkpointing during training
        for block in self.denoise_blocks:
            if self.training:
                X = torch.utils.checkpoint.checkpoint(block, X, use_reentrant=False)
            else:
                X = block(X)
            X = X * mask.unsqueeze(-1).float()
        
        # Predict noise
        eps_pred = self.output_head(X)
        eps_pred = eps_pred * mask.unsqueeze(-1).float()
        
        return eps_pred
    
# ==============================================================================
# STAGE C: TRAINING FUNCTION
# ==============================================================================

def plot_losses(history, plot_dir, current_epoch):
    """
    Plot and save loss curves.
    
    Args:
        history: dict with 'epoch' and 'epoch_avg' keys
        plot_dir: directory to save plots
        current_epoch: current epoch number
    """
    epochs = history['epoch']
    avg = history['epoch_avg']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Stage C Training Losses (Epoch {current_epoch})', fontsize=16, fontweight='bold')
    
    # Plot each loss component
    loss_names = ['total', 'score', 'gram', 'heat', 'sw', 'triplet']
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, color) in enumerate(zip(loss_names, colors)):
        ax = axes[idx // 3, idx % 3]
        ax.plot(epochs, avg[name], color=color, linewidth=2, label=name.capitalize())
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{name.capitalize()} Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add smoothed trend line if enough data
        if len(epochs) > 10:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(avg[name], sigma=2)
            ax.plot(epochs, smoothed, '--', color=color, alpha=0.5, linewidth=1.5, label='Trend')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'losses_epoch_{current_epoch:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save a combined plot showing all losses on same axes (log scale)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for name, color in zip(loss_names[1:], colors[1:]):  # Skip 'total'
        ax.plot(epochs, avg[name], color=color, linewidth=2, label=name.capitalize(), marker='o', markersize=3, markevery=max(1, len(epochs)//20))
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('All Loss Components (Epoch-Averaged)', fontsize=16, fontweight='bold')
    ax.set_yscale('log')  # Log scale to see all components
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'combined_losses_epoch_{current_epoch:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_dir}/")


def train_stageC_diffusion_generator(
    context_encoder: 'SetEncoderContext',
    generator: 'MetricSetGenerator',
    score_net: 'DiffusionScoreNet',
    st_dataset: 'STSetDataset',
    sc_dataset: 'SCSetDataset',
    prototype_bank: Dict,
    n_epochs: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-4,
    n_timesteps: int = 500,  # CHANGED from 600
    sigma_min: float = 0.01,
    sigma_max: float = 5.0,
    device: str = 'cuda',
    outf: str = 'output',
    fabric: Optional['Fabric'] = None,
    precision: str = '16-mixed' # "32-true" | "16-mixed" | "bf16-mixed"
):
    """
    Train diffusion generator with mixed ST/SC regimen.
    OPTIMIZED with: AMP, loss alternation, CFG, reduced overlap computation.
    """
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup AMP with new API
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # scaler = torch.amp.GradScaler(enabled=not use_bf16)  # auto-noop for bf16
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if fabric is None else None
    
    context_encoder = context_encoder.to(device).train()
    score_net = score_net.to(device).train()
    
    params = list(context_encoder.parameters()) + list(score_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    # --- NEW: wrap models + optimizer for DDP ---
    if fabric is not None:
        context_encoder, score_net, optimizer = fabric.setup(context_encoder, score_net, optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # VE SDE
    sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), n_timesteps, device=device))
    
    # Loss modules - OPTIMIZED: Heat loss with Hutchinson
    loss_gram = uet.FrobeniusGramLoss()
    loss_heat = uet.HeatKernelLoss(
        use_hutchinson=True,
        num_probes=64,
        chebyshev_degree=30,
        knn_k=12,
        t_list=(0.5, 1.0),
        laplacian='sym'
    )
    loss_sw = uet.SlicedWassersteinLoss1D()
    loss_triplet = uet.OrdinalTripletLoss()
    
    # DataLoaders - OPTIMIZED
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_minisets, collate_sc_minisets

    if fabric is not None:
        device = str(fabric.device)
    
    st_loader = DataLoader(
        st_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_minisets, 
        num_workers=0,          # CHANGED from 0
        pin_memory=False
    )
    sc_loader = DataLoader(
        sc_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_sc_minisets, 
        num_workers=0,         # CHANGED from 0
        pin_memory=False
    )

    from utils_et import build_sc_knn_cache

    # Build kNN cache from SC dataset embeddings
    print("Building SC kNN cache...")
    sc_knn = build_sc_knn_cache(
        sc_dataset.Z_cpu,
        k_pos=25,
        block_q=2048,
        device=device
    )
    POS_IDX = sc_knn["pos_idx"]
    K_POS = int(sc_knn["k_pos"])
    print(f"SC kNN cache built: {POS_IDX.shape}")

    # Optional: save to disk
    # torch.save(sc_knn, f"{outf}/sc_knn_cache.pt")

    #wrap dataloaders for ddp sharding
    if fabric is not None:
        st_loader = fabric.setup_dataloaders(st_loader)
        sc_loader = fabric.setup_dataloaders(sc_loader)
    
    os.makedirs(outf, exist_ok=True)
    plot_dir = os.path.join(outf, 'plots')
    #only rank 0 touches the filesystem
    if fabric is None or fabric.is_global_zero:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Loss weights
    WEIGHTS = {
        'score': 1.0,
        'gram': 0.5,
        'heat': 0.25,
        'sw_st': 0.5,
        'sw_sc': 0.3,
        'overlap': 0.25,
        'ordinal_sc': 0.5
    }
    
    # Overlap config - OPTIMIZED
    EVERY_K_STEPS = 2   # CHANGED from 2
    MAX_OVERLAP_POINTS = 64  # CHANGED from 64
    MIN_OVERLAP_ABS = 5

    # NEW: compute-heavy loss cadence + subsampling
    heat_every_k = 1          # compute heat loss every 4th step only
    sw_every_k = 1            # sliced-W every 2nd step
    gram_pair_cap = 8000      # sample up to 8k pairs for Gram (mask upper-tri)
    triplet_cap = 10000       # cap triplets per batch
    hist_bins = 48            # fewer bins → faster histogram distances
    
    # CFG config (same as original)
    p_uncond = 0.15  # 10-20% works well
    
    history = {
        'epoch': [],
        'batch_losses': [],
        'epoch_avg': {
            'total': [], 'score': [], 'gram': [], 'heat': [], 
            'sw_st': [], 'sw_sc': [], 'overlap': [], 'ordinal_sc': []
        }
    }
    
    # Compute sigma_data once at start
    print("Computing sigma_data from data statistics...")
    with torch.no_grad():
        sample_stds = []
        for _ in range(min(10, len(st_loader))):
            sample_batch = next(iter(st_loader))
            # CRITICAL: Move batch to device first
            G_batch = sample_batch['G_target']
            if not G_batch.is_cuda:
                G_batch = G_batch.to(device)
            for i in range(min(4, G_batch.shape[0])):
                V_temp = uet.factor_from_gram(G_batch[i], score_net.D_latent)
                sample_stds.append(V_temp.std().item())
        sigma_data = np.median(sample_stds)
        print(f"Computed sigma_data = {sigma_data:.4f}")

    #--amp scaler choice based on precision---
    use_fp16 = (precision == '16-mixed')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    global_step = 0
    
    for epoch in range(n_epochs):
        st_iter = iter(st_loader)
        sc_iter = iter(sc_loader)
        
        epoch_losses = {k: 0.0 for k in WEIGHTS.keys()}
        epoch_losses['total'] = 0.0
        n_batches = 0
        
        # Mixed schedule: [ST, ST, SC] repeat
        schedule = ['ST', 'ST', 'SC'] * (max(len(st_loader), len(sc_loader)) // 3 + 1)
        
        # Batch progress bar
        batch_pbar = tqdm(schedule, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True)
        
        for batch_type in batch_pbar:
            if batch_type == 'ST':
                batch = next(st_iter, None)
                if batch is None:
                    st_iter = iter(st_loader)
                    batch = next(st_iter, None)
                    if batch is None:
                        continue
            else:
                batch = next(sc_iter, None)
                if batch is None:
                    sc_iter = iter(sc_loader)
                    batch = next(sc_iter, None)
                    if batch is None:
                        continue
            
            is_sc = batch.get('is_sc', False)
            
            Z_set = batch['Z_set'].to(device)
            mask = batch['mask'].to(device)
            n_list = batch['n']
            batch_size_real = Z_set.shape[0]
            
            D_latent = score_net.D_latent
            
            # ===== FORWARD PASS WITH AMP =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                # Context encoding
                H = context_encoder(Z_set, mask)
                
                # Sample noise level
                t_idx = torch.randint(0, n_timesteps, (batch_size_real,), device=device)
                t_norm = t_idx.float() / n_timesteps
                sigma_t = sigmas[t_idx].view(-1, 1, 1)
                
                # Generate V_0 using generator (works for both ST and SC)
                V_0 = generator(H, mask)

                # For ST: still load G_target (needed for Gram loss later)
                if not is_sc:
                    G_target = batch['G_target'].to(device)
                
                # Add noise
                eps = torch.randn_like(V_0)
                V_t = V_0 + sigma_t * eps
                V_t = V_t * mask.unsqueeze(-1).float()
                
                # CFG: Drop context randomly during training (RESTORED)
                drop_mask = (torch.rand(batch_size_real, device=device) < p_uncond).float().view(-1, 1, 1)
                H_train = H * (1 - drop_mask)  # Zero context for random subset
                
                # Predict noise
                eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask)
            
            # ===== SCORE LOSS (in fp32 for numerical stability) =====
            with torch.autocast(device_type='cuda', enabled=False):
                sigma_t_fp32 = sigma_t.float()
                eps_pred_fp32 = eps_pred.float()
                eps_fp32 = eps.float()
                mask_fp32 = mask.float()
                
                # EDM loss weighting
                w = (sigma_t_fp32**2 + sigma_data**2) / ((sigma_t_fp32 * sigma_data)**2)
                L_score = (w * (eps_pred_fp32 - eps_fp32)**2 * mask_fp32.unsqueeze(-1)).sum() / mask_fp32.sum()
            
            # Initialize other losses
            L_gram = torch.tensor(0.0, device=device)
            L_heat = torch.tensor(0.0, device=device)
            L_sw_st = torch.tensor(0.0, device=device)
            L_sw_sc = torch.tensor(0.0, device=device)
            L_overlap = torch.tensor(0.0, device=device)
            L_ordinal_sc = torch.tensor(0.0, device=device)
            
            # Denoise to get V_hat
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                V_hat = V_t - sigma_t * eps_pred
                V_hat = V_hat * mask.unsqueeze(-1).float()
            
            # ===== LOSS ALTERNATION =====
            if not is_sc:
                # ===== ST STEP: Score + Gram + Heat + SW_ST =====
                
                # Gram loss (in fp32)
                # with torch.autocast(device_type='cuda', enabled=False):
                #     V_fp32 = V_hat.float()
                #     G_pred = V_fp32 @ V_fp32.transpose(1, 2)
                #     G_targ = G_target.float()

                #     #use masked frob function
                #     L_gram = uet.masked_frobenius_loss(G_pred, G_targ, mask)

                with torch.autocast(device_type='cuda', enabled=False):
                    V = V_hat.float()
                    Gt = G_target.float()
                    m  = mask

                    # center + scale-match as above
                    mw   = m.float()
                    cnt  = mw.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean = (V * mw.unsqueeze(-1)).sum(dim=1, keepdim=True) / cnt.unsqueeze(-1)
                    Vc   = V - mean
                    Gp   = Vc @ Vc.transpose(1, 2)

                    tr_p = (Gp.diagonal(dim1=1, dim2=2) * mw).sum(dim=1, keepdim=True).clamp_min(1e-8)
                    tr_t = (Gt.diagonal(dim1=1, dim2=2) * mw).sum(dim=1, keepdim=True).clamp_min(1e-8)
                    s2   = (tr_t / tr_p)
                    Gp   = Gp * s2.unsqueeze(-1)

                    L_gram = uet.masked_frobenius_loss(Gp, Gt, m, drop_diag=True)

                
                # Heat kernel loss (batched)
                if (global_step % heat_every_k) == 0:
                    L_info_batch = batch.get('L_info', [])
                    if L_info_batch:
                        with torch.autocast(device_type='cuda', enabled=False):
                            # Pre-allocate batched Laplacian tensors
                            n_max = V_hat.size(1)
                            L_pred_batch = torch.zeros(batch_size_real, n_max, n_max, device=device, dtype=torch.float32)
                            L_targ_batch = torch.zeros(batch_size_real, n_max, n_max, device=device, dtype=torch.float32)
                            
                            # Build Laplacians (loop needed for kNN graph construction)
                            for i in range(batch_size_real):
                                n_valid = mask[i].sum().item()  # Single .item() call per sample - acceptable
                                V_i = V_hat[i, :n_valid].float()
                                
                                # Build predicted Laplacian
                                D_V = torch.cdist(V_i, V_i)
                                edge_index, edge_weight = uet.build_knn_graph_from_distance(
                                    D_V, k=8, device=device
                                )
                                L_pred_i = uet.compute_graph_laplacian(edge_index, edge_weight, n_valid)
                                
                                # Convert sparse to dense if needed
                                if L_pred_i.layout != torch.strided:
                                    L_pred_i = L_pred_i.to_dense()
                                
                                L_pred_batch[i, :n_valid, :n_valid] = L_pred_i
                                
                                # Target Laplacian
                                L_tgt_i = L_info_batch[i]['L'].to(device).float()
                                if L_tgt_i.layout != torch.strided:
                                    L_tgt_i = L_tgt_i.to_dense()
                                
                                L_targ_batch[i, :n_valid, :n_valid] = L_tgt_i
                            
                            #SINGLE BATCHED LOSS CALL
                            t_list = L_info_batch[0].get('t_list', [0.5, 1.0])
                            L_heat = loss_heat(L_pred_batch, L_targ_batch, mask=mask, t_list=t_list)
                
                # Distance histogram SW (in fp32)
                if (global_step % sw_every_k) == 0:
                    D_target = batch['D_target'].to(device)
                    H_target = batch['H_target'].to(device)
                    H_bins = batch['H_bins'][0].to(device)
                    
                    with torch.autocast(device_type='cuda', enabled=False):
                        V_fp32 = V_hat.float() #(B, N, D)
                        D_all = torch.cdist(V_fp32, V_fp32) 

                        #extract upper triangualr distances per sample
                        n_max = D_all.size(1)
                        iu, ju = torch.triu_indices(n_max, n_max, 1, device=D_all.device)
                        d_batch = D_all[:, iu, ju]

                        #mask for valid pairs
                        mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2) #(B, N, N)
                        valid_pairs = mask_2d[:, iu, ju]

                        #batch histogram computation
                        nb = H_bins.numel() - 1
                        bin_ids = torch.bucketize(d_batch, H_bins) - 1
                        bin_ids = bin_ids.clamp(0, nb-1)
                        bin_ids = bin_ids.masked_fill(~valid_pairs, -1)

                        #vectorized bincount
                        batch_offsets = torch.arange(batch_size_real, device=device).unsqueeze(1) * nb
                        flat_ids = (bin_ids + batch_offsets).view(-1)
                        flat_ids = flat_ids[flat_ids >= 0]

                        counts = torch.bincount(flat_ids, minlength=batch_size_real * nb)
                        counts = counts.view(batch_size_real, nb).float()

                        H_pred = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)

                        L_sw_st = loss_sw(H_pred, H_target)

                
            else:
                # ========================= SC STEP (INSTRUMENTED) =========================
                PROFILE_SC = True
                PROFILE_PRINT_EVERY = 1  # print each SC step; bump to 10 if too chatty
                sc_prof = {}  # timings bucket

                # ===== SC STEP: Score + SW_SC + Ordinal =====

                # ----------------- (1) Ordinal from Z (in fp32) -----------------
                L_ordinal_sc = torch.tensor(0.0, device=device)
                sc_global_indices = batch.get('sc_global_indices', None)
                if sc_global_indices is None:
                    raise ValueError("Batch missing 'sc_global_indices' - update collate_sc_minisets")
                with torch.autocast(device_type='cuda', enabled=False):
                    with timed("ord_total", sc_prof):
                        L_ordinal_sc = torch.tensor(0.0, device=device)
                        
                        sc_global_indices = batch.get('sc_global_indices', None)
                        if sc_global_indices is None:
                            raise ValueError("Batch missing 'sc_global_indices' - update collate_sc_minisets")
                        
                        for i in range(batch_size_real):
                            n_valid = int(n_list[i].item())
                            if n_valid <= 2:
                                continue
                            
                            V_i = V_hat[i, :n_valid].float()
                            set_global_idx = sc_global_indices[i, :n_valid]
                            
                            mask_valid = set_global_idx >= 0
                            set_global_idx = set_global_idx[mask_valid]
                            V_i = V_i[mask_valid]
                            n_valid = set_global_idx.numel()
                            
                            if n_valid <= 2:
                                continue
                            
                            with timed("ord_triplet_sample", sc_prof):
                                triplets_local = uet.build_triplets_from_cache_for_set(
                                    set_global_idx=set_global_idx,
                                    pos_idx_cpu=POS_IDX,
                                    n_per_anchor=10,
                                    triplet_cap=20000
                                )
                            
                            if triplets_local.numel() == 0:
                                continue
                            
                            if triplets_local.size(0) > triplet_cap:
                                with timed("ord_triplet_cap", sc_prof):
                                    sel = torch.randperm(triplets_local.size(0))[:triplet_cap]
                                    triplets_local = triplets_local[sel]
                            
                            triplets_gpu = triplets_local.to(device, non_blocking=True)
                            a_idx, p_idx, n_idx = triplets_gpu.unbind(dim=1)
                            
                            with timed("ord_cdist", sc_prof):
                                Va = V_i[a_idx]
                                Vp = V_i[p_idx]
                                Vn = V_i[n_idx]
                                d_ap = (Va - Vp).pow(2).sum(dim=1)
                                d_an = (Va - Vn).pow(2).sum(dim=1)
                            
                            with timed("ord_loss", sc_prof):
                                margin = 0.2
                                L_i = torch.relu(d_ap - d_an + margin).mean()
                                L_ordinal_sc = L_ordinal_sc + L_i
                        
                        if batch_size_real > 0:
                            L_ordinal_sc = L_ordinal_sc / batch_size_real


                # ----------------- (2) Z-matched distogram (in fp32) -----------------
                L_sw_sc = torch.tensor(0.0, device=device)
                if (global_step % sw_every_k) == 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        with timed("dist_total", sc_prof):
                            with timed("dist_target_match", sc_prof):
                                H_target_sc = match_prototypes_batched(Z_set, mask, prototype_bank)  # (B, nb)

                            with timed("dist_cdist", sc_prof):
                                V_fp32 = V_hat.float()
                                D_all = torch.cdist(V_fp32, V_fp32)

                            with timed("dist_triu", sc_prof):
                                n_max = D_all.size(1)
                                iu, ju = torch.triu_indices(n_max, n_max, 1, device=D_all.device)
                                d_batch = D_all[:, iu, ju]
                                mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                                valid_pairs = mask_2d[:, iu, ju]

                            with timed("dist_bucketize", sc_prof):
                                proto_bins = prototype_bank['bins']
                                bin_ids = torch.bucketize(d_batch, proto_bins) - 1
                                bin_ids = bin_ids.clamp(0, proto_bins.numel() - 2)
                                bin_ids = bin_ids.masked_fill(~valid_pairs, -1)

                            with timed("dist_bincount", sc_prof):
                                nb = proto_bins.numel() - 1
                                batch_offsets = torch.arange(batch_size_real, device=device).unsqueeze(1) * nb
                                flat_ids = (bin_ids + batch_offsets).view(-1)
                                flat_ids = flat_ids[flat_ids >= 0]
                                counts = torch.bincount(flat_ids, minlength=batch_size_real * nb)
                                counts = counts.view(batch_size_real, nb).float()

                            with timed("dist_norm", sc_prof):
                                H_pred_sc = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)

                            with timed("dist_loss", sc_prof):
                                L_sw_sc = loss_sw(H_pred_sc, H_target_sc)

                # ----------------- (3) Distance-only overlap (SC ONLY, every K steps) -----------------
                L_overlap = torch.tensor(0.0, device=device)
                if (global_step % EVERY_K_STEPS) == 0:
                    need = ("pair_idxA","pair_idxB","shared_A_idx","shared_B_idx")
                    has_all = all(k in batch for k in need)

                    if has_all and batch["pair_idxA"].numel() > 0:
                        with torch.autocast(device_type='cuda', enabled=False):
                            P = batch['pair_idxA'].size(0)

                            # Gather per-pair predicted coordinates
                            idxA, idxB = batch['pair_idxA'], batch['pair_idxB']     # (P,)
                            VA, VB = V_hat[idxA].float(), V_hat[idxB].float()       # (P, N, D)

                            SA, SB = batch['shared_A_idx'], batch['shared_B_idx']   # (P, Kmax)
                            maskA, maskB = SA.ge(0), SB.ge(0)                       # True where valid

                            SAx = SA.clamp_min(0).unsqueeze(-1).expand(-1, -1, VA.size(-1))
                            SBx = SB.clamp_min(0).unsqueeze(-1).expand(-1, -1, VB.size(-1))
                            A = torch.gather(VA, 1, SAx)    # (P, Kmax, D)
                            B = torch.gather(VB, 1, SBx)    # (P, Kmax, D)

                            # Pairwise validity mask built from masks (no ordering assumptions)
                            pairmask = (maskA.unsqueeze(2) & maskA.unsqueeze(1) &
                                        maskB.unsqueeze(2) & maskB.unsqueeze(1))    # (P, Kmax, Kmax)

                            # Remove diagonal
                            Kmax = SA.size(1)
                            eye = torch.eye(Kmax, dtype=torch.bool, device=device).unsqueeze(0)
                            pairmask = pairmask & (~eye)

                            if pairmask.any():
                                DA = torch.cdist(A, A)  # (P, Kmax, Kmax)
                                DB = torch.cdist(B, B)

                                iu, ju = torch.triu_indices(Kmax, Kmax, 1, device=device)
                                tri_mask = pairmask[:, iu, ju]          # (P, M)
                                dA = DA[:, iu, ju][tri_mask]            # (total_valid,)
                                dB = DB[:, iu, ju][tri_mask]

                                L_overlap = torch.mean((dA - dB) ** 2)
                            else:
                                # nothing valid this step; leave as 0.0 or skip logging
                                L_overlap = torch.tensor(0.0, device=device)


                # ----------------- (4) Compact timing print -----------------
                # if PROFILE_SC and (global_step % PROFILE_PRINT_EVERY == 0):
                #     g = lambda k: sc_prof.get(k, 0.0)
                #     print(
                #         f"[SC-PROFILE] step={global_step} "
                #         f"| ord: total={g('ord_total'):.4f}s samp={g('ord_triplet_sample'):.4f} "
                #         f"cap={g('ord_triplet_cap'):.4f} cdist={g('ord_cdist'):.4f} loss={g('ord_loss'):.4f} "
                #         f"| dist: total={g('dist_total'):.4f}s tgt={g('dist_target_match'):.4f} "
                #         f"cdist={g('dist_cdist'):.4f} triu={g('dist_triu'):.4f} bucket={g('dist_bucketize'):.4f} "
                #         f"bincount={g('dist_bincount'):.4f} norm={g('dist_norm'):.4f} loss={g('dist_loss'):.4f} "
                #         f"| ov: total={g('ov_total'):.4f}s gather={g('ov_gather'):.4f} "
                #         f"cdist={g('ov_cdist'):.4f} tri={g('ov_tri_mask'):.4f} mse={g('ov_mse'):.4f}"
                #     )
                    sc_prof.clear()
# ======================= END SC STEP (INSTRUMENTED) =======================

                # ===== SC STEP: Score + SW_SC + Ordinal =====
                
                # Ordinal from Z (in fp32)
                # with torch.autocast(device_type='cuda', enabled=False):
                #     for i in range(batch_size_real):
                #         n_valid = int(n_list[i].item())
                #         Z_i = Z_set[i, :n_valid].float()
                #         V_i = V_hat[i, :n_valid].float()
                        
                #         triplets = uet.sample_ordinal_triplets_from_Z(Z_i, n_per_anchor=10, k_nn=25)
                #         if len(triplets) > triplet_cap:
                #             sel = torch.randperm(len(triplets))[:triplet_cap]
                #             triplets = triplets[sel]
                #         if len(triplets) > 0:
                #             D_V = torch.cdist(V_i, V_i)
                #             L_ordinal_sc += loss_triplet(D_V, triplets)
                    
                #     L_ordinal_sc = L_ordinal_sc / batch_size_real
                

                # # Z-matched distogram (in fp32)
                # # Z-matched distogram (in fp32)
                # if (global_step % sw_every_k) == 0:
                #     with torch.autocast(device_type='cuda', enabled=False):
                #         # Use the dedicated batched function
                #         H_target_sc = match_prototypes_batched(Z_set, mask, prototype_bank)  # (B, nb)
                        
                #         # Compute predicted histograms
                #         V_fp32 = V_hat.float()
                #         D_all = torch.cdist(V_fp32, V_fp32)
                        
                #         # Extract upper triangular distances
                #         n_max = D_all.size(1)
                #         iu, ju = torch.triu_indices(n_max, n_max, 1, device=D_all.device)
                #         d_batch = D_all[:, iu, ju]
                        
                #         # Mask for valid pairs
                #         mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                #         valid_pairs = mask_2d[:, iu, ju]
                        
                #         # Use proto_bins from bank
                #         proto_bins = prototype_bank['bins']
                #         bin_ids = torch.bucketize(d_batch, proto_bins) - 1
                #         bin_ids = bin_ids.clamp(0, proto_bins.numel() - 2)
                #         bin_ids = bin_ids.masked_fill(~valid_pairs, -1)
                        
                #         # Vectorized bincount
                #         nb = proto_bins.numel() - 1
                #         batch_offsets = torch.arange(batch_size_real, device=device).unsqueeze(1) * nb
                #         flat_ids = (bin_ids + batch_offsets).view(-1)
                #         flat_ids = flat_ids[flat_ids >= 0]
                        
                #         counts = torch.bincount(flat_ids, minlength=batch_size_real * nb)
                #         counts = counts.view(batch_size_real, nb).float()
                #         H_pred_sc = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
                        
                #         L_sw_sc = loss_sw(H_pred_sc, H_target_sc)

                # # if (global_step % sw_every_k) == 0:
                # #     with torch.autocast(device_type='cuda', enabled=False):
                # #         # FIX: Correct centroid computation
                # #         Z_centroids = (Z_set * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, h_dim)

                # #         # Batched cosine similarity to all prototypes
                # #         proto_centroids = prototype_bank['centroids']  # (P, h_dim) on GPU
                # #         proto_hists = prototype_bank['hists']  # (P, nb) on GPU
                # #         proto_bins = prototype_bank['bins']  # (nb+1,) on GPU

                # #         # Normalized cosine similarity
                # #         Z_norm = F.normalize(Z_centroids, dim=1)
                # #         P_norm = F.normalize(proto_centroids, dim=1)
                # #         similarity = Z_norm @ P_norm.T  # (B, P)

                # #         best_proto_idx = similarity.argmax(dim=1)  # (B,)
                # #         H_target_sc = proto_hists[best_proto_idx]  # (B, nb)

                # #         # Compute distances
                # #         V_fp32 = V_hat.float()  # (B, N, D)
                # #         D_all = torch.cdist(V_fp32, V_fp32)  # (B, N, N)

                # #         # Extract upper triangular distances
                # #         n_max = D_all.size(1)
                # #         iu, ju = torch.triu_indices(n_max, n_max, 1, device=D_all.device)
                # #         d_batch = D_all[:, iu, ju]  # (B, M)
                        
                # #         # Mask for valid pairs
                # #         mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                # #         valid_pairs = mask_2d[:, iu, ju]  # (B, M)

                # #         # Use fixed proto_bins for all samples (fast)
                # #         bin_ids = torch.bucketize(d_batch, proto_bins) - 1  # (B, M)
                # #         bin_ids = bin_ids.clamp(0, proto_bins.numel() - 2)
                # #         bin_ids = bin_ids.masked_fill(~valid_pairs, -1)

                # #         # Vectorized bincount
                # #         nb = proto_bins.numel() - 1
                # #         batch_offsets = torch.arange(batch_size_real, device=device).unsqueeze(1) * nb
                # #         flat_ids = (bin_ids + batch_offsets).view(-1)
                # #         flat_ids = flat_ids[flat_ids >= 0]

                # #         counts = torch.bincount(flat_ids, minlength=batch_size_real * nb)
                # #         counts = counts.view(batch_size_real, nb).float()
                # #         H_pred_sc = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)

                # #         L_sw_sc = loss_sw(H_pred_sc, H_target_sc)

                
                # # Distance-only overlap (SC ONLY, every 8 steps) - OPTIMIZED
                # if (global_step % EVERY_K_STEPS) == 0:
                #     shared_info = batch.get('shared_info', [])
                    
                #     if len(shared_info) > 0:
                #         pair_losses = []
                #         with torch.autocast(device_type='cuda', enabled=False):                            
                #             if 'pair_idxA' in batch and batch['pair_idxA'].numel() > 0:
                #                 P = batch['pair_idxA'].size(0)
                #                 Kmax = batch['shared_A_idx'].size(1)
                                
                #                 # Gather V for all pairs at once
                #                 idxA = batch['pair_idxA']  # (P,)
                #                 idxB = batch['pair_idxB']  # (P,)
                #                 VA = V_hat[idxA]  # (P, N, D)
                #                 VB = V_hat[idxB]  # (P, N, D)
                                
                #                 # Gather shared points
                #                 SA = batch['shared_A_idx']  # (P, Kmax)
                #                 SB = batch['shared_B_idx']  # (P, Kmax)
                                
                #                 maskA = SA >= 0
                #                 maskB = SB >= 0
                                
                #                 # Gather rows (handle -1 padding)
                #                 SA_safe = SA.clamp_min(0).unsqueeze(-1).expand(-1, -1, VA.size(-1))
                #                 SB_safe = SB.clamp_min(0).unsqueeze(-1).expand(-1, -1, VB.size(-1))
                                
                #                 rowA = torch.gather(VA, 1, SA_safe) * maskA.unsqueeze(-1).float()  # (P, Kmax, D)
                #                 rowB = torch.gather(VB, 1, SB_safe) * maskB.unsqueeze(-1).float()  # (P, Kmax, D)
                                
                #                 # Batched cdist
                #                 DA = torch.cdist(rowA.float(), rowA.float())  # (P, Kmax, Kmax)
                #                 DB = torch.cdist(rowB.float(), rowB.float())  # (P, Kmax, Kmax)
                                
                #                 # Upper triangle mask per pair
                #                 iu, ju = torch.triu_indices(Kmax, Kmax, 1, device=DA.device)
                #                 dA_flat = DA[:, iu, ju]  # (P, M)
                #                 dB_flat = DB[:, iu, ju]  # (P, M)
                                
                #                 # Valid mask based on shared_len
                #                 shared_lens = batch['shared_len']  # (P,)
                #                 valid_mask = (iu.unsqueeze(0) < shared_lens.unsqueeze(1)) & \
                #                             (ju.unsqueeze(0) < shared_lens.unsqueeze(1))  # (P, M)
                                
                #                 dA_valid = dA_flat[valid_mask]
                #                 dB_valid = dB_flat[valid_mask]
                                
                #                 L_overlap = ((dA_valid - dB_valid) ** 2).mean()
                #             else:
                #                 L_overlap = torch.tensor(0.0, device=device)

            
            # Total loss
            L_total = (WEIGHTS['score'] * L_score +
                      WEIGHTS['gram'] * L_gram +
                      WEIGHTS['heat'] * L_heat +
                      WEIGHTS['sw_st'] * L_sw_st +
                      WEIGHTS['sw_sc'] * L_sw_sc +
                      WEIGHTS['overlap'] * L_overlap +
                      WEIGHTS['ordinal_sc'] * L_ordinal_sc)
            
            # Backward with gradient scaling
            optimizer.zero_grad(set_to_none=True)

            if fabric is not None:
                # When using Fabric, it handles backward and gradient scaling
                fabric.backward(L_total)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            else:
                # When not using Fabric, use manual GradScaler
                scaler.scale(L_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            # Log
            epoch_losses['total'] += L_total.item()
            epoch_losses['score'] += L_score.item()
            epoch_losses['gram'] += L_gram.item()
            epoch_losses['heat'] += L_heat.item()
            epoch_losses['sw_st'] += L_sw_st.item()
            epoch_losses['sw_sc'] += L_sw_sc.item()
            epoch_losses['overlap'] += L_overlap.item()
            epoch_losses['ordinal_sc'] += L_ordinal_sc.item()
            
            n_batches += 1
            global_step += 1
            # batch_pbar.update(1)
        
            # (optional) metrics logging
            if fabric is None or fabric.is_global_zero:
                pass  # print / tqdm here if you want
        
        scheduler.step()

        # Epoch averages
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            history['epoch_avg'][k].append(epoch_losses[k])
        
        history['epoch'].append(epoch)
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_losses['total']:.4f} | "
                  f"Score: {epoch_losses['score']:.4f} | Gram: {epoch_losses['gram']:.4f} | "
                  f"SW_ST: {epoch_losses['sw_st']:.4f} | Ord_SC: {epoch_losses['ordinal_sc']:.4f}")
        
        # Save checkpoint
        # if (epoch + 1) % 100 == 0:
        if fabric is not None:
            fabric.barrier()

        # --- save checkpoints only on rank-0 ---
        if (epoch + 1) % 50 == 0:
            if fabric is None or fabric.is_global_zero:
                ckpt = {
                    'epoch': epoch,
                    'context_encoder': context_encoder.state_dict(),
                    'score_net': score_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history
                }
                torch.save(ckpt, os.path.join(outf, f'ckpt_epoch_{epoch+1}.pt'))
    
    if fabric is None or fabric.is_global_zero:
        print("Training complete!")

# ==============================================================================
# STAGE D: SC INFERENCE
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import gc


def sample_sc_edm_anchored(
    sc_gene_expr: torch.Tensor,
    encoder: 'SharedEncoder',
    context_encoder: SetEncoderContext,
    score_net: DiffusionScoreNet,
    n_timesteps_sample: int = 160,
    sigma_min: float = 0.01,
    sigma_max: float = 5.0,
    return_coords: bool = True,
    anchor_size: int = 384,
    batch_size: int = 512,
    eta: float = 0.0,
    device: str = 'cuda',
    guidance_scale = 8.0
) -> Dict[str, torch.Tensor]:
    """
    ANCHOR-CONDITIONED batched inference for SC data.
    """
    print("\n" + "="*70)
    print("ANCHOR-CONDITIONED SC INFERENCE")
    print("="*70)
    
    encoder.eval()
    context_encoder.eval()
    score_net.eval()
    
    n_sc = sc_gene_expr.shape[0]
    D_latent = score_net.D_latent
    
    print(f"Total cells: {n_sc}")
    print(f"Anchor cells: {anchor_size}")
    print(f"Batch size: {batch_size}")
    print(f"Timesteps: {n_timesteps_sample}")
    
    # Encode all SC cells
    print("\nStep 1: Encoding SC cells and selecting anchors...")
    
    with torch.no_grad():
        Z_all = []
        encode_bs = 1024
        for i in range(0, n_sc, encode_bs):
            z = encoder(sc_gene_expr[i:i+encode_bs].to(device)).cpu()
            Z_all.append(z)
        Z_all = torch.cat(Z_all, dim=0)
    
    # FPS for anchors
    import utils_et as uet
    anchor_size = min(anchor_size, n_sc)
    Z_unit = Z_all / (Z_all.norm(dim=1, keepdim=True) + 1e-8)
    anchor_idx = uet.farthest_point_sampling(Z_unit, anchor_size, device='cpu')
    non_anchor_idx = torch.tensor(
        [i for i in range(n_sc) if i not in set(anchor_idx.tolist())],
        dtype=torch.long
    )
    
    print(f"Selected {len(anchor_idx)} anchors")
    print(f"Remaining cells: {len(non_anchor_idx)}")
    
    # Sample anchors once
    print("\nStep 2: Sampling anchor positions...")
    
    Z_A = Z_all[anchor_idx].to(device).unsqueeze(0)
    mask_A = torch.ones(1, Z_A.shape[1], dtype=torch.bool, device=device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        H_A = context_encoder(Z_A, mask_A)
    
    # Sigma schedule (decreasing: sigma_max → sigma_min)
    sigmas = torch.exp(torch.linspace(
        np.log(sigma_max), np.log(sigma_min), 
        n_timesteps_sample, device=device
    ))
    
    # Initialize with high noise
    V_t = torch.randn(1, Z_A.shape[1], D_latent, device=device) * sigmas[0]
    
    # Reverse diffusion (FORWARD iteration: high noise → low noise)
    with torch.no_grad(), torch.cuda.amp.autocast():
        for t_idx in range(n_timesteps_sample):
            sigma_t = sigmas[t_idx]
            t_norm = torch.tensor([[t_idx / (n_timesteps_sample - 1)]], device=device)
            
            # CFG
            H_null = torch.zeros_like(H_A)
            eps_uncond = score_net(V_t, t_norm, H_null, mask_A)
            eps_cond = score_net(V_t, t_norm, H_A, mask_A)
            
            guidance_scale = 10.0
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # Log first few steps
            if t_idx < 5:
                diff_norm = (eps_cond - eps_uncond).norm().item()
                uncond_norm = eps_uncond.norm().item()
                ratio = diff_norm / (uncond_norm + 1e-6)
                print(f"  Step {t_idx}, sigma={sigma_t:.4f}: guidance ratio = {ratio:.4f}")
            
            # Update
            if t_idx < n_timesteps_sample - 1:
                sigma_next = sigmas[t_idx + 1]
                
                # Tweedie: predict V_0
                V_0_pred = V_t - sigma_t * eps
                
                # Step toward V_0_pred
                V_t = V_0_pred + sigma_next * (V_t - V_0_pred) / sigma_t
                
                # Stochasticity
                if eta > 0:
                    noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - (sigma_next**2 / sigma_t**2) * sigma_t**2, min=0))
                    V_t = V_t + noise_scale * torch.randn_like(V_t)
            else:
                # Final step
                V_t = V_t - sigma_t * eps
    
    V_A = V_t.squeeze(0).detach()
    print(f"Anchor coordinates sampled: {V_A.shape}")
    
    # Batch remaining cells with frozen anchors
    print("\nStep 3: Processing remaining cells with frozen anchors...")
    
    all_V = [V_A.cpu()]
    order = [anchor_idx.tolist()]
    
    n_batches = (non_anchor_idx.numel() + batch_size - 1) // batch_size
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx in tqdm(range(n_batches), desc="Batching"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, non_anchor_idx.numel())
            idx_B = non_anchor_idx[start_idx:end_idx]
            
            Z_B = Z_all[idx_B].to(device).unsqueeze(0)
            Z_C = torch.cat([Z_A, Z_B], dim=1)
            mask_C = torch.ones(1, Z_C.shape[1], dtype=torch.bool, device=device)
            
            H_C = context_encoder(Z_C, mask_C)
            
            # Initialize: anchors fixed, new cells noisy
            V_t = torch.empty(1, Z_C.shape[1], D_latent, device=device)
            V_t[:, :V_A.shape[0], :] = V_A.to(device)
            V_t[:, V_A.shape[0]:, :] = torch.randn(
                1, Z_C.shape[1] - V_A.shape[0], D_latent, device=device
            ) * sigmas[0]
            
            # Freeze mask (0 = frozen anchors, 1 = update new cells)
            upd_mask = torch.ones_like(V_t)
            upd_mask[:, :V_A.shape[0], :] = 0.0
            
            # Reverse diffusion
            for t_idx in range(n_timesteps_sample):
                sigma_t = sigmas[t_idx]
                t_norm = torch.tensor([[t_idx / (n_timesteps_sample - 1)]], device=device)
                
                # CFG
                H_null_C = torch.zeros_like(H_C)
                eps_uncond = score_net(V_t, t_norm, H_null_C, mask_C)
                eps_cond = score_net(V_t, t_norm, H_C, mask_C)
                
                # guidance_scale = 10.0
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                
                # Update
                if t_idx < n_timesteps_sample - 1:
                    sigma_next = sigmas[t_idx + 1]
                    V_0_pred = V_t - sigma_t * eps
                    V_new = V_0_pred + sigma_next * (V_t - V_0_pred) / sigma_t
                    
                    if eta > 0:
                        noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - (sigma_next**2 / sigma_t**2) * sigma_t**2, min=0))
                        V_new = V_new + noise_scale * torch.randn_like(V_new)
                else:
                    V_new = V_t - sigma_t * eps
                
                # Apply freeze mask
                V_t = upd_mask * V_new + (1.0 - upd_mask) * V_t
            
            # Extract new cells only
            V_B_only = V_t.squeeze(0)[V_A.shape[0]:, :].detach().cpu()
            
            all_V.append(V_B_only)
            order.append(idx_B.tolist())
            
            del Z_B, Z_C, H_C, V_t, eps_uncond, eps_cond, eps
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Reassemble in original order
    print("\nStep 4: Reassembling coordinates...")
    
    V_0_full = torch.empty(n_sc, D_latent)
    V_0_full[anchor_idx] = all_V[0]
    for idxs, Vs in zip(order[1:], all_V[1:]):
        V_0_full[torch.tensor(idxs, dtype=torch.long)] = Vs
    
    # ONE GLOBAL CENTERING
    V_0_full = V_0_full - V_0_full.mean(dim=0, keepdim=True)
    
    print(f"Final latent shape: {V_0_full.shape}")
    
    # ONE GLOBAL EDM
    print("\nStep 5: Computing global EDM...")
    
    G = V_0_full @ V_0_full.t()
    diag = torch.diag(G).unsqueeze(1)
    D = torch.sqrt(torch.clamp(diag + diag.t() - 2 * G, min=0))
    D_edm = uet.edm_project(D)
    
    result = {'D_edm': D_edm.cpu()}
    
    if return_coords:
        print("Computing coordinates via MDS...")
        n = D_edm.shape[0]
        J = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
        B = -0.5 * J @ (D_edm.to(device) ** 2) @ J
        
        coords = uet.classical_mds(B, d_out=2)
        coords_canon = uet.canonicalize_coords(coords)
        
        result['coords'] = coords.cpu()
        result['coords_canon'] = coords_canon.cpu()
        
        del J, B, coords, coords_canon
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print("SC inference complete!")
    return result
    