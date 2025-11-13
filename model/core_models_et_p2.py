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
import torch.distributed as dist


from tqdm import tqdm
import gc


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

#debug tools
DEBUG = True #master switch for debug logging
LOG_EVERY = 20 #per-batch prints
SAVE_EVERY_EPOCH = 5 #extra plots/dumps

#debug tracking variables 
debug_state = {
    'score_bins': None,
    'score_bin_sum': None,
    'score_bin_cnt': None,
    'dbg_overlap_seen': 0,
    'dbg_overlap_pairs': 0,
    'dbg_k_mean': 0.0,
    'overlap_count_this_epoch': 0,
    'last_gram_trace_ratio': 85.0 
}

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

def _intersect1d_with_indices(a: torch.Tensor, b: torch.Tensor):
    """
    PyTorch-only replacement for torch.intersect1d(..., return_indices=True).
    Returns (common, idx_a, idx_b) where:
      common = a[idx_a] = b[idx_b]
    Keeps the order of 'a' (stable in a).
    """
    # a, b: 1D Long tensors on same device
    a = a.view(-1)
    b = b.view(-1)
    # boolean match matrix (sizes here are small: shared ~ O(10^2))
    M = (a[:, None] == b[None, :])
    idx_a, idx_b = torch.nonzero(M, as_tuple=True)
    if idx_a.numel() == 0:
        empty = a.new_empty((0,), dtype=a.dtype)
        return empty, idx_a, idx_b
    # keep a-order stable
    order = torch.argsort(idx_a)
    idx_a = idx_a[order]
    idx_b = idx_b[order]
    common = a[idx_a]
    return common, idx_a, idx_b


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
import math

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
        ln: bool = True,
        #new params
        use_canonicalize: bool = True,
        use_dist_bias: bool = True,
        dist_bins: int = 16,
        dist_head_shared: bool = True,
        use_angle_features: bool = True,
        angle_bins: int = 8,
        knn_k: int = 12,
        self_conditioning: bool = True,
        sc_feat_mode: str ='concat'
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

        # Store new config
        self.use_canonicalize = use_canonicalize
        self.use_dist_bias = use_dist_bias
        self.dist_bins = dist_bins
        self.dist_head_shared = dist_head_shared
        self.use_angle_features = use_angle_features
        self.angle_bins = angle_bins
        self.knn_k = knn_k
        self.self_conditioning = self_conditioning
        self.sc_feat_mode = sc_feat_mode

        #distance bias params
        if use_dist_bias:
            d_emb = 32
            self.E_bin = nn.Parameter(torch.randn(dist_bins, d_emb) / math.sqrt(d_emb))
            out_dim = 1 if dist_head_shared else n_heads
            self.W_bias = nn.Parameter(torch.randn(d_emb, out_dim) * 0.01)
            self.alpha_bias = nn.Parameter(torch.tensor(0.1))

        #self conditioning MLP if needed
        if self.self_conditioning and sc_feat_mode == 'mlp':
            self.sc_mlp = nn.Sequential(
                nn.Linear(D_latent, c_dim // 2),
                nn.ReLU(),
                nn.Linear(c_dim // 2, D_latent)
            )

        #update inpiut projection dimension
        extra_dims = 0

        if self_conditioning:
            extra_dims += D_latent
        if use_angle_features:
            extra_dims += angle_bins

        
        # Time embedding (Fourier features)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, c_dim)
        )
        
        # Input projection: V_t + H → combined features
        # self.input_proj = nn.Linear(D_latent + c_dim, c_dim)
        # self.input_proj = nn.Linear(c_dim + 128 + extra_dims, c_dim)  # 128 is time_emb dim
        self.input_proj = nn.Linear(c_dim + c_dim + extra_dims + D_latent, c_dim) 
        self.bias_sab = SAB(c_dim, c_dim, n_heads, ln=ln)

        
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

    def forward(self, V_t: torch.Tensor, t: torch.Tensor, H: torch.Tensor, 
           mask: torch.Tensor, self_cond: torch.Tensor = None, 
           attn_cached: dict = None) -> torch.Tensor:
        """
        Args:
            V_t: (B, N, D_latent) noisy coords at time t
            t: (B,) or (B,1) diffusion time
            H: (B, N, c_dim) context from SetEncoderContext
            mask: (B, N)
            self_cond: optional (B, N, D_latent) predicted V_0 from previous step
            attn_cached: optional dict to reuse distance bias
        
        Returns:
            eps_hat: (B, N, D_latent)
        """
        B, N, D = V_t.shape
        
        # Step 1: Canonicalization
        if self.use_canonicalize:
            V_in, _, _ = uet.canonicalize(V_t, mask)
            if self_cond is not None:
                self_cond_canon, _, _ = uet.canonicalize(self_cond, mask)
            else:
                self_cond_canon = None
        else:
            V_in = V_t
            self_cond_canon = self_cond
        
        # Step 2: Compose node features
        features = [H]  # Start with context
        
        # Add time embedding (expanded to all nodes)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.get_time_embedding(t)  #First get Fourier features
        t_emb = self.time_mlp(t_emb)  # then pass through MLP
        # t_emb_expanded = t_emb.expand(-1, N, -1)  # (B, N, c_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1) 
        features.append(t_emb_expanded)
        
        # Add self-conditioning
        if self.self_conditioning and self_cond_canon is not None:
            if self.sc_feat_mode == "mlp":
                sc_feat = self.sc_mlp(self_cond_canon)
            else:  # concat
                sc_feat = self_cond_canon
            features.append(sc_feat)
        elif self.self_conditioning:  # No self_cond provided, use zeros
            sc_feat = torch.zeros(B, N, self.D_latent, device=V_t.device)
            features.append(sc_feat)
        
        # Add angle features
        if self.use_angle_features:
            idx = uet.knn_graph(V_in, mask, k=self.knn_k)
            angle_feat = uet.angle_features(V_in, mask, idx, n_angle_bins=self.angle_bins)
            features.append(angle_feat)
        
        features.append(V_in)

        # Concatenate all features
        X = torch.cat(features, dim=-1)  # (B, N, c_dim + 128 + extras)
        X = self.input_proj(X)  # (B, N, c_dim)
        
        # Step 3: Distance-bucket attention bias
        attn_bias = None
        if self.use_dist_bias:
            if attn_cached is not None and 'bias' in attn_cached:
                attn_bias = attn_cached['bias']
            else:
                D2 = uet.pairwise_dist2(V_in, mask)
                attn_bias, _ = uet.make_distance_bias(
                    D2, mask, 
                    n_bins=self.dist_bins,
                    d_emb=32,
                    share_across_heads=self.dist_head_shared,
                    E_bin=self.E_bin,
                    W=self.W_bias,
                    alpha_bias=self.alpha_bias,
                    device=V_t.device
                )
                if attn_cached is not None:
                    attn_cached['bias'] = attn_bias

        # Apply X↔X distance-biased attention exactly once
        if self.use_dist_bias and attn_bias is not None:
            X = self.bias_sab(X, mask=mask, attn_bias=attn_bias)
        else:
            # Even without bias, keep a homogeneous pass for depth
            X = self.bias_sab(X, mask=mask, attn_bias=None)
        X = X * mask.unsqueeze(-1).float()
        
        # Step 4: Apply ISAB blocks with attention bias
        for isab in self.denoise_blocks:
            X = isab(X, mask=mask, attn_bias=None)
            X = X * mask.unsqueeze(-1).float()
        
        # Step 5: Output head
        eps_hat = self.output_head(X)  # (B, N, D_latent)
        eps_hat = eps_hat * mask.unsqueeze(-1).float()
        
        return eps_hat
    
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
    sigma_max: float = 53.0,
    device: str = 'cuda',
    outf: str = 'output',
    fabric: Optional['Fabric'] = None,
    precision: str = '16-mixed', # "32-true" | "16-mixed" | "bf16-mixed"
    logger = None,
    log_interval: int = 20
):
    
    # Initialize debug tracking
    global debug_state
    debug_state = {
        'score_bins': None,
        'score_bin_sum': None,
        'score_bin_cnt': None,
        'dbg_overlap_seen': 0,
        'dbg_overlap_pairs': 0,
        'dbg_k_mean': 0.0,
        'overlap_count_this_epoch': 0
    }

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
    # scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if fabric is None else None
    use_fp16 = (precision == '16-mixed')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    
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

    # DEBUG: First-time sigma info
    if DEBUG:
       print(f"[sigma_schedule] min={sigmas.min().item():.4f} max={sigmas.max().item():.4f} "
             f"n_steps={n_timesteps}")
    
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


    # ============================================================================
    # DEBUG: MINISETS SANITY CHECK
    # ============================================================================
    from utils_debug_minisets import sample_dataloader_and_report
    import os

    def _is_rank0():
        try:
            return (not dist.is_initialized()) or (dist.get_rank() == 0)
        except Exception:
            return True

    if _is_rank0():
        print("\n" + "="*80)
        print("MINISETS DRY CHECK (BEFORE TRAINING)")
        print("="*80 + "\n")
        sample_dataloader_and_report(
            st_loader=st_loader,
            sc_loader=sc_loader,
            batches=3,
            device=device,
            is_global_zero=_is_rank0(),
            save_json_path=os.path.join(outf, "minisets_check.json")
        )
        print("\n" + "="*80)
        print("MINISETS CHECK COMPLETE - CHECK OUTPUT ABOVE")
        print("="*80 + "\n")
    # ============================================================================

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

    WEIGHTS = {
        'score': 0.5,
        'gram': 8.0,
        'gram_scale': 2.0,
        'heat': 1.0,
        'sw_st': 1.0,
        'sw_sc': 0.6,
        'overlap': 0.25,
        'ordinal_sc': 0.5,
    }
    
    # Overlap config - OPTIMIZED
    EVERY_K_STEPS = 1   # CHANGED from 2
    MAX_OVERLAP_POINTS = 128  # CHANGED from 64
    MIN_OVERLAP_ABS = 5

    # NEW: compute-heavy loss cadence + subsampling
    heat_every_k = 1          # compute heat loss every 4th step only
    sw_every_k = 1            # sliced-W every 2nd step
    gram_pair_cap = 8000      # sample up to 8k pairs for Gram (mask upper-tri)
    triplet_cap = 10000       # cap triplets per batch
    hist_bins = 48            # fewer bins → faster histogram distances

    SW_SC_KPROJ = 64     # number of random projections
    SW_SC_NCAP  = 512    # cap points per set for SW speed

    
    # CFG config
    p_uncond = 0.15
    
    history = {
        'epoch': [],
        'batch_losses': [],
        'epoch_avg': {
            'total': [], 'score': [], 'gram': [], 'gram_scale': [], 'heat': [],
            'sw_st': [], 'sw_sc': [], 'overlap': [], 'ordinal_sc': []
        }
    }
    
    # Compute sigma_data once at start
    print("Computing sigma_data from data statistics...")

    def sync_scalar(value: float, device: str) -> float:
        t = torch.tensor([value], device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.broadcast(t, src=0)
        return float(t.item())

    with torch.no_grad():
        if fabric is None or fabric.is_global_zero:
            sample_stds = []
            it = iter(st_loader)  # Create iterator ONCE
            for _ in range(min(10, len(st_loader))):
                sample_batch = next(it, None)
                if sample_batch is None:
                    break
                G_batch = sample_batch['G_target'].to(device, non_blocking=True)
                for i in range(min(4, G_batch.shape[0])):
                    V_temp = uet.factor_from_gram(G_batch[i], score_net.D_latent)
                    sample_stds.append(V_temp.std().item())
            sigma_data = float(np.median(sample_stds)) if sample_stds else 1.0
        else:
            sigma_data = 0.0

    sigma_data = sync_scalar(sigma_data, device)
    print(f"[synced] sigma_data = {sigma_data:.4f}")

    #--amp scaler choice based on precision---
    use_fp16 = (precision == '16-mixed')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    global_step = 0

    # EMA-based autobalance (stable)
    ema_grads = {
        'score': 1.0,
        'gram': 0.1,
        'heat': 0.1,
        'sw_st': 0.1
    }
    ema_sc = {
        'score': 1.0,
        'sw_sc': 0.1,
        'overlap': 0.1,
        'ordinal_sc': 0.1
    }
    EMA_BETA = 0.98  # CHANGED from 0.95
    
    TARGET = {'gram': 0.05, 'heat': 0.30, 'sw_st': 0.10}
    TARGET_SC = {'sw_sc': 0.08, 'overlap': 0.02, 'ordinal_sc': 0.06}
    AUTOBALANCE_EVERY = 100
    AUTOBALANCE_START = 200

    #self conditioning probability
    p_sc = 0.5 #prob of using self-conditioning in training
    
    for epoch in range(n_epochs):
        st_iter = iter(st_loader)
        sc_iter = iter(sc_loader)
        
        epoch_losses = {k: 0.0 for k in WEIGHTS.keys()}
        epoch_losses['total'] = 0.0
        n_batches = 0
        c_overlap = 0

        st_batches = 0
        sc_batches = 0
        
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

            if not is_sc:
                st_batches += 1
            else:
                sc_batches += 1
                
            Z_set = batch['Z_set'].to(device)
            mask = batch['mask'].to(device)
            n_list = batch['n']
            batch_size_real = Z_set.shape[0]
            
            D_latent = score_net.D_latent

            # if DEBUG and epoch == 0:
            #     if fabric is None or fabric.is_global_zero:
            #         print("\n[WEIGHTS]", {k: float(v) for k, v in WEIGHTS.items()})
            #         print()
            
            # ===== FORWARD PASS WITH AMP =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                # Context encoding
                H = context_encoder(Z_set, mask)
                
                # Sample noise level with quadratic bias toward low noise
                u = torch.rand(batch_size_real, device=device)
                t_cont = (u ** 2) * (n_timesteps - 1)
                t_idx = t_cont.long()
                t_norm = t_cont / (n_timesteps - 1)
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

                # CFG: Drop context randomly during training
                drop_mask = (torch.rand(batch_size_real, device=device) < p_uncond).float().view(-1, 1, 1)
                H_train = H * (1 - drop_mask)

                # DEBUG: CFG stats (first epoch only)
                if DEBUG and epoch == 0 and global_step < 10:
                   n_dropped = drop_mask.sum().item()
                   print(f"[CFG] step={global_step} dropped={int(n_dropped)}/{batch_size_real} contexts")

                # Self-conditioning logic
                use_self_cond = torch.rand(1, device=device).item() < p_sc

                if use_self_cond and score_net.self_conditioning:
                    # First pass without self-conditioning to get V_0 prediction
                    with torch.no_grad():
                        eps_hat_0 = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None)
                        V_pred_0 = V_t - sigma_t * eps_hat_0
                    
                    # Second pass with self-conditioning
                    eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=V_pred_0)
                else:
                    # Single pass without self-conditioning
                    eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None)

                
                # # Predict noise
                # eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask)
            
            # ===== SCORE LOSS (in fp32 for numerical stability) =====
            with torch.autocast(device_type='cuda', enabled=False):
                sigma_t_fp32 = sigma_t.float()
                eps_pred_fp32 = eps_pred.float()
                eps_fp32 = eps.float()
                mask_fp32 = mask.float()
                

                # EDM loss weighting
                w = (sigma_t_fp32**2 + sigma_data**2) / ((sigma_t_fp32 * sigma_data)**2)

                # Soften extremes (prevents score from dominating at low sigma)
                w = w.clamp_max(50.0)  # cap huge weights
                w = w.sqrt()           # gentler than linear

                # Average over latent dimension to reduce scale by ~D_latent factor
                err2 = (eps_pred_fp32 - eps_fp32).pow(2).mean(dim=2)  # (B,N)
                w_squeezed = w.squeeze(-1)  # (B,N,1) → (B,N)
                L_score = (w_squeezed * err2 * mask_fp32).sum() / mask_fp32.sum()

                # === SANITY CHECK FOR L_SCORE SCALE ===
                if DEBUG and (global_step % LOG_EVERY == 0) and (not is_sc):
                    with torch.no_grad():
                        # Flatten valid entries
                        m = mask_fp32 > 0
                        err2_mean = (err2[m]).mean().item()  # should be ~O(1)
                        w_vals = (w_squeezed.expand_as(err2))[m]
                        w_min, w_mean, w_max = w_vals.min().item(), w_vals.mean().item(), w_vals.max().item()
                        approx_L = w_mean * err2_mean  # rough expectation for L_score
                        
                        print(f"[score/check] err2_mean={err2_mean:.3e}  "
                            f"w_min={w_min:.3e} w_mean={w_mean:.3e} w_max={w_max:.3e}  "
                            f"approx_L={approx_L:.3e}  L_score={L_score.item():.3e}")

            if DEBUG:
                if debug_state['score_bins'] is None:
                    edges = torch.linspace(0, 1, 6, device=device)
                    debug_state['score_bins'] = edges
                    debug_state['score_bin_sum'] = torch.zeros(5, device=device)
                    debug_state['score_bin_cnt'] = torch.zeros(5, device=device)

                edges = debug_state['score_bins']
                b = torch.bucketize(t_norm.squeeze(), edges) - 1
                with torch.no_grad():
                    per = ((eps_pred - eps) ** 2).mean(dim=(1,2))
                    for k in range(5):
                        m = (b == k)
                        if m.any():
                            debug_state['score_bin_sum'][k] += per[m].sum()
                            debug_state['score_bin_cnt'][k] += m.sum()
            
            # Initialize other losses
            L_gram = torch.tensor(0.0, device=device)
            L_gram_scale = torch.tensor(0.0, device=device)
            L_heat = torch.tensor(0.0, device=device)
            L_sw_st = torch.tensor(0.0, device=device)
            L_sw_sc = torch.tensor(0.0, device=device)
            L_overlap = torch.tensor(0.0, device=device)
            L_ordinal_sc = torch.tensor(0.0, device=device)
            
            # Denoise to get V_hat
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                V_hat = V_t - sigma_t * eps_pred
                V_hat = V_hat * mask.unsqueeze(-1).float()

            # ----------------- Cone Loss (PSD penalty) for ST -----------------
            L_cone = torch.tensor(0.0, device=device)
            # if not is_sc:
            #     with torch.autocast(device_type='cuda', enabled=False):
            #         BATCH_SUB_M = 64  # sub-miniset size
            #         for i in range(batch_size_real):
            #             n_valid = int(n_list[i].item())
            #             if n_valid < 8:
            #                 continue

            #             # Sample a sub-miniset S ⊂ {1..n_valid}
            #             m = min(BATCH_SUB_M, n_valid)
            #             idx = torch.randperm(n_valid, device=device)[:m]
            #             V_i = V_hat[i, :n_valid].float()
            #             V_sub = V_i[idx]  # (m, D)

            #             # Build distances, center, compute B = -1/2 J D^2 J
            #             D_sub = torch.cdist(V_sub, V_sub)
            #             Jm = torch.eye(m, device=device) - torch.ones(m, m, device=device) / m
            #             B = -0.5 * (Jm @ (D_sub**2) @ Jm)

            #             # PSD hinge on negative eigenvalues only
            #             eigs = torch.linalg.eigvalsh(B)
            #             neg = torch.relu(-eigs)  # only negative parts
            #             if neg.numel() > 0:
            #                 L_cone += neg.mean()

            #         L_cone = L_cone / max(1, batch_size_real)

            #     # DEBUG: Cone PSD stats
            #     if DEBUG and (global_step % LOG_EVERY == 0) and not is_sc:
            #        print(f"[cone] penalty={L_cone.item():.6f} (should decrease over time)")

            if not is_sc:
                # ===== ST STEP: Score + Gram + Heat + SW_ST =====

                # Canonicalize V_hat for geometry: center only, no RMS scaling
                with torch.autocast(device_type='cuda', enabled=False):
                    V_hat_f32 = V_hat.float()
                    m_bool = mask.bool()
                    m_float = mask.float().unsqueeze(-1)              # (B,N,1)

                    # center per set over valid nodes
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
                    mean = (V_hat_f32 * m_float).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
                    V_geom = (V_hat_f32 - mean) * m_float             # (B,N,D), centered, scale-preserving

                    Gt = G_target.float()
                    B, N, _ = V_geom.shape

                    # ==================== GRAM LOSS (COSINE-NORMALIZED) ====================
                    
                    # Build predicted Gram *with* true scale
                    Gp_raw = V_geom @ V_geom.transpose(1, 2)          # (B,N,N)
                    
                    # Build masks
                    MM = (m.unsqueeze(-1) & m.unsqueeze(-2)).float()  # (B,N,N) valid pairs
                    eye = torch.eye(N, dtype=torch.bool, device=Gp_raw.device).unsqueeze(0)
                    P_off = (MM.bool() & (~eye)).float()  # off-diagonal valid entries
                    
                    # --- DEBUG 1: Target checksum (ensure G_target varies across batches) ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        gt_checksum = float(Gt.abs().sum().item())
                        print(f"[gram/gt] checksum={gt_checksum:.6e} shape={tuple(Gt.shape)}")
                    
                    # --- DEBUG 2: Raw statistics (before any normalization) ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        gp_off_raw = Gp_raw[P_off.bool()]
                        gt_off_raw = Gt[P_off.bool()]
                        if gp_off_raw.numel() > 0 and gt_off_raw.numel() > 0:
                            diff_raw = gp_off_raw - gt_off_raw
                            rel_frob_raw = diff_raw.norm() / (gt_off_raw.norm() + 1e-12)
                            cos_sim_raw = float(F.cosine_similarity(gp_off_raw, gt_off_raw, dim=0).item()) if (gp_off_raw.norm() > 0 and gt_off_raw.norm() > 0) else float('nan')
                            print(f"[gram/raw] offdiag stats | "
                                f"P(mean={gp_off_raw.mean().item():.3e}, std={gp_off_raw.std().item():.3e})  "
                                f"T(mean={gt_off_raw.mean().item():.3e}, std={gt_off_raw.std().item():.3e})  "
                                f"ΔF/TF={rel_frob_raw:.3e}  cos={cos_sim_raw:.3f}")
                    
                    # --- DEBUG 3: Trace statistics (raw space) ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        tr_p_raw = (Gp_raw.diagonal(dim1=1, dim2=2) * m.float()).sum(dim=1)
                        tr_t_raw = (Gt.diagonal(dim1=1, dim2=2) * m.float()).sum(dim=1)
                        ratio_tr = (tr_p_raw.mean() / tr_t_raw.mean().clamp_min(1e-12) * 100).item()
                        print(f"[gram/trace] pred={tr_p_raw.mean().item():.1f} "
                            f"target={tr_t_raw.mean().item():.1f} ratio={ratio_tr:.1f}%")
                        
                        # Store for epoch-end health check
                        debug_state['last_gram_trace_ratio'] = ratio_tr

                    
                    # ===== COSINE GRAM NORMALIZATION (diagonal normalization) =====
                    diag_p = Gp_raw.diagonal(dim1=1, dim2=2).clamp_min(1e-8)  # (B,N)
                    diag_t = Gt.diagonal(dim1=1, dim2=2).clamp_min(1e-8)      # (B,N)
                    
                    # Outer product of sqrt(diagonals) for normalization
                    scale_p = torch.sqrt(diag_p.unsqueeze(2) * diag_p.unsqueeze(1))  # (B,N,N)
                    scale_t = torch.sqrt(diag_t.unsqueeze(2) * diag_t.unsqueeze(1))  # (B,N,N)
                    
                    # Cosine-normalized Grams (values in [-1,1] range)
                    Cp = (Gp_raw / scale_p).where(MM.bool(), torch.zeros_like(Gp_raw))
                    Ct = (Gt / scale_t).where(MM.bool(), torch.zeros_like(Gt))
                    Cp = Cp.masked_fill(eye, 0.0)  # zero out diagonals
                    Ct = Ct.masked_fill(eye, 0.0)
                    
                    # --- DEBUG 4: Cosine-normalized statistics ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        cp_off = Cp[P_off.bool()]
                        ct_off = Ct[P_off.bool()]
                        if cp_off.numel() > 0 and ct_off.numel() > 0:
                            diff_cos = cp_off - ct_off
                            rel_frob_cos = diff_cos.norm() / (ct_off.norm() + 1e-12)
                            cos_sim_cos = float(F.cosine_similarity(cp_off, ct_off, dim=0).item()) if (cp_off.norm() > 0 and ct_off.norm() > 0) else float('nan')
                            print(f"[gram/cosine] offdiag stats | "
                                f"C_P(mean={cp_off.mean().item():.3e}, std={cp_off.std().item():.3e})  "
                                f"C_T(mean={ct_off.mean().item():.3e}, std={ct_off.std().item():.3e})  "
                                f"ΔF/TF={rel_frob_cos:.3e}  cos={cos_sim_cos:.3f}")
                    
                    # ===== RELATIVE FROBENIUS LOSS (per-set, then mean) =====
                    diff = (Cp - Ct) * P_off
                    numerator = (diff.pow(2)).sum(dim=(1,2))  # (B,)
                    denominator = ((Ct.pow(2)) * P_off).sum(dim=(1,2)).clamp_min(1e-12)  # (B,)
                    per_set_relative_loss = numerator / denominator  # (B,)

                    # Sigma compensation: geometry gradients scale with sigma, so compensate
                    sigma_vec = sigma_t.view(-1).float()  # (B,)
                    w_geom = (1.0 / sigma_vec.clamp_min(0.3)).pow(1.0)  # gamma=1.0, floor sigma at 0.3
                    per_set_relative_loss = per_set_relative_loss * w_geom

                    # Gate geometry to conditional + reasonable sigma samples
                    low_noise = (sigma_vec <= 1.5)  # geometry meaningful here
                    cond_only = (drop_mask.view(-1) < 0.5)  # not unconditional
                    geo_gate = (low_noise & cond_only)  # (B,)

                    if geo_gate.any():
                        L_gram = per_set_relative_loss[geo_gate].mean()
                    else:
                        L_gram = V_hat.new_tensor(0.0)
                    
                    # L_gram = per_set_relative_loss.mean()  # scalar, O(1) magnitude
                    
                    # --- DEBUG 5: Loss statistics ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        print(f"[gram/loss] L_gram={L_gram.item():.3e} | "
                            f"per_set: mean={per_set_relative_loss.mean().item():.3e} "
                            f"med={per_set_relative_loss.median().item():.3e} "
                            f"min={per_set_relative_loss.min().item():.3e} "
                            f"max={per_set_relative_loss.max().item():.3e}")
                        
                        # Mask coverage
                        denom_per_set = P_off.sum(dim=(1,2))
                        n_valid = m.sum(dim=1).float()
                        print(f"[gram/mask] offdiag_counts: min={denom_per_set.min().item():.0f} "
                            f"mean={denom_per_set.mean().item():.0f} max={denom_per_set.max().item():.0f} | "
                            f"n_valid: mean={n_valid.mean().item():.1f}")
                    
                    # --- DEBUG 6: Gradient flow check ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        g_probe = torch.autograd.grad(L_gram, V_hat, retain_graph=True, allow_unused=True)[0]
                        gnorm = float(g_probe.norm().item()) if g_probe is not None else 0.0
                        print(f"[gram/grad] ||∂L_gram/∂V_hat||={gnorm:.3e}")
                    
                    # --- DEBUG 7: Comparison with raw (no centering) as a sanity check ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        V_raw = (V_hat.float() * m_float)                    # just mask, no centering
                        Gp_raw_nocenter = V_raw @ V_raw.transpose(1,2)
                        diff_nocenter = (Gp_raw_nocenter[P_off.bool()] - Gt[P_off.bool()])
                        rel_frob_nocenter = diff_nocenter.norm() / (Gt[P_off.bool()].norm() + 1e-12)
                        print(f"[gram/raw-no-center-probe] ΔF/TF={rel_frob_nocenter:.3e}")
                    
                    # --- DEBUG 8: Old unit-trace normalization (for comparison) ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        tr_p_ut = (Gp_raw.diagonal(dim1=1, dim2=2) * m.float()).sum(dim=1, keepdim=True).clamp_min(1e-8)
                        tr_t_ut = (Gt.diagonal(dim1=1, dim2=2) * m.float()).sum(dim=1, keepdim=True).clamp_min(1e-8)
                        Gp_ut = Gp_raw / tr_p_ut.unsqueeze(-1)
                        Gt_ut = Gt / tr_t_ut.unsqueeze(-1)
                        diff_sq_ut = (Gp_ut - Gt_ut).pow(2)
                        per_set_mse_ut = (diff_sq_ut * P_off).sum(dim=(1,2)) / P_off.sum(dim=(1,2)).clamp_min(1.0)
                        L_gram_ut_probe = per_set_mse_ut.mean()
                        print(f"[gram/unit-trace-probe] L_old={L_gram_ut_probe.item():.3e} (for comparison)")


                    # ==================== SCALE-MATCHING PENALTY ====================
                    # Force predicted Gram trace to match target Gram trace
                    with torch.autocast(device_type='cuda', enabled=False):
                        tr_p = (Gp_raw.diagonal(dim1=1, dim2=2) * m_float.squeeze(-1)).sum(dim=1)  # (B,)
                        tr_t = (Gt.diagonal(dim1=1, dim2=2) * m_float.squeeze(-1)).sum(dim=1)      # (B,)

                        # Log-ratio for scale-invariant gradients
                        log_ratio = torch.log(tr_p + 1e-8) - torch.log(tr_t + 1e-8)      # (B,)

                        if geo_gate.any():
                            # Only use low-noise, conditional samples (same gate as L_gram)
                            L_gram_scale = (log_ratio[geo_gate] ** 2).mean()
                        else:
                            L_gram_scale = V_hat.new_tensor(0.0)
                
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

                L_sw_st = torch.zeros((), device=device)
                if (global_step % sw_every_k) == 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # build a paired batch: (pred_i, tgt_i) for each i
                        B2, N, D = V_hat.shape  # B2 is your current batch_size_real
                        # pre-alloc
                        VT = torch.zeros(B2, N, D, device=device, dtype=torch.float32)
                        mask_st = mask.clone()

                        for i in range(batch_size_real):
                            n_valid = int(n_list[i].item())
                            if n_valid < 3:
                                continue

                            # reconstruct a D-dim embedding from Gram target
                            G_i = batch['G_target'][i, :n_valid, :n_valid].to(device).float()
                            V_tgt_i = uet.factor_from_gram(G_i, D).to(device)  # (n_valid, D)

                            VT[i, :n_valid] = V_tgt_i

                        # stack (pred, tgt) as pairs: (0,1), (2,3), ...
                        V_pairs = torch.zeros(batch_size_real * 2, N, D, device=device, dtype=torch.float32)
                        M_pairs = torch.zeros(batch_size_real * 2, N, device=device, dtype=torch.bool)

                        for i in range(batch_size_real):
                            n_valid = int(n_list[i].item())
                            if n_valid < 3:
                                continue
                            # predicted
                            V_pairs[2*i, :n_valid] = V_hat[i, :n_valid].float()
                            M_pairs[2*i, :n_valid] = True
                            # target (from Gram)
                            V_pairs[2*i+1, :n_valid] = VT[i, :n_valid]
                            M_pairs[2*i+1, :n_valid] = True

                        pairA = torch.arange(0, 2*batch_size_real, 2, device=device, dtype=torch.long)
                        pairB = torch.arange(1, 2*batch_size_real, 2, device=device, dtype=torch.long)

                        L_sw_st = uet.sliced_wasserstein_sc(
                            V_pairs, M_pairs,
                            pair_idxA=pairA,
                            pair_idxB=pairB,
                            K_proj=64,
                            N_cap=512,
                            use_canon=True,   # center + unit-RMS per set
                        )

                        if DEBUG and (global_step % LOG_EVERY == 0):
                            print(f"[sw_st] step={global_step} val={float(L_sw_st.item()):.6f} K=64 cap=512 pairs={int(pairA.numel())}")

                # ============================================================================
                # DEBUG: RIGID TRANSFORM INVARIANCE TEST (ST batch only)
                # ============================================================================
                if DEBUG and (global_step % 200 == 0):
                    with torch.no_grad():
                        i_test = 0
                        n_test = int(n_list[i_test].item())
                        if n_test >= 3:
                            print("\n" + "="*80)
                            print(f"[RIGID_INVARIANCE_TEST] step={global_step}")
                            print("="*80)

                            Vf32 = V_hat[i_test, :n_test].detach().float()
                            V = Vf32.double()  # fp64 for numerical headroom
                            D = V.shape[1]

                            # Stable SO(D) rotation via SVD (orthonormal to machine precision)
                            M = torch.randn(D, D, device=V.device, dtype=torch.float64)
                            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                            R = U @ Vh
                            if torch.linalg.det(R) < 0:  # force determinant +1
                                R[:, 0] = -R[:, 0]

                            # Translation scaled to data spread to avoid huge magnitudes
                            t = torch.randn(D, device=V.device, dtype=torch.float64) * (V.std() + 1e-6)

                            Vt = (V @ R.T) + t

                            # Translation/rotation-invariant distance computation in fp64
                            def _pdist(X):
                                x2 = (X * X).sum(dim=1, keepdim=True)
                                D2 = (x2 + x2.T - 2.0 * (X @ X.T)).clamp_min(0.0)
                                return D2.sqrt()

                            D0 = _pdist(V)
                            D1 = _pdist(Vt)

                            diff = (D1 - D0)
                            abs_max = diff.abs().max().item()
                            rel = (diff.norm() / (D0.norm().clamp_min(1e-12))).item()

                            print(f"[rigid] abs_max={abs_max:.3e}  rel={rel:.3e}")

                            # Use relaxed, numerically sane tolerances
                            tol_abs = 5e-4
                            tol_rel = 1e-8
                            if not (abs_max < tol_abs or rel < tol_rel):
                                print("⚠ rigid check not within tolerance; continuing (diagnostic only)")
                            else:
                                print("✓ rigid distances preserved within tolerance")

                            print("="*80 + "\n")          
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

                # ---- (2) Sliced Wasserstein between paired SC sets (A,B) ----
                L_sw_sc = torch.zeros((), device=device)
                if (global_step % sw_every_k) == 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # predicted embeddings + mask for ALL sets in this SC batch
                        V_sc    = V_hat.float()   # shape: (2B, N, D)
                        mask_sc = mask            # shape: (2B, N), bool

                        # pair indices from collate; fallback to (0,1), (2,3), ...
                        pairA = batch.get('pair_idxA', None)
                        pairB = batch.get('pair_idxB', None)
                        if (pairA is None) or (pairB is None):
                            # fallback if you haven't added these to collate yet
                            pairA = torch.arange(0, V_sc.size(0), 2, device=device, dtype=torch.long)
                            pairB = torch.arange(1, V_sc.size(0), 2, device=device, dtype=torch.long)

                        L_sw_sc = uet.sliced_wasserstein_sc(
                            V_sc, mask_sc,
                            pair_idxA=pairA,
                            pair_idxB=pairB,
                            K_proj=SW_SC_KPROJ,
                            N_cap=SW_SC_NCAP,
                            use_canon=True,
                        )

                        if DEBUG and (global_step % LOG_EVERY == 0):
                            print(f"[sw_sc] step={global_step} val={float(L_sw_sc.item()):.6f} "
                                f"K={SW_SC_KPROJ} cap={SW_SC_NCAP} pairs={int(pairA.numel())}")
                else:
                    L_sw_sc = torch.zeros((), device=device)

                # ----------------- (3) Distance-only overlap (SC ONLY) -----------------
                L_overlap = torch.tensor(0.0, device=device)
                overlap_pairs = 0

                # DEBUG: Check if overlap keys exist (first 10 steps only)
                if DEBUG and global_step < 10 and (global_step % EVERY_K_STEPS) == 0:
                   has_keys = all(k in batch for k in ["pair_idxA","pair_idxB","shared_A_idx","shared_B_idx"])
                   if has_keys:
                       print(f"[overlap_check] step={global_step} pair_idxA.shape={batch['pair_idxA'].shape} "
                             f"shared_A_idx.shape={batch['shared_A_idx'].shape}")
                   else:
                       print(f"[overlap_check] step={global_step} ⚠️ MISSING OVERLAP KEYS - dataloader broken!")

                need = ("pair_idxA","pair_idxB","shared_A_idx","shared_B_idx")
                has_all = all(k in batch for k in need)

                if has_all and batch["pair_idxA"].numel() > 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        P = batch['pair_idxA'].size(0)

                        idxA, idxB = batch['pair_idxA'], batch['pair_idxB']
                        VA_raw, VB_raw = V_hat[idxA].float(), V_hat[idxB].float()
                        maskA, maskB   = mask[idxA], mask[idxB]

                        SA, SB = batch['shared_A_idx'], batch['shared_B_idx']
                        validA, validB = SA.ge(0), SB.ge(0)
                        P, Kmax = SA.size(0), SA.size(1)

                        is_lmk = batch.get('is_landmark', None)
                        if is_lmk is not None:
                            is_lmkA = is_lmk[idxA]; is_lmkB = is_lmk[idxB]

                        L_overlap = V_hat.new_tensor(0.0)
                        overlap_pairs = 0

                        for p in range(P):
                            mA = validA[p]; mB = validB[p]
                            if not (mA.any() and mB.any()):
                                continue

                            gA_all = batch['sc_global_indices'][idxA[p]]
                            gB_all = batch['sc_global_indices'][idxB[p]]
                            gA = gA_all[SA[p].clamp_min(0)][mA]
                            gB = gB_all[SB[p].clamp_min(0)][mB]

                            if is_lmk is not None:
                                keepA = ~is_lmkA[p, SA[p].clamp_min(0)][mA]
                                keepB = ~is_lmkB[p, SB[p].clamp_min(0)][mB]
                                gA = gA[keepA]; gB = gB[keepB]

                            common, a_pos, b_pos = _intersect1d_with_indices(gA, gB)
                            k = common.numel()
                            if k < 5:
                                continue

                            if k > MAX_OVERLAP_POINTS:
                                sel = torch.randperm(k, device=V_hat.device)[:MAX_OVERLAP_POINTS]
                                a_pos = a_pos[sel]; b_pos = b_pos[sel]
                                k = a_pos.numel()

                            A_all = VA_raw[p]; B_all = VB_raw[p]
                            A_sel = SA[p].clamp_min(0)[mA]
                            B_sel = SB[p].clamp_min(0)[mB]
                            A_pts = A_all[A_sel[a_pos]].float()
                            B_pts = B_all[B_sel[b_pos]].float()

                            muA = A_pts.mean(0, keepdim=True)
                            muB = B_pts.mean(0, keepdim=True)
                            Ac = A_pts - muA
                            Bc = B_pts - muB

                            U, S, Vh = torch.linalg.svd(Ac.t() @ Bc, full_matrices=False)
                            R = (U @ Vh).detach()
                            denom = (Ac.pow(2).sum() + 1e-12)
                            s = (S.sum() / denom).detach()

                            A_aligned = s * (Ac @ R)
                            L_pair = (A_aligned - Bc).pow(2).mean()

                            L_overlap = L_overlap + L_pair
                            overlap_pairs += 1

                            if DEBUG:
                                debug_state['dbg_k_mean'] += float(k)
                                debug_state['dbg_overlap_pairs'] += 1

                        if overlap_pairs > 0:
                            L_overlap = L_overlap / overlap_pairs

                            if DEBUG:
                                debug_state['dbg_overlap_seen'] += 1
                                debug_state['overlap_count_this_epoch'] += 1

                    sc_prof.clear()
            
            
            # Total loss with optional score deprioritization on geometry-heavy batches
            score_multiplier = 1.0
            if not is_sc:
                if 'geo_gate' in locals() and geo_gate.float().mean() > 0.5:
                    score_multiplier = 0.25

            L_total = (WEIGHTS['score'] * score_multiplier * L_score +
                                # WEIGHTS['cone'] * L_cone +
                                WEIGHTS['gram'] * L_gram +
                                WEIGHTS['gram_scale'] * L_gram_scale +
                                WEIGHTS['heat'] * L_heat +
                                WEIGHTS['sw_st'] * L_sw_st +
                                WEIGHTS['sw_sc'] * L_sw_sc +
                                WEIGHTS['overlap'] * L_overlap +
                                WEIGHTS['ordinal_sc'] * L_ordinal_sc)
            
            # ==================== GRADIENT PROBE (IMPROVED) ====================
            if DEBUG and (global_step % LOG_EVERY == 0):
                def vhat_gn(loss):
                    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                        return 0.0
                    g = torch.autograd.grad(loss, V_hat, retain_graph=True, allow_unused=True)[0]
                    return 0.0 if (g is None) else float(g.norm().item())

                # Score: interface-equivalent grad via eps_pred (chain rule: dL/dV = dL/deps / sigma)
                def vhat_gn_from_eps(loss, eps_pred_tensor, sigma_tensor):
                    if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
                        return 0.0
                    g_eps = torch.autograd.grad(loss, eps_pred_tensor, retain_graph=True, allow_unused=True)[0]
                    if g_eps is None:
                        return 0.0
                    inv_sigma = (1.0 / sigma_tensor.float()).expand_as(g_eps)  # (B,1,1) -> (B,N,D)
                    g_vhat_equiv = g_eps * inv_sigma  # ||∂L/∂V_hat|| = ||∂L/∂eps|| / σ
                    return float(g_vhat_equiv.norm().item())

                vgn_score = vhat_gn_from_eps(L_score, eps_pred, sigma_t)
                vgn_gram = vhat_gn(L_gram) if not is_sc else 0.0
                vgn_heat = vhat_gn(L_heat) if not is_sc else 0.0
                vgn_swst = vhat_gn(L_sw_st) if not is_sc else 0.0
                vgn_cone = vhat_gn(L_cone) if not is_sc else 0.0
                vgn_swsc = vhat_gn(L_sw_sc) if is_sc else 0.0
                vgn_ord = vhat_gn(L_ordinal_sc) if is_sc else 0.0
                vgn_overlap = vhat_gn(L_overlap) if is_sc else 0.0
                
                # Add batch type label
                batch_type = 'SC' if is_sc else 'ST'
                
                print(f"[vhatprobe][{batch_type}] score={vgn_score:.3e} gram={vgn_gram:.3e} heat={vgn_heat:.3e} "
                    f"sw_st={vgn_swst:.3e} sw_sc={vgn_swsc:.3e} overlap={vgn_overlap:.3e} ord={vgn_ord:.3e}")
                
                # Seed EMAs from first ST batch (prevents wrong-direction updates)
                if (global_step == 0) and (not is_sc):
                    ema_grads['score'] = max(vgn_score, 1e-12)
                    ema_grads['gram']  = max(vgn_gram,  1e-12)
                    ema_grads['heat']  = max(vgn_heat,  1e-12)
                    ema_grads['sw_st'] = max(vgn_swst,  1e-12)
                    if DEBUG:
                        print(f"[ema_init] score={ema_grads['score']:.3e} gram={ema_grads['gram']:.3e} "
                            f"heat={ema_grads['heat']:.3e} sw_st={ema_grads['sw_st']:.3e}")

                # Seed SC EMAs from first SC batch
                if (global_step <= 10) and is_sc and (vgn_score > 0):
                    if ema_sc['score'] == 1.0:  # Only init once
                        ema_sc['score'] = max(vgn_score, 1e-12)
                        ema_sc['sw_sc'] = max(vgn_swsc, 1e-12)
                        ema_sc['overlap'] = 0.1  # Will be set after we add overlap to vhatprobe
                        ema_sc['ordinal_sc'] = max(vgn_ord, 1e-12)
                
                # Param probe - better parameter selection, skip on SC to avoid confusing zeros
                if not is_sc:
                    # Pick evenly-spaced parameters across the network
                    def pick_probe_params(model, k=8):
                        params = [p for p in model.parameters() if p.requires_grad]
                        if len(params) <= k:
                            return params
                        # Even spread over the parameter list to hit early/mid/late layers
                        idx = torch.linspace(0, len(params)-1, steps=k).round().long().tolist()
                        return [params[i] for i in idx]
                    
                    probe_params = pick_probe_params(score_net, k=8)
                    
                    def grad_norm(loss):
                        if not isinstance(loss, torch.Tensor) or (not loss.requires_grad):
                            return 0.0
                        g = torch.autograd.grad(loss, probe_params, retain_graph=True, allow_unused=True)
                        s = torch.tensor(0.0, device=device)
                        for gi in g:
                            if gi is not None:
                                s = s + gi.norm().pow(2)
                        return float(s.sqrt().item()) if s.item() > 0 else 0.0
                    
                    gn_score = grad_norm(L_score)
                    gn_gram = grad_norm(L_gram)
                    gn_heat = grad_norm(L_heat)
                    gn_swst = grad_norm(L_sw_st)
                    # gn_cone = grad_norm(L_cone)
                    gn_swsc = 0.0
                    gn_ord = 0.0
                    
                    print(f"[gradprobe][{batch_type}] score={gn_score:.3e} gram={gn_gram:.3e} heat={gn_heat:.3e} "
                        f"sw_st={gn_swst:.3e} sw_sc={gn_swsc:.3e} ord={gn_ord:.3e}")
                else:
                    # Skip param probe on SC batches to avoid confusion
                    print(f"[gradprobe][{batch_type}] (skipped on SC batches)")

                # ---- Stable EMA-based autobalance ----
                if DEBUG and (global_step % LOG_EVERY == 0):
                    ratio_gs   = vgn_gram / max(vgn_score, 1e-12)
                    ratio_hs   = vgn_heat / max(vgn_score, 1e-12)
                    ratio_swst = vgn_swst / max(vgn_score, 1e-12)
                    print(f"[ratios] gram/score={ratio_gs:.3e} heat/score={ratio_hs:.3e} "
                        f"sw_st/score={ratio_swst:.3e} | w_gram={WEIGHTS['gram']:.3g}")

                # Update EMAs (separate for ST vs SC)
                if not is_sc:
                    ema_grads['score'] = EMA_BETA * ema_grads['score'] + (1 - EMA_BETA) * (vgn_score + 1e-12)
                    ema_grads['gram']  = EMA_BETA * ema_grads['gram']  + (1 - EMA_BETA) * (vgn_gram  + 1e-12)
                    ema_grads['heat']  = EMA_BETA * ema_grads['heat']  + (1 - EMA_BETA) * (vgn_heat  + 1e-12)
                    ema_grads['sw_st'] = EMA_BETA * ema_grads['sw_st'] + (1 - EMA_BETA) * (vgn_swst + 1e-12)
                else:
                    ema_sc['score']      = EMA_BETA * ema_sc['score']      + (1 - EMA_BETA) * (vgn_score   + 1e-12)
                    ema_sc['sw_sc']      = EMA_BETA * ema_sc['sw_sc']      + (1 - EMA_BETA) * (vgn_swsc    + 1e-12)
                    ema_sc['overlap']    = EMA_BETA * ema_sc['overlap']    + (1 - EMA_BETA) * (vgn_overlap + 1e-12)
                    ema_sc['ordinal_sc'] = EMA_BETA * ema_sc['ordinal_sc'] + (1 - EMA_BETA) * (vgn_ord     + 1e-12)

                # Adjust weights (ST batches only, after warmup, at intervals)
                if (not is_sc) and (global_step >= AUTOBALANCE_START) and (global_step % AUTOBALANCE_EVERY == 0):
                    bounds = {'gram': (6.0, 20.0), 'heat': (0.6, 3.0), 'sw_st': (0.8, 3.0)}

                    def _update(key):
                        ratio = ema_grads[key] / (ema_grads['score'] + 1e-12)
                        target = TARGET[key]
                        delta = max(-0.1, min(0.1, math.log(target / (ratio + 1e-12))))
                        new_w = WEIGHTS[key] * math.exp(delta)
                        lo, hi = bounds[key]
                        WEIGHTS[key] = float(min(max(new_w, lo), hi))

                    rank = 0
                    try:
                        if dist.is_available() and dist.is_initialized():
                            rank = dist.get_rank()
                    except Exception:
                        pass

                    if rank == 0:
                        for key in ['gram', 'heat', 'sw_st']:
                            _update(key)

                    try:
                        if dist.is_available() and dist.is_initialized():
                            w = torch.tensor(
                                [WEIGHTS['gram'], WEIGHTS['heat'], WEIGHTS['sw_st']],
                                device=device, dtype=torch.float32
                            )
                            dist.broadcast(w, src=0)
                            WEIGHTS['gram'], WEIGHTS['heat'], WEIGHTS['sw_st'] = map(float, w.tolist())
                    except Exception:
                        pass

                    if DEBUG and (rank == 0):
                        print(f"[autobalance @ step {global_step}] "
                            f"gram={WEIGHTS['gram']:.3g} heat={WEIGHTS['heat']:.3g} sw_st={WEIGHTS['sw_st']:.3g}")

                # SC-side autobalance (separate from ST)
                if is_sc and (global_step >= AUTOBALANCE_START) and (global_step % AUTOBALANCE_EVERY == 0):
                    bounds_sc = {
                        'sw_sc': (0.4, 1.2),
                        'overlap': (0.10, 0.6),
                        'ordinal_sc': (0.2, 1.0),
                    }

                    def _update_sc(key):
                        ratio = ema_sc[key] / (ema_sc['score'] + 1e-12)
                        target = TARGET_SC[key]
                        delta = max(-0.1, min(0.1, math.log(target / (ratio + 1e-12))))
                        new_w = WEIGHTS[key] * math.exp(delta)
                        lo, hi = bounds_sc[key]
                        WEIGHTS[key] = float(min(max(new_w, lo), hi))

                    rank = 0
                    try:
                        if dist.is_available() and dist.is_initialized():
                            rank = dist.get_rank()
                    except Exception:
                        pass

                    if rank == 0:
                        for key in ['sw_sc', 'overlap', 'ordinal_sc']:
                            _update_sc(key)

                    try:
                        if dist.is_available() and dist.is_initialized():
                            w = torch.tensor(
                                [WEIGHTS['sw_sc'], WEIGHTS['overlap'], WEIGHTS['ordinal_sc']],
                                device=device, dtype=torch.float32
                            )
                            dist.broadcast(w, src=0)
                            WEIGHTS['sw_sc'], WEIGHTS['overlap'], WEIGHTS['ordinal_sc'] = map(float, w.tolist())
                    except Exception:
                        pass

                    if DEBUG and (rank == 0):
                        print(f"[autobalance_SC @ step {global_step}] "
                            f"sw_sc={WEIGHTS['sw_sc']:.3g} overlap={WEIGHTS['overlap']:.3g} "
                            f"ordinal_sc={WEIGHTS['ordinal_sc']:.3g}")
                # =========================================================
            
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
            # epoch_losses['cone'] += L_cone.item()
            epoch_losses['gram'] += L_gram.item()
            epoch_losses['gram_scale'] += L_gram_scale.item()
            epoch_losses['heat'] += L_heat.item()
            epoch_losses['sw_st'] += L_sw_st.item()
            epoch_losses['sw_sc'] += L_sw_sc.item()
            epoch_losses['overlap'] += L_overlap.item()
            epoch_losses['ordinal_sc'] += L_ordinal_sc.item()

            def _is_rank0():
                return (not dist.is_initialized()) or dist.get_rank() == 0

            if L_overlap.item() > 0:  
                c_overlap += 1       
            
            n_batches += 1
            global_step += 1
            # batch_pbar.update(1)
        
            # (optional) metrics logging
            if fabric is None or fabric.is_global_zero:
                pass  # print / tqdm here if you want

            # DEBUG: Per-batch logging
            if DEBUG and (global_step % LOG_EVERY == 0):
                Vn = float(V_hat.norm().item()) / (mask.sum().item()**0.5 * D_latent**0.5 + 1e-8)
                bt = 'SC' if is_sc else 'ST'
                print(f"[{bt} step {global_step}] ||V_hat||_rms={Vn:.3f} "
                    f"score={L_score.item():.4f} gram={L_gram.item():.3e} "
                    f"sw_st={L_sw_st.item():.4f} sw_sc={L_sw_sc.item():.4f} "
                    f"overlap={L_overlap.item():.4f}")

        
        scheduler.step()

        # Adjust weights after initial epochs
        if epoch == 5:
            print("\n⚡ [Weight Adjustment] Reducing overlap weight from 0.25 → 0.15")
            WEIGHTS['overlap'] = 0.15

        # MLflow logging
        if logger and (fabric is None or fabric.is_global_zero):
            if global_step % log_interval == 0:
                metrics = {
                    "train/total_loss": L_total.item(),
                    "train/L_score": L_score.item(),
                    "train/L_gram": L_gram.item(),
                    "train/L_heat": L_heat.item(),
                    "train/L_sw_st": L_sw_st.item(),
                    "train/L_sw_sc": L_sw_sc.item(),
                    "train/L_overlap": L_overlap.item(),
                    "train/L_ordinal_sc": L_ordinal_sc.item(),
                    "opt/lr": optimizer.param_groups[0]["lr"],
                }
                metrics.update(logger.gpu_stats(gpu_id=0))
                logger.log_metrics(metrics, step=global_step)
                logger.set_step(global_step)

        # -------------------- END-OF-EPOCH AVERAGING (FIXED) --------------------

        # sum losses across ranks
        sum_losses = {k: torch.tensor([v], device=device, dtype=torch.float32) for k, v in epoch_losses.items()}
        for k in sum_losses:
            if dist.is_initialized():
                dist.all_reduce(sum_losses[k], op=dist.ReduceOp.SUM)

        # sum batch counts across ranks
        sum_batches = torch.tensor([float(n_batches)], device=device, dtype=torch.float32)
        cnt_st = torch.tensor([float(st_batches)], device=device, dtype=torch.float32)
        cnt_sc = torch.tensor([float(sc_batches)], device=device, dtype=torch.float32)

        if dist.is_initialized():
            dist.all_reduce(sum_batches, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_st, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_sc, op=dist.ReduceOp.SUM)

        # sum overlap batch count
        sum_overlap_batches = torch.tensor([float(debug_state['overlap_count_this_epoch'])], device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(sum_overlap_batches, op=dist.ReduceOp.SUM)

        # Determine denominator per loss type
        def _denom_for(key):
            if key in ('gram', 'gram_scale', 'heat', 'sw_st', 'cone'):
                return cnt_st
            if key in ('sw_sc', 'ordinal_sc'):
                return cnt_sc
            if key == 'overlap':
                return sum_overlap_batches
            return sum_batches  # 'total' and 'score'

        # compute true epoch means
        epoch_means = {}
        for k in sum_losses:
            denom = _denom_for(k).clamp_min(1.0)
            epoch_means[k] = (sum_losses[k] / denom).item()

        # store + history
        epoch_losses = epoch_means
        for k, v in epoch_losses.items():
            history['epoch_avg'][k].append(v)
        history['epoch'].append(epoch + 1)

        # Recompute weighted total from means (don't use the raw 'total' sum)
        epoch_losses['total'] = (
            WEIGHTS['score'] * epoch_losses['score'] +
            WEIGHTS['gram'] * epoch_losses['gram'] +
            WEIGHTS['gram_scale'] * epoch_losses['gram_scale'] +
            WEIGHTS['heat'] * epoch_losses['heat'] +
            WEIGHTS['sw_st'] * epoch_losses['sw_st'] +
            WEIGHTS['sw_sc'] * epoch_losses['sw_sc'] +
            WEIGHTS['overlap'] * epoch_losses['overlap'] +
            WEIGHTS['ordinal_sc'] * epoch_losses['ordinal_sc']
        )

        # Override the history['epoch_avg']['total'] with correct value
        history['epoch_avg']['total'][-1] = epoch_losses['total']

        # reset overlap batch counter for next epoch
        debug_state['overlap_count_this_epoch'] = 0
        # -----------------------------------------------------------------------

        # DEBUG: Epoch summary
        if DEBUG:
            # Score by sigma
            if debug_state['score_bin_cnt'].sum() > 0:
                msg = " | ".join([f"s{k}:{(debug_state['score_bin_sum'][k]/(debug_state['score_bin_cnt'][k].clamp_min(1))).item():.4f}"
                                for k in range(5)])
                print(f"[epoch {epoch+1}] score_by_sigma: {msg}")
            
            # Overlap stats
            if debug_state['dbg_overlap_pairs'] > 0:
                avg_k = debug_state['dbg_k_mean'] / debug_state['dbg_overlap_pairs']
                print(f"[epoch {epoch+1}] overlap_batches={debug_state['dbg_overlap_seen']} "
                    f"pairs={debug_state['dbg_overlap_pairs']} avg_shared={avg_k:.1f}")
            else:
                print(f"[epoch {epoch+1}] overlap_batches=0 (check dataloader plumbing)")
            
            # Reset for next epoch
            debug_state['dbg_overlap_seen'] = 0
            debug_state['dbg_overlap_pairs'] = 0
            debug_state['dbg_k_mean'] = 0.0
            debug_state['overlap_count_this_epoch'] = 0


        # ==================== HEALTH METRICS TRACKING ====================
        if _is_rank0() and (epoch) % 2 == 0:
            # 1. Heat share
            heat_share = epoch_losses['heat'] / max(epoch_losses['total'], 1e-12)
            heat_pct = heat_share * 100
            
            # 2. High-noise score inflation (from score_by_sigma)
            if debug_state['score_bin_cnt'].sum() > 0:
                s0 = (debug_state['score_bin_sum'][0] / debug_state['score_bin_cnt'][0].clamp_min(1)).item()
                s3 = (debug_state['score_bin_sum'][3] / debug_state['score_bin_cnt'][3].clamp_min(1)).item()
                s4 = (debug_state['score_bin_sum'][4] / debug_state['score_bin_cnt'][4].clamp_min(1)).item()
                r_hi = (s3 + s4) / (2 * max(s0, 1e-12))
            else:
                r_hi = float('nan')
            
            # 3. Gram trace ratio (use last value from epoch)
            gram_trace_ratio = debug_state.get('last_gram_trace_ratio', float('nan'))
            
            # 4. SC-side losses: ordinal + overlap shares
            ord_share = epoch_losses['ordinal_sc'] / max(epoch_losses['total'], 1e-12)
            ovl_share = epoch_losses['overlap'] / max(epoch_losses['total'], 1e-12)
            ord_pct = 100.0 * ord_share
            ovl_pct = 100.0 * ovl_share
            
            # Print health metrics
            print(f"\n{'='*70}")
            print(f"HEALTH METRICS (Epoch {epoch+1})")
            print(f"{'='*70}")
            print(f"1. Heat Share:           {heat_pct:.1f}%  (healthy: 25-50%)")
            print(f"2. Hi-Noise Inflation:   {r_hi:.2f}   (healthy: ≤1.3)")
            print(f"3. Gram Trace Ratio:     {gram_trace_ratio:.1f}%  (healthy: 85-92%)")
            print(f"4. Ordinal_SC Share:     {ord_pct:.1f}%  (healthy: ~5-15%)")
            print(f"5. Overlap Share:        {ovl_pct:.1f}%  (healthy: ~2-10%)")
            
            # Warning triggers
            warnings = []
            if heat_pct > 50:
                warnings.append("⚠️  HEAT SHARE >50% - Consider reducing heat weight or increasing heat_every_k")
            if r_hi > 1.5:
                warnings.append("⚠️  HI-NOISE INFLATION >1.5 - Consider capping sigma_max or reweighting timesteps")
            if gram_trace_ratio < 83 or gram_trace_ratio > 95:
                warnings.append("⚠️  GRAM TRACE RATIO outside 85-92% - Check canonicalization or adjust gram weight")
            if ord_pct < 2.0:
                warnings.append("⚠️  ORDINAL_SC SHARE <2% - Check SC kNN cache or triplet construction")
            if ovl_pct < 1e-2:
                warnings.append("⚠️  OVERLAP SHARE ≈0 - Overlap keys or shared indices may be broken")
            
            if warnings:
                print(f"\n{'WARNING':^70}")
                for w in warnings:
                    print(f"  {w}")
            else:
                print(f"\n  ✓ All metrics within healthy ranges")
            
            print(f"{'='*70}\n")
            
            # Store in history for plotting later
            if 'health_metrics' not in history:
                history['health_metrics'] = {
                    'heat_share': [],
                    'hi_noise_inflation': [],
                    'gram_trace_ratio': [],
                    'ordinal_sc_share': [],
                    'overlap_share': []
                }
            history['health_metrics']['heat_share'].append(heat_pct)
            history['health_metrics']['hi_noise_inflation'].append(r_hi)
            history['health_metrics']['gram_trace_ratio'].append(gram_trace_ratio)
            history['health_metrics']['ordinal_sc_share'].append(ord_pct)
            history['health_metrics']['overlap_share'].append(ovl_pct)

            # After using the score histogram for health metrics, reset for next epoch
            if 'score_bin_sum' in debug_state:
                debug_state['score_bin_sum'].zero_()
                debug_state['score_bin_cnt'].zero_()
        # ================================================================
            
        # Log epoch averages to MLflow
        if logger and (fabric is None or fabric.is_global_zero):
            epoch_metrics = {
                f"epoch/{k}": v for k, v in epoch_losses.items()
            }
            logger.log_metrics(epoch_metrics, step=epoch)
        
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

    return history if _is_rank0() else None

# ==============================================================================
# STAGE D: SC INFERENCE
# ==============================================================================


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
    guidance_scale: float = 8.0,
    # stitching knobs
    K_ANCH: int = 64,            # anchors used per cell for triangulation (>= D + 8)
    L2_REG: float = 1e-4,        # Tikhonov regularization
    CARRY_K: int = 256,          # carry-over frozen cells per batch (sampling only)
    RES_CAP: int = 2000          # reservoir cap (sampling only)
) -> Dict[str, torch.Tensor]:
    """
    Stage D (Inference) -- Structure-first stitching
    ------------------------------------------------
    1) Choose anchors in Z space.
    2) Sample anchors ONCE via diffusion to estimate anchor-anchor distances.
    3) Build a single global anchor frame via MDS on those distances.
    4) Process remaining cells in batches:
       - Sample with anchors+carry frozen (for local quality),
       - Extract only distances to anchors,
       - Triangulate each new cell into the global frame from those distances.
    5) Return global EDM and (optional) global 2D coords.
    """

    #debug flag for inference
    DEBUG_FLAG = True

    import numpy as np
    import torch.nn.functional as F
    import utils_et as uet
    from tqdm import tqdm

    # -------- Init & encode Z --------
    encoder.eval(); context_encoder.eval(); score_net.eval()
    n_sc = sc_gene_expr.shape[0]
    D_latent = score_net.D_latent

    if DEBUG_FLAG:
        print("\n" + "="*72)
        print("STAGE D — STRUCTURE-FIRST ANCHORED INFERENCE")
        print("="*72)
        print(f"[cfg] n_sc={n_sc}  anchor_size={anchor_size}  batch_size={batch_size}  "
              f"timesteps={n_timesteps_sample}  D_latent={D_latent}")
        print(f"[stitch] K_ANCH={K_ANCH}  L2_REG={L2_REG:.1e}  CARRY_K={CARRY_K}  RES_CAP={RES_CAP}")

    # Encode all SC cells to Z (on CPU to save VRAM)
    Z_chunks, encode_bs = [], 1024
    for i in range(0, n_sc, encode_bs):
        z = encoder(sc_gene_expr[i:i+encode_bs].to(device)).cpu()
        Z_chunks.append(z)
    Z_all = torch.cat(Z_chunks, dim=0)                    # (n_sc, h)
    Z_all_unit = Z_all / (Z_all.norm(dim=1, keepdim=True) + 1e-8)

    # -------- Choose anchors (FPS in Z) --------
    anchor_size = min(anchor_size, n_sc)
    anchor_idx = uet.farthest_point_sampling(Z_all_unit, anchor_size, device='cpu')
    mask_non_anchor = torch.ones(n_sc, dtype=torch.bool)
    mask_non_anchor[anchor_idx] = False
    non_anchor_idx = mask_non_anchor.nonzero(as_tuple=False).squeeze(1)

    if DEBUG_FLAG:
        print(f"[anchors] chosen={len(anchor_idx)}  non_anchors={len(non_anchor_idx)}")

    # Prepare context encoder input for anchors
    Z_A = Z_all[anchor_idx].to(device).unsqueeze(0)       # (1, a, h)
    mask_A = torch.ones(1, Z_A.shape[1], dtype=torch.bool, device=device)

    # -------- Sigma schedule --------
    sigmas = torch.exp(torch.linspace(
        torch.log(torch.tensor(sigma_max, device=device)),
        torch.log(torch.tensor(sigma_min, device=device)),
        n_timesteps_sample, device=device
    ))  # (T,)

    # -------- Step 1/2: Sample anchors ONCE to estimate D_AA --------
    if DEBUG_FLAG:
        print("\n[Step A] Sampling anchors once to estimate anchor-anchor distances...")

    H_A = context_encoder(Z_A, mask_A)

    V_t = torch.randn(1, Z_A.shape[1], D_latent, device=device) * sigmas[0]
    for t_idx in range(n_timesteps_sample):
        sigma_t = sigmas[t_idx]
        t_norm = torch.tensor([[t_idx / float(n_timesteps_sample)]], device=device)
        H_null = torch.zeros_like(H_A)

        eps_uncond = score_net(V_t, t_norm, H_null, mask_A)
        eps_cond   = score_net(V_t, t_norm, H_A,    mask_A)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        if DEBUG_FLAG and t_idx < 5:
            ratio = (eps_cond - eps_uncond).norm() / (eps_uncond.norm() + 1e-8)
            print(f"  [anchors] t={t_idx:3d} sigma={float(sigma_t):.4f} CFG_ratio={float(ratio):.4f}")

        if t_idx < n_timesteps_sample - 1:
            sigma_next = sigmas[t_idx + 1]
            V_0_pred = V_t - sigma_t * eps
            V_t = V_0_pred + (sigma_next / sigma_t) * (V_t - V_0_pred)
            if eta > 0:
                noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma_t**2, min=0))
                V_t = V_t + noise_scale * torch.randn_like(V_t)
        else:
            V_t = V_t - sigma_t * eps

    # Structure only: take distances, not coords
    V_A_sample = V_t.squeeze(0).detach()                  # throwaway coords after building D_AA
    D_AA = torch.cdist(V_A_sample, V_A_sample)            # (a, a)
    # Build one global anchor frame from structure (MDS)
    a = D_AA.shape[0]
    J = torch.eye(a, device=device) - (1.0 / a) * torch.ones(a, a, device=device)
    B_A = -0.5 * (J @ (D_AA**2) @ J)
    X_A = uet.classical_mds(B_A, d_out=D_latent)          # (a, D)
    X_A = uet.canonicalize_coords(X_A)
    # Precompute norms for triangulation
    XA = X_A.to(device)
    XA_norm2 = (XA**2).sum(dim=1)                         # (a,)

    # Global anchor scale (p95 of upper-tri distances)
    iu, ju = torch.triu_indices(a, a, 1, device=device)
    anchor_scale_p95 = torch.quantile(D_AA[iu, ju], 0.95).detach()
    if DEBUG_FLAG:
        print(f"[anchors] a={a}  p95={float(anchor_scale_p95):.4f}")

    # -------- Reservoir for carry (sampling-only) --------
    reservoir_idx = anchor_idx.clone().cpu().tolist()
    reservoir_V   = [X_A.detach().cpu()]                  # use stitched anchor coords for carry
    if DEBUG_FLAG:
        print(f"[reservoir] init size={len(reservoir_idx)}  carry_k={CARRY_K}")

    # -------- Step 3/4: Process non-anchors in batches --------
    all_X = [X_A.detach().cpu()]
    order = [anchor_idx.tolist()]
    n_batches = (non_anchor_idx.numel() + batch_size - 1) // batch_size

    ZA = Z_all[anchor_idx].to(device)                     # (a, h) for anchor–Z matching

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        s = batch_idx * batch_size
        e = min(s + batch_size, non_anchor_idx.numel())
        idx_B = non_anchor_idx[s:e]                       # LongTensor
        Z_B = Z_all[idx_B].to(device)                     # (|B|, h)

        # --- Build set for sampling: anchors + optional carry + new cells ---
        carry_idx = []
        carry_coords = None
        kcarry = 0
        if len(reservoir_idx) > 0 and CARRY_K > 0:
            sel = np.random.choice(len(reservoir_idx),
                                   size=min(CARRY_K, len(reservoir_idx)),
                                   replace=False)
            carry_idx = [reservoir_idx[s] for s in sel]
            carry_coords = torch.cat(reservoir_V, dim=0)[sel]  # (kcarry, D) CPU
            kcarry = len(carry_idx)

            Z_carry = Z_all[carry_idx].to(device).unsqueeze(0) # (1, kcarry, h)
            Z_C = torch.cat([Z_A, Z_carry, Z_B.unsqueeze(0)], dim=1)
        else:
            Z_C = torch.cat([Z_A, Z_B.unsqueeze(0)], dim=1)

        mask_C = torch.ones(1, Z_C.shape[1], dtype=torch.bool, device=device)
        H_C = context_encoder(Z_C, mask_C)

        # --- Reverse diffusion: anchors+carry frozen, new noisy ---
        V_t = torch.empty(1, Z_C.shape[1], D_latent, device=device)
        V_t[:, :X_A.shape[0], :] = X_A                    # use stitched anchors in global frame
        if kcarry > 0:
            V_t[:, X_A.shape[0]:X_A.shape[0]+kcarry, :] = carry_coords.to(device)
        V_t[:, X_A.shape[0]+kcarry:, :] = (
            torch.randn(1, Z_C.shape[1] - X_A.shape[0] - kcarry, D_latent, device=device) * sigmas[0]
        )

        upd_mask = torch.ones_like(V_t)                   # 1 = update; 0 = freeze
        upd_mask[:, :X_A.shape[0]+kcarry, :] = 0.0

        for t_idx in range(n_timesteps_sample):
            sigma_t = sigmas[t_idx]
            t_norm = torch.tensor([[t_idx / float(n_timesteps_sample)]], device=device)

            H_null_C = torch.zeros_like(H_C)
            eps_uncond = score_net(V_t, t_norm, H_null_C, mask_C)
            eps_cond   = score_net(V_t, t_norm, H_C,        mask_C)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # masked EDM step (do not move anchors+carry)
            if t_idx < n_timesteps_sample - 1:
                sigma_next = sigmas[t_idx + 1]
                update = (sigma_t - sigma_next) * eps * upd_mask
            else:
                update = sigma_t * eps * upd_mask
            V_t = V_t - update

        # --- Structure harvesting: distances only ---
        V_all = V_t.squeeze(0)                            # (a + kcarry + |B|, D) throwaway coords
        V_new = V_all[X_A.shape[0] + kcarry:]             # (|B|, D)

        # Safety: per-batch scale alignment using anchor–anchor p95
        D_AA_batch = torch.cdist(V_all[:X_A.shape[0]], V_all[:X_A.shape[0]])
        # scale = (anchor_scale_p95 / (torch.quantile(D_AA_batch[iu, ju], 0.95) + 1e-8)).clamp(0.5, 2.0)
        scale = 1.0

        # distances to anchors (scaled)
        D_newA = torch.cdist(V_new, V_all[:X_A.shape[0]]) * scale  # (|B|, a)

        if DEBUG_FLAG and (batch_idx % 10 == 0):
            print(f"[batch {batch_idx+1}/{n_batches}] |B|={len(idx_B)} kcarry={kcarry} scale={float(scale):.3f}")

        # --- Triangulation in the global anchor frame (per cell) ---
        # Use Z-similarity to pick K_ANCH nearest anchors (robust)
        sim = F.normalize(Z_B, dim=1) @ F.normalize(ZA, dim=1).T  # (|B|, a)
        topk = torch.topk(sim, k=min(K_ANCH, ZA.shape[0]), dim=1).indices  # (|B|, k)

        X_B_list = []
        for j in range(D_newA.shape[0]):
            Sj = topk[j]                                     # (k,)
            # reference anchor: closest by predicted distance
            r_local = torch.argmin(D_newA[j, Sj]).item()
            r_idx = Sj[r_local].item()

            Xj = XA[Sj]                                      # (k, D)
            xr = XA[r_idx]                                   # (D,)
            Aj = 2.0 * (Xj - xr)                             # (k, D)

            dj  = D_newA[j, Sj]                              # (k,)
            djr = D_newA[j, r_idx]                           # scalar
            bj = (djr**2 - dj**2 + (XA_norm2[Sj] - XA_norm2[r_idx]))  # (k,)

            # Solve (A^T A + λ I) x = A^T b  (regularized least-squares)
            AtA = Aj.T @ Aj                                  # (D,D)
            rhs = Aj.T @ bj                                  # (D,)
            xj = torch.linalg.solve(AtA + L2_REG*torch.eye(D_latent, device=device), rhs)  # (D,)
            X_B_list.append(xj)

        X_B = torch.stack(X_B_list, dim=0).detach().cpu()    # (|B|, D)

        # --- Keep stitched coords; throw away sampled coords ---
        all_X.append(X_B)
        order.append(idx_B.tolist())

        # Update reservoir with STITCHED coords (global frame)
        reservoir_idx.extend(idx_B.cpu().tolist())
        reservoir_V.append(X_B)
        if len(reservoir_idx) > RES_CAP:
            keep = np.random.choice(len(reservoir_idx), size=RES_CAP, replace=False)
            bigV = torch.cat(reservoir_V, dim=0)
            reservoir_idx = [reservoir_idx[i] for i in keep]
            reservoir_V   = [bigV[keep]]

        # housekeeping
        del Z_B, Z_C, H_C, V_t, eps_uncond, eps_cond, eps, V_all, V_new, D_AA_batch, D_newA, sim, topk
        if device == 'cuda':
            torch.cuda.empty_cache()

    # -------- Step 5: Reassemble global coordinates, compute EDM --------
    if DEBUG_FLAG:
        print("\n[Step Z] Reassembling global structure...")
    X_full = torch.empty(n_sc, D_latent)
    X_full[anchor_idx] = all_X[0]
    for idxs, Xs in zip(order[1:], all_X[1:]):
        X_full[torch.tensor(idxs, dtype=torch.long)] = Xs
    X_full = X_full - X_full.mean(dim=0, keepdim=True)

    # Global distances and EDM projection
    Xd = X_full.to(device)
    D = torch.cdist(Xd, Xd)
    D_edm = uet.edm_project(D).detach().cpu()

    result = {'D_edm': D_edm}

    if return_coords:
        # For visualization, compute 2D MDS from the final EDM
        n = D_edm.shape[0]
        Jn = torch.eye(n) - torch.ones(n, n) / n
        B = -0.5 * (Jn @ (D_edm**2) @ Jn)
        coords = uet.classical_mds(B.to(device), d_out=2).detach().cpu()
        coords_canon = uet.canonicalize_coords(coords).detach().cpu()
        result['coords'] = coords
        result['coords_canon'] = coords_canon

    def _approx_upper_quantile(D_cpu: torch.Tensor, q: float = 0.95, max_pairs: int = 1_000_000) -> float:
        """
        Robust quantile on the upper triangle using <= max_pairs random pairs.
        Avoids materializing the full 50M-element vector.
        """
        n = D_cpu.shape[0]
        total_pairs = n * (n - 1) // 2
        k = min(max_pairs, max(1, total_pairs))

        # Sample random (i,j) with i<j
        # Generate a bit more than k and filter to i<j to avoid bias
        m = int(k * 1.3)
        i = torch.randint(0, n, (m,), device=D_cpu.device)
        j = torch.randint(0, n, (m,), device=D_cpu.device)
        keep = i < j
        i = i[keep][:k]
        j = j[keep][:k]

        vals = D_cpu[i, j]  # length <= k
        return torch.quantile(vals, q).item()

    if DEBUG_FLAG:
        try:
            p95 = _approx_upper_quantile(D_edm, q=0.95, max_pairs=1_000_000)
            print(f"[done] ||X||_std={float(X_full.std()):.3f}  D_p95≈{p95:.3f}  (structure-first stitching complete)")
        except Exception as e:
            print(f"[done] ||X||_std={float(X_full.std()):.3f}  (p95 skipped: {e})  (structure-first stitching complete)")


    return result