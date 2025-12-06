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
DEBUG = False #master switch for debug logging

# LOG_EVERY = 500 #per-batch prints
# SAVE_EVERY_EPOCH = 5 #extra plots/dumps

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
    Generator that produces V ∈ R ^{n x D} from context H.
    
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
        # V_centered = V - V.mean(dim=1, keepdim=True)
        
        # # Apply mask
        # V_centered = V_centered * mask.unsqueeze(-1).float()

        mask_f = mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mean = (V * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        V_centered = (V - mean) * mask_f.unsqueeze(-1)

        
        return V_centered
    
# ==============================================================================
# STAGE C: DIFFUSION SCORE NETWORK sψ
# ==============================================================================
import math

class DiffusionScoreNet(nn.Module):
    """
    Conditional denoiser for V_t → ε̂.
    
    Set-equivariant architecture with time embedding.
    VE SDE: sigma_min=0.01, sigma_max=50
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
        sc_feat_mode: str ='concat',
        use_st_dist_head: bool = True
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
        self.use_st_dist_head = use_st_dist_head

        #distance bias params
        if use_dist_bias:
            d_emb = 32
            self.E_bin = nn.Parameter(torch.randn(dist_bins, d_emb) / math.sqrt(d_emb))
            out_dim = 1 if dist_head_shared else n_heads
            self.W_bias = nn.Parameter(torch.randn(d_emb, out_dim) * 0.01)
            self.alpha_bias = nn.Parameter(torch.tensor(0.1))

        #ST distogram head for supervised bin prediction
        if use_st_dist_head:
            d_emb = 32
            self.st_dist_head = nn.Sequential(
                nn.Linear(d_emb, 64),
                nn.ReLU(),
                nn.Linear(64, dist_bins)
            )

        #buffer for data=driven bin edges 
        self.register_buffer('st_dist_bin_edges', None)

        #self conditioning MLP if needed
        if self.self_conditioning and sc_feat_mode == 'mlp':
            self.sc_mlp = nn.Sequential(
                nn.Linear(D_latent, c_dim // 2),
                nn.ReLU(),
                nn.Linear(c_dim // 2, D_latent)
            )

        #update input projection dimension
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
           attn_cached: dict = None, return_dist_aux: bool = False) -> torch.Tensor:
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
        
        if self.use_canonicalize:
            V_in, _ = uet.center_only(V_t, mask)
            if self_cond is not None:
                self_cond_canon, _ = uet.center_only(self_cond, mask)
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
        bin_ids = None
        bin_embeddings = None

        if self.use_dist_bias:
            if attn_cached is not None and 'bias' in attn_cached:
                attn_bias = attn_cached['bias']
                bin_ids = attn_cached.get('bin_ids', None)
                bin_embeddings = attn_cached.get('bin_embeddings', None)
            else:
                D2 = uet.pairwise_dist2(V_in, mask)
                attn_bias, _, bin_ids = uet.make_distance_bias(
                    D2, mask, 
                    n_bins=self.dist_bins,
                    d_emb=32,
                    share_across_heads=self.dist_head_shared,
                    E_bin=self.E_bin,
                    W=self.W_bias,
                    alpha_bias=self.alpha_bias,
                    device=V_t.device,
                    bin_edges=  self.st_dist_bin_edges
                )

                #get bin embeddings for distogram head
                if self.use_st_dist_head:
                    bin_embeddings = self.E_bin[bin_ids]

                if attn_cached is not None:
                    attn_cached['bias'] = attn_bias
                    attn_cached['bin_ids'] = bin_ids
                    attn_cached['bin_embeddings'] = bin_embeddings

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

        #return distogram auxiliary outputs if requested
        if return_dist_aux and self.st_dist_head and bin_embeddings is not None:
            dist_logits = self.st_dist_head(bin_embeddings)
            return eps_hat, {'dist_logits': dist_logits, 'bin_ids': bin_ids}
        
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

# ==============================================================================
# STAGE C 2.0: SIMPLIFIED ST-ONLY TRAINING WITH GRAPH-AWARE LOSSES
# ==============================================================================

def train_stageC_st_only(
    context_encoder: 'SetEncoderContext',
    generator: 'MetricSetGenerator',
    score_net: 'DiffusionScoreNet',
    st_dataset: 'STSetDataset',
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    n_timesteps: int = 500,
    sigma_min: float = 0.01,
    sigma_max: float = 5.0,
    device: str = 'cuda',
    outf: str = 'output',
    fabric: Optional['Fabric'] = None,
    precision: str = '16-mixed',
    # Loss weights
    w_score: float = 1.0,
    w_edge: float = 1.0,
    w_repel: float = 0.1,
    # Repulsion config
    r_min: float = 0.02,  # Will be auto-computed if None
    # Early stopping
    enable_early_stop: bool = True,
    early_stop_patience: int = 10,
    early_stop_min_epochs: int = 20,
):
    """
    Stage C 2.0: ST-only training with graph-aware losses.
    
    Losses:
        1. Score loss (EDM-weighted denoising)
        2. kNN edge distance loss (index-aware geometry)
        3. Repulsion loss (prevent collapse)
    
    This replaces the histogram-based losses that allowed ring degeneracy.
    """
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_minisets

    print("\n" + "="*70)
    print("STAGE C 2.0: ST-ONLY TRAINING (GRAPH-AWARE LOSSES)")
    print("="*70)
    print(f"Losses: score (w={w_score}) + knn_edge (w={w_edge}) + repel (w={w_repel})")
    print(f"Epochs: {n_epochs}, Batch: {batch_size}, LR: {lr}")
    print("="*70 + "\n")

    # ========== SETUP ==========
    context_encoder = context_encoder.to(device).train()
    generator = generator.to(device).train()
    score_net = score_net.to(device).train()
    
    params = (
        list(context_encoder.parameters()) +
        list(generator.parameters()) +
        list(score_net.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    if fabric is not None:
        context_encoder, generator, score_net, optimizer = fabric.setup(
            context_encoder, generator, score_net, optimizer
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Sigma schedule (VE SDE)
    sigmas = torch.exp(torch.linspace(
        np.log(sigma_min), np.log(sigma_max), n_timesteps, device=device
    ))
    print(f"[Sigma] min={sigmas.min().item():.4f}, max={sigmas.max().item():.4f}")

    # DataLoader
    st_loader = DataLoader(
        st_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_minisets,
        num_workers=0,
        pin_memory=False
    )
    print(f"[DataLoader] {len(st_dataset)} samples, {len(st_loader)} batches/epoch")
    
    if fabric is not None:
        st_loader = fabric.setup_dataloaders(st_loader)

    # AMP setup
    use_fp16 = (precision == '16-mixed')
    use_bf16 = torch.cuda.is_bf16_supported() and precision == 'bf16-mixed'
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Loss modules
    loss_knn_edge = uet.KNNEdgeDistanceLoss(max_edges_per_sample=1024)
    loss_repel = uet.RepulsionLoss(r_min=r_min, max_pairs=2000)

    # Compute sigma_data from ST data
    print("Computing sigma_data from ST statistics...")
    with torch.no_grad():
        sample_stds = []
        it = iter(st_loader)
        for _ in range(min(10, len(st_loader))):
            batch = next(it, None)
            if batch is None:
                break
            G_batch = batch['G_target'].to(device)
            for i in range(min(4, G_batch.shape[0])):
                V_temp = uet.factor_from_gram(G_batch[i], score_net.D_latent)
                sample_stds.append(V_temp.std().item())
        sigma_data = float(np.median(sample_stds)) if sample_stds else 1.0
    print(f"[sigma_data] = {sigma_data:.4f}")

    # Auto-compute r_min from ST nearest-neighbor distances
    if r_min is None or r_min <= 0:
        print("Computing r_min from ST NN distances...")
        with torch.no_grad():
            nn_dists = []
            it = iter(st_loader)
            for _ in range(min(20, len(st_loader))):
                batch = next(it, None)
                if batch is None:
                    break
                V_target = batch['V_target'].to(device)
                mask = batch['mask'].to(device)
                for b in range(V_target.shape[0]):
                    m = mask[b]
                    if m.sum() < 2:
                        continue
                    V = V_target[b, m]
                    D = torch.cdist(V, V)
                    D.fill_diagonal_(float('inf'))
                    nn_dist = D.min(dim=1)[0]
                    nn_dists.append(nn_dist.cpu())
            if nn_dists:
                all_nn = torch.cat(nn_dists)
                r_min = float(all_nn.quantile(0.15).item())
            else:
                r_min = 0.02
        print(f"[r_min] = {r_min:.4f} (15th percentile of NN distances)")
        loss_repel.r_min = r_min

    # Output directory
    os.makedirs(outf, exist_ok=True)
    
    # CFG config
    p_uncond = 0.25
    p_sc = 0.5  # self-conditioning probability

    # History tracking
    history = {
        'epoch': [],
        'epoch_avg': {'total': [], 'score': [], 'edge': [], 'repel': []}
    }
    
    best_loss = float('inf')
    patience_counter = 0
    D_latent = score_net.D_latent

    # ========== TRAINING LOOP ==========
    for epoch in range(n_epochs):
        epoch_losses = {'total': 0.0, 'score': 0.0, 'edge': 0.0, 'repel': 0.0}
        n_batches = 0
        
        pbar = tqdm(st_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        
        for batch in pbar:
            # ===== LOAD BATCH =====
            Z_set = batch['Z_set'].to(device)
            mask = batch['mask'].to(device)
            V_target = batch['V_target'].to(device)
            knn_edge_index = batch['knn_edge_index']  # list of tensors
            knn_edge_dists = batch['knn_edge_dists']  # list of tensors
            
            B = Z_set.shape[0]
            
            # ===== CONTEXT ENCODING (PRESERVED) =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_fp16):
                H = context_encoder(Z_set, mask)

                # ===== SAMPLE NOISE LEVEL (PRESERVED - quadratic bias) =====
                u = torch.rand(B, device=device)
                t_cont = (u ** 2) * (n_timesteps - 1)
                t_idx = t_cont.long()
                t_norm = t_cont / (n_timesteps - 1)
                sigma_t = sigmas[t_idx].view(-1, 1, 1)
                
                # ===== GENERATE V_0 (PRESERVED) =====
                V_0 = generator(H, mask)
                
                # ===== FORWARD DIFFUSION (PRESERVED) =====
                eps = torch.randn_like(V_0)
                V_t = V_0 + sigma_t * eps
                V_t = V_t * mask.unsqueeze(-1).float()

                # ===== CFG DROPOUT (PRESERVED) =====
                drop_mask = (torch.rand(B, device=device) < p_uncond).float().view(-1, 1, 1)
                H_train = H * (1 - drop_mask)
                
                # ===== SELF-CONDITIONING (PRESERVED) =====
                use_self_cond = torch.rand(1, device=device).item() < p_sc
                
                if use_self_cond and score_net.self_conditioning:
                    with torch.no_grad():
                        eps_hat_0 = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None)
                        V_pred_0 = V_t - sigma_t * eps_hat_0
                    eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=V_pred_0)
                else:
                    eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None)

                # Handle tuple return (if score_net returns aux outputs)
                if isinstance(eps_pred, tuple):
                    eps_pred = eps_pred[0]

            # ===== SCORE LOSS (PRESERVED - EDM weighting) =====
            with torch.autocast(device_type='cuda', enabled=False):
                sigma_t_fp32 = sigma_t.float()
                eps_pred_fp32 = eps_pred.float()
                eps_fp32 = eps.float()
                mask_fp32 = mask.float()
                
                sigma_t_squeezed = sigma_t_fp32.squeeze(-1)
                if sigma_t_squeezed.dim() == 1:
                    sigma_t_squeezed = sigma_t_squeezed.unsqueeze(-1)
                
                # EDM weight
                w = (sigma_t_squeezed**2 + sigma_data**2) / (sigma_data**2)
                w = w.clamp(max=100.0)
                
                err2 = (eps_pred_fp32 - eps_fp32).pow(2).mean(dim=2)
                if w.shape[-1] == 1:
                    w = w.expand_as(err2)
                
                L_score = (w * err2 * mask_fp32).sum() / mask_fp32.sum().clamp(min=1)

            # ===== DENOISE TO GET V_HAT (PRESERVED) =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_fp16):
                V_hat = V_t - sigma_t * eps_pred
                V_hat = V_hat * mask.unsqueeze(-1).float()
            
            # ===== NEW GEOMETRY LOSSES =====
            with torch.autocast(device_type='cuda', enabled=False):
                V_hat_fp32 = V_hat.float()
                
                # kNN Edge Distance Loss (NEW - index-aware)
                L_edge = loss_knn_edge(
                    V_pred=V_hat_fp32,
                    mask=mask,
                    knn_edge_index_list=knn_edge_index,
                    knn_edge_dists_list=knn_edge_dists
                )
                
                # Repulsion Loss (NEW - prevent collapse)
                L_repel = loss_repel(V_pred=V_hat_fp32, mask=mask)

            # ===== TOTAL LOSS =====
            L_total = w_score * L_score + w_edge * L_edge + w_repel * L_repel
            
            # ===== BACKWARD (PRESERVED) =====
            optimizer.zero_grad(set_to_none=True)
            
            if fabric is not None:
                fabric.backward(scaler.scale(L_total))
            else:
                scaler.scale(L_total).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # ===== LOGGING =====
            epoch_losses['total'] += L_total.item()
            epoch_losses['score'] += L_score.item()
            epoch_losses['edge'] += L_edge.item()
            epoch_losses['repel'] += L_repel.item()
            n_batches += 1
            
            pbar.set_postfix({
                'score': f"{L_score.item():.4f}",
                'edge': f"{L_edge.item():.4f}",
                'repel': f"{L_repel.item():.4f}"
            })
        
        scheduler.step()

        # ===== EPOCH SUMMARY =====
        avg_total = epoch_losses['total'] / max(n_batches, 1)
        avg_score = epoch_losses['score'] / max(n_batches, 1)
        avg_edge = epoch_losses['edge'] / max(n_batches, 1)
        avg_repel = epoch_losses['repel'] / max(n_batches, 1)
        
        history['epoch'].append(epoch + 1)
        history['epoch_avg']['total'].append(avg_total)
        history['epoch_avg']['score'].append(avg_score)
        history['epoch_avg']['edge'].append(avg_edge)
        history['epoch_avg']['repel'].append(avg_repel)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}] total={avg_total:.4f} | "
                  f"score={avg_score:.4f} | edge={avg_edge:.4f} | repel={avg_repel:.4f}")
        
        # ===== EARLY STOPPING =====
        if enable_early_stop and epoch >= early_stop_min_epochs:
            if avg_total < best_loss * 0.99:  # 1% improvement threshold
                best_loss = avg_total
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"\n[Early Stop] No improvement for {early_stop_patience} epochs. Stopping.")
                break
    
    print("\n" + "="*70)
    print("STAGE C 2.0 TRAINING COMPLETE")
    print("="*70)
    
    return history

# ==============================================================================
# SC ENCODER FINE-TUNING (FROZEN GEOMETRY PRIOR)
# ==============================================================================

def finetune_encoder_on_sc(
    encoder: 'SharedEncoder',
    context_encoder: 'SetEncoderContext',
    generator: 'MetricSetGenerator',
    score_net: 'DiffusionScoreNet',
    sc_gene_expr: torch.Tensor,  # (n_sc, n_genes) - RAW expression
    sc_dataset: 'SCSetDataset',
    D_st_reference: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 3e-5,
    device: str = 'cuda',
    outf: str = 'output',
    fabric: Optional['Fabric'] = None,
    precision: str = '16-mixed',
    # Loss weights
    w_overlap: float = 1.0,
    w_geom: float = 1.0,
    w_dim: float = 0.1,
):
    """
    SC Encoder Fine-tuning with Frozen Geometry Prior.
    
    Generator and score_net weights are FROZEN but used as differentiable 
    functions so gradients flow through them into encoder/context_encoder.
    
    Losses:
        1. Overlap consistency (shared cells in A/B land at same location)
        2. ST distance distribution matching
        3. Intrinsic dimension regularizer
    """
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_sc_minisets
    
    print("\n" + "="*70)
    print("SC ENCODER FINE-TUNING (FROZEN GEOMETRY PRIOR)")
    print("="*70)
    print(f"Losses: overlap (w={w_overlap}) + geom (w={w_geom}) + dim (w={w_dim})")
    print(f"Epochs: {n_epochs}, Batch: {batch_size}, LR: {lr}")
    print("="*70 + "\n")
    
    # ========== FREEZE GEOMETRY PRIOR (weights only, still differentiable) ==========
    generator.to(device).train()
    score_net.to(device).train()
    for p in generator.parameters():
        p.requires_grad = False
    for p in score_net.parameters():
        p.requires_grad = False
    print("[Frozen weights] generator, score_net (still differentiable)")
    
    # ========== TRAINABLE COMPONENTS ==========
    encoder = encoder.to(device).train()
    context_encoder = context_encoder.to(device).train()
    
    trainable_params = (
        list(encoder.parameters()) +
        list(context_encoder.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    if fabric is not None:
        encoder, context_encoder, generator, optimizer = fabric.setup(
            encoder, context_encoder, generator, optimizer
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Keep sc_gene_expr on device for re-encoding
    sc_gene_expr = sc_gene_expr.to(device)
    
    # DataLoader
    sc_loader = DataLoader(
        sc_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sc_minisets,
        num_workers=0,
        pin_memory=False
    )
    print(f"[SC DataLoader] {len(sc_dataset)} samples, {len(sc_loader)} batches/epoch")
    
    if fabric is not None:
        sc_loader = fabric.setup_dataloaders(sc_loader)
    
    # AMP setup
    use_fp16 = (precision == '16-mixed')
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    # Loss modules
    loss_geom_fn = uet.DistanceDistributionLoss(n_samples=512)
    loss_dim_fn = uet.IntrinsicDimensionLoss(k_neighbors=20, target_dim=2.0)
    
    # Move ST reference to device
    D_st_reference = D_st_reference.to(device)
    
    os.makedirs(outf, exist_ok=True)
    
    # History
    history = {
        'epoch': [],
        'epoch_avg': {'total': [], 'overlap': [], 'geom': [], 'dim': []}
    }
    
    # ========== TRAINING LOOP ==========
    for epoch in range(n_epochs):
        epoch_losses = {'total': 0.0, 'overlap': 0.0, 'geom': 0.0, 'dim': 0.0}
        n_batches = 0
        
        pbar = tqdm(sc_loader, desc=f"SC Epoch {epoch+1}/{n_epochs}", leave=False)
        
        for batch in pbar:
            # ===== GET DATA FROM BATCH =====
            # collate_sc_minisets returns sc_global_indices: (2B, n_max) with -1 for padding
            sc_global_indices = batch['sc_global_indices'].to(device)  # (2B, n_max)
            mask = batch['mask'].to(device)  # (2B, n_max)
            n_vec = batch['n'].to(device)  # (2B,)
            
            # Overlap info
            pair_idxA = batch.get('pair_idxA', None)  # (P,) - indices into batch dim
            pair_idxB = batch.get('pair_idxB', None)  # (P,)
            shared_A_idx = batch.get('shared_A_idx', None)  # (P, Kmax) local positions
            shared_B_idx = batch.get('shared_B_idx', None)  # (P, Kmax) local positions
            
            B_total = sc_global_indices.shape[0]  # 2B (A/B interleaved)
            n_max = sc_global_indices.shape[1]
            
            # ===== RE-ENCODE FROM RAW EXPRESSION (encoder in gradient graph) =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_fp16):
                # Build Z_set by re-encoding each cell from raw expression
                h_dim = encoder.n_embedding[-1]
                Z_set = torch.zeros(B_total, n_max, h_dim, device=device)
                
                for b in range(B_total):
                    n_b = n_vec[b].item()
                    if n_b == 0:
                        continue
                    # Get global indices for this set
                    global_idx = sc_global_indices[b, :n_b]  # (n_b,)
                    # Index into raw expression and re-encode
                    expr_b = sc_gene_expr[global_idx]  # (n_b, n_genes)
                    Z_b = encoder(expr_b)  # (n_b, h_dim) - ENCODER IS USED HERE
                    Z_set[b, :n_b] = Z_b
                
                # ===== CONTEXT ENCODING =====
                H = context_encoder(Z_set, mask)
                
                # ===== GENERATE COORDS VIA FROZEN PRIOR (no torch.no_grad!) =====
                # Weights frozen but graph is differentiable
                V = generator(H, mask)  # (B_total, n_max, D_latent)
            
            # ===== COMPUTE ALL LOSSES ON SAME V =====
            with torch.autocast(device_type='cuda', enabled=False):
                V_fp32 = V.float()
                
                # ----- OVERLAP CONSISTENCY LOSS -----
                L_overlap = torch.tensor(0.0, device=device, requires_grad=True)
                
                if pair_idxA is not None and shared_A_idx is not None:
                    P = pair_idxA.shape[0]
                    overlap_losses = []
                    
                    for p in range(P):
                        iA = pair_idxA[p].item()
                        iB = pair_idxB[p].item()
                        sA = shared_A_idx[p].to(device)  # local positions in set A
                        sB = shared_B_idx[p].to(device)  # local positions in set B
                        
                        # Filter valid (non-padded) shared indices
                        valid_mask = (sA >= 0) & (sB >= 0)
                        if valid_mask.sum() < 2:
                            continue
                        
                        sA_valid = sA[valid_mask].clamp(0, n_vec[iA].item() - 1)
                        sB_valid = sB[valid_mask].clamp(0, n_vec[iB].item() - 1)
                        
                        coords_A = V_fp32[iA, sA_valid]  # (K, D_latent)
                        coords_B = V_fp32[iB, sB_valid]  # (K, D_latent)
                        
                        # L2 distance between same cells in A and B
                        overlap_losses.append(((coords_A - coords_B) ** 2).sum(dim=-1).mean())
                    
                    if overlap_losses:
                        L_overlap = torch.stack(overlap_losses).mean()
                
                # ----- DISTANCE DISTRIBUTION LOSS -----
                geom_losses = []
                for b in range(B_total):
                    m = mask[b]
                    if m.sum() < 3:
                        continue
                    coords_b = V_fp32[b, m]
                    geom_losses.append(loss_geom_fn(coords_b, D_st_reference))
                
                L_geom = torch.stack(geom_losses).mean() if geom_losses else torch.tensor(0.0, device=device, requires_grad=True)
                
                # ----- INTRINSIC DIMENSION LOSS -----
                L_dim = loss_dim_fn(V_pred=V_fp32, V_target=None, mask=mask, use_target=False)
            
            # ===== TOTAL LOSS (all three terms) =====
            L_total = w_overlap * L_overlap + w_geom * L_geom + w_dim * L_dim
            
            # ===== BACKWARD =====
            optimizer.zero_grad(set_to_none=True)
            
            if fabric is not None:
                fabric.backward(scaler.scale(L_total))
            else:
                scaler.scale(L_total).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # ===== LOGGING (same losses used for training) =====
            epoch_losses['total'] += L_total.item()
            epoch_losses['overlap'] += L_overlap.item()
            epoch_losses['geom'] += L_geom.item()
            epoch_losses['dim'] += L_dim.item()
            n_batches += 1
            
            pbar.set_postfix({
                'overlap': f"{L_overlap.item():.4f}",
                'geom': f"{L_geom.item():.4f}",
                'dim': f"{L_dim.item():.4f}",
            })
        
        scheduler.step()
        
        # ===== EPOCH SUMMARY =====
        avg_total = epoch_losses['total'] / max(n_batches, 1)
        avg_overlap = epoch_losses['overlap'] / max(n_batches, 1)
        avg_geom = epoch_losses['geom'] / max(n_batches, 1)
        avg_dim = epoch_losses['dim'] / max(n_batches, 1)
        
        history['epoch'].append(epoch + 1)
        history['epoch_avg']['total'].append(avg_total)
        history['epoch_avg']['overlap'].append(avg_overlap)
        history['epoch_avg']['geom'].append(avg_geom)
        history['epoch_avg']['dim'].append(avg_dim)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[SC Epoch {epoch+1:3d}] total={avg_total:.4f} | "
                  f"overlap={avg_overlap:.4f} | geom={avg_geom:.4f} | dim={avg_dim:.4f}")
    
    # Unfreeze for future use
    for p in generator.parameters():
        p.requires_grad = True
    for p in score_net.parameters():
        p.requires_grad = True
    
    print("\n" + "="*70)
    print("SC ENCODER FINE-TUNING COMPLETE")
    print("="*70)
    
    return history
            


# ==============================================================================
# STAGE D: SC INFERENCE (PATCH-BASED GLOBAL ALIGNMENT)
# ==============================================================================

def sample_sc_edm_patchwise(
    sc_gene_expr: torch.Tensor,
    encoder: "SharedEncoder",
    context_encoder: "SetEncoderContext",
    score_net: "DiffusionScoreNet",
    target_st_p95: Optional[float] = None,
    n_timesteps_sample: int = 160,
    sigma_min: float = 0.01,
    sigma_max: float = 5.0,
    guidance_scale: float = 8.0,
    eta: float = 0.0,
    device: str = "cuda",
    # patch / coverage knobs
    patch_size: int = 384,
    coverage_per_cell: float = 4.0,
    # global alignment knobs
    n_align_iters: int = 10,
    # misc
    return_coords: bool = True,
    DEBUG_FLAG: bool = True,
    DEBUG_EVERY: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Stage D: Patch-based SC inference via global alignment.

    1) Encode all SC cells -> Z.
    2) Build overlapping SC patches using k-NN in Z-space.
    3) For each patch:
       - Run diffusion sampling on the patch (all points free),
       - Canonicalize sample to match training invariances.
    4) Align all patch samples into a single global embedding X via
       alternating Procrustes (similarity transforms per patch).
    5) Compute global EDM from X, optionally rescale to match ST scale.
    6) Optionally run 2D MDS for visualization.
    """

    import numpy as np
    import torch.nn.functional as F
    from tqdm import tqdm
    import utils_et as uet
    from typing import List, Dict



    print(f"[PATCHWISE] THIS IS THE NEW LOGIC BROTHER Running on device={device}, starting inference...", flush=True)

    encoder.eval()
    context_encoder.eval()
    score_net.eval()

    n_sc = sc_gene_expr.shape[0]
    D_latent = score_net.D_latent
    patch_size = int(min(patch_size, n_sc))

    if DEBUG_FLAG:
        print("\n" + "=" * 72)
        print("STAGE D — PATCH-BASED GLOBAL SC INFERENCE")
        print("=" * 72)
        print(f"[cfg] n_sc={n_sc}  patch_size≈{patch_size}  coverage_per_cell={coverage_per_cell}")
        print(f"[cfg] timesteps={n_timesteps_sample}  D_latent={D_latent}")
        print(f"[cfg] sigma_min={sigma_min}  sigma_max={sigma_max}  guidance_scale={guidance_scale}")
        print(f"[align] n_align_iters={n_align_iters}")
        if target_st_p95 is not None:
            print(f"[scale] target_st_p95={target_st_p95:.4f}")

    # ------------------------------------------------------------------
    # 1) Encode all SC cells into Z space
    # ------------------------------------------------------------------
    encode_bs = 1024
    Z_chunks = []
    for i in range(0, n_sc, encode_bs):
        z = encoder(sc_gene_expr[i:i + encode_bs].to(device)).detach().cpu()
        Z_chunks.append(z)
    Z_all = torch.cat(Z_chunks, dim=0)          # (N, h)
    del Z_chunks
    if DEBUG_FLAG:
        print(f"[ENC] Z_all shape={tuple(Z_all.shape)}")

    # ------------------------------------------------------------------
    # 2) Build k-NN index in Z-space for patch construction
    # ------------------------------------------------------------------
    K_nbrs = patch_size
    nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs)  # (N, K_nbrs)

    # ------------------------------------------------------------------
    # 3) Define overlapping patches S_k
    # ------------------------------------------------------------------
    total_slots = int(np.ceil(coverage_per_cell * n_sc))
    n_patches_est = max(int(np.ceil(total_slots / patch_size)), 1)

    if DEBUG_FLAG:
        print(f"[PATCH] estimated n_patches={n_patches_est} (total_slots={total_slots})")

    centers = torch.randint(low=0, high=n_sc, size=(n_patches_est,), dtype=torch.long)

    patch_indices: List[torch.Tensor] = []
    memberships: List[List[int]] = [[] for _ in range(n_sc)]

    # First pass: random patches around centers
    for k, c in enumerate(centers.tolist()):
        S_k = nbr_idx[c, :patch_size]
        S_k = torch.unique(S_k, sorted=False)
        patch_indices.append(S_k)
        for idx in S_k.tolist():
            memberships[idx].append(k)

    # Ensure every cell appears in at least one patch
    for i in range(n_sc):
        if len(memberships[i]) == 0:
            k = len(patch_indices)
            S_k = nbr_idx[i, :patch_size]
            S_k = torch.unique(S_k, sorted=False)
            patch_indices.append(S_k)
            memberships[i].append(k)
            for idx in S_k.tolist():
                memberships[idx].append(k)

    K = len(patch_indices)

    if DEBUG_FLAG:
        cover_counts = torch.tensor([len(memberships[i]) for i in range(n_sc)], dtype=torch.float32)
        patch_sizes = torch.tensor([len(S_k) for S_k in patch_indices], dtype=torch.float32)
        print(f"[PATCH] final n_patches={K}")
        print(f"[PATCH] per-cell coverage: "
              f"min={cover_counts.min().item():.1f} "
              f"p25={cover_counts.quantile(0.25).item():.1f} "
              f"p50={cover_counts.quantile(0.50).item():.1f} "
              f"p75={cover_counts.quantile(0.75).item():.1f} "
              f"max={cover_counts.max().item():.1f}")
        print(f"[PATCH] patch sizes: "
              f"min={patch_sizes.min().item():.0f} "
              f"p25={patch_sizes.quantile(0.25).item():.0f} "
              f"p50={patch_sizes.quantile(0.50).item():.0f} "
              f"p75={patch_sizes.quantile(0.75).item():.0f} "
              f"max={patch_sizes.max().item():.0f}")

    # ------------------------------------------------------------------
    # 4) For each patch: run diffusion sampling (all points free),
    #    then canonicalize to match training.
    # ------------------------------------------------------------------
    sigmas = torch.exp(torch.linspace(
        torch.log(torch.tensor(sigma_max, device=device)),
        torch.log(torch.tensor(sigma_min, device=device)),
        n_timesteps_sample,
        device=device,
    ))  # (T,)

    patch_coords: List[torch.Tensor] = []

    if DEBUG_FLAG:
        print("\n[STEP] Sampling local geometries for patches...")

    with torch.no_grad():
        for k in tqdm(range(K), desc="Sampling patches"):
            S_k = patch_indices[k]
            m_k = S_k.numel()
            Z_k = Z_all[S_k].to(device)         # (m_k, h)

            Z_k_batched = Z_k.unsqueeze(0)      # (1, m_k, h)
            mask_k = torch.ones(1, m_k, dtype=torch.bool, device=device)
            H_k = context_encoder(Z_k_batched, mask_k)

            V_t = torch.randn(1, m_k, D_latent, device=device) * sigmas[0]

            for t_idx in range(n_timesteps_sample):
                sigma_t = sigmas[t_idx]
                t_norm = torch.tensor([[t_idx / float(n_timesteps_sample - 1)]],
                                      device=device)

                H_null = torch.zeros_like(H_k)
                eps_uncond = score_net(V_t, t_norm, H_null, mask_k)
                eps_cond   = score_net(V_t, t_norm, H_k,    mask_k)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                # Extra debug on first patch
                if DEBUG_FLAG and k == 0 and t_idx < 3:
                    du = eps_uncond.norm(dim=[1, 2]).mean().item()
                    dc = eps_cond.norm(dim=[1, 2]).mean().item()
                    diff = (eps_cond - eps_uncond).norm(dim=[1, 2]).mean().item()
                    ratio = diff / (du + 1e-8)
                    print(f"  [PATCH0] t={t_idx:3d} sigma={float(sigma_t):.4f} "
                          f"||eps_u||={du:.3f} ||eps_c||={dc:.3f} "
                          f"||diff||={diff:.3f} CFG_ratio={ratio:.3f}")

                if t_idx < n_timesteps_sample - 1:
                    sigma_next = sigmas[t_idx + 1]
                    V_0_pred = V_t - sigma_t * eps
                    V_t = V_0_pred + (sigma_next / sigma_t) * (V_t - V_0_pred)
                    if eta > 0:
                        noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma_t**2, min=0))
                        V_t = V_t + noise_scale * torch.randn_like(V_t)
                else:
                    V_t = V_t - sigma_t * eps


            # if DEBUG_FLAG and (k % max(1, K // 5) == 0):
            #     rms = V_canon.pow(2).mean().sqrt().item()
            # NEW: Only center, do NOT apply unit RMS (matches training)
            V_final = V_t.squeeze(0)  # (m_k, D)
            V_centered = V_final - V_final.mean(dim=0, keepdim=True)
            patch_coords.append(V_centered.detach().cpu())

            if DEBUG_FLAG and (k % max(1, K // 5) == 0):
                rms = V_centered.pow(2).mean().sqrt().item()
                print(f"  [PATCH {k}/{K}] RMS={rms:.3f} (centered, natural scale)")

                # mean_norm = V_canon.mean(dim=0).norm().item()
                mean_norm = V_centered.mean(dim=0).norm().item()
                print(f"[PATCH] k={k}/{K} m_k={m_k} "
                      f"coords_rms={rms:.3f} center_norm={mean_norm:.3e}")

            # del Z_k, Z_k_batched, H_k, V_t, eps_uncond, eps_cond, eps, V_final, V_canon
            del Z_k, Z_k_batched, H_k, V_t, eps_uncond, eps_cond, eps, V_final

            if 'cuda' in device:
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5) Global alignment: alternately solve patch transforms and X
    # ------------------------------------------------------------------
    if DEBUG_FLAG:
        print("\n[ALIGN] Starting global alignment...")

    # 5.1 Initialize X by averaging patch coords per cell
    X_global = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
    W_global = torch.zeros(n_sc, 1, dtype=torch.float32, device=device)

    if DEBUG_FLAG:
        print("\n[ALIGN] Stitching with centrality weighting...")

    for k in range(K):
        # Global indices for this patch
        S_k = patch_indices[k].to(device)              # (m_k,)
        V_k = patch_coords[k].to(device)               # (m_k, D)

        # --- centrality weights in patch coordinates ---
        # center of patch in its local frame
        center_k = V_k.mean(dim=0, keepdim=True)       # (1, D)
        # distance of each point from center
        dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
        max_d = dists.max().clamp_min(1e-6)
        # linear taper: 1.0 at center, ~0.2 at edge, clamped at 0.01
        weights_k = 1.0 - (dists / (max_d * 1.2))
        weights_k = weights_k.clamp(min=0.01)          # (m_k, 1)
        # -----------------------------------------------

        # accumulate weighted coords and total weight
        X_global.index_add_(0, S_k, V_k * weights_k)   # (N,D) += (m_k,D)
        W_global.index_add_(0, S_k, weights_k)         # (N,1) += (m_k,1)

    # normalize by total weight where seen
    mask_seen = W_global.squeeze(-1) > 0
    X_global[mask_seen] /= W_global[mask_seen]

    # global recentering (keep learned scale)
    X_global = X_global - X_global.mean(dim=0, keepdim=True)
    rms = X_global.pow(2).mean().sqrt().item()
    if DEBUG_FLAG:
        print(f"[ALIGN] Init X_global: coords_rms={rms:.3f} (centrality-weighted)")

    # 5.2 Alternating Procrustes alignment (SIMPLIFIED: fixed scale, single iteration)
    s_global = 1.0
    
    print(f"\n[ALIGN] Using FIXED global scale s_global={s_global} (no dynamic scaling)")
    print(f"[ALIGN] Running simplified alignment with {n_align_iters} iteration(s)...")
    
    for it in range(n_align_iters):
        if DEBUG_FLAG:
            print(f"\n[ALIGN] Iteration {it + 1}/{n_align_iters}")

        R_list: List[torch.Tensor] = []
        t_list: List[torch.Tensor] = []
        
        # For global alignment loss tracking
        per_patch_mse = []

        # ======================================================================
        # Step A: Compute rotations (NO scale accumulation)
        # ======================================================================
        for k in range(K):
            S_k = patch_indices[k]
            V_k = patch_coords[k].to(X_global.device)   # (m_k, D)
            X_k = X_global[S_k]                         # (m_k, D)
            m_k = V_k.shape[0]

            # Centrality weights (same as Step B initialization)
            center_k = V_k.mean(dim=0, keepdim=True)       # (1, D)
            dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)          # (m_k, 1)

            # Weighted centroids
            w_sum = weights_k.sum()
            mu_X = (weights_k * X_k).sum(dim=0, keepdim=True) / w_sum
            mu_V = (weights_k * V_k).sum(dim=0, keepdim=True) / w_sum
            
            # Center
            Xc = X_k - mu_X
            Vc = V_k - mu_V

            # Weighted cross-covariance
            C = (Xc.T * weights_k.squeeze(-1)) @ Vc    # (D, D)
            
            # SVD for rotation
            U, S_vals, Vh = torch.linalg.svd(C, full_matrices=False)
            R_k = U @ Vh
            if torch.det(R_k) < 0:
                U[:, -1] *= -1
                R_k = U @ Vh

            # NO numerator/denominator accumulation - scale is fixed!
            R_list.append(R_k)

        # ======================================================================
        # NO GLOBAL SCALE RECOMPUTATION - use fixed s_global = 1.0
        # ======================================================================
        if DEBUG_FLAG and it == 0:
            print(f"[ALIGN] Using FIXED s_global={s_global} (not recomputed from patches)")

        # ======================================================================
        # Step A (cont.): Compute translations and track alignment error
        # ======================================================================
        for k in range(K):
            S_k = patch_indices[k]
            V_k = patch_coords[k].to(X_global.device)
            X_k = X_global[S_k]
            R_k = R_list[k]

            # Recompute centrality weights (or cache from above)
            center_k = V_k.mean(dim=0, keepdim=True)
            dists = torch.norm(V_k - center_k, dim=1, keepdim=True)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)

            w_sum = weights_k.sum()
            mu_X = (weights_k * X_k).sum(dim=0, keepdim=True) / w_sum
            mu_V = (weights_k * V_k).sum(dim=0, keepdim=True) / w_sum

            # Translation using FIXED s_global = 1.0
            t_k = (mu_X - s_global * (mu_V @ R_k.T)).squeeze(0)
            t_list.append(t_k)

            # Track patch alignment error with current X_global
            X_hat_k = s_global * (V_k @ R_k.T) + t_k  # (m_k, D)
            sqerr = (X_hat_k - X_k).pow(2).sum(dim=1)  # (m_k,)
            patch_mse = sqerr.mean().item()
            per_patch_mse.append(patch_mse)

        if DEBUG_FLAG:
            per_patch_mse_t = torch.tensor(per_patch_mse)
            print(f"[ALIGN] per-patch mse: "
                f"p10={per_patch_mse_t.quantile(0.10).item():.4e} "
                f"p50={per_patch_mse_t.quantile(0.50).item():.4e} "
                f"p90={per_patch_mse_t.quantile(0.90).item():.4e}")

        # ======================================================================
        # Step B: update X from all patch transforms (centrality-weighted)
        # ======================================================================
        device_X = X_global.device

        new_X = torch.zeros_like(X_global)
        W_X = torch.zeros(n_sc, 1, dtype=torch.float32, device=device_X)

        for k in range(K):
            S_k = patch_indices[k].to(device_X)          # (m_k,)
            V_k = patch_coords[k].to(device_X)           # (m_k, D)
            R_k = R_list[k].to(device_X)                 # (D, D)
            t_k = t_list[k].to(device_X)                 # (D,)

            # Centrality weights in local patch coordinates
            center_k = V_k.mean(dim=0, keepdim=True)     # (1, D)
            dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)        # (m_k, 1)

            # Transformed patch in global frame using FIXED s_global = 1.0
            X_hat_k = s_global * (V_k @ R_k.T) + t_k     # (m_k, D)

            # Weighted accumulation
            new_X.index_add_(0, S_k, X_hat_k * weights_k)
            W_X.index_add_(0, S_k, weights_k)

        # Finish Step B: normalize and recenter
        mask_seen2 = W_X.squeeze(-1) > 0
        new_X[mask_seen2] /= W_X[mask_seen2]
        # Cells never hit by any patch: keep previous
        new_X[~mask_seen2] = X_global[~mask_seen2]


        new_X = new_X - new_X.mean(dim=0, keepdim=True)
        X_global = new_X

        if DEBUG_FLAG:
            rms_new = new_X.pow(2).mean().sqrt().item()
            print(f"[new ALIGN] iter={it + 1} coords_rms={rms_new:.3f} (global scale)")

    # ------------------------------------------------------------------
    # 6) Compute EDM and optional ST-scale alignment
    # ------------------------------------------------------------------
    if DEBUG_FLAG:
        print("\n[GLOBAL] Computing EDM from global coordinates...")

    X_full = X_global
    X_full = X_full - X_full.mean(dim=0, keepdim=True)

    Xd = X_full.to(device)
    D = torch.cdist(Xd, Xd)
    D_edm = uet.edm_project(D).detach().cpu()

    if target_st_p95 is not None:
        N = D_edm.shape[0]
        iu_s, ju_s = torch.triu_indices(N, N, 1, device=D_edm.device)
        D_vec = D_edm[iu_s, ju_s]
        current_p95 = torch.quantile(D_vec, 0.95).clamp_min(1e-6)
        scale_factor = (target_st_p95 / current_p95).clamp(0.5, 4.0)
        D_edm = D_edm * scale_factor
        if DEBUG_FLAG:
            print(f"[SCALE] current_p95={current_p95:.3f} "
                  f"target_p95={target_st_p95:.3f} "
                  f"scale={scale_factor:.3f}")

    result: Dict[str, torch.Tensor] = {"D_edm": D_edm}

    # ------------------------------------------------------------------
    # 7) Debug stats and optional 2D coords
    # ------------------------------------------------------------------
    if DEBUG_FLAG:
        N = D_edm.shape[0]
        total_pairs = N * (N - 1) // 2
        print(f"\n[GLOBAL] N={N} (total_pairs={total_pairs})")

        MAX_SAMPLES = 1_000_000
        if total_pairs <= MAX_SAMPLES:
            iu_all, ju_all = torch.triu_indices(N, N, 1, device=D_edm.device)
            D_sample = D_edm[iu_all, ju_all].float()
        else:
            k = MAX_SAMPLES
            i = torch.randint(0, N, (int(k * 1.3),), device=D_edm.device)
            j = torch.randint(0, N, (int(k * 1.3),), device=D_edm.device)
            keep = i < j
            i = i[keep][:k]
            j = j[keep][:k]
            D_sample = D_edm[i, j].float()
            print(f"[GLOBAL] (sampled {len(D_sample)} pairs for stats)")

        print(f"[GLOBAL] dist: "
              f"p50={D_sample.quantile(0.50):.3f} "
              f"p90={D_sample.quantile(0.90):.3f} "
              f"p99={D_sample.quantile(0.99):.3f} "
              f"max={D_sample.max():.3f}")

        coords_rms = X_full.pow(2).mean().sqrt().item()
        print(f"[GLOBAL] coords_rms={coords_rms:.3f}")

        cov = torch.cov(X_full.float().T)
        eigs = torch.linalg.eigvalsh(cov)
        ratio = float(eigs.max() / (eigs.min().clamp(min=1e-8)))
        print(f"[GLOBAL] coord_cov eigs: "
              f"min={eigs.min():.3e} "
              f"max={eigs.max():.3e} "
              f"ratio={ratio:.1f}")

    if return_coords:
        n = D_edm.shape[0]
        Jn = torch.eye(n) - torch.ones(n, n) / n
        B = -0.5 * (Jn @ (D_edm**2) @ Jn)
        coords = uet.classical_mds(B.to(device), d_out=2).detach().cpu()
        coords_canon = uet.canonicalize_coords(coords).detach().cpu()
        result["coords"] = coords
        result["coords_canon"] = coords_canon

    # Cleanup GPU tensors to allow process exit
    # Cleanup GPU tensors to allow process exit
    del Xd, D, X_full
    del Z_all, patch_indices, memberships, patch_coords
    # del X_global, R_list, t_list
    
    if "cuda" in device:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    if DEBUG_FLAG:
        print("=" * 72)
        print("STAGE D (PATCH-BASED) COMPLETE")
        print("=" * 72 + "\n")

    return result



