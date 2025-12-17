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

# ===================================================================
# DIAGNOSTIC HELPER FUNCTIONS FOR PATCH-BASED INFERENCE
# ===================================================================

def knn_sets(X, k=10):
    """Get k-NN sets for each point."""
    D = torch.cdist(X, X)
    idx = D.topk(k+1, largest=False).indices[:, 1:]
    return [set(row.tolist()) for row in idx]

def mean_jaccard(knnA, knnB):
    """Compute mean Jaccard similarity between two lists of neighbor sets."""
    js = []
    for a, b in zip(knnA, knnB):
        inter = len(a & b)
        uni = len(a | b)
        js.append(inter / max(1, uni))
    return float(np.mean(js))

def compute_overlap_graph_stats(patch_indices, n_sc, min_overlap=20):
    """Compute patch graph connectivity statistics."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    
    K = len(patch_indices)
    overlap_sizes = []
    edges = []
    
    # Compute all pairwise overlaps
    for i in range(K):
        for j in range(i+1, K):
            S_i = set(patch_indices[i].tolist())
            S_j = set(patch_indices[j].tolist())
            overlap = len(S_i & S_j)
            overlap_sizes.append(overlap)
            
            if overlap >= min_overlap:
                edges.append((i, j))
    
    # Build adjacency matrix
    adj = sp.lil_matrix((K, K), dtype=bool)
    for i, j in edges:
        adj[i, j] = True
        adj[j, i] = True
    
    # Connected components
    n_components, labels = connected_components(adj, directed=False)
    component_sizes = np.bincount(labels)
    giant_component_size = component_sizes.max() if len(component_sizes) > 0 else 0
    
    return {
        'overlap_sizes': overlap_sizes,
        'n_components': n_components,
        'giant_component_size': giant_component_size,
        'giant_component_frac': giant_component_size / K if K > 0 else 0,
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
        self.st_dist_head = None
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
    
    def forward_edm(
            self,
            x: torch.Tensor,           # (B, N, D) noisy input
            sigma: torch.Tensor,       # (B,) noise level
            H: torch.Tensor,           # (B, N, c_dim) context
            mask: torch.Tensor,        # (B, N)
            sigma_data: float,         # data std
            self_cond: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            EDM-preconditioned forward pass.
            Returns denoised estimate x0_pred.
            
            D_θ(x, σ) = c_skip · x + c_out · F_θ(c_in · x; c_noise, H)
            """
            B, N, D = x.shape
            
            # Compute preconditioning
            c_skip, c_out, c_in, c_noise = uet.edm_precond(sigma, sigma_data)
            # c_skip, c_out, c_in: (B, 1, 1)
            # c_noise: (B, 1)
            
            # Scale input
            x_in = c_in * x  # (B, N, D)
            
            # Call underlying network with c_noise as time embedding
            # The network expects t in shape (B, 1), c_noise is already (B, 1)
            F_x = self.forward(x_in, c_noise, H, mask, self_cond=self_cond)
            
            # If forward returns tuple (for dist_aux), extract just the prediction
            if isinstance(F_x, tuple):
                F_x = F_x[0]
            
            # EDM denoiser output
            x0_pred = c_skip * x + c_out * F_x
            x0_pred = x0_pred * mask.unsqueeze(-1).float()
            
            return x0_pred
    
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
    st_dataset: Optional['STSetDataset'],
    sc_dataset: Optional['SCSetDataset'],
    prototype_bank: Dict,
    n_epochs: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-4,
    n_timesteps: int = 500,  # CHANGED from 600
    sigma_min: float = 0.002, #edm default
    sigma_max: float = 80.0, #edm default (wll be clamped by sigma_data)
    device: str = 'cuda',
    outf: str = 'output',
    fabric: Optional['Fabric'] = None,
    precision: str = '16-mixed', # "32-true" | "16-mixed" | "bf16-mixed"
    logger = None,
    log_interval: int = 20,
    # Early stopping parameters
    enable_early_stop: bool = True,
    early_stop_min_epochs: int = 12,
    early_stop_patience: int = 6,
    early_stop_threshold: float = 0.01,  # 1% relative improvement
    # NEW: EDM parameters
    P_mean: float = -1.2,          # Log-normal mean for sigma sampling
    P_std: float = 1.2,            # Log-normal std for sigma sampling
    use_edm: bool = True,          # Enable EDM mode
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

    # After you initialize loss_knn_nca but BEFORE training starts:
    print("\n[Computing reference-based tau for KNN NCA]")

    # Collect all ST reference coordinates
    all_ref_coords = []
    print("\n[Computing reference-based tau for KNN NCA]")

    slide_d15_medians = []
    for slide_id in st_dataset.targets_dict:
        y_hat = st_dataset.targets_dict[slide_id].y_hat.to(device)  # (n_slide, 2)
        n_slide = y_hat.shape[0]
        
        if n_slide < 20:
            continue
        
        with torch.no_grad():
            D_slide = torch.cdist(y_hat, y_hat)  # (n_slide, n_slide)
            D_slide[torch.arange(n_slide), torch.arange(n_slide)] = float('inf')
            
            knn_dists, _ = torch.topk(D_slide, k=min(15, n_slide-1), dim=1, largest=False)
            d_15th = knn_dists[:, -1]
            slide_d15_medians.append(d_15th.median().item())

    if slide_d15_medians:
        r_15_median = float(np.median(slide_d15_medians))
    else:
        r_15_median = 1.0  # fallback

    tau_reference = r_15_median ** 2

    print(f"  Per-slide 15-NN median distances: {slide_d15_medians}")
    print(f"  Global median: {r_15_median:.6f}")
    print(f"  Setting tau = {tau_reference:.6f}\n")

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
    generator = generator.to(device).train()
    
    params = (
        list(context_encoder.parameters()) +
        list(generator.parameters()) +
        list(score_net.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    # --- NEW: wrap models + optimizer for DDP ---    
    if fabric is not None:
        context_encoder, generator, score_net, optimizer = fabric.setup(
            context_encoder, generator, score_net, optimizer
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # VE SDE
    if not use_edm:
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
        t_list=(0.25, 1.0),
        laplacian='sym'
    )
    loss_sw = uet.SlicedWassersteinLoss1D()
    loss_triplet = uet.OrdinalTripletLoss()
    # loss_knn_nca = uet.KNNSoftmaxLoss(tau=1.0, k=15)
    # NEW: topology aware losses
    loss_edge = uet.EdgeLengthLoss()
    loss_topo = uet.TopologyLoss()
    loss_shape_spectrum = uet.ShapeSpectrumLoss()


    loss_dim = uet.IntrinsicDimensionLoss(k_neighbors=20, target_dim=2.0)
    loss_triangle = uet.TriangleAreaLoss(num_triangles_per_sample=500, knn_k=12)
    loss_radial = uet.RadialHistogramLoss(num_bins=20)

    
    # DataLoaders - OPTIMIZED
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_minisets, collate_sc_minisets

    if fabric is not None:
        device = str(fabric.device)
    
    #compute data deriven bin edges form ST coords
    # ST loader (conditional)
    use_st = (st_dataset is not None)
    if use_st:
        #compute data deriven bin edges form ST coords
        all_st_coords = []
        for slide_id in st_dataset.targets_dict:
            y_hat = st_dataset.targets_dict[slide_id].y_hat
            all_st_coords.append(y_hat.cpu().numpy() if torch.is_tensor(y_hat) else y_hat)

        #concatenate all slides
        st_coords_np = np.concatenate(all_st_coords, axis=0)

        st_dist_bin_edges = uet.init_st_dist_bins_from_data(
            st_coords_np, n_bins = score_net.dist_bins,
            mode='log'
        ).to(device)

        #register in score_net
        score_net.st_dist_bin_edges = st_dist_bin_edges
        print(f"[Stage C] Computed ST distance bin edges from {len(all_st_coords)} slides, {st_coords_np.shape[0]} total spots")
        print(f"[Stage C] Bin edges: min={st_dist_bin_edges[0].item():.4f}, max={st_dist_bin_edges[-1].item():.4f}")

        st_loader = DataLoader(
            st_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_minisets, 
            num_workers=0,
            pin_memory=False
        )
        print(f"[ST Loader] {len(st_dataset)} samples, {len(st_loader)} batches/epoch")
    else:
        st_loader = None
        print("[ST Loader] DISABLED (st_dataset=None)")
        print("[WARNING] ST dataset is empty - ST training disabled!")

    # SC loader (conditional)
    use_sc = (sc_dataset is not None)
    if use_sc:
        sc_loader = DataLoader(
            sc_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_sc_minisets, 
            num_workers=0,
            pin_memory=False
        )
        print(f"[SC Loader] {len(sc_dataset)} samples, {len(sc_loader)} batches/epoch")
    else:
        sc_loader = None
        print("[SC Loader] DISABLED (sc_dataset=None)")

    if use_st and use_sc:
        steps_per_epoch = len(st_loader) + len(sc_loader)
    elif use_st:
        steps_per_epoch = len(st_loader)
    elif use_sc:
        steps_per_epoch = len(sc_loader)
    else:
        steps_per_epoch = 1  # fallback

    LOG_EVERY = steps_per_epoch  # Print once per epoch

    # ========== COMPUTE TRIANGLE EPSILON FOR ST ONLY ==========
    print("\n[Triangle Loss] Computing ST-specific threshold (ST batches only)...")
    
    triangle_refs = []
    with torch.no_grad():
        for _ in range(20):
            try:
                batch = next(iter(st_loader))
            except:
                st_loader_temp = DataLoader(
                    st_dataset, batch_size=4, shuffle=True,
                    collate_fn=collate_minisets, num_workers=0
                )
                batch = next(iter(st_loader_temp))
            
            V_target = batch['V_target'].to(device)
            mask = batch['mask'].to(device)
            A_ref = loss_triangle._compute_avg_triangle_area(V_target, mask)
            triangle_refs.extend(A_ref.cpu().tolist())
    
    # triangle_epsilon_st = 1.0 * float(torch.tensor(triangle_refs).median())
    areas = np.array(triangle_refs)
    triangle_epsilon_st = float(np.quantile(areas, 0.10))
    
    # SC uses FIXED small epsilon (ST-independent)
    triangle_epsilon_sc = 0.01  # Just "don't collapse to line", no ST bias
    
    print(f"[Triangle Loss] ST epsilon (60% of median): {triangle_epsilon_st:.4f}")
    print(f"[Triangle Loss] SC epsilon (fixed, ST-independent): {triangle_epsilon_sc:.4f}")
    print(f"[Triangle Loss] → ST: Hinge mode with ST-derived threshold")
    print(f"[Triangle Loss] → SC: Weak guardrail, no ST influence\n")
    # =============================================================


    # ========== COMPUTE REPEL EPSILON FOR ST (NEAREST-NEIGHBOR DISTANCE) ==========
    print("\n[Repel Loss] Computing ST nearest-neighbor distance threshold (ST batches only)...")

    nn_dists = []
    debug_batches_processed = 0
    debug_samples_processed = 0

    with torch.no_grad():
        st_iter_repel = iter(st_loader)
        for batch_idx in range(20):
            try:
                batch = next(st_iter_repel)
            except StopIteration:
                st_iter_repel = iter(st_loader)
                batch = next(st_iter_repel)
            
            D_target = batch['D_target'].to(device)
            mask = batch['mask'].to(device)
            
            B, N, _ = D_target.shape
            debug_batches_processed += 1
            
            for b in range(B):
                m_b = mask[b]
                n_valid = m_b.sum().item()
                
                if n_valid < 2:
                    continue
                
                debug_samples_processed += 1
                
                D_b = D_target[b][m_b][:, m_b]  # (n_valid, n_valid)
                
                # Mask out diagonal
                D_b_nodiag = D_b + torch.eye(n_valid, device=device) * 1e6
                
                # Get nearest neighbor distance for each point
                nn_dist_b = D_b_nodiag.min(dim=1)[0]  # (n_valid,)
                
                
                nn_dists.extend(nn_dist_b.cpu().tolist())


    nn_dists_arr = np.array(nn_dists)

    if len(nn_dists_arr) == 0:
        raise RuntimeError("[Repel Loss] No ST NN distances collected!")

    print(f"  NN distances array stats:")
    print(f"    min: {nn_dists_arr.min():.6f}")
    print(f"    max: {nn_dists_arr.max():.6f}")
    print(f"    mean: {nn_dists_arr.mean():.6f}")
    print(f"    median: {np.median(nn_dists_arr):.6f}")
    print(f"    p05: {np.quantile(nn_dists_arr, 0.05):.6f}")
    print(f"    p10: {np.quantile(nn_dists_arr, 0.10):.6f}")
    print(f"    p15: {np.quantile(nn_dists_arr, 0.15):.6f}")
    print(f"    p25: {np.quantile(nn_dists_arr, 0.25):.6f}")
    print(f"  Number of zeros: {(nn_dists_arr == 0.0).sum()}")
    print(f"  Number < 0.001: {(nn_dists_arr < 0.001).sum()}")
    print(f"  First 20 values: {nn_dists_arr[:20].tolist()}")

    repel_epsilon_st = float(np.quantile(nn_dists_arr, 0.01))

    print(f"\n[Repel Loss] ST epsilon (15% quantile of NN dists): {repel_epsilon_st:.6f}")
    print(f"[Repel Loss] → Penalizes predicted distances < {repel_epsilon_st:.6f}\n")
    # =============================================================================


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
            sc_loader=sc_loader if use_sc else None,  # ← Pass None if disabled
            batches=3,
            device=device,
            is_global_zero=_is_rank0(),
            # save_json_path=os.path.join(outf, "minisets_check.json"),
            save_json_path=None

        )
        print("\n" + "="*80)
        print("MINISETS CHECK COMPLETE - CHECK OUTPUT ABOVE")
        print("="*80 + "\n")
    # ============================================================================

    from utils_et import build_sc_knn_cache

    # Build kNN cache from SC dataset embeddings (conditional)
    if use_sc:
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
    else:
        POS_IDX = None
        K_POS = 0
        print("SC kNN cache SKIPPED (no SC data)")

    # Optional: save to disk
    # torch.save(sc_knn, f"{outf}/sc_knn_cache.pt")

    #wrap dataloaders for ddp sharding
    if fabric is not None:
        if use_st:
            st_loader = fabric.setup_dataloaders(st_loader)
        if use_sc:
            sc_loader = fabric.setup_dataloaders(sc_loader)
    
    os.makedirs(outf, exist_ok=True)
    plot_dir = os.path.join(outf, 'plots')
    #only rank 0 touches the filesystem
    if fabric is None or fabric.is_global_zero:
        os.makedirs(plot_dir, exist_ok=True)

    # WEIGHTS = {
    #     'score': 1.0,
    #     'gram': 0.5,
    #     'gram_scale': 0.3,
    #     'heat': 0.25,
    #     'sw_st': 0.2,
    #     'sw_sc': 0.2,
    #     'overlap': 0.25,
    #     'ordinal_sc': 0.5,
    #     'st_dist': 0.3,
    #     'edm_tail': 0.3,
    #     'gen_align': 0.3,
    #     'dim': 0.1,
    #     'triangle': 0.5,
    #     'radial': 1.0,
    #     'knn_nca': 0.5,
    #     'repel': 0.0,      # NEW: ST repulsion loss
    #     'shape': 0.0       # NEW: ST anisotropy/shape loss
    # }


    # WEIGHTS = {
    #     'score': 1.0,
    #     'gram': 0.5,
    #     'gram_scale': 0.5,
    #     'heat': 0.0,
    #     'sw_st': 0.0,
    #     'sw_sc': 0.25,
    #     'overlap': 0.25,
    #     'ordinal_sc': 0.5,
    #     'st_dist': 0.0,
    #     'edm_tail': 0.0,
    #     'gen_align': 0.0,
    #     'dim': 0.0,
    #     'triangle': 0.0,
    #     'radial': 0.0,
    #     'knn_nca': 0.7,
    #     'repel': 0.0,      # NEW: ST repulsion loss
    #     'shape': 0.0       # NEW: ST anisotropy/shape loss
    # }

    WEIGHTS = {
        'score': 1.0,
        'gram': 0.5,
        'gram_scale': 0.3,
        'heat': 0.25,
        'sw_st': 0.0,
        'sw_sc': 0.3,
        'overlap': 0.25,
        'ordinal_sc': 0.5,
        'st_dist': 0.0,
        'edm_tail': 0.3,
        'gen_align': 0.0,
        'dim': 0.0,
        'triangle': 0.0,
        'radial': 0.5,
        'knn_nca': 0.0,
        'repel': 0.0,      # NEW: ST repulsion loss
        'shape': 0.0,       # NEW: ST anisotropy/shape loss
        # NEW
        'edge': 0.5,       # Edge-length preservation (local metric)
        'topo': 0.3,       # Topology preservation (persistent homology proxy)
        'shape_spec': 0.3,      # Shape spectrum (anisotropy matching)
    }

    # Heat warmup schedule
    HEAT_WARMUP_EPOCHS = 10
    HEAT_TARGET_WEIGHT = 0.05
    
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
    p_uncond = 0.25
    

    history = {
        'epoch': [],
        'batch_losses': [],
        'epoch_avg': {
            'total': [], 'score': [], 'gram': [], 'gram_scale': [], 'heat': [],
            'sw_st': [], 'sw_sc': [], 'overlap': [], 'ordinal_sc': [], 'st_dist': [],
            'edm_tail': [], 'gen_align': [], 'dim': [], 'triangle': [], 'radial': [], 'knn_nca': [], 
            'repel': [], 'shape': [],  # NEW: ST shape constraints
            'edge': [],     # NEW
            'topo': [],     # NEW
            'shape_spec': []
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
            if use_st:
                it = iter(st_loader)
                for _ in range(min(10, len(st_loader))):
                    sample_batch = next(it, None)
                    if sample_batch is None:
                        break
                    V_batch = sample_batch['V_target'].to(device, non_blocking=True)
                    mask_batch = sample_batch['mask'].to(device, non_blocking=True)
                    for i in range(min(4, V_batch.shape[0])):
                        m = mask_batch[i]
                        if m.sum() > 0:
                            V_temp = V_batch[i, m]
                            sample_stds.append(V_temp.std().item())
            sigma_data = float(np.median(sample_stds)) if sample_stds else 1.0
        else:
            sigma_data = 0.0

    sigma_data = sync_scalar(sigma_data, device)

    # EDM: Clamp sigma_max based on sigma_data
    if use_edm:
        sigma_max = min(sigma_max, sigma_data * 100)  # Reasonable upper bound
        sigma_min = max(sigma_min, sigma_data * 0.001)  # Reasonable lower bound
    
    print(f"[StageC] sigma_data = {sigma_data:.4f}")
    print(f"[StageC] sigma_min = {sigma_min:.6f}, sigma_max = {sigma_max:.2f}")
    if use_edm:
        print(f"[StageC] EDM mode ENABLED: P_mean={P_mean}, P_std={P_std}")


    #--amp scaler choice based on precision---
    use_fp16 = (precision == '16-mixed')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    global_step = 0

    ema_grads = {
        'score': 1.0,
        'gram': 0.1,
        'gram_scale': 0.1,
        'heat': 0.1,
        'sw_st': 0.1,
        'st_dist': 0.1,
        'edm_tail': 0.1,
        'gen_align': 0.1,
        'radial': 0.1,
        'edge': 0.1,
        'topo': 0.1,
        'shape_spec': 0.1,
        'shape': 0.1
        # dim and triangle are regularizers - keep manual for now
    }
    ema_sc = {
        'score': 1.0,
        'sw_sc': 0.1,
        'overlap': 0.1,
        'ordinal_sc': 0.1
    }
    EMA_BETA = 0.98  # CHANGED from 0.95
 
    # Target gradient ratios relative to score loss
    # These control how much each loss contributes to the gradient
    # TARGET = {
    #     'gram': 0.05,           # Keep Gram gradients at ~5% of score
    #     'gram_scale': 0.03,     # Scale regularizer weaker
    #     'heat': 0.15,           # Heat kernel at ~15% of score
    #     'sw_st': 0.10,          # Sliced-Wasserstein at ~10%
    #     'st_dist': 0.08,        # ST distance classification at ~8%
    #     'edm_tail': 0.12,       # EDM tail loss at ~12%
    #     'gen_align': 0.06,      # Procrustes alignment at ~6%
    #     'radial': 0.04,         # Radial histogram at ~4%
    #     'edge': 0.12,      # Similar to edm_tail (local geometry)
    #     'topo': 0.10,      # Similar to sw_st (distribution matching)
    #     'shape_spec': 0.06,
    # }

    TARGET = {
        # Global losses - REDUCED (your global structure is OK)
        'gram': 0.03,           # was 0.05 - reduce global Gram influence
        'gram_scale': 0.02,     # was 0.03
        'heat': 0.10,           # was 0.15 - heat is multi-scale, keep moderate
        'sw_st': 0.05,          # was 0.10 - reduce global distribution matching
        'radial': 0.02,         # was 0.04 - reduce radial histogram
        
        # Local losses - INCREASED (your local structure is bad)
        'edge': 0.25,           # was 0.12 - DOUBLE for local edge preservation
        'topo': 0.20,           # was 0.10 - DOUBLE for topology (MST edges are local-ish)
        
        # Keep these moderate
        'st_dist': 0.06,        # was 0.08
        'edm_tail': 0.08,       # was 0.12 - tail is semi-local
        'gen_align': 0.04,      # was 0.06
        'shape_spec': 0.04,     # was 0.06 (shape is global, not helping local)
    }

    TARGET_SC = {
        'sw_sc': 0.10,        # Match sw_st (was 0.08) - distribution matching is important
        'overlap': 0.05,      # Increase from 0.01 - too weak at 1%
        'ordinal_sc': 0.08,   # Increase from 0.06 - ordinal structure is important
        }
    AUTOBALANCE_EVERY = 100
    AUTOBALANCE_START = 200

    USE_AUTOBALANCE = True

    #self conditioning probability
    p_sc = 0.5 #prob of using self-conditioning in training

    # Early stopping state
    early_stop_best = float('inf')
    early_stop_no_improve = 0
    early_stopped = False
    early_stop_epoch = -1

    if enable_early_stop:
        print(f"\n[Early Stop] Enabled: min_epochs={early_stop_min_epochs}, "
            f"patience={early_stop_patience}, threshold={early_stop_threshold:.1%}")
    
    for epoch in range(n_epochs):

        epoch_cv_sum = 0.0
        epoch_qent_sum = 0.0
        epoch_nca_loss_sum = 0.0
        epoch_nca_count = 0


        st_iter = iter(st_loader) if use_st else None
        sc_iter = iter(sc_loader) if use_sc else None
        
        epoch_losses = {k: 0.0 for k in WEIGHTS.keys()}
        epoch_losses['total'] = 0.0
        n_batches = 0
        c_overlap = 0

        st_batches = 0
        sc_batches = 0

        
        # Schedule based on what loaders we have
        if use_st and use_sc:
            max_len = max(len(st_loader), len(sc_loader))
            # schedule = ['ST', 'ST', 'SC'] * (max_len // 3 + 1)
            schedule = ['SC', 'SC', 'ST'] * (max_len // 3 + 1)
            mode_str = "ST+SC"
        elif use_st:
            schedule = ['ST'] * len(st_loader)
            mode_str = "ST-only"
        elif use_sc:
            schedule = ['SC'] * len(sc_loader)
            mode_str = "SC-only"
        else:
            raise ValueError("Must have at least one of ST or SC data")
        
        # Ensure all ranks use same schedule length
        if fabric is not None:
            fabric.barrier()

                
        # Batch progress bar
        # batch_pbar = tqdm(schedule, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True)
        batch_pbar = tqdm(schedule, desc=f"Epoch {epoch+1}/{n_epochs} [{mode_str}]", leave=False)
        
        for batch_type in batch_pbar:
            if batch_type == 'ST':
                if not use_st:
                    continue
                batch = next(st_iter, None)
                if batch is None:
                    st_iter = iter(st_loader)
                    batch = next(st_iter, None)
                    if batch is None:
                        continue
            else:  # SC
                if not use_sc:
                    continue  # Skip SC batches if disabled
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
            
            # ===== FORWARD PASS WITH AMP =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                # Context encoding
                H = context_encoder(Z_set, mask)

                # === Define self-conditioning flag ONCE ===
                use_self_cond = (torch.rand(1, device=device).item() < p_sc)

                # === EDM: sample sigma from log-normal ===
                if use_edm:
                    sigma = uet.sample_sigma_lognormal(batch_size_real, P_mean, P_std, device)
                    sigma = sigma.clamp(sigma_min, sigma_max)
                    sigma_t = sigma.view(-1, 1, 1)
                    
                    # Create t_norm proxy for gating/debug (log-space normalization)
                    log_sigma = sigma.log()
                    log_min = math.log(sigma_min)
                    log_max = math.log(sigma_max)
                    t_norm = ((log_sigma - log_min) / (log_max - log_min + 1e-8)).clamp(0, 1)  # (B,)
                else:
                    # Old: quadratic bias toward low noise
                    u = torch.rand(batch_size_real, device=device)
                    t_cont = (u ** 2) * (n_timesteps - 1)
                    t_idx = t_cont.long()
                    t_norm = t_cont / (n_timesteps - 1)
                    sigma_t = sigmas[t_idx].view(-1, 1, 1)

                # ===== EDM: x0 FROM ST TARGET, NOT GENERATOR =====
                if use_edm and not is_sc:
                    # ST batch: x0 is the ground truth geometry
                    V_0 = batch['V_target'].to(device)
                else:
                    # SC batch or non-EDM: use generator
                    V_0 = generator(H, mask)
                
                # Debug: Check V_0 scale variation (first 10 steps only)
                if DEBUG and global_step < 10:
                    with torch.no_grad():
                        rms_per_set = []
                        for b in range(V_0.shape[0]):
                            m = mask[b]
                            if m.sum() > 0:
                                v = V_0[b, m]
                                rms = v.pow(2).mean().sqrt().item()
                                rms_per_set.append(rms)
                        if rms_per_set:
                            print(f"[V_0 scale] step={global_step} RMS range: [{min(rms_per_set):.3f}, \
                                  {max(rms_per_set):.3f}], \
                                  mean={sum(rms_per_set)/len(rms_per_set):.3f}")

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
                #    print(f"[CFG] step={global_step} dropped={int(n_dropped)}/{batch_size_real} contexts")

                # === forward pass ===
                if use_edm:
                    # EDM: use preconditioned forward, get x0_pred directly
                    sigma_flat = sigma_t.view(-1)

                    if use_self_cond and score_net.self_conditioning:
                        with torch.no_grad():
                            x0_pred_0 = score_net.forward_edm(V_t, sigma_flat, H_train, mask, sigma_data, self_cond=None)
                        x0_pred = score_net.forward_edm(V_t, sigma_flat, H_train, mask, sigma_data, self_cond=x0_pred_0)
                    else:
                        x0_pred = score_net.forward_edm(V_t, sigma_flat, H_train, mask, sigma_data, self_cond=None)

                    # for geom losses, V_hat is x0_pred
                    V_hat = x0_pred
                    dist_aux = None
                else:
                    # Self-conditioning logic
                    use_self_cond = torch.rand(1, device=device).item() < p_sc

                    if use_self_cond and score_net.self_conditioning:
                        # First pass without self-conditioning to get V_0 prediction
                        with torch.no_grad():
                            eps_hat_0 = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None)
                            V_pred_0 = V_t - sigma_t * eps_hat_0

                        result = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=V_pred_0, return_dist_aux=(not is_sc))
                        if isinstance(result, tuple):
                            eps_pred, dist_aux = result
                        else:
                            eps_pred = result
                            dist_aux = None
                    else:
                        # Single pass without self-conditioning
                        # eps_pred = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None)
                        result = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None, return_dist_aux=(not is_sc))
                        if isinstance(result, tuple):
                            eps_pred, dist_aux = result
                        else:
                            eps_pred = result
                            dist_aux = None

       
            # ===== SCORE LOSS (in fp32 for numerical stability) =====
            with torch.autocast(device_type='cuda', enabled=False):
                mask_fp32 = mask.float()
                if use_edm:
                    # EDM: Loss on denoised prediction vs ground truth
                    x0_pred_fp32 = x0_pred.float()
                    V_0_fp32 = V_0.float()
                    sigma_flat = sigma_t.view(-1).float()
                    
                    # EDM weight: (σ² + σ_d²) / (σ · σ_d)²
                    w = uet.edm_loss_weight(sigma_flat, sigma_data)  # (B,)
                    w = w.view(-1, 1)  # (B, 1)
                    w = w.clamp(max=1000.0)  # Prevent extreme weights
                    
                    # MSE on (x0_pred - x0)
                    err2 = (x0_pred_fp32 - V_0_fp32).pow(2).mean(dim=2)  # (B, N)
                    L_score = (w * err2 * mask_fp32).sum() / mask_fp32.sum()

                else:
                    sigma_t_fp32 = sigma_t.float()
                    eps_pred_fp32 = eps_pred.float()
                    eps_fp32 = eps.float()
                    mask_fp32 = mask.float()

                    # ===== CORRECT EDM WEIGHTING FOR NOISE PREDICTION =====
                    sigma_t_squeezed = sigma_t_fp32.squeeze(-1)  # (B, N) or (B,)
                    if sigma_t_squeezed.dim() == 1:
                        sigma_t_squeezed = sigma_t_squeezed.unsqueeze(-1)  # (B, 1)

                    # Simple EDM weight for noise prediction
                    # w(σ) = (σ² + σ_d²) / σ_d², with σ_d = 1
                    w = (sigma_t_squeezed**2 + sigma_data**2) / (sigma_data**2)  # here sigma_data=1

                    # Just cap very large weights; no SNR magic
                    w = w.clamp(max=100.0)

                    # Compute loss
                    err2 = (eps_pred_fp32 - eps_fp32).pow(2).mean(dim=2)  # (B, N)

                    # Handle broadcasting
                    if w.shape[-1] == 1:
                        w = w.expand_as(err2)

                    L_score = (w * err2 * mask_fp32).sum() / mask_fp32.sum()

                # === SANITY CHECK ===
                if DEBUG and (global_step % LOG_EVERY == 0) and (not is_sc):
                    with torch.no_grad():
                        m = mask_fp32 > 0
                        err2_mean = err2[m].mean().item()
                        w_vals = w.expand_as(err2)[m]
                        w_mean = w_vals.mean().item()
                        print(f"[score/check] err2={err2_mean:.3e} w_mean={w_mean:.3f} "
                            f"L_score={L_score.item():.3e}")
                

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
            L_knn_nca = torch.tensor(0.0, device=device)
            L_heat = torch.tensor(0.0, device=device)
            L_sw_st = torch.tensor(0.0, device=device)
            L_sw_sc = torch.tensor(0.0, device=device)
            L_overlap = torch.tensor(0.0, device=device)
            L_ordinal_sc = torch.tensor(0.0, device=device)
            L_st_dist = torch.tensor(0.0, device=device)
            L_edm_tail = torch.tensor(0.0, device=device)
            L_gen_align = torch.tensor(0.0, device=device)
            L_dim = torch.zeros((), device=device)
            L_triangle = torch.zeros((), device=device)
            L_radial = torch.zeros((), device=device)
            L_repel = torch.tensor(0.0, device=device)   # NEW: ST repulsion
            L_shape = torch.tensor(0.0, device=device)   # NEW: ST shape/anisotropy
            L_edge = torch.tensor(0.0, device=device)   # NEW: Edge-length loss
            L_topo = torch.tensor(0.0, device=device)   # NEW: Topology loss  
            L_shape_spec = torch.tensor(0.0, device=device)  # NEW: Shape spectrum loss

      
            # Denoise to get V_hat
            if not use_edm:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    V_hat = V_t - sigma_t * eps_pred
                    V_hat = V_hat * mask.unsqueeze(-1).float()

            # ==================== NEW GLOBAL GEOMETRY BLOCK ====================
            # 1. Canonicalize V_hat (Center per set) - NOW GLOBAL for ST and SC
            with torch.autocast(device_type='cuda', enabled=False):
                V_hat_f32 = V_hat.float()
                m_bool = mask.bool()
                m_float = mask.float().unsqueeze(-1)
                
                # center per set over valid nodes
                valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean = (V_hat_f32 * m_float).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
                V_geom = (V_hat_f32 - mean) * m_float  # (B,N,D), centered


            if not is_sc:
                # ===== ST STEP: Score + Gram + Heat + SW_ST =====

                #ST distogram loss
                if not is_sc and WEIGHTS['st_dist'] > 0 and dist_aux is not None and 'dist_logits' in dist_aux:
                    dist_logits = dist_aux['dist_logits']  # (B, N, N, dist_bins)
                    bin_ids_target = dist_aux['bin_ids']   # (B, N, N)
                    
                    # Mask: exclude diagonal and invalid nodes
                    B_d, N_d = mask.shape
                    diag_mask = ~torch.eye(N_d, dtype=torch.bool, device=device).unsqueeze(0)
                    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & diag_mask  # (B, N, N)
                    
                    # Flatten valid pairs
                    logits_flat = dist_logits[pair_mask]  # (num_pairs, dist_bins)
                    targets_flat = bin_ids_target[pair_mask].long()  # (num_pairs,)
                    
                    if logits_flat.numel() > 0:
                        L_st_dist = F.cross_entropy(logits_flat, targets_flat)
                    else:
                        L_st_dist = torch.tensor(0.0, device=device)
                else:
                    L_st_dist = torch.tensor(0.0, device=device)

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

                    Gt = G_target.float()
                    B, N, _ = V_geom.shape

                    # ==================== GRAM LOSS (COSINE-NORMALIZED) ====================
                    
                    # Build predicted Gram *with* true scale
                    Gp_raw = V_geom @ V_geom.transpose(1, 2)          # (B,N,N)
                    
                    # Build masks
                    # MM = (m.unsqueeze(-1) & m.unsqueeze(-2)).float()  # (B,N,N) valid pairs
                    MM = (m_bool.unsqueeze(-1) & m_bool.unsqueeze(-2)).float()
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
                        tr_p_raw = (Gp_raw.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1)
                        tr_t_raw = (Gt.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1)
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
                    geo_gate = (low_noise & cond_only).float()  # (B,) as float

                    # Always compute, weight by gate (ensures same graph across ranks)
                    gate_sum = geo_gate.sum().clamp(min=1.0)
                    L_gram = (per_set_relative_loss * geo_gate).sum() / gate_sum
                    
                    
                    # --- DEBUG 5: Loss statistics ---
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        print(f"[gram/loss] L_gram={L_gram.item():.3e} | "
                            f"per_set: mean={per_set_relative_loss.mean().item():.3e} "
                            f"med={per_set_relative_loss.median().item():.3e} "
                            f"min={per_set_relative_loss.min().item():.3e} "
                            f"max={per_set_relative_loss.max().item():.3e}")
                        
                        # Mask coverage
                        denom_per_set = P_off.sum(dim=(1,2))
                        n_valid = m_bool.sum(dim=1).float()
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
                        tr_p_ut = (Gp_raw.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1, keepdim=True).clamp_min(1e-8)
                        tr_t_ut = (Gt.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1, keepdim=True).clamp_min(1e-8)
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

                        # Use the same weighted approach to avoid divergence
                        L_gram_scale = ((log_ratio ** 2) * geo_gate).sum() / gate_sum

                # ==================== COMPUTE V_TARGET_BATCH ONCE ====================
                # Needed by: edge loss, topo loss, shape loss, etc.
                # Only compute if at least one of these losses is active
                need_v_target = (WEIGHTS['edge'] > 0 or WEIGHTS['topo'] > 0 or 
                                WEIGHTS['shape_spec'] > 0 or WEIGHTS['edm_tail'] > 0 or
                                WEIGHTS['gen_align'] > 0)
                
                if need_v_target:
                    with torch.autocast(device_type='cuda', enabled=False):
                        V_target_batch = torch.zeros_like(V_geom)  # (B, N, D)
                        
                        for i in range(batch_size_real):
                            n_valid = int(n_list[i].item())
                            if n_valid < 3:
                                continue
                            G_i = G_target[i, :n_valid, :n_valid].float()
                            V_tgt_i = uet.factor_from_gram(G_i, D_latent)  # (n_valid, D)
                            V_target_batch[i, :n_valid] = V_tgt_i

                # ==================== EDGE-LENGTH LOSS ====================
                # if WEIGHTS['edge'] > 0 and 'knn_indices' in batch:
                #     with torch.autocast(device_type='cuda', enabled=False):
                #         knn_indices_batch = batch['knn_indices'].to(device)
                                                
                        # Low-noise gating
                        # low_noise = (t_norm.squeeze() < 0.3)
                        # cond_only = (torch.rand(batch_size_real, device=device) >= p_uncond)
                        # geo_gate_edge = (low_noise & cond_only).float()

                        # Low-noise gating - use actual drop_mask for consistency with CFG
                if WEIGHTS['edge'] > 0 and 'knn_spatial' in batch:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # Use SPATIAL kNN for geometry, not expression kNN
                        knn_indices_batch = batch['knn_spatial'].to(device)
                        low_noise = (t_norm.squeeze() < 0.6)
                        cond_only = (drop_mask.view(-1) < 0.5)  # True when NOT dropped
                        geo_gate_edge = (low_noise & cond_only).float()
                        gate_sum_edge = geo_gate_edge.sum().clamp(min=1.0)
                        
                        L_edge_raw = loss_edge(
                            V_pred=V_geom,
                            V_target=V_target_batch,  # Already computed!
                            knn_indices=knn_indices_batch,
                            mask=mask
                        )
                        
                        L_edge = (L_edge_raw * geo_gate_edge).sum() / gate_sum_edge
                
                # ==================== TOPOLOGY LOSS ====================
                if WEIGHTS['topo'] > 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        topo_info = batch.get('topo_info', None)
                        
                        # V_target_batch already computed above!
                        
                        # Low-noise gating (more aggressive for topology)
                        # low_noise = (t_norm.squeeze() < 0.2)
                        # cond_only = (torch.rand(batch_size_real, device=device) >= p_uncond)
                        # geo_gate_topo = (low_noise & cond_only).float()

                        # Low-noise gating (more aggressive for topology)
                        low_noise = (t_norm.squeeze() < 0.4)
                        cond_only = (drop_mask.view(-1) < 0.5)  # True when NOT dropped
                        geo_gate_topo = (low_noise & cond_only).float()
                        gate_sum_topo = geo_gate_topo.sum().clamp(min=1.0)
                        
                        L_topo_raw = loss_topo(
                            V_pred=V_geom,
                            V_target=V_target_batch,  # Already computed!
                            mask=mask,
                            topo_info=topo_info
                        )
                        
                        L_topo = (L_topo_raw * geo_gate_topo).sum() / gate_sum_topo
                
                # ==================== SHAPE SPECTRUM LOSS ====================
                if WEIGHTS['shape_spec'] > 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # V_target_batch already computed above!
                        
                        # Low-noise gating
                        # low_noise = (t_norm.squeeze() < 0.3)
                        # cond_only = (torch.rand(batch_size_real, device=device) >= p_uncond)
                        # geo_gate_shape = (low_noise & cond_only).float()

                        # Low-noise gating
                        low_noise = (t_norm.squeeze() < 0.6)
                        cond_only = (drop_mask.view(-1) < 0.5)  # True when NOT dropped
                        geo_gate_shape = (low_noise & cond_only).float()
                        gate_sum_shape = geo_gate_shape.sum().clamp(min=1.0)
                        
                        L_shape_spectrum_raw = loss_shape_spectrum(
                            V_pred=V_geom,
                            V_target=V_target_batch,  # Already computed!
                            mask=mask
                        )
                        
                        L_shape_spec = (L_shape_spectrum_raw * geo_gate_shape).sum() / gate_sum_shape

                        # ==================== DEBUG PRINTS FOR NEW LOSSES ====================
                        if (global_step % 50 == 0) and (not is_sc):
                            # Topo pair counts
                            ti = batch.get("topo_info", None)
                            if ti is None:
                                print("[DBG] topo_info=None")
                            else:
                                c0 = [int(x["pairs_0"].shape[0]) if x is not None else -1 for x in ti]
                                c1 = [int(x["pairs_1"].shape[0]) if x is not None else -1 for x in ti]
                                print(f"[DBG] topo pairs_0: {c0}, pairs_1: {c1}")

                                # Check uniqueness and distance ranges for first sample
                                info0 = ti[0]
                                if info0 is not None:
                                    p0 = info0["pairs_0"].detach().cpu().numpy()
                                    p1 = info0["pairs_1"].detach().cpu().numpy()
                                    d0 = info0["dists_0"].detach().cpu().numpy()
                                    d1 = info0["dists_1"].detach().cpu().numpy()
                                    
                                    u0 = len(set(map(tuple, map(sorted, p0.tolist())))) if len(p0) > 0 else 0
                                    u1 = len(set(map(tuple, map(sorted, p1.tolist())))) if len(p1) > 0 else 0
                                    print(f"[DBG] topo uniq pairs: {u0}/{len(p0)} (0D) and {u1}/{len(p1)} (1D)")
                                    
                                    if len(d0) > 0:
                                        print(f"[DBG] topo d0 range: [{d0.min():.4f}, {d0.max():.4f}]")
                                    if len(d1) > 0:
                                        print(f"[DBG] topo d1 range: [{d1.min():.4f}, {d1.max():.4f}]")
                            
                            # Gate hit rates
                            edge_hits = geo_gate_edge.sum().item() if 'geo_gate_edge' in dir() else 0
                            topo_hits = geo_gate_topo.sum().item() if 'geo_gate_topo' in dir() else 0
                            shape_hits = geo_gate_shape.sum().item() if 'geo_gate_shape' in dir() else 0
                            print(f"[DBG] gates: edge={edge_hits:.0f} topo={topo_hits:.0f} shape={shape_hits:.0f} / {batch_size_real}")
                            
                            # Raw losses
                            print(f"[DBG] raw: gram={float(L_gram.detach().cpu()):.4e} "
                                f"edge={float(L_edge.detach().cpu()):.4e} "
                                f"topo={float(L_topo.detach().cpu()):.4e} "
                                f"shape={float(L_shape_spec.detach().cpu()):.4e}")
                            
                            # Weights
                            print(f"[DBG] W: gram={WEIGHTS['gram']:.3g} edge={WEIGHTS['edge']:.3g} "
                                f"topo={WEIGHTS['topo']:.3g} shape_spec={WEIGHTS['shape_spec']:.3g}")

                # Heat kernel loss (batched)
                if WEIGHTS['heat'] > 0 and (global_step % heat_every_k) == 0:
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
                                knn_k = getattr(st_dataset, 'knn_k', 12)  # Match dataset k
                                edge_index, edge_weight = uet.build_knn_graph_from_distance(
                                    D_V, k=knn_k, device=device
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
                else:
                    L_heat = torch.tensor(0.0, device=device)    


                L_sw_st = torch.zeros((), device=device)
                if WEIGHTS['sw_st'] > 0 and (global_step % sw_every_k) == 0:
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
                            use_canon=True,   # center per set
                        )

                        # ==================== NEW: EDM TAIL LOSS ====================
                        # Penalize compression of far distances
                        if WEIGHTS['edm_tail'] > 0:
                            with torch.autocast(device_type='cuda', enabled=False):
                                # We already have V_geom (centered V_hat) from Gram loss above
                                # Need to build V_target from G_target for each sample
                                V_target_batch = torch.zeros_like(V_geom)  # (B, N, D)
                                
                                for i in range(batch_size_real):
                                    n_valid = int(n_list[i].item())
                                    if n_valid < 3:
                                        continue
                                    G_i = G_target[i, :n_valid, :n_valid].float()
                                    V_tgt_i = uet.factor_from_gram(G_i, D_latent)  # (n_valid, D)
                                    V_target_batch[i, :n_valid] = V_tgt_i
                                
                                # Compute EDM tail loss
                                L_edm_tail = uet.compute_edm_tail_loss(
                                    V_pred=V_geom,           # centered V_hat
                                    V_target=V_target_batch,  # from factor_from_gram
                                    mask=mask,
                                    tail_quantile=0.85,       # top 20% distances
                                    weight_tail=2.0
                                )
                        else:
                            L_edm_tail = torch.tensor(0.0, device=device)
                        
                        # ==================== NEW: GENERATOR ALIGNMENT LOSS ====================
                        # Align generator output V_0 to target geometry via Procrustes
                        if WEIGHTS['gen_align'] > 0:
                            with torch.autocast(device_type='cuda', enabled=False):
                                # Center V_0 (generator output before noise)
                                V_0_f32 = V_0.float()
                                m_float = mask.float().unsqueeze(-1)
                                valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                                mean_V0 = (V_0_f32 * m_float).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
                                V_0_centered = (V_0_f32 - mean_V0) * m_float
                                
                                # V_target_batch already computed above for EDM tail loss
                                L_gen_align = uet.procrustes_alignment_loss(
                                    V_pred=V_0_centered,
                                    V_target=V_target_batch,
                                    mask=mask
                                )
                        else:
                            L_gen_align = torch.tensor(0.0, device=device)

                        # ==================== NEW MANIFOLD-AWARE REGULARIZERS ====================
                        # B1: Intrinsic dimension regularizer
                        if WEIGHTS['dim'] > 0:
                            with torch.autocast(device_type='cuda', enabled=False):
                                L_dim = loss_dim(
                                    V_pred=V_geom,
                                    V_target=V_target_batch,
                                    mask=mask,
                                    use_target=True
                                )
                        else:
                            L_dim = torch.tensor(0.0, device=device)
                        
                        # B2: Triangle area regularizer (ST: match normalized area to ST median)
                        if WEIGHTS['triangle'] > 0:
                            with torch.autocast(device_type='cuda', enabled=False):
                                # Compute average normalized triangle area per sample (B,)
                                A_pred = loss_triangle._compute_avg_triangle_area(V_geom, mask)
                                
                                # Target
                                target = torch.full_like(A_pred, triangle_epsilon_st)
                                
                                # MSE loss to pull areas toward ST-style geometry
                                L_triangle = (A_pred - target).pow(2).mean()
                        else:
                            L_triangle = torch.tensor(0.0, device=device)
                        
                        # B3: Radial histogram loss
                        if WEIGHTS['radial'] > 0:
                            with torch.autocast(device_type='cuda', enabled=False):
                                L_radial = loss_radial(
                                    V_pred=V_geom,
                                    V_target=V_target_batch,
                                    mask=mask
                                )
                        else:
                            L_radial = torch.tensor(0.0, device=device)


                        if DEBUG and (global_step % LOG_EVERY == 0):
                            print(f"[sw_st] step={global_step} val={float(L_sw_st.item()):.6f} K=64 cap=512 pairs={int(pairA.numel())}")


                # ==================== NEW: REPULSION LOSS (ST ONLY) ====================
                # Penalize very small predicted pairwise distances (avoid local overlaps)
                if False:
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        print(f"\n[REPEL] Computing repulsion loss for ST batch...")

                    B_rep, N_rep, D_rep = V_geom.shape
                    repel_per_set = torch.zeros(B_rep, device=device)  # ← INITIALIZE HERE
                    repel_debug = {"total_pairs": 0, "violating_pairs": 0, "min_dist": float('inf'), "samples_checked": 0}

                    for b in range(B_rep):
                        m_b = mask[b]
                        n_valid = m_b.sum().item()
                        if n_valid < 2:
                            continue
                        
                        V_b = V_geom[b, m_b]  # (n_valid, D)
                        
                        # ROTATION-INVARIANT 2D PROJECTION (PCA)
                        V_b_c = V_b - V_b.mean(0, keepdim=True)  # center
                        Cov_b = V_b_c.T @ V_b_c / n_valid  # (D, D) covariance
                        eigvals, eigvecs = torch.linalg.eigh(Cov_b)  # ascending order
                        U2 = eigvecs[:, -2:]  # top-2 eigenvectors (last 2 columns)
                        V_b_2d = V_b_c @ U2  # project to principal 2D plane (n_valid, 2)
                        
                        D_pred_b = torch.cdist(V_b_2d, V_b_2d)  # (n_valid, n_valid)
                        
                        # Mask out diagonal
                        eye_b = torch.eye(n_valid, device=device, dtype=torch.bool)
                        D_pred_b_nodiag = D_pred_b.masked_fill(eye_b, 1e6)
                        
                        # DEBUG: Track statistics
                        repel_debug["samples_checked"] += 1
                        min_dist_sample = D_pred_b_nodiag.min().item()
                        repel_debug["min_dist"] = min(repel_debug["min_dist"], min_dist_sample)
                        repel_debug["total_pairs"] += (n_valid * (n_valid - 1)) // 2
                        
                        # Find pairs with distance < repel_epsilon_st
                        violating_mask = (D_pred_b_nodiag < repel_epsilon_st) & (~eye_b)
                        n_violations = violating_mask.sum().item() // 2  # Divide by 2 (symmetric)
                        repel_debug["violating_pairs"] += n_violations
                        
                        if violating_mask.any():
                            # Penalty: (epsilon - D_ij)^2 for violating pairs
                            violations = D_pred_b_nodiag[violating_mask]
                            penalty = (repel_epsilon_st - violations).pow(2).mean()
                            repel_per_set[b] = penalty  # ← STORE IN TENSOR
                        # else: repel_per_set[b] stays 0.0 (already initialized)

                    # Gate by low-noise + conditional (same as Gram)
                    # geo_gate is already computed above in Gram loss section
                    gate_sum = geo_gate.sum().clamp_min(1.0)
                    L_repel = (repel_per_set * geo_gate).sum() / gate_sum  # ← GATED AGGREGATION

                # if DEBUG and (global_step % LOG_EVERY == 0):
                if False:
                    print(f"[REPEL] L_repel={L_repel.item():.6f} | epsilon={repel_epsilon_st:.6f}")
                    print(f"[REPEL] Samples checked: {repel_debug['samples_checked']}, Total pairs: {repel_debug['total_pairs']}, Violating: {repel_debug['violating_pairs']}")
                    print(f"[REPEL] Min predicted distance across all samples: {repel_debug['min_dist']:.6f}")
                
                # ==================== NEW: SHAPE/ANISOTROPY LOSS (ST ONLY) ====================
                # Match eigenvalue ratio of predicted Gram to target Gram
                if False:
                    if DEBUG and (global_step % LOG_EVERY == 0):
                        print(f"\n[SHAPE] Computing anisotropy/shape loss for ST batch...")

                    shape_per_set = torch.zeros(B_rep, device=device)  # ← INITIALIZE HERE
                    shape_debug = {"skipped_too_few": 0, "skipped_exception": 0, "computed": 0}

                    for b in range(B_rep):
                        m_b = mask[b].bool()
                        n_valid = m_b.sum().item()
                        if n_valid < 3:  # Need at least 3 points for meaningful eigenvalues
                            shape_debug["skipped_too_few"] += 1
                            continue
                        
                        # ROTATION-INVARIANT 2D PROJECTION (PCA)
                        V_b = V_geom[b, m_b]  # (n_valid, 32)
                        V_b_c = V_b - V_b.mean(0, keepdim=True)  # center
                        Cov_b = V_b_c.T @ V_b_c / n_valid  # (32, 32) covariance
                        eigvals_cov, eigvecs_cov = torch.linalg.eigh(Cov_b)  # ascending order
                        U2 = eigvecs_cov[:, -2:]  # top-2 eigenvectors
                        V_b_2d = V_b_c @ U2  # project to principal 2D plane (n_valid, 2)
                        
                        Gp_b = V_b_2d @ V_b_2d.T  # Recompute Gram in 2D space
                        
                        Gt_b = Gt[b][m_b][:, m_b]   

                        # Compute eigenvalues (symmetric, so use eigvalsh)
                        try:
                            eig_pred = torch.linalg.eigvalsh(Gp_b)  # (n_valid,), ascending order
                            eig_targ = torch.linalg.eigvalsh(Gt_b)  # (n_valid,), ascending order
                            
                            # Clamp from below to avoid log(0)
                            eig_pred = eig_pred.clamp_min(1e-8)
                            eig_targ = eig_targ.clamp_min(1e-8)
                            
                            # Take top 2 eigenvalues (largest)
                            lambda1_pred = eig_pred[-1]
                            lambda2_pred = eig_pred[-2]
                            lambda1_targ = eig_targ[-1]
                            lambda2_targ = eig_targ[-2]
                            
                            # Anisotropy ratio (how elongated vs round)
                            r_pred = lambda1_pred / lambda2_pred.clamp_min(1e-8)
                            r_targ = lambda1_targ / lambda2_targ.clamp_min(1e-8)
                            
                            # Log-ratio penalty (scale-invariant)
                            log_ratio_loss = (torch.log(r_pred) - torch.log(r_targ)).pow(2)
                            shape_per_set[b] = log_ratio_loss  # ← STORE IN TENSOR
                            shape_debug["computed"] += 1
                            
                            # DEBUG: Print first few samples
                            if DEBUG and (global_step % LOG_EVERY == 0) and shape_debug["computed"] <= 3:
                                print(f"  [SHAPE sample {b}] n_valid={n_valid}")
                                print(f"    λ_pred: {lambda1_pred:.4f}, {lambda2_pred:.4f} → ratio: {r_pred:.4f}")
                                print(f"    λ_targ: {lambda1_targ:.4f}, {lambda2_targ:.4f} → ratio: {r_targ:.4f}")
                                print(f"    log_ratio_loss: {log_ratio_loss:.6f}")
                            
                        except Exception as e:
                            shape_debug["skipped_exception"] += 1
                            if DEBUG and (global_step % LOG_EVERY == 0):
                                print(f"[SHAPE] Warning: eigendecomp failed for sample {b}: {e}")
                            continue

                    # Gate by low-noise + conditional (same as Gram)
                    gate_sum = geo_gate.sum().clamp_min(1.0)
                    L_shape = (shape_per_set * geo_gate).sum() / gate_sum  # ← GATED AGGREGATION

                    if DEBUG and (global_step % LOG_EVERY == 0):
                        print(f"[SHAPE] L_shape={L_shape.item():.6f} | computed={shape_debug['computed']}, skipped_few={shape_debug['skipped_too_few']}, skipped_exc={shape_debug['skipped_exception']}")
                    # ==============================================================================


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
                if not use_sc:
                    # Safety check - should never reach here
                    continue
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


                # ===== SC MANIFOLD REGULARIZER: Weak dimension prior =====
                L_dim_sc = torch.tensor(0.0, device=device)
                with torch.autocast(device_type='cuda', enabled=False):
                    # Apply weak dimension regularizer to SC (no ground truth, use target_dim=2)
                    L_dim_sc = loss_dim(
                        V_pred=V_hat.float(),
                        V_target=None,
                        mask=mask,
                        use_target=False
                    )
                    L_dim_sc = L_dim_sc * 0.3

                # ===== SC TRIANGLE REGULARIZER: Weak guardrail (ST-independent) =====
                L_triangle_sc = torch.tensor(0.0, device=device)
                with torch.autocast(device_type='cuda', enabled=False):
                    # Use FIXED small epsilon (not ST-derived) for SC
                    L_triangle_sc = loss_triangle(
                        V_pred=V_hat.float(),
                        V_target=None,
                        mask=mask,
                        use_target=False,
                        hinge_mode=True,
                        epsilon=triangle_epsilon_sc  # Fixed, ST-independent
                    )
                    # Scale down further for SC (very weak guardrail)
                    L_triangle_sc = L_triangle_sc * 0.1


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


            L_total = (WEIGHTS['score'] * score_multiplier * L_score +
                                WEIGHTS['gram'] * L_gram +
                                WEIGHTS['gram_scale'] * L_gram_scale +
                                WEIGHTS['heat'] * L_heat +
                                WEIGHTS['sw_st'] * L_sw_st +
                                WEIGHTS['sw_sc'] * L_sw_sc +
                                WEIGHTS['overlap'] * L_overlap +
                                WEIGHTS['ordinal_sc'] * L_ordinal_sc + 
                                WEIGHTS['st_dist'] * L_st_dist +
                                WEIGHTS['edm_tail'] * L_edm_tail +      
                                WEIGHTS['gen_align'] * L_gen_align +
                                WEIGHTS['dim'] * L_dim +
                                WEIGHTS['triangle'] * L_triangle +
                                WEIGHTS['knn_nca'] * L_knn_nca +
                                WEIGHTS['radial'] * L_radial +
                                WEIGHTS['repel'] * L_repel +      # NEW: ST repulsion
                                WEIGHTS['shape'] * L_shape + 
                                WEIGHTS['edge'] * L_edge +           # NEW
                                WEIGHTS['topo'] * L_topo +           # NEW
                                WEIGHTS['shape_spec'] * L_shape_spec )       # NEW: ST shape/anisotropy

            
            # Add SC dimension prior if this is an SC batch
            if is_sc:
                L_total = L_total + WEIGHTS['dim'] * L_dim_sc + WEIGHTS['triangle'] * L_triangle_sc
    
            
            # ==================== GRADIENT PROBE (IMPROVED) ====================
            # if DEBUG and (global_step % LOG_EVERY == 0):
            if USE_AUTOBALANCE and (global_step >= AUTOBALANCE_START) and (global_step % AUTOBALANCE_EVERY == 0):
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
                # ST-side gradient norms
                vgn_gram = vhat_gn(L_gram) if not is_sc else 0.0
                vgn_gram_scale = vhat_gn(L_gram_scale) if not is_sc else 0.0
                vgn_heat = vhat_gn(L_heat) if not is_sc else 0.0
                vgn_swst = vhat_gn(L_sw_st) if not is_sc else 0.0
                vgn_stdist = vhat_gn(L_st_dist) if not is_sc else 0.0
                vgn_edm = vhat_gn(L_edm_tail) if not is_sc else 0.0
                vgn_align = vhat_gn(L_gen_align) if not is_sc else 0.0
                vgn_radial = vhat_gn(L_radial) if not is_sc else 0.0

                # NEW: Gradient probes for new losses
                vgn_edge = vhat_gn(L_edge) if not is_sc else 0.0
                vgn_topo = vhat_gn(L_topo) if not is_sc else 0.0
                vgn_shape_spectrum = vhat_gn(L_shape_spec) if not is_sc else 0.0
 
                # SC-side gradient norms
                vgn_swsc = vhat_gn(L_sw_sc) if is_sc else 0.0
                vgn_ord = vhat_gn(L_ordinal_sc) if is_sc else 0.0
                vgn_overlap = vhat_gn(L_overlap) if is_sc else 0.0
 
                # Add batch type label
                batch_type = 'SC' if is_sc else 'ST'
 
                if not is_sc:
                    print(f"[vhatprobe][{batch_type}] score={vgn_score:.3e} gram={vgn_gram:.3e} gram_scale={vgn_gram_scale:.3e} "
                        f"heat={vgn_heat:.3e} sw_st={vgn_swst:.3e} st_dist={vgn_stdist:.3e} "
                        f"edm={vgn_edm:.3e} align={vgn_align:.3e} radial={vgn_radial:.3e}"
                        f"edge={vgn_edge:.3e} topo={vgn_topo:.3e} shape_spec={vgn_shape_spectrum:.3e}")
                else:
                    print(f"[vhatprobe][{batch_type}] score={vgn_score:.3e} sw_sc={vgn_swsc:.3e} "
                        f"overlap={vgn_overlap:.3e} ord={vgn_ord:.3e}")
                
                # Seed EMAs from first ST batch (prevents wrong-direction updates)
                if (global_step == 0) and (not is_sc):
                    ema_grads['score'] = max(vgn_score, 1e-12)
                    ema_grads['gram']  = max(vgn_gram,  1e-12)
                    ema_grads['gram_scale'] = max(vgn_gram_scale, 1e-12)
                    ema_grads['heat']  = max(vgn_heat,  1e-12)
                    ema_grads['sw_st'] = max(vgn_swst,  1e-12)
                    ema_grads['st_dist'] = max(vgn_stdist, 1e-12)
                    ema_grads['edm_tail'] = max(vgn_edm, 1e-12)
                    ema_grads['gen_align'] = max(vgn_align, 1e-12)
                    ema_grads['radial'] = max(vgn_radial, 1e-12)
                    ema_grads['edge'] = max(vgn_edge, 1e-12)
                    ema_grads['topo'] = max(vgn_topo, 1e-12)
                    ema_grads['shape_spec'] = max(vgn_shape_spectrum, 1e-12)
                    if DEBUG:
                        print(f"[ema_init] score={ema_grads['score']:.3e} gram={ema_grads['gram']:.3e} "
                            f"gram_scale={ema_grads['gram_scale']:.3e} heat={ema_grads['heat']:.3e} "
                            f"sw_st={ema_grads['sw_st']:.3e} st_dist={ema_grads['st_dist']:.3e} "
                            f"edm={ema_grads['edm_tail']:.3e} align={ema_grads['gen_align']:.3e} "
                            f"radial={ema_grads['radial']:.3e}")

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
                    # ST-side EMA updates for all geometric losses
                    ema_grads['score'] = EMA_BETA * ema_grads['score'] + (1 - EMA_BETA) * (vgn_score + 1e-12)
                    ema_grads['gram']  = EMA_BETA * ema_grads['gram']  + (1 - EMA_BETA) * (vgn_gram  + 1e-12)
                    ema_grads['gram_scale'] = EMA_BETA * ema_grads['gram_scale'] + (1 - EMA_BETA) * (vgn_gram_scale + 1e-12)
                    ema_grads['heat']  = EMA_BETA * ema_grads['heat']  + (1 - EMA_BETA) * (vgn_heat  + 1e-12)
                    ema_grads['sw_st'] = EMA_BETA * ema_grads['sw_st'] + (1 - EMA_BETA) * (vgn_swst + 1e-12)
                    ema_grads['st_dist'] = EMA_BETA * ema_grads['st_dist'] + (1 - EMA_BETA) * (vgn_stdist + 1e-12)
                    ema_grads['edm_tail'] = EMA_BETA * ema_grads['edm_tail'] + (1 - EMA_BETA) * (vgn_edm + 1e-12)
                    ema_grads['gen_align'] = EMA_BETA * ema_grads['gen_align'] + (1 - EMA_BETA) * (vgn_align + 1e-12)
                    ema_grads['radial'] = EMA_BETA * ema_grads['radial'] + (1 - EMA_BETA) * (vgn_radial + 1e-12)
                else:
                    # SC-side EMA updates
                    ema_sc['score']      = EMA_BETA * ema_sc['score']      + (1 - EMA_BETA) * (vgn_score   + 1e-12)
                    ema_sc['sw_sc']      = EMA_BETA * ema_sc['sw_sc']      + (1 - EMA_BETA) * (vgn_swsc    + 1e-12)
                    ema_sc['overlap']    = EMA_BETA * ema_sc['overlap']    + (1 - EMA_BETA) * (vgn_overlap + 1e-12)
                    ema_sc['ordinal_sc'] = EMA_BETA * ema_sc['ordinal_sc'] + (1 - EMA_BETA) * (vgn_ord     + 1e-12)

                # Adjust weights (ST batches only, after warmup, at intervals)
                if (not is_sc) and (global_step >= AUTOBALANCE_START) and (global_step % AUTOBALANCE_EVERY == 0):
                    # Define bounds for each loss (min_weight, max_weight)
                    # Conservative bounds prevent any single loss from dominating
                    # bounds = {
                    #     'gram': (6.0, 20.0),          # Original range
                    #     'gram_scale': (0.05, 0.8),    # Scale regularizer - keep modest
                    #     'heat': (0.6, 3.0),           # Original range
                    #     'sw_st': (0.8, 3.0),          # Original range
                    #     'st_dist': (0.01, 0.5),       # Distance classification - moderate
                    #     'edm_tail': (0.05, 0.8),      # EDM tail - moderate
                    #     'gen_align': (0.05, 0.6),     # Procrustes alignment - moderate
                    #     'radial': (0.2, 2.0),         # Radial histogram - moderate
                    #     'edge': (0.3, 1.5),           # Keep edge consistency moderate
                    #     'topo': (0.2, 1.0),           # Topology can be expensive, keep check
                    #     'shape_spec': (0.2, 1.0), # Shape spectrum bounds
                    # }

                    bounds = {
                        # Global losses - TIGHTER upper bounds (don't let them dominate)
                        'gram': (0.3, 3.0),         # was (6.0, 20.0) - much lower ceiling
                        'gram_scale': (0.02, 0.5),  # was (0.05, 0.8)
                        'heat': (0.3, 2.0),         # was (0.6, 3.0)
                        'sw_st': (0.2, 1.5),        # was (0.8, 3.0)
                        'radial': (0.1, 1.0),       # was (0.2, 2.0)
                        
                        # Local losses - WIDER bounds (let them grow stronger)
                        'edge': (0.5, 4.0),         # was (0.3, 1.5) - allow much higher
                        'topo': (0.3, 3.0),         # was (0.2, 1.0) - allow much higher
                        
                        # Others - keep reasonable
                        'st_dist': (0.01, 0.4),
                        'edm_tail': (0.05, 0.6),
                        'gen_align': (0.03, 0.4),
                        'shape_spec': (0.1, 0.8),
                    }
 
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
                        # Update all ST-side autobalanced losses
                        for key in ['gram', 'gram_scale', 'heat', 'sw_st', 'st_dist',
                                    'edm_tail', 'gen_align', 'radial', 'edge', 'topo', 'shape_spec']:
                            _update(key)
 
                    try:
                        if dist.is_available() and dist.is_initialized():
                            # Broadcast all updated weights to other ranks
                            w = torch.tensor([
                                WEIGHTS['gram'], WEIGHTS['gram_scale'], WEIGHTS['heat'],
                                WEIGHTS['sw_st'], WEIGHTS['st_dist'], WEIGHTS['edm_tail'],
                                WEIGHTS['gen_align'], WEIGHTS['radial'],
                                WEIGHTS['edge'], WEIGHTS['topo'], WEIGHTS['shape_spec']
                            ], device=device, dtype=torch.float32)

                            dist.broadcast(w, src=0)

                            (WEIGHTS['gram'], WEIGHTS['gram_scale'], WEIGHTS['heat'],
                             WEIGHTS['sw_st'], WEIGHTS['st_dist'], WEIGHTS['edm_tail'],
                             WEIGHTS['gen_align'], WEIGHTS['radial'],
                             WEIGHTS['edge'], WEIGHTS['topo'], WEIGHTS['shape_spec']) = map(float, w.tolist())
                    except Exception:
                        pass
 
                    if DEBUG and (rank == 0):
                        print(f"[autobalance @ step {global_step}] "
                            f"gram={WEIGHTS['gram']:.3g} gram_scale={WEIGHTS['gram_scale']:.3g} "
                            f"heat={WEIGHTS['heat']:.3g} sw_st={WEIGHTS['sw_st']:.3g} "
                            f"st_dist={WEIGHTS['st_dist']:.3g} edm={WEIGHTS['edm_tail']:.3g} "
                            f"align={WEIGHTS['gen_align']:.3g} radial={WEIGHTS['radial']:.3g}")

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
            epoch_losses['st_dist'] += L_st_dist.item()
            epoch_losses['gram'] += L_gram.item()
            epoch_losses['gram_scale'] += L_gram_scale.item()
            epoch_losses['knn_nca'] += L_knn_nca.item()
            epoch_losses['heat'] += L_heat.item()
            epoch_losses['sw_st'] += L_sw_st.item()
            epoch_losses['sw_sc'] += L_sw_sc.item()
            epoch_losses['overlap'] += L_overlap.item()
            epoch_losses['ordinal_sc'] += L_ordinal_sc.item()
            epoch_losses['edm_tail'] += L_edm_tail.item()
            epoch_losses['gen_align'] += L_gen_align.item()
            epoch_losses['dim'] += L_dim.item()
            epoch_losses['triangle'] += L_triangle.item()
            epoch_losses['radial'] += L_radial.item()
            epoch_losses['repel'] += L_repel.item()
            epoch_losses['shape'] += L_shape.item()
            epoch_losses['edge'] += L_edge.item()
            epoch_losses['topo'] += L_topo.item()
            epoch_losses['shape_spec'] += L_shape_spec.item()




            def _is_rank0():
                return (not dist.is_initialized()) or dist.get_rank() == 0

            if L_overlap.item() > 0:  
                c_overlap += 1       
            
            n_batches += 1
            global_step += 1
            # batch_pbar.update(1)
            # Sync global_step across ranks to prevent divergence
            if dist.is_initialized():
                global_step_tensor = torch.tensor([global_step], device=device, dtype=torch.long)
                dist.broadcast(global_step_tensor, src=0)
                global_step = int(global_step_tensor.item())
        
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
                
                print(f"[NEW LOSSES] L_repel={L_repel.item():.3e} L_shape={L_shape.item():.3e}")

                
        #============ END OF EPOCH SUMMARY ============
        if use_sc:
            print(f"\n[Epoch {epoch+1}] ST batches: {st_batches}, SC batches: {sc_batches}")
        else:
            print(f"\n[Epoch {epoch+1}] ST batches: {st_batches} (SC disabled)")

        should_stop = False 

        # Print loss summary
        if fabric is None or fabric.is_global_zero:
            # Average the accumulated losses
            avg_score = epoch_losses['score'] / max(n_batches, 1)
            avg_gram = epoch_losses['gram'] / max(n_batches, 1)
            avg_total = epoch_losses['total'] / max(n_batches, 1)
            
            # print(f"[Epoch {epoch+1}] Avg Losses: score={avg_score:.4f}, gram={avg_gram:.4f}, total={avg_total:.4f}")
            
            # Print detailed losses every 5 epochs, simple summary otherwise
            if (epoch + 1) % 5 == 0:
                avg_gram_scale = epoch_losses['gram_scale'] / max(n_batches, 1)
                avg_heat = epoch_losses['heat'] / max(n_batches, 1)
                avg_sw_st = epoch_losses['sw_st'] / max(n_batches, 1)
                avg_sw_sc = epoch_losses['sw_sc'] / max(n_batches, 1)
                avg_overlap = epoch_losses['overlap'] / max(n_batches, 1)
                avg_ordinal_sc = epoch_losses['ordinal_sc'] / max(n_batches, 1)
                avg_st_dist = epoch_losses['st_dist'] / max(n_batches, 1)
                avg_edm_tail = epoch_losses['edm_tail'] / max(n_batches, 1)
                avg_gen_align = epoch_losses['gen_align'] / max(n_batches, 1)
                avg_dim = epoch_losses['dim'] / max(n_batches, 1)
                avg_triangle = epoch_losses['triangle'] / max(n_batches, 1)
                avg_radial = epoch_losses['radial'] / max(n_batches, 1)
                avg_knn_nca = epoch_losses['knn_nca'] / max(n_batches, 1)

                avg_edge = epoch_losses['edge'] / max(n_batches, 1)
                avg_topo = epoch_losses['topo'] / max(n_batches, 1)
                avg_shape_spec = epoch_losses['shape_spec'] / max(n_batches, 1)

                print(f"[Epoch {epoch+1}] DETAILED LOSSES:")
                print(f"  total={avg_total:.4f} | score={avg_score:.4f} | gram={avg_gram:.4f} | gram_scale={avg_gram_scale:.4f}")
                print(f"  heat={avg_heat:.4f} | sw_st={avg_sw_st:.4f} | sw_sc={avg_sw_sc:.4f} | knn_nca={avg_knn_nca:.4f}")
                print(f"  overlap={avg_overlap:.4f} | ordinal_sc={avg_ordinal_sc:.4f} | st_dist={avg_st_dist:.4f}")
                print(f"  edm_tail={avg_edm_tail:.4f} | gen_align={avg_gen_align:.4f}")
                print(f"  dim={avg_dim:.4f} | triangle={avg_triangle:.4f} | radial={avg_radial:.4f}")
                print(f"  edge={avg_edge:.4f} | topo={avg_topo:.4f} | shape_spec={avg_shape_spec:.4f}")

            else:
                print(f"[Epoch {epoch+1}] Avg Losses: score={avg_score:.4f}, gram={avg_gram:.4f}, total={avg_total:.4f}")

            # Add this summary print every 10 epochs
            # if epoch % 10 == 0:
            #     # Compute averages
            #     cv_avg = (epoch_cv_sum / max(epoch_nca_count, 1)) * 100  # Convert to percentage
            #     loss_nca = epoch_nca_loss_sum / max(epoch_nca_count, 1)
                
            #     # Get Q entropy average from global accumulator
            #     from utils_et import _nca_qent_accum
            #     if len(_nca_qent_accum) > 0:
            #         qent_avg = (sum(_nca_qent_accum) / len(_nca_qent_accum)) * 100  # Convert to %
            #         _nca_qent_accum.clear()  # Reset for next epoch
            #     else:
            #         qent_avg = 100.0  # Default if no data
                
                # print(f"\n{'='*60}")
                # print(f"EPOCH {epoch} PROGRESS CHECK")
                # print(f"{'='*60}")
                # print(f"Metric          | Current | Target  | Status")
                # print(f"----------------|---------|---------|--------")
                # print(f"d² CV          | {cv_avg:.1f}%   | >30%    | {'✓' if cv_avg > 30 else '⚠️'}")
                # print(f"Q entropy ratio| {qent_avg:.1f}% | <70%    | {'✓' if qent_avg < 70 else '⚠️'}")
                # print(f"NCA loss       | {loss_nca:.2f}  | <3.5    | {'✓' if loss_nca < 3.5 else '⚠️'}")
                # print(f"{'='*60}\n")
            
            if enable_early_stop and (epoch + 1) >= early_stop_min_epochs:
                # Use total weighted loss as validation metric
                # This includes all geometry regularizers (dim, triangle, radial)
                val_metric = avg_total

                
                # Check for improvement
                if val_metric < early_stop_best:
                    rel_improv = (early_stop_best - val_metric) / max(early_stop_best, 1e-8)
                    
                    if rel_improv > early_stop_threshold:
                        # Significant improvement
                        print(f"[Early Stop] Improvement: {rel_improv:.2%} (val_metric: {early_stop_best:.4f} → {val_metric:.4f})")
                        early_stop_best = val_metric
                        early_stop_no_improve = 0
                    else:
                        # Minor improvement, doesn't count
                        early_stop_no_improve += 1
                        print(f"[Early Stop] Minor improvement: {rel_improv:.2%} < {early_stop_threshold:.1%} "
                            f"(no_improve: {early_stop_no_improve}/{early_stop_patience})")
                else:
                    # No improvement
                    early_stop_no_improve += 1
                    print(f"[Early Stop] No improvement: val_metric={val_metric:.4f} >= best={early_stop_best:.4f} "
                        f"(no_improve: {early_stop_no_improve}/{early_stop_patience})")
                
                # Check if we should stop
                if early_stop_no_improve >= early_stop_patience:
                    should_stop = True
                    early_stopped = True
                    early_stop_epoch = epoch + 1
                    print(f"\n{'='*80}")
                    print(f"[Early Stop] PLATEAU DETECTED after {early_stop_epoch} epochs")
                    print(f"[Early Stop] Best val_metric: {early_stop_best:.4f}")
                    print(f"[Early Stop] No improvement for {early_stop_patience} epochs")
                    print(f"{'='*80}\n")

            elif enable_early_stop and (epoch + 1) < early_stop_min_epochs:
                # Just track best, don't stop yet
                val_metric = avg_total
                if val_metric < early_stop_best:
                    early_stop_best = val_metric
                    print(f"[Early Stop] Warmup: best={early_stop_best:.4f} (min_epochs not reached)")
          
        # Broadcast stop decision to ALL ranks
        if fabric is not None:
            # Use dist.broadcast directly - fabric.broadcast is buggy
            should_stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.long, device=fabric.device)
            dist.broadcast(should_stop_tensor, src=0)
            should_stop = bool(should_stop_tensor.item())

        # ALL ranks break together
        if should_stop:

            del st_iter
            if sc_iter is not None:
                del sc_iter
            break

        # Broadcast stop decision to ALL ranks
        if fabric is not None:
            should_stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=fabric.device)
            fabric.broadcast(should_stop_tensor, src=0)
            should_stop = should_stop_tensor.item() > 0.5

        
        scheduler.step()

        if epoch + 1 == HEAT_WARMUP_EPOCHS:
            # if rank == 0:
            #     print(f"\n[Schedule] Enabling heat loss at epoch {epoch+1}")
            WEIGHTS['heat'] = HEAT_TARGET_WEIGHT

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
            if key in ('gram', 'gram_scale', 'heat', 'sw_st', 'cone', 'edm_tail', 'gen_align', 'dim', 'triangle', 'radial', 'st_dist', 'edge', 'topo', 'shape_spec'):
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
            WEIGHTS['score']    * epoch_losses['score']    +
            WEIGHTS['gram']     * epoch_losses['gram']     +
            WEIGHTS['gram_scale'] * epoch_losses['gram_scale'] +
            WEIGHTS['knn_nca'] * epoch_losses['knn_nca'] +
            WEIGHTS['heat']     * epoch_losses['heat']     +
            WEIGHTS['sw_st']    * epoch_losses['sw_st']    +
            WEIGHTS['sw_sc']    * epoch_losses['sw_sc']    +
            WEIGHTS['overlap']  * epoch_losses['overlap']  +
            WEIGHTS['ordinal_sc'] * epoch_losses['ordinal_sc'] +
            WEIGHTS['st_dist']  * epoch_losses['st_dist']  +   
            WEIGHTS['edm_tail'] * epoch_losses['edm_tail'] +
            WEIGHTS['gen_align']* epoch_losses['gen_align']+
            WEIGHTS['dim']      * epoch_losses['dim']      +
            WEIGHTS['triangle'] * epoch_losses['triangle'] +
            WEIGHTS['radial']   * epoch_losses['radial']   +
            WEIGHTS['repel']    * epoch_losses['repel']    +   
            WEIGHTS['shape']    * epoch_losses['shape']     +
            WEIGHTS['shape_spec'] * epoch_losses['shape_spec'] +
            WEIGHTS['edge'] * epoch_losses['edge'] +
            WEIGHTS['topo'] * epoch_losses['topo']
        )

        # Override the history['epoch_avg']['total'] with correct value
        history['epoch_avg']['total'][-1] = epoch_losses['total']

        # reset overlap batch counter for next epoch
        debug_state['overlap_count_this_epoch'] = 0
        # -----------------------------------------------------------------------

        # DEBUG: Epoch summary
        if DEBUG:
            # Score by sigma
            if debug_state['score_bin_cnt'] is not None and debug_state['score_bin_cnt'].sum() > 0:
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
            if debug_state['score_bin_cnt'] is not None and debug_state['score_bin_cnt'].sum() > 0:
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
            if 'score_bin_sum' in debug_state and debug_state['score_bin_sum'] is not None:
                debug_state['score_bin_sum'].zero_()
                debug_state['score_bin_cnt'].zero_()

            # Add early stopping info to history
            if _is_rank0():
                if 'early_stop' not in history:
                    history['early_stop'] = {}
                
                history['early_stop']['stopped'] = early_stopped
                history['early_stop']['epoch'] = early_stop_epoch if early_stopped else n_epochs
                history['early_stop']['best_metric'] = early_stop_best if early_stopped else None
                
                if early_stopped:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"{'='*70}")
                    print(f"Stopped at epoch:     {early_stop_epoch}")
                    print(f"Best val metric:      {early_stop_best:.4f}")
                    print(f"Patience exhausted:   {early_stop_patience} epochs without improvement")
                    print(f"{'='*70}\n")
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
        if (epoch + 1) % 100 == 0:
            if fabric is None or fabric.is_global_zero:
                ckpt = {
                    'epoch': epoch,
                    'context_encoder': context_encoder.state_dict(),
                    'score_net': score_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                    'sigma_data': sigma_data,
                }
                torch.save(ckpt, os.path.join(outf, f'ckpt_epoch_{epoch+1}.pt'))
    
    # Save final checkpoint after training loop
    if fabric is None or fabric.is_global_zero:
        ckpt_final = {
            'epoch': epoch,
            'context_encoder': context_encoder.state_dict(),
            'score_net': score_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history
        }
        torch.save(ckpt_final, os.path.join(outf, 'ckpt_final.pt'))
        print(f"Saved final checkpoint at epoch {epoch+1}")

    if fabric is not None:
        fabric.barrier()

    if fabric is None or fabric.is_global_zero:
            print("Training complete!")

    print(f"[DEBUG train_stageC] Rank {fabric.global_rank if fabric else 0} - deleting loaders")
    del st_loader
    if use_sc:
        del sc_loader

    # CRITICAL: Force CUDA synchronization before exiting training
    print(f"[DEBUG train_stageC] Rank {fabric.global_rank if fabric else 0} - syncing CUDA")
    torch.cuda.synchronize()
    print(f"[DEBUG train_stageC] Rank {fabric.global_rank if fabric else 0} - CUDA synced")

    # Sync before exit
    if fabric is not None:
        print(f"[DEBUG train_stageC] Rank {fabric.global_rank} - hitting barrier")
        fabric.barrier()
        print(f"[DEBUG train_stageC] Rank {fabric.global_rank} - passed barrier, syncing CUDA again")
        torch.cuda.synchronize()
        print(f"[DEBUG train_stageC] Rank {fabric.global_rank} - about to return")

    print(f"[DEBUG train_stageC LOOP EXIT] Rank {fabric.global_rank if fabric else 0} - exited epoch loop")
    
    if fabric is not None:
        print(f"[DEBUG train_stageC POST-LOOP] Rank {fabric.global_rank} - BEFORE post-loop barrier")
        fabric.barrier()
        print(f"[DEBUG train_stageC POST-LOOP] Rank {fabric.global_rank} - AFTER post-loop barrier")


    return history if _is_rank0() else None


# ==============================================================================
# STAGE D: SC INFERENCE (SINGLE-PATCH MODE - NO STITCHING)
# ==============================================================================

def sample_sc_edm_single_patch(
    sc_gene_expr: torch.Tensor,
    encoder: "SharedEncoder",
    context_encoder: "SetEncoderContext",
    score_net: "DiffusionScoreNet",
    target_st_p95: Optional[float] = None,
    n_timesteps_sample: int = 500,
    sigma_min: float = 0.01,
    sigma_max: float = 3.0,
    guidance_scale: float = 2.0,
    eta: float = 0.0,
    device: str = "cuda",
    DEBUG_FLAG: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Stage D: Single-patch SC inference (NO patchwise stitching).
    
    This is a reference implementation that treats all SC cells as one patch.
    Useful for:
    1. Debugging Stage C without patchwise alignment artifacts
    2. Small datasets where patchwise isn't needed
    3. Comparing patchwise vs single-patch geometry
    
    Returns:
        Dictionary with:
        - 'D_edm': (n_sc, n_sc) Euclidean distance matrix
        - 'coords': (n_sc, D_latent) raw coordinates
        - 'coords_canon': (n_sc, D_latent) canonicalized coordinates
    """
    import torch.nn.functional as F
    import utils_et as uet
    
    print(f"\n{'='*72}")
    print("STAGE D — SINGLE-PATCH SC INFERENCE (NO PATCHWISE STITCHING)")
    print(f"{'='*72}")
    
    encoder.eval()
    context_encoder.eval()
    score_net.eval()
    
    n_sc = sc_gene_expr.shape[0]
    D_latent = score_net.D_latent
    
    if DEBUG_FLAG:
        print(f"[cfg] n_sc={n_sc}  D_latent={D_latent}")
        print(f"[cfg] timesteps={n_timesteps_sample}  guidance_scale={guidance_scale}")
        print(f"[cfg] sigma_min={sigma_min}  sigma_max={sigma_max}")
    
    # 1) Encode all SC cells
    encode_bs = 1024
    Z_chunks = []
    for i in range(0, n_sc, encode_bs):
        z = encoder(sc_gene_expr[i:i + encode_bs].to(device)).detach()
        Z_chunks.append(z)
    Z_all = torch.cat(Z_chunks, dim=0).to(device)  # (n_sc, h)
    
    if DEBUG_FLAG:
        print(f"[ENC] Z_all shape={tuple(Z_all.shape)}")
    
    # 2) Build single-patch context
    Z_set = Z_all.unsqueeze(0)  # (1, n_sc, h)
    mask = torch.ones(1, n_sc, dtype=torch.bool, device=device)
    H = context_encoder(Z_set, mask)  # (1, n_sc, c_dim)

    # Apply CORAL if available
    if hasattr(context_encoder, 'coral_transform') and context_encoder.coral_transform is not None:
        H = context_encoder.coral_transform(H)
    
    # 3) Diffusion sampling
    sigmas = torch.exp(torch.linspace(
        torch.log(torch.tensor(sigma_max, device=device)),
        torch.log(torch.tensor(sigma_min, device=device)),
        n_timesteps_sample,
        device=device,
    ))
    
    V_t = torch.randn(1, n_sc, D_latent, device=device) * sigmas[0]
    
    if DEBUG_FLAG:
        print(f"\n[SAMPLE] Starting diffusion with {n_timesteps_sample} steps...")
    
    with torch.no_grad():
        for t_idx in range(n_timesteps_sample):
            sigma_t = sigmas[t_idx]
            t_norm = torch.tensor([[t_idx / float(n_timesteps_sample - 1)]], device=device)
            
            # Classifier-free guidance
            H_null = torch.zeros_like(H)
            eps_uncond = score_net(V_t, t_norm, H_null, mask)
            eps_cond = score_net(V_t, t_norm, H, mask)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            if DEBUG_FLAG and t_idx % 100 == 0:
                print(f"  [STEP] t={t_idx:3d}/{n_timesteps_sample} sigma={float(sigma_t):.4f}")
            
            # DDIM/EDM step
            if t_idx < n_timesteps_sample - 1:
                sigma_next = sigmas[t_idx + 1]
                V_0_pred = V_t - sigma_t * eps
                V_t = V_0_pred + (sigma_next / sigma_t) * (V_t - V_0_pred)
                if eta > 0:
                    noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma_t**2, min=0))
                    V_t = V_t + noise_scale * torch.randn_like(V_t)
            else:
                V_t = V_t - sigma_t * eps
    
    # 4) Canonicalize: center only, no RMS scaling (match training)
    V_final = V_t.squeeze(0)  # (n_sc, D_latent)
    V_canon = V_final - V_final.mean(dim=0, keepdim=True)
    
    # 5) Compute EDM
    D_edm = torch.cdist(V_canon, V_canon)  # (n_sc, n_sc)

    triu_mask = torch.triu(torch.ones_like(D_edm, dtype=torch.bool), diagonal=1)

    
    # 6) Optional: rescale to ST p95
    if target_st_p95 is not None:
        current_p95 = D_edm[triu_mask].quantile(0.95).item()
        scale_factor = target_st_p95 / (current_p95 + 1e-8)
        D_edm = D_edm * scale_factor
        V_canon = V_canon * scale_factor
        
        if DEBUG_FLAG:
            print(f"\n[SCALE] Rescaled to match ST p95={target_st_p95:.4f} (factor={scale_factor:.4f})")
    
    # 7) Stats
    if DEBUG_FLAG:
        print(f"\n[RESULT] EDM shape: {tuple(D_edm.shape)}")
        print(f"[RESULT] Coords RMS: {V_canon.pow(2).mean().sqrt().item():.4f}")
        D_upper = D_edm[triu_mask]
        print(f"[RESULT] Distance stats: "
              f"p50={D_upper.quantile(0.50).item():.4f} "
              f"p95={D_upper.quantile(0.95).item():.4f} "
              f"max={D_upper.max().item():.4f}")
        print(f"{'='*72}\n")
    
    return {
        'D_edm': D_edm,
        'coords': V_final,
        'coords_canon': V_canon,
    }


# ==============================================================================
# STAGE D: SC INFERENCE (PATCH-BASED GLOBAL ALIGNMENT)
# ==============================================================================

def sample_sc_edm_patchwise(
    sc_gene_expr: torch.Tensor,
    encoder: "SharedEncoder",
    context_encoder: "SetEncoderContext",
    score_net: "DiffusionScoreNet",
    sigma_data: float,
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
    fixed_patch_graph: Optional[dict] = None,
    coral_params: Optional[Dict] = None
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

    import random


    print(f"[PATCHWISE] FINAL IDK WHAT IS GOING ON Running on device={device}, starting inference...", flush=True)

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
    # 3) Define overlapping patches S_k (or reload from file)
    # ------------------------------------------------------------------
    if fixed_patch_graph is not None:
        # RELOAD existing patch graph - do NOT rebuild
        patch_indices = [p.to(torch.long) for p in fixed_patch_graph["patch_indices"]]
        memberships = fixed_patch_graph["memberships"]
        K = len(patch_indices)
        print(f"[PATCHWISE] Loaded fixed patch graph with {K} patches")
    else:
        # BUILD new patch graph
        n_patches_est = int(math.ceil((coverage_per_cell * n_sc) / patch_size))
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
        print(f"[PATCHWISE] Built new patch graph with {K} patches")
        # Reload pre-computed patch graph
        patch_indices = [p.to(torch.long) for p in fixed_patch_graph["patch_indices"]]
        memberships = fixed_patch_graph["memberships"]
        K = len(patch_indices)
        if DEBUG_FLAG:
            print(f"[PATCH] Loaded fixed patch graph: K={K} patches")

        # ===================================================================
        # DIAGNOSTIC A: PATCH GRAPH ANALYSIS
        # ===================================================================
        if DEBUG_FLAG:
            print("\n" + "="*70)
            print("DIAGNOSTIC A: PATCH GRAPH ANALYSIS")
            print("="*70)
            
            stats = compute_overlap_graph_stats(patch_indices, n_sc, min_overlap=20)
            overlap_sizes = stats['overlap_sizes']
            
            # A1: Overlap size distribution
            print(f"\n[A1] Overlap Statistics:")
            print(f"  Total patch pairs: {len(overlap_sizes)}")
            print(f"  Mean overlap: {np.mean(overlap_sizes):.1f}")
            print(f"  Median overlap: {np.median(overlap_sizes):.1f}")
            print(f"  % pairs with ≥20: {100*np.mean(np.array(overlap_sizes) >= 20):.1f}%")
            print(f"  % pairs with ≥50: {100*np.mean(np.array(overlap_sizes) >= 50):.1f}%")
            print(f"  % pairs with ≥100: {100*np.mean(np.array(overlap_sizes) >= 100):.1f}%")
            
            # Plot overlap distribution
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].hist(overlap_sizes, bins=50, edgecolor='black', alpha=0.7)
            axes[0].axvline(20, color='red', linestyle='--', label='Min overlap=20')
            axes[0].set_xlabel('Overlap Size')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Patch Overlap Distribution')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # A2: Connectivity
            print(f"\n[A2] Patch Graph Connectivity (min_overlap=20):")
            print(f"  # Connected components: {stats['n_components']}")
            print(f"  Giant component size: {stats['giant_component_size']}/{K} ({stats['giant_component_frac']*100:.1f}%)")
            
            # Plot component sizes
            if stats['n_components'] > 1:
                print("  ⚠️  WARNING: Patch graph is fragmented!")
            
            axes[1].bar(['Giant', 'Other'], 
                    [stats['giant_component_size'], K - stats['giant_component_size']],
                    color=['green', 'red'], alpha=0.7)
            axes[1].set_ylabel('# Patches')
            axes[1].set_title(f'Patch Graph Components (n={stats["n_components"]})')
            axes[1].grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
            print("="*70 + "\n")

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
    # DEBUG 1: Patch graph summary
    # ------------------------------------------------------------------
    if DEBUG_FLAG:
        cover_counts = torch.tensor([len(memberships[i]) for i in range(n_sc)], dtype=torch.float32)
        patch_sizes = torch.tensor([len(S_k) for S_k in patch_indices], dtype=torch.float32)

        print(f"\n[PATCH] final n_patches={K}")
        print(f"[PATCH] per-cell coverage stats: "
              f"min={cover_counts.min().item():.1f} "
              f"p25={cover_counts.quantile(0.25).item():.1f} "
              f"p50={cover_counts.quantile(0.50).item():.1f} "
              f"p75={cover_counts.quantile(0.75).item():.1f} "
              f"max={cover_counts.max().item():.1f}")
        print(f"[PATCH] cells with coverage==1: {(cover_counts==1).sum().item()} "
              f"({(cover_counts==1).float().mean().item()*100:.1f}%)")
        print(f"[PATCH] patch size stats: "
              f"min={patch_sizes.min().item():.0f} "
              f"p25={patch_sizes.quantile(0.25).item():.0f} "
              f"p50={patch_sizes.quantile(0.50).item():.0f} "
              f"p75={patch_sizes.quantile(0.75).item():.0f} "
              f"max={patch_sizes.max().item():.0f}")
        
        # Save patch graph for reproducibility testing
        incidence = torch.zeros(n_sc, K, dtype=torch.bool)
        for k_idx, S_k in enumerate(patch_indices):
            incidence[S_k, k_idx] = True
        
        torch.save(
            {
                "patch_indices": [p.cpu() for p in patch_indices],
                "memberships": memberships,
                "incidence": incidence.cpu(),
                "n_sc": n_sc,
                "patch_size": patch_size,
            },
            f"debug_patch_graph_seed.pt",
        )
        print("[DEBUG] Saved patch graph to debug patch graph")

    # ------------------------------------------------------------------
    # 4) For each patch: run diffusion sampling (all points free),
    #    then canonicalize to match training.
    # ------------------------------------------------------------------
    # sigmas = torch.exp(torch.linspace(
    #     torch.log(torch.tensor(sigma_max, device=device)),
    #     torch.log(torch.tensor(sigma_min, device=device)),
    #     n_timesteps_sample,
    #     device=device,
    # ))  # (T,)

    # EDM Karras sigma schedule
    sigmas = uet.edm_sigma_schedule(n_timesteps_sample, sigma_min, sigma_max, rho=7.0, device=device)

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
            
            # APPLY CORAL TRANSFORMATION IF ENABLED
            if coral_params is not None:
                from core_models_et_p3 import GEMSModel
                H_k = GEMSModel.apply_coral_transform(
                    H_k,
                    mu_sc=coral_params['mu_sc'],
                    A=coral_params['A'],
                    B=coral_params['B'],
                    mu_st=coral_params['mu_st']
                )

            V_t = torch.randn(1, m_k, D_latent, device=device) * sigmas[0]

            # for t_idx in range(n_timesteps_sample):
            #     sigma_t = sigmas[t_idx]
            #     t_norm = torch.tensor([[t_idx / float(n_timesteps_sample - 1)]],
            #                           device=device)

            # sigma_data = 1.0  # Use stored or estimated value
            
            for i in range(n_timesteps_sample):
                sigma_cur = sigmas[i]
                sigma_next = sigmas[i + 1] if i < n_timesteps_sample - 1 else torch.tensor(0.0, device=device)
                
                # EDM denoised prediction with CFG
                H_null = torch.zeros_like(H_k)
                x0_uncond = score_net.forward_edm(V_t, sigma_cur.expand(1), H_null, mask_k, sigma_data)
                x0_cond = score_net.forward_edm(V_t, sigma_cur.expand(1), H_k, mask_k, sigma_data)
                x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
                
                # Debug on first patch
                if DEBUG_FLAG and k == 0 and i < 3:
                    du = x0_uncond.norm(dim=[1, 2]).mean().item()
                    dc = x0_cond.norm(dim=[1, 2]).mean().item()
                    diff = (x0_cond - x0_uncond).norm(dim=[1, 2]).mean().item()
                    print(f"  [PATCH0] i={i:3d} sigma={float(sigma_cur):.4f} "
                          f"||x0_u||={du:.3f} ||x0_c||={dc:.3f} ||diff||={diff:.3f}")
                
                # EDM update
                if sigma_next > 0:
                    d_cur = (V_t - x0_pred) / sigma_cur
                    V_t = V_t + (sigma_next - sigma_cur) * d_cur
                    
                    if eta > 0:
                        noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma_cur**2, min=0))
                        V_t = V_t + noise_scale * torch.randn_like(V_t)
                else:
                    V_t = x0_pred
                
            # for t_idx in range(n_timesteps_sample):
            #     sigma_t = sigmas[t_idx]
            #     t_norm = torch.tensor([[1.0 - t_idx / float(n_timesteps_sample - 1)]], device=device)

            #     H_null = torch.zeros_like(H_k)
            #     eps_uncond = score_net(V_t, t_norm, H_null, mask_k)
            #     eps_cond   = score_net(V_t, t_norm, H_k,    mask_k)
            #     eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            #     # Extra debug on first patch
            #     if DEBUG_FLAG and k == 0 and t_idx < 3:
            #         du = eps_uncond.norm(dim=[1, 2]).mean().item()
            #         dc = eps_cond.norm(dim=[1, 2]).mean().item()
            #         diff = (eps_cond - eps_uncond).norm(dim=[1, 2]).mean().item()
            #         ratio = diff / (du + 1e-8)
            #         print(f"  [PATCH0] t={t_idx:3d} sigma={float(sigma_t):.4f} "
            #               f"||eps_u||={du:.3f} ||eps_c||={dc:.3f} "
            #               f"||diff||={diff:.3f} CFG_ratio={ratio:.3f}")

            #     if t_idx < n_timesteps_sample - 1:
            #         sigma_next = sigmas[t_idx + 1]
            #         V_0_pred = V_t - sigma_t * eps
            #         V_t = V_0_pred + (sigma_next / sigma_t) * (V_t - V_0_pred)
            #         if eta > 0:
            #             noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma_t**2, min=0))
            #             V_t = V_t + noise_scale * torch.randn_like(V_t)
            #     else:
            #         V_t = V_t - sigma_t * eps
    

            # if DEBUG_FLAG and (k % max(1, K // 5) == 0):
            #     rms = V_canon.pow(2).mean().sqrt().item()
            # NEW: Only center, do NOT apply unit RMS (matches training)
            V_final = V_t.squeeze(0)  # (m_k, D)
            V_centered = V_final - V_final.mean(dim=0, keepdim=True)

            # ===================================================================
            # DIAGNOSTIC B: PATCH SAMPLING QUALITY (first 3 patches only)
            # ===================================================================
            if DEBUG_FLAG and k < 3:
                print(f"\n[DIAGNOSTIC B] Patch {k} sampling analysis:")
                
                # B1: Check if patch is isotropic (blob)
                cov = torch.cov(V_centered.float().T)
                eigs = torch.linalg.eigvalsh(cov)
                aniso_ratio = float(eigs.max() / (eigs.min() + 1e-8))
                print(f"  Anisotropy ratio: {aniso_ratio:.2f} (higher=more structured)")
                
                if aniso_ratio < 2.0:
                    print(f"  ⚠️  WARNING: Patch {k} is very isotropic (blob-like)!")
            patch_coords.append(V_centered.detach().cpu())

            # DEBUG 2: Per-patch sample diagnostics
            if DEBUG_FLAG and (k < 5 or k % max(1, K // 5) == 0):
                rms = V_centered.pow(2).mean().sqrt().item()

                # Coord covariance eigs to see anisotropy / effective dimension
                cov_k = torch.cov(V_centered.float().T)
                eigs_k = torch.linalg.eigvalsh(cov_k)
                dim_eff = float((eigs_k.sum() ** 2) / (eigs_k ** 2).sum())
                aniso = float(eigs_k.max() / (eigs_k.min().clamp(min=1e-8)))

                print(f"[PATCH-SAMPLE] k={k}/{K} m_k={m_k} "
                      f"rms={rms:.3f} dim_eff={dim_eff:.2f} aniso={aniso:.1f} "
                      f"eigs_min={eigs_k.min().item():.3e} eigs_max={eigs_k.max().item():.3e}")

            if DEBUG_FLAG and (k % max(1, K // 5) == 0):
                rms = V_centered.pow(2).mean().sqrt().item()
                print(f"  [PATCH {k}/{K}] RMS={rms:.3f} (centered, natural scale)")

                # mean_norm = V_canon.mean(dim=0).norm().item()
                mean_norm = V_centered.mean(dim=0).norm().item()
                print(f"[PATCH] k={k}/{K} m_k={m_k} "
                      f"coords_rms={rms:.3f} center_norm={mean_norm:.3e}")

            if 'cuda' in device:
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # DEBUG: Save patch coords (AFTER all patches sampled, BEFORE alignment)
    # ------------------------------------------------------------------
    if DEBUG_FLAG:
        all_rms = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
        print(f"\n[PATCH-SAMPLE] Final RMS distribution across patches: "
              f"min={all_rms.min().item():.3f} "
              f"p25={all_rms.quantile(0.25).item():.3f} "
              f"p50={all_rms.quantile(0.50).item():.3f} "
              f"p75={all_rms.quantile(0.75).item():.3f} "
              f"max={all_rms.max().item():.3f}")
        
        torch.save(
            {
                "patch_indices": [p.cpu() for p in patch_indices],
                "patch_coords": [pc.cpu() for pc in patch_coords],
            },
            f"debug_patch_coords_seed.pt",
        )
        print(f"[DEBUG] Saved patch coords to debug_patch_coords_seed.pt\n")



    # ===================================================================
    # DIAGNOSTIC B3: OVERLAP NEIGHBOR AGREEMENT (JACCARD)
    # ===================================================================
    if DEBUG_FLAG:
        print("\n" + "="*70)
        print("DIAGNOSTIC B3: OVERLAP NEIGHBOR AGREEMENT")
        print("="*70)
        
        overlap_jaccards = []
        overlap_dist_corrs = []
        
        # Sample 20 random overlapping patch pairs
        overlap_pairs = []
        for i in range(min(K, 50)):
            for j in range(i+1, min(K, 50)):
                S_i = set(patch_indices[i].tolist())
                S_j = set(patch_indices[j].tolist())
                shared = S_i & S_j
                if len(shared) >= 30:  # Need enough for meaningful kNN
                    overlap_pairs.append((i, j, list(shared)))
        
        # Analyze up to 20 pairs
        for i, j, shared_list in overlap_pairs[:20]:
            # Get coords for shared cells in both patches

            # Build position lookup dicts (O(1) lookup)
            pos_i = {int(cid): p for p, cid in enumerate(patch_indices[i].tolist())}
            pos_j = {int(cid): p for p, cid in enumerate(patch_indices[j].tolist())}

            shared_idx_i = torch.tensor([pos_i[c] for c in shared_list], dtype=torch.long)
            shared_idx_j = torch.tensor([pos_j[c] for c in shared_list], dtype=torch.long)
            
            V_i_shared = patch_coords[i][shared_idx_i]
            V_j_shared = patch_coords[j][shared_idx_j]
            
            # Compute kNN Jaccard
            knn_i = knn_sets(V_i_shared, k=min(10, len(shared_list)-1))
            knn_j = knn_sets(V_j_shared, k=min(10, len(shared_list)-1))
            jaccard = mean_jaccard(knn_i, knn_j)
            overlap_jaccards.append(jaccard)
            
            # Also compute distance correlation
            D_i = torch.cdist(V_i_shared, V_i_shared).cpu().numpy()
            D_j = torch.cdist(V_j_shared, V_j_shared).cpu().numpy()
            triu = np.triu_indices(len(shared_list), k=1)
            dist_corr = np.corrcoef(D_i[triu], D_j[triu])[0, 1]
            overlap_dist_corrs.append(dist_corr)
        
        if overlap_jaccards:
            print(f"\n[B3] Analyzed {len(overlap_jaccards)} overlapping patch pairs:")
            print(f"  kNN Jaccard: {np.mean(overlap_jaccards):.3f} ± {np.std(overlap_jaccards):.3f}")
            print(f"  Distance corr: {np.mean(overlap_dist_corrs):.3f} ± {np.std(overlap_dist_corrs):.3f}")
            
            if np.mean(overlap_jaccards) < 0.3:
                print("  ⚠️  LOW JACCARD: Patches disagree on neighborhoods → DIFFUSION PROBLEM")
            elif np.mean(overlap_jaccards) > 0.6:
                print("  ✓ GOOD JACCARD: Patches agree on neighborhoods")
            
            # Plot
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].hist(overlap_jaccards, bins=20, edgecolor='black', alpha=0.7, color='blue')
            axes[0].axvline(np.mean(overlap_jaccards), color='red', linestyle='--', 
                           label=f'Mean={np.mean(overlap_jaccards):.3f}')
            axes[0].set_xlabel('kNN Jaccard')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Overlap Neighbor Agreement (Jaccard)')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].scatter(overlap_dist_corrs, overlap_jaccards, alpha=0.6)
            axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
            axes[1].axvline(0.8, color='red', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Distance Correlation')
            axes[1].set_ylabel('kNN Jaccard')
            axes[1].set_title('Distance Corr vs Neighbor Agreement')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        print("="*70 + "\n")

    # Compute median patch RMS as target scale for global alignment
    patch_rms_list = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
    rms_target = patch_rms_list.median().item()

    if DEBUG_FLAG:
        print(f"[ALIGN] Target RMS for global space: {rms_target:.3f} (median of patches)")


    # DEBUG 3: RMS distribution across all patches
    if DEBUG_FLAG:
        all_rms = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
        print(f"\n[PATCH-SAMPLE] RMS distribution across patches: "
            f"min={all_rms.min().item():.3f} "
            f"p25={all_rms.quantile(0.25).item():.3f} "
            f"p50={all_rms.quantile(0.50).item():.3f} "
            f"p75={all_rms.quantile(0.75).item():.3f} "
            f"max={all_rms.max().item():.3f}")
        
        # Save patch coords for reproducibility testing
        torch.save(
            {
                "patch_indices": [p.cpu() for p in patch_indices],
                "patch_coords": [pc.cpu() for pc in patch_coords],
            },
            "debug_patch_coords.pt",
        )
        print("[DEBUG] Saved patch coords to debug_patch_coords.pt\n")

    # ===== DIAGNOSTIC: Patch-level geometry BEFORE any stitching =====
    if DEBUG_FLAG and K > 1:
        print("\n" + "="*60)
        print("PATCH OVERLAP DIAGNOSTIC (pre-alignment)")
        print("="*60)
        
        # Find pairs of overlapping patches
        overlap_corrs = []
        
        for k1 in range(min(10, K)):  # Check first 10 patches
            S_k1 = set(patch_indices[k1].cpu().tolist())
            V_k1 = patch_coords[k1].cpu()  # (m_k1, D)
            
            for k2 in range(k1+1, min(k1+5, K)):  # Check next 4 patches
                S_k2 = set(patch_indices[k2].cpu().tolist())
                
                # Find shared cells
                shared = S_k1 & S_k2
                if len(shared) < 20:  # Need enough overlap
                    continue
                
                shared_list = sorted(list(shared))
                
                # Get positions in each patch
                S_k1_list = patch_indices[k1].cpu().tolist()
                S_k2_list = patch_indices[k2].cpu().tolist()
                
                pos_k1 = [S_k1_list.index(s) for s in shared_list]
                pos_k2 = [S_k2_list.index(s) for s in shared_list]
                
                # Extract shared cell coords from each patch
                V_shared_k1 = V_k1[pos_k1]  # (n_shared, D)
                V_shared_k2 = patch_coords[k2].cpu()[pos_k2]
                
                # Compute pairwise distances within shared cells
                D_k1 = torch.cdist(V_shared_k1, V_shared_k1).numpy()
                D_k2 = torch.cdist(V_shared_k2, V_shared_k2).numpy()
                
                # Compare distances (upper triangle)
                tri = np.triu_indices(len(shared_list), k=1)
                if len(tri[0]) > 10:
                    from scipy.stats import pearsonr
                    corr = pearsonr(D_k1[tri], D_k2[tri])[0]
                    overlap_corrs.append(corr)
        
        if overlap_corrs:
            overlap_corrs = np.array(overlap_corrs)
            print(f"[OVERLAP] Checked {len(overlap_corrs)} patch pairs")
            print(f"[OVERLAP] Distance correlation in shared cells:")
            print(f"  min={overlap_corrs.min():.3f} "
                f"p25={np.percentile(overlap_corrs, 25):.3f} "
                f"median={np.median(overlap_corrs):.3f} "
                f"p75={np.percentile(overlap_corrs, 75):.3f} "
                f"max={overlap_corrs.max():.3f}")
            
            if np.median(overlap_corrs) < 0.7:
                print("\n⚠️  WARNING: Low overlap consistency!")
                print("    Patches disagree about shared cells' neighborhoods")
                print("    → Problem is in DIFFUSION MODEL (Stage C)")
            elif np.median(overlap_corrs) > 0.85:
                print("\n✓ Good overlap consistency")
                print("    Patches agree about shared cells")
                print("    → If final result is bad, problem is in STITCHING")
            else:
                print("\n⚠️  Moderate overlap consistency")
                print("    Some disagreement between patches")
        
        print("="*60 + "\n")

    # DEBUG: Patch-level correlation to GT (if GT is available)
    if DEBUG_FLAG and hasattr(sc_gene_expr, 'gt_coords'):
        from scipy.spatial.distance import cdist
        from scipy.stats import pearsonr
        
        gt_coords = sc_gene_expr.gt_coords  # Assume passed in somehow
        patch_local_corrs = []
        
        for k in range(K):
            S_k_np = patch_indices[k].cpu().numpy()
            V_k = patch_coords[k].cpu().numpy()  # (m_k, D_latent)
            gt_k = gt_coords[S_k_np]  # (m_k, 2)
            
            D_pred = cdist(V_k, V_k)
            D_gt = cdist(gt_k, gt_k)
            
            tri = np.triu_indices(len(S_k_np), k=1)
            if len(tri[0]) > 0:
                r = pearsonr(D_pred[tri], D_gt[tri])[0]
                patch_local_corrs.append(r)
        
        patch_local_corrs = np.array(patch_local_corrs)
        print(f"\n[PATCH-LOCAL] Pearson vs GT: "
              f"min={patch_local_corrs.min():.3f} "
              f"p25={np.percentile(patch_local_corrs, 25):.3f} "
              f"p50={np.median(patch_local_corrs):.3f} "
              f"p75={np.percentile(patch_local_corrs, 75):.3f} "
              f"max={patch_local_corrs.max():.3f}")

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

    # global recentering
    X_global = X_global - X_global.mean(dim=0, keepdim=True)
    rms_init = X_global.pow(2).mean().sqrt().item()

    # Rescale to match median patch RMS (data-driven)
    scale_factor = (rms_target / (rms_init + 1e-8))
    scale_factor = torch.clamp(torch.tensor(scale_factor), 0.25, 4.0).item()  # Safety bounds
    X_global = X_global * scale_factor

    rms_final = X_global.pow(2).mean().sqrt().item()
    if DEBUG_FLAG:
        print(f"[ALIGN] Init X_global: rms_raw={rms_init:.3f} "
            f"→ rescaled to {rms_final:.3f} (target={rms_target:.3f}, scale={scale_factor:.3f})")

    # 5.2 Alternating Procrustes alignment (SIMPLIFIED: fixed scale, single iteration)
    s_global = 1.0
    
    print(f"\n[ALIGN] Using FIXED global scale s_global={s_global} (no dynamic scaling)")
    print(f"[ALIGN] Running simplified alignment with {n_align_iters} iteration(s)...")
    
    for it in range(n_align_iters):
        if DEBUG_FLAG:
            print(f"\n[ALIGN] Iteration {it + 1}/{n_align_iters}")

        R_list: List[torch.Tensor] = []
        t_list: List[torch.Tensor] = []
        s_list: List[torch.Tensor] = []
        
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

            # Apply sqrt weights for proper weighted Procrustes
            w_sqrt = weights_k.sqrt()
            Xc_w = Xc * w_sqrt
            Vc_w = Vc * w_sqrt

            # Weighted cross-covariance
            C = Xc_w.T @ Vc_w
            
            # SVD for rotation
            U, S_vals, Vh = torch.linalg.svd(C, full_matrices=False)
            R_k = U @ Vh
            if torch.det(R_k) < 0:
                U[:, -1] *= -1
                R_k = U @ Vh

            # Compute per-patch scale (no clamp initially - let's see natural values)
            numer = S_vals.sum()
            denom = (Vc_w ** 2).sum().clamp_min(1e-8)
            s_k_raw = numer / denom

            # Gentle safety clamp (wide range to allow data-driven values)
            s_k = s_k_raw.clamp(0.3, 3.0)

            R_list.append(R_k)
            s_list.append(s_k)

        # ======================================================================
        # NO GLOBAL SCALE RECOMPUTATION - use fixed s_global = 1.0
        # ======================================================================
        if DEBUG_FLAG and it == 0:
            # print(f"[ALIGN] Using FIXED s_global={s_global} (not recomputed from patches)")
            s_tensor = torch.stack(s_list)
            print(f"[ALIGN] per-patch s_k: "
                  f"min={s_tensor.min().item():.3f} "
                  f"p25={s_tensor.quantile(0.25).item():.3f} "
                  f"p50={s_tensor.quantile(0.50).item():.3f} "
                  f"p75={s_tensor.quantile(0.75).item():.3f} "
                  f"max={s_tensor.max().item():.3f}")
            
            # Show how many are hitting clamps
            n_clamp_low = (s_tensor < 0.31).sum().item()
            n_clamp_high = (s_tensor > 2.99).sum().item()
            if n_clamp_low > 0 or n_clamp_high > 0:
                print(f"[ALIGN] WARNING: {n_clamp_low} patches hit lower clamp, "
                    f"{n_clamp_high} hit upper clamp - consider adjusting bounds")

        # ======================================================================
        # Step A (cont.): Compute translations and track alignment error
        # ======================================================================
        for k in range(K):
            S_k = patch_indices[k]
            V_k = patch_coords[k].to(X_global.device)
            X_k = X_global[S_k]
            R_k = R_list[k]
            s_k = s_list[k]

            # Recompute centrality weights (or cache from above)
            center_k = V_k.mean(dim=0, keepdim=True)
            dists = torch.norm(V_k - center_k, dim=1, keepdim=True)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)

            w_sum = weights_k.sum()
            mu_X = (weights_k * X_k).sum(dim=0, keepdim=True) / w_sum
            mu_V = (weights_k * V_k).sum(dim=0, keepdim=True) / w_sum

            # Translation using this patch's scale
            t_k = (mu_X - s_k * (mu_V @ R_k.T)).squeeze(0)
            t_list.append(t_k)

            # Track patch alignment error with current X_global
            X_hat_k = s_k * (V_k @ R_k.T) + t_k  # (m_k, D)
            sqerr = (X_hat_k - X_k).pow(2).sum(dim=1)  # (m_k,)
            patch_mse = sqerr.mean().item()
            per_patch_mse.append(patch_mse)

        if DEBUG_FLAG:
            per_patch_mse_t = torch.tensor(per_patch_mse)
            print(f"[ALIGN] per-patch mse: "
                f"p10={per_patch_mse_t.quantile(0.10).item():.4e} "
                f"p50={per_patch_mse_t.quantile(0.50).item():.4e} "
                f"p90={per_patch_mse_t.quantile(0.90).item():.4e}")

            # DEBUG 5: Transform magnitudes
            R_norms = []
            t_norms = []
            for k_t in range(K):
                R_k_t = R_list[k_t]
                t_k_t = t_list[k_t]
                # how far from identity?
                R_norms.append((R_k_t - torch.eye(D_latent, device=R_k_t.device)).pow(2).sum().sqrt().item())
                t_norms.append(t_k_t.norm().item())
            R_norms_t = torch.tensor(R_norms)
            t_norms_t = torch.tensor(t_norms)
            print(f"[ALIGN-TRANSFORMS] iter={it+1} "
                f"R_dev_from_I: p50={R_norms_t.quantile(0.5).item():.3f} "
                f"p90={R_norms_t.quantile(0.9).item():.3f} "
                f"t_norm: p50={t_norms_t.quantile(0.5).item():.3f} "
                f"p90={t_norms_t.quantile(0.9).item():.3f}")
            
        # ===================================================================
        # DIAGNOSTIC C: STITCHING QUALITY
        # ===================================================================
        if DEBUG_FLAG and it == 0:  # Only on first iteration
            print(f"\n[DIAGNOSTIC C] Stitching Analysis (Iteration {it+1}):")
            
            # C2: Per-cell multi-patch disagreement
            sum_x = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
            sum_x2 = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
            count = torch.zeros(n_sc, 1, dtype=torch.float32, device=device)
            
            for k_idx in range(K):
                S_k = patch_indices[k_idx].to(device)
                V_k = patch_coords[k_idx].to(device)
                R_k = R_list[k_idx].to(device)
                t_k = t_list[k_idx].to(device)
                s_k = s_list[k_idx].to(device)
                
                X_hat = s_k * (V_k @ R_k.T) + t_k
                
                sum_x.index_add_(0, S_k, X_hat)
                sum_x2.index_add_(0, S_k, X_hat**2)
                count.index_add_(0, S_k, torch.ones(len(S_k), 1, device=device))
            
            mean = sum_x / count.clamp_min(1)
            var = (sum_x2 / count.clamp_min(1)) - mean**2
            cell_var = var.mean(dim=1).cpu().numpy()
            
            print(f"  Per-cell disagreement:")
            print(f"    p50: {np.median(cell_var):.4f}")
            print(f"    p90: {np.percentile(cell_var, 90):.4f}")
            print(f"    p95: {np.percentile(cell_var, 95):.4f}")
            
            if np.median(cell_var) > 0.5:
                print("  ⚠️  HIGH VARIANCE: Patches strongly disagree → STITCHING PROBLEM")
            
            # Plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.hist(cell_var, bins=50, edgecolor='black', alpha=0.7, color='purple')
            plt.axvline(np.median(cell_var), color='red', linestyle='--', 
                       label=f'Median={np.median(cell_var):.3f}')
            plt.xlabel('Per-cell Variance')
            plt.ylabel('Count')
            plt.title('Multi-Patch Coordinate Disagreement')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

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
            s_k = s_list[k].to(device_X)                 # scalar

            # Centrality weights in local patch coordinates
            center_k = V_k.mean(dim=0, keepdim=True)     # (1, D)
            dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)        # (m_k, 1)

            # Transformed patch in global frame using this patch's scale
            X_hat_k = s_k * (V_k @ R_k.T) + t_k     # (m_k, D)

            # Weighted accumulation
            new_X.index_add_(0, S_k, X_hat_k * weights_k)
            W_X.index_add_(0, S_k, weights_k)

        # Finish Step B: normalize and recenter
        mask_seen2 = W_X.squeeze(-1) > 0
        new_X[mask_seen2] /= W_X[mask_seen2]
        # Cells never hit by any patch: keep previous
        new_X[~mask_seen2] = X_global[~mask_seen2]


        new_X = new_X - new_X.mean(dim=0, keepdim=True)
        
        # Enforce target scale at every iteration (data-driven)
        rms_current = new_X.pow(2).mean().sqrt()
        scale_correction = rms_target / (rms_current + 1e-8)
        new_X = new_X * scale_correction
        
        X_global = new_X

        if DEBUG_FLAG:
            rms_new = new_X.pow(2).mean().sqrt().item()
            print(f"[new ALIGN] iter={it + 1} coords_rms={rms_new:.3f} (global scale)")

        # DEBUG 4: Patch consistency after alignment iteration
        patch_fit_errs = []
        for k_check in range(K):
            S_k_check = patch_indices[k_check].to(device_X)
            V_k_check = patch_coords[k_check].to(device_X)
            X_k_check = X_global[S_k_check]

            # Compare distances within patch before/after stitching
            D_V = torch.cdist(V_k_check, V_k_check)
            D_X = torch.cdist(X_k_check, X_k_check)
            # normalize by RMS to ignore global scale
            D_V = D_V / (D_V.pow(2).mean().sqrt().clamp_min(1e-6))
            D_X = D_X / (D_X.pow(2).mean().sqrt().clamp_min(1e-6))
            err = (D_V - D_X).abs().mean().item()
            patch_fit_errs.append(err)

        patch_fit_errs_t = torch.tensor(patch_fit_errs)
        print(f"[ALIGN-CHECK] iter={it+1} patch dist mismatch: "
              f"p10={patch_fit_errs_t.quantile(0.10).item():.4e} "
              f"p50={patch_fit_errs_t.quantile(0.50).item():.4e} "
              f"p90={patch_fit_errs_t.quantile(0.90).item():.4e} "
              f"max={patch_fit_errs_t.max().item():.4e}")

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
    # DEBUG: Save final EDM
    # ------------------------------------------------------------------
    if DEBUG_FLAG:
        torch.save(
            {
                "D_edm": D_edm.cpu(),
            },
            f"debug_final_edm_seed.pt",
        )
        print(f"[DEBUG] Saved final EDM to debug_final_edm_seed.pt")

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