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
    
def precompute_st_prototypes(
    targets_dict: Dict[int, 'STTargets'],
    encoder: 'SharedEncoder',
    st_gene_expr_dict: Dict[int, torch.Tensor],
    n_prototypes: int = 3000,
    n_min: int = 64,
    n_max: int = 256,
    device: str = 'cuda'
) -> Dict:
    '''
    precompute ST prototypes for Z-matched distogram loss
    '''

    print(f'processing {n_prototypes} ST protypes.....')

    encoder.eval()
    Z_dict = {}
    with torch.no_grad():
        for slide_id, st_expr in st_gene_expr_dict.items():
            Z = encoder(st_expr.to(device))
            Z_dict[slide_id] = Z

    prototypes = []
    slide_ids = list(targets_dict.keys())

    for _ in range(n_prototypes):
        slide_id = slide_ids[torch.randint(len(slide_ids), (1,)).item()]
        targets = targets_dict[slide_id]
        Z_slide = Z_dict[slide_id]

        m = targets.D.shape[0]
        n = torch.randint(n_min, min(n_max + 1, m+1), (1, )).item()
        indices = torch.randperm(m)[:n]

        Z_subset = Z_slide[indices]
        centroid_Z = Z_subset.mean(dim=0)

        D_subset = targets.D[indices][:, indices]
        d_95 = torch.quantile(D_subset[torch.triu(torch.ones_like(D_subset), diagonal=1).bool()], 0.95)
        bins = torch.linspace(0, d_95, 64, device=device)
        hist = uet.compute_distance_hist(D_subset, bins)

        prototypes.append({
            'centroid_Z': centroid_Z.cpu(),
            'hist': hist.cpu(),
            'bins': bins.cpu(),
            'D': D_subset.cpu()
        })

    #stack centroid for fast lookup
    centroids = torch.stack([p['centroid_Z'] for p in prototypes])

    print(f'precomputed {len(prototypes)} protoypes')
    return {'prototypes': prototypes, 'centroids': centroids}

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
    n_timesteps: int = 400,  # CHANGED from 600
    sigma_min: float = 0.01,
    sigma_max: float = 5.0,
    device: str = 'cuda',
    outf: str = 'output'
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
    scaler = torch.amp.GradScaler(enabled=not use_bf16)  # auto-noop for bf16
    
    context_encoder = context_encoder.to(device).train()
    score_net = score_net.to(device).train()
    
    params = list(context_encoder.parameters()) + list(score_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # VE SDE
    sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), n_timesteps, device=device))
    
    # Loss modules - OPTIMIZED: Heat loss with Hutchinson
    loss_gram = uet.FrobeniusGramLoss()
    loss_heat = uet.HeatKernelLoss(
        use_hutchinson=True,
        num_probes=8,
        chebyshev_degree=10,
        knn_k=8,
        t_list=(0.5, 1.0),
        laplacian='sym'
    )
    loss_sw = uet.SlicedWassersteinLoss1D()
    loss_triplet = uet.OrdinalTripletLoss()
    
    # DataLoaders - OPTIMIZED
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_minisets, collate_sc_minisets
    
    st_loader = DataLoader(
        st_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_minisets, 
        num_workers=4,           # CHANGED from 0
        pin_memory=True,         # NEW
        persistent_workers=True  # NEW
    )
    sc_loader = DataLoader(
        sc_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_sc_minisets, 
        num_workers=4,           # CHANGED from 0
        pin_memory=True,         # NEW
        persistent_workers=True  # NEW
    )
    
    os.makedirs(outf, exist_ok=True)
    plot_dir = os.path.join(outf, 'plots')
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
    EVERY_K_STEPS = 8   # CHANGED from 2
    MAX_OVERLAP_POINTS = 32  # CHANGED from 64
    MIN_OVERLAP_ABS = 5
    
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
            G_batch = sample_batch['G_target'].to(device)
            for i in range(min(4, G_batch.shape[0])):
                V_temp = uet.factor_from_gram(G_batch[i], score_net.D_latent)
                sample_stds.append(V_temp.std().item())
        sigma_data = np.median(sample_stds)
        print(f"Computed sigma_data = {sigma_data:.4f}")
    
    global_step = 0
    
    epoch_pbar = tqdm(range(n_epochs), desc="Training Epochs", position=0)

    for epoch in epoch_pbar:
        st_iter = iter(st_loader)
        sc_iter = iter(sc_loader)
        
        epoch_losses = {k: 0.0 for k in WEIGHTS.keys()}
        epoch_losses['total'] = 0.0
        n_batches = 0
        
        # Mixed schedule: [ST, ST, SC] repeat
        schedule = ['ST', 'ST', 'SC'] * (max(len(st_loader), len(sc_loader)) // 3 + 1)
        
        # Batch progress bar
        batch_pbar = tqdm(schedule, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False, position=1)
        
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
                
                # Generate V_0 from Gram (for ST) or random (for SC)
                # if not is_sc:
                #     G_target = batch['G_target'].to(device)
                #     V_0 = torch.stack([uet.factor_from_gram(G_target[i], D_latent) for i in range(batch_size_real)])
                # else:
                #     V_0 = torch.randn(batch_size_real, Z_set.shape[1], D_latent, device=device) * 0.1

                # Generate V_0 using generator (works for both ST and SC)
                V_0 = generator(H, mask)
                
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
                with torch.autocast(device_type='cuda', enabled=False):
                    V_hat_fp32 = V_hat.float()
                    G_target_fp32 = G_target.float()
                    
                    for i in range(batch_size_real):
                        n_valid = int(n_list[i].item())
                        mask_i = mask[i, :n_valid]
                        
                        G_p = V_hat_fp32[i, :n_valid] @ V_hat_fp32[i, :n_valid].T
                        G_t = G_target_fp32[i, :n_valid, :n_valid]
                        
                        # Compute pair mask
                        P = mask_i.unsqueeze(-1) & mask_i.unsqueeze(-2)
                        
                        # Masked Frobenius
                        diff_sq = (G_p - G_t)**2 * P.float()
                        L_gram += diff_sq.sum() / P.sum().clamp_min(1)
                    
                    L_gram = L_gram / batch_size_real
                
                # Heat kernel loss (with Hutchinson)
                L_info_batch = batch.get('L_info', [])
                if L_info_batch:
                    with torch.autocast(device_type='cuda', enabled=False):
                        for i in range(batch_size_real):
                            n_valid = int(n_list[i].item())
                            mask_i = mask[i, :n_valid]
                            V_i = V_hat[i, :n_valid].float()
                            
                            # Build predicted Laplacian from V_i
                            D_V = torch.cdist(V_i, V_i)
                            edge_index, edge_weight = uet.build_knn_graph_from_distance(
                                D_V, k=8, device=device
                            )
                            L_pred = uet.compute_graph_laplacian(edge_index, edge_weight, n_valid)
                            
                            L_tgt = L_info_batch[i]['L'].to(device).float()
                            t_list = L_info_batch[i].get('t_list', [0.5, 1.0])
                            
                            L_heat += loss_heat(L_pred, L_tgt, mask=mask_i, t_list=t_list)
                        
                        L_heat = L_heat / batch_size_real
                
                # Distance histogram SW (in fp32)
                D_target = batch['D_target'].to(device)
                H_target = batch['H_target'].to(device)
                H_bins = batch['H_bins'][0].to(device)
                
                with torch.autocast(device_type='cuda', enabled=False):
                    for i in range(batch_size_real):
                        n_valid = int(n_list[i].item())
                        mask_i = mask[i, :n_valid]
                        V_i = V_hat[i, :n_valid].float()
                        D_p = torch.cdist(V_i, V_i)
                        
                        # Extract upper triangle with mask
                        triu_i, triu_j = torch.triu_indices(n_valid, n_valid, 1, device=device)
                        P_i = mask_i.unsqueeze(-1) & mask_i.unsqueeze(-2)
                        valid_pairs = P_i[triu_i, triu_j]
                        d_vec = D_p[triu_i, triu_j][valid_pairs]
                        
                        if d_vec.numel() > 0:
                            d_95_p = torch.quantile(d_vec, 0.95)
                            bins_p = torch.linspace(0, d_95_p, len(H_bins), device=device)
                            H_p = uet.compute_distance_hist(D_p.detach(), bins_p)
                            L_sw_st += loss_sw(H_p.unsqueeze(0), H_target[i].unsqueeze(0))
                    
                    L_sw_st = L_sw_st / batch_size_real
                
            else:
                # ===== SC STEP: Score + SW_SC + Ordinal =====
                
                # Ordinal from Z (in fp32)
                with torch.autocast(device_type='cuda', enabled=False):
                    for i in range(batch_size_real):
                        n_valid = int(n_list[i].item())
                        Z_i = Z_set[i, :n_valid].float()
                        V_i = V_hat[i, :n_valid].float()
                        
                        triplets = uet.sample_ordinal_triplets_from_Z(Z_i, n_per_anchor=10, k_nn=25)
                        if len(triplets) > 0:
                            D_V = torch.cdist(V_i, V_i)
                            L_ordinal_sc += loss_triplet(D_V, triplets)
                    
                    L_ordinal_sc = L_ordinal_sc / batch_size_real
                
                # Z-matched distogram (in fp32)
                with torch.autocast(device_type='cuda', enabled=False):
                    for i in range(batch_size_real):
                        n_valid = int(n_list[i].item())
                        Z_i = Z_set[i, :n_valid].float()
                        V_i = V_hat[i, :n_valid].float()
                        
                        centroid = Z_i.mean(dim=0).cpu()
                        proto_idx = find_nearest_prototypes(centroid, prototype_bank)
                        proto = prototype_bank['prototypes'][proto_idx]
                        
                        D_p = torch.cdist(V_i, V_i)
                        d_95_p = torch.quantile(D_p[torch.triu(torch.ones_like(D_p), diagonal=1).bool()], 0.95)
                        bins_p = torch.linspace(0, d_95_p, len(proto['bins']), device=device)
                        H_p = uet.compute_distance_hist(D_p.detach(), bins_p)
                        
                        H_t = proto['hist'].to(device)
                        L_sw_sc += loss_sw(H_p.unsqueeze(0), H_t.unsqueeze(0))
                    
                    L_sw_sc = L_sw_sc / batch_size_real
                
                # Distance-only overlap (SC ONLY, every 8 steps) - OPTIMIZED
                if (global_step % EVERY_K_STEPS) == 0:
                    shared_info = batch.get('shared_info', [])
                    
                    if len(shared_info) > 0:
                        pair_losses = []
                        with torch.autocast(device_type='cuda', enabled=False):
                            for info in shared_info:
                                idx_A = info['idx_A']
                                idx_B = info['idx_B']
                                shared_A = info['shared_A']
                                shared_B = info['shared_B']
                                
                                if len(shared_A) < MIN_OVERLAP_ABS:
                                    continue
                                
                                # Subsample if needed (OPTIMIZED: 32 instead of 64)
                                if len(shared_A) > MAX_OVERLAP_POINTS:
                                    sel = torch.randperm(len(shared_A))[:MAX_OVERLAP_POINTS]
                                    shared_A = shared_A[sel]
                                    shared_B = shared_B[sel]
                                
                                n_A = int(n_list[idx_A].item())
                                n_B = int(n_list[idx_B].item())
                                
                                V_A = V_hat[idx_A, :n_A].float()
                                V_B = V_hat[idx_B, :n_B].float()
                                
                                shared_A_dev = shared_A.to(device)
                                shared_B_dev = shared_B.to(device)
                                
                                V_A_shared = V_A[shared_A_dev]
                                V_B_shared = V_B[shared_B_dev]
                                
                                D_A = torch.cdist(V_A_shared, V_A_shared)
                                D_B = torch.cdist(V_B_shared, V_B_shared)
                                
                                m = len(shared_A_dev)
                                triu_idx = torch.triu_indices(m, m, offset=1, device=device)
                                d_A = D_A[triu_idx[0], triu_idx[1]]
                                d_B = D_B[triu_idx[0], triu_idx[1]]
                                
                                pair_losses.append(((d_A - d_B) ** 2).mean())
                            
                            if len(pair_losses) > 0:
                                L_overlap = torch.stack(pair_losses).mean()
            
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
            batch_pbar.update(1)
        
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
        if (epoch + 1) % 100 == 0:
            ckpt = {
                'epoch': epoch,
                'context_encoder': context_encoder.state_dict(),
                'score_net': score_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history
            }
            torch.save(ckpt, os.path.join(outf, f'ckpt_epoch_{epoch+1}.pt'))
        epoch_pbar.update(1)
    
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

def infer_sc_coordinates_optimized(model, sc_expr, batch_size=512, device='cuda'):
    """
    OPTIMIZED inference wrapper with automatic batch size selection.
    
    Args:
        model: GEMSModel instance
        sc_expr: (n_sc, n_genes) tensor
        batch_size: cells per batch (reduce if still getting OOM)
        device: 'cuda' or 'cpu'
    """
    print("\n" + "="*70)
    print("SC COORDINATE INFERENCE (BATCHED)")
    print("="*70)
    print(f"Total cells: {sc_expr.shape[0]}")
    print(f"Batch size: {batch_size}")
    
    # Clear cache before starting
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    results = sample_sc_edm_batched(
        sc_gene_expr=sc_expr,
        encoder=model.encoder,
        context_encoder=model.context_encoder,
        score_net=model.score_net,
        n_timesteps_sample=250,
        sigma_min=0.01,
        sigma_max=50.0,
        return_coords=True,
        batch_size=batch_size,
        device=device
    )
    
    return results

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
    device: str = 'cuda'
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
                
                guidance_scale = 10.0
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
    