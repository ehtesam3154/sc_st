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

# Import from project knowledge Set-Transformer components
from modules import MAB, SAB, ISAB, PMA
import utils_et as uet


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


def train_encoder(
    model: SharedEncoder,
    st_gene_expr: torch.Tensor,
    st_coords: torch.Tensor,
    sc_gene_expr: torch.Tensor,
    slide_ids: Optional[torch.Tensor] = None,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    sigma: Optional[float] = None,
    alpha: float = 0.9,
    mmdbatch: float = 0.1,
    device: str = 'cuda',
    outf: str = 'output'
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
        n_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        sigma: RBF bandwidth (auto-computed if None)
        alpha: circular coupling weight
        mmdbatch: MMD batch fraction
        device: torch device
        outf: output directory
    """
    model = model.to(device)
    model.train()
    
    # Normalize ST coordinates (pose-invariant)
    st_coords_norm, center, radius = uet.normalize_coordinates_isotropic(st_coords)
    
    # Auto-compute sigma if not provided
    if sigma is None:
        with torch.no_grad():
            D = torch.cdist(st_coords_norm, st_coords_norm)
            k = 15
            kth_distances = torch.kthvalue(D, k+1, dim=1).values
            sigma = torch.median(kth_distances).item() * 0.8
            print(f"Auto-computed sigma = {sigma:.4f}")
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    print(f"Training encoder for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # Sample batches
        idx_st = torch.randperm(st_gene_expr.shape[0], device=device)[:batch_size]
        idx_sc = torch.randperm(sc_gene_expr.shape[0], device=device)[:batch_size]
        
        X_st_batch = st_gene_expr[idx_st]
        X_sc_batch = sc_gene_expr[idx_sc]
        
        # Forward pass
        Z_st = model(X_st_batch)
        Z_sc = model(X_sc_batch)
        
        # Loss 1: RBF adjacency prediction (with slide mask)
        Z_st_sim = F.softmax(Z_st @ Z_st.t() / 0.1, dim=1)  # Temperature = 0.1
        A_target_batch = A_target[idx_st][:, idx_st]
        slide_mask_batch = slide_mask[idx_st][:, idx_st]
        
        loss_pred = F.mse_loss(Z_st_sim * slide_mask_batch.float(), 
                               A_target_batch * slide_mask_batch.float())
        
        # Loss 2: Circular coupling
        # ST → SC (use all SC)
        Z_sc_all = model(sc_gene_expr)
        sim_st_to_sc = Z_st @ Z_sc_all.t()
        P_st_to_sc = F.softmax(sim_st_to_sc / 0.1, dim=1)
        Z_st_recon = P_st_to_sc @ Z_sc_all
        
        # SC → ST (use all ST)
        Z_st_all = model(st_gene_expr)
        sim_sc_to_st = Z_sc @ Z_st_all.t()
        P_sc_to_st = F.softmax(sim_sc_to_st / 0.1, dim=1)
        Z_sc_recon = P_sc_to_st @ Z_st_all
        
        loss_circle = F.mse_loss(Z_st, Z_st_recon) + F.mse_loss(Z_sc, Z_sc_recon)
        
        # Loss 3: MMD
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
        
        # Total loss
        ratio = min(1.0, epoch / (n_epochs * 0.2))  # Warmup for circular coupling
        loss = loss_pred + alpha * ratio * loss_circle + 0.1 * loss_mmd
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs} | Loss: {loss.item():.4f} | "
                  f"Pred: {loss_pred.item():.4f} | Circle: {loss_circle.item():.4f} | "
                  f"MMD: {loss_mmd.item():.4f}")
    
    # Save checkpoint
    os.makedirs(outf, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outf, 'encoder_final.pt'))
    print("Encoder training complete!")
    
    return model


# ==============================================================================
# STAGE B: GEOMETRIC TARGET PRECOMPUTATION
# ==============================================================================

@dataclass
class STTargets:
    """Per-slide geometric targets (pose-free)."""
    slide_id: int
    center: torch.Tensor  # (2,)
    scale: float
    y_hat: torch.Tensor  # (m, 2) pose-normalized coords
    G: torch.Tensor  # (m, m) Gram matrix
    D: torch.Tensor  # (m, m) distance matrix
    H: torch.Tensor  # (num_bins,) distance histogram
    t_list: List[float]  # Heat kernel times
    k: int  # kNN parameter
    sigma_policy: str  # "median_knn"
    triplets: torch.Tensor  # (T, 3) ordinal triplets
    Z_indices: torch.Tensor  # (m,) indices into encoder embeddings


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
        y_hat, center, scale = uet.normalize_pose_scale(st_coords)
        
        # 2. Pairwise distances & Gram
        D = torch.cdist(y_hat, y_hat, p=2)
        G = uet.gram_from_coords(y_hat)
        
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
            Z_indices=Z_indices
        )
        
        return targets
    
    def precompute(
        self,
        slides: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        encoder: SharedEncoder,
        outdir: str = 'stage_b_cache'
    ) -> Dict[int, STTargets]:
        """
        Precompute targets for multiple slides and save to disk.
        
        Args:
            slides: dict of {slide_id: (st_coords, st_gene_expr, ...)}
            encoder: frozen encoder
            outdir: output directory for caches
            
        Returns:
            dict of {slide_id: STTargets}
        """
        os.makedirs(outdir, exist_ok=True)
        
        all_targets = {}
        for slide_id, (st_coords, st_gene_expr) in slides.items():
            print(f"Precomputing targets for slide {slide_id}...")
            
            targets = self.precompute_slide(slide_id, st_coords, encoder, st_gene_expr)
            all_targets[slide_id] = targets
            
            # Save to disk
            save_path = os.path.join(outdir, f'slide_{slide_id}_targets.pt')
            torch.save({
                'slide_id': targets.slide_id,
                'center': targets.center.cpu(),
                'scale': targets.scale,
                'y_hat': targets.y_hat.cpu(),
                'G': targets.G.cpu(),
                'D': targets.D.cpu(),
                'H': targets.H.cpu(),
                't_list': targets.t_list,
                'k': targets.k,
                'sigma_policy': targets.sigma_policy,
                'triplets': targets.triplets.cpu(),
                'Z_indices': targets.Z_indices.cpu()
            }, save_path)
            
            print(f"  Saved to {save_path}")
        
        print("Stage B precomputation complete!")
        return all_targets
    

# ==============================================================================
# STAGE B: MINI-SET DATASET
# ==============================================================================

class STSetDataset(Dataset):
    """
    Dataset yielding mini-sets from precomputed ST targets.
    
    Each item is a mini-set of size n ∈ [n_min, n_max]:
    - Z_set: (n, h) encoder embeddings
    - V_target: (n, D) target factor
    - G_target: (n, n) target Gram
    - D_target: (n, n) target distances
    - H_target: (num_bins,) distance histogram
    - L_info: dict with Laplacian metadata
    - triplets: (T, 3) ordinal triplets (subset)
    - mask: (n,) boolean mask
    """
    
    def __init__(
        self,
        targets_dict: Dict[int, STTargets],
        encoder: SharedEncoder,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_min: int = 256,
        n_max: int = 1024,
        D_latent: int = 16,
        num_samples: int = 10000,
        device: str = 'cuda'
    ):
        """
        Args:
            targets_dict: precomputed ST targets per slide
            encoder: frozen encoder
            st_gene_expr_dict: {slide_id: st_gene_expr}
            n_min: minimum mini-set size
            n_max: maximum mini-set size
            D_latent: latent dimension for V
            num_samples: number of mini-sets to generate
            device: torch device
        """
        self.targets_dict = targets_dict
        self.encoder = encoder.eval()
        self.st_gene_expr_dict = st_gene_expr_dict
        self.n_min = n_min
        self.n_max = n_max
        self.D_latent = D_latent
        self.num_samples = num_samples
        self.device = device
        
        # Precompute encoder embeddings for all slides
        self.Z_dict = {}
        with torch.no_grad():
            for slide_id, st_expr in st_gene_expr_dict.items():
                Z = self.encoder(st_expr.to(device))
                self.Z_dict[slide_id] = Z
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Sample slide and subset size
        slide_id = np.random.choice(list(self.targets_dict.keys()))
        targets = self.targets_dict[slide_id]
        
        n = np.random.randint(self.n_min, self.n_max + 1)
        m = targets.y_hat.shape[0]
        n = min(n, m)  # Don't exceed slide size
        
        # Sample random subset of spots
        indices = torch.randperm(m, device=self.device)[:n]
        
        # Extract subset data
        Z_set = self.Z_dict[slide_id][indices]  # (n, h)
        y_hat_subset = targets.y_hat[indices].to(self.device)  # (n, 2)
        G_subset = targets.G[indices][:, indices].to(self.device)  # (n, n)
        D_subset = targets.D[indices][:, indices].to(self.device)  # (n, n)
        
        # Factor Gram to get V_target
        V_target = uet.factor_from_gram(G_subset, self.D_latent)  # (n, D)
        
        # Recompute Laplacian on subset (kNN changes with subset)
        edge_index, edge_weight = uet.build_knn_graph(y_hat_subset, k=targets.k, device=self.device)
        L_subset = uet.compute_graph_laplacian(edge_index, edge_weight, n)
        
        # Recompute distance histogram for subset
        d_95 = torch.quantile(D_subset[torch.triu(torch.ones_like(D_subset), diagonal=1).bool()], 0.95)
        bins = torch.linspace(0, d_95, 64, device=self.device)
        H_subset = uet.compute_distance_hist(D_subset, bins)

        
        # Resample ordinal triplets within subset
        triplets_subset = uet.sample_ordinal_triplets(D_subset, n_triplets=min(500, n), margin_ratio=0.05)
        
        # L_info for computing heat kernel traces
        L_info = {
            'L': L_subset,
            't_list': targets.t_list
        }
        
        return {
            'Z_set': Z_set,
            'V_target': V_target,
            'G_target': G_subset,
            'D_target': D_subset,
            'H_target': H_subset,
            'H_bins': bins,
            'L_info': L_info,
            'triplets': triplets_subset,
            'n': n
        }  
    
def collate_minisets(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate mini-sets with padding.
    
    Args:
        batch: list of mini-set dicts
        
    Returns:
        padded batch dict with masks
    """
    device = batch[0]['Z_set'].device
    n_max = max(item['n'] for item in batch)
    batch_size = len(batch)
    h_dim = batch[0]['Z_set'].shape[1]
    D_latent = batch[0]['V_target'].shape[1]
    num_bins = batch[0]['H_target'].shape[0]
    
    # Initialize padded tensors
    Z_batch = torch.zeros(batch_size, n_max, h_dim, device=device)
    V_batch = torch.zeros(batch_size, n_max, D_latent, device=device)
    G_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    D_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    H_batch = torch.zeros(batch_size, num_bins, device=device)
    mask_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    
    L_info_batch = []
    triplets_batch = []
    H_bins_batch = []
    
    for i, item in enumerate(batch):
        n = item['n']
        Z_batch[i, :n] = item['Z_set']
        V_batch[i, :n] = item['V_target']
        G_batch[i, :n, :n] = item['G_target']
        D_batch[i, :n, :n] = item['D_target']
        H_batch[i] = item['H_target']
        mask_batch[i, :n] = True
        L_info_batch.append(item['L_info'])
        triplets_batch.append(item['triplets'])
        H_bins_batch.append(item['H_bins'])
    
    return {
        'Z_set': Z_batch,
        'V_target': V_batch,
        'G_target': G_batch,
        'D_target': D_batch,
        'H_target': H_batch,
        'L_info': L_info_batch,
        'triplets': triplets_batch,
        'H_bins': H_bins_batch,
        'mask': mask_batch
    }
