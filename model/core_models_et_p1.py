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


def _cpu(x):
    import torch
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
    ratio_start: float = 0.0,
    ratio_end: float = 1.0,
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
    
    st_gene_expr = st_gene_expr.to(device)
    st_coords = st_coords.to(device)
    sc_gene_expr = sc_gene_expr.to(device)
    if slide_ids is not None:
        slide_ids = slide_ids.to(device)

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

    nettrue = A_target
    
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
               
        #loss 1: rbf adjacency prediction (with slide mask)
        netpred = Z_st @ Z_st.t()
        A_target_batch = A_target[idx_st][:, idx_st]
        slide_mask_batch = slide_mask[idx_st][:, idx_st]

        #apply mask to both prediction and target
        netpred_masked = netpred * slide_mask_batch.float()
        A_target_masked = A_target_batch * slide_mask_batch.float()

        #normalize (row-wise) where mask is activae
        row_sums = A_target_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_target_masked = A_target_masked / row_sums

        # loss_pred = F.cross_entropy(netpred_masked, A_target_masked, reduction='mean')
        logits = netpred_masked / 0.1
        log_probs = F.log_softmax(logits, dim=1)
        loss_pred = F.kl_div(log_probs, A_target_masked, reduction='batchmean')
        
        #loss 2: circular coupling
        # Get all embeddings
        Z_st_all = model(st_gene_expr)
        Z_sc_all = model(sc_gene_expr)

        # ST batch → SC → ST batch mapping
        st2sc = F.softmax(Z_st @ Z_sc_all.t(), dim=1)
        sc2st = F.softmax(Z_sc_all @ Z_st.t(), dim=1)
        st2st_unnorm = st2sc @ sc2st
        st2st = torch.log(st2st_unnorm + 1e-7)

        # Get target and RE-NORMALIZE for batch
        nettrue_batch = nettrue[idx_st][:, idx_st]
        nettrue_batch = nettrue_batch / (nettrue_batch.sum(dim=1, keepdim=True) + 1e-8)  # ✅ ADD THIS LINE

        loss_circle = F.kl_div(st2st, nettrue_batch, reduction='none').sum(1).mean()
        
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
        # ratio = min(1.0, epoch / (n_epochs * 0.2))  # Warmup for circular coupling
        # ratio = min(1.0, epoch / (n_epochs * 0.8))  # Warmup for circular coupling
        ratio = ratio_start + (ratio_end - ratio_start) * min(epoch / (n_epochs * 0.8), 1.0)
        # loss = loss_pred + alpha * ratio * loss_circle + 0.1 * loss_mmd
        loss = loss_pred + alpha * loss_mmd + ratio * loss_circle
        
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
            if use_affine_whitening:
                y_hat, scale = uet.affine_whitening(st_coords, eps=1e-6)
            else:
                y_hat = uet.normalize_coordinates_isotropic(st_coords)
                scale = 1.0
            
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
                scale=scale
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
        knn_k: int = 12,  # NEW parameter
        device: str = 'cuda'
    ):
        self.targets_dict = targets_dict
        self.encoder = encoder
        self.st_gene_expr_dict = st_gene_expr_dict
        self.n_min = n_min
        self.n_max = n_max
        self.D_latent = D_latent
        self.num_samples = num_samples
        self.knn_k = knn_k  # Store knn_k
        self.device = device
        self.slide_ids = list(targets_dict.keys())
        
        # Precompute encoder embeddings for all slides
        self.Z_dict = {}
        with torch.no_grad():
            for slide_id, st_expr in st_gene_expr_dict.items():
                Z = self.encoder(st_expr.to(device))
                self.Z_dict[slide_id] = Z.cpu() #changed to store on cpu
    
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Pick a slide and subset size
        slide_id = np.random.choice(list(self.targets_dict.keys()))
        targets = self.targets_dict[slide_id]

        n = np.random.randint(self.n_min, self.n_max + 1)
        m = targets.y_hat.shape[0]
        n = min(n, m)  # don't exceed slide size

        # --- Sample on CPU ---
        indices = torch.randperm(m)[:n]

        overlap_info = {
            'slide_id': slide_id,
            'indices': indices,  # keep CPU bookkeeping
        }

        # All indexing on CPU (Z_dict is now CPU, targets are CPU)
        Z_set = self.Z_dict[slide_id][indices]
        y_hat_subset = targets.y_hat[indices]
        G_subset = targets.G[indices][:, indices]
        D_subset = targets.D[indices][:, indices]

        # Factor / graphs / hist / triplets all on CPU
        V_target = uet.factor_from_gram(G_subset, self.D_latent)
        edge_index, edge_weight = uet.build_knn_graph(y_hat_subset, k=self.knn_k)
        L_subset = uet.compute_graph_laplacian(edge_index, edge_weight, n)

        triu_mask = torch.triu(torch.ones_like(D_subset, dtype=torch.bool), diagonal=1)
        d_95 = torch.quantile(D_subset[triu_mask], 0.95)
        bins = torch.linspace(0, float(d_95), 64)
        H_subset = uet.compute_distance_hist(D_subset, bins)

        triplets_subset = uet.sample_ordinal_triplets(D_subset, n_triplets=min(500, n), margin_ratio=0.05)

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
            'n': n,
            'overlap_info': overlap_info
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
    
    # Initialize padded tensors
    Z_batch = torch.zeros(batch_size, n_max, h_dim, device=device)
    V_batch = torch.zeros(batch_size, n_max, D_latent, device=device)
    G_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    D_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    H_batch = torch.zeros(batch_size, num_bins, device=device)
    mask_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    n_batch = torch.zeros(batch_size, dtype=torch.long)
    
    L_info_batch = []
    triplets_batch = []
    H_bins_batch = []
    overlap_info_batch = []
    
    for i, item in enumerate(batch):
        n = item['n']
        Z_batch[i, :n] = item['Z_set']
        V_batch[i, :n] = item['V_target']
        G_batch[i, :n, :n] = item['G_target']
        D_batch[i, :n, :n] = item['D_target']
        H_batch[i] = item['H_target']
        mask_batch[i, :n] = True
        n_batch[i] = n
        L_info_batch.append(item['L_info'])
        triplets_batch.append(item['triplets'])
        H_bins_batch.append(item['H_bins'])
        overlap_info_batch.append(item['overlap_info'])
    
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
        'is_sc': False 
    }


class SCSetDataset(Dataset):
    """
    SC mini-sets with intentional overlap, using a prebuilt K-NN index.
    Returns CPU-pinned tensors; collate moves to CUDA in batch.
    """
    def __init__(
        self,
        sc_gene_expr: torch.Tensor,
        encoder: "SharedEncoder",
        n_min: int = 64,
        n_max: int = 256,
        n_large_min: int = 384,
        n_large_max: int = 512,
        large_fraction: float = 0.15,
        overlap_min: int = 20,
        overlap_max: int = 512,
        num_samples: int = 5000,
        K_nbrs: int = 2048,
        device: str = "cuda",
    ):
        self.n_min = n_min
        self.n_max = n_max
        self.n_large_min = n_large_min
        self.n_large_max = n_large_max
        self.large_fraction = large_fraction
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        self.num_samples = num_samples
        self.device = device

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

        # Build neighbor index once (CPU pinned)
        self.nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs, block=min(4096, Z_all.shape[0]))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        N = self.Z_cpu.shape[0]
        # Decide set size
        if torch.rand(1).item() < self.large_fraction:
            n_set = int(torch.randint(self.n_large_min, self.n_large_max + 1, (1,)).item())
        else:
            n_set = int(torch.randint(self.n_min, self.n_max + 1, (1,)).item())
        n_set = min(n_set, N)

        # Overlap size
        n_ov_max = min(self.overlap_max, n_set // 2 if n_set >= 2 else 0)
        n_overlap = int(torch.randint(self.overlap_min, max(self.overlap_min + 1, n_ov_max + 1), (1,)).item()) if n_ov_max >= self.overlap_min else 0

        # Seed & neighbor-based sets (CPU index)
        i = int(torch.randint(N, (1,)).item())
        A = self.nbr_idx[i, :n_set]        # (n_set,) cpu long

        if n_overlap > 0:
            ov_pos = torch.randperm(n_set)[:n_overlap]         # positions in A
            shared_global = A[ov_pos]
        else:
            ov_pos = torch.tensor([], dtype=torch.long)
            shared_global = ov_pos

        # pick nearby seed j
        k_nn = min(50, self.nbr_idx.shape[1] - 1) if self.nbr_idx.shape[1] > 1 else 0
        if k_nn > 0:
            near = self.nbr_idx[i, 1:k_nn+1]
            j = int(near[torch.randint(len(near), (1,))].item())
        else:
            j = (i + 1) % N

        cand = self.nbr_idx[j]             # (K,)
        # remove A \ overlap
        non_sharedA_pos = torch.arange(n_set)
        if n_overlap > 0:
            non_sharedA_pos = non_sharedA_pos[~torch.isin(non_sharedA_pos, ov_pos)]
        non_shared_A = A[non_sharedA_pos]
        mask = ~torch.isin(cand, non_shared_A)
        cand = cand[mask]

        n_new = n_set - n_overlap
        new_B = cand[:n_new]
        B = torch.cat([shared_global, new_B])[:n_set]

        # positions of shared in A and B
        mapB = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
        mapB[B] = torch.arange(B.numel(), dtype=torch.long)
        shared_A_pos = ov_pos
        shared_B_pos = mapB[shared_global]

        # Pull CPU pinned slices (do NOT move to CUDA here)
        Z_A = self.Z_cpu[A]
        Z_B = self.Z_cpu[B]

        return {
            "Z_A": Z_A, "Z_B": Z_B,
            "n_A": int(A.numel()), "n_B": int(B.numel()),
            "shared_A": shared_A_pos,
            "shared_B": shared_B_pos,
            "is_sc": True,
            "global_indices_A": A,  # ADD THIS
            "global_indices_B": B   # ADD THIS
        }

from typing import List, Dict

def collate_sc_minisets(batch: List[Dict]) -> Dict:
    """
    Collate SC pairs: moves Z_* from CPU pinned -> CUDA (non_blocking),
    builds a (2B, n_max, h) tensor, and emits padded overlap tensors for batched loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = len(batch)
    n_max_A = max(x["n_A"] for x in batch)
    n_max_B = max(x["n_B"] for x in batch)
    n_max = max(n_max_A, n_max_B)
    h_dim = batch[0]["Z_A"].shape[1]

    Z_batch = torch.zeros(B * 2, n_max, h_dim, device=device)
    mask = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)
    n_vec = torch.zeros(B * 2, dtype=torch.long, device=device)

    # ADD THIS: Global indices tensor (padded with -1)
    global_indices = torch.full((B * 2, n_max), -1, dtype=torch.long, device=device)
    

    # Overlap tensors (padded)
    Kmax = max((len(x["shared_A"]) for x in batch), default=0)
    shared_A_idx = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    shared_B_idx = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    shared_len   = torch.zeros(B, dtype=torch.long, device=device)
    idx_A_pairs  = torch.arange(0, 2 * B, 2, device=device, dtype=torch.long)
    idx_B_pairs  = torch.arange(1, 2 * B, 2, device=device, dtype=torch.long)

    for i, item in enumerate(batch):
        nA, nB = item["n_A"], item["n_B"]

        # move CPU pinned -> CUDA
        ZA = item["Z_A"].to(device, non_blocking=True)
        ZB = item["Z_B"].to(device, non_blocking=True)

        # place & pad
        Z_batch[2*i, :nA] = ZA
        Z_batch[2*i+1, :nB] = ZB
        mask[2*i, :nA] = True
        mask[2*i+1, :nB] = True
        n_vec[2*i] = nA
        n_vec[2*i+1] = nB

        # ADD THIS: Copy global indices
        global_indices[2*i, :nA] = item["global_indices_A"].to(device, non_blocking=True)
        global_indices[2*i+1, :nB] = item["global_indices_B"].to(device, non_blocking=True)
        

        # overlaps
        k = len(item["shared_A"])
        if k > 0:
            shared_A_idx[i, :k] = item["shared_A"].to(device, non_blocking=True)
            shared_B_idx[i, :k] = item["shared_B"].to(device, non_blocking=True)
            shared_len[i] = k

    return {
        "Z_set": Z_batch,          # (2B, n_max, h)
        "mask": mask,              # (2B, n_max)
        "n": n_vec,                # (2B,)
        "idx_A_pairs": idx_A_pairs,         # (B,)
        "idx_B_pairs": idx_B_pairs,         # (B,)
        "shared_A_idx": shared_A_idx,       # (B, Kmax) padded -1
        "shared_B_idx": shared_B_idx,       # (B, Kmax) padded -1
        "shared_len": shared_len,           # (B,)
        "sc_global_indices": global_indices,  # ADD THIS LINE
        "is_sc": True
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