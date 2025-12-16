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
    # st_coords_norm, center, radius = uet.normalize_coordinates_isotropic(st_coords)
    # NEW: Coords are already normalized from run_mouse_brain_2.py
    st_coords_norm = st_coords
    print("[Stage A] Using pre-normalized coordinates")
    
    # Auto-compute sigma if not provided
    if sigma is None:
        with torch.no_grad():
            sigmas_per_slide = []
            unique_slides = torch.unique(slide_ids)
            
            for sid in unique_slides:
                mask_slide = (slide_ids == sid)
                coords_slide = st_coords_norm[mask_slide]
                n_spots = coords_slide.shape[0]
                
                # Within-slide distances only
                D_slide = torch.cdist(coords_slide, coords_slide)
                k = min(15, n_spots - 1)
                kth_dists = torch.kthvalue(D_slide, k+1, dim=1).values
                sigma_slide = torch.median(kth_dists).item()
                sigmas_per_slide.append(sigma_slide)
            
            # Aggregate across slides
            sigma = float(np.median(sigmas_per_slide)) * 0.8
            print(f"Auto-computed sigma (per-slide median) = {sigma:.4f}")
            print(f"  Per-slide sigmas: {sigmas_per_slide}")
    
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
    knn_indices: torch.Tensor  # (n, k) k-NN indices for each spot
    knn_spatial: torch.Tensor   # (n, k) k-NN indices from SPATIAL coordinates (not embeddings)
    # NEW: Persistent homology topology info (for full slide)
    topo_pairs_0: torch.Tensor  # (m0, 2) index pairs for 0D features (connected components)
    topo_dists_0: torch.Tensor  # (m0,) birth distances for 0D features
    topo_pairs_1: torch.Tensor  # (m1, 2) index pairs for 1D features (loops)
    topo_dists_1: torch.Tensor  # (m1,) birth distances for 1D features



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
        # y_hat, center, scale = uet.normalize_pose_scale(st_coords)
        # 1. Coords already normalized from run_mouse_brain_2.py
        y_hat = st_coords
        center = torch.zeros(2, device=device)
        scale = 1.0
        print(f"[Stage B] Using pre-normalized coordinates for slide {slide_id}")
        
        # 2. Pairwise distances & Gram
        D = torch.cdist(y_hat, y_hat, p=2)
        G = uet.gram_from_coords(y_hat)

        # Compute k-NN indices from EMBEDDINGS (not coordinates)
        # This makes the loss biologically meaningful (expression similarity preservation)
        print(f"[Stage B] Computing kNN from embeddings (not coordinates)")
        Z = encoder(st_gene_expr.to(device))  # (n, 128) embeddings
        D_Z = torch.cdist(Z, Z)  # Distance in embedding space

        knn_k = self.k
        n_spots = y_hat.shape[0]
        knn_indices = torch.zeros(n_spots, knn_k, dtype=torch.long, device=device)

        for i in range(n_spots):
            # Get distances from spot i to all others IN EMBEDDING SPACE
            dists_from_i = D_Z[i]  # (n,)
            # Set self-distance to infinity to exclude
            dists_from_i_copy = dists_from_i.clone()
            dists_from_i_copy[i] = float('inf')
            # Get indices of k smallest distances
            _, indices = torch.topk(dists_from_i_copy, k=min(knn_k, n_spots-1), largest=False)
            knn_indices[i, :len(indices)] = indices
            # If we have fewer than k neighbors, pad with -1
            if len(indices) < knn_k:
                knn_indices[i, len(indices):] = -1

        print(f"[Stage B] kNN computed from embeddings: {knn_indices.shape}")
        
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
            Z_indices=Z_indices,
            knn_indices=knn_indices
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
            # if use_affine_whitening:
            #     y_hat, scale = uet.affine_whitening(st_coords, eps=1e-6)
            # else:
            #     y_hat = uet.normalize_coordinates_isotropic(st_coords)
            #     scale = 1.0

            # NEW: Coords already normalized from run_mouse_brain_2.py
            y_hat = st_coords
            scale = 1.0
            print(f"[Stage B] Using pre-normalized coordinates for slide {slide_id}")
            
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

            # Compute k-NN indices for each spot
            knn_k = self.k
            n_spots = y_hat.shape[0]
            knn_indices = torch.zeros(n_spots, knn_k, dtype=torch.long, device=self.device)

            for i in range(n_spots):
                # Get distances from spot i to all others
                dists_from_i = D[i]  # (n,)
                # Set self-distance to infinity to exclude
                dists_from_i_copy = dists_from_i.clone()
                dists_from_i_copy[i] = float('inf')
                # Get indices of k smallest distances
                _, indices = torch.topk(dists_from_i_copy, k=min(knn_k, n_spots-1), largest=False)
                knn_indices[i, :len(indices)] = indices
                # If we have fewer than k neighbors, pad with -1
                if len(indices) < knn_k:
                    knn_indices[i, len(indices):] = -1

            # NEW: Compute SPATIAL kNN (from coordinates, not embeddings)
            print(f"[Stage B] Computing spatial kNN for slide {slide_id}...")
            D_spatial = torch.cdist(y_hat, y_hat)  # (n, n)
            D_spatial_noself = D_spatial + torch.eye(n_spots, device=y_hat.device) * 1e10
            _, knn_spatial = torch.topk(D_spatial_noself, k=knn_k, dim=1, largest=False)
            # knn_spatial is (n, k) indices in global slide indexing

            # NEW: Compute persistent homology for topology preservation
            print(f"[Stage B] Computing persistent homology for slide {slide_id}...")
            y_hat_np = y_hat.cpu().numpy()
            topo_info = uet.compute_persistent_pairs(y_hat_np, max_pairs_0d=10, max_pairs_1d=5)
            print(f"[Stage B]   0D pairs: {topo_info['pairs_0'].shape[0]}, 1D pairs: {topo_info['pairs_1'].shape[0]}")

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
                scale=scale,
                knn_indices=knn_indices.cpu(),
                knn_spatial=knn_spatial.cpu(),
                topo_pairs_0=torch.from_numpy(topo_info['pairs_0']).long(),
                topo_dists_0=torch.from_numpy(topo_info['dists_0']).float(),
                topo_pairs_1=torch.from_numpy(topo_info['pairs_1']).long(),
                topo_dists_1=torch.from_numpy(topo_info['dists_1']).float()
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
        device: str = 'cuda',
        landmarks_L: int=32
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
        self.landmarks_L = 0
        
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
        slide_id = np.random.choice(self.slide_ids)
        targets = self.targets_dict[slide_id]

        n = np.random.randint(self.n_min, self.n_max + 1)
        m = targets.y_hat.shape[0]
        n = min(n, m)  # don't exceed slide size

        # DEBUG: Only print for first 3 minisets per epoch
        debug_this_sample = (idx < 3)

        # ------------------------------------------------------------------
        # NEW sampling: pure local kNN patch in TRUE ST distance space
        # ------------------------------------------------------------------
        # Pick a random center
        center_idx = np.random.randint(0, m)

        # Distances from this center in true ST distance matrix
        D_row = targets.D[center_idx]  # (m,)

        # Exclude the center itself
        all_idx = torch.arange(m)
        mask_self = all_idx != center_idx
        dists = D_row[mask_self]        # (m-1,)
        idx_no_self = all_idx[mask_self]

        if idx_no_self.numel() == 0:
            # Tiny slide fallback
            indices = all_idx[:n]
        else:
            # Sort by distance (nearest first)
            sort_order = torch.argsort(dists)          # ascending
            sorted_neighbors = idx_no_self[sort_order] # (m-1,)

            # Always include the center + closest neighbors
            n_neighbors = min(n - 1, sorted_neighbors.numel())
            neighbors = sorted_neighbors[:n_neighbors]

            indices = torch.cat([
                torch.tensor([center_idx], dtype=torch.long),
                neighbors
            ])

            # If we still have fewer than n (tiny slide), pad with random others
            if indices.numel() < n:
                missing = n - indices.numel()
                mask_extra = ~torch.isin(all_idx, indices)
                extra_pool = all_idx[mask_extra]
                if extra_pool.numel() > 0:
                    perm = torch.randperm(extra_pool.numel())
                    add = extra_pool[perm[:min(missing, extra_pool.numel())]]
                    indices = torch.cat([indices, add])

        # Shuffle order to avoid any positional bias
        indices = indices[torch.randperm(indices.numel())]

        # # ------------------------------------------------------------------
        # # New sampling: center-based near / mid / far using true distances
        # # ------------------------------------------------------------------
        # # Pick a random center
        # center_idx = np.random.randint(0, m)

        # # Full distance row for this center (true ST nuclear distances)
        # D_row = targets.D[center_idx]  # (m,)

        # # Exclude the center itself
        # all_idx = torch.arange(m)
        # mask_self = all_idx != center_idx
        # dists = D_row[mask_self]           # (m-1,)
        # idx_no_self = all_idx[mask_self]   # (m-1,)

        # # Safeguard for tiny slides
        # if idx_no_self.numel() == 0:
        #     indices = all_idx[:n]
        # else:
        #     # Compute distance quantiles (on CPU tensors)
        #     # You can tune these
        #     q_near = torch.quantile(dists, 0.2)
        #     q_mid  = torch.quantile(dists, 0.6)
        #     q_far  = torch.quantile(dists, 0.9)

        #     # Define buckets
        #     near_mask = dists <= q_near
        #     mid_mask  = (dists > q_near) & (dists <= q_mid)
        #     far_mask  = dists >= q_far

        #     near_idx_all = idx_no_self[near_mask]
        #     mid_idx_all  = idx_no_self[mid_mask]
        #     far_idx_all  = idx_no_self[far_mask]

        #     # Allocate counts per tier (you can tune these ratios)
        #     n_near = int(0.4 * n)
        #     n_mid  = int(0.3 * n)
        #     n_far  = n - n_near - n_mid

        #     # Helper to sample from a bucket with fallback
        #     def sample_from_bucket(bucket, k, fallback_pool):
        #         bucket = bucket
        #         if bucket.numel() >= k:
        #             # random sample without replacement
        #             perm = torch.randperm(bucket.numel())
        #             return bucket[perm[:k]]
        #         else:
        #             # take what we have, fill remaining from fallback_pool
        #             taken = bucket
        #             remaining = k - bucket.numel()
        #             if remaining > 0 and fallback_pool.numel() > 0:
        #                 # exclude already taken
        #                 mask_fp = ~torch.isin(fallback_pool, taken)
        #                 pool2 = fallback_pool[mask_fp]
        #                 if pool2.numel() > 0:
        #                     perm = torch.randperm(pool2.numel())
        #                     add = pool2[perm[:min(remaining, pool2.numel())]]
        #                     taken = torch.cat([taken, add])
        #             return taken

        #     # Global fallback pool (all non-center indices)
        #     fallback_pool = idx_no_self

        #     # Sample from each bucket
        #     near_idx = sample_from_bucket(near_idx_all, n_near, fallback_pool)
        #     mid_idx  = sample_from_bucket(mid_idx_all,  n_mid,  fallback_pool)
        #     far_idx  = sample_from_bucket(far_idx_all,  n_far,  fallback_pool)

        #     # Combine, maybe shuffle
        #     indices = torch.cat([near_idx, mid_idx, far_idx])
        #     # If, due to fallbacks, we got more than n, trim
        #     if indices.numel() > n:
        #         perm = torch.randperm(indices.numel())
        #         indices = indices[perm[:n]]
        #     # If we got fewer, pad with random others (unlikely but safe)
        #     elif indices.numel() < n:
        #         missing = n - indices.numel()
        #         mask_extra = ~torch.isin(all_idx, indices)
        #         extra_pool = all_idx[mask_extra]
        #         if extra_pool.numel() > 0:
        #             perm = torch.randperm(extra_pool.numel())
        #             add = extra_pool[perm[:min(missing, extra_pool.numel())]]
        #             indices = torch.cat([indices, add])

        # # Shuffle to avoid any ordering bias
        # indices = indices[torch.randperm(indices.numel())]

        # Store actual base size BEFORE adding landmarks
        base_n = indices.shape[0]

        # Extract kNN indices for this miniset
        # targets.knn_indices is (m, k_full) in global indexing
        # We need to map to local indexing [0..n-1]

        # Create mapping from global to local indices
        global_to_local = torch.full((m,), -1, dtype=torch.long)
        global_to_local[indices] = torch.arange(n, dtype=torch.long)

        # Get kNN indices for selected spots (still in global indexing)
        knn_global = targets.knn_indices[indices]  # (n, k)

        # Map to local indexing
        knn_local = global_to_local[knn_global]  # (n, k)
        # Any neighbor not in this miniset will be -1 (invalid)

        # Also get SPATIAL kNN for geometry losses
        knn_spatial_global = targets.knn_spatial[indices]  # (n, k)
        knn_spatial_local = global_to_local[knn_spatial_global]

        # ------------------------------------------------------------------
        # Landmarks via FPS in embedding space (unchanged)
        # ------------------------------------------------------------------
        if self.landmarks_L > 0:
            Z_subset = self.Z_dict[slide_id][indices]
            n_landmarks = min(self.landmarks_L, base_n)
            landmark_indices = uet.farthest_point_sampling(Z_subset, n_landmarks)
            landmark_global = indices[landmark_indices]
            indices = torch.cat([indices, landmark_global])

            is_landmark = torch.zeros(indices.shape[0], dtype=torch.bool)
            is_landmark[base_n:] = True
        else:
            is_landmark = torch.zeros(base_n, dtype=torch.bool)

        overlap_info = {
            'slide_id': slide_id,
            'indices': indices,
        }

        # ------------------------------------------------------------------
        # Rest of your code: unchanged
        # ------------------------------------------------------------------
        Z_set = self.Z_dict[slide_id][indices]
        y_hat_subset = targets.y_hat[indices]

        y_hat_centered = y_hat_subset - y_hat_subset.mean(dim=0, keepdim=True)
        G_subset = y_hat_centered @ y_hat_centered.t()

        D_subset = targets.D[indices][:, indices]

        V_target = uet.factor_from_gram(G_subset, self.D_latent)

        edge_index, edge_weight = uet.build_knn_graph(y_hat_subset, k=self.knn_k)
        L_subset = uet.compute_graph_laplacian(edge_index, edge_weight, n_nodes=len(indices))

        triu_mask = torch.triu(torch.ones_like(D_subset, dtype=torch.bool), diagonal=1)
        d_95 = torch.quantile(D_subset[triu_mask], 0.95)
        bins = torch.linspace(0, float(d_95), 64)
        H_subset = uet.compute_distance_hist(D_subset, bins)

        n_nodes = len(indices)
        triplets_subset = uet.sample_ordinal_triplets(D_subset, n_triplets=min(500, n_nodes), margin_ratio=0.05)

        L_info = {
            'L': L_subset,
            't_list': targets.t_list
        }

        # ------------------------------------------------------------------
        # NEW: Extract topology pairs for this miniset
        # ------------------------------------------------------------------
        # Map global topology pairs to local indices
        # Only keep pairs where BOTH points are in the miniset
        
        def extract_topo_pairs(pairs_global, dists_global, indices, global_to_local):
            """Extract topology pairs that exist within miniset"""
            if len(pairs_global) == 0:
                return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.float32)
            
            pairs_local = []
            dists_local = []
            
            for idx_pair, (i_global, j_global) in enumerate(pairs_global):
                # Check if both points are in miniset
                i_local = global_to_local[i_global.item()]
                j_local = global_to_local[j_global.item()]
                
                if i_local != -1 and j_local != -1:
                    pairs_local.append([i_local, j_local])
                    dists_local.append(dists_global[idx_pair])
            
            if len(pairs_local) == 0:
                return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.float32)
            
            return (
                torch.tensor(pairs_local, dtype=torch.long),
                torch.tensor(dists_local, dtype=torch.float32)
            )
        
        # # global_to_local is already defined above (line 683-684)
        # topo_pairs_0_local, topo_dists_0_local = extract_topo_pairs(
        #     targets.topo_pairs_0, targets.topo_dists_0, indices, global_to_local
        # )
        # topo_pairs_1_local, topo_dists_1_local = extract_topo_pairs(
        #     targets.topo_pairs_1, targets.topo_dists_1, indices, global_to_local
        # )
        
        # topo_info = {
        #     'pairs_0': topo_pairs_0_local,
        #     'dists_0': topo_dists_0_local,
        #     'pairs_1': topo_pairs_1_local,
        #     'dists_1': topo_dists_1_local,
        # }

        # ------------------------------------------------------------------
        # NEW: Compute PH directly on this miniset (not from full slide)
        # ------------------------------------------------------------------
        # Get miniset GT coordinates
        y_hat_miniset = targets.y_hat[indices].cpu().numpy()  # (n, 2)
        
        # Compute PH pairs for THIS miniset (already in local indices)
        topo_result = uet.compute_persistent_pairs(
            y_hat_miniset, 
            max_pairs_0d=min(20, n // 4),  # Scale with miniset size
            max_pairs_1d=min(10, n // 8)
        )
        
        topo_info = {
            'pairs_0': torch.from_numpy(topo_result['pairs_0']).long(),
            'dists_0': torch.from_numpy(topo_result['dists_0']).float(),
            'pairs_1': torch.from_numpy(topo_result['pairs_1']).long(),
            'dists_1': torch.from_numpy(topo_result['dists_1']).float(),
        }

        return {
            'Z_set': Z_set,
            'is_landmark': is_landmark,
            'V_target': V_target,
            'G_target': G_subset,
            'D_target': D_subset,
            'H_target': H_subset,
            'H_bins': bins,
            'L_info': L_info,
            'triplets': triplets_subset,
            'n': len(indices),
            'overlap_info': overlap_info,
            'knn_indices': knn_local,
            'topo_info': topo_info,  # NEW
            'knn_spatial': knn_spatial_local
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
        
    L_info_batch = []
    triplets_batch = []
    H_bins_batch = []
    overlap_info_batch = []
    topo_info_batch = [] 

    #init padded tensors
    Z_batch = torch.zeros(batch_size, n_max, h_dim, device=device)
    V_batch = torch.zeros(batch_size, n_max, D_latent, device=device)
    G_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    D_batch = torch.zeros(batch_size, n_max, n_max, device=device)
    H_batch = torch.zeros(batch_size, num_bins, device=device)
    mask_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    n_batch = torch.zeros(batch_size, dtype=torch.long)
    is_landmark_batch = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)  # ADD THIS

    knn_k = batch[0]['knn_indices'].shape[1]  # Get k from first item
    knn_batch = torch.full((batch_size, n_max, knn_k), -1, dtype=torch.long, device=device)
    knn_spatial_batch = torch.full((batch_size, n_max, knn_k), -1, dtype=torch.long, device=device)


    for i, item in enumerate(batch):
        n = item['n']
        Z_batch[i, :n] = item['Z_set']
        V_batch[i, :n] = item['V_target']
        G_batch[i, :n, :n] = item['G_target']
        D_batch[i, :n, :n] = item['D_target']
        H_batch[i] = item['H_target']
        mask_batch[i, :n] = True
        n_batch[i] = n
        is_landmark_batch[i, :n] = item['is_landmark']  # ADD THIS
        L_info_batch.append(item['L_info'])
        triplets_batch.append(item['triplets'])
        H_bins_batch.append(item['H_bins'])
        overlap_info_batch.append(item['overlap_info'])
        topo_info_batch.append(item.get('topo_info', None))  # NEW!
        knn_batch[i, :n] = item['knn_indices']

        if 'knn_spatial' in item:
            knn_spatial_batch[i, :n, :] = item['knn_spatial'][:n, :]
    

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
        'is_landmark': is_landmark_batch,  # ADD THIS
        'is_sc': False,
        'knn_indices': knn_batch,
        'topo_info': topo_info_batch,
        'knn_spatial': knn_spatial_batch,
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
        landmarks_L: int = 32
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
        self.landmarks_L = 0

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

        # Compute embedding-based kNN indices (for NCA loss)
        print(f"[SC Dataset] Computing embedding-based kNN indices...")
        n_sc = self.Z_cpu.shape[0]
        knn_k = min(K_nbrs, 20)  # Use smaller k for NCA loss

        # Compute distance matrix in chunks to save memory
        self.knn_indices = torch.zeros(n_sc, knn_k, dtype=torch.long)

        chunk_size = 2048
        for i in range(0, n_sc, chunk_size):
            end_i = min(i + chunk_size, n_sc)
            D_chunk = torch.cdist(self.Z_cpu[i:end_i], self.Z_cpu)
            
            for local_i, global_i in enumerate(range(i, end_i)):
                dists_from_i = D_chunk[local_i]
                dists_from_i_copy = dists_from_i.clone()
                dists_from_i_copy[global_i] = float('inf')  # Exclude self
                _, indices = torch.topk(dists_from_i_copy, k=min(knn_k, n_sc-1), largest=False)
                self.knn_indices[global_i, :len(indices)] = indices
                if len(indices) < knn_k:
                    self.knn_indices[global_i, len(indices):] = -1

        print(f"[SC Dataset] kNN indices computed: {self.knn_indices.shape}")

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
        # non_shared_A = A[non_sharedA_pos]
        # mask = ~torch.isin(cand, non_shared_A)
        # cand = cand[mask]

        # n_new = n_set - n_overlap
        # new_B = cand[:n_new]
        # B = torch.cat([shared_global, new_B])[:n_set]
        mask = ~torch.isin(cand, A)
        cand = cand[mask]

        n_new = n_set - n_overlap
        new_B = cand[:n_new]
        B = torch.cat([shared_global, new_B])[:n_set]

        # positions of shared in A and B
        mapB = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
        mapB[B] = torch.arange(B.numel(), dtype=torch.long)
        shared_A_pos = ov_pos
        shared_B_pos = mapB[shared_global]

        # Add landmarks from union (but cap total size at n_set)
        if self.landmarks_L > 0:
            # Get union of A and B
            union = torch.unique(torch.cat([A, B]))
            Z_union = self.Z_cpu[union]

            # FPS on union to select landmarks
            landmark_local = uet.farthest_point_sampling(
                Z_union, 
                min(self.landmarks_L, len(union)),
                device='cpu'
            )

            landmark_global = union[landmark_local]
            
            # Remove any landmarks already in A or B (prevent duplicates)
            new_landmarks_A = landmark_global[~torch.isin(landmark_global, A)]
            new_landmarks_B = landmark_global[~torch.isin(landmark_global, B)]

            # Cap landmarks to ensure final size doesn't exceed n_set
            space_left_A = n_set - A.numel()
            space_left_B = n_set - B.numel()
            
            new_landmarks_A = new_landmarks_A[:space_left_A]
            new_landmarks_B = new_landmarks_B[:space_left_B]

            # Append only NEW landmarks to both sets
            A = torch.cat([A, new_landmarks_A])
            B = torch.cat([B, new_landmarks_B])

            # Update shared indices to include landmarks
            n_landmarks = len(landmark_global)
            landmark_pos_A = torch.arange(len(A) - len(new_landmarks_A), len(A))
            landmark_pos_B = torch.arange(len(B) - len(new_landmarks_B), len(B))

            shared_A_pos = torch.cat([shared_A_pos, landmark_pos_A])
            shared_B_pos = torch.cat([shared_B_pos, landmark_pos_B])

            # Create is_landmark masks
            is_landmark_A = torch.zeros(len(A), dtype=torch.bool)
            is_landmark_A[-len(new_landmarks_A):] = True
            is_landmark_B = torch.zeros(len(B), dtype=torch.bool)
            is_landmark_B[-len(new_landmarks_B):] = True
        else:
            is_landmark_A = torch.zeros(len(A), dtype=torch.bool)
            is_landmark_B = torch.zeros(len(B), dtype=torch.bool)

        # Pull CPU pinned slices (do NOT move to CUDA here)
        Z_A = self.Z_cpu[A]
        Z_B = self.Z_cpu[B]

        # Duplicate guards
        assert A.numel() == A.unique().numel(), f"Set A has duplicates: {A.numel()} vs {A.unique().numel()}"
        assert B.numel() == B.unique().numel(), f"Set B has duplicates: {B.numel()} vs {B.unique().numel()}"

        # Extract kNN indices for minisets A and B, map to local indexing
        global_to_local_A = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
        global_to_local_A[A] = torch.arange(len(A), dtype=torch.long)
        knn_global_A = self.knn_indices[A]  # (n_A, k)
        knn_local_A = global_to_local_A[knn_global_A]  # Map to local indices

        global_to_local_B = torch.full((self.Z_cpu.shape[0],), -1, dtype=torch.long)
        global_to_local_B[B] = torch.arange(len(B), dtype=torch.long)
        knn_global_B = self.knn_indices[B]  # (n_B, k)
        knn_local_B = global_to_local_B[knn_global_B]  # Map to local indices

        return {
            "Z_A": Z_A, "Z_B": Z_B,
            "n_A": int(A.numel()), "n_B": int(B.numel()),
            "shared_A": shared_A_pos,
            "shared_B": shared_B_pos,
            "is_sc": True,
            "is_landmark_A": is_landmark_A,  # ADD THIS
            "is_landmark_B": is_landmark_B,  # ADD THIS
            "global_indices_A": A,  # ADD THIS
            "global_indices_B": B,   # ADD THIS
            "knn_indices_A": knn_local_A,  # ADD THIS
            "knn_indices_B": knn_local_B 
        }

from typing import List, Dict


def collate_sc_minisets(batch: List[Dict]) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = len(batch)
    n_max_A = max(x["n_A"] for x in batch)
    n_max_B = max(x["n_B"] for x in batch)
    n_max = max(n_max_A, n_max_B)
    h_dim = batch[0]["Z_A"].shape[1]

    Z_batch = torch.zeros(B * 2, n_max, h_dim, device=device)
    mask = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)
    n_vec = torch.zeros(B * 2, dtype=torch.long, device=device)

    # keep your global indices tensor
    global_indices = torch.full((B * 2, n_max), -1, dtype=torch.long, device=device)

    # Overlap tensors (padded)
    Kmax = max((len(x["shared_A"]) for x in batch), default=0)
    shared_A_idx = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    shared_B_idx = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
    shared_len   = torch.zeros(B, dtype=torch.long, device=device)

    # --- these two ARE the pair indices the debug trainer wants ---
    pair_idxA = torch.arange(0, 2 * B, 2, device=device, dtype=torch.long)
    pair_idxB = torch.arange(1, 2 * B, 2, device=device, dtype=torch.long)

    # we'll also build the older "shared_info" list for core_models_et_p2.py
    shared_info: List[Dict] = []

    for i, item in enumerate(batch):
        nA, nB = item["n_A"], item["n_B"]

        ZA = item["Z_A"].to(device, non_blocking=True)
        ZB = item["Z_B"].to(device, non_blocking=True)

        Z_batch[2*i,   :nA] = ZA
        Z_batch[2*i+1, :nB] = ZB
        mask[2*i,   :nA] = True
        mask[2*i+1, :nB] = True
        n_vec[2*i]   = nA
        n_vec[2*i+1] = nB

        # global ids (if you produce them; fine if you don't)
        if "global_indices_A" in item:
            global_indices[2*i,   :nA] = item["global_indices_A"].to(device, non_blocking=True)
        if "global_indices_B" in item:
            global_indices[2*i+1, :nB] = item["global_indices_B"].to(device, non_blocking=True)

        # overlaps (local positions within each set)
        k = len(item["shared_A"])
        if k > 0:
            shared_A_idx[i, :k] = item["shared_A"].to(device, non_blocking=True)
            shared_B_idx[i, :k] = item["shared_B"].to(device, non_blocking=True)
            shared_len[i] = k

        # build old-style entry too (for core_models_et_p2.py)
        shared_info.append({
            "idx_A": int(2*i),
            "idx_B": int(2*i+1),
            # keep these on CPU; the trainer .to(device)s them later
            "shared_A": item["shared_A"].detach().cpu(),
            "shared_B": item["shared_B"].detach().cpu(),
        })

    # Landmark masks (optional; keep your existing logic)
    if 'is_landmark_A' in batch[0]:
        is_landmark_batch = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)
        for i, item in enumerate(batch):
            nA, nB = item["n_A"], item["n_B"]
            is_landmark_batch[2*i,   :nA] = item["is_landmark_A"].to(device, non_blocking=True)
            is_landmark_batch[2*i+1, :nB] = item["is_landmark_B"].to(device, non_blocking=True)
    else:
        is_landmark_batch = torch.zeros(B * 2, n_max, dtype=torch.bool, device=device)

    # Collate kNN indices
    knn_k = batch[0]['knn_indices_A'].shape[1]
    knn_batch = torch.full((B * 2, n_max, knn_k), -1, dtype=torch.long, device=device)

    for i, item in enumerate(batch):
        nA, nB = item["n_A"], item["n_B"]
        knn_batch[2*i, :nA] = item["knn_indices_A"].to(device, non_blocking=True)
        knn_batch[2*i+1, :nB] = item["knn_indices_B"].to(device, non_blocking=True)

    return {
        "Z_set": Z_batch,                 # (2B, n_max, h)
        "mask": mask,                     # (2B, n_max)
        "n": n_vec,                       # (2B,)
        "pair_idxA": pair_idxA,           # (P=B,)
        "pair_idxB": pair_idxB,           # (P=B,)
        "shared_A_idx": shared_A_idx,     # (B, Kmax) padded -1
        "shared_B_idx": shared_B_idx,     # (B, Kmax) padded -1
        "shared_len": shared_len,         # (B,)
        "shared_info": shared_info,
        "sc_global_indices": global_indices,
        "is_sc": True,
        'knn_indices': knn_batch,
        "is_landmark": is_landmark_batch,
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