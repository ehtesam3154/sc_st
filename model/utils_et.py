"""
GEMS (Generative Euclidean Metric Synthesis) - Utility Functions
Coordinate-free supervision for spatial transcriptomics reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import warnings

# ==============================================================================
# PART 1: POSE NORMALIZATION (reuse existing for continuity)
# ==============================================================================

def normalize_coordinates_isotropic(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Isotropic normalization (center + scale).
    REUSE from existing utils.py to maintain continuity.
    
    Args:
        coords: (n, 2) raw spatial coordinates
        
    Returns:
        coords_norm: (n, 2) normalized coordinates
        center: (2,) centroid
        radius: float, scaling factor (Frobenius-based)
    """
    center = coords.mean(dim=0)
    coords_centered = coords - center
    
    # Frobenius norm for isotropic scaling
    radius = torch.sqrt((coords_centered ** 2).sum() / coords.shape[0])
    if radius < 1e-8:
        radius = torch.tensor(1.0, device=coords.device)
    
    coords_norm = coords_centered / radius
    return coords_norm, center, radius


def normalize_pose_scale(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Thin wrapper around normalize_coordinates_isotropic for Stage B.
    
    Args:
        y: (m, 2) coordinates
        
    Returns:
        y_hat: (m, 2) pose-normalized coordinates
        center: (2,) translation
        scale: float, scaling factor
    """
    return normalize_coordinates_isotropic(y)


def denormalize_coordinates(coords_norm: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """Inverse of normalize_coordinates_isotropic."""
    return coords_norm * radius + center


# ==============================================================================
# PART 2: GRAPH & ADJACENCY CONSTRUCTION
# ==============================================================================

def rbf_affinity_from_dists(d2: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute RBF affinity from squared distances.
    
    Args:
        d2: (n, n) squared distances
        sigma: RBF bandwidth
        
    Returns:
        A: (n, n) RBF affinity matrix
    """
    return torch.exp(-d2 / (2 * sigma**2))


def build_knn_graph(
    coords_or_features: torch.Tensor,
    k: int = 20,
    metric: str = 'euclidean',
    return_weights: bool = True,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build kNN graph and return edge_index + weights.
    Vectorized, no Python loops.
    
    Args:
        coords_or_features: (n, d) coordinates or features
        k: number of neighbors
        metric: distance metric
        return_weights: whether to compute RBF weights
        device: torch device
        
    Returns:
        edge_index: (2, E) edge indices
        edge_weight: (E,) RBF weights (if return_weights=True)
    """
    n = coords_or_features.shape[0]
    
    # Compute pairwise distances
    D = torch.cdist(coords_or_features, coords_or_features, p=2)  # (n, n)
    
    # Find k nearest neighbors (excluding self)
    D_noself = D + torch.eye(n, device=D.device) * 1e10  # Mask diagonal
    knn_dists, knn_indices = torch.topk(D_noself, k, dim=1, largest=False)  # (n, k)
    
    # Build edge_index
    src = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten()  # (n*k,)
    dst = knn_indices.flatten()  # (n*k,)
    edge_index = torch.stack([src, dst], dim=0)  # (2, n*k)
    
    if return_weights:
        # Compute RBF weights with adaptive sigma (median kNN distance)
        knn_dists_sq = knn_dists ** 2
        sigma = torch.median(knn_dists).item()
        if sigma < 1e-8:
            sigma = 1.0
        
        edge_weight = torch.exp(-knn_dists_sq.flatten() / (2 * sigma**2))
        return edge_index.long(), edge_weight.float()
    else:
        return edge_index.long(), None


def compute_graph_laplacian(edge_index: torch.Tensor, edge_weight: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Compute graph Laplacian L = D - W from edge list.
    
    Args:
        edge_index: (2, E)
        edge_weight: (E,)
        n_nodes: number of nodes
        
    Returns:
        L: (n, n) Laplacian matrix (dense)
    """
    device = edge_index.device
    
    # Build adjacency matrix W
    W = torch.zeros(n_nodes, n_nodes, device=device)
    src, dst = edge_index[0], edge_index[1]
    W[src, dst] = edge_weight
    
    # Symmetrize (kNN is directed, make undirected)
    W = (W + W.t()) / 2
    
    # Degree matrix
    deg = W.sum(dim=1)
    D = torch.diag(deg)
    
    # Laplacian
    L = D - W
    return L


# ==============================================================================
# PART 3: SPECTRAL & HEAT KERNEL COMPUTATIONS
# ==============================================================================

def compute_spectral_targets(
    L: torch.Tensor,
    t_list: List[float] = [0.25, 1.0, 4.0],
    subsample: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute spectral targets for heat kernel matching.
    
    Args:
        L: (n, n) graph Laplacian
        t_list: diffusion times
        subsample: if provided, subsample to this many nodes for Frobenius
        
    Returns:
        dict with 'heat_traces', 'eigenvalues', 'eigenvectors' (optional)
    """
    device = L.device
    n = L.shape[0]
    
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L)  # Ascending order
    eigvals = eigvals.clamp(min=0)  # Numerical safety
    
    # Compute heat kernel traces for each t
    heat_traces = []
    for t in t_list:
        trace_t = torch.sum(torch.exp(-t * eigvals))
        heat_traces.append(trace_t)
    
    result = {
        'heat_traces': torch.tensor(heat_traces, device=device),
        'eigenvalues': eigvals,
        't_list': t_list
    }
    
    # Optional: store subsampled Frobenius info
    if subsample is not None and subsample < n:
        indices = torch.randperm(n, device=device)[:subsample]
        result['subsample_indices'] = indices
    
    return result


# ==============================================================================
# PART 4: DISTANCE & GEOMETRY FUNCTIONS
# ==============================================================================

def compute_distance_hist(D: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Compute histogram of pairwise distances (upper triangle).
    
    Args:
        D: (n, n) distance matrix
        bins: (num_bins+1,) bin edges
        
    Returns:
        hist: (num_bins,) histogram counts (normalized)
    """
    # Extract upper triangle
    triu_indices = torch.triu_indices(D.shape[0], D.shape[0], offset=1, device=D.device)
    dists = D[triu_indices[0], triu_indices[1]]
    
    # Compute histogram
    hist = torch.histc(dists, bins=len(bins)-1, min=bins[0].item(), max=bins[-1].item())
    hist = hist / hist.sum()  # Normalize
    return hist


def sample_ordinal_triplets(
    D: torch.Tensor,
    n_triplets: int = 1000,
    margin_ratio: float = 0.05
) -> torch.Tensor:
    """
    Sample ordinal triplets (i, j, k) such that D[i,j] + δ < D[i,k].
    Vectorized sampling.
    
    Args:
        D: (n, n) distance matrix
        n_triplets: number of triplets to sample
        margin_ratio: margin as fraction of median distance
        
    Returns:
        triplets: (n_triplets, 3) indices [i, j, k]
    """
    n = D.shape[0]
    device = D.device
    
    # Compute margin
    triu_indices = torch.triu_indices(n, n, offset=1, device=device)
    median_dist = torch.median(D[triu_indices[0], triu_indices[1]])
    margin = margin_ratio * median_dist
    
    # Sample random triplets
    triplets = []
    attempts = 0
    max_attempts = n_triplets * 10
    
    while len(triplets) < n_triplets and attempts < max_attempts:
        # Sample batch of random triplets
        batch_size = min(1000, n_triplets - len(triplets))
        i = torch.randint(0, n, (batch_size,), device=device)
        j = torch.randint(0, n, (batch_size,), device=device)
        k = torch.randint(0, n, (batch_size,), device=device)
        
        # Check validity: i!=j!=k and D[i,j] + margin < D[i,k]
        valid = (i != j) & (j != k) & (i != k) & (D[i, j] + margin < D[i, k])
        valid_triplets = torch.stack([i[valid], j[valid], k[valid]], dim=1)
        
        if len(valid_triplets) > 0:
            triplets.append(valid_triplets)
        
        attempts += batch_size
    
    if len(triplets) == 0:
        # Fallback: return random triplets
        warnings.warn(f"Could not find valid ordinal triplets with margin {margin:.4f}")
        i = torch.randint(0, n, (n_triplets,), device=device)
        j = torch.randint(0, n, (n_triplets,), device=device)
        k = torch.randint(0, n, (n_triplets,), device=device)
        return torch.stack([i, j, k], dim=1)
    
    triplets = torch.cat(triplets, dim=0)[:n_triplets]
    return triplets


def gram_from_coords(y_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix from pose-normalized coordinates.
    
    Args:
        y_hat: (n, 2) pose-normalized coordinates
        
    Returns:
        G: (n, n) Gram matrix = y_hat @ y_hat.T
    """
    return y_hat @ y_hat.t()


def factor_from_gram(G: torch.Tensor, D_latent: int) -> torch.Tensor:
    """
    Factor Gram matrix to get V such that G ≈ V V^T.
    
    Args:
        G: (n, n) Gram matrix (PSD)
        D_latent: target latent dimension
        
    Returns:
        V: (n, D_latent) factor matrix
    """
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(G)  # Ascending order
    eigvals = eigvals.flip(0).clamp(min=0)  # Descending, non-negative
    eigvecs = eigvecs.flip(1)
    
    # Take top D_latent eigenpairs
    D = min(D_latent, eigvals.shape[0])
    eigvals_top = eigvals[:D]
    eigvecs_top = eigvecs[:, :D]
    
    # V = U @ sqrt(Λ)
    V = eigvecs_top @ torch.diag(torch.sqrt(eigvals_top))
    
    # Pad with zeros if rank < D_latent
    if D < D_latent:
        padding = torch.zeros(V.shape[0], D_latent - D, device=V.device)
        V = torch.cat([V, padding], dim=1)
    
    # Center rows (translation neutrality)
    V = V - V.mean(dim=0, keepdim=True)
    
    return V


def edm_project(D: torch.Tensor) -> torch.Tensor:
    """
    Project distance matrix to Euclidean Distance Matrix (EDM) cone.
    Uses double-centering and eigenvalue thresholding.
    
    Args:
        D: (n, n) distance matrix
        
    Returns:
        D_proj: (n, n) projected EDM
    """
    n = D.shape[0]
    device = D.device
    
    # Double-centering: B = -0.5 * J * (D ∘ D) * J
    J = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(B)
    
    # Threshold negative eigenvalues
    eigvals_pos = eigvals.clamp(min=0)
    
    # Reconstruct B_+
    B_plus = eigvecs @ torch.diag(eigvals_pos) @ eigvecs.t()
    
    # Reconstruct distances from B_+
    diag = torch.diag(B_plus).unsqueeze(1)
    D_proj = torch.sqrt(torch.clamp(diag + diag.t() - 2 * B_plus, min=0))
    
    return D_proj


def classical_mds(B: torch.Tensor, d_out: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """
    Classical multidimensional scaling from Gram matrix.
    
    Args:
        B: (n, n) Gram-like matrix (from double-centered squared distances)
        d_out: output dimensionality
        
    Returns:
        coords: (n, d_out) embedded coordinates
    """
    # Adaptive regularization based on matrix scale
    B_scale = torch.abs(B).max()
    if B_scale < 1e-6:
        B_scale = torch.tensor(1.0, device=B.device)
    
    eps_adaptive = max(eps, 1e-4 * B_scale)  # Scale eps with matrix magnitude
    B_reg = B + eps_adaptive * torch.eye(B.shape[0], device=B.device)
    
    eigvals, eigvecs = torch.linalg.eigh(B_reg)
    eigvals = eigvals.flip(0).clamp(min=0)
    eigvecs = eigvecs.flip(1)
    
    # Take top d_out eigenpairs
    D = min(d_out, eigvals.shape[0])
    eigvals_top = eigvals[:D]
    eigvecs_top = eigvecs[:, :D]
    
    # coords = U @ sqrt(Λ)
    coords = eigvecs_top @ torch.diag(torch.sqrt(eigvals_top))
    
    # Pad if needed
    if D < d_out:
        padding = torch.zeros(coords.shape[0], d_out - D, device=coords.device)
        coords = torch.cat([coords, padding], dim=1)
    
    return coords


def canonicalize_coords(y: torch.Tensor) -> torch.Tensor:
    """
    Deterministic canonicalization: center, isotropic scale, proper rotation, fixed sign.
    
    Args:
        y: (n, d) coordinates
        
    Returns:
        y_canon: (n, d) canonicalized coordinates
    """
    # Center
    y_centered = y - y.mean(dim=0, keepdim=True)
    
    # Isotropic scale
    scale = torch.sqrt((y_centered ** 2).sum() / y.shape[0])
    if scale < 1e-8:
        scale = torch.tensor(1.0, device=y.device)
    y_scaled = y_centered / scale
    
    # SVD for proper rotation (align to principal axes)
    U, S, Vt = torch.linalg.svd(y_scaled, full_matrices=False)
    
    # Ensure determinant = +1 (proper rotation)
    if torch.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    y_aligned = y_scaled @ Vt.t()
    
    # Fixed sign convention (e.g., first point has positive x-coordinate)
    if y_aligned[0, 0] < 0:
        y_aligned[:, 0] *= -1
    
    return y_aligned


# ==============================================================================
# PART 5: BATCHING & MASKING UTILITIES
# ==============================================================================

def pad_lists_to_maxlen(
    tensors: List[torch.Tensor],
    max_len: int,
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length tensors to max_len.
    
    Args:
        tensors: list of (n_i, d) tensors
        max_len: maximum length
        pad_value: padding value
        
    Returns:
        padded: (batch, max_len, d) padded tensor
        mask: (batch, max_len) boolean mask (True = real data)
    """
    batch_size = len(tensors)
    d = tensors[0].shape[1] if tensors[0].ndim > 1 else 1
    device = tensors[0].device
    
    if tensors[0].ndim == 1:
        padded = torch.full((batch_size, max_len), pad_value, device=device)
        for i, t in enumerate(tensors):
            n = t.shape[0]
            padded[i, :n] = t
    else:
        padded = torch.full((batch_size, max_len, d), pad_value, device=device)
        for i, t in enumerate(tensors):
            n = t.shape[0]
            padded[i, :n, :] = t
    
    # Create mask
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        mask[i, :n] = True
    
    return padded, mask


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask from lengths.
    
    Args:
        lengths: (batch,) lengths of each sequence
        max_len: maximum length
        
    Returns:
        mask: (batch, max_len) boolean mask
    """
    batch_size = lengths.shape[0]
    device = lengths.device
    
    mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask


def block_diag_mask(slide_ids: torch.Tensor) -> torch.Tensor:
    """
    Create block-diagonal mask for multi-slide batches.
    
    Args:
        slide_ids: (batch,) slide identifiers
        
    Returns:
        mask: (batch, batch) boolean mask (True = same slide)
    """
    return slide_ids.unsqueeze(0) == slide_ids.unsqueeze(1)


# ==============================================================================
# PART 6: LOSS FUNCTIONS
# ==============================================================================

class HeatKernelLoss(nn.Module):
    """Heat kernel trace matching loss."""
    
    def __init__(self, t_list: List[float] = [0.25, 1.0, 4.0]):
        super().__init__()
        self.t_list = t_list
    
    def forward(
        self,
        L_pred: torch.Tensor,
        L_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute heat kernel trace loss.
        
        Args:
            L_pred: (n, n) predicted Laplacian
            L_target: (n, n) target Laplacian
            mask: (n,) optional mask for valid nodes
            
        Returns:
            loss: scalar heat kernel trace loss
        """
        # Apply mask to Laplacians
        if mask is not None:
            n_valid = mask.sum().item()
            valid_idx = torch.where(mask)[0]
            L_pred = L_pred[valid_idx][:, valid_idx]
            L_target = L_target[valid_idx][:, valid_idx]
        
        # Eigendecomposition
        eigvals_pred = torch.linalg.eigvalsh(L_pred).clamp(min=0)
        eigvals_target = torch.linalg.eigvalsh(L_target).clamp(min=0)
        
        # Compute traces for each t
        loss = 0.0
        for t in self.t_list:
            trace_pred = torch.sum(torch.exp(-t * eigvals_pred))
            trace_target = torch.sum(torch.exp(-t * eigvals_target))
            loss += (trace_pred - trace_target) ** 2
        
        return loss / len(self.t_list)


class SlicedWassersteinLoss1D(nn.Module):
    """1D Sliced Wasserstein distance for distance histograms."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hist_pred: torch.Tensor, hist_target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D Sliced Wasserstein distance between histograms.
        For 1D, this is just the L1 distance between CDFs.
        
        Args:
            hist_pred: (num_bins,) predicted histogram (normalized)
            hist_target: (num_bins,) target histogram (normalized)
            
        Returns:
            loss: scalar SW distance
        """
        # Compute CDFs
        cdf_pred = torch.cumsum(hist_pred, dim=0)
        cdf_target = torch.cumsum(hist_target, dim=0)
        
        # L1 distance between CDFs
        loss = torch.mean(torch.abs(cdf_pred - cdf_target))
        return loss


class OrdinalTripletLoss(nn.Module):
    """Ordinal triplet hinge loss."""
    
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        D_pred: torch.Tensor,
        triplets: torch.Tensor,
        margin: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute ordinal triplet loss.
        
        Args:
            D_pred: (n, n) predicted distance matrix
            triplets: (T, 3) triplet indices [i, j, k]
            margin: optional margin override
            
        Returns:
            loss: scalar triplet loss
        """
        if margin is None:
            margin = self.margin
        
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        
        # D[i,j]^2 should be less than D[i,k]^2
        d_ij_sq = D_pred[i, j] ** 2
        d_ik_sq = D_pred[i, k] ** 2
        
        # Hinge loss: max(0, d_ij^2 - d_ik^2 + margin)
        loss = torch.clamp(d_ij_sq - d_ik_sq + margin, min=0).mean()
        return loss


class FrobeniusGramLoss(nn.Module):
    """Frobenius norm loss for Gram matrices."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        G_pred: torch.Tensor,
        G_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Frobenius loss between Gram matrices.
        
        Args:
            G_pred: (n, n) predicted Gram matrix
            G_target: (n, n) target Gram matrix
            mask: (n,) optional mask for valid nodes
            
        Returns:
            loss: scalar Frobenius loss
        """
        if mask is not None:
            # Apply mask to both dimensions
            valid_idx = torch.where(mask)[0]
            G_pred = G_pred[valid_idx][:, valid_idx]
            G_target = G_target[valid_idx][:, valid_idx]
        
        diff = G_pred - G_target
        loss = torch.mean(diff ** 2)
        return loss