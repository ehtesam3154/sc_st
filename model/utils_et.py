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

def canonicalize(V, mask, eps=1e-8):
    '''
    center and scale V to unit RMS per set
    V: (B, N, D_latent), mask: (B, N) in {0,1}
    returns V_centered_sacled, mean, scale
    '''
    B, N, D = V.shape
    mask_expanded = mask.unsqueeze(-1) #(B, N, 1)

    #compute mean over valid rows
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (V * mask_expanded).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1) #(B, 1, D)

    #center 
    V_centered = V - mean
    V_centered = V_centered * mask_expanded

    #compute rms scale
    sq_norms = (V_centered ** 2).sum(dim=-1) #(B, N)
    rms_sq = (sq_norms * mask).sum(dim=1) / valid_counts.squeeze(-1)
    scale = torch.sqrt(rms_sq + eps).unsqueeze(1).unsqueeze(2)

    #scale
    V_scaled = V_centered / scale
    V_scaled = V_scaled * mask_expanded

    return V_scaled, mean, scale

def canonicalize_unit_rms(V, mask, eps=1e-8):
    """
    Center and scale V to unit RMS per set.
    Used ONLY for scale-invariant losses (SW losses).
    """
    B, N, D = V.shape
    mask_expanded = mask.unsqueeze(-1)
    
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (V * mask_expanded).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
    
    V_centered = (V - mean) * mask_expanded
    
    sq_norms = (V_centered ** 2).sum(dim=-1)
    rms_sq = (sq_norms * mask).sum(dim=1) / valid_counts.squeeze(-1)
    scale = torch.sqrt(rms_sq + eps).unsqueeze(1).unsqueeze(2)
    
    V_scaled = V_centered / scale
    V_scaled = V_scaled * mask_expanded
    
    return V_scaled, mean, scale


def center_only(V, mask, eps=1e-8):
    """
    Center V per set, DO NOT rescale.
    This is what diffusion + Gram + heat should use.
    """
    B, N, D = V.shape
    mask_expanded = mask.unsqueeze(-1)
    
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (V * mask_expanded).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
    
    V_centered = (V - mean) * mask_expanded
    return V_centered, mean

def pairwise_dist2(V, mask):
    '''
    squared pairwise distances
    V: (B, N, D), mask: (B, N)
    returns D2: (B, N, N) with masked pairs set to 0
    '''
    B, N, D = V.shape

    #compute squared distances
    V_norm = (V ** 2).sum(dim=-1, keepdim=True) # (B, N, 1)
    D2 = V_norm + V_norm.transpose(1, 2) - 2 * torch.bmm(V, V.transpose(1, 2))

    #mask invalid pairs
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    D2 = D2 * mask_2d 

    return D2

def make_distance_bias(D2, mask, n_bins=16, d_emb=32, share_across_heads=True,
                       E_bin=None, W=None, alpha_bias=None, device=None, bin_edges=None):
    '''
    convert squared distances to attention bias
    returns attn_bias: (B, 1, N, N) if share_across_heads else (B, H, N, N)
    also returns bin_ids for distogram supervision
    '''

    #set bin edges to None to not have distogram loss

    B, N, _ = D2.shape
    device = D2.device if device is None else device

    # Init learnable params if not provided
    if E_bin is None:
        E_bin = nn.Parameter(torch.randn(n_bins, d_emb, device=device) / math.sqrt(d_emb))
    if W is None:
        out_dim = 1 if share_across_heads else 8
        W = nn.Parameter(torch.randn(d_emb, out_dim, device=device) * 0.01)
    if alpha_bias is None:
        alpha_bias = nn.Parameter(torch.tensor(0.1, device=device))

    D = torch.sqrt(D2.clamp(min=1e-8))

    # Use provided bin_edges or fall back to linear spacing
    if bin_edges is not None:
        edges = bin_edges.to(device)
        bin_ids = torch.bucketize(D.contiguous(), edges.contiguous()) - 1
        bin_ids = bin_ids.clamp(min=0, max=n_bins - 1)
    else:
        edges = torch.linspace(0, 3.0, n_bins, device=device)
        bin_ids = torch.searchsorted(edges, D.flatten()).reshape(B, N, N)
        bin_ids = torch.clamp(bin_ids, 0, n_bins - 1)

    bin_embeddings = E_bin[bin_ids]

    bias_raw = torch.matmul(bin_embeddings, W)
    
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    
    if share_across_heads:
        bias = bias_raw.squeeze(-1) * mask_2d
        attn_bias = alpha_bias * bias.unsqueeze(1)
    else:
        bias = bias_raw.permute(0, 3, 1, 2) * mask_2d.unsqueeze(1)
        attn_bias = alpha_bias * bias
    
    return attn_bias, (E_bin, W, alpha_bias), bin_ids

def knn_graph(V, mask, k=12):
    """
    Build kNN graph on coordinates.
    Returns idx: (B, N, k) with -1 for invalid neighbors
    """
    B, N, D = V.shape
    device = V.device
    
    # Compute distances
    D2 = pairwise_dist2(V, mask)
    
    # Mask self-connections
    D2 = D2 + torch.eye(N, device=device).unsqueeze(0) * 1e10
    
    # Mask invalid nodes
    invalid_mask = (~mask).unsqueeze(1).float() * 1e10
    D2 = D2 + invalid_mask
    
    # Find k nearest neighbors
    _, idx = torch.topk(D2, k, dim=-1, largest=False)  # (B, N, k)
    
    # Mark invalid neighbors as -1
    valid_neighbors = mask.unsqueeze(2).expand(-1, -1, k)  # (B, N, k)
    neighbor_mask = torch.gather(mask.unsqueeze(1).expand(-1, N, -1), 2, idx)  # (B, N, k)
    idx = torch.where(valid_neighbors & neighbor_mask, idx, torch.tensor(-1, device=device))
    
    return idx

from functools import lru_cache

@lru_cache(maxsize=64)
def _triu_indices_cached(k: int, device: torch.device):
    #upper-triangular pair indices (j>k) once per k
    return torch.triu_indices(k, k, offset=1, device=device)


def angle_features(V, mask, idx, n_angle_bins: int = 8, eps: float = 1e-8):
    """
    Vectorized angle histogram per node from neighbor triangles (no Python loops).

    Args
    ----
    V   : (B, N, D) latent coordinates
    mask: (B, N) bool, True = valid node
    idx : (B, N, k) long, neighbor indices per node, -1 indicates missing
    n_angle_bins: number of bins for cos(theta) in [-1,1]
    eps : numerical epsilon

    Returns
    -------
    angle_hist : (B, N, n_angle_bins), rows sum to 1 for valid nodes (0 for pads)
    """
    B, N, D = V.shape
    k = idx.shape[-1]
    device = V.device

    # Clamp negative indices so we can safely index, but remember validity
    idx_clamped = idx.clamp_min(0)                              # (B, N, k)
    # Build neighbor validity: (neighbor exists) & (neighbor not padded)
    # Gather neighbor-node masks with advanced indexing
    b_ix = torch.arange(B, device=device)[:, None, None].expand(B, N, k)
    nb_mask_from_nodes = mask[b_ix, idx_clamped]                # (B, N, k)
    neighbor_valid = (idx >= 0) & nb_mask_from_nodes            # (B, N, k)

    # Gather neighbor coordinates: V_neighbors[b,i,j,:] = V[b, idx[b,i,j], :]
    V_neighbors = V[b_ix, idx_clamped, :]                       # (B, N, k, D)

    # Centered neighbor rays U = V_j - V_i
    V_center = V.unsqueeze(2)                                   # (B, N, 1, D)
    U = V_neighbors - V_center                                  # (B, N, k, D)

    # Normalize rays to unit length
    U_norm = torch.linalg.norm(U, dim=-1).clamp_min(eps)        # (B, N, k)
    U_unit = U / U_norm.unsqueeze(-1)                           # (B, N, k, D)

    # Zero-out invalid neighbors to avoid polluting Gram
    U_unit = U_unit * neighbor_valid.unsqueeze(-1).to(U.dtype)

    # Neighbor–neighbor Gram per node: all cosines at once
    # G[b,n,j,k] = <u_j, u_k>; shape (B, N, k, k)
    G = torch.matmul(U_unit, U_unit.transpose(-1, -2))

    # Select upper-triangular pairs j<k (each is an angle at node center)
    i_idx, j_idx = _triu_indices_cached(k, device)
    # Cosines for all pairs at once: (B, N, P)
    cos_all = G[:, :, i_idx, j_idx]

    # Valid-pair mask: both neighbors must be valid
    pair_valid = neighbor_valid[:, :, i_idx] & neighbor_valid[:, :, j_idx]  # (B, N, P)

    # Bin edges and bucketize cos(theta) in [-1,1]
    bin_edges = torch.linspace(-1.0, 1.0, n_angle_bins + 1, device=device)
    BN = B * N
    P = cos_all.shape[-1]

    cos_flat = cos_all.reshape(BN, P)
    valid_flat = pair_valid.reshape(BN, P).to(cos_flat.dtype)

    # Bucketize (no grad needed for histogramming)
    bin_idx = torch.bucketize(cos_flat, bin_edges) - 1          # (BN, P) in [-1, n_bins-1]
    bin_idx = bin_idx.clamp_(0, n_angle_bins - 1)

    # Scatter-add into per-node histograms
    # Map (row r, bin b) -> flat index r*n_bins + b
    row_offsets = (torch.arange(BN, device=device) * n_angle_bins).unsqueeze(1)  # (BN,1)
    flat_idx = (row_offsets + bin_idx).reshape(-1)                                # (BN*P,)
    weights = valid_flat.reshape(-1)                                              # (BN*P,)

    hist_flat = torch.zeros(BN * n_angle_bins, device=device, dtype=cos_flat.dtype)
    hist_flat.index_add_(0, flat_idx, weights)
    hist = hist_flat.view(BN, n_angle_bins)

    # Normalize per node; zeros for padded rows
    hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    hist = hist.view(B, N, n_angle_bins)

    # Zero-out padded nodes explicitly
    hist = hist * mask.unsqueeze(-1).to(hist.dtype)

    return hist


def edm_cone_penalty_from_V(V, mask, min_eigs=8):
    """
    Compute EDM cone penalty (sum of negative eigenvalues).
    V: (B, N, D_latent), mask: (B, N)
    Returns scalar penalty averaged over batch
    """
    B = V.shape[0]
    penalties = []
    
    for b in range(B):
        valid_idx = torch.where(mask[b])[0]
        n_valid = len(valid_idx)
        
        if n_valid < 2:
            penalties.append(torch.tensor(0.0, device=V.device))
            continue
        
        # Extract valid subset
        V_valid = V[b, valid_idx]  # (n_valid, D)
        
        # Compute squared distances
        D2 = torch.cdist(V_valid, V_valid, p=2) ** 2
        
        # Double-center to get Gram
        ones = torch.ones(n_valid, 1, device=V.device)
        J = torch.eye(n_valid, device=V.device) - ones @ ones.T / n_valid
        G = -0.5 * J @ D2 @ J
        
        # Compute eigenvalues
        eigvals = torch.linalg.eigvalsh(G)
        
        # Sum negative eigenvalues
        neg_eigvals = torch.clamp(-eigvals, min=0)
        penalty = neg_eigvals.sum()
        penalties.append(penalty)
    
    return torch.stack(penalties).mean()


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

def compute_graph_laplacian(edge_index: torch.Tensor,
                            edge_weight: torch.Tensor,
                            n_nodes: int,
                            normalized: bool = True,
                            eps: float = 1e-8) -> torch.Tensor:
    """
    Graph Laplacian. Returns a **sparse** tensor.
    If normalized=True, returns L_sym = I - D^{-1/2} W D^{-1/2}.
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]
    w = edge_weight

    # make undirected
    idx_i = torch.cat([src, dst], dim=0)
    idx_j = torch.cat([dst, src], dim=0)
    vals  = torch.cat([w,   w  ], dim=0)

    W = torch.sparse_coo_tensor(
        torch.stack([idx_i, idx_j], dim=0),
        vals,
        (n_nodes, n_nodes),
        device=device
    ).coalesce()

    # degree
    deg = torch.sparse.sum(W, dim=1).to_dense().clamp_min(eps)

    if normalized:
        # S = D^{-1/2} W D^{-1/2}
        di = 1.0 / torch.sqrt(deg)
        # scale edge weights
        row, col = W.indices()
        s_vals = W.values() * di[row] * di[col]
        S = torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0),
            s_vals,
            (n_nodes, n_nodes),
            device=device
        ).coalesce()
        # L = I - S  (sparse)
        eye_idx = torch.arange(n_nodes, device=device)
        I = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx], dim=0),
            torch.ones(n_nodes, device=device),
            (n_nodes, n_nodes),
            device=device
        )
        L = (I - S).coalesce()
        return L
    else:
        # L = D - W
        eye_idx = torch.arange(n_nodes, device=device)
        D = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx], dim=0),
            deg,
            (n_nodes, n_nodes),
            device=device
        )
        L = (D - W).coalesce()
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
    
def _symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.mT)

def safe_eigh(A, cpu=False, return_vecs=True, regularize=1e-6):
    """
    Compute eigendecomposition on GPU with numerical stability.
    
    Args:
        A: Symmetric matrix (N, N) or (B, N, N)
        cpu: Ignored (kept for API compatibility)
        return_vecs: Whether to return eigenvectors
        regularize: Small value added to diagonal for stability
    
    Returns:
        eigvals: Eigenvalues (N,) or (B, N)
        eigvecs: Eigenvectors if return_vecs=True, else None
    """
    # Ensure float32 minimum for numerical stability
    if A.dtype == torch.float16 or A.dtype == torch.bfloat16:
        A = A.float()
    
    # Add small regularization to diagonal for positive definiteness
    if A.dim() == 2:
        eye = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_reg = A + regularize * eye
    elif A.dim() == 3:
        B, N, _ = A.shape
        eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        A_reg = A + regularize * eye
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {A.shape}")
    
    # Compute eigendecomposition on GPU
    if return_vecs:
        eigvals, eigvecs = torch.linalg.eigh(A_reg)
        return eigvals, eigvecs
    else:
        eigvals = torch.linalg.eigvalsh(A_reg)
        return eigvals, None


# utils_et.py
import torch
import torch.nn.functional as F
from functools import lru_cache

# def compute_distance_hist(D: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
#     """
#     CUDA histogram over the upper triangle.
#     D: (n, n) or (B, n, n) distances; bins: (nb+1,)
#     Returns: (nb,) or (B, nb) normalized.
#     """
#     assert D.is_cuda and bins.is_cuda, "Put D and bins on CUDA"
#     if D.dim() == 2:
#         n = D.size(0)
#         iu, ju = torch.triu_indices(n, n, 1, device=D.device)
#         d = D[iu, ju]
#         hist, _ = torch.histogram(d, bins=bins)
#         return hist.float() / hist.sum().clamp_min(1)
#     else:
#         B, n, _ = D.shape
#         iu, ju = torch.triu_indices(n, n, 1, device=D.device)
#         d = D[:, iu, ju]                           # (B, M)
#         nb = bins.numel() - 1
#         ids = torch.bucketize(d, bins) - 1
#         ids = ids.clamp(0, nb - 1)
#         offs = torch.arange(B, device=D.device).unsqueeze(1) * nb
#         flat = (ids + offs).reshape(-1)
#         counts = torch.bincount(flat, minlength=B * nb).view(B, nb).float()
#         return counts / counts.sum(dim=1, keepdim=True).clamp_min(1)

import torch

@torch.no_grad()
def compute_distance_hist(
    D: torch.Tensor,
    bins: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Distance histogram over upper triangle using bucketize+bincount.
    Works on CPU and CUDA; batched (B,n,n) or single (n,n).

    Args
    ----
    D : (n,n) or (B,n,n) tensor of distances
    bins : (nb+1,) monotonically increasing bin edges (on any device)
    pair_mask : optional (n,n) or (B,n,n) boolean mask over pairs.
                True means "include". If None, uses upper-tri mask.

    Returns
    -------
    hist : (nb,) or (B, nb) normalized histogram (sums to 1, safe-div)
    """
    # Ensure tensors are on the same device & contiguous
    if bins.device != D.device:
        bins = bins.to(D.device, non_blocking=True)
    D = D.contiguous()
    bins = bins.contiguous()

    if D.dim() == 2:
        n = D.size(0)
        if n < 2:
            nb = bins.numel() - 1
            return torch.zeros(nb, device=D.device, dtype=D.dtype)
        iu, ju = torch.triu_indices(n, n, offset=1, device=D.device)
        d = D[iu, ju]  # (M,)
        if pair_mask is not None:
            pm = pair_mask.to(torch.bool, non_blocking=True)
            m = pm[iu, ju]
            d = d[m]
        if d.numel() == 0:
            nb = bins.numel() - 1
            return torch.zeros(nb, device=D.device, dtype=D.dtype)

        # Map to bins: right-closed last bin behavior
        nb = bins.numel() - 1
        ids = torch.bucketize(d, bins, right=True) - 1
        ids = ids.clamp_(0, nb - 1)

        counts = torch.bincount(ids, minlength=nb).to(dtype=D.dtype)
        total = counts.sum().clamp_min(1)
        return counts / total

    elif D.dim() == 3:
        B, n, _ = D.shape
        if n < 2:
            nb = bins.numel() - 1
            return torch.zeros(B, nb, device=D.device, dtype=D.dtype)

        iu, ju = torch.triu_indices(n, n, offset=1, device=D.device)
        # (B, M)
        d = D[:, iu, ju]
        if pair_mask is not None:
            pm = pair_mask.to(torch.bool, non_blocking=True)
            valid = pm[:, iu, ju]
            # mark invalid as a sentinel outside all bins (we'll drop them)
            d = torch.where(valid, d, torch.full_like(d, float("nan")))

        nb = bins.numel() - 1
        # bucketize does not support NaN — mask them before binning
        if pair_mask is not None:
            nanmask = torch.isnan(d)
            d = torch.where(nanmask, torch.full_like(d, bins[0] - 1), d)

        ids = torch.bucketize(d, bins, right=True) - 1  # (B, M)
        ids = ids.clamp(0, nb - 1)

        if pair_mask is not None:
            # drop the previously NaN positions
            ids = torch.where(nanmask, torch.full_like(ids, -1), ids)

        # Batched bincount via offsets
        offs = (torch.arange(B, device=D.device).unsqueeze(1) * nb)  # (B,1)
        flat = (ids + offs).reshape(-1)
        if pair_mask is not None:
            flat = flat[flat >= 0]

        counts = torch.bincount(flat, minlength=B * nb).view(B, nb).to(dtype=D.dtype)
        totals = counts.sum(dim=1, keepdim=True).clamp_min(1)
        return counts / totals

    else:
        raise ValueError(f"compute_distance_hist: expected D dim 2 or 3, got {D.dim()}")


import torch
import torch.nn.functional as F

def _get_device_for_build(x: torch.Tensor) -> torch.device:
    if x.is_cuda:
        return x.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def build_topk_index(
    Z_all: torch.Tensor,
    K: int = 2048,
    block: int = 2048,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Build a (N, K) top-k neighbor index for embeddings Z_all.
    - Runs on CUDA if available.
    - Uses block matmul to bound memory.
    - Casts to float32 for stable top-k.
    - If torch.distributed is initialized, rank 0 computes once and broadcasts to all ranks.
    - Returns a CPU pinned (N, K) LongTensor for fast DataLoader use.
    """
    import torch.distributed as dist

    device = _get_device_for_build(Z_all)
    Z = Z_all.to(device, non_blocking=True).contiguous()
    Z = Z.float()  # AMP-safe & numerically stable for similarity
    N, d = Z.shape

    # Clamp args to dataset size
    K = int(max(1, min(K, N)))          # at least 1, at most N
    block = int(max(1, min(block, N)))  # at least 1, at most N

    # DDP: compute once on rank 0, broadcast to others
    ddp = torch.distributed.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0

    if rank == 0:
        if metric == "cosine":
            Zn = F.normalize(Z, dim=1)
            Zt = Zn.t().contiguous()                     # (d, N)
            chunks = []
            for s in range(0, N, block):
                e = min(s + block, N)
                # (b, N) similarities
                S = Zn[s:e] @ Zt                         # CUDA GEMM
                # NOTE: includes self; OK for our sampler
                _, idx = torch.topk(S, K, dim=1, largest=True, sorted=False)
                chunks.append(idx)
            nbr_idx_cuda = torch.cat(chunks, dim=0)      # (N, K) on device
        else:  # 'l2'
            nbr_idx_cuda = torch.empty(N, K, dtype=torch.long, device=device)
            for s in range(0, N, block):
                e = min(s + block, N)
                D = torch.cdist(Z[s:e], Z)               # (b, N)
                _, idx = torch.topk(D, K, dim=1, largest=False, sorted=False)
                nbr_idx_cuda[s:e] = idx

        # allocate the broadcast buffer on device
        bcast_buf = nbr_idx_cuda
    else:
        # non-src ranks: allocate empty buffer with correct shape/dtype on device
        bcast_buf = torch.empty(N, K, dtype=torch.long, device=device)

    if ddp:
        # all ranks participate; src=0 sends
        dist.broadcast(bcast_buf, src=0)

    # move to CPU pinned for DataLoader workers
    nbr_idx = bcast_buf.cpu().pin_memory()
    return nbr_idx



def cached_triu_idx(n: int, k: int = 1):
    iu, ju = torch.triu_indices(n, n, k)
    return iu, ju 


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
    # eigvals, eigvecs = torch.linalg.eigh(G)  # Ascending order
    eigvals, eigvecs = safe_eigh(G, return_vecs=True)
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
    # eigvals, eigvecs = torch.linalg.eigh(B)
    eigvals, eigvecs = safe_eigh(B, return_vecs=True)
    
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

    # B_reg = B + eps_adaptive * torch.eye(B.shape[0], device=B.device) 
    # eigvals, eigvecs = torch.linalg.eigh(B_reg)

    B_reg = _symmetrize(B) + eps_adaptive * torch.eye(B.shape[0], device=B.device)
    eigvals, eigvecs = safe_eigh(B_reg, return_vecs=True)

    eigvals = eigvals.flip(0).clamp(min=0)
    eigvecs = eigvecs.flip(1)

    if eigvals.max() < 1e-12:
        # utterly degenerate; return zeros to avoid crashes
        return torch.zeros(B.shape[0], d_out, device=B.device, dtype=B.dtype)
    
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



class HeatKernelLoss(nn.Module):
    """
    Heat kernel trace matching loss with optional Hutchinson trace estimation.
    
    For batched inputs (B, N, N), uses fast batched eigendecomposition.
    For single inputs (N, N), can use Hutchinson for very large graphs.
    
    Args:
        use_hutchinson: Use Hutchinson trace estimation for single samples
        num_probes: Number of Rademacher probes for Hutchinson
        chebyshev_degree: Degree of Chebyshev polynomial approximation
        knn_k: k for kNN graph Laplacian (unused in loss, kept for API)
        t_list: Heat kernel diffusion times
        laplacian: Type of Laplacian (unused, kept for API)
    """
    
    def __init__(
        self,
        use_hutchinson: bool = True,
        num_probes: int = 8,
        chebyshev_degree: int = 10,
        knn_k: int = 8,
        t_list: Tuple[float, ...] = (0.5, 1.0),
        laplacian: str = 'sym'
    ):
        super().__init__()
        self.use_hutchinson = use_hutchinson
        self.num_probes = num_probes
        self.chebyshev_degree = chebyshev_degree
        self.knn_k = knn_k
        self.t_list = t_list
        self.laplacian = laplacian
    
    def forward(
        self,
        L_pred: torch.Tensor,
        L_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        t_list: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Compute heat kernel trace loss.
        
        Args:
            L_pred: (N, N) or (B, N, N) predicted Laplacian
            L_target: (N, N) or (B, N, N) target Laplacian  
            mask: (N,) or (B, N) optional mask for valid nodes
            t_list: Optional override for diffusion times
            
        Returns:
            loss: Scalar heat kernel trace loss
        """
        if t_list is None:
            t_list = self.t_list
        
        is_batched = L_pred.dim() == 3
        
        if not is_batched:
            # ===== SINGLE SAMPLE PATH =====
            if mask is not None:
                mask = mask.bool().view(-1)
                
                # Convert sparse to dense if needed
                if L_pred.layout != torch.strided:
                    L_pred = L_pred.to_dense()
                if L_target.layout != torch.strided:
                    L_target = L_target.to_dense()
                
                # Index select valid nodes
                valid_idx = mask.nonzero(as_tuple=False).squeeze(1)
                L_pred = L_pred.index_select(0, valid_idx).index_select(1, valid_idx).contiguous()
                L_target = L_target.index_select(0, valid_idx).index_select(1, valid_idx).contiguous()
            
            # Use Hutchinson or full eigendecomposition
            if self.use_hutchinson and L_pred.size(0) > 100:
                traces_pred = self._hutchinson_heat_trace(L_pred, t_list)
                traces_target = self._hutchinson_heat_trace(L_target, t_list)
            else:
                # Full eigendecomposition (fast for small matrices)
                eigvals_pred, _ = safe_eigh(L_pred, return_vecs=False)
                eigvals_target, _ = safe_eigh(L_target, return_vecs=False)
                eigvals_pred = eigvals_pred.clamp(min=0)
                eigvals_target = eigvals_target.clamp(min=0)
                
                traces_pred = torch.stack([torch.sum(torch.exp(-t * eigvals_pred)) for t in t_list])
                traces_target = torch.stack([torch.sum(torch.exp(-t * eigvals_target)) for t in t_list])
            
            # loss = torch.mean((traces_pred - traces_target) ** 2)
            # return loss
        
            # Relative error (scale-invariant)
            diff = traces_pred - traces_target
            num = (diff ** 2).sum()
            den = (traces_target ** 2).sum().clamp_min(1e-12)
            loss = num / den
            return loss
        
        else:
            # ===== BATCHED PATH (B, N, N) - FULLY VECTORIZED =====
            B, N, _ = L_pred.shape
            device = L_pred.device
            
            # Convert sparse to dense if needed
            if L_pred.layout != torch.strided:
                L_pred = L_pred.to_dense()
            if L_target.layout != torch.strided:
                L_target = L_target.to_dense()
            
            # Apply mask to zero out invalid entries
            if mask is not None:
                mask = mask.bool()  # (B, N)
                mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
                L_pred = L_pred * mask_2d.float()
                L_target = L_target * mask_2d.float()
            
            # Batched eigendecomposition - fast for small N!
            eigvals_pred, _ = safe_eigh(L_pred, return_vecs=False)  # (B, N)
            eigvals_target, _ = safe_eigh(L_target, return_vecs=False)  # (B, N)
            
            # Clamp negative eigenvalues
            eigvals_pred = eigvals_pred.clamp(min=0)
            eigvals_target = eigvals_target.clamp(min=0)
            
            # Compute heat kernel traces: tr(exp(-t*L)) = sum_i exp(-t*lambda_i)
            # Vectorized over batch and time
            traces_pred_list = []
            traces_target_list = []
            
            for t in t_list:
                # (B, N) -> (B,) by summing over eigenvalues
                tr_pred = torch.sum(torch.exp(-t * eigvals_pred), dim=1)
                tr_target = torch.sum(torch.exp(-t * eigvals_target), dim=1)
                traces_pred_list.append(tr_pred)
                traces_target_list.append(tr_target)
            
            traces_pred = torch.stack(traces_pred_list, dim=1)  # (B, T)
            traces_target = torch.stack(traces_target_list, dim=1)  # (B, T)
            
            # MSE loss over batch and time points
            # loss = ((traces_pred - traces_target) ** 2).mean()
            # return loss

            # Relative Frobenius error (scale-invariant)
            diff = traces_pred - traces_target  # (B, T)
            num = (diff ** 2).sum(dim=1)  # (B,)
            den = (traces_target ** 2).sum(dim=1).clamp_min(1e-12)  # (B,)
            loss = (num / den).mean()
            return loss
    
    def _hutchinson_heat_trace(
        self,
        L: torch.Tensor,
        t_list: List[float]
    ) -> torch.Tensor:
        """
        Hutchinson trace estimation with Lanczos tridiagonalization.
        Used for single large matrices only.
        
        Args:
            L: (N, N) Laplacian matrix
            t_list: List of diffusion times
            
        Returns:
            traces: (len(t_list),) trace estimates
        """
        device = L.device
        n = L.shape[0]
        
        # Estimate max eigenvalue
        lambda_max = self._power_iter_max_eig(L, num_iter=20)
        
        # Scale L to [-1, 1]
        L_scaled = L / (lambda_max + 1e-8)
        
        # Hutchinson trace estimation
        traces = torch.zeros(len(t_list), device=device, dtype=torch.float32)
        
        # Average over multiple random probes
        for _ in range(self.num_probes):
            # Rademacher probe: random ±1
            v = torch.empty(n, device=device, dtype=torch.float32).uniform_(-1.0, 1.0).sign()
            v = v / (n ** 0.5)
            
            # Lanczos tridiagonalization
            T = self._lanczos_tridiag(L_scaled, v, self.chebyshev_degree)
            
            # Compute trace for each diffusion time
            for i, t in enumerate(t_list):
                t_scaled = t * lambda_max
                trace_t = self._chebyshev_exp_trace(T, t_scaled)
                traces[i] += trace_t
        
        traces = traces / float(self.num_probes)
        return traces
    
    @torch.no_grad()
    def _power_iter_max_eig(self, A: torch.Tensor, num_iter: int = 20) -> torch.Tensor:
        """Estimate maximum eigenvalue using power iteration."""
        n = A.shape[0]
        v = torch.randn(n, device=A.device, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)
        
        for _ in range(num_iter):
            Av = A @ v
            v = Av / (Av.norm() + 1e-12)
        
        lambda_max = torch.dot(v, A @ v)
        return lambda_max.abs()
    
    @torch.no_grad()
    def _lanczos_tridiag(
        self,
        A: torch.Tensor,
        v0: torch.Tensor,
        m: int,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """
        Lanczos tridiagonalization to reduce A to small tridiagonal matrix T.
        
        Returns:
            T: (m, m) tridiagonal matrix (or smaller if converged early)
        """
        device = A.device
        dtype = torch.float32
        
        alpha = torch.zeros(m, device=device, dtype=dtype)
        beta = torch.zeros(m - 1, device=device, dtype=dtype)
        
        v = v0 / (v0.norm() + 1e-12)
        w = A @ v
        alpha[0] = torch.dot(v, w)
        w = w - alpha[0] * v
        
        for j in range(1, m):
            beta[j - 1] = w.norm()
            if beta[j - 1] <= tol:
                # Early termination
                alpha = alpha[:j]
                beta = beta[:j - 1]
                break
            
            v_next = w / beta[j - 1]
            w = A @ v_next - beta[j - 1] * v
            alpha[j] = torch.dot(v_next, w)
            w = w - alpha[j] * v_next
            v = v_next
        
        # Build tridiagonal matrix
        T = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
        return T
    
    @torch.no_grad()
    def _chebyshev_exp_trace(self, T: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute trace of exp(-t*T) where T is small tridiagonal matrix.
        Uses eigendecomposition since T is small.
        """
        evals, evecs = torch.linalg.eigh(T)
        weights = evecs[0, :] ** 2
        return (weights * torch.exp(-t * evals)).sum()


class SlicedWassersteinLoss1D(nn.Module):
    """1D Sliced Wasserstein distance for distance histograms."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hist_pred: torch.Tensor, hist_target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D Sliced Wasserstein distance between histograms.
        For 1D, this is just the L1 distance between CDFs.
        
        Args:
            hist_pred: (num_bins,) OR (batch, num_bins) predicted histogram (normalized)
            hist_target: (num_bins,) OR (batch, num_bins) target histogram (normalized)
            
        Returns:
            loss: scalar SW distance (averaged over batch if batched)
        """
        # Handle both single and batched inputs
        if hist_pred.dim() == 1:
            hist_pred = hist_pred.unsqueeze(0)
            hist_target = hist_target.unsqueeze(0)
        
        # Compute CDFs along the last dimension (bins)
        cdf_pred = torch.cumsum(hist_pred, dim=-1)      # (B, nb)
        cdf_target = torch.cumsum(hist_target, dim=-1)  # (B, nb)
        
        # L1 distance between CDFs
        # torch.mean averages over ALL elements (batch * bins)
        # This is equivalent to: mean_over_batch(mean_over_bins(|diff|))
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

def masked_frobenius_loss(A, B, mask, drop_diag: bool = True):
    """
    ||A - B||_F^2 over valid entries.
    A, B: (B, N, N) or (N, N)
    mask: (B, N) or (N,) boolean. Valid nodes.
    """
    if A.dim() == 3:
        Bsz, N, _ = A.shape
        P = (mask.unsqueeze(-1) & mask.unsqueeze(-2))  # (B,N,N) boolean
        if drop_diag:
            eye = torch.eye(N, dtype=torch.bool, device=A.device).unsqueeze(0)
            P = P & (~eye)
        P = P.float()
        diff_sq = (A - B).pow(2)
        return (diff_sq * P).sum() / P.sum().clamp_min(1.0)
    else:
        N = A.size(0)
        P = (mask.unsqueeze(1) & mask.unsqueeze(0))
        if drop_diag:
            eye = torch.eye(N, dtype=torch.bool, device=A.device)
            P = P & (~eye)
        P = P.float()
        diff_sq = (A - B).pow(2)
        return (diff_sq * P).sum() / P.sum().clamp_min(1.0)


def build_knn_graph_from_distance(D: torch.Tensor, k: int = 20, device: str = 'cuda'):
    """
    kNN graph from a distance matrix (no coordinates needed).
    D: (n, n) with zeros on diag, symmetric.
    """
    n = D.shape[0]
    D = _symmetrize(D)
    D = D + torch.eye(n, device=D.device, dtype=D.dtype) * 1e10  # mask self
    knn_dists, knn_idx = torch.topk(D, k, dim=1, largest=False)
    src = torch.arange(n, device=D.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    # Adaptive sigma from knn distances
    sigma = torch.median(knn_dists).item()
    if sigma < 1e-8:
        sigma = 1.0
    edge_weight = torch.exp(-(knn_dists.reshape(-1) ** 2) / (2 * sigma ** 2))
    return edge_index.long(), edge_weight.float()

def build_knn_graph(
    coords_or_features: torch.Tensor,
    k: int = 20,
    metric: str = 'euclidean',
    return_weights: bool = True,
    device: str = 'cuda'  # Keep param for backward compatibility, but ignore it
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build kNN graph and return edge_index + weights.
    Vectorized, no Python loops.
    
    Args:
        coords_or_features: (n, d) coordinates or features
        k: number of neighbors
        metric: distance metric
        return_weights: whether to compute RBF weights
        device: IGNORED - device is inferred from input tensor
        
    Returns:
        edge_index: (2, E) edge indices
        edge_weight: (E,) RBF weights (if return_weights=True)
    """
    # CRITICAL FIX: Infer device from input tensor, ignore parameter
    device = coords_or_features.device
    n = coords_or_features.shape[0]
    
    # Compute pairwise distances
    D = torch.cdist(coords_or_features, coords_or_features, p=2)  # (n, n)
    
    # Find k nearest neighbors (excluding self)
    D_noself = D + torch.eye(n, device=device) * 1e10  # Mask diagonal
    knn_dists, knn_indices = torch.topk(D_noself, k, dim=1, largest=False)  # (n, k)
    
    # Build edge_index - all tensors inherit device from input
    src = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten()  # (n*k,)
    dst = knn_indices.flatten()  # Already on correct device
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

import math 
import torch
import torch.nn as nn

#fast heat trace via strochastic lanczos quadratur (dense L)
@torch.no_grad()

def lanczos_dense(A, v0, m, tol=1e-6):
    """
    m-step Lanczos on symmetric A (dense or sparse). Returns tridiagonal T.
    """
    device, dtype = A.device, A.dtype
    n = v0.numel()
    Q = torch.zeros(n, m, device=device, dtype=dtype)
    alpha = torch.zeros(m, device=device, dtype=dtype)
    beta  = torch.zeros(m-1, device=device, dtype=dtype)

    def matvec(x):
        if A.is_sparse:
            return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        else:
            return A @ x

    v = v0 / (v0.norm() + 1e-12)
    w = matvec(v)
    alpha[0] = torch.dot(v, w)
    Q[:, 0] = v
    w = w - alpha[0] * v

    for j in range(1, m):
        beta[j-1] = w.norm()
        if beta[j-1] <= tol:
            alpha = alpha[:j]
            beta  = beta[:j-1]
            Q     = Q[:, :j]
            break
        v_next = w / beta[j-1]
        Q[:, j] = v_next
        w = matvec(v_next) - beta[j-1] * v
        alpha[j] = torch.dot(v_next, w)
        w = w - alpha[j] * v_next
        v = v_next

    T = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
    return T


@torch.no_grad()
def _quad_form_exp_from_T(T, t):
    """
    Compute e1^T exp(-t T) e1 for small symmetric T (j x j).
    """
    # j is small (<= m ~ 25-40), so dense eig is cheap
    evals, evecs = torch.linalg.eigh(T)          # on current device
    weights = (evecs[0, :] ** 2)                 # (j,)
    return (weights * torch.exp(-t * evals)).sum()


def heat_trace_slq_dense(L, t_list, num_probe=4, m=12):
    """
    Tr(exp(-t L)) per node via SLQ (supports sparse L).
    Returns tensor [len(t_list)], **already divided by n**.
    """
    device, dtype = L.device, L.dtype
    n = L.size(0)
    traces = torch.zeros(len(t_list), device=device, dtype=dtype)

    for _ in range(num_probe):
        v = torch.empty(n, device=device, dtype=dtype).uniform_(-1.0, 1.0).sign()
        v = v / (n ** 0.5)
        T = lanczos_dense(L, v, m)
        evals, evecs = torch.linalg.eigh(T)      # small j x j
        w1 = (evecs[0, :] ** 2)                  # e1^T f(T) e1 weights
        for j, t in enumerate(t_list):
            traces[j] += (w1 * torch.exp(-t * evals)).sum()

    traces = traces / float(num_probe)
    # Hutchinson gives (1/n) * trace when v is 1/sqrt(n)-scaled
    return traces  # per-node trace, roughly in [e^{-2t}, 1] for normalized L


import torch
import torch.nn.functional as F

@torch.no_grad()
def _upper_tri_vec(D):
    n = D.size(0)
    iu = torch.triu_indices(n, n, offset=1, device=D.device)
    return D[iu[0], iu[1]]  # (n*(n-1)/2,)


def wasserstein_1d_quantile_loss(
    D_pred, D_tgt, m_pairs=8192, p=1, norm="p95", rescale_back=True
):
    """
    Differentiable 1D OT on pairwise distances (upper-triangle).
    Returns scalar loss on the SAME device/dtype as inputs.
    - p=1 gives stronger, less-squashed gradients than p=2 early on.
    - norm in {"none","median","p95"} rescales *both* by a robust target scale s.
    - rescale_back=True multiplies loss by s**p to put it back in distance units.
    """
    dp = _upper_tri_vec(D_pred)
    dt = _upper_tri_vec(D_tgt)

    # optional subsample (same indices for both)
    if dp.numel() > m_pairs:
        idx = torch.randperm(dp.numel(), device=dp.device)[:m_pairs]
        dp = dp.index_select(0, idx)
        dt = dt.index_select(0, idx)

    # robust normalization (decouples global scale from shape)
    s = dp.new_tensor(1.0)
    if norm != "none":
        if norm == "median":
            s = dt.median().clamp_min(1e-6)
        else:  # "p95"
            s = torch.quantile(dt, 0.95).clamp_min(1e-6)
        dp = dp / s
        dt = dt / s

    dp_sorted = torch.sort(dp).values
    dt_sorted = torch.sort(dt).values

    if p == 1:
        diff = (dp_sorted - dt_sorted).abs()
    else:
        diff = (dp_sorted - dt_sorted).pow(2)

    # restore physical units so the magnitude isn't microscopic
    if rescale_back:
        if p == 1:
            diff = diff * s
        else:
            diff = diff * (s * s)

    return diff.mean()

import torch

def farthest_point_sampling(Z: torch.Tensor, k: int, device='cpu') -> torch.Tensor:
    """
    Farthest Point Sampling to select k representative points.
    
    Args:
        Z: (N, d) embeddings
        k: number of points to select
        device: 'cpu' recommended for large N
        
    Returns:
        (k,) tensor of selected indices
    """
    Z = Z.to(device)
    N = Z.shape[0]
    k = min(k, N)
    
    # Start with random point
    idx = torch.randint(0, N, (1,), device=device).item()
    selected = [idx]
    dist = torch.full((N,), float('inf'), device=device)
    
    for _ in range(1, k):
        # Distance from last selected to all points
        d = torch.cdist(Z[selected[-1]].unsqueeze(0), Z, p=2).squeeze(0)
        dist = torch.minimum(dist, d)
        selected.append(int(torch.argmax(dist)))
    
    return torch.tensor(selected, dtype=torch.long)


def mds_from_latent(X: torch.Tensor, d_out: int = 2) -> torch.Tensor:
    """
    Fast classical MDS directly from centered latent V_0 (N x D).
    
    Equivalent to classical MDS on the EDM induced by X, but O(N D²) instead of O(N³).
    Since distances are constructed from V_0 itself, this is mathematically exact.
    
    Args:
        X: (N, D) centered latent matrix
        d_out: output dimensionality (typically 2)
        
    Returns:
        (N, d_out) coordinates
        
    Complexity: O(N D²) for SVD of N×D matrix (D=16, so ~256 ops per row)
    """
    # Ensure centered
    Xc = X - X.mean(dim=0, keepdim=True)
    
    # Thin SVD on tall matrix (no N×N eigen!)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    
    # U Σ gives classical MDS coordinates
    coords = U[:, :d_out] * S[:d_out].unsqueeze(0)
    
    return coords

# Add these imports at the top if not already present
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

def affine_whitening(coords: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """
    Affine whitening with eigendecomp. Returns whitened coords and scale factor.
    """
    n = coords.shape[0]
    coords_centered = coords - coords.mean(dim=0, keepdim=True)
    
    cov = (coords_centered.T @ coords_centered) / (n - 1)
    cov = cov + eps * torch.eye(cov.shape[0], device=coords.device, dtype=coords.dtype)
    
    eigvals, eigvecs = safe_eigh(cov)  # Use your safe version
    eigvals = eigvals.clamp(min=eps)
    Sigma_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    
    coords_whitened = coords_centered @ Sigma_inv_sqrt
    
    # Scale to median 1-NN ≈ 1
    D_1nn = torch.cdist(coords_whitened, coords_whitened)
    D_1nn.fill_diagonal_(float('inf'))
    nn_dists = D_1nn.min(dim=1)[0]
    median_1nn = nn_dists.median().item()
    scale = median_1nn if median_1nn > 1e-8 else 1.0
    coords_whitened = coords_whitened / scale
    
    return coords_whitened, scale


def compute_geodesic_distances(coords: torch.Tensor, k: int = 15, device: str = 'cuda') -> torch.Tensor:
    """
    Compute graph-geodesic distances via shortest-path on kNN graph.
    """
    n = coords.shape[0]
    coords_np = coords.cpu().numpy()
    
    # Build kNN graph with Euclidean edge weights
    from sklearn.neighbors import kneighbors_graph
    knn_graph = kneighbors_graph(coords_np, n_neighbors=k, mode='distance', metric='euclidean', include_self=False)
    
    # Symmetrize
    knn_graph = (knn_graph + knn_graph.T) / 2.0
    
    # Shortest paths
    D_geo = shortest_path(knn_graph, method='auto', directed=False)
    D_geo = torch.from_numpy(D_geo).float().to(device)
    
    # Handle infinities (disconnected components)
    D_geo = torch.nan_to_num(D_geo, nan=0.0, posinf=D_geo[D_geo != float('inf')].max().item() * 2)
    
    return D_geo


def gram_from_geodesic(D_geo: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix from geodesic distances: B = -0.5 * H * D_geo^2 * H
    """
    n = D_geo.shape[0]
    H = torch.eye(n, device=D_geo.device) - torch.ones(n, n, device=D_geo.device) / n
    B = -0.5 * H @ (D_geo ** 2) @ H
    return B


def sample_ordinal_triplets_from_Z(Z: torch.Tensor, n_per_anchor: int = 10, k_nn: int = 25, margin_ratio: float = 0.2) -> torch.Tensor:
    """
    Sample ordinal triplets from Z-space neighborhoods.
    For each anchor i: j from k-NN, k from outside k-NN, enforce d_Z(i,j) < d_Z(i,k).
    Returns (T, 3) tensor of triplet indices.
    """
    n = Z.shape[0]
    if n < k_nn + 5:
        k_nn = max(3, n // 3)
    
    D_Z = torch.cdist(Z, Z)
    
    triplets = []
    for i in range(n):
        dists = D_Z[i]
        sorted_idx = torch.argsort(dists)
        
        neighbors = sorted_idx[1:k_nn+1]  # Exclude self
        non_neighbors = sorted_idx[k_nn+1:min(k_nn*4+1, n)]
        
        if len(neighbors) == 0 or len(non_neighbors) == 0:
            continue
        
        for _ in range(n_per_anchor):
            if len(neighbors) == 0 or len(non_neighbors) == 0:
                break
            j = neighbors[torch.randint(len(neighbors), (1,))].item()
            k_idx = non_neighbors[torch.randint(len(non_neighbors), (1,))].item()
            
            d_ij = dists[j].item()
            d_ik = dists[k_idx].item()
            
            if d_ij < d_ik:
                triplets.append([i, j, k_idx])
    
    if len(triplets) == 0:
        # Fallback
        return torch.zeros((0, 3), dtype=torch.long, device=Z.device)
    
    return torch.tensor(triplets, dtype=torch.long, device=Z.device)


from typing import Tuple

def create_sc_miniset_pair(
    Z_all: torch.Tensor,
    n_set: int,
    n_overlap: int,
    k_nn: int = 50,
    device: str | torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create two overlapping SC mini-sets from Z-space (indices only).
    Returns: indices_A, indices_B, shared_indices_in_A, shared_indices_in_B
    """
    # *** CRITICAL: keep everything on the same device as Z_all ***
    dev = Z_all.device if device is None else (torch.device(device) if isinstance(device, str) else device)

    n_total = Z_all.shape[0]

    # A: pick a seed and take n_set nearest
    seed_i = torch.randint(n_total, (1,), device=dev).item()
    D_from_seed = torch.cdist(Z_all[seed_i:seed_i+1], Z_all)[0]        # (n_total,) on dev
    indices_A   = torch.argsort(D_from_seed)[:n_set]                   # (n_set,) on dev

    # overlap positions inside set-A (positions 0..n_set-1)
    n_overlap = min(n_overlap, n_set - 5)
    overlap_positions = torch.randperm(n_set, device=dev)[:n_overlap]  # (n_overlap,) on dev
    shared_global     = indices_A[overlap_positions]                   # (n_overlap,) on dev

    # pick a nearby seed for set-B
    nearby = torch.argsort(D_from_seed)[1:k_nn+1]
    seed_j = nearby[torch.randint(len(nearby), (1,), device=dev)].item()

    D_from_seed_j = torch.cdist(Z_all[seed_j:seed_j+1], Z_all)[0]
    candidates    = torch.argsort(D_from_seed_j)                        # (n_total,) on dev

    # remove A \ overlap from candidates for B’s "new" cells
    all_pos_A   = torch.arange(n_set, device=dev)
    non_sharedA_pos = all_pos_A[~torch.isin(all_pos_A, overlap_positions)]
    non_shared_A    = indices_A[non_sharedA_pos]
    mask            = ~torch.isin(candidates, non_shared_A)
    candidates      = candidates[mask]

    n_new    = n_set - n_overlap
    new_B    = candidates[:n_new]
    indices_B = torch.cat([shared_global, new_B])[:n_set]               # (n_set,) on dev

    # positions of the shared elements inside A and B (vectorized, device-safe)
    shared_in_A = overlap_positions                                     # already positions in A
    pos_map_B   = torch.full((n_total,), -1, device=dev, dtype=torch.long)
    pos_map_B[indices_B] = torch.arange(indices_B.numel(), device=dev, dtype=torch.long)
    shared_in_B = pos_map_B[shared_global]

    return indices_A, indices_B, shared_in_A, shared_in_B


import torch
import torch.nn.functional as F
from typing import Dict

@torch.no_grad()
def build_sc_knn_cache(
    Z_sc: torch.Tensor,           # [N_sc, D], on CPU or GPU
    k_pos: int = 25,
    block_q: int = 2048,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Precompute per-cell kNN (positives) once. Tiled matmul -> topk.
    Returns dict with:
      - 'pos_idx': int32 [N_sc, k_pos] (global indices of nearest neighbors)
    """
    assert Z_sc.dim() == 2
    N, D = Z_sc.shape
    dev = torch.device(device)

    Z = Z_sc.to(dev, dtype=torch.float16, non_blocking=True)
    # Z = F.normalize(Z, dim=1)  # cosine kNN; if you prefer L2, normalize or switch to -L2
    Z_norm_sq = (Z * Z).sum(dim=1, keepdim=True)  # [N, 1], precompute ||z_i||^2

    pos_idx_cpu = torch.empty((N, k_pos), dtype=torch.int32)

    for start in range(0, N, block_q):
        end = min(start + block_q, N)
        bq = end - start
        Q = Z[start:end]                        # [bq, D]
        # sims = Q @ Z.t()                        # [bq, N], FP16 GEMM
        Q_norm_sq = (Q * Q).sum(dim=1, keepdim=True)  # [bq, 1]
        dist_sq = Q_norm_sq + Z_norm_sq.t() - 2 * (Q @ Z.t())  # [bq, N], squared L2

        # mask self-sim for in-block rows (so we don't pick self)
        row = torch.arange(bq, device=dev)
        col = start + row
        # sims[row, col] = -float("inf")

        # top-k neighbors (positives)
        # topk = torch.topk(sims, k=k_pos, dim=1, largest=True, sorted=True).indices
        topk = torch.topk(dist_sq, k=k_pos, dim=1, largest=False, sorted=True).indices
        pos_idx_cpu[start:end] = topk.to("cpu", dtype=torch.int32)

        # del Q, sims, topk
        del Q, Q_norm_sq, dist_sq, topk
        torch.cuda.empty_cache()

    return {"pos_idx": pos_idx_cpu, "k_pos": torch.tensor(k_pos, dtype=torch.int32)}

@torch.no_grad()
def build_triplets_from_cache_for_set(
    set_global_idx: torch.Tensor,
    pos_idx_cpu: torch.Tensor,
    n_per_anchor: int = 10,
    triplet_cap: int = 20000
) -> torch.Tensor:
    """
    Assemble triplets (a, p, n) for a single set using precomputed global kNN.
    Returns LongTensor [T, 3] with LOCAL indices in the set.
    Negatives sampled from within set but not in positive list or self.
    """
    if set_global_idx.is_cuda:
        set_global_idx_cpu = set_global_idx.detach().to("cpu")
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

    triplets_cpu = []

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
        triplets_cpu.append(torch.stack([a_col, pl.long(), neg_sel.long()], dim=1))

    if len(triplets_cpu) == 0:
        return torch.empty((0, 3), dtype=torch.long)

    triplets = torch.cat(triplets_cpu, dim=0)
    if triplets.size(0) > triplet_cap:
        idx = torch.randperm(triplets.size(0))[:triplet_cap]
        triplets = triplets[idx]
    return triplets


def canonicalize_center_only(V, mask, eps=1e-8):
    """
    Center V per set over valid points, but DO NOT rescale to unit RMS.
    This keeps absolute scale for geometry-sensitive losses.
    """
    B, N, D = V.shape
    mask_expanded = mask.unsqueeze(-1)          # (B,N,1)
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)

    mean = (V * mask_expanded).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
    V_centered = (V - mean) * mask_expanded

    # Keep a dummy scale tensor for API symmetry if needed
    scale = torch.ones(B, 1, 1, device=V.device, dtype=V.dtype)
    return V_centered, mean, scale


def sliced_wasserstein_sc(
    V: torch.Tensor,             # (B2, N, D) batched embeddings (A and B interleaved)
    mask: torch.Tensor,          # (B2, N) boolean mask
    pair_idxA: torch.Tensor,     # (P=B, ) indices into dim0 of V for A-sets: 0,2,4...
    pair_idxB: torch.Tensor,     # (P=B, ) indices into dim0 of V for B-sets: 1,3,5...
    *,
    K_proj: int = 64,            # number of random projections
    N_cap: int = 512,            # cap elements per set for speed
    eps: float = 1e-8,
    use_canon: bool = True,
) -> torch.Tensor:
    
    """
    Differentiable SW2 between paired SC sets (A_p, B_p) across p in [0..B-1].
    Returns mean SW2 over pairs (scalar tensor).
    """

    device = V.device
    P = pair_idxA.numel()
    if P == 0:
        return V.sum() * 0.0  # scalar zero on same device/dtype

    # # optional: canonicalize each set (center + unit RMS) to remove scale
    # if use_canon:
    #     # canonicalize per set using the same function you use elsewhere
    #     from utils_et import canonicalize as _canon
    #     Vc, _, _ = _canon(V.float(), mask)  # do math in fp32
    # else:
    #     Vc = V.float()

    # optional: canonicalize each set (center + unit RMS) to remove scale
    if use_canon:
        # canonicalize per set using the same function you use elsewhere
        # from utils_et import canonicalize_unit_rms as _canon
        from utils_et import canonicalize_center_only as _canon
        Vc, _, _ = _canon(V.float(), mask)  # do math in fp32
    else:
        Vc = V.float()

    B2, N, D = Vc.shape
    # random Gaussian projections, shared across pairs for variance reduction
    # shape: (K_proj, D)
    proj = torch.randn(K_proj, D, device=device, dtype=Vc.dtype)
    proj = F.normalize(proj, dim=1)  # unit vectors

    sw_total = Vc.new_tensor(0.0)

    # loop pairs (P is batch size of pairs, usually small); proj is vectorized
    for a, b in zip(pair_idxA.tolist(), pair_idxB.tolist()):
        ma = mask[a]  # (N,)
        mb = mask[b]  # (N,)
        Va = Vc[a, ma]  # (Na, D)
        Vb = Vc[b, mb]  # (Nb, D)

        # (optional) subsample for speed if very large
        Na = Va.shape[0]
        Nb = Vb.shape[0]
        if Na == 0 or Nb == 0:
            continue
        if Na > N_cap:
            idx = torch.randperm(Na, device=device)[:N_cap]
            Va = Va[idx]
            Na = N_cap
        if Nb > N_cap:
            idx = torch.randperm(Nb, device=device)[:N_cap]
            Vb = Vb[idx]
            Nb = N_cap

        # project to 1D for all K at once: (K, Na)/(K, Nb)
        Pa = Va @ proj.t()  # (Na, K)
        Pb = Vb @ proj.t()  # (Nb, K)
        Pa = Pa.transpose(0, 1)  # (K, Na)
        Pb = Pb.transpose(0, 1)  # (K, Nb)

        # sort along the set dimension
        Pa_sorted, _ = torch.sort(Pa, dim=1)  # (K, Na)
        Pb_sorted, _ = torch.sort(Pb, dim=1)  # (K, Nb)

        # match by quantiles: take the smaller length and interpolate the longer
        n = min(Na, Nb)
        if n == 0:
            continue

        q = torch.linspace(0.0, 1.0, n, device=device, dtype=Vc.dtype)
        # index helper
        def _interp_rows(X):
            # X: (K, Nrow)
            Nrow = X.shape[1]
            if Nrow == n:
                return X
            # positions in [0, Nrow-1]
            pos = q * (Nrow - 1)
            lo = pos.floor().long()
            hi = (lo + 1).clamp_max(Nrow - 1)
            w = (pos - lo.to(pos.dtype)).unsqueeze(0)  # (1, n)
            Xlo = X.gather(1, lo.unsqueeze(0).expand(X.size(0), -1))
            Xhi = X.gather(1, hi.unsqueeze(0).expand(X.size(0), -1))
            return Xlo * (1 - w) + Xhi * w

        A_q = _interp_rows(Pa_sorted)  # (K, n)
        B_q = _interp_rows(Pb_sorted)  # (K, n)

        # SW2 over K projections: mean_k mean_i (A_q - B_q)^2
        sw_pair = (A_q - B_q).pow(2).mean()
        sw_total = sw_total + sw_pair

    # average over existing pairs
    return sw_total / max(P, 1)

def init_st_dist_bins_from_data(
    coords: np.ndarray,   # (N, 2) or (N, d)
    n_bins: int = 24,
    mode: str = "log",    # or "quantile" / "linear"
    max_quantile: float = 0.99,
):
    """
    Build distance bin edges automatically from ST coordinates.
    Returns: torch.tensor of shape (n_bins+1,) = bin edges (increasing).
    """
    from scipy.spatial.distance import pdist
    
    # coords: use a random subset if huge
    if coords.shape[0] > 2000:
        idx = np.random.choice(coords.shape[0], 2000, replace=False)
        coords = coords[idx]

    D = pdist(coords, metric="euclidean")
    D = torch.from_numpy(D.astype(np.float32))

    D_pos = D[D > 0]

    if mode == "quantile":
        qs = torch.linspace(0., max_quantile, n_bins + 1)
        edges = torch.quantile(D_pos, qs)
        edges[0] = 0.0
    elif mode == "log":
        eps = 1e-3
        d_min = torch.quantile(D_pos, 0.01).item()
        d_max = torch.quantile(D_pos, max_quantile).item()
        edges = torch.exp(
            torch.linspace(
                np.log(d_min + eps),
                np.log(d_max + eps),
                n_bins + 1
            )
        )
        edges[0] = 0.0
    else:  # "linear"
        d_max = torch.quantile(D_pos, max_quantile).item()
        edges = torch.linspace(0.0, d_max, n_bins + 1)

    return edges

def canonicalize_st_coords_per_slide(
    coords: torch.Tensor,      # (N, d), typically d=2
    slide_ids: torch.Tensor,   # (N,) long, slide index per spot
    eps: float = 1e-8,
) -> tuple:
    """
    Per-slide canonicalization for ST coordinates:
      - center each slide by its mean,
      - scale each slide by a single isotropic factor (RMS radius).

    Args:
        coords   : (N, d) raw spatial coords for all spots
        slide_ids: (N,) slide index for each spot (0..S-1)
        eps      : minimum allowed scale

    Returns:
        coords_canon : (N, d) canonicalized coordinates
        mu_per_slide : (S, d) mean per slide in original units
        scale_per_slide : (S,) RMS radius per slide in original units
    """
    assert coords.shape[0] == slide_ids.shape[0]
    device = coords.device
    N, d = coords.shape
    S = int(slide_ids.max().item()) + 1

    coords_canon = coords.clone()
    mu_per_slide = torch.zeros(S, d, device=device, dtype=coords.dtype)
    scale_per_slide = torch.zeros(S, device=device, dtype=coords.dtype)

    for s in range(S):
        mask = (slide_ids == s)
        if not mask.any():
            continue

        xs = coords[mask]                        # (n_s, d)
        mu = xs.mean(dim=0, keepdim=True)        # (1, d)
        xc = xs - mu
        # RMS radius over points
        rms = (xc.pow(2).sum(dim=1).mean().sqrt()).clamp_min(eps)

        coords_canon[mask] = xc / rms
        mu_per_slide[s] = mu.squeeze(0)
        scale_per_slide[s] = rms

    return coords_canon, mu_per_slide, scale_per_slide


# =========================================
# EDM TAIL LOSS & GENERATOR ALIGNMENT
# =========================================

def compute_edm_tail_loss(
    V_pred: torch.Tensor,      # (B, N, D_latent) predicted coordinates
    V_target: torch.Tensor,    # (B, N, D_latent) target coordinates from factor_from_gram
    mask: torch.Tensor,        # (B, N) validity mask
    tail_quantile: float = 0.80,  # pairs above this quantile are "tail"
    weight_tail: float = 2.0,     # weight for tail pairs vs others
) -> torch.Tensor:
    """
    EDM tail loss: penalize distance compression, especially for far pairs.
    
    For each sample in batch:
      1. Compute pairwise distances D_pred and D_target
      2. Identify "tail" pairs (distances > quantile in D_target)
      3. Apply weighted MSE on distances, heavier on tail
    
    Args:
        V_pred: predicted latent coordinates (centered)
        V_target: target latent coordinates (centered)
        mask: (B, N) boolean mask
        tail_quantile: quantile threshold for "far" pairs (0.8 = top 20%)
        weight_tail: extra weight for tail pairs
    
    Returns:
        scalar loss
    """
    B, N, D = V_pred.shape
    device = V_pred.device
    
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    for b in range(B):
        m = mask[b]  # (N,)
        n_valid = m.sum().item()
        if n_valid < 2:
            continue
        
        # Extract valid points
        V_p = V_pred[b, m]  # (n_valid, D)
        V_t = V_target[b, m]  # (n_valid, D)
        
        # Pairwise distances
        D_pred = torch.cdist(V_p, V_p)  # (n_valid, n_valid)
        D_target = torch.cdist(V_t, V_t)  # (n_valid, n_valid)
        
        # Get upper triangular (exclude diagonal)
        triu_mask = torch.triu(torch.ones_like(D_target, dtype=torch.bool), diagonal=1)
        d_pred_flat = D_pred[triu_mask]
        d_target_flat = D_target[triu_mask]
        
        if d_target_flat.numel() == 0:
            continue
        
        # Identify tail pairs
        threshold = torch.quantile(d_target_flat, tail_quantile)
        is_tail = d_target_flat >= threshold
        
        # Weighted MSE
        diff_sq = (d_pred_flat - d_target_flat) ** 2
        weights = torch.where(is_tail, 
                             torch.tensor(weight_tail, device=device), 
                             torch.tensor(1.0, device=device))
        loss_b = (weights * diff_sq).mean()
        total_loss = total_loss + loss_b
        valid_samples += 1
    
    if valid_samples == 0:
        return torch.tensor(0.0, device=device)
    
    return total_loss / valid_samples


# def compute_edm_tail_loss(
#     V_pred: torch.Tensor,      # (B, N, D)
#     V_target: torch.Tensor,    # (B, N, D)
#     mask: torch.Tensor,        # (B, N) bool
#     tail_quantile: float = 0.80,
#     weight_tail: float = 1.0,    # smaller than 2.0
#     clip_quantile: float = 0.995 # clip extreme outliers
# ) -> torch.Tensor:
#     """
#     Weighted MSE on pairwise distances, emphasizing far pairs in target.
#     Distances are normalized by median(target_distances) to stabilize scale.
#     """
#     B, N, D = V_pred.shape
#     device = V_pred.device

#     total_loss = V_pred.new_tensor(0.0)
#     valid_samples = 0

#     for b in range(B):
#         m = mask[b]
#         n_valid = int(m.sum())
#         if n_valid < 2:
#             continue

#         Vp = V_pred[b, m]    # (n_valid, D)
#         Vt = V_target[b, m]  # (n_valid, D)

#         Dp = torch.cdist(Vp, Vp)
#         Dt = torch.cdist(Vt, Vt)

#         triu = torch.triu(
#             torch.ones_like(Dt, dtype=torch.bool), diagonal=1
#         )
#         d_p = Dp[triu]
#         d_t = Dt[triu]

#         if d_t.numel() == 0:
#             continue

#         # robust scale and optional clipping
#         med = d_t.median()
#         med = med.clamp_min(1e-6)
#         d_t = d_t / med
#         d_p = d_p / med

#         if clip_quantile is not None:
#             q_clip = torch.quantile(d_t, clip_quantile)
#             keep = d_t <= q_clip
#             d_t = d_t[keep]
#             d_p = d_p[keep]
#             if d_t.numel() == 0:
#                 continue

#         # far tail mask
#         q_hi = torch.quantile(d_t, tail_quantile)
#         is_tail = d_t >= q_hi

#         diff_sq = (d_p - d_t) ** 2
#         w = torch.where(
#             is_tail,
#             d_p.new_tensor(weight_tail),
#             d_p.new_tensor(1.0)
#         )
#         loss_b = (w * diff_sq).mean()

#         total_loss += loss_b
#         valid_samples += 1

#     if valid_samples == 0:
#         return V_pred.new_tensor(0.0)

#     return total_loss / valid_samples


def procrustes_alignment_loss(
    V_pred: torch.Tensor,      # (B, N, D_latent) from generator
    V_target: torch.Tensor,    # (B, N, D_latent) from factor_from_gram
    mask: torch.Tensor,        # (B, N)
) -> torch.Tensor:
    """
    Generator alignment loss using Procrustes.
    
    For each sample:
      1. Center both V_pred and V_target
      2. Find optimal rotation R via SVD: C = V_pred^T @ V_target, R = U @ V^T
      3. Compute MSE between V_pred @ R and V_target
    
    This encourages generator to produce geometry isometric to the target Gram matrix.
    
    Args:
        V_pred: generator output (should already be centered per-miniset)
        V_target: target from factor_from_gram (already centered)
        mask: validity mask
    
    Returns:
        scalar loss
    """
    B, N, D = V_pred.shape
    device = V_pred.device
    
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    for b in range(B):
        m = mask[b]  # (N,)
        n_valid = m.sum().item()
        if n_valid < D:  # need at least D points for meaningful Procrustes
            continue
        
        # Extract valid points
        Vp = V_pred[b, m]  # (n_valid, D)
        Vt = V_target[b, m]  # (n_valid, D)
        
        # Center (should already be centered, but ensure it)
        Vp_c = Vp - Vp.mean(dim=0, keepdim=True)
        Vt_c = Vt - Vt.mean(dim=0, keepdim=True)
        
        # Compute cross-covariance
        C = Vp_c.t() @ Vt_c  # (D, D)
        
        # SVD
        try:
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            R = U @ Vh  # Optimal rotation
            
            # Align V_pred to V_target
            Vp_aligned = Vp_c @ R
            
            # MSE loss
            loss_b = ((Vp_aligned - Vt_c) ** 2).mean()
            total_loss = total_loss + loss_b
            valid_samples += 1
        except:
            # SVD failed (rare), skip this sample
            continue
    
    if valid_samples == 0:
        return torch.tensor(0.0, device=device)
    
    return total_loss / valid_samples


# ==============================================================================
# FIXED-ADJACENCY GRAPH CONSTRUCTION FOR HEAT LOSS
# ==============================================================================

def precompute_st_graph_fixed(
    y_hat: torch.Tensor,    # (N_total, d) - full ST coordinates for one slide
    k: int = 10,            # kNN parameter
    sigma: Optional[float] = None  # Gaussian kernel bandwidth (None = auto from median)
) -> Dict[str, torch.Tensor]:
    """
    Precompute fixed kNN graph structure on full ST coordinates.
    
    Returns dictionary with:
        - edges_full: (2, E) edge indices
        - w_full_gt: (E,) Gaussian weights based on GT distances
        - sigma: bandwidth used
        - N_total: total number of nodes
    """
    from sklearn.neighbors import NearestNeighbors
    
    N_total = y_hat.shape[0]
    device = y_hat.device
    
    # Use sklearn for efficient kNN (on CPU)
    y_np = y_hat.cpu().numpy() if torch.is_tensor(y_hat) else y_hat
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(y_np)
    dist, idx = nbrs.kneighbors(y_np)  # (N, k+1)
    
    # Skip self-neighbor at position 0
    idx = idx[:, 1:]      # (N, k)
    dist = dist[:, 1:]    # (N, k)
    
    # Build edge list
    rows = np.repeat(np.arange(N_total), k)
    cols = idx.reshape(-1)
    d_ij = dist.reshape(-1)
    
    edges_full = torch.tensor(
        np.vstack([rows, cols]),
        dtype=torch.long,
        device=device
    )  # (2, E)
    
    # Compute Gaussian weights
    if sigma is None:
        sigma = float(np.median(d_ij))
    
    d_ij_tensor = torch.tensor(d_ij, dtype=torch.float32, device=device)
    w_full_gt = torch.exp(-(d_ij_tensor**2) / (2 * sigma**2))
    
    return {
        'edges_full': edges_full,
        'w_full_gt': w_full_gt,
        'sigma': sigma,
        'N_total': N_total
    }


def extract_miniset_fixed_graph(
    indices: torch.Tensor,      # (n_valid,) global indices in this mini-set
    graph_info: Dict[str, torch.Tensor]  # from precompute_st_graph_fixed
) -> Dict[str, torch.Tensor]:
    """
    Extract the induced subgraph for a mini-set with fixed adjacency.
    
    Returns:
        - edges_local: (2, E_sub) edges in local indexing [0..n_valid-1]
        - w_gt: (E_sub,) GT weights for these edges
        - sigma: bandwidth for computing predicted weights
    """
    edges_full = graph_info['edges_full']
    w_full_gt = graph_info['w_full_gt']
    N_total = graph_info['N_total']
    device = edges_full.device
    
    # Create mask for nodes in this mini-set
    node_mask = torch.zeros(N_total, dtype=torch.bool, device=device)
    node_mask[indices] = True
    
    # Select edges where both endpoints are in mini-set
    i = edges_full[0]
    j = edges_full[1]
    in_set = node_mask[i] & node_mask[j]
    
    edges_sub = edges_full[:, in_set]  # (2, E_sub)
    w_sub_gt = w_full_gt[in_set]       # (E_sub,)
    
    # Map global indices to local indices [0..n_valid-1]
    local_id = torch.full((N_total,), -1, dtype=torch.long, device=device)
    local_id[indices] = torch.arange(len(indices), device=device)
    
    edges_local = local_id[edges_sub]  # (2, E_sub) in local indexing
    
    return {
        'edges_local': edges_local,
        'w_gt': w_sub_gt,
        'sigma': graph_info['sigma']
    }


def build_laplacian_from_edges(
    n_nodes: int,
    edges: torch.Tensor,      # (2, E) edge indices
    weights: torch.Tensor,    # (E,) edge weights
    laplacian_type: str = 'sym'
) -> torch.Tensor:
    """
    Build graph Laplacian from edge list.
    
    Args:
        n_nodes: number of nodes
        edges: (2, E) edge indices
        weights: (E,) edge weights
        laplacian_type: 'sym' (normalized) or 'unnorm' (combinatorial)
    
    Returns:
        L: (n_nodes, n_nodes) Laplacian matrix
    """
    device = edges.device
    
    # Make symmetric (add reverse edges)
    edges_sym = torch.cat([edges, edges.flip(0)], dim=1)
    weights_sym = torch.cat([weights, weights], dim=0)
    
    # Build adjacency matrix (sparse -> dense)
    adj = torch.sparse_coo_tensor(
        edges_sym, weights_sym, (n_nodes, n_nodes)
    ).to_dense()
    
    # Degree matrix
    deg = adj.sum(dim=1)
    
    if laplacian_type == 'sym':
        # Symmetric normalized: L = I - D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(n_nodes, device=device) - deg_inv_sqrt @ adj @ deg_inv_sqrt
    else:
        # Combinatorial: L = D - A
        L = torch.diag(deg) - adj
    
    return L


class IntrinsicDimensionLoss(nn.Module):
    """
    Levina-Bickel MLE intrinsic dimension regularizer.
    
    Estimates local intrinsic dimension from pairwise distances (EDM)
    and matches it to ground truth.
    
    Args:
        k_neighbors: number of nearest neighbors for local estimation (default: 20)
        target_dim: target intrinsic dimension for SC (default: 2.0)
    """
    
    def __init__(self, k_neighbors: int = 20, target_dim: float = 2.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.target_dim = target_dim
    
    def forward(
        self,
        V_pred: torch.Tensor,
        V_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_target: bool = True
    ) -> torch.Tensor:
        """
        Compute intrinsic dimension loss.
        
        Args:
            V_pred: (B, N, D) predicted coordinates
            V_target: (B, N, D) target coordinates (for ST) or None (for SC)
            mask: (B, N) boolean mask
            use_target: if True, match to V_target's dimension; else use self.target_dim
            
        Returns:
            loss: scalar dimension mismatch loss
        """
        B, N, D = V_pred.shape
        device = V_pred.device
        
        # Ensure float32 for numerical stability
        V_pred = V_pred.float()
        if V_target is not None:
            V_target = V_target.float()
        
        # Compute intrinsic dimension for predictions
        m_pred = self._estimate_dimension_batch(V_pred, mask)
        
        if use_target and V_target is not None:
            # Match to ground truth dimension
            m_true = self._estimate_dimension_batch(V_target, mask)
            loss = (m_pred - m_true).pow(2).mean()
        else:
            # Match to target dimension (for SC)
            m_target = torch.tensor(self.target_dim, device=device, dtype=torch.float32)
            loss = (m_pred - m_target).pow(2).mean()
        
        return loss
    
    def _estimate_dimension_batch(
        self,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate intrinsic dimension for each sample in batch.
        
        Args:
            V: (B, N, D) coordinates
            mask: (B, N) boolean mask
            
        Returns:
            m: (B,) estimated intrinsic dimensions
        """
        B, N, D = V.shape
        device = V.device
        k = min(self.k_neighbors, N - 1)
        
        if k < 2:
            # Not enough neighbors, return default
            return torch.full((B,), 2.0, device=device, dtype=torch.float32)
        
        # Compute pairwise distances
        D_dist = torch.cdist(V, V)  # (B, N, N)
        
        # For each point, get k nearest neighbors
        # topk excludes self (distance=0) by taking k+1 and slicing [1:]
        distances, indices = torch.topk(D_dist, k=k+1, dim=-1, largest=False, sorted=True)
        r_k = distances[:, :, 1:]  # (B, N, k) - exclude self
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, k)
            r_k = r_k.where(mask_expanded, torch.full_like(r_k, 1.0))  # Use 1.0 instead of inf
        
        # Levina-Bickel MLE estimator: m_i = [1/(k-1) * sum_{j=1}^{k-1} log(r_k / r_j)]^{-1}
        r_k_outer = r_k[:, :, -1:]  # (B, N, 1) - largest distance
        r_j = r_k[:, :, :-1]  # (B, N, k-1) - all but largest
        
        # CRITICAL FIX: Avoid log(0), log(negative), and division by zero
        # Add larger epsilon to prevent near-zero ratios
        eps = 1e-6
        ratio = (r_k_outer.clamp(min=eps)) / (r_j.clamp(min=eps))
        
        # Clamp ratio to reasonable range before log
        ratio = ratio.clamp(min=1.0 + eps, max=100.0)  # Ensure ratio > 1
        
        log_ratio = torch.log(ratio)
        
        # Average log ratios per point
        mean_log_ratio = log_ratio.mean(dim=-1)  # (B, N)
        
        # CRITICAL FIX: Clamp mean_log_ratio away from zero before inversion
        mean_log_ratio = mean_log_ratio.clamp(min=1e-3)
        
        # Inverse gives dimension estimate per point
        m_i = 1.0 / mean_log_ratio
        
        # Clamp to reasonable range [0.5, 10] to avoid numerical issues
        m_i = m_i.clamp(min=0.5, max=10.0)
        
        # CRITICAL FIX: Check for NaN and replace with default
        m_i = torch.where(torch.isnan(m_i), torch.tensor(2.0, device=device), m_i)
        
        # Apply mask and average over points
        if mask is not None:
            valid_counts = mask.float().sum(dim=1).clamp(min=1.0)
            m_patch = (m_i * mask.float()).sum(dim=1) / valid_counts
        else:
            m_patch = m_i.mean(dim=1)
        
        # Final NaN check
        m_patch = torch.where(torch.isnan(m_patch), torch.tensor(2.0, device=device), m_patch)
        
        return m_patch  # (B,)


class TriangleAreaLoss(nn.Module):
    """
    Local triangle area regularizer using Heron's formula.
    
    Penalizes degenerate (1-D) manifolds by enforcing non-zero triangle areas.
    FULLY DIFFERENTIABLE - all operations keep gradient graph intact.
    
    Args:
        num_triangles_per_sample: number of triangles to sample per point cloud
        knn_k: use neighbors within knn_k for triangle sampling
    """
    
    def __init__(self, num_triangles_per_sample: int = 500, knn_k: int = 12):
        super().__init__()
        self.num_triangles_per_sample = num_triangles_per_sample
        self.knn_k = knn_k
    
    def forward(
        self,
        V_pred: torch.Tensor,
        V_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_target: bool = True,
        hinge_mode: bool = False,
        epsilon: float = 1e-4
    ) -> torch.Tensor:
        """
        Compute triangle area loss.
        
        Args:
            V_pred: (B, N, D) predicted coordinates
            V_target: (B, N, D) target coordinates (for ST) or None
            mask: (B, N) boolean mask
            use_target: if True, match to V_target areas; else use hinge
            hinge_mode: if True, enforce A >= epsilon; else match to target
            epsilon: minimum triangle area for hinge mode
            
        Returns:
            loss: scalar area loss
        """
        B, N, D = V_pred.shape
        device = V_pred.device
        
        V_pred = V_pred.float()
        if V_target is not None:
            V_target = V_target.float()
        
        # Compute average triangle area for predictions
        A_pred = self._compute_avg_triangle_area(V_pred, mask)
        
        if use_target and V_target is not None:
            # Match to ground truth areas
            A_true = self._compute_avg_triangle_area(V_target, mask)
            loss = (A_pred - A_true).pow(2).mean()
        elif hinge_mode:
            # Hinge: enforce A_pred >= epsilon
            violation = torch.clamp(epsilon - A_pred, min=0.0)
            loss = violation.pow(2).mean()

            # DEBUG: Log actual values to diagnose zero loss
            if torch.rand(1).item() < 0.01:  # Log 1% of the time
                print(f"[TriangleLoss DEBUG] epsilon={epsilon:.6f}, "
                      f"A_pred mean={A_pred.mean().item():.6f}, "
                      f"A_pred min={A_pred.min().item():.6f}, "
                      f"A_pred max={A_pred.max().item():.6f}, "
                      f"violation mean={violation.mean().item():.6f}, "
                      f"loss={loss.item():.6f}")
        else:
            # Default: match to a reasonable target area
            A_target = torch.tensor(0.1, device=device, dtype=torch.float32)
            loss = (A_pred - A_target).pow(2).mean()
        
        return loss
    
    def _compute_avg_triangle_area(
        self,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute average triangle area for each sample.
        
        Args:
            V: (B, N, D) coordinates
            mask: (B, N) boolean mask
            
        Returns:
            A_avg: (B,) average area per sample
        """
        B, N, D = V.shape
        device = V.device
        
        # For each sample, sample triangles from knn neighborhoods
        areas_batch = []
        
        for b in range(B):
            V_b = V[b]  # (N, D)
            if mask is not None:
                valid_mask = mask[b]
                n_valid = valid_mask.sum().item()
                if n_valid < 3:
                    areas_batch.append(torch.tensor(0.0, device=device))
                    continue
                V_b = V_b[valid_mask]  # (n_valid, D)
            else:
                n_valid = N
            
            # Sample triangles from knn neighborhoods (FULLY DIFFERENTIABLE)
            areas = self._sample_local_triangles(V_b, n_valid)
            
            if areas.numel() == 0:
                areas_batch.append(torch.tensor(0.0, device=device))
            else:
                areas_batch.append(areas.mean())
        
        return torch.stack(areas_batch)  # (B,)
    
    def _sample_local_triangles(
        self,
        V: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Sample triangles within local knn neighborhoods.
        FULLY DIFFERENTIABLE - no .item() calls, all tensor operations.
        
        Args:
            V: (n, D) coordinates
            n: number of valid points
            
        Returns:
            areas_normalized: (T,) normalized areas of sampled triangles
        """
        device = V.device
        k = min(self.knn_k, n - 1)
        
        if k < 2 or n < 3:
            return torch.zeros(0, device=device, dtype=torch.float32)
        
        # Compute pairwise distances (DIFFERENTIABLE)
        D_dist = torch.cdist(V, V)  # (n, n)
        
        # Get knn for each point (DIFFERENTIABLE - indices don't need gradients)
        _, knn_idx = torch.topk(D_dist, k=k+1, dim=-1, largest=False, sorted=True)
        knn_idx = knn_idx[:, 1:]  # exclude self, shape (n, k)
        
        # Sample triangles
        num_to_sample = min(self.num_triangles_per_sample, n * (k * (k-1)) // 2)
        
        if num_to_sample == 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        
        # Sample anchor points (i) - shape (T,)
        i_idx = torch.randint(0, n, (num_to_sample,), device=device)
        
        # Sample 2 neighbors per anchor - shape (T, 2)
        nbr_perm = torch.randint(0, k, (num_to_sample, 2), device=device)
        
        # Get neighbor indices j and k_pt - shape (T,)
        j_idx = knn_idx[i_idx, nbr_perm[:, 0]]
        k_idx = knn_idx[i_idx, nbr_perm[:, 1]]
        
        # CRITICAL: Compute edge lengths as TENSORS (not .item())
        a = D_dist[i_idx, j_idx]  # (T,) - distance i to j
        b = D_dist[i_idx, k_idx]  # (T,) - distance i to k
        c = D_dist[j_idx, k_idx]  # (T,) - distance j to k
        
        # Heron's formula in TENSOR FORM (DIFFERENTIABLE)
        s = 0.5 * (a + b + c)  # semi-perimeter (T,)
        area_sq = s * (s - a) * (s - b) * (s - c)  # (T,)
        area_sq = area_sq.clamp(min=0.0)  # prevent negative from numerical errors
        A = torch.sqrt(area_sq + 1e-12)  # (T,)
        
        # Normalize by scale to make dimensionless (DIFFERENTIABLE)
        # avg_edge = (a + b + c) / 3.0  # (T,)
        # scale = (avg_edge ** 2).clamp(min=1e-8)  # (T,)
        # A_normalized = A / scale  # (T,)


        # Normalize by scale to make dimensionless (DIFFERENTIABLE)
        avg_edge = (a + b + c) / 3.0  # (T,)
        scale = (avg_edge ** 2).clamp(min=1e-8)  # (T,)
        A_normalized = A / scale  # (T,)
 
        # DEBUG: Occasionally log raw vs normalized areas
        # if torch.rand(1).item() < 0.005:  # 0.5% of calls
        #     print(f"[Triangle DEBUG] Raw area: mean={A.mean().item():.6f}, "
        #           f"Normalized: mean={A_normalized.mean().item():.6f}, "
        #           f"avg_edge mean={avg_edge.mean().item():.4f}")
        
        return A_normalized  # (T,) - TENSOR with gradient graph intact


class RadialHistogramLoss(nn.Module):
    """
    Occupancy / radial histogram loss for ST patches.
    
    Canonicalizes coordinates (center, PCA, scale) and matches radial distribution.
    Directly attacks hollow centers by comparing radial occupancy.
    Uses DIFFERENTIABLE soft binning instead of torch.histogram.
    
    Args:
        num_bins: number of radial bins (default: 20)
        temperature: softmax temperature for soft binning (default: 0.1)
    """
    
    def __init__(self, num_bins: int = 20, temperature: float = 0.1):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
    
    def forward(
        self,
        V_pred: torch.Tensor,
        V_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute radial histogram loss.
        
        Args:
            V_pred: (B, N, D) predicted coordinates
            V_target: (B, N, D) target coordinates
            mask: (B, N) boolean mask
            
        Returns:
            loss: scalar histogram matching loss
        """
        B, N, D = V_pred.shape
        device = V_pred.device
        
        V_pred = V_pred.float()
        V_target = V_target.float()
        
        # Canonicalize both pred and target, compute histograms
        hist_pred = self._compute_radial_histogram_batch(V_pred, mask)
        hist_target = self._compute_radial_histogram_batch(V_target, mask)
        
        # L2 loss between histograms
        loss = (hist_pred - hist_target).pow(2).sum(dim=-1).mean()
        
        return loss
    
    def _compute_radial_histogram_batch(
        self,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Canonicalize coordinates and compute radial histogram using soft binning.
        
        Args:
            V: (B, N, D) coordinates
            mask: (B, N) boolean mask
            
        Returns:
            hist: (B, num_bins) normalized radial histograms
        """
        B, N, D = V.shape
        device = V.device
        
        hists = []
        
        for b in range(B):
            V_b = V[b]  # (N, D)
            
            if mask is not None:
                valid_mask = mask[b]
                n_valid = valid_mask.sum().item()
                if n_valid < 3:
                    # Not enough points, return uniform
                    hists.append(torch.ones(self.num_bins, device=device) / self.num_bins)
                    continue
                V_b = V_b[valid_mask]  # (n_valid, D)
            else:
                n_valid = N
            
            # Canonicalize
            V_canon = self._canonicalize(V_b)
            
            # Compute radii
            radii = torch.norm(V_canon, dim=-1)  # (n_valid,)
            
            # Normalize radii to [0, 1]
            max_r = radii.max()
            if max_r > 1e-8:
                radii_norm = radii / max_r
            else:
                radii_norm = radii
            
            # DIFFERENTIABLE soft binning
            hist = self._soft_histogram(radii_norm, self.num_bins, self.temperature)
            
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-8)
            hists.append(hist)
        
        return torch.stack(hists)  # (B, num_bins)
    
    @staticmethod
    def _soft_histogram(
        values: torch.Tensor,
        num_bins: int,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Differentiable soft histogram using Gaussian kernels.
        
        Args:
            values: (N,) values in range [0, 1]
            num_bins: number of bins
            temperature: kernel width (smaller = sharper bins)
            
        Returns:
            hist: (num_bins,) soft histogram counts
        """
        device = values.device
        n = values.shape[0]
        
        # Bin centers uniformly spaced in [0, 1]
        bin_centers = torch.linspace(0.0, 1.0, num_bins, device=device)
        
        # Compute distance from each value to each bin center
        # values: (N,) -> (N, 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        dist = (values.unsqueeze(1) - bin_centers.unsqueeze(0)).abs()  # (N, num_bins)
        
        # Soft assignment using Gaussian kernel
        # weight[i,j] = how much value i contributes to bin j
        weights = torch.exp(-dist.pow(2) / (2 * temperature**2))  # (N, num_bins)
        
        # Sum weights across values to get histogram
        hist = weights.sum(dim=0)  # (num_bins,)
        
        return hist
    
    @staticmethod
    def _canonicalize(V: torch.Tensor) -> torch.Tensor:
        """
        Canonicalize coordinates: center, PCA rotate, scale to unit max radius.
        
        Args:
            V: (n, D) coordinates
            
        Returns:
            V_canon: (n, D) canonicalized coordinates
        """
        n, D = V.shape
        device = V.device
        
        # Center
        V_centered = V - V.mean(dim=0, keepdim=True)
        
        # PCA rotation (optional - can skip if causing issues)
        if D > 1 and n > D:
            try:
                # Use SVD for differentiable PCA
                U, S, Vh = torch.svd(V_centered)
                V_rotated = V_centered @ Vh
            except:
                # Fallback if SVD fails
                V_rotated = V_centered
        else:
            V_rotated = V_centered
        
        # Scale to unit max radius
        radii = torch.norm(V_rotated, dim=-1)
        max_r = radii.max()
        if max_r > 1e-8:
            V_canon = V_rotated / max_r
        else:
            V_canon = V_rotated
        
        return V_canon


class KNNSoftmaxLoss(nn.Module):
    """
    NCA/SNE-style neighbor preservation loss.
    
    For each anchor point i, defines target distribution P_ij over neighbors
    (uniform over k-NN, zero elsewhere), and predicted distribution Q_ij 
    from softmax over predicted distances.
    
    Loss = sum_i sum_j P_ij * (-log Q_ij)  (cross-entropy)
    
    Args:
        tau: Temperature for softmax (smaller = sharper)
        k: Number of nearest neighbors to preserve
    """
    
    def __init__(self, tau: float = 1.0, k: int = 10):
        super().__init__()
        self.tau = tau
        self.k = k
    
    def forward(
        self,
        D_pred: torch.Tensor,
        knn_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute NCA-style neighbor softmax loss.
        
        Args:
            D_pred: (B, N, N) or (N, N) predicted distance matrix
            knn_indices: (B, N, k) or (N, k) ground truth k-NN indices (local indexing)
            mask: (B, N) or (N,) optional validity mask
            tau: Optional temperature override
            k: Optional k override
            
        Returns:
            loss: Scalar cross-entropy loss
        """
        if tau is None:
            tau = self.tau
        if k is None:
            k = self.k
            
        is_batched = D_pred.dim() == 3
        
        if not is_batched:
            # Add batch dimension
            D_pred = D_pred.unsqueeze(0)          # (1, N, N)
            knn_indices = knn_indices.unsqueeze(0)  # (1, N, k)
            if mask is not None:
                mask = mask.unsqueeze(0)  # (1, N)
        
        B, N, _ = D_pred.shape
        device = D_pred.device
        
        # Create target distribution P_ij: uniform over k-NN, zero elsewhere
        # Shape: (B, N, N)
        P = torch.zeros(B, N, N, device=device, dtype=D_pred.dtype)
        
        # For each batch and each anchor i, set P[b,i,j]=1/k for j in knn_indices[b,i]
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
        anchor_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
        
        # Handle invalid kNN indices (padded with -1)
        valid_knn = knn_indices >= 0  # (B, N, k)
        knn_indices_clamped = knn_indices.clamp(min=0)  # Replace -1 with 0 temporarily
        
        P[batch_idx, anchor_idx, knn_indices_clamped] = 1.0 / k
        P = P * valid_knn.unsqueeze(1).float()  # Zero out invalid entries
        
        # Normalize rows to handle variable k (in case some points have <k neighbors)
        row_sums = P.sum(dim=2, keepdim=True).clamp(min=1e-12)
        P = P / row_sums
        
        # Compute predicted distribution Q_ij from softmax over distances
        # Q_ij = exp(-d_ij^2 / tau) / sum_l exp(-d_il^2 / tau)
        
        # Negative squared distances (for numerical stability)
        neg_sq_dist = -(D_pred ** 2) / tau  # (B, N, N)
        
        # Mask out self-connections (diagonal)
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        neg_sq_dist = neg_sq_dist.masked_fill(eye, -float('inf'))
        
        # Apply validity mask if provided
        if mask is not None:
            # Mask both anchors and neighbors
            anchor_mask = mask.unsqueeze(2)  # (B, N, 1)
            neighbor_mask = mask.unsqueeze(1)  # (B, 1, N)
            valid_pairs = anchor_mask & neighbor_mask  # (B, N, N)
            neg_sq_dist = neg_sq_dist.masked_fill(~valid_pairs, -float('inf'))
        
        # Softmax to get Q
        Q = F.softmax(neg_sq_dist, dim=2)  # (B, N, N)
        
        # Cross-entropy: -sum_j P_ij * log(Q_ij)
        # Add epsilon to avoid log(0)
        loss_per_anchor = -(P * torch.log(Q + 1e-12)).sum(dim=2)  # (B, N)
        
        # Average over valid anchors
        if mask is not None:
            loss = (loss_per_anchor * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss_per_anchor.mean()
        
        return loss