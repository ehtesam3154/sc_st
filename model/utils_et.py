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

# def safe_eigh(A: torch.Tensor, return_vecs: bool = True, cpu: bool = True):
#     """
#     Robust Hermitian eigensolver:
#     - symmetrize
#     - adaptive jitter
#     - float64 on CPU
#     - numpy fallback
#     """

#     A = _symmetrize(A)
#     n = A.shape[0]
#     dev, dtype = A.device, A.dtype

#     #adaptive jitter scaled to matrix magnitude
#     scale = torch.nan_to_num(A.abs().max(), nan=1.0, posinf=1.0, neginf=1.0)
#     jitter = (1e-8 + 1e-6 * float(scale)) * torch.eye(n, device=dev, dtype=dtype)
#     A = A + jitter 

#     A64 = A.double().cpu() if cpu else A.double()
#     try:
#         if return_vecs:
#             w, V = torch.linalg.eigh(A64)
#         else:
#             w = torch.linalg.eigvalsh(A64)
#             V = None
#     except Exception:
#         import numpy as np
#         a_np = A64.numpy()
#         if return_vecs:
#             w_np, V_np = np.linalg.eigh(a_np)
#             w = torch.from_numpy(w_np)
#             V = torch.from_numpy(V_np)
#         else:
#             w_np = np.linalg.eigvalsh(a_np)
#             w = torch.from_numpy(w_np)
#             V = None

#     if cpu:
#         w = w.to(device=dev, dtype=dtype)
#         if V is not None:
#             V = V.to(device=dev, dtype=dtype)
#     return (w, V) if return_vecs else (w, None)

# def compute_distance_hist(D: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
#     """
#     Compute normalized histogram of pairwise distances (upper triangle only).
#     FIXED: moves to CPU for histogram computation.
#     """
#     n = D.shape[0]
#     triu_i, triu_j = torch.triu_indices(n, n, offset=1)
#     d_vec = D[triu_i, triu_j]
    
#     # Move to CPU for histogram (not supported on CUDA)
#     device = d_vec.device
#     d_vec_cpu = d_vec.cpu()
#     bins_cpu = bins.cpu()
    
#     # Compute histogram on CPU
#     hist, _ = torch.histogram(d_vec_cpu, bins=bins_cpu)
#     hist = hist.float() / hist.sum().clamp_min(1)
    
#     # Move back to original device
#     return hist.to(device)

# def compute_distance_hist(D, bins):
#     """Compute distance histogram entirely on GPU"""
#     n = D.shape[-1]
#     iu, ju = torch.triu_indices(n, n, 1, device=D.device)
#     d = D[iu, ju]
#     hist, _ = torch.histogram(d, bins=bins)
#     return hist.float() / hist.sum().clamp_min(1.0)

import torch

# def compute_distance_hist(D: torch.Tensor, bins: torch.Tensor):
#     """
#     GPU-only distance histogram using bucketize + bincount.
#     Matches torch.histogram with explicit bin edges:
#       - left-inclusive, right-inclusive for last bin
#       - under/overflow values are ignored (not clamped into edge bins)

#     Args:
#         D: (N,N) or (B,N,N) distance matrix (CUDA ok)
#         bins: (nb+1,) sorted bin edges tensor on same device as D

#     Returns:
#         (nb,) for single, or (B, nb) for batched
#     """
#     is_batched = (D.dim() == 3)
#     if not is_batched:
#         n = D.shape[0]
#         iu, ju = torch.triu_indices(n, n, 1, device=D.device)
#         d = D[iu, ju]  # (M,)
#     else:
#         B, n, _ = D.shape
#         iu, ju = torch.triu_indices(n, n, 1, device=D.device)
#         d = D[:, iu, ju]  # (B, M)

#     nb = bins.numel() - 1

#     # Keep only in-range values (under/overflow dropped)
#     if not is_batched:
#         mask = (d >= bins[0]) & (d <= bins[-1])
#         d = d[mask]
#     else:
#         mask = (d >= bins[0]) & (d <= bins[-1])
#         # If you expect many out-of-range, compact per batch:
#         # gather valid per row
#         # But for simplicity, we keep mask and leave out-of-range with a dummy id
#         # We'll set those to -1 and ignore them via masking below.
    
#     # Compute bin ids (left-inclusive). For rightmost edge, force last bin.
#     # bucketize returns insertion index; right=False => first idx with edge >= x
#     if not is_batched:
#         ids = torch.bucketize(d, bins, right=False) - 1  # [0..nb-1], but safe because d in-range
#         ids = ids.clamp(0, nb - 1)
#         counts = torch.bincount(ids, minlength=nb).float()[:nb]
#         return counts / counts.sum().clamp_min(1.0)
#     else:
#         # For batched, do bucketize first then drop masked
#         ids = torch.bucketize(d, bins, right=False) - 1
#         ids = ids.clamp(0, nb - 1)

#         # Set out-of-range (mask==False) to a sentinel that will be ignored by bincount trick
#         # We'll map (invalid) to nothing by not adding it to any real bin via masking.
#         # Build flat ids only for valid entries.
#         valid_ids = ids[mask]
#         # Batch offsets only for rows that contribute
#         # Compute per-row offsets for valid positions
#         # Make an index of row for each valid entry
#         row_idx = torch.arange(B, device=D.device).unsqueeze(1).expand_as(ids)
#         valid_row = row_idx[mask]
#         flat_ids = valid_ids + valid_row * nb  # (K,)

#         counts = torch.bincount(flat_ids, minlength=B * nb).float().view(B, nb)
#         return counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)


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


# ==============================================================================
# PART 6: LOSS FUNCTIONS
# ==============================================================================

# class HeatKernelLoss(nn.Module):
#     """
#     Heat kernel trace matching loss with Hutchinson trace estimation.
    
#     Args:
#         use_hutchinson: Use stochastic Lanczos quadrature (faster)
#         num_probes: Number of Rademacher probes for Hutchinson
#         chebyshev_degree: Degree of Chebyshev polynomial approximation
#         knn_k: k for kNN graph Laplacian
#         t_list: Heat kernel diffusion times
#         laplacian: Type of Laplacian ('sym' or 'rw')
#     """
    
#     def __init__(
#         self,
#         use_hutchinson: bool = True,
#         num_probes: int = 8,
#         chebyshev_degree: int = 10,
#         knn_k: int = 8,
#         t_list: Tuple[float, ...] = (0.5, 1.0),
#         laplacian: str = 'sym'
#     ):
#         super().__init__()
#         self.use_hutchinson = use_hutchinson
#         self.num_probes = num_probes
#         self.chebyshev_degree = chebyshev_degree
#         self.knn_k = knn_k
#         self.t_list = t_list
#         self.laplacian = laplacian
    
#     def forward(
#         self,
#         L_pred: torch.Tensor,
#         L_target: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         t_list: Optional[List[float]] = None
#     ) -> torch.Tensor:
#         """
#         Compute heat kernel trace loss.
        
#         Args:
#             L_pred: (n, n) predicted Laplacian
#             L_target: (n, n) target Laplacian
#             mask: (n,) optional mask for valid nodes
#             t_list: optional override for diffusion times
            
#         Returns:
#             loss: scalar heat kernel trace loss
#         """
#         if t_list is None:
#             t_list = self.t_list
        
#         # Apply mask to Laplacians
#         # if mask is not None:
#         #     valid_idx = torch.where(mask)[0]
#         #     L_pred = L_pred[valid_idx][:, valid_idx]
#         #     L_target = L_target[valid_idx][:, valid_idx]

#         # NEW (GPU-safe)
#         if mask is not None:
#             # Ensure boolean 1-D
#             mask = mask.bool().view(-1)

#             # If sparse, make dense on GPU (mini-sets are small)
#             if L_pred.layout != torch.strided:
#                 L_pred = L_pred.to_dense()
#             if L_target.layout != torch.strided:
#                 L_target = L_target.to_dense()

#             valid_idx = mask.nonzero(as_tuple=False).squeeze(1)
#             L_pred   = L_pred.index_select(0, valid_idx).index_select(1, valid_idx).contiguous()
#             L_target = L_target.index_select(0, valid_idx).index_select(1, valid_idx).contiguous()

        
#         if self.use_hutchinson:
#             # Hutchinson trace estimation
#             traces_pred = self._hutchinson_heat_trace(L_pred, t_list)
#             traces_target = self._hutchinson_heat_trace(L_target, t_list)
#         else:
#             # Full eigendecomposition (fallback)
#             eigvals_pred, _ = safe_eigh(L_pred, return_vecs=False)
#             eigvals_target, _ = safe_eigh(L_target, return_vecs=False)
#             eigvals_pred = eigvals_pred.clamp(min=0)
#             eigvals_target = eigvals_target.clamp(min=0)
            
#             traces_pred = []
#             traces_target = []
#             for t in t_list:
#                 traces_pred.append(torch.sum(torch.exp(-t * eigvals_pred)))
#                 traces_target.append(torch.sum(torch.exp(-t * eigvals_target)))
            
#             traces_pred = torch.tensor(traces_pred, device=L_pred.device)
#             traces_target = torch.tensor(traces_target, device=L_target.device)
        
#         # Compute loss
#         loss = torch.mean((traces_pred - traces_target) ** 2)
#         return loss
    
#     def _hutchinson_heat_trace(
#         self,
#         L: torch.Tensor,
#         t_list: List[float]
#     ) -> torch.Tensor:
#         """
#         Hutchinson trace estimation with Chebyshev polynomials.
        
#         Args:
#             L: (n, n) Laplacian matrix
#             t_list: list of diffusion times
            
#         Returns:
#             traces: (len(t_list),) trace estimates
#         """
#         device = L.device
#         n = L.shape[0]
        
#         # Estimate max eigenvalue with power iteration
#         lambda_max = self._power_iter_max_eig(L, num_iter=20)
        
#         # Scale L to [-1, 1]
#         L_scaled = L / (lambda_max + 1e-8)
        
#         # Hutchinson trace estimation
#         traces = torch.zeros(len(t_list), device=device, dtype=torch.float32)
        
#         for _ in range(self.num_probes):
#             # Rademacher probe
#             v = torch.empty(n, device=device, dtype=torch.float32).uniform_(-1.0, 1.0).sign()
#             v = v / (n ** 0.5)
            
#             # Lanczos tridiagonalization
#             T = self._lanczos_tridiag(L_scaled, v, self.chebyshev_degree)
            
#             # Compute trace for each t using Chebyshev
#             for i, t in enumerate(t_list):
#                 # Scale t by lambda_max
#                 t_scaled = t * lambda_max
#                 trace_t = self._chebyshev_exp_trace(T, t_scaled)
#                 traces[i] += trace_t
        
#         traces = traces / float(self.num_probes)
#         return traces
    
#     @torch.no_grad()
#     def _power_iter_max_eig(self, A: torch.Tensor, num_iter: int = 20) -> torch.Tensor:
#         """Power iteration to estimate max eigenvalue."""
#         n = A.shape[0]
#         v = torch.randn(n, device=A.device, dtype=torch.float32)
#         v = v / (v.norm() + 1e-12)
        
#         for _ in range(num_iter):
#             v = A @ v
#             v = v / (v.norm() + 1e-12)
        
#         lambda_max = torch.dot(v, A @ v)
#         return lambda_max.abs()
    
#     @torch.no_grad()
#     def _lanczos_tridiag(
#         self,
#         A: torch.Tensor,
#         v0: torch.Tensor,
#         m: int,
#         tol: float = 1e-6
#     ) -> torch.Tensor:
#         """Lanczos tridiagonalization."""
#         device, dtype = A.device, torch.float32
#         n = v0.numel()
        
#         alpha = torch.zeros(m, device=device, dtype=dtype)
#         beta = torch.zeros(m - 1, device=device, dtype=dtype)
        
#         v = v0 / (v0.norm() + 1e-12)
#         w = A @ v
#         alpha[0] = torch.dot(v, w)
#         w = w - alpha[0] * v
        
#         for j in range(1, m):
#             beta[j - 1] = w.norm()
#             if beta[j - 1] <= tol:
#                 alpha = alpha[:j]
#                 beta = beta[:j - 1]
#                 break
            
#             v_next = w / beta[j - 1]
#             w = A @ v_next - beta[j - 1] * v
#             alpha[j] = torch.dot(v_next, w)
#             w = w - alpha[j] * v_next
#             v = v_next
        
#         T = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
#         return T
    
#     @torch.no_grad()
#     def _chebyshev_exp_trace(self, T: torch.Tensor, t: float) -> torch.Tensor:
#         """Compute e1^T exp(-t*T) e1 using eigendecomposition of small T."""
#         evals, evecs = torch.linalg.eigh(T)
#         weights = (evecs[0, :] ** 2)
#         return (weights * torch.exp(-t * evals)).sum()

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
            
            loss = torch.mean((traces_pred - traces_target) ** 2)
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
            loss = ((traces_pred - traces_target) ** 2).mean()
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


# class SlicedWassersteinLoss1D(nn.Module):
#     """1D Sliced Wasserstein distance for distance histograms."""
    
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, hist_pred: torch.Tensor, hist_target: torch.Tensor) -> torch.Tensor:
#         """
#         Compute 1D Sliced Wasserstein distance between histograms.
#         For 1D, this is just the L1 distance between CDFs.
        
#         Args:
#             hist_pred: (num_bins,) predicted histogram (normalized)
#             hist_target: (num_bins,) target histogram (normalized)
            
#         Returns:
#             loss: scalar SW distance
#         """
#         # Compute CDFs
#         cdf_pred = torch.cumsum(hist_pred, dim=0)
#         cdf_target = torch.cumsum(hist_target, dim=0)
        
#         # L1 distance between CDFs
#         loss = torch.mean(torch.abs(cdf_pred - cdf_target))
#         return loss

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
    
# def masked_frobenius_loss(A, B, mask):
#     '''
#     compute ||a-b||_F^2 only over valid masked entries
#     a, b: (batch, n, n)
#     mask: (batch, n) boolean
#     '''

#     #create 2D mask: (batch, n, n)
#     P = mask.unsqueeze(-1) & mask.unsqueeze(-2) #(B, N, N)
#     P = P.float()

#     diff_sq = (A - B).pow(2)
#     return (diff_sq * P).sum() / P.sum().clamp_min(1.0)

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

def wasserstein_1d_quantile_loss(D_pred, D_tgt, m_pairs=4096, p=1, norm="p95"):
    """
    Differentiable 1D OT on distances. Returns scalar loss.
    D_pred, D_tgt: (n,n), symmetric with zeros on diag.
    m_pairs: subsample count for speed.
    p: 1 -> W1 (L1), 2 -> W2 (MSE on quantiles).
    norm: 'none' | 'median' | 'p95' (normalize by target scale).
    """
    dp = _upper_tri_vec(D_pred)
    dt = _upper_tri_vec(D_tgt)

    # optional subsample (same indices for both)
    if dp.numel() > m_pairs:
        idx = torch.randperm(dp.numel(), device=dp.device)[:m_pairs]
        dp = dp.index_select(0, idx)
        dt = dt.index_select(0, idx)

    # robust normalization (decouples global scale from shape)
    if norm != "none":
        if norm == "median":
            s = dt.median().clamp_min(1e-6)
        else:  # p95
            s = torch.quantile(dt, 0.95).clamp_min(1e-6)
        dp = dp / s
        dt = dt / s

    dp_sorted = torch.sort(dp).values
    dt_sorted = torch.sort(dt).values
    if p == 1:
        return F.l1_loss(dp_sorted, dt_sorted)
    else:
        return F.mse_loss(dp_sorted, dt_sorted)


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
