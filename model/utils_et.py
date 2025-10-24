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

def safe_eigh(A: torch.Tensor, return_vecs: bool = True, cpu: bool = True):
    """
    Robust Hermitian eigensolver:
    - symmetrize
    - adaptive jitter
    - float64 on CPU
    - numpy fallback
    """

    A = _symmetrize(A)
    n = A.shape[0]
    dev, dtype = A.device, A.dtype

    #adaptive jitter scaled to matrix magnitude
    scale = torch.nan_to_num(A.abs().max(), nan=1.0, posinf=1.0, neginf=1.0)
    jitter = (1e-8 + 1e-6 * float(scale)) * torch.eye(n, device=dev, dtype=dtype)
    A = A + jitter 

    A64 = A.double().cpu() if cpu else A.double()
    try:
        if return_vecs:
            w, V = torch.linalg.eigh(A64)
        else:
            w = torch.linalg.eigvalsh(A64)
            V = None
    except Exception:
        import numpy as np
        a_np = A64.numpy()
        if return_vecs:
            w_np, V_np = np.linalg.eigh(a_np)
            w = torch.from_numpy(w_np)
            V = torch.from_numpy(V_np)
        else:
            w_np = np.linalg.eigvalsh(a_np)
            w = torch.from_numpy(w_np)
            V = None

    if cpu:
        w = w.to(device=dev, dtype=dtype)
        if V is not None:
            V = V.to(device=dev, dtype=dtype)
    return (w, V) if return_vecs else (w, None)

def compute_distance_hist(D: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized histogram of pairwise distances (upper triangle only).
    FIXED: moves to CPU for histogram computation.
    """
    n = D.shape[0]
    triu_i, triu_j = torch.triu_indices(n, n, offset=1)
    d_vec = D[triu_i, triu_j]
    
    # Move to CPU for histogram (not supported on CUDA)
    device = d_vec.device
    d_vec_cpu = d_vec.cpu()
    bins_cpu = bins.cpu()
    
    # Compute histogram on CPU
    hist, _ = torch.histogram(d_vec_cpu, bins=bins_cpu)
    hist = hist.float() / hist.sum().clamp_min(1)
    
    # Move back to original device
    return hist.to(device)


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

class HeatKernelLoss(nn.Module):
    """
    Heat kernel trace matching loss with Hutchinson trace estimation.
    
    Args:
        use_hutchinson: Use stochastic Lanczos quadrature (faster)
        num_probes: Number of Rademacher probes for Hutchinson
        chebyshev_degree: Degree of Chebyshev polynomial approximation
        knn_k: k for kNN graph Laplacian
        t_list: Heat kernel diffusion times
        laplacian: Type of Laplacian ('sym' or 'rw')
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
            L_pred: (n, n) predicted Laplacian
            L_target: (n, n) target Laplacian
            mask: (n,) optional mask for valid nodes
            t_list: optional override for diffusion times
            
        Returns:
            loss: scalar heat kernel trace loss
        """
        if t_list is None:
            t_list = self.t_list
        
        # Apply mask to Laplacians
        if mask is not None:
            valid_idx = torch.where(mask)[0]
            L_pred = L_pred[valid_idx][:, valid_idx]
            L_target = L_target[valid_idx][:, valid_idx]
        
        if self.use_hutchinson:
            # Hutchinson trace estimation
            traces_pred = self._hutchinson_heat_trace(L_pred, t_list)
            traces_target = self._hutchinson_heat_trace(L_target, t_list)
        else:
            # Full eigendecomposition (fallback)
            eigvals_pred, _ = safe_eigh(L_pred, return_vecs=False)
            eigvals_target, _ = safe_eigh(L_target, return_vecs=False)
            eigvals_pred = eigvals_pred.clamp(min=0)
            eigvals_target = eigvals_target.clamp(min=0)
            
            traces_pred = []
            traces_target = []
            for t in t_list:
                traces_pred.append(torch.sum(torch.exp(-t * eigvals_pred)))
                traces_target.append(torch.sum(torch.exp(-t * eigvals_target)))
            
            traces_pred = torch.tensor(traces_pred, device=L_pred.device)
            traces_target = torch.tensor(traces_target, device=L_target.device)
        
        # Compute loss
        loss = torch.mean((traces_pred - traces_target) ** 2)
        return loss
    
    def _hutchinson_heat_trace(
        self,
        L: torch.Tensor,
        t_list: List[float]
    ) -> torch.Tensor:
        """
        Hutchinson trace estimation with Chebyshev polynomials.
        
        Args:
            L: (n, n) Laplacian matrix
            t_list: list of diffusion times
            
        Returns:
            traces: (len(t_list),) trace estimates
        """
        device = L.device
        n = L.shape[0]
        
        # Estimate max eigenvalue with power iteration
        lambda_max = self._power_iter_max_eig(L, num_iter=20)
        
        # Scale L to [-1, 1]
        L_scaled = L / (lambda_max + 1e-8)
        
        # Hutchinson trace estimation
        traces = torch.zeros(len(t_list), device=device, dtype=torch.float32)
        
        for _ in range(self.num_probes):
            # Rademacher probe
            v = torch.empty(n, device=device, dtype=torch.float32).uniform_(-1.0, 1.0).sign()
            v = v / (n ** 0.5)
            
            # Lanczos tridiagonalization
            T = self._lanczos_tridiag(L_scaled, v, self.chebyshev_degree)
            
            # Compute trace for each t using Chebyshev
            for i, t in enumerate(t_list):
                # Scale t by lambda_max
                t_scaled = t * lambda_max
                trace_t = self._chebyshev_exp_trace(T, t_scaled)
                traces[i] += trace_t
        
        traces = traces / float(self.num_probes)
        return traces
    
    @torch.no_grad()
    def _power_iter_max_eig(self, A: torch.Tensor, num_iter: int = 20) -> torch.Tensor:
        """Power iteration to estimate max eigenvalue."""
        n = A.shape[0]
        v = torch.randn(n, device=A.device, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)
        
        for _ in range(num_iter):
            v = A @ v
            v = v / (v.norm() + 1e-12)
        
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
        """Lanczos tridiagonalization."""
        device, dtype = A.device, torch.float32
        n = v0.numel()
        
        alpha = torch.zeros(m, device=device, dtype=dtype)
        beta = torch.zeros(m - 1, device=device, dtype=dtype)
        
        v = v0 / (v0.norm() + 1e-12)
        w = A @ v
        alpha[0] = torch.dot(v, w)
        w = w - alpha[0] * v
        
        for j in range(1, m):
            beta[j - 1] = w.norm()
            if beta[j - 1] <= tol:
                alpha = alpha[:j]
                beta = beta[:j - 1]
                break
            
            v_next = w / beta[j - 1]
            w = A @ v_next - beta[j - 1] * v
            alpha[j] = torch.dot(v_next, w)
            w = w - alpha[j] * v_next
            v = v_next
        
        T = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
        return T
    
    @torch.no_grad()
    def _chebyshev_exp_trace(self, T: torch.Tensor, t: float) -> torch.Tensor:
        """Compute e1^T exp(-t*T) e1 using eigendecomposition of small T."""
        evals, evecs = torch.linalg.eigh(T)
        weights = (evecs[0, :] ** 2)
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


def create_sc_miniset_pair(Z_all: torch.Tensor, n_set: int, n_overlap: int, k_nn: int = 50, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create two overlapping SC mini-sets from Z-space.
    Returns: indices_A, indices_B, shared_indices_in_A, shared_indices_in_B
    """
    n_total = Z_all.shape[0]
    
    # Sample seed for set A
    seed_i = torch.randint(n_total, (1,)).item()
    D_from_seed = torch.cdist(Z_all[seed_i:seed_i+1], Z_all)[0]
    indices_A = torch.argsort(D_from_seed)[:n_set]
    
    # Select overlap subset
    n_overlap = min(n_overlap, n_set - 5)
    overlap_positions = torch.randperm(n_set)[:n_overlap]
    shared_global = indices_A[overlap_positions]
    
    # Sample seed for set B near seed_i
    D_from_seed_all = torch.cdist(Z_all[seed_i:seed_i+1], Z_all)[0]
    nearby = torch.argsort(D_from_seed_all)[1:k_nn]
    seed_j = nearby[torch.randint(len(nearby), (1,))].item()
    
    # Build set B: shared + new cells
    D_from_seed_j = torch.cdist(Z_all[seed_j:seed_j+1], Z_all)[0]
    candidates = torch.argsort(D_from_seed_j)
    
    # Exclude cells already in A\S
    non_shared_A = indices_A[~torch.isin(torch.arange(n_set, device=device), overlap_positions)]
    mask = ~torch.isin(candidates, non_shared_A)
    candidates = candidates[mask]
    
    n_new = n_set - n_overlap
    new_cells = candidates[:n_new] if len(candidates) >= n_new else candidates
    
    indices_B = torch.cat([shared_global, new_cells])[:n_set]
    
    # Find shared positions in both sets
    shared_in_A = overlap_positions
    shared_in_B = torch.tensor([torch.where(indices_B == g)[0][0].item() for g in shared_global if g in indices_B], device=device)
    
    return indices_A, indices_B, shared_in_A, shared_in_B