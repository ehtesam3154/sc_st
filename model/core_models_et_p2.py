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
from core_models_et_p1 import collate_minisets, collate_sc_minisets, STPairSetDataset, collate_pair_minisets
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

import math

from typing import List, Dict, Optional, Tuple

from scipy.stats import spearmanr, pearsonr



#debug tools
DEBUG = False #master switch for debug logging


#debug tracking variables 
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


# ==============================================================================
# OVERLAP CONSISTENCY LOSS FUNCTIONS (Candidate 1)
# ==============================================================================

def compute_overlap_losses(
    x0_pred_1: torch.Tensor,  # (B, N1, D) denoised predictions for view1
    x0_pred_2: torch.Tensor,  # (B, N2, D) denoised predictions for view2
    idx1_I: torch.Tensor,     # (B, max_I) positions in view1 of overlap points
    idx2_I: torch.Tensor,     # (B, max_I) positions in view2 of overlap points
    I_mask: torch.Tensor,     # (B, max_I) validity mask for overlap points
    mask_1: torch.Tensor,     # (B, N1) validity mask for view1
    mask_2: torch.Tensor,     # (B, N2) validity mask for view2
    kl_tau: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute overlap consistency losses for paired minisets.

    Returns dict with:
    - L_ov_shape: Scale-free Gram consistency loss
    - L_ov_scale: Log-trace scale consistency loss
    - L_ov_kl: Symmetric KL divergence of neighbor distributions
    - debug_dict: Diagnostic information
    """
    B = x0_pred_1.shape[0]
    device = x0_pred_1.device

    # Initialize losses
    L_ov_shape = torch.tensor(0.0, device=device)
    L_ov_scale = torch.tensor(0.0, device=device)
    L_ov_kl = torch.tensor(0.0, device=device)

    valid_batch_count = 0
    debug_info = {
        'I_sizes': [],
        'trG1': [],
        'trG2': [],
        'log_tr_diff': [],
        'mean_pairwise_dist1': [],
        'mean_pairwise_dist2': [],
        'jaccard_k10': [],
    }

    for b in range(B):
        # Get overlap indices for this batch element
        I_valid = I_mask[b]  # (max_I,) bool
        n_I = I_valid.sum().item()

        if n_I < 4:  # Need at least 4 points for meaningful overlap loss
            continue

        # Extract overlap point indices
        idx1 = idx1_I[b, I_valid]  # (n_I,) positions in view1
        idx2 = idx2_I[b, I_valid]  # (n_I,) positions in view2

        # Extract predictions for overlap points
        V1_I = x0_pred_1[b, idx1]  # (n_I, D)
        V2_I = x0_pred_2[b, idx2]  # (n_I, D)

        # Center predictions (remove translation)
        V1_centered = V1_I - V1_I.mean(dim=0, keepdim=True)
        V2_centered = V2_I - V2_I.mean(dim=0, keepdim=True)

        # Compute Gram matrices
        G1 = V1_centered @ V1_centered.T  # (n_I, n_I)
        G2 = V2_centered @ V2_centered.T  # (n_I, n_I)

        tr_G1 = G1.trace() + eps
        tr_G2 = G2.trace() + eps

        # (1) Shape loss: normalized Gram Frobenius distance (scale-free)
        G1_norm = G1 / tr_G1
        G2_norm = G2 / tr_G2
        shape_loss_b = (G1_norm - G2_norm).pow(2).sum()
        L_ov_shape = L_ov_shape + shape_loss_b

        # (2) Scale loss: squared log-trace difference
        log_tr_diff = (torch.log(tr_G1) - torch.log(tr_G2)).pow(2)
        L_ov_scale = L_ov_scale + log_tr_diff

        # (3) KL neighbor distribution loss
        # Compute pairwise squared distances
        D1_sq = torch.cdist(V1_centered, V1_centered).pow(2)  # (n_I, n_I)
        D2_sq = torch.cdist(V2_centered, V2_centered).pow(2)  # (n_I, n_I)

        # Build soft neighbor distributions (excluding self)
        # P1[i,j] = softmax(-d1[i,j]^2 / tau) over j != i
        # Mask out diagonal with large value before softmax
        D1_masked = D1_sq.clone()
        D2_masked = D2_sq.clone()
        diag_mask = torch.eye(n_I, device=device, dtype=torch.bool)
        D1_masked[diag_mask] = float('inf')
        D2_masked[diag_mask] = float('inf')

        P1 = F.softmax(-D1_masked / kl_tau, dim=1)  # (n_I, n_I)
        P2 = F.softmax(-D2_masked / kl_tau, dim=1)  # (n_I, n_I)

        # Clamp for numerical stability
        P1 = P1.clamp(min=eps)
        P2 = P2.clamp(min=eps)

        # Symmetric KL divergence: (KL(P1||P2) + KL(P2||P1)) / 2
        # KL(P||Q) = sum(P * log(P/Q))
        kl_12 = (P1 * (torch.log(P1) - torch.log(P2))).sum(dim=1).mean()  # Mean over points
        kl_21 = (P2 * (torch.log(P2) - torch.log(P1))).sum(dim=1).mean()
        kl_sym = (kl_12 + kl_21) / 2
        L_ov_kl = L_ov_kl + kl_sym

        valid_batch_count += 1

        # Debug info
        debug_info['I_sizes'].append(n_I)
        debug_info['trG1'].append(tr_G1.item())
        debug_info['trG2'].append(tr_G2.item())
        debug_info['log_tr_diff'].append(log_tr_diff.sqrt().item())

        # Mean pairwise distances (for collapse detection)
        D1_upper = D1_sq[torch.triu(torch.ones_like(D1_sq, dtype=torch.bool), diagonal=1)]
        D2_upper = D2_sq[torch.triu(torch.ones_like(D2_sq, dtype=torch.bool), diagonal=1)]
        if D1_upper.numel() > 0:
            debug_info['mean_pairwise_dist1'].append(D1_upper.mean().sqrt().item())
            debug_info['mean_pairwise_dist2'].append(D2_upper.mean().sqrt().item())

        # Jaccard@10 as diagnostic (training-time proxy)
        if n_I >= 10:
            k = min(10, n_I - 1)
            _, knn1 = torch.topk(D1_sq, k=k+1, largest=False, dim=1)  # Include self
            _, knn2 = torch.topk(D2_sq, k=k+1, largest=False, dim=1)
            knn1 = knn1[:, 1:]  # Exclude self
            knn2 = knn2[:, 1:]

            jaccard_sum = 0.0
            for i in range(n_I):
                set1 = set(knn1[i].tolist())
                set2 = set(knn2[i].tolist())
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                if union > 0:
                    jaccard_sum += intersection / union
            jaccard_mean = jaccard_sum / n_I
            debug_info['jaccard_k10'].append(jaccard_mean)

    # Normalize by valid batch count
    if valid_batch_count > 0:
        L_ov_shape = L_ov_shape / valid_batch_count
        L_ov_scale = L_ov_scale / valid_batch_count
        L_ov_kl = L_ov_kl / valid_batch_count

    return {
        'L_ov_shape': L_ov_shape,
        'L_ov_scale': L_ov_scale,
        'L_ov_kl': L_ov_kl,
        'valid_batch_count': valid_batch_count,
        'debug_info': debug_info,
    }


def apply_coupled_noise(
    eps_1: torch.Tensor,      # (B, N1, D) noise for view1
    eps_2: torch.Tensor,      # (B, N2, D) noise for view2
    idx1_I: torch.Tensor,     # (B, max_I) positions in view1 of overlap points
    idx2_I: torch.Tensor,     # (B, max_I) positions in view2 of overlap points
    I_mask: torch.Tensor,     # (B, max_I) validity mask for overlap points
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply coupled noise: for overlapping points, copy noise from view1 to view2.
    This ensures the same noise is used for the same physical points across views.

    Returns: (eps_1, eps_2_coupled) where eps_2 has matching noise for overlap points.
    """
    B = eps_1.shape[0]
    eps_2_coupled = eps_2.clone()

    for b in range(B):
        I_valid = I_mask[b]  # (max_I,) bool
        n_I = I_valid.sum().item()

        if n_I == 0:
            continue

        idx1 = idx1_I[b, I_valid]  # (n_I,) positions in view1
        idx2 = idx2_I[b, I_valid]  # (n_I,) positions in view2

        # Copy noise from view1 to corresponding positions in view2
        eps_2_coupled[b, idx2] = eps_1[b, idx1]

    return eps_1, eps_2_coupled


def apply_z_ln(Z_set: torch.Tensor, context_encoder: nn.Module) -> torch.Tensor:
    """
    Apply LayerNorm to Z_set features only.
    If anchor channel exists (last dim), leave it untouched.
    """
    input_dim = Z_set.shape[-1]
    anchor_train = getattr(context_encoder, "anchor_train", False)
    expected_in = getattr(context_encoder, "input_dim", None)
    if expected_in is None and hasattr(context_encoder, "input_proj"):
        expected_in = context_encoder.input_proj.in_features

    # If anchor channel present: normalize all but last dim
    if anchor_train and expected_in is not None and input_dim == expected_in:
        z_feat = Z_set[..., :-1]
        z_anchor = Z_set[..., -1:]
        z_feat = F.layer_norm(z_feat, (z_feat.shape[-1],))
        return torch.cat([z_feat, z_anchor], dim=-1)

    # Otherwise normalize full vector
    return F.layer_norm(Z_set, (Z_set.shape[-1],))


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


def _validate_st_uid_uniqueness(batch: Dict, device: str = 'cuda'):
    """
    Smoke test: Verify ST UIDs are correct.
    
    Checks:
    1. Within each miniset, UIDs should be unique (no duplicate spots in one patch)
    2. Cross-slide disjointness: UIDs from different slides should not overlap
       (verifies the (slide_id << 32) + idx packing is correct)
    
    NOTE: Duplicates ACROSS different minisets in the same batch are EXPECTED
    because different patches can sample overlapping spots from the same slide.
    """
    if batch.get('is_sc', False):
        return  # Skip SC batches
    
    global_indices = batch.get('global_indices', None)
    if global_indices is None:
        return
    
    overlap_info = batch.get('overlap_info', None)
    if overlap_info is None:
        return
    
    mask_batch = batch.get('mask', None)
    B, N = global_indices.shape
    
    # ========== CHECK 1: Uniqueness WITHIN each miniset ==========
    for b in range(B):
        # Get valid (non-padding) UIDs for this miniset
        if mask_batch is not None:
            valid = mask_batch[b] & (global_indices[b] >= 0)
        else:
            valid = global_indices[b] >= 0
        
        uids_b = global_indices[b][valid]
        
        if uids_b.numel() == 0:
            continue
        
        # Check for duplicates within this miniset
        n_unique = uids_b.unique().numel()
        n_total = uids_b.numel()
        if n_unique != n_total:
            n_dups = n_total - n_unique
            raise AssertionError(
                f"[SPOT-IDENTITY] Duplicate UID within miniset b={b}! "
                f"{n_dups} duplicates (total={n_total}, unique={n_unique})"
            )
    
    # ========== CHECK 2: Cross-slide disjointness (only if multiple slides) ==========
    # Collect UIDs per slide (across all minisets)
    uids_by_slide = {}
    for b in range(B):
        if mask_batch is not None:
            valid = mask_batch[b] & (global_indices[b] >= 0)
        else:
            valid = global_indices[b] >= 0
        
        uids_b = global_indices[b][valid]
        slide_b = overlap_info[b]['slide_id']
        
        if slide_b not in uids_by_slide:
            uids_by_slide[slide_b] = set()
        uids_by_slide[slide_b].update(uids_b.tolist())
    
    # Only check cross-slide if we have multiple slides
    slide_ids = list(uids_by_slide.keys())
    if len(slide_ids) <= 1:
        return  # Single slide - no cross-slide check needed
    
    for i, slide_a in enumerate(slide_ids):
        for slide_b in slide_ids[i+1:]:
            intersection = uids_by_slide[slide_a] & uids_by_slide[slide_b]
            if len(intersection) > 0:
                # Decode to check if this is a real collision or packing error
                sample_uid = list(intersection)[0]
                decoded_slide = sample_uid >> 32
                decoded_idx = sample_uid & 0xFFFFFFFF
                raise AssertionError(
                    f"[SPOT-IDENTITY] Cross-slide collision! "
                    f"Slide {slide_a} and {slide_b} share {len(intersection)} UIDs. "
                    f"Example UID={sample_uid} decodes to slide={decoded_slide}, idx={decoded_idx}. "
                    f"This indicates a UID packing bug."
                )



# ==============================================================================
# STAGE C: SET-EQUIVARIANT CONTEXT ENCODER
# ==============================================================================

class SetEncoderContext(nn.Module):
    """
    Permutation-equivariant context encoder using Set Transformer.
    
    Takes set of embeddings Z_set and produces context H.
    Uses ISAB blocks for O(mn) complexity.
    
    NEW: Supports optional anchor channel (h_dim+1 input) for anchored training.
    """
    
    def __init__(
        self,
        h_dim: int = 128,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 3,
        isab_m: int = 64,
        ln: bool = True,
        anchor_train: bool = False,  # NEW: if True, expect h_dim+1 input
    ):
        """
        Args:
            h_dim: input embedding dimension (base, without anchor channel)
            c_dim: output context dimension
            n_heads: number of attention heads
            n_blocks: number of ISAB blocks
            isab_m: number of inducing points in ISAB
            ln: use layer normalization
            anchor_train: if True, input projection accepts h_dim+1
        """
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.anchor_train = anchor_train
        
        # Input projection - handle anchor channel
        input_dim = h_dim + 1 if anchor_train else h_dim
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, c_dim)
        
        # Track if we've logged the padding warning
        self._logged_pad_warning = False
        
        # Stack of ISAB blocks
        self.isab_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])
    
    def forward(self, Z_set: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_set: (batch, n, h_dim) or (batch, n, h_dim+1) set of embeddings
            mask: (batch, n) boolean mask (True = valid)
            
        Returns:
            H: (batch, n, c_dim) context features
        """
        batch_size, n, input_dim_actual = Z_set.shape
        
        # ========== NEW: Handle missing anchor channel (for backward compatibility) ==========
        if input_dim_actual == self.input_dim - 1:
            # Input is missing anchor channel - pad with zeros
            # This allows old sampling code to work with anchored checkpoints
            zeros = torch.zeros(batch_size, n, 1, device=Z_set.device, dtype=Z_set.dtype)
            Z_set = torch.cat([Z_set, zeros], dim=-1)
            
            if not self._logged_pad_warning:
                # Only log on rank 0 to avoid DDP spam
                should_log = True
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        should_log = (dist.get_rank() == 0)
                except:
                    pass
                
                if should_log:
                    print(f"[ANCHOR-CONTEXT] padded_missing_anchor_channel=1 "
                          f"(input_dim={input_dim_actual}, expected={self.input_dim})")
                self._logged_pad_warning = True

                
        elif input_dim_actual != self.input_dim:
            raise ValueError(
                f"SetEncoderContext: expected input dim {self.input_dim} or {self.input_dim-1}, "
                f"got {input_dim_actual}"
            )
        
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


def check_target_scale_consistency(V_target, G_target, mask):
    """
    Checks if V_target and G_target imply the same physical scale.
    Returns the scale ratio s = RMS(V_target) / RMS(V_from_G).
    If s != 1.0, your targets are fighting each other.
    """
    with torch.no_grad():
        # 1. Get RMS of raw V_target (centered)
        # Center per sample
        mask_f = mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_v = (V_target * mask_f).sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        V_centered = (V_target - mean_v) * mask_f
        
        # RMS = Frobenius norm / sqrt(N*D)
        denom = (n_valid * V_target.shape[-1]).sqrt()
        rms_v = V_centered.norm(dim=(1,2)) / denom.squeeze()

        # 2. Reconstruct V from Gram and get its RMS
        # Factor G -> V
        # Note: uet.factor_from_gram should already return centered V
        V_from_G = torch.zeros_like(V_target)
        batch_size = V_target.shape[0]
        rms_g_list = []
        
        for i in range(batch_size):
            n = int(mask[i].sum().item())
            if n < 3: 
                rms_g_list.append(1.0) # dummy
                continue
            
            G_i = G_target[i, :n, :n].float()
            # Reconstruct V (this usually does Eigendecomp)
            V_rec = uet.factor_from_gram(G_i, V_target.shape[-1]) 
            
            # RMS of reconstruction
            rms_rec = V_rec.norm() / math.sqrt(n * V_target.shape[-1])
            rms_g_list.append(rms_rec.item())
            
        rms_g = torch.tensor(rms_g_list, device=V_target.device)

        # 3. Ratio
        ratio = rms_v / (rms_g + 1e-8)
        
        # print(f"[SCALE CHECK] V_target RMS: {rms_v.mean():.4f}")
        # print(f"[SCALE CHECK] G_target RMS: {rms_g.mean():.4f}")
        # print(f"[SCALE CHECK] Ratio (V/G):  {ratio.mean():.4f} (Should be 1.00)")
        
        return ratio.mean().item()
    

def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> dict:
    """Compute statistics over valid (masked) entries only."""
    mask_bool = mask.bool()
    
    # --- FIX START ---
    if mask.dim() == 2 and x.dim() == 3:  # (B,N,D)
        # Don't unsqueeze the mask. 
        # x[mask_bool] returns shape (Total_Valid_Nodes, D)
        x_valid = x[mask_bool] 
    # --- FIX END ---
    
    elif mask.dim() == 2 and x.dim() == 2:  # (B,N)
        x_valid = x[mask_bool]
    else:
        x_valid = x.flatten()
    
    if x_valid.numel() == 0:
        return {'mean': 0.0, 'std': 0.0, 'rms': 0.0, 'absmean': 0.0, 
                'min': 0.0, 'max': 0.0}
    
    return {
        'mean': float(x_valid.mean().item()),
        'std': float(x_valid.std().item()),
        'rms': float(x_valid.pow(2).mean().sqrt().item()),
        'absmean': float(x_valid.abs().mean().item()),
        'min': float(x_valid.min().item()),
        'max': float(x_valid.max().item()),
    }
# ==============================================================================
# STAGE C: DIFFUSION SCORE NETWORK sψ
# ==============================================================================


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

        # NEW: Store sigma_data for EDM preconditioning reference
        self.sigma_data = None  # Will be set during training

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

        #FiLM conditioning layers (per block modulation)
        # input [H, t_emb] concatenated -> output: gamma, beta for modulation
        # H is (B, N, c_dim), t_emb is (B, 1, c_dim) -> expanded to (B, N, c_dim)
        # so FiLM input is 2 * c_dim, output is 2 * c_dim (gamma + beta)
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * c_dim, c_dim),
                nn.SiLU(),
                nn.Linear(c_dim, 2 * c_dim)
            )
            for _ in range(n_blocks)
        ])

        # init film output layers to near-zero for stable training
        for film in self.film_layers:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

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
        attn_cached: dict = None, return_dist_aux: bool = False,
        sigma_raw: torch.Tensor = None,
        x_raw: torch.Tensor = None,     # NEW: raw centered geometry (Fix #1)
        c_in: torch.Tensor = None,       # NEW: for self-cond feature scaling (Fix #3)
        center_mask: torch.Tensor = None
        ) -> torch.Tensor:
        """
        Args:
            V_t: (B, N, D_latent) noisy coords at time t
            t: (B,) or (B,1) diffusion time
            H: (B, N, c_dim) context from SetEncoderContext
            mask: (B, N)
            self_cond: optional (B, N, D_latent) predicted V_0 from previous step
            attn_cached: optional dict to reuse distance bias
            sigma_raw: optional (B,) raw sigma values from EDM
            x_raw: optional (B, N, D_latent) raw centered geometry for distance bias (Fix #1)
            c_in: optional (B, 1, 1) preconditioning scalar for self_cond scaling (Fix #3)
        
        Returns:
            eps_hat: (B, N, D_latent)
        """
        B, N, D = V_t.shape
        
        if self.use_canonicalize:
            mask_center = center_mask if center_mask is not None else mask
            V_in, _ = uet.center_only(V_t, mask_center)
            if self_cond is not None:
                self_cond_canon, _ = uet.center_only(self_cond, mask_center)
            else:
                self_cond_canon = None
        else:
            V_in = V_t
            self_cond_canon = self_cond

        # --- Compute sigma for gating decisions ---
        if sigma_raw is not None:
            sigma_for_gating = sigma_raw.detach().view(-1).float()  # (B,)
        else:
            # Fallback: assume t is c_noise from EDM
            sigma_for_gating = torch.exp(4.0 * t.squeeze(-1)).detach().view(-1).float()
        
        # --- FIX #2: Smooth σ gating instead of hard threshold ---
        # geom_gate: 1 at low σ, 0 at high σ (smooth transition around 0.30)
        sigma_cut = 0.30
        sigma_k = 20.0  # Steepness of transition
        geom_gate = torch.sigmoid((sigma_cut - sigma_for_gating) * sigma_k)  # (B,)
        
        # --- FIX #1: Raw-space geometry source for distance bias ---
        # Use x_raw if provided (from forward_edm), otherwise fallback to V_in
        if x_raw is not None:
            V_geom_for_bias = x_raw  # Raw centered x_t (units match st_dist_bin_edges)
        else:
            V_geom_for_bias = V_in  # Fallback for backward compatibility
        
        # === CHATGPT FIX 3 (REVISED): Pick ONE geometry source, NO blending ===
        # Policy: If self_cond exists, use it; else use x_raw (raw centered input).
        # This prevents geometry distortions from mixing two different estimates.
        # IMPORTANT: geom_gate is NO LONGER used for source selection or attenuation.
        
        if self_cond_canon is not None:
            V_bias_geom = self_cond_canon  # Use self-conditioning for geometry bias
        else:
            V_bias_geom = V_geom_for_bias  # Fallback to raw centered input (x_raw passed in)
        
        # V_geom is now ONLY used for distance bias computation (not blended)
        V_geom = V_bias_geom
        
        # geom_ok: geometry is usable when we have valid points (mask-based, NOT σ-based)
        # We always have x_raw, so geometry is always "ok" as long as mask has valid points
        geom_ok = (mask.sum(dim=-1) >= 2)  # (B,) True if sample has at least 2 valid points


        # DEBUG: Log geometry source usage
        if self.training and (torch.rand(()).item() < 0.01):
            with torch.no_grad():
                has_sc = (self_cond_canon is not None)
                n_low = int((geom_gate > 0.5).sum().item())
                B_dbg = int(sigma_for_gating.numel())
                geom_gate_mean = geom_gate.mean().item()
                print(f"[DEBUG-GEOM-SRC] B={B_dbg} σ_mean={sigma_for_gating.mean():.3f} "
                    f"self_cond={'YES' if has_sc else 'NO'} "
                    f"geom_gate_mean={geom_gate_mean:.3f} n_lowσ={n_low}/{B_dbg}")

        # Step 2: Compose node features
        features = [H]  # Start with context
        
        # Add time embedding (expanded to all nodes)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1) 
        features.append(t_emb_expanded)
        
        # --- FIX #3 CORRECTED: Consistent self-conditioning feature scaling for ALL modes ---
        if self.self_conditioning and self_cond_canon is not None:
            # FIRST: Scale self_cond to match preconditioned V_in scale
            # This ensures consistent units regardless of sc_feat_mode
            if c_in is not None:
                self_cond_feat_input = self_cond_canon * c_in  # (B, 1, 1) broadcasts
            else:
                self_cond_feat_input = self_cond_canon  # Fallback
            
            # THEN: Apply mode-specific processing to the SCALED input
            if self.sc_feat_mode == "mlp":
                sc_feat = self.sc_mlp(self_cond_feat_input)  # MLP sees scaled input
            else:  # concat
                sc_feat = self_cond_feat_input  # Already scaled
            
            features.append(sc_feat)
        elif self.self_conditioning:  # No self_cond provided, use zeros
            sc_feat = torch.zeros(B, N, self.D_latent, device=V_t.device)
            features.append(sc_feat)

        # === FIX: SC-SCALE identity check ===
        # Identity: rms(sc_feat)/rms(V_in) ≈ rms(sc_canon)/rms(x_raw)
        # This is because V_in = c_in * x_raw and sc_feat = c_in * sc_canon
        
        if self.training and self_cond_canon is not None and (torch.rand(()).item() < 0.005):
            with torch.no_grad():
                mask_f = mask.unsqueeze(-1).float()
                
                # Per-sample RMS computation
                def _per_sample_rms(tensor, mf):
                    """Compute RMS per sample, return (B,) tensor."""
                    B = tensor.shape[0]
                    denom = mf.sum(dim=(1, 2)).clamp_min(1)  # (B,)
                    sq_sum = (tensor.pow(2) * mf).sum(dim=(1, 2))  # (B,)
                    return (sq_sum / denom).sqrt()  # (B,)
                
                rms_vin = _per_sample_rms(V_in, mask_f)  # (B,)
                rms_sc_scaled = _per_sample_rms(self_cond_feat_input, mask_f)  # (B,)
                rms_sc_canon = _per_sample_rms(self_cond_canon, mask_f)  # (B,)
                
                # Get x_raw_canon for identity check
                if x_raw is not None:
                    x_raw_canon, _ = uet.center_only(x_raw, mask)
                    rms_xraw = _per_sample_rms(x_raw_canon, mask_f)
                else:
                    rms_xraw = rms_vin  # Fallback
                
                # Compute per-sample ratios
                ratio_scaled = rms_sc_scaled / rms_vin.clamp_min(1e-8)  # (B,)
                ratio_raw = rms_sc_canon / rms_xraw.clamp_min(1e-8)  # (B,)
                ratio_error = (ratio_scaled - ratio_raw).abs()  # (B,)
                
                # Print statistics using torch (no numpy needed)
                print(f"\n[DEBUG-SC-SCALE] Self-cond identity check:")
                print(f"  IDENTITY: rms(sc_feat)/rms(V_in) should ≈ rms(sc_canon)/rms(x_raw)")
                print(f"  ratio_scaled: median={ratio_scaled.median():.3f} "
                      f"p10={ratio_scaled.quantile(0.1):.3f} p90={ratio_scaled.quantile(0.9):.3f}")
                print(f"  ratio_raw:    median={ratio_raw.median():.3f} "
                      f"p10={ratio_raw.quantile(0.1):.3f} p90={ratio_raw.quantile(0.9):.3f}")
                print(f"  ratio_error:  median={ratio_error.median():.4f} max={ratio_error.max():.4f}")
                
                if ratio_error.median() > 0.1:
                    print(f"  ⚠️ WARNING: Identity violated! Check self_cond scaling logic.")

        
        # Add angle features (use V_geom which is already blended)
        if self.use_angle_features:
            if geom_ok.any():
                idx = uet.knn_graph(V_geom, mask, k=self.knn_k)
                angle_feat = uet.angle_features(V_geom, mask, idx, n_angle_bins=self.angle_bins)
                # Gate by smooth geom_gate (not binary geom_ok)
                gate_angle = geom_gate.view(-1, 1, 1)
                angle_feat = angle_feat * gate_angle
                features.append(angle_feat)
            else:
                angle_feat = torch.zeros(B, N, self.angle_bins, device=V_t.device)
                features.append(angle_feat)

        # --- CORRECTED: Coordinate features are NEVER gated ---
        # The network must always see the noisy coordinates V_in for denoising.
        # Only gate AUXILIARY geometry features (distance bias, angles), not the core coordinates.
        V_coord_feat = V_in  # NO GATING - always pass coordinates
        features.append(V_coord_feat)


        # --- DEBUG: Verify coordinate features are not gated ---
        if self.training and (torch.rand(()).item() < 0.000000005):  # ~0.5% of steps
            with torch.no_grad():
                sigma_for_dbg = sigma_for_gating if sigma_raw is not None else torch.zeros(B, device=V_t.device)
                
                # Check coordinate feature is same as V_in
                coord_ratio = V_coord_feat.norm() / V_in.norm().clamp(min=1e-8)
                
                # Per-sigma-bin check
                sigma_bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, float('inf'))]
                print(f"\n[DEBUG-COORD-FEAT] Coordinate feature verification:")
                for lo, hi in sigma_bins:
                    in_bin = (sigma_for_dbg >= lo) & (sigma_for_dbg < hi)
                    if in_bin.any():
                        v_in_bin = V_in[in_bin]
                        v_coord_bin = V_coord_feat[in_bin]
                        mask_bin = mask[in_bin].unsqueeze(-1).float()
                        
                        rms_vin = (v_in_bin.pow(2) * mask_bin).sum().div(mask_bin.sum()).sqrt().item()
                        rms_coord = (v_coord_bin.pow(2) * mask_bin).sum().div(mask_bin.sum()).sqrt().item()
                        ratio = rms_coord / max(rms_vin, 1e-8)
                        
                        status = "✅" if abs(ratio - 1.0) < 0.01 else "❌ GATED!"
                        print(f"  σ∈[{lo:.1f},{hi:.1f}): rms(V_in)={rms_vin:.4f}, "
                            f"rms(V_coord)={rms_coord:.4f}, ratio={ratio:.4f} {status}")


        # Concatenate all features
        X = torch.cat(features, dim=-1)
        X = self.input_proj(X)
        
        # Step 3: Distance-bucket attention bias
        attn_bias = None
        bin_ids = None
        bin_embeddings = None

        if self.use_dist_bias:
            if attn_cached is not None and 'bias' in attn_cached:
                attn_bias = attn_cached['bias']
                bin_ids = attn_cached.get('bin_ids', None)
                bin_embeddings = attn_cached.get('bin_embeddings', None)
            elif geom_ok.any():
                # --- FIX #1 CONTINUED: Use V_geom (raw-space blended) for distance bias ---
                D2 = uet.pairwise_dist2(V_geom, mask)
                attn_bias, _, bin_ids = uet.make_distance_bias(
                    D2, mask, 
                    n_bins=self.dist_bins,
                    d_emb=32,
                    share_across_heads=self.dist_head_shared,
                    E_bin=self.E_bin,
                    W=self.W_bias,
                    alpha_bias=self.alpha_bias,
                    device=V_t.device,
                    bin_edges=self.st_dist_bin_edges
                )
                
                # === CHATGPT FIX 4 (REVISED): NO σ-attenuation of bias strength ===
                # Gate ONLY on geometry validity (mask-based), NOT on σ.
                # The bias should be full-strength whenever we have valid geometry.
                # geom_ok is (B,) bool - True if sample has at least 2 valid points.
                
                # Create validity gate (1.0 if valid geometry, 0.0 otherwise)
                validity_gate = geom_ok.float().view(-1, 1, 1, 1)  # (B, 1, 1, 1)
                attn_bias = attn_bias * validity_gate
                
                # DEBUG: Log when bias is being zeroed due to invalid geometry (should be rare)
                if self.training and (torch.rand(()).item() < 0.001):
                    n_valid_geom = geom_ok.sum().item()
                    n_total = geom_ok.numel()
                    if n_valid_geom < n_total:
                        print(f"[DEBUG-BIAS-GATE] {n_total - n_valid_geom}/{n_total} samples had bias zeroed (invalid geometry)")

                if self.use_st_dist_head:
                    bin_embeddings = self.E_bin[bin_ids]

                if attn_cached is not None:
                    attn_cached['bias'] = attn_bias
                    attn_cached['bin_ids'] = bin_ids
                    attn_cached['bin_embeddings'] = bin_embeddings
            else:
                attn_bias = None
                bin_ids = None
                bin_embeddings = None

        # Apply X↔X distance-biased attention exactly once
        if self.use_dist_bias and attn_bias is not None:
            X = self.bias_sab(X, mask=mask, attn_bias=attn_bias)
        else:
            X = self.bias_sab(X, mask=mask, attn_bias=None)
        X = X * mask.unsqueeze(-1).float()
        
        # Step 4: Apply ISAB blocks with FiLM conditioning
        film_cond = torch.cat([H, t_emb_expanded], dim=-1)

        for i, isab in enumerate(self.denoise_blocks):
            X = isab(X, mask=mask, attn_bias=None)
            X = X * mask.unsqueeze(-1).float()

            gamma_beta = self.film_layers[i](film_cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            X = X * (1.0 + gamma) + beta
            X = X * mask.unsqueeze(-1).float()
        
        # Step 5: Output head
        eps_hat = self.output_head(X)
        eps_hat = eps_hat * mask.unsqueeze(-1).float()

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
            return_debug: bool = False,
            center_mask: torch.Tensor = None,  # NEW: optional mask for centering (fixes ctx-drop frame shift)
        ) -> torch.Tensor:
            """
            EDM-preconditioned forward pass.
            Returns denoised estimate x0_pred.
            
            D_θ(x, σ) = c_skip · x + c_out · F_θ(c_in · x; c_noise, H)
            
            Args:
                center_mask: If provided, use this mask for centering instead of `mask`.
                             This ensures consistent coordinate frames between normal and
                             context-dropped forward passes.
            """
            B, N, D = x.shape
            
            # Compute preconditioning
            c_skip, c_out, c_in, c_noise = uet.edm_precond(sigma, sigma_data)
            # c_skip, c_out, c_in: (B, 1, 1)
            # c_noise: (B, 1)
            
            # [PHASE 4] Debug dict initialization
            debug_dict = {} if return_debug else None
            
            if return_debug:
                debug_dict['c_skip'] = c_skip[:2].detach().cpu()
                debug_dict['c_out'] = c_out[:2].detach().cpu()
                debug_dict['c_in'] = c_in[:2].detach().cpu()
                debug_dict['c_noise'] = c_noise[:2].detach().cpu()
            
            # --- FIX #1: Center in raw space BEFORE preconditioning ---
            # x_c is the raw centered geometry (units match V_target / st_dist_bin_edges)
            # Use center_mask if provided (for ctx-drop consistency), otherwise use mask
            mask_for_centering = center_mask if center_mask is not None else mask
            x_c, mean_shift = uet.center_only(x, mask_for_centering)  # (B, N, D)

            
            # Scale input for network (preconditioned space)
            x_in = c_in * x_c
            
            if return_debug:
                debug_dict['x_in_stats'] = masked_stats(x_in, mask)
                debug_dict['H_stats'] = masked_stats(H, mask)
            
            # --- FIX #1 CONTINUED: Pass raw centered geometry to forward() ---
            # This ensures distance bias is computed in raw units (matching st_dist_bin_edges)
            F_x = self.forward(
                x_in, 
                c_noise, 
                H, 
                mask, 
                self_cond=self_cond, 
                sigma_raw=sigma,
                x_raw=x_c,      # NEW: raw centered geometry for distance bias
                c_in=c_in,       # NEW: for consistent self-cond feature scaling
                center_mask=mask_for_centering
            )
            
            # If forward returns tuple (for dist_aux), extract just the prediction
            if isinstance(F_x, tuple):
                F_x = F_x[0]
            
            if return_debug:
                debug_dict['F_x_stats'] = masked_stats(F_x, mask)
            
            # EDM denoiser output - use centered x_c for skip path
            x0_pred = c_skip * x_c + c_out * F_x
            x0_pred = x0_pred * mask.unsqueeze(-1).float()
            
            if return_debug:
                skip_term = c_skip * x_c * mask.unsqueeze(-1).float()
                out_term = c_out * F_x * mask.unsqueeze(-1).float()
                debug_dict['skip_term_stats'] = masked_stats(skip_term, mask)
                debug_dict['out_term_stats'] = masked_stats(out_term, mask)
                debug_dict['x0_pred_stats'] = masked_stats(x0_pred, mask)
                debug_dict['out_skip_ratio'] = (
                    debug_dict['out_term_stats']['std'] / 
                    (debug_dict['skip_term_stats']['std'] + 1e-8)
                )
            
            if return_debug:
                return x0_pred, debug_dict
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


# ==================== CONTEXT AUGMENTATION ====================
def apply_context_augmentation(Z_set, mask, noise_std=0.02, dropout_rate=0.1):
    """
    Apply stochastic augmentation to Z_set (embedding inputs).
    Only affects conditioning, not V_target (geometry stays unchanged).
    
    Args:
        Z_set: (B, N, h_dim) embeddings
        mask: (B, N) valid mask
        noise_std: relative std for Gaussian noise
        dropout_rate: fraction of features to zero out
    
    Returns:
        Z_aug: augmented embeddings
    """
    B, N, h_dim = Z_set.shape
    Z_aug = Z_set.clone()
    
    # Compute feature-wise RMS for relative noise scaling
    # Only consider valid positions
    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
    valid_count = mask_expanded.sum(dim=(0, 1)).clamp(min=1)  # (h_dim,)
    feature_rms = ((Z_set ** 2) * mask_expanded).sum(dim=(0, 1)) / valid_count
    feature_rms = torch.sqrt(feature_rms + 1e-8)  # (h_dim,)
    
    # Option 1: Add Gaussian noise (scaled by feature RMS)
    if noise_std > 0:
        noise = torch.randn_like(Z_aug) * (noise_std * feature_rms)
        Z_aug = Z_aug + noise * mask_expanded
    
    # Option 2: Feature dropout (randomly zero out some features)
    if dropout_rate > 0:
        # Create dropout mask: (B, 1, h_dim) - same for all positions in a batch
        dropout_mask = (torch.rand(B, 1, h_dim, device=Z_aug.device) > dropout_rate).float()
        # Scale remaining features to preserve expected value
        Z_aug = Z_aug * dropout_mask / (1.0 - dropout_rate + 1e-8)
    
    return Z_aug


# ==============================================================================
# A/B TESTS FOR SIGMA/TIME CONDITIONING DIAGNOSIS
# ==============================================================================

def run_ab_tests_fixed_batch(
    score_net,
    context_encoder,
    fixed_batch_data,
    sigma_data: float,
    fixed_eval_sigmas: list,
    device: str,
    epoch: int,
):
    """
    Run A/B tests on fixed batch to diagnose σ/time conditioning issues.
    
    Test 1 (A/B 1): Is σ-conditioning actually being used?
        - A: Normal time/σ signal
        - B: Constant time (force σ=0.3 for time embedding, but keep true preconditioning)
        
    Test 2 (Decomposition): Is shrink coming from the learned branch?
        - Log RMS of x_skip, x_out, x0_pred separately
        
    Test 3 (A/B 6): Does a different time signal change behavior?
        - A: Current time input (c_noise = log(σ)/4)
        - B: Alternative: c_noise = log(σ) (unscaled)
    
    These tests are EVAL ONLY and do not affect training.
    """
    print(f"\n{'='*80}")
    print(f"A/B TESTS - Epoch {epoch}")
    print(f"{'='*80}")
    
    # Set eval mode
    was_training_score = score_net.training
    was_training_ctx = context_encoder.training
    score_net.eval()
    context_encoder.eval()
    
    # Handle DDP wrapper for score_net.forward
    if hasattr(score_net, 'module'):
        score_net_unwrapped = score_net.module
    else:
        score_net_unwrapped = score_net
    
    with torch.no_grad():
        Z_fixed = fixed_batch_data['Z_set'].to(device)
        mask_fixed = fixed_batch_data['mask'].to(device)
        V_target_fixed = fixed_batch_data['V_target'].to(device)
        
        B_fixed = Z_fixed.shape[0]
        
        # Get D_latent from score_net (handle DDP wrapper)
        D_lat = score_net_unwrapped.D_latent

        Z_fixed = apply_z_ln(Z_fixed, context_encoder)
        
        H_fixed = context_encoder(Z_fixed, mask_fixed)
        
        mask_f = mask_fixed.unsqueeze(-1).float()
        valid_count = mask_f.sum()
        
        # Helper: compute RMS over valid entries
        def masked_rms(x, mask_f):
            return (x.pow(2) * mask_f).sum().div(mask_f.sum().clamp(min=1)).sqrt().item()
        
        # Helper: compute Jaccard@10
        def compute_jaccard_at_k(pred, target, mask, k=10):
            jaccard_sum = 0.0
            jaccard_count = 0
            B = pred.shape[0]
            for b in range(min(4, B)):
                m_b = mask[b].bool()
                n_valid = int(m_b.sum().item())
                if n_valid < k + 5:
                    continue
                
                pred_b = pred[b, m_b]
                tgt_b = target[b, m_b]
                
                D_pred_b = torch.cdist(pred_b, pred_b)
                D_tgt_b = torch.cdist(tgt_b, tgt_b)
                
                k_j = min(k, n_valid - 1)
                _, knn_pred = D_pred_b.topk(k_j + 1, largest=False)
                _, knn_tgt = D_tgt_b.topk(k_j + 1, largest=False)
                
                knn_pred = knn_pred[:, 1:]  # Exclude self
                knn_tgt = knn_tgt[:, 1:]
                
                for i in range(n_valid):
                    set_pred = set(knn_pred[i].tolist())
                    set_tgt = set(knn_tgt[i].tolist())
                    inter = len(set_pred & set_tgt)
                    union = len(set_pred | set_tgt)
                    if union > 0:
                        jaccard_sum += inter / union
                        jaccard_count += 1
            
            return jaccard_sum / max(jaccard_count, 1)
        
        # Fixed seed for reproducible noise
        rng_state = torch.get_rng_state()
        
        # =====================================================================
        # TEST 1: Is σ-conditioning actually being used?
        # =====================================================================
        # print(f"\n{'='*70}")
        # print(f"TEST 1 (A/B 1): Is σ-conditioning actually being used?")
        # print(f"{'='*70}")
        # print(f"  A = Normal σ/time signal")
        # print(f"  B = Force constant time (σ=0.30 for time embedding only)")
        # print(f"  EDM preconditioning (c_skip/c_in/c_out) uses TRUE σ in both cases")
        # print()
        # print(f"  {'σ':>8} | {'scale_r_A':>10} | {'scale_r_B':>10} | {'Jacc_A':>8} | {'Jacc_B':>8} | {'mean_diff':>10}")
        # print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*10}")
        
        CONSTANT_SIGMA_FOR_TIME = 0.30  # The constant σ used for time embedding in test B
        
        for sigma_val in fixed_eval_sigmas:
            torch.manual_seed(42 + int(sigma_val * 1000))
            
            sigma_fixed = torch.full((B_fixed,), sigma_val, device=device)
            sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
            
            eps_fixed = torch.randn_like(V_target_fixed)
            V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
            V_t_fixed = V_t_fixed * mask_f
            
            # A: Normal forward
            x0_pred_A = score_net_unwrapped.forward_edm(
                V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
                sigma_data, self_cond=None
            )
            if isinstance(x0_pred_A, tuple):
                x0_pred_A = x0_pred_A[0]
            
            # B: Force constant time embedding, but use TRUE sigma for preconditioning
            sigma_const_for_time = torch.full((B_fixed,), CONSTANT_SIGMA_FOR_TIME, device=device)
            
            # Manually compute preconditioning with TRUE sigma
            c_skip, c_out, c_in, _ = uet.edm_precond(sigma_fixed, sigma_data)
            
            # Compute c_noise with CONSTANT sigma (this is what goes to time embedding)
            c_noise_const = sigma_const_for_time.log() / 4  # Same formula as edm_precond
            c_noise_const = c_noise_const.view(-1, 1)
            
            # Center x
            x_c, _ = uet.center_only(V_t_fixed, mask_fixed)
            x_in = c_in * x_c
            
            # Call forward() with constant c_noise
            F_x_B = score_net_unwrapped.forward(
                x_in, 
                c_noise_const,  # CONSTANT time signal
                H_fixed, 
                mask_fixed, 
                self_cond=None, 
                sigma_raw=sigma_fixed,  # TRUE sigma for gating
                x_raw=x_c,
                c_in=c_in
            )
            if isinstance(F_x_B, tuple):
                F_x_B = F_x_B[0]
            
            # Apply EDM denoiser formula with TRUE preconditioning
            x0_pred_B = c_skip * x_c + c_out * F_x_B
            x0_pred_B = x0_pred_B * mask_f
            
            # Compute metrics
            V_pred_A_c, _ = uet.center_only(x0_pred_A, mask_fixed)
            V_pred_B_c, _ = uet.center_only(x0_pred_B, mask_fixed)
            V_tgt_c, _ = uet.center_only(V_target_fixed, mask_fixed)
            
            rms_pred_A = masked_rms(V_pred_A_c, mask_f)
            rms_pred_B = masked_rms(V_pred_B_c, mask_f)
            rms_tgt = masked_rms(V_tgt_c, mask_f)
            
            scale_r_A = rms_pred_A / max(rms_tgt, 1e-8)
            scale_r_B = rms_pred_B / max(rms_tgt, 1e-8)
            
            jacc_A = compute_jaccard_at_k(x0_pred_A, V_target_fixed, mask_fixed, k=10)
            jacc_B = compute_jaccard_at_k(x0_pred_B, V_target_fixed, mask_fixed, k=10)
            
            # Mean absolute difference between A and B predictions
            diff_AB = (x0_pred_A - x0_pred_B).abs()
            mean_abs_diff = (diff_AB * mask_f).sum() / valid_count
            mean_abs_diff = mean_abs_diff.item()
            
            print(f"  {sigma_val:8.3f} | {scale_r_A:10.4f} | {scale_r_B:10.4f} | {jacc_A:8.4f} | {jacc_B:8.4f} | {mean_abs_diff:10.6f}")
        
        # print()
        # print(f"  INTERPRETATION:")
        # print(f"    If mean_diff is SMALL (< 0.01) across all σ:")
        # print(f"      → Time conditioning is NOT being used effectively (or broken)")
        # print(f"    If mean_diff is LARGE (> 0.05) especially at mid/high σ:")
        # print(f"      → Time conditioning IS alive and affecting predictions")
        
        # # =====================================================================
        # # TEST 2: Decomposition - Is shrink coming from learned branch?
        # # =====================================================================
        # print(f"\n{'='*70}")
        # print(f"TEST 2 (Decomposition): Is shrink from the learned branch?")
        # print(f"{'='*70}")
        # print(f"  x0_pred = c_skip * x_c + c_out * F_x")
        # print(f"  x_skip = c_skip * x_c")
        # print(f"  x_out = c_out * F_x (learned contribution)")
        # print()
        # print(f"  {'σ':>8} | {'c_skip':>8} | {'c_out':>8} | {'rms_skip':>10} | {'rms_out':>10} | {'rms_pred':>10} | {'rms_tgt':>10} | {'out/tgt':>8}")
        # print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
        
        for sigma_val in fixed_eval_sigmas:
            torch.manual_seed(42 + int(sigma_val * 1000))
            
            sigma_fixed = torch.full((B_fixed,), sigma_val, device=device)
            sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
            
            eps_fixed = torch.randn_like(V_target_fixed)
            V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
            V_t_fixed = V_t_fixed * mask_f
            
            # Compute preconditioning
            c_skip, c_out, c_in, c_noise = uet.edm_precond(sigma_fixed, sigma_data)
            
            # Center x
            x_c, _ = uet.center_only(V_t_fixed, mask_fixed)
            x_in = c_in * x_c
            
            # Get F_x from network
            F_x = score_net_unwrapped.forward(
                x_in, c_noise, H_fixed, mask_fixed, 
                self_cond=None, sigma_raw=sigma_fixed, x_raw=x_c, c_in=c_in
            )
            if isinstance(F_x, tuple):
                F_x = F_x[0]
            
            # Decompose
            x_skip = c_skip * x_c * mask_f
            x_out = c_out * F_x * mask_f
            x0_pred = (x_skip + x_out) * mask_f
            
            # Center target for comparison
            V_tgt_c, _ = uet.center_only(V_target_fixed, mask_fixed)
            V_tgt_c = V_tgt_c * mask_f
            
            # RMS values
            rms_skip = masked_rms(x_skip, mask_f)
            rms_out = masked_rms(x_out, mask_f)
            rms_pred = masked_rms(x0_pred, mask_f)
            rms_tgt = masked_rms(V_tgt_c, mask_f)
            
            out_over_tgt = rms_out / max(rms_tgt, 1e-8)
            
            # c_skip and c_out values (take mean over batch)
            c_skip_val = c_skip.mean().item()
            c_out_val = c_out.mean().item()
            
            print(f"  {sigma_val:8.3f} | {c_skip_val:8.4f} | {c_out_val:8.4f} | {rms_skip:10.4f} | {rms_out:10.4f} | {rms_pred:10.4f} | {rms_tgt:10.4f} | {out_over_tgt:8.4f}")
        
        # print()
        # print(f"  INTERPRETATION:")
        # print(f"    As σ increases, c_skip → 0 and c_out → σ_data")
        # print(f"    If rms(x_out)/rms(target) is << 1 at mid/high σ:")
        # print(f"      → Learned branch (F_x) is under-scaled, causing shrink")
        # print(f"    If rms(x_out)/rms(target) ≈ 1 but x0_pred still shrinks:")
        # print(f"      → Bug in combination or centering")
        
        # # =====================================================================
        # # TEST 3: Does a different time signal change behavior?
        # # =====================================================================
        # print(f"\n{'='*70}")
        # print(f"TEST 3 (A/B 6): Does a different time signal change behavior?")
        # print(f"{'='*70}")
        # print(f"  A = Current: c_noise = log(σ)/4")
        # print(f"  B = Alternative: c_noise = log(σ) (unscaled)")
        # print()
        # print(f"  {'σ':>8} | {'scale_r_A':>10} | {'scale_r_B':>10} | {'Jacc_A':>8} | {'Jacc_B':>8} | {'mean_diff':>10}")
        # print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*10}")
        
        for sigma_val in fixed_eval_sigmas:
            torch.manual_seed(42 + int(sigma_val * 1000))
            
            sigma_fixed = torch.full((B_fixed,), sigma_val, device=device)
            sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
            
            eps_fixed = torch.randn_like(V_target_fixed)
            V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
            V_t_fixed = V_t_fixed * mask_f
            
            # A: Normal forward (uses c_noise = log(σ)/4)
            x0_pred_A = score_net_unwrapped.forward_edm(
                V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
                sigma_data, self_cond=None
            )
            if isinstance(x0_pred_A, tuple):
                x0_pred_A = x0_pred_A[0]
            
            # B: Alternative time signal (c_noise = log(σ), unscaled)
            c_skip, c_out, c_in, _ = uet.edm_precond(sigma_fixed, sigma_data)
            
            # Alternative c_noise: just log(σ), no /4 scaling
            c_noise_alt = sigma_fixed.log().view(-1, 1)  # Note: no /4
            
            x_c, _ = uet.center_only(V_t_fixed, mask_fixed)
            x_in = c_in * x_c
            
            F_x_B = score_net_unwrapped.forward(
                x_in, 
                c_noise_alt,  # Alternative time signal
                H_fixed, 
                mask_fixed, 
                self_cond=None, 
                sigma_raw=sigma_fixed,
                x_raw=x_c,
                c_in=c_in
            )
            if isinstance(F_x_B, tuple):
                F_x_B = F_x_B[0]
            
            x0_pred_B = c_skip * x_c + c_out * F_x_B
            x0_pred_B = x0_pred_B * mask_f
            
            # Metrics
            V_pred_A_c, _ = uet.center_only(x0_pred_A, mask_fixed)
            V_pred_B_c, _ = uet.center_only(x0_pred_B, mask_fixed)
            V_tgt_c, _ = uet.center_only(V_target_fixed, mask_fixed)
            
            scale_r_A = masked_rms(V_pred_A_c, mask_f) / max(masked_rms(V_tgt_c, mask_f), 1e-8)
            scale_r_B = masked_rms(V_pred_B_c, mask_f) / max(masked_rms(V_tgt_c, mask_f), 1e-8)
            
            jacc_A = compute_jaccard_at_k(x0_pred_A, V_target_fixed, mask_fixed, k=10)
            jacc_B = compute_jaccard_at_k(x0_pred_B, V_target_fixed, mask_fixed, k=10)
            
            diff_AB = (x0_pred_A - x0_pred_B).abs()
            mean_abs_diff = (diff_AB * mask_f).sum() / valid_count
            mean_abs_diff = mean_abs_diff.item()
            
            print(f"  {sigma_val:8.3f} | {scale_r_A:10.4f} | {scale_r_B:10.4f} | {jacc_A:8.4f} | {jacc_B:8.4f} | {mean_abs_diff:10.6f}")
        
        # print()
        # print(f"  INTERPRETATION:")
        # print(f"    If A and B show LARGE differences in metrics:")
        # print(f"      → Time signal scaling matters; current /4 scaling may be sub-optimal")
        # print(f"    If A and B are nearly IDENTICAL:")
        # print(f"      → Either time conditioning path is broken, or both signals saturate similarly")


        #         # =====================================================================
        # # TEST 4: Is conditioning H actually being used at high σ?
        # # =====================================================================
        # print(f"\n{'='*70}")
        # print(f"TEST 4: Is conditioning H actually being used?")
        # print(f"{'='*70}")
        # print(f"  Normal   = Standard H from context encoder")
        # print(f"  Zero-H   = Replace H with zeros (ablation)")
        # print(f"  Shuffle-H = Permute H across batch (wrong conditioning)")
        # print()
        
        # # First, let's log H statistics to understand what we're working with
        # print(f"  H STATISTICS:")
        # H_mean = (H_fixed * mask_f).sum() / valid_count
        # H_std = ((H_fixed - H_mean).pow(2) * mask_f).sum().div(valid_count).sqrt()
        # H_norm_per_sample = (H_fixed.pow(2) * mask_f).sum(dim=(1,2)).sqrt()
        # H_mean_norm = H_norm_per_sample.mean()
        # print(f"    H_mean: {H_mean.item():.6f}")
        # print(f"    H_std:  {H_std.item():.6f}")
        # print(f"    H_mean_norm (per sample): {H_mean_norm.item():.4f}")
        # print(f"    H shape: {H_fixed.shape}")
        # print()
        
        # Create ablated versions of H
        H_zero = torch.zeros_like(H_fixed)
        
        # Shuffle H across batch dimension (if B > 1)
        if B_fixed > 1:
            perm = torch.randperm(B_fixed, device=device)
            # Make sure permutation actually shuffles (not identity)
            while (perm == torch.arange(B_fixed, device=device)).all():
                perm = torch.randperm(B_fixed, device=device)
            H_shuffle = H_fixed[perm]
        else:
            # If only 1 sample, shuffle within the sample (permute nodes)
            n_valid = int(mask_fixed[0].sum().item())
            node_perm = torch.randperm(n_valid, device=device)
            H_shuffle = H_fixed.clone()
            H_shuffle[0, :n_valid] = H_fixed[0, node_perm]
        
        print(f"  {'σ':>8} | {'scale_N':>8} | {'scale_Z':>8} | {'scale_S':>8} | {'Jacc_N':>7} | {'Jacc_Z':>7} | {'Jacc_S':>7} | {'rel_diff_Z':>10} | {'rel_diff_S':>10}")
        print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*10} | {'-'*10}")
        
        # Storage for per-sigma analysis
        sigma_analysis = []
        
        for sigma_val in fixed_eval_sigmas:
            torch.manual_seed(42 + int(sigma_val * 1000))
            
            sigma_fixed = torch.full((B_fixed,), sigma_val, device=device)
            sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
            
            eps_fixed = torch.randn_like(V_target_fixed)
            V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
            V_t_fixed = V_t_fixed * mask_f
            
            # Normal: with real H
            x0_pred_normal = score_net_unwrapped.forward_edm(
                V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
                sigma_data, self_cond=None
            )
            if isinstance(x0_pred_normal, tuple):
                x0_pred_normal = x0_pred_normal[0]
            
            # Zero-H: with H = 0
            x0_pred_zero = score_net_unwrapped.forward_edm(
                V_t_fixed, sigma_fixed, H_zero, mask_fixed, 
                sigma_data, self_cond=None
            )
            if isinstance(x0_pred_zero, tuple):
                x0_pred_zero = x0_pred_zero[0]
            
            # Shuffle-H: with shuffled H
            x0_pred_shuffle = score_net_unwrapped.forward_edm(
                V_t_fixed, sigma_fixed, H_shuffle, mask_fixed, 
                sigma_data, self_cond=None
            )
            if isinstance(x0_pred_shuffle, tuple):
                x0_pred_shuffle = x0_pred_shuffle[0]
            
            # Compute metrics for each
            V_tgt_c, _ = uet.center_only(V_target_fixed, mask_fixed)
            rms_tgt = masked_rms(V_tgt_c, mask_f)
            
            # Normal
            V_pred_N_c, _ = uet.center_only(x0_pred_normal, mask_fixed)
            scale_N = masked_rms(V_pred_N_c, mask_f) / max(rms_tgt, 1e-8)
            jacc_N = compute_jaccard_at_k(x0_pred_normal, V_target_fixed, mask_fixed, k=10)
            
            # Zero-H
            V_pred_Z_c, _ = uet.center_only(x0_pred_zero, mask_fixed)
            scale_Z = masked_rms(V_pred_Z_c, mask_f) / max(rms_tgt, 1e-8)
            jacc_Z = compute_jaccard_at_k(x0_pred_zero, V_target_fixed, mask_fixed, k=10)
            
            # Shuffle-H
            V_pred_S_c, _ = uet.center_only(x0_pred_shuffle, mask_fixed)
            scale_S = masked_rms(V_pred_S_c, mask_f) / max(rms_tgt, 1e-8)
            jacc_S = compute_jaccard_at_k(x0_pred_shuffle, V_target_fixed, mask_fixed, k=10)
            
            # Relative difference: ||x_normal - x_ablated|| / ||x_normal||
            rms_normal = masked_rms(x0_pred_normal, mask_f)
            
            diff_zero = (x0_pred_normal - x0_pred_zero).pow(2)
            rel_diff_Z = masked_rms(x0_pred_normal - x0_pred_zero, mask_f) / max(rms_normal, 1e-8)
            
            diff_shuffle = (x0_pred_normal - x0_pred_shuffle).pow(2)
            rel_diff_S = masked_rms(x0_pred_normal - x0_pred_shuffle, mask_f) / max(rms_normal, 1e-8)
            
            print(f"  {sigma_val:8.3f} | {scale_N:8.4f} | {scale_Z:8.4f} | {scale_S:8.4f} | {jacc_N:7.4f} | {jacc_Z:7.4f} | {jacc_S:7.4f} | {rel_diff_Z:10.4f} | {rel_diff_S:10.4f}")
            
            # Store for detailed analysis
            sigma_analysis.append({
                'sigma': sigma_val,
                'scale_N': scale_N, 'scale_Z': scale_Z, 'scale_S': scale_S,
                'jacc_N': jacc_N, 'jacc_Z': jacc_Z, 'jacc_S': jacc_S,
                'rel_diff_Z': rel_diff_Z, 'rel_diff_S': rel_diff_S,
            })
        
        # Detailed analysis
        print()
        print(f"  DETAILED ANALYSIS:")
        print(f"  {'-'*60}")
        
        # Check if conditioning matters at high sigma
        high_sigma_entries = [e for e in sigma_analysis if e['sigma'] >= 0.70]
        low_sigma_entries = [e for e in sigma_analysis if e['sigma'] <= 0.15]
        
        if high_sigma_entries:
            avg_rel_diff_Z_high = sum(e['rel_diff_Z'] for e in high_sigma_entries) / len(high_sigma_entries)
            avg_rel_diff_S_high = sum(e['rel_diff_S'] for e in high_sigma_entries) / len(high_sigma_entries)
            avg_jacc_drop_Z_high = sum(e['jacc_N'] - e['jacc_Z'] for e in high_sigma_entries) / len(high_sigma_entries)
            avg_jacc_drop_S_high = sum(e['jacc_N'] - e['jacc_S'] for e in high_sigma_entries) / len(high_sigma_entries)
            
            print(f"  HIGH σ (≥0.70) - This is where c_skip→0, learned branch dominates:")
            print(f"    Avg rel_diff (Zero-H):    {avg_rel_diff_Z_high:.4f}")
            print(f"    Avg rel_diff (Shuffle-H): {avg_rel_diff_S_high:.4f}")
            print(f"    Avg Jacc drop (Zero-H):   {avg_jacc_drop_Z_high:.4f}")
            print(f"    Avg Jacc drop (Shuffle-H):{avg_jacc_drop_S_high:.4f}")
            
            if avg_rel_diff_Z_high < 0.05 and avg_rel_diff_S_high < 0.05:
                print(f"    ⚠️  PROBLEM: H has MINIMAL effect at high σ!")
                print(f"       → Conditioning pipeline may be broken or H is not reaching the network")
            elif avg_rel_diff_Z_high > 0.10 or avg_rel_diff_S_high > 0.10:
                print(f"    ✓ H IS being used at high σ (rel_diff > 0.10)")
                print(f"       → Issue is likely objective/weighting, not conditioning pipeline")
            else:
                print(f"    ⚡ H has MODERATE effect at high σ (0.05 < rel_diff < 0.10)")
                print(f"       → Conditioning works but may be weak")
        
        if low_sigma_entries:
            avg_rel_diff_Z_low = sum(e['rel_diff_Z'] for e in low_sigma_entries) / len(low_sigma_entries)
            avg_rel_diff_S_low = sum(e['rel_diff_S'] for e in low_sigma_entries) / len(low_sigma_entries)
            
            print(f"\n  LOW σ (≤0.15) - Skip connection dominates here:")
            print(f"    Avg rel_diff (Zero-H):    {avg_rel_diff_Z_low:.4f}")
            print(f"    Avg rel_diff (Shuffle-H): {avg_rel_diff_S_low:.4f}")
        
        # Check scale collapse pattern
        print(f"\n  SCALE COLLAPSE PATTERN:")
        for e in sigma_analysis:
            collapse_indicator = ""
            if e['scale_N'] < 0.80:
                collapse_indicator = "⚠️ SHRINK"
            elif e['scale_N'] > 1.20:
                collapse_indicator = "⚠️ EXPAND"
            else:
                collapse_indicator = "✓ OK"
            
            # Does Zero-H make it worse or better?
            zero_effect = ""
            if abs(e['scale_Z'] - 1.0) < abs(e['scale_N'] - 1.0):
                zero_effect = "(Zero-H closer to 1.0!)"
            elif abs(e['scale_Z'] - 1.0) > abs(e['scale_N'] - 1.0) * 1.5:
                zero_effect = "(Zero-H makes it worse)"
            
            print(f"    σ={e['sigma']:.2f}: scale_N={e['scale_N']:.3f} {collapse_indicator} {zero_effect}")
        
        print()
        print(f"  INTERPRETATION GUIDE:")
        print(f"  {'-'*60}")
        print(f"  If rel_diff is SMALL (<0.05) at high σ:")
        print(f"    → Model ignores H when noise is high")
        print(f"    → Focus on: conditioning injection strength, H normalization")
        print(f"  If rel_diff is LARGE (>0.10) but scale still collapses:")
        print(f"    → H is used, but F_x output is under-scaled")
        print(f"    → Focus on: scale enforcement loss, output head initialization")
        print(f"  If Jacc drops significantly with Zero-H/Shuffle-H:")
        print(f"    → H carries useful geometric info")
        print(f"  If Jacc stays same with Zero-H/Shuffle-H:")
        print(f"    → H is not helping geometry prediction")
        
        # =====================================================================
        # TEST 4B: Decomposition with H ablation
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"TEST 4B: F_x magnitude with vs without H")
        print(f"{'='*70}")
        print(f"  Shows whether F_x (learned branch) changes magnitude when H is ablated")
        print()
        print(f"  {'σ':>8} | {'rms_Fx_N':>10} | {'rms_Fx_Z':>10} | {'rms_Fx_S':>10} | {'Fx_ratio_Z':>10} | {'Fx_ratio_S':>10}")
        print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
        
        for sigma_val in fixed_eval_sigmas:
            torch.manual_seed(42 + int(sigma_val * 1000))
            
            sigma_fixed = torch.full((B_fixed,), sigma_val, device=device)
            sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
            
            eps_fixed = torch.randn_like(V_target_fixed)
            V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
            V_t_fixed = V_t_fixed * mask_f
            
            # Compute preconditioning
            c_skip, c_out, c_in, c_noise = uet.edm_precond(sigma_fixed, sigma_data)
            
            x_c, _ = uet.center_only(V_t_fixed, mask_fixed)
            x_in = c_in * x_c
            
            # Get F_x with normal H
            F_x_N = score_net_unwrapped.forward(
                x_in, c_noise, H_fixed, mask_fixed, 
                self_cond=None, sigma_raw=sigma_fixed, x_raw=x_c, c_in=c_in
            )
            if isinstance(F_x_N, tuple):
                F_x_N = F_x_N[0]
            
            # Get F_x with Zero H
            F_x_Z = score_net_unwrapped.forward(
                x_in, c_noise, H_zero, mask_fixed, 
                self_cond=None, sigma_raw=sigma_fixed, x_raw=x_c, c_in=c_in
            )
            if isinstance(F_x_Z, tuple):
                F_x_Z = F_x_Z[0]
            
            # Get F_x with Shuffle H
            F_x_S = score_net_unwrapped.forward(
                x_in, c_noise, H_shuffle, mask_fixed, 
                self_cond=None, sigma_raw=sigma_fixed, x_raw=x_c, c_in=c_in
            )
            if isinstance(F_x_S, tuple):
                F_x_S = F_x_S[0]
            
            rms_Fx_N = masked_rms(F_x_N, mask_f)
            rms_Fx_Z = masked_rms(F_x_Z, mask_f)
            rms_Fx_S = masked_rms(F_x_S, mask_f)
            
            Fx_ratio_Z = rms_Fx_Z / max(rms_Fx_N, 1e-8)
            Fx_ratio_S = rms_Fx_S / max(rms_Fx_N, 1e-8)
            
            print(f"  {sigma_val:8.3f} | {rms_Fx_N:10.4f} | {rms_Fx_Z:10.4f} | {rms_Fx_S:10.4f} | {Fx_ratio_Z:10.4f} | {Fx_ratio_S:10.4f}")
        
        print()
        print(f"  INTERPRETATION:")
        print(f"    If Fx_ratio ≈ 1.0: F_x magnitude doesn't change when H is ablated")
        print(f"      → Network may be ignoring H in the learned branch")
        print(f"    If Fx_ratio significantly != 1.0: F_x magnitude depends on H")
        print(f"      → Conditioning IS affecting the learned output magnitude")
        
        # =====================================================================
        # TEST 4C: Per-sample conditioning effect
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"TEST 4C: Per-sample conditioning effect at σ=1.20")
        print(f"{'='*70}")
        print(f"  Shows if effect varies across samples (some use H, some don't)")
        print()
        
        sigma_test = 1.20  # High sigma where learned branch dominates
        torch.manual_seed(42 + int(sigma_test * 1000))
        
        sigma_fixed = torch.full((B_fixed,), sigma_test, device=device)
        sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
        
        eps_fixed = torch.randn_like(V_target_fixed)
        V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
        V_t_fixed = V_t_fixed * mask_f
        
        x0_pred_normal = score_net_unwrapped.forward_edm(
            V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
            sigma_data, self_cond=None
        )
        if isinstance(x0_pred_normal, tuple):
            x0_pred_normal = x0_pred_normal[0]
        
        x0_pred_zero = score_net_unwrapped.forward_edm(
            V_t_fixed, sigma_fixed, H_zero, mask_fixed, 
            sigma_data, self_cond=None
        )
        if isinstance(x0_pred_zero, tuple):
            x0_pred_zero = x0_pred_zero[0]
        
        print(f"  {'Sample':>8} | {'n_valid':>8} | {'rms_pred_N':>10} | {'rms_pred_Z':>10} | {'rel_diff':>10} | {'scale_N':>8}")
        print(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
        
        V_tgt_c, _ = uet.center_only(V_target_fixed, mask_fixed)
        
        for b in range(min(B_fixed, 8)):  # Show up to 8 samples
            m_b = mask_fixed[b].bool()
            n_valid = int(m_b.sum().item())
            
            if n_valid < 5:
                continue
            
            pred_N_b = x0_pred_normal[b, m_b]
            pred_Z_b = x0_pred_zero[b, m_b]
            tgt_b = V_tgt_c[b, m_b]
            
            rms_pred_N = pred_N_b.pow(2).mean().sqrt().item()
            rms_pred_Z = pred_Z_b.pow(2).mean().sqrt().item()
            rms_tgt_b = tgt_b.pow(2).mean().sqrt().item()
            
            diff_b = (pred_N_b - pred_Z_b).pow(2).mean().sqrt().item()
            rel_diff_b = diff_b / max(rms_pred_N, 1e-8)
            scale_N_b = rms_pred_N / max(rms_tgt_b, 1e-8)
            
            indicator = ""
            if rel_diff_b < 0.02:
                indicator = "⚠️ H ignored"
            elif rel_diff_b > 0.15:
                indicator = "✓ H matters"
            
            print(f"  {b:8d} | {n_valid:8d} | {rms_pred_N:10.4f} | {rms_pred_Z:10.4f} | {rel_diff_b:10.4f} | {scale_N_b:8.4f} {indicator}")
        
        print()
        print(f"  If rel_diff varies a lot across samples:")
        print(f"    → Some samples may have more informative H than others")
        print(f"  If rel_diff is uniformly low:")
        print(f"    → Systematic issue with H conditioning pathway")

        
        # Restore RNG state
        torch.set_rng_state(rng_state)
    
    # Restore training mode
    if was_training_score:
        score_net.train()
    if was_training_ctx:
        context_encoder.train()
    
    print(f"\n{'='*80}")
    print(f"END A/B TESTS")
    print(f"{'='*80}\n")


def run_probe_eval_multi_sigma(
    probe_state: Dict,
    score_net: nn.Module,
    context_encoder: nn.Module,
    sigma_data: float,
    device: str,
    sigma_list: List[float],
    seed: int = 42,
    global_step: int = 0,
    epoch: int = 0,
    fabric = None,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate the model on a fixed probe batch at MULTIPLE sigma levels.
    
    This tests whether the model can recover structure at different noise levels.
    High-sigma probes test "structure formation from far away" which is critical
    for understanding inference failures.
    
    Args:
        probe_state: dict with 'Z_set', 'V_target', 'mask', 'knn_spatial'
        score_net: diffusion score network
        context_encoder: context encoder
        sigma_data: EDM sigma_data parameter
        device: torch device string
        sigma_list: list of sigma values to probe (e.g., [0.2, 0.8, 1.5, 2.0, 2.5, 3.0])
        seed: random seed for reproducible noise
        global_step: current training step
        epoch: current epoch
        fabric: optional Fabric object
    
    Returns:
        Dict mapping sigma -> metrics dict
    """
    if not probe_state.get('batch_captured', False):
        return {}
    
    # Move probe tensors to device
    Z_probe = probe_state['Z_set'].unsqueeze(0).to(device)  # (1, N, D_z)
    V_target = probe_state['V_target'].unsqueeze(0).to(device)  # (1, N, D)
    mask_probe = probe_state['mask'].unsqueeze(0).to(device)  # (1, N)
    n_probe = probe_state['n_probe']
    
    # Get model in eval mode
    was_training_score = score_net.training
    was_training_ctx = context_encoder.training
    score_net.eval()
    context_encoder.eval()
    
    # Unwrap if DDP wrapped
    score_net_unwrapped = score_net.module if hasattr(score_net, 'module') else score_net
    
    all_results = {}
    
    with torch.no_grad():
        B, N, D = V_target.shape
        Z_probe = apply_z_ln(Z_probe, context_encoder)

        
        # Get context (same for all sigmas)
        H = context_encoder(Z_probe, mask_probe)
        
        # Center V_target for fair comparison (do once)
        V_target_centered, _ = uet.center_only(V_target, mask_probe)
        
        # ========== GT-MARGIN: Compute ground truth margin (once) ==========
        # This tells us if low margin is inherent to the dataset geometry
        V_t_flat = V_target_centered[0, mask_probe[0].bool()]  # (n_valid, D)
        D_gt = torch.cdist(V_t_flat, V_t_flat)
        D_gt.fill_diagonal_(float('inf'))
        D_gt_sorted, _ = D_gt.sort(dim=1)
        
        n_valid = V_t_flat.shape[0]
        if n_valid >= 12:
            d_10_gt = D_gt_sorted[:, 9]   # 10th nearest
            d_11_gt = D_gt_sorted[:, 10]  # 11th nearest
            gt_margin = (d_11_gt / (d_10_gt + 1e-8))
            gt_margin_p50 = gt_margin.median().item()
            gt_margin_p90 = gt_margin.quantile(0.9).item()
        else:
            gt_margin_p50 = 0.0
            gt_margin_p90 = 0.0
        
        # Store GT margin in results (only need to report once)
        all_results['gt_margin'] = {
            'd11_d10_p50': gt_margin_p50,
            'd11_d10_p90': gt_margin_p90
        }
        
        # ========== Loop over sigma levels ==========
        for sigma_val in sigma_list:
            # Generate deterministic noise (same noise pattern for each sigma, scaled differently)
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            eps = torch.randn(B, N, D, generator=g, device=device)
            
            # Build noisy input at this sigma
            sigma = torch.tensor([sigma_val], device=device)
            V_t = V_target + sigma.view(1, 1, 1) * eps
            
            # Center V_t (same preprocessing as training)
            V_t_centered, _ = uet.center_only(V_t, mask_probe)
            
            # Forward pass to get x0_pred
            result = score_net_unwrapped.forward_edm(
                V_t_centered, sigma, H, mask_probe, sigma_data, 
                self_cond=None, return_debug=False
            )
            
            if isinstance(result, tuple):
                x0_pred = result[0]
            else:
                x0_pred = result
            
            # Center x0_pred (same as training V_geom computation)
            x0_pred_centered, _ = uet.center_only(x0_pred, mask_probe)
            
            # Compute probe metrics
            probe_metrics = uet.compute_probe_metrics(
                x0_pred_centered, 
                V_target_centered, 
                mask_probe,
                knn_spatial=None,
                k=10
            )
            
            probe_metrics['sigma'] = sigma_val
            probe_metrics['n_probe'] = n_probe
            
            all_results[sigma_val] = probe_metrics
    
    # Restore training mode
    if was_training_score:
        score_net.train()
    if was_training_ctx:
        context_encoder.train()
    
    return all_results


def print_probe_results_multi_sigma(all_results: Dict, global_step: int, epoch: int):
    """Print probe results for all sigma levels with standardized tags."""
    if not all_results:
        return
    
    # Print GT-MARGIN first (only once, not per sigma)
    if 'gt_margin' in all_results:
        gt_m = all_results['gt_margin']
        print(f"\n[GT-MARGIN] epoch={epoch} d11_d10_p50={gt_m['d11_d10_p50']:.4f} "
              f"d11_d10_p90={gt_m['d11_d10_p90']:.4f}")
        if gt_m['d11_d10_p50'] < 1.05:
            print(f"[GT-MARGIN] ⚠️ GT itself has tiny margins - kNN@10 is inherently fragile for this geometry")
    
    # Print per-sigma results
    sigma_keys = sorted([k for k in all_results.keys() if isinstance(k, float)])
    
    for sigma_val in sigma_keys:
        results = all_results[sigma_val]
        
        # [PROBE-KNN] tag
        print(f"\n[PROBE-KNN] step={global_step} epoch={epoch} sigma={sigma_val:.3f} "
              f"knn10={results.get('knn10_overlap', 0):.4f} "
              f"knn20={results.get('knn20_overlap', 0):.4f} "
              f"knn50={results.get('knn50_overlap', 0):.4f}")
        
        # [PROBE-NEARMISS] tag
        print(f"[PROBE-NEARMISS] step={global_step} epoch={epoch} sigma={sigma_val:.3f} "
              f"ratio10_p50={results.get('nearmiss_ratio_p50', 0):.4f} "
              f"ratio10_p90={results.get('nearmiss_ratio_p90', 0):.4f}")
        
        # [PROBE-MARGIN] tag
        print(f"[PROBE-MARGIN] step={global_step} epoch={epoch} sigma={sigma_val:.3f} "
              f"d11_d10_p50={results.get('margin_d11_d10_p50', 0):.4f} "
              f"d11_d10_p90={results.get('margin_d11_d10_p90', 0):.4f}")
        
        # [PROBE-EDGE] tag
        print(f"[PROBE-EDGE] step={global_step} epoch={epoch} sigma={sigma_val:.3f} "
              f"spear_edge10={results.get('edge_spearman', 0):.4f}")
        
        # [PROBE-DIST] tag
        print(f"[PROBE-DIST] step={global_step} epoch={epoch} sigma={sigma_val:.3f} "
              f"spear={results.get('dist_spearman', 0):.4f} "
              f"pear={results.get('dist_pearson', 0):.4f}")
    
    # Summary interpretation
    print(f"\n[PROBE-SUMMARY] epoch={epoch}")
    
    # Compare low vs high sigma
    if 0.2 in all_results and 3.0 in all_results:
        low = all_results[0.2]
        high = all_results[3.0]
        
        knn_drop = low.get('knn10_overlap', 0) - high.get('knn10_overlap', 0)
        nearmiss_rise = high.get('nearmiss_ratio_p50', 0) - low.get('nearmiss_ratio_p50', 0)
        
        print(f"  σ=0.2 → σ=3.0: kNN10 drop={knn_drop:.4f}, nearmiss rise={nearmiss_rise:.4f}")
        
        if knn_drop > 0.3:
            print(f"  ⚠️ Large kNN degradation at high sigma - structure formation is weak")
        if nearmiss_rise > 1.0:
            print(f"  ⚠️ Near-miss explodes at high sigma - high-σ conditioning needs work")


def run_probe_open_loop_sample(
    probe_state: Dict,
    score_net: nn.Module,
    context_encoder: nn.Module,
    sigma_data: float,
    device: str,
    n_steps: int = 20,
    sigma_max: float = 3.0,
    sigma_min: float = 0.02,
    rho: float = 7.0,
    seed: int = 42,
    global_step: int = 0,
    epoch: int = 0,
    trace_sigmas: List[float] = None,
    fabric = None,
) -> Dict[str, float]:
    """
    Run open-loop EDM sampling on the fixed probe batch.
    
    This tests whether the model can generate correct structure from noise,
    not just denoise a noisy GT. This is the missing discriminator between
    "teacher-forced looks great" and "inference fails".
    
    Args:
        probe_state: dict with 'Z_set', 'V_target', 'mask'
        score_net: diffusion score network
        context_encoder: context encoder
        sigma_data: EDM sigma_data parameter
        device: torch device string
        n_steps: number of sampling steps
        sigma_max: starting sigma
        sigma_min: ending sigma
        rho: Karras schedule parameter
        seed: random seed for reproducible noise
        global_step: current training step
        epoch: current epoch
        trace_sigmas: list of sigmas at which to print intermediate metrics
        fabric: optional Fabric object
    
    Returns:
        Dict with final metrics
    
    Tags: [PROBE-SAMPLE-END], [PROBE-SAMPLE-TRACE]
    """
    if not probe_state.get('batch_captured', False):
        return {}
    
    if trace_sigmas is None:
        trace_sigmas = [3.0, 1.5, 0.8, 0.2]
    
    # Move probe tensors to device
    Z_probe = probe_state['Z_set'].unsqueeze(0).to(device)  # (1, N, D_z)
    V_target = probe_state['V_target'].unsqueeze(0).to(device)  # (1, N, D)
    mask_probe = probe_state['mask'].unsqueeze(0).to(device)  # (1, N)
    
    B, N, D = V_target.shape
    
    # Get model in eval mode
    was_training_score = score_net.training
    was_training_ctx = context_encoder.training
    score_net.eval()
    context_encoder.eval()
    
    # Unwrap if DDP wrapped
    score_net_unwrapped = score_net.module if hasattr(score_net, 'module') else score_net
    
    results = {
        'n_steps': n_steps,
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
    }
    trace_results = []
    
    with torch.no_grad():
        # ========== Build Karras sigma schedule ==========
        # sigmas[i] = (sigma_max^(1/rho) + i/(n_steps-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        step_indices = torch.linspace(0, 1, n_steps + 1, device=device)
        sigmas = (sigma_max ** (1/rho) + step_indices * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        sigmas[-1] = 0  # Final sigma is 0 (clean)

        Z_probe = apply_z_ln(Z_probe, context_encoder)
        # ========== Get context (once) ==========
        H = context_encoder(Z_probe, mask_probe)
        
        # ========== Initialize from noise ==========
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x = sigmas[0] * torch.randn(B, N, D, generator=g, device=device)
        
        # Center target for comparison
        V_target_centered, _ = uet.center_only(V_target, mask_probe)
        
        # ========== Sampling loop (Euler) ==========
        for i in range(n_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Center current state
            x_centered, _ = uet.center_only(x, mask_probe)
            
            # Get denoised prediction
            sigma_batch = sigma.view(1)
            denoised = score_net_unwrapped.forward_edm(
                x_centered, sigma_batch, H, mask_probe, sigma_data,
                self_cond=None, return_debug=False
            )
            
            if isinstance(denoised, tuple):
                denoised = denoised[0]
            
            # Center denoised
            denoised_centered, _ = uet.center_only(denoised, mask_probe)
            
            # ========== Check if we should trace at this sigma ==========
            sigma_val = sigma.item()
            for trace_sig in trace_sigmas:
                # Check if this is the closest step to the trace sigma
                if i == 0 or sigmas[i-1].item() >= trace_sig > sigma_val:
                    # Compute quick metrics at this point
                    trace_metrics = uet.compute_probe_metrics(
                        denoised_centered, V_target_centered, mask_probe, k=10
                    )
                    trace_results.append({
                        'sigma': trace_sig,
                        'knn10': trace_metrics.get('knn10_overlap', 0),
                        'nearmiss_p50': trace_metrics.get('nearmiss_ratio_p50', 0),
                    })
                    break
            
            # ========== Euler step ==========
            if sigma_next > 0:
                # d = (x - denoised) / sigma
                d = (x_centered - denoised_centered) / sigma
                # x_next = x + (sigma_next - sigma) * d
                x = x_centered + (sigma_next - sigma) * d
            else:
                # Final step: just use denoised
                x = denoised_centered
        
        # ========== Final metrics ==========
        # x is now the final sample (should be close to clean)
        x_final_centered, _ = uet.center_only(x, mask_probe)
        
        final_metrics = uet.compute_probe_metrics(
            x_final_centered, V_target_centered, mask_probe, k=10
        )
        
        results.update(final_metrics)
        results['trace'] = trace_results
    
    # Restore training mode
    if was_training_score:
        score_net.train()
    if was_training_ctx:
        context_encoder.train()
    
    return results


def print_probe_sample_results(results: Dict, global_step: int, epoch: int):
    """Print open-loop probe sampling results."""
    if not results:
        return
    
    n_steps = results.get('n_steps', 20)
    sigma_max = results.get('sigma_max', 3.0)
    sigma_min = results.get('sigma_min', 0.02)
    
    # Print trace (intermediate sigmas)
    if 'trace' in results and results['trace']:
        for t in results['trace']:
            print(f"[PROBE-SAMPLE-TRACE] epoch={epoch} sigma={t['sigma']:.3f} "
                  f"knn10={t['knn10']:.4f} nearmiss_p50={t['nearmiss_p50']:.4f}")
    
    # Print final
    print(f"\n[PROBE-SAMPLE-END] step={global_step} epoch={epoch} steps={n_steps} "
          f"sigmax={sigma_max:.2f} sigmin={sigma_min:.3f}")
    print(f"[PROBE-SAMPLE-END] knn10={results.get('knn10_overlap', 0):.4f} "
          f"nearmiss_p50={results.get('nearmiss_ratio_p50', 0):.4f} "
          f"margin_p50={results.get('margin_d11_d10_p50', 0):.4f} "
          f"edge_spear={results.get('edge_spearman', 0):.4f} "
          f"dist_spear={results.get('dist_spearman', 0):.4f}")
    
    # Interpretation
    knn10 = results.get('knn10_overlap', 0)
    nearmiss = results.get('nearmiss_ratio_p50', 0)
    
    if knn10 < 0.3 and nearmiss > 2.0:
        print(f"[PROBE-SAMPLE-END] ⚠️ Open-loop sampling fails! "
              f"Structure not recovered from noise.")
    elif knn10 > 0.5 and nearmiss < 1.5:
        print(f"[PROBE-SAMPLE-END] ✓ Open-loop sampling works well!")


def train_stageC_diffusion_generator(
    context_encoder: 'SetEncoderContext',
    generator: 'MetricSetGenerator',
    score_net: 'DiffusionScoreNet',
    st_dataset: Optional['STSetDataset'],
    sc_dataset: Optional['SCSetDataset'],
    prototype_bank: Dict,
    encoder: Optional['SharedEncoder'] = None,
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
    sigma_refine_max: float = None,  # Max sigma for EDM refinement (if None, uses 20.0 * sigma_data)
    #context augmentation params
    z_noise_std: float= 0.02, # guassian noise std (relative to feature rms)
    z_dropout_rate: float = 0.1, # feature dropout rate (5-20%)
    aug_prob: float = 0.5, #prob to apply augmentation per batch
    # ========== COMPETITOR TRAINING PARAMS (ChatGPT hypothesis test) ==========
    compete_train: bool = False,
    compete_n_extra: int = 128,
    compete_n_rand: int = 64,
    compete_n_hard: int = 64,
    compete_use_pos_closure: bool = True,
    compete_k_pos: int = 10,
    compete_expr_knn_k: int = 50,
    compete_anchor_only: bool = True,
    compete_diag_every: int = 200,
    # ========== NEW: Anchored training params ==========
    anchor_train: bool = False,
    anchor_p_uncond: float = 0.50,
    anchor_clamp_clean: bool = True,
    anchor_mask_score_loss: bool = True,
    anchor_pointweight_nca: bool = True,
    anchor_debug_every: int = 200,
    anchor_warmup_steps: int = 0,
    # ========== ANCHOR GEOMETRY LOSSES (structure loss anchor-awareness) ==========
    anchor_geom_losses: bool = True,  # Default ON when anchor_train is ON
    anchor_geom_mode: str = "clamp_only",  # "clamp_only" or "clamp_and_mask"
    anchor_geom_min_unknown: int = 8,
    anchor_geom_debug_every: int = 200,
    # ---- Resume Stage C ----
    resume_ckpt_path: Optional[str] = None,
    resume_reset_optimizer: bool = False,
    # ========== CONTEXT REPLACEMENT INVARIANCE ==========
    ctx_replace_variant: str = 'permute',
    ctx_loss_weight: float = 0.0,
    ctx_replace_p: float = 0.5,
    ctx_snr_thresh: float = 0.3,
    ctx_warmup_steps: int = 1000,
    ctx_debug_every: int = 100,
    # ========== SELF-CONDITIONING MODE ==========
    self_cond_mode: str = 'standard',
    # ========== PAIRED OVERLAP TRAINING (Candidate 1) ==========
    train_pair_overlap: bool = False,
    pair_overlap_alpha: float = 0.5,
    pair_overlap_min_I: int = 16,
    overlap_loss_weight_shape: float = 1.0,
    overlap_loss_weight_scale: float = 0.5,
    overlap_loss_weight_kl: float = 1.0,
    overlap_kl_tau: float = 0.5,
    overlap_sigma_thresh: float = 0.5,
    disable_ctx_loss_when_overlap: bool = True,
    overlap_debug_every: int = 100,

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

    # --- FIX: Validate and clamp tau_reference ---
    # tau_reference is in SQUARED-DISTANCE units (raw target space)
    # This matches the squared distances used in knn_nca_loss
    tau_reference = r_15_median ** 2

    # Safety clamp to prevent numerical issues
    tau_reference = max(1e-4, min(tau_reference, 1e2))

    # Make it a plain Python float (no gradients)
    tau_reference = float(tau_reference)

    # Debug logging
    print(f"[NCA TEMP] r_15_median={r_15_median:.4f}, tau_reference={tau_reference:.6f}")
    print(f"[NCA TEMP] This represents the squared 15th-NN distance in raw target space")


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
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    
    context_encoder = context_encoder.to(device).train()
    score_net = score_net.to(device).train()
    generator = generator.to(device).train()
    
    params = (
        list(context_encoder.parameters()) +
        list(generator.parameters()) +
        list(score_net.parameters())
    )
    # optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def safe_loss(loss, name, max_val=100.0, global_step=0, verbose=True):
        """Clamp loss and warn if it was extreme. Replace NaN/Inf with 0."""
        if not torch.isfinite(loss):
            if verbose and global_step % 10 == 0:
                print(f"[SAFE-LOSS] {name} is {'NaN' if torch.isnan(loss) else 'Inf'} at step {global_step}, replacing with 0")
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        
        if loss.abs() > max_val:
            if verbose and global_step % 10 == 0:
                print(f"[SAFE-LOSS] {name}={loss.item():.4f} exceeds {max_val}, clamping")
            return loss.clamp(-max_val, max_val)
        
        return loss

    # PATCH 8C: Separate LR for generator (needs bigger LR due to small gradients)
    # optimizer = torch.optim.AdamW([
    #     {"params": context_encoder.parameters(), "lr": lr},
    #     {"params": score_net.parameters(),       "lr": lr},
    #     {"params": generator.parameters(),       "lr": lr * 5.0},  # generator needs bigger LR
    # ], weight_decay=1e-4)

    # ==================== EXPERIMENT FLAGS (A/B Testing) ====================
    # These flags control experimental changes to fix high-σ learned-branch under-scaling.
    # 
    # REASONING: At high σ, c_skip ≈ 0, so x0_pred ≈ c_out * F_x (learned branch dominates).
    # But training signal for high-σ is weak because:
    #   1. EDM weight may not compensate enough
    #   2. Weight decay may shrink learned-branch output weights
    #   3. Low-σ samples dominate the gradient
    #
    # A/B 1: Disable weight decay for score_net to prevent amplitude suppression
    EXP_SCORE_WD0 = False
    
    # A/B 2a: Normalize score loss by sum of weights (removes global w scaling)
    # This is the KEY fix - it prevents low-σ from dominating the gradient
    EXP_SCORE_WNORM = True
    
    # A/B 2b: Boost high-noise samples in L_score (distribution shaping)
    EXP_SCORE_HI_BOOST = True
    SCORE_HI_TARGET_RATE = 0.25        # Top 25% noisiest samples
    SCORE_HI_BOOST_FACTOR = 4.0        # 4x weight on high-noise subset
    SCORE_HI_BOOST_WARMUP = 0       # Steps before boost starts
    SCORE_HI_BOOST_RAMP = 200          # Ramp boost in over this many steps

        
    # ==================== CHANGE 7: TAIL SAFETY PARAMETERS (Option 1 Change C) ====================
    TAIL_SAFETY_ENABLED = True        # Enable tail protection
    TAIL_QUANTILE = 0.95              # Top 5% of sigma is "tail"
    TAIL_BOOST_CAP = 2.0              # Max boost multiplier for tail samples (vs normal 4x)

    
    # A/B 3: F_x-space supervision for high-noise (learned branch direct supervision)
    EXP_SCORE_FX_HI = False
    FX_HI_WEIGHT = 1.0                 # Multiplier for Fx loss term
    FX_USE_COUT2_MATCH = True          # If True, multiply Fx-MSE by c_out^2 (unit-matching)
    
    print(f"[EXP FLAGS] WD0={EXP_SCORE_WD0}, WNORM={EXP_SCORE_WNORM}, HI_BOOST={EXP_SCORE_HI_BOOST}, FX_HI={EXP_SCORE_FX_HI}")
    # =========================================================================


    # PATCH 8C: Separate LR for generator (needs bigger LR due to small gradients)
    # A/B 1: Optionally disable weight decay for score_net to prevent high-σ amplitude suppression
    wd_default = 1e-4
    wd_score = 0.0 if EXP_SCORE_WD0 else wd_default
    
    optimizer = torch.optim.AdamW([
        {"params": context_encoder.parameters(), "lr": lr,       "weight_decay": wd_default},
        {"params": score_net.parameters(),       "lr": lr,       "weight_decay": wd_score},
        {"params": generator.parameters(),       "lr": lr * 5.0, "weight_decay": wd_default},
    ])
    
    print(f"[EXP] EXP_SCORE_WD0={EXP_SCORE_WD0} (score_net weight_decay={wd_score})")



    # ==============================================================================
    # ADAPTIVE QUANTILE GATES (dataset-independent)
    # ==============================================================================
    # These gates automatically learn thresholds based on noise distribution
    # Uses c_skip-based noise metric: noise = -log(c_skip), normalized by sigma_data
    # 
    # Target rates:
    #   - gram/gram_scale: 40-60% (global structure, more tolerant of noise)
    #   - edge/nca: 15% (local features, need clean signal)
    #   - learn_hi: 50% high-noise (for learned-branch geometry at high sigma)
    
    adaptive_gates = {
        "gram": uet.AdaptiveQuantileGate(target_rate=0.50, mode="low", warmup_steps=200),
        "gram_scale": uet.AdaptiveQuantileGate(target_rate=0.60, mode="low", warmup_steps=200),
        "edge": uet.AdaptiveQuantileGate(target_rate=0.30, mode="low", warmup_steps=200),  # Was 0.15
        "nca": uet.AdaptiveQuantileGate(target_rate=0.40, mode="low", warmup_steps=200),   # Was 0.15
        "learn_hi": uet.AdaptiveQuantileGate(target_rate=0.50, mode="high", warmup_steps=200),
    }

    # ==============================================================================
    # BOOST READINESS STATE MACHINE (CHANGE 2)
    # ==============================================================================
    # Boost only activates when high-σ scale is stable (Fx_pred/Fx_tgt ≈ 1.0)
    boost_state = {
        'ready': False,
        'start_step': None,
        'ema_fx_ratio_hi': 1.0,
        'last_fx_ratio_hi': None,    # ADD THIS KEY
        'stable_count': 0,
        'stability_tol': 0.10,
        'min_stable_checks': 3,
        'ema_decay': 0.98,
    }
    
    # A/B 2: High-noise gate for score loss boosting (dataset-independent)
    # This gate selects the noisiest fraction of samples to boost their L_score contribution
    score_hi_gate = uet.AdaptiveQuantileGate(
        target_rate=SCORE_HI_TARGET_RATE,
        mode="high",                  # Pass the noisiest fraction
        warmup_steps=200,
        reservoir_size=8192,
        update_every=50,
        ema=0.9,
    )
    
    print(f"[StageC] Initialized adaptive quantile gates:")
    for name, gate in adaptive_gates.items():
        print(f"  {name}: target_rate={gate.target_rate:.0%} mode={gate.mode}")
    print(f"[EXP] score_hi_gate: target_rate={SCORE_HI_TARGET_RATE:.0%} mode=high (for A/B 2)")


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

    # ==================== EMA SETUP ====================
    import copy
    ema_decay = 0.999  # Try 0.999 for small datasets, 0.9999 for large
    score_net_ema = copy.deepcopy(score_net).eval()
    for p in score_net_ema.parameters():
        p.requires_grad_(False)
    
    # Also create EMA for context_encoder
    context_encoder_ema = copy.deepcopy(context_encoder).eval()
    for p in context_encoder_ema.parameters():
        p.requires_grad_(False)
        
    @torch.no_grad()
    def ema_update(ema_model, model, decay: float):
        """Update EMA model weights."""
        msd = model.state_dict()
        for k, v_ema in ema_model.state_dict().items():
            v = msd[k]
            if not torch.is_floating_point(v):
                v_ema.copy_(v)
            else:
                v_ema.mul_(decay).add_(v, alpha=1.0 - decay)
    # ===================================================

    # ========== UNANCHORED MODE SAFETY CHECK ==========
    if not anchor_train and (fabric is None or fabric.is_global_zero):
        print("\n" + "="*70)
        print("UNANCHORED DIFFUSION TRAINING MODE")
        print("="*70)
        print("  anchor_train=False: No anchor channel, no anchor CFG, no anchor loss masking")
        print(f"  self_cond_mode={self_cond_mode}")
        print(f"  ctx_loss_weight={ctx_loss_weight}")
        print(f"  ctx_replace_variant={ctx_replace_variant}")
        
        # Warn if any anchor flags are set
        anchor_flags_active = []
        if anchor_p_uncond != 0.50:
            anchor_flags_active.append(f"anchor_p_uncond={anchor_p_uncond}")
        if anchor_clamp_clean:
            anchor_flags_active.append("anchor_clamp_clean=True")
        if anchor_mask_score_loss:
            anchor_flags_active.append("anchor_mask_score_loss=True")
        if anchor_warmup_steps > 0:
            anchor_flags_active.append(f"anchor_warmup_steps={anchor_warmup_steps}")
        
        if anchor_flags_active:
            print("\n⚠️  WARNING: Anchor flags are set but anchor_train=False. These will be IGNORED:")
            for flag in anchor_flags_active:
                print(f"    - {flag}")
        print("="*70 + "\n")
 
    # DataLoaders - OPTIMIZED
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_minisets, collate_sc_minisets, STPairSetDataset, collate_pair_minisets

    if fabric is not None:
        device = str(fabric.device)

    # ========== PAIRED OVERLAP TRAINING SETUP ==========
    # When enabled, disable ctx_loss to avoid competing objectives
    effective_ctx_loss_weight = ctx_loss_weight
    if train_pair_overlap and disable_ctx_loss_when_overlap:
        effective_ctx_loss_weight = 0.0
        if fabric is None or fabric.is_global_zero:
            print(f"[PAIR-OVERLAP] Disabling ctx_loss (weight {ctx_loss_weight} -> 0.0) to avoid competing objectives")

    #compute data deriven bin edges form ST coords
    # ST loader (conditional)
    use_st = (st_dataset is not None)
    st_pair_loader = None  # Will be set if train_pair_overlap is enabled

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

        # ========== CREATE PAIRED DATASET IF ENABLED ==========
        if train_pair_overlap:
            st_pair_dataset = STPairSetDataset(
                targets_dict=st_dataset.targets_dict,
                encoder=st_dataset.encoder,
                st_gene_expr_dict=st_dataset.st_gene_expr_dict,
                n_min=st_dataset.n_min,
                n_max=st_dataset.n_max,
                D_latent=st_dataset.D_latent,
                num_samples=st_dataset.num_samples,
                knn_k=st_dataset.knn_k,
                device=st_dataset.device,
                pool_mult=st_dataset.pool_mult,
                stochastic_tau=st_dataset.stochastic_tau,
                pair_overlap_alpha=pair_overlap_alpha,
                pair_overlap_min_I=pair_overlap_min_I,
                compete_train=st_dataset.compete_train,
                compete_n_extra=st_dataset.compete_n_extra,
                compete_n_rand=st_dataset.compete_n_rand,
                compete_n_hard=st_dataset.compete_n_hard,
                compete_use_pos_closure=st_dataset.compete_use_pos_closure,
                compete_k_pos=st_dataset.compete_k_pos,
                compete_expr_knn_k=st_dataset.compete_expr_knn_k,
                compete_anchor_only=st_dataset.compete_anchor_only,
            )
            # Use paired loader as the MAIN ST loader
            st_loader = DataLoader(
                st_pair_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_pair_minisets,
                num_workers=0,
                pin_memory=False
            )
            st_pair_loader = None  # Not needed separately - st_loader IS the paired loader
            if fabric is None or fabric.is_global_zero:
                print(f"[PAIR-OVERLAP] Using ONLY paired minisets for ST training")
                print(f"[PAIR-OVERLAP] alpha={pair_overlap_alpha}, min_I={pair_overlap_min_I}")
                print(f"[PAIR-OVERLAP] Overlap loss weights: shape={overlap_loss_weight_shape}, scale={overlap_loss_weight_scale}, kl={overlap_loss_weight_kl}")
        else:
            # Standard unpaired training
            st_loader = DataLoader(
                st_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_minisets,
                num_workers=0,
                pin_memory=False
            )


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
    else:
        sc_loader = None

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
            
            # If paired batch, use view1 for main losses
            if isinstance(batch, dict) and batch.get('is_pair_batch', False):
                batch = batch['view1']

            V_target = batch['V_target'].to(device)
            mask = batch['mask'].to(device)
            A_ref = loss_triangle._compute_avg_triangle_area(V_target, mask)
            triangle_refs.extend(A_ref.cpu().tolist())

    
    # triangle_epsilon_st = 1.0 * float(torch.tensor(triangle_refs).median())
    areas = np.array(triangle_refs)
    triangle_epsilon_st = float(np.quantile(areas, 0.10))
    
    # SC uses FIXED small epsilon (ST-independent)
    triangle_epsilon_sc = 0.01  # Just "don't collapse to line", no ST bias
    # =============================================================


    # ========== COMPUTE REPEL EPSILON FOR ST (NEAREST-NEIGHBOR DISTANCE) ==========

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

            # If paired batch, use view1 for main losses
            if isinstance(batch, dict) and batch.get('is_pair_batch', False):
                batch = batch['view1']
            
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

    repel_epsilon_st = float(np.quantile(nn_dists_arr, 0.01))
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
        sample_dataloader_and_report(
            st_loader=st_loader,
            sc_loader=sc_loader if use_sc else None,  # ← Pass None if disabled
            batches=3,
            device=device,
            is_global_zero=_is_rank0(),
            # save_json_path=os.path.join(outf, "minisets_check.json"),
            save_json_path=None
        )

    # ============================================================================

    from utils_et import build_sc_knn_cache

    # Build kNN cache from SC dataset embeddings (conditional)
    if use_sc:
        sc_knn = build_sc_knn_cache(
            sc_dataset.Z_cpu,
            k_pos=25,
            block_q=2048,
            device=device
        )
        POS_IDX = sc_knn["pos_idx"]
        K_POS = int(sc_knn["k_pos"])
    else:
        POS_IDX = None
        K_POS = 0

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

    # WEIGHTS = {
    #     'score': 1.0,
    #     'gram': 0.5,
    #     'gram_scale': 0.3,
    #     'heat': 0.25,
    #     'sw_st': 0.0,
    #     'sw_sc': 0.3,
    #     'overlap': 0.25,
    #     'ordinal_sc': 0.5,
    #     'st_dist': 0.0,
    #     'edm_tail': 0.3,
    #     'gen_align': 0.0,
    #     'dim': 0.0,
    #     'triangle': 0.0,
    #     'radial': 0.5,
    #     'knn_nca': 0.0,
    #     'repel': 0.0,      # NEW: ST repulsion loss
    #     'shape': 0.0,       # NEW: ST anisotropy/shape loss
    #     # NEW
    #     'edge': 0.5,       # Edge-length preservation (local metric)
    #     'topo': 0.3,       # Topology preservation (persistent homology proxy)
    #     'shape_spec': 0.3,      # Shape spectrum (anisotropy matching)
    # }

    # WEIGHTS = {
    #     'score': 1.0,
    #     'gram': 1.0,
    #     'gram_scale': 1.0,
    #     'heat': 0.0,
    #     'sw_st': 0.0,
    #     'sw_sc': 0.0,
    #     'overlap': 0.0,
    #     'ordinal_sc': 0.0,
    #     'st_dist': 0.0,
    #     'edm_tail': 0.0,
    #     'gen_align': 0.5,
    #     'dim': 0.0,
    #     'triangle': 0.0,
    #     'radial': 0.0,
    #     'knn_nca': 0.0,
    #     'repel': 0.0,
    #     'shape': 0.0,
    #     'edge': 2.0,       # Can re-enable later at 0.02-0.1 for low-sigma only
    #     'topo': 0.0,
    #     'shape_spec': 0.0,
    # }

    WEIGHTS = {
        'score': 16.0,         # was 1.0, but score is now ~32x smaller; 16 keeps it strong
        'gram': 2.0,           # was 1.0
        'gram_scale': 2.0,     # was 1.0
        'out_scale': 1.0,
        'gram_learn': 1.0,
        'knn_scale': 0.2,     # NEW: kNN distance scale calibration
        'heat': 0.0,
        'sw_st': 0.0,
        'sw_sc': 0.0,
        'overlap': 0.0,
        'ordinal_sc': 0.0,
        'st_dist': 0.0,
        'edm_tail': 0.0,
        'gen_align': 10.0,     # was 0.5 - will use new gen losses in Patch 8
        'gen_scale': 10.0,     # NEW: add this key for Patch 8
        'dim': 0.0,
        'triangle': 0.0,
        'radial': 0.0,
        'knn_nca': 2.0,
        'repel': 0.0,
        'shape': 0.0,
        'edge': 4.0,           # was 2.0
        'topo': 0.0,
        'shape_spec': 0.0,
        'subspace': 0.5,       # NEW: add this key for Patch 7
        'ctx_edge': 0.05,
        # ========== OVERLAP CONSISTENCY LOSSES (Candidate 1) ==========
        'ov_shape': 0.0,       # Weight set dynamically via overlap_loss_weight_shape
        'ov_scale': 0.0,       # Weight set dynamically via overlap_loss_weight_scale
        'ov_kl': 0.0,          # Weight set dynamically via overlap_loss_weight_kl
    }

    # ========== CONTEXT INVARIANCE LOSS CONFIG ==========
    ENABLE_CTX_EDGE = True  # Master switch - set False to disable ctx loss entirely
    CTX_WARMUP_STEPS = 2000  # Warmup steps before full ctx loss weight
    CTX_SNR_THRESH = 0.20  # SNR gate: only apply when snr_w >= this (try 0.10-0.30)
    CTX_KEEP_P = 0.8  # Fraction of extra points to keep (was 0.5, now milder)
    CTX_MIN_EXTRA = 16  # Minimum extra points to keep (was 8, now more)
    CTX_DEBUG_EVERY = 50  # Print debug info every N steps
    CTX_K = 8  # Number of neighbors for core->core kNN
    
    # Print ctx config once at startup
    if fabric is None or fabric.is_global_zero:
        print(f"\n[CTX-EDGE CONFIG] ENABLE_CTX_EDGE={ENABLE_CTX_EDGE}")
        if ENABLE_CTX_EDGE:
            print(f"  weight={WEIGHTS.get('ctx_edge', 0)}, warmup={CTX_WARMUP_STEPS}, "
                  f"snr_thresh={CTX_SNR_THRESH}, keep_p={CTX_KEEP_P}, min_extra={CTX_MIN_EXTRA}, K={CTX_K}")


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
    p_uncond = 0.10
    # sigma_gate = 5.0

    # ======== A/B TEST 2: Force Conditional-Only Training ========
    # Environment-controlled override for testing hypothesis:
    # "CFG context-drop training encourages averaged geometry that blurs 
    #  fine neighborhoods"
    #
    # When GEMS_FORCE_COND_ONLY=1, we bypass all CFG dropout and train
    # with full context at all times. This tests whether conditional-drop
    # is the source of "local blur" that hurts kNN@10.
    import os
    FORCE_COND_ONLY = (os.getenv("GEMS_FORCE_COND_ONLY", "0") == "1")
    if FORCE_COND_ONLY:
        print("[A/B TEST 2] FORCE_COND_ONLY=True: Disabling CFG context dropout entirely")
    # ======== END A/B TEST 2 INIT ========


    # CFG config - PATCH 9: schedule p_uncond to avoid blocking early learning
    p_uncond_max = 0.10
    cfg_warmup_start = 20   # Don't use CFG dropout for first N epochs
    cfg_warmup_len = 20     # Ramp up over this many epochs
    sigma_gate = 5.0


    # FiLM/CFG Debug config
    DEBUG_FILM_EVERY = 100   # Check FiLM effect every N steps
    DEBUG_FILM = False      # Enable/disable FiLM debugging
    

    history = {
        'epoch': [],
        'batch_losses': [],
        'epoch_avg': {
            'total': [], 'score': [], 'gram': [], 'gram_scale': [], 'out_scale': [], 
            'gram_learn': [], 'heat': [],
            'sw_st': [], 'sw_sc': [], 'overlap': [], 'ordinal_sc': [], 'st_dist': [],
            'edm_tail': [], 'gen_align': [], 'gen_scale': [], 'subspace': [],
            'dim': [], 'triangle': [], 'radial': [],
            'knn_nca': [], 'knn_scale': [], 'repel': [], 'shape': [], 'edge': [], 'topo': [], 'shape_spec': [],
            'ov_shape': [], 'ov_scale': [], 'ov_kl': [], 'score_2': [],
            'ctx_edge': [],  # NEW: context invariance loss
            'ctx_replace': [],
            'ctx_snr_med': [],
            'ctx_apply_rate': [],
            'ctx_hard_sim': [],  # Only used if variant=hard
            'ctx_perm_fixed': [],  # Only used if variant=permute
        }
    }

    start_epoch = 0
    if resume_ckpt_path:
        print(f"[RESUME] Loading Stage C checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)

        if 'context_encoder' in ckpt:
            context_encoder.load_state_dict(ckpt['context_encoder'])
        if 'score_net' in ckpt:
            score_net.load_state_dict(ckpt['score_net'])
        if 'generator' in ckpt:
            generator.load_state_dict(ckpt['generator'])
        else:
            print("[RESUME] WARNING: generator not found in checkpoint.")

        if 'context_encoder_ema' in ckpt:
            context_encoder_ema.load_state_dict(ckpt['context_encoder_ema'])
        if 'score_net_ema' in ckpt:
            score_net_ema.load_state_dict(ckpt['score_net_ema'])

        if 'optimizer' in ckpt and not resume_reset_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])

        if 'epoch' in ckpt:
            start_epoch = int(ckpt['epoch']) + 1

        if 'history' in ckpt:
            history = ckpt['history']
            if 'epoch_avg' in history:
                history['epoch_avg'].setdefault('ov_shape', [])
                history['epoch_avg'].setdefault('ov_scale', [])
                history['epoch_avg'].setdefault('ov_kl', [])
                history['epoch_avg'].setdefault('score_2', [])

        if 'sigma_data' in ckpt:
            sigma_data = ckpt['sigma_data']
        if 'sigma_min' in ckpt:
            sigma_min = ckpt['sigma_min']
        if 'sigma_max' in ckpt:
            sigma_max = ckpt['sigma_max']

        print(f"[RESUME] start_epoch={start_epoch}, reset_optimizer={resume_reset_optimizer}")


    # ========== EDM DEBUG STATE INITIALIZATION ==========
    edm_debug_state = {
        'fixed_batch': None,  # Phase 0: fixed eval batch
        'sigma_bins': None,   # Phase 6: per-sigma-bin accumulation
        'sigma_bin_edges': None,
        'sigma_bin_sum_err2': None,
        'sigma_bin_sum_w': None,
        'sigma_bin_sum_werr2': None,
        'sigma_bin_count': None,
        'baseline_mse': None,  # Phase 1: baseline reference
    }

    # ========== PROBE BATCH STATE INITIALIZATION (ChatGPT Hypothesis 3 & 4) ==========
    # These probes provide discriminative metrics that cannot "look healthy" while kNN@10 fails
    probe_state = {
        'batch_captured': False,        # Whether probe batch has been captured
        'Z_set': None,                  # (N_probe, D_z) encoder embeddings
        'V_target': None,               # (N_probe, D_latent) ground truth geometry
        'mask': None,                   # (N_probe,) validity mask (all True for probe)
        'knn_spatial': None,            # (N_probe, k) spatial kNN indices
        'n_probe': 0,                   # Number of probe points
        'slide_id': None,               # Source slide ID
        'probe_indices': None,          # Global indices in original slide
        'st_ident_computed': False,     # Whether ST-IDENT audit was run
    }
    
    # Probe evaluation config
    # ========== PROBE CONFIGURATION (hardcoded defaults) ==========
    PROBE_SIGMAS = [0.2, 0.8, 1.5, 2.0, 2.5, 3.0]  # Multiple sigma levels to probe
    PROBE_SEED = 42             # Fixed seed for reproducible noise  
    PROBE_K = 10                # Match target kNN metric (k=10)
    N_PROBE_TARGET = 256        # Target probe batch size

    # ========== OPEN-LOOP PROBE SAMPLER CONFIG ==========
    PROBE_SAMPLE_STEPS = 20
    PROBE_SAMPLE_SIGMA_MAX = 3.0
    PROBE_SAMPLE_SIGMA_MIN = 0.02
    PROBE_SAMPLE_RHO = 7.0  # Karras schedule
    PROBE_SAMPLE_TRACE_SIGMAS = [3.0, 2.3, 1.5, 0.8, 0.2]  # Sigmas to log during trajectory

    # ========== NEW: Anchored training state ==========
    anchor_state = {
        'enabled': anchor_train,
        'total_anchored_batches': 0,
        'total_unanchored_batches': 0,
    }
    
    if anchor_train and (fabric is None or fabric.is_global_zero):
        print(f"\n{uet.ANCHOR_TAG} Anchored training ENABLED")
        print(f"  anchor_p_uncond: {anchor_p_uncond}")
        print(f"  anchor_clamp_clean: {anchor_clamp_clean}")
        print(f"  anchor_mask_score_loss: {anchor_mask_score_loss}")
        print(f"  anchor_warmup_steps: {anchor_warmup_steps}")
    
    # Log active losses at startup (rank-0 only)
    if fabric is None or fabric.is_global_zero:
        active_losses = {k: v for k, v in WEIGHTS.items() if v != 0}
        print(f"\n[ANCHOR-GEOM] active_losses (weight>0): {list(active_losses.keys())}")
        print(f"[ANCHOR-GEOM] anchor_train={anchor_train} anchor_geom_losses={anchor_geom_losses}")
        if anchor_train and anchor_geom_losses:
            print(f"[ANCHOR-GEOM] mode={anchor_geom_mode} min_unknown={anchor_geom_min_unknown}")



    # Compute sigma_data once at start    
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

                    if isinstance(sample_batch, dict) and sample_batch.get('is_pair_batch', False):
                        sample_batch = sample_batch['view1']

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

    # Store sigma_data on score_net for reference in forward pass
    print(f"[StageC] sigma_data = {sigma_data:.4f}")
    
    # Store sigma_data on score_net (handle DDP/Fabric wrapper)
    if hasattr(score_net, 'module'):
        score_net.module.sigma_data = sigma_data
    else:
        score_net.sigma_data = sigma_data




    # ========== PHASE 1: DATA + TARGET SCALE SANITY ==========
    if fabric is None or fabric.is_global_zero:
        print("\n" + "="*70)
        print("[PHASE 1] DATA + TARGET SCALE SANITY CHECK")
        print("="*70)
        
        # Phase 0.3: Store fixed evaluation batch
        try:
            fixed_batch_raw = next(iter(st_loader))

            if isinstance(fixed_batch_raw, dict) and fixed_batch_raw.get('is_pair_batch', False):
                fixed_batch_raw = fixed_batch_raw['view1']

            edm_debug_state['fixed_batch'] = {
                'Z_set': fixed_batch_raw['Z_set'].cpu(),
                'mask': fixed_batch_raw['mask'].cpu(),
                'V_target': fixed_batch_raw['V_target'].cpu(),
                'G_target': fixed_batch_raw['G_target'].cpu(),
                'n': fixed_batch_raw['n'],
            }
            print("[PHASE 0] Fixed evaluation batch stored")
        except:
            print("[PHASE 0] WARNING: Could not store fixed batch")

        
        # 1.1 Target stats (masked, per-set)
        print("\n[PHASE 1.1] Target Statistics (V_target):")
        v_target_stds = []
        v_target_rms_list = []
        
        it_temp = iter(st_loader)
        for _ in range(min(10, len(st_loader))):
            try:
                batch_temp = next(it_temp)
            except:
                break

            if isinstance(batch_temp, dict) and batch_temp.get('is_pair_batch', False):
                batch_temp = batch_temp['view1']

            V_temp = batch_temp['V_target'].to(device)
            mask_temp = batch_temp['mask'].to(device)
            
            for b in range(V_temp.shape[0]):
                m_b = mask_temp[b]
                n_valid = m_b.sum().item()
                if n_valid < 3:
                    continue
                V_b = V_temp[b, m_b]  # (n_valid, D)
                v_target_stds.append(V_b.std().item())
                v_target_rms_list.append(V_b.pow(2).mean().sqrt().item())

        
        if v_target_stds:
            v_std_arr = np.array(v_target_stds)
            v_rms_arr = np.array(v_target_rms_list)
            print(f"  V_target STD:  min={v_std_arr.min():.4f} "
                  f"median={np.median(v_std_arr):.4f} max={v_std_arr.max():.4f}")
            print(f"  V_target RMS:  min={v_rms_arr.min():.4f} "
                  f"median={np.median(v_rms_arr):.4f} max={v_rms_arr.max():.4f}")
        
        # 1.2 sigma_data verification
        print(f"\n[PHASE 1.2] sigma_data Verification:")
        print(f"  sigma_data (from median std): {sigma_data:.6f}")
        sigma_data_rms = float(np.median(v_rms_arr)) if v_rms_arr.size > 0 else sigma_data
        print(f"  sigma_data (from median rms): {sigma_data_rms:.6f}")
        print(f"  Ratio (std/rms): {sigma_data/sigma_data_rms:.4f}")
        
        # 1.3 Baseline reference error
        print(f"\n[PHASE 1.3] Baseline Reference Error:")
        baseline_mse_list = []
        sigma_test = sigma_data  # Test at sigma_data scale
        
        it_baseline = iter(st_loader)
        for _ in range(5):
            try:
                batch_bl = next(it_baseline)
            except:
                break

            if isinstance(batch_bl, dict) and batch_bl.get('is_pair_batch', False):
                batch_bl = batch_bl['view1']

            V_bl = batch_bl['V_target'].to(device)
            mask_bl = batch_bl['mask'].to(device)
            
            eps_bl = torch.randn_like(V_bl)
            V_t_bl = V_bl + sigma_test * eps_bl
            V_t_bl = V_t_bl * mask_bl.unsqueeze(-1).float()

            
            # Baseline: no denoising (x0_baseline = V_t)
            err_bl = (V_t_bl - V_bl).pow(2).sum(dim=-1)  # (B, N)
            mask_f = mask_bl.float()
            mse_bl = (err_bl * mask_f).sum() / mask_f.sum()
            baseline_mse_list.append(mse_bl.item())
        
        baseline_mse = float(np.mean(baseline_mse_list)) if baseline_mse_list else 0.0
        edm_debug_state['baseline_mse'] = baseline_mse
        print(f"  Baseline MSE (sigma={sigma_test:.4f}, no denoising): {baseline_mse:.6f}")
        print(f"  Model should beat this to show learning")
        
        print("="*70 + "\n")


    # EDM: Clamp sigma_max based on sigma_data
    if use_edm:
        sigma_max = min(sigma_max, sigma_data * 100)  # Reasonable upper bound
        sigma_min = max(sigma_min, sigma_data * 0.001)  # Reasonable lower bound
        # NEW: Set refinement sigma range (train as refiner, not full generator)
        if sigma_refine_max is None:
            sigma_refine_max = 20.0 * sigma_data  # Default: 1x sigma_data
        sigma_refine_max = min(sigma_refine_max, sigma_max)  # Don't exceed sigma_max
    
    print(f"[StageC] sigma_data = {sigma_data:.4f}")
    print(f"[StageC] sigma_min = {sigma_min:.6f}, sigma_max = {sigma_max:.2f}")
    if use_edm:
        print(f"[StageC] EDM mode ENABLED: P_mean={P_mean}, P_std={P_std}")
        print(f"[StageC] EDM refinement mode: sigma_refine_max = {sigma_refine_max:.4f}")



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

    USE_AUTOBALANCE = False

    #self conditioning probability
    p_sc = 0.5 #prob of using self-conditioning in training

    # Early stopping state
    early_stop_best = float('inf')
    early_stop_no_improve = 0
    early_stopped = False
    early_stop_epoch = -1


    # --- DEBUG 5: Sigma-stratified sample selection helper ---
    def get_debug_samples_by_sigma(sigma_batch, n_per_bin=1):
        """
        Return dict of sample indices stratified by sigma bins.
        
        Args:
            sigma_batch: (B,) tensor of sigma values
            n_per_bin: how many samples to return per bin
            
        Returns:
            dict like {'very_low': [0, 3], 'low': [1], 'mid': [5], ...}
        """
        indices = {}
        bins = [
            (0.0, 0.1, 'very_low'),   # σ < 0.1: nearly clean
            (0.1, 0.3, 'low'),         # σ ∈ [0.1, 0.3): geometry still useful
            (0.3, 0.5, 'mid'),         # σ ∈ [0.3, 0.5): geometry questionable
            (0.5, 1.0, 'high'),        # σ ∈ [0.5, 1.0): mostly noise
            (1.0, float('inf'), 'very_high')  # σ > 1.0: pure noise
        ]
        
        for lo, hi, name in bins:
            in_bin = (sigma_batch >= lo) & (sigma_batch < hi)
            if in_bin.any():
                bin_indices = in_bin.nonzero(as_tuple=True)[0]
                indices[name] = bin_indices[:n_per_bin].tolist()
            else:
                indices[name] = []
        
        return indices
    
    def get_one_sample_per_sigma_bin(sigma_batch):
        """Convenience: get exactly one sample index per available sigma bin."""
        samples = get_debug_samples_by_sigma(sigma_batch, n_per_bin=1)
        # Flatten to list of (bin_name, idx) tuples
        result = []
        for bin_name, idx_list in samples.items():
            if idx_list:
                result.append((bin_name, idx_list[0]))
        return result
    

    # Track pin-rate for adaptive clamp (persists across steps within this training run)
    last_pin_rate = 1.0  # Assume worst case initially
    
    for epoch in range(start_epoch, n_epochs):
        epoch_cv_sum = 0.0
        epoch_qent_sum = 0.0
        epoch_nca_loss_sum = 0.0
        epoch_nca_count = 0

        st_iter = iter(st_loader) if use_st else None
        sc_iter = iter(sc_loader) if use_sc else None
        # st_pair_iter = iter(st_pair_loader) if (train_pair_overlap and st_pair_loader is not None) else None

        epoch_losses = {k: 0.0 for k in WEIGHTS.keys()}
        epoch_losses['total'] = 0.0

        # Context replacement loss tracking (separate from WEIGHTS)
        ctx_replace_sum = 0.0
        ctx_apply_count = 0
        ctx_snr_sum = 0.0
        ctx_hard_sim_sum = 0.0
        ctx_perm_fixed_sum = 0.0

        # ========== OVERLAP CONSISTENCY LOSS TRACKING (Candidate 1) ==========
        ov_loss_sum = {'shape': 0.0, 'scale': 0.0, 'kl': 0.0, 'total': 0.0}
        ov_apply_count = 0
        ov_skipped_sigma = 0
        ov_pair_batches = 0
        ov_no_valid = 0
        ov_I_sizes = []
        ov_jaccard_k10 = []

        n_batches = 0
        c_overlap = 0

        st_batches = 0
        sc_batches = 0
        
        # Schedule based on what loaders we have
        if use_st and use_sc:
            max_len = max(len(st_loader), len(sc_loader))
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
                
                # Handle paired batches: extract view1 as main batch, keep view2 for overlap
                is_pair_batch = batch.get('is_pair_batch', False)
                if is_pair_batch:
                    pair_batch_full = batch  # Keep full paired batch for overlap loss
                    batch = batch['view1']   # Use view1 as the main batch
                else:
                    pair_batch_full = None
                    
            else:  # SC
                if not use_sc:
                    continue  # Skip SC batches if disabled
                batch = next(sc_iter, None)
                if batch is None:
                    sc_iter = iter(sc_loader)
                    batch = next(sc_iter, None)
                    if batch is None:
                        continue
                is_pair_batch = False
                pair_batch_full = None
            
            is_sc = batch.get('is_sc', False)

            
            # ========== SPOT IDENTITY: Validation (once per epoch) ==========
            if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                _validate_st_uid_uniqueness(batch, device=device)


            if not is_sc:
                st_batches += 1
            else:
                sc_batches += 1
                
            Z_set = batch['Z_set'].to(device)
            mask = batch['mask'].to(device)
            # === Z_ln conditioning: normalize Z_set before augmentation ===
            Z_set = apply_z_ln(Z_set, context_encoder)


            # Core membership mask (core points of the sampled ST miniset).
            # This is distinct from anchor_cond_mask (conditioning anchors).
            anchor_mask = batch.get('anchor_mask', None)
            if anchor_mask is not None:
                anchor_mask = anchor_mask.to(device).bool() & mask


            n_list = batch['n']
            batch_size_real = Z_set.shape[0]
            
            D_latent = score_net.D_latent

            
            # ========== Extract anchor_cond_mask BEFORE any processing ==========
            if anchor_train and not is_sc:
                anchor_cond_mask = batch.get('anchor_cond_mask', torch.zeros_like(mask)).to(device)
                # Ensure anchor_cond_mask respects padding
                anchor_cond_mask = anchor_cond_mask & mask
                
                # Apply warmup: linearly ramp anchored probability
                if anchor_warmup_steps > 0:
                    warmup_progress = min(1.0, global_step / anchor_warmup_steps)
                    p_anchor_effective = (1.0 - anchor_p_uncond) * warmup_progress
                    # With probability (1 - p_anchor_effective), force unanchored
                    if torch.rand(1).item() > p_anchor_effective:
                        anchor_cond_mask = torch.zeros_like(anchor_cond_mask)
                
                # Track statistics
                batch_is_anchored = anchor_cond_mask.any(dim=1).sum().item()
                anchor_state['total_anchored_batches'] += batch_is_anchored
                anchor_state['total_unanchored_batches'] += (batch_size_real - batch_is_anchored)
                
                # Debug logging
                if global_step % anchor_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                    stats = uet.anchor_mask_stats(anchor_cond_mask, mask)
                    print(f"\n{uet.ANCHOR_TAG} step={global_step}")
                    print(f"  [ANCHOR-BATCH] n_anchor_mean={stats['n_anchor_mean']:.1f} "
                            f"frac_mean={stats['frac_anchor_mean']:.2%} "
                            f"n_unknown_mean={stats['n_unknown_mean']:.1f}")
            else:
                anchor_cond_mask = None

            # Apply context augmentation stochastically (ONLY to base Z_set, NOT anchor channel)
            if torch.rand(1).item() < aug_prob:
                Z_set = apply_context_augmentation(
                    Z_set, mask, 
                    noise_std=z_noise_std, 
                    dropout_rate=z_dropout_rate
                )
            
            # ========== Append anchor channel AFTER augmentation ==========
            # This ensures the anchor indicator is NOT corrupted by noise/dropout
            if anchor_train and anchor_cond_mask is not None and not is_sc:
                anchor_channel = anchor_cond_mask.float().unsqueeze(-1)  # (B, N, 1)
                Z_set = torch.cat([Z_set, anchor_channel], dim=-1)
       
            # ===== FORWARD PASS WITH AMP =====
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                # Context encoding
                H = context_encoder(Z_set, mask)

                # === Define self-conditioning flag ONCE ===
                use_self_cond = (torch.rand(1, device=device).item() < p_sc)

                # === EDM: sample sigma from log-normal ===
                if use_edm:
                    # sigma = uet.sample_sigma_lognormal(batch_size_real, P_mean, P_std, device)
                    sigma = uet.sample_sigma_lognormal_stratified(
                        batch_size_real, P_mean, P_std,
                        high_sigma_fraction=0.4,  # 25% of batch guaranteed high-σ
                        high_sigma_threshold=0.5,  # σ >= 0.5 is "high"
                        device=device
                    )
                    sigma = sigma.clamp(sigma_min, sigma_refine_max)
                    sigma_t = sigma.view(-1, 1, 1)

                    # ========== PHASE 2: SIGMA SAMPLING SANITY ==========
                    if global_step % 20 == 0 and (fabric is None or fabric.is_global_zero):
                        print(f"\n[PHASE 2] Sigma Sampling (step {global_step}):")
                        print(f"  sigma: min={sigma.min():.6f} median={sigma.median():.6f} max={sigma.max():.6f}")
                        
                        log_sigma = sigma.log()
                        log_min = math.log(sigma_min)
                        log_refine = math.log(sigma_refine_max)
                        print(f"  log(sigma): min={log_sigma.min():.4f} median={log_sigma.median():.4f} max={log_sigma.max():.4f}")
                        
                        # Quantiles
                        print(f"  Quantiles: p05={sigma.quantile(0.05):.6f} p25={sigma.quantile(0.25):.6f} "
                              f"p75={sigma.quantile(0.75):.6f} p95={sigma.quantile(0.95):.6f}")
                        
                        # Clamp hit rates
                        clamp_low = (sigma <= sigma_min + 1e-6).float().mean().item()
                        clamp_high = (sigma >= sigma_refine_max - 1e-6).float().mean().item()
                        print(f"  Clamp hit rate: at sigma_min={clamp_low*100:.1f}% at sigma_refine_max={clamp_high*100:.1f}%")
                        
                        if clamp_low > 0.3 or clamp_high > 0.3:
                            print(f"  ⚠️ WARNING: >30% samples hit clamp bounds!")
                        
                        # t_norm mapping sanity (for time-aware architectures)
                        t_norm_check = ((log_sigma - log_min) / (log_refine - log_min + 1e-8)).clamp(0, 1)
                        print(f"  t_norm: min={t_norm_check.min():.4f} median={t_norm_check.median():.4f} max={t_norm_check.max():.4f}")
                    
                    # Create t_norm proxy for gating/debug (log-space normalization)
                    log_sigma = sigma.log()
                    log_min = math.log(sigma_min)
                    log_max = math.log(sigma_refine_max)
                    t_norm = ((log_sigma - log_min) / (log_max - log_min + 1e-8)).clamp(0, 1)  # (B,)
                else:
                    # Old: quadratic bias toward low noise
                    u = torch.rand(batch_size_real, device=device)
                    t_cont = (u ** 2) * (n_timesteps - 1)
                    t_idx = t_cont.long()
                    t_norm = t_cont / (n_timesteps - 1)
                    sigma_t = sigmas[t_idx].view(-1, 1, 1)

                if not is_sc:
                    # ST batch: Ground truth from Gram matrix
                    V_target_raw = batch['V_target'].to(device)
                    G_target = batch['G_target'].to(device)  # For geometry losses
                    D_latent = score_net.D_latent

                    # --- PATCH 3: Canonicalize ST target coordinates from Gram ---
                    V_target = torch.zeros_like(V_target_raw)
                    for i in range(batch_size_real):
                        n_valid = int(mask[i].sum().item())
                        if n_valid <= 1:
                            continue
                        G_i = G_target[i, :n_valid, :n_valid].float()
                        V_i = uet.factor_from_gram(G_i, D_latent).to(V_target_raw.dtype)
                        V_target[i, :n_valid] = V_i
                    # --------------------------------------------------------

                    # ========== PROBE BATCH CAPTURE (once, rank-0 only) ==========
                    # Must be AFTER V_target canonicalization so we capture the same
                    # targets the model is trained on
                    if not probe_state['batch_captured'] and (fabric is None or fabric.is_global_zero):
                        n_valid_total = mask.sum().item()
                        
                        if n_valid_total >= N_PROBE_TARGET:
                            # Find the batch item with most valid points
                            n_per_sample = mask.sum(dim=1)  # (B,)
                            best_idx = n_per_sample.argmax().item()
                            n_best = int(n_per_sample[best_idx].item())
                            
                            if n_best >= N_PROBE_TARGET:
                                # Take the first N_PROBE_TARGET valid points from this sample
                                m_best = mask[best_idx].bool()
                                valid_indices = torch.where(m_best)[0][:N_PROBE_TARGET]
                                
                                # Store probe batch (CPU for persistence)
                                # Use CANONICALIZED V_target (what model trains on)
                                probe_state['Z_set'] = Z_set[best_idx, valid_indices].detach().cpu()
                                probe_state['V_target'] = V_target[best_idx, valid_indices].detach().cpu()
                                probe_state['mask'] = torch.ones(len(valid_indices), dtype=torch.bool)
                                probe_state['n_probe'] = len(valid_indices)
                                probe_state['batch_captured'] = True
                                
                                # Compute spatial kNN for this probe (fixed reference)
                                V_probe = probe_state['V_target']
                                D_probe = torch.cdist(V_probe, V_probe)
                                D_probe.fill_diagonal_(float('inf'))
                                _, knn_spatial_probe = D_probe.topk(PROBE_K, dim=1, largest=False)
                                probe_state['knn_spatial'] = knn_spatial_probe
                                
                                # Get slide info if available
                                # ========== SPOT IDENTITY: Probe uses spot_indices (within-slide) ==========
                                if 'overlap_info' in batch:
                                    oi = batch['overlap_info'][best_idx]
                                    probe_state['slide_id'] = oi.get('slide_id', None)
                                    probe_state['probe_spot_indices'] = oi.get('indices', None)  # Within-slide
                                    probe_state['probe_uid'] = oi.get('global_uid', None)  # UIDs
                                
                                print(f"\n[PROBE-SNAP] step={global_step} n={probe_state['n_probe']} "
                                      f"n_valid={n_best} k_spatial={PROBE_K}")
                                print(f"[PROBE-SNAP] Probe batch captured and fixed for training evaluation")

                    # ========== ST-IDENT AUDIT (once, after probe capture) ==========
                    if probe_state['batch_captured'] and not probe_state['st_ident_computed'] and \
                       (fabric is None or fabric.is_global_zero):
                        print(f"\n[ST-IDENT] Running identifiability audit on probe batch...")
                        
                        # Run audit: can expression recover spatial neighbors?
                        Z_probe = probe_state['Z_set'].unsqueeze(0).to(device)  # (1, N, D_z)
                        V_probe = probe_state['V_target'].unsqueeze(0).to(device)  # (1, N, D)
                        m_probe = probe_state['mask'].unsqueeze(0).to(device)  # (1, N)
                        
                        ident_results = uet.st_ident_audit(Z_probe, V_probe, m_probe, k=PROBE_K)
                        
                        print(f"[ST-IDENT] step={global_step} "
                              f"expr_knn{PROBE_K}={ident_results['expr_knn_overlap']:.4f} "
                              f"median_rank_spatial{PROBE_K}_in_expr={ident_results['median_rank_spatial_in_expr']:.1f} "
                              f"p90_rank={ident_results['p90_rank_spatial_in_expr']:.1f}")
                        print(f"[ST-IDENT] local_ambiguity_index={ident_results['local_ambiguity_index']:.4f}")
                        
                        # Interpretation guidance
                        if ident_results['expr_knn_overlap'] < 0.35:
                            print(f"[ST-IDENT] ⚠️ WARNING: Expression kNN overlap is low ({ident_results['expr_knn_overlap']:.2f})")
                            print(f"[ST-IDENT] This suggests kNN@{PROBE_K} may be near data ceiling - "
                                  "consider adding celltype/morphology features")
                        
                        if ident_results['median_rank_spatial_in_expr'] > 50:
                            print(f"[ST-IDENT] ⚠️ WARNING: Spatial neighbors rank poorly in expression space")
                            print(f"[ST-IDENT] Median rank {ident_results['median_rank_spatial_in_expr']:.0f} >> 10 "
                                  "means expression doesn't strongly encode spatial locality")
                        
                        probe_state['st_ident_computed'] = True
                        probe_state['st_ident_results'] = ident_results

                    
                    # Optional: Compute generator output for alignment training
                    if WEIGHTS['gen_align'] > 0:
                        V_gen = generator(H, mask)
                    else:
                        V_gen = None  # Don't waste compute if not training it
                else:
                    # SC batch: No ground truth, use generator as target
                    V_gen = generator(H, mask)
                    V_target = V_gen


                # =================================================================
                # [DEBUG] GENERATOR SCALE SANITY CHECK
                # =================================================================
                if (global_step % 25 == 0) and (not is_sc) and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        # 1. Force compute V_gen to check what the generator is producing
                        # (Even if gen_align is off, we need to see what it outputs)
                        V_gen_debug = generator(H, mask)
                        
                        # 2. Inspect the first valid sample in the batch
                        b_idx = 0
                        m_b = mask[b_idx].bool()
                        n_valid = m_b.sum().item()
                        
                        if n_valid > 10:
                            v_gen = V_gen_debug[b_idx, m_b].float()
                            v_tgt = batch['V_target'][b_idx, m_b].to(device).float()
                            
                            # --- Metric A: RMS Ratio (Global Scale) ---
                            # Are the points too far from zero?
                            rms_gen = v_gen.pow(2).mean().sqrt()
                            rms_tgt = v_tgt.pow(2).mean().sqrt()
                            rms_ratio = rms_gen / (rms_tgt + 1e-8)
                            
                            # --- Metric B: Median kNN Edge Ratio (Local Density) ---
                            # Are the points too far from each other?
                            
                            # Compute raw distance matrices
                            d_gen = torch.cdist(v_gen, v_gen)
                            d_tgt = torch.cdist(v_tgt, v_tgt)
                            
                            # Mask diagonal to ignore self-distance
                            eye = torch.eye(n_valid, device=device).bool()
                            d_gen.masked_fill_(eye, float('inf'))
                            d_tgt.masked_fill_(eye, float('inf'))
                            
                            # Get nearest neighbor distances (k=1 is strict NN, k=5 is robust)
                            k_check = 5
                            d_gen_knn, _ = d_gen.topk(k_check, largest=False)
                            d_tgt_knn, _ = d_tgt.topk(k_check, largest=False)
                            
                            # Median edge length
                            med_edge_gen = d_gen_knn.median()
                            med_edge_tgt = d_tgt_knn.median()
                            edge_ratio = med_edge_gen / (med_edge_tgt + 1e-8)
                            
                            print(f"\n[GEN SANITY] step={global_step} (sample 0)")
                            print(f"  Global Scale (RMS): gen={rms_gen:.3f} vs tgt={rms_tgt:.3f} -> Ratio={rms_ratio:.3f}")
                            print(f"  Local Density (NN): gen={med_edge_gen:.3f} vs tgt={med_edge_tgt:.3f} -> Ratio={edge_ratio:.3f}")
                            
                            if rms_ratio > 5.0 or edge_ratio > 5.0:
                                print("  ⚠️ CRITICAL: Generator is initializing too large! Gradients will explode.")
                            if rms_ratio < 0.1 or edge_ratio < 0.1:
                                print("  ⚠️ CRITICAL: Generator has collapsed to zero!")
                            print("="*60)

                if (global_step % 25 == 0) and (not is_sc) and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        v_gen_scale = generator(H, mask)
                        mask_f = mask.unsqueeze(-1).float()
                        valid_count = mask_f.sum() * v_gen_scale.shape[-1]
                        rms_gen = (v_gen_scale.pow(2) * mask_f).sum().div(valid_count).sqrt()
                        rms_tgt = (V_target.pow(2) * mask_f).sum().div(valid_count).sqrt()
                        print(f"[GEN SCALE] RMS={rms_gen:.4f} (target ~{rms_tgt:.4f}) ratio={rms_gen/rms_tgt:.4f}")


                # ===== CFG: Drop context ONLY for score network =====
                # PATCH 9: Schedule p_uncond to avoid blocking early learning
                # A/B TEST 2: Override when FORCE_COND_ONLY is set
                
                if FORCE_COND_ONLY:
                    # Bypass CFG dropout entirely for testing
                    p_uncond_curr = 0.0
                    drop_bool = torch.zeros(batch_size_real, device=device)
                    view_shape = [batch_size_real] + [1] * (H.ndim - 1)
                    drop_mask = drop_bool.view(*view_shape)
                    H_train = H  # Full context always
                else:
                    # Original CFG schedule logic
                    if epoch < cfg_warmup_start:
                        p_uncond_curr = 0.0
                    else:
                        t_warmup = (epoch - cfg_warmup_start) / float(max(cfg_warmup_len, 1))
                        t_warmup = max(0.0, min(1.0, t_warmup))
                        p_uncond_curr = p_uncond_max * t_warmup
                    
                    drop_probs = torch.rand(batch_size_real, device=device)
                    drop_bool = (drop_probs < p_uncond_curr).float()
                    
                    # Dynamically reshape drop_mask to match H's dimensions
                    # If H is (B, C), makes (B, 1). If H is (B, N, C), makes (B, 1, 1)
                    view_shape = [batch_size_real] + [1] * (H.ndim - 1)
                    drop_mask = drop_bool.view(*view_shape)
                    
                    H_train = H * (1.0 - drop_mask)

                # ======== A/B TEST 2 DEBUG: CFG Status ========
                if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                    uncond_rate = drop_bool.mean().item()
                    print(f"[CFG] epoch={epoch} step={global_step} p_uncond_curr={p_uncond_curr:.4f} "
                          f"drop_rate={uncond_rate:.4f} FORCE_COND_ONLY={FORCE_COND_ONLY}")
                # ======== END A/B TEST 2 DEBUG ========

                
                # For ST: still load G_target (needed for Gram loss later)
                if not is_sc:
                    G_target = batch['G_target'].to(device)
                
                # ===== NOISE AROUND GROUND TRUTH (not generator!) =====
                eps = torch.randn_like(V_target)
                # eps = eps * mask.unsqueeze(-1).float()
                V_t = V_target + sigma_t * eps  # ← KEY FIX: noise around V_target
                V_t = V_t * mask.unsqueeze(-1).float()
                
                # ========== NEW: Anchored training - clamp anchor positions to clean values ==========
                if anchor_train and anchor_clamp_clean and anchor_cond_mask is not None and not is_sc:
                    V_t = uet.apply_anchor_clamp(V_t, V_target, anchor_cond_mask, mask)
                    
                    # Debug: verify clamp worked
                    if global_step % anchor_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            if anchor_cond_mask.any():
                                anchor_diff = (V_t - V_target).abs()
                                anchor_positions = anchor_cond_mask.unsqueeze(-1).expand_as(V_t)
                                max_anchor_diff = anchor_diff[anchor_positions].max().item()
                                sigma_median = sigma_t.median().item()
                                print(f"  [ANCHOR-NOISE] sigma_p50={sigma_median:.4f} "
                                      f"max_abs_anchor_diff={max_anchor_diff:.6f} (should be ~0)")

                # ✅ SAVE THE ORIGINAL for diagnostics
                V_target_orig = V_target.clone().detach()
                eps_orig = eps.clone().detach()
                sigma_t_orig = sigma_t.clone().detach()
                mask_orig = mask.clone().detach()

                if global_step % 2500 == 0 and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        # --- DEBUG 5: Check noise across sigma bins ---
                        sigma_for_debug = sigma_t.view(-1) if use_edm else torch.exp(4.0 * t_norm)
                        debug_samples = get_one_sample_per_sigma_bin(sigma_for_debug)
                        
                        for bin_name, b_idx in debug_samples[:2]:  # Check 2 bins max
                            m_b = mask_orig[b_idx].bool()
                            if m_b.sum() > 5:
                                noise_actual = (V_t[b_idx] - V_target_orig[b_idx])[m_b]
                                sigma_val = sigma_for_debug[b_idx].item()
                                noise_expected = (sigma_t_orig[b_idx] * eps_orig[b_idx])[m_b]
                                diff = (noise_actual - noise_expected).abs().max().item()
                                
                                noise_b = (V_t[b_idx] - V_target_orig[b_idx])[m_b]
                                noise_std = noise_b.std(dim=0).mean().item()
                                noise_mean_norm = noise_b.mean(dim=0).norm().item()
                                sig_b = float(sigma_t_orig[b_idx].view(-1)[0])

                                print(f"[NOISE CHECK] bin={bin_name}, σ={sigma_val:.3f}, sample={b_idx}")

                                print(f"\n[NOISE] step={global_step}")
                                print(f"  equation_diff: {diff:.6f}")
                                print(f"  sigma: {sig_b:.4f}")
                                print(f"  noise_std: {noise_std:.4f}")
                                print(f"  noise_mean: {noise_mean_norm:.4f}")
                                print(f"  ratio: {noise_std/sig_b:.3f}")

                                # --- FIX #7: Use proper expected value for mean shift threshold ---
                                # Expected: ||mean(eps)|| ≈ σ * sqrt(D) / sqrt(n_valid)
                                n_valid = m_b.sum().item()
                                D_check = noise_b.shape[-1]
                                expected_mean_norm = sig_b * math.sqrt(D_check) / math.sqrt(max(n_valid, 1))

                                if diff > 1e-4:
                                    print(f"  🔴 EQUATION FAIL")
                                elif noise_std < 0.7 * sig_b:
                                    print(f"  🔴 TRANSLATION NOISE")
                                elif noise_mean_norm > 3.0 * expected_mean_norm:  # FIX: 3σ rule
                                    print(f"  🔴 MEAN SHIFT (expected≈{expected_mean_norm:.4f})")
                                else:
                                    print(f"  ✅ PASS")

                if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                    check_target_scale_consistency(V_target, G_target, mask)


                # ======== A/B TEST 1 DEBUG: kNN Coverage Metrics ========
                # Measures what fraction of nodes have their true spatial kNN neighbors
                # present in the miniset. This tells us if training has enough signal
                # to learn kNN@10 ordering.
                if global_step % 100 == 0 and (not is_sc) and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        # Get knn_spatial from batch (on CPU to avoid overhead)
                        knn_spatial_cpu = batch.get('knn_spatial', None)
                        mask_cpu = batch['mask'].cpu()
                        
                        if knn_spatial_cpu is not None:
                            if knn_spatial_cpu.is_cuda:
                                knn_spatial_cpu = knn_spatial_cpu.cpu()
                            
                            k_eval = 10  # Evaluate coverage for top-10 neighbors
                            k_avail = knn_spatial_cpu.shape[2] if knn_spatial_cpu.ndim == 3 else knn_spatial_cpu.shape[1]
                            k_eval = min(k_eval, k_avail)
                            
                            # For each batch element and each valid node, check how many of its
                            # k_eval nearest spatial neighbors are present (not -1)
                            # knn_spatial_cpu shape: (B, N, K)
                            B_size = knn_spatial_cpu.shape[0]
                            
                            total_valid_nodes = 0
                            total_present_neighbors = 0
                            total_possible_neighbors = 0
                            nodes_with_full_k = 0
                            present_counts = []
                            
                            for b_idx in range(B_size):
                                valid_mask = mask_cpu[b_idx]  # (N,) bool
                                n_valid = valid_mask.sum().item()
                                if n_valid == 0:
                                    continue
                                
                                knn_b = knn_spatial_cpu[b_idx, valid_mask, :k_eval]  # (n_valid, k_eval)
                                present = (knn_b != -1)  # (n_valid, k_eval) bool
                                
                                n_present = present.sum().item()
                                n_possible = n_valid * k_eval
                                
                                total_valid_nodes += n_valid
                                total_present_neighbors += n_present
                                total_possible_neighbors += n_possible
                                
                                # Count nodes that have all k_eval neighbors present
                                full_k_mask = present.all(dim=1)  # (n_valid,) bool
                                nodes_with_full_k += full_k_mask.sum().item()
                                
                                # Track per-node present count for median
                                present_counts.extend(present.sum(dim=1).tolist())
                            
                            if total_possible_neighbors > 0:
                                cov10 = total_present_neighbors / total_possible_neighbors
                                full10 = nodes_with_full_k / max(total_valid_nodes, 1)
                                med10 = float(torch.tensor(present_counts).float().median().item()) if present_counts else 0.0
                                
                                # Get patch_mode from dataset if available
                                patch_mode_str = getattr(st_dataset, 'patch_mode', 'unknown') if st_dataset else 'unknown'
                                
                                print(f"[KNN_COVERAGE] step={global_step} epoch={epoch} "
                                      f"cov10={cov10:.4f} full10={full10:.4f} med10={med10:.1f}/10 "
                                      f"mode={patch_mode_str}")
                # ======== END A/B TEST 1 DEBUG ========


                # DEBUG: Check which samples have correct noise
                # DEBUG: Check which samples have correct noise
                if global_step % 50 == 0 and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        for b_idx in range(V_t.shape[0]):
                            m_b = mask_orig[b_idx].bool()
                            if m_b.sum() < 5:
                                continue
                            noise_actual = (V_t[b_idx] - V_target_orig[b_idx])[m_b]
                            noise_expected = (sigma_t_orig[b_idx] * eps_orig[b_idx])[m_b]
                            max_diff = (noise_actual - noise_expected).abs().max().item()
                            if max_diff > 0.01:
                                print(f"  SAMPLE {b_idx}: max_diff={max_diff:.4f} ← CORRUPTED")


                # ========== PHASE 3: NOISE INJECTION SANITY ==========
                    
                    # === C. REALIZED NOISE vs SIGMA (after multiplication) ===
                    print(f"\n[3C] Realized Noise vs Sigma:")
                    noise_stats = []
                    # for b_idx in range(min(2, V_t.shape[0])):
                    for b_idx in range(V_t.shape[0]):
                        m_b = mask[b_idx].bool()
                        if m_b.sum() > 1:
                            noise_b = (V_t[b_idx] - V_target[b_idx])[m_b]  # shape: (n_valid, D)
                            
                            # Extract sigma for this sample (sigma is constant per sample)
                            # sigma_t shape: (B, 1, 1) or (B, 1) or (B,)
                            sig = float(sigma_t[b_idx].view(-1)[0])
                            
                            noise_point_std = noise_b.std(dim=0).mean().item()
                            noise_mean_norm = noise_b.mean(dim=0).norm().item()
                            noise_over_sigma_std = (noise_b / (sig + 1e-8)).std(dim=0).mean().item()
                            
                            noise_stats.append((noise_point_std, noise_mean_norm, noise_over_sigma_std, sig))
                                        
                    # === E. EXPECTED VARIANCE CHECK (original check) ===
                    # print(f"\n[3E] Global Variance Check:")
                    valid_mask = mask.bool().unsqueeze(-1)
                    vt_valid = V_t.masked_select(valid_mask)
                    vtar_valid = V_target.masked_select(valid_mask)
                    
                    std_target = vtar_valid.std().item()
                    std_vt = vt_valid.std().item()
                    sigma_rms = sigma_t.pow(2).mean().sqrt().item()
                    
                    expected_std = math.sqrt(std_target**2 + sigma_rms**2)
                    ratio = std_vt / expected_std
                    
                    print(f"  std(V_target): {std_target:.4f}")
                    print(f"  sigma_rms:     {sigma_rms:.4f}")
                    print(f"  std(V_t):      {std_vt:.4f}")
                    print(f"  Expected std:  {expected_std:.4f}")
                    print(f"  Ratio:         {ratio:.4f} (should be ~1.0)")
                    
                    # if not (0.9 < ratio < 1.1):
                    #     print(f"  ⚠️ WARNING: V_t scale mismatch!")
                    # else:
                    #     print(f"  ✓ PASS: Global variance matches expectation")
                    
                    print(f"{'='*70}\n")

                # --- [FIX 1] STRICT DIAGNOSTIC FOR V_T SCALING ---
                if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        # 1. Check for garbage in padding
                        invalid_mask = (~mask.bool()).unsqueeze(-1)
                        if invalid_mask.any():
                            inv_max = V_t.masked_select(invalid_mask).abs().max().item()
                            # print(f"[V_T CHECK] invalid_absmax={inv_max:.6f} (Should be 0.0)")

                        # 2. Stats on VALID entries only
                        valid_mask = mask.bool().unsqueeze(-1)
                        vt_valid = V_t.masked_select(valid_mask)
                        vtar_valid = V_target.masked_select(valid_mask)
                        
                        # 3. Compute Expected using RMS (Correct for mixed batches)
                        sigma_mean = sigma_t.mean().item()
                        sigma_rms = sigma_t.pow(2).mean().sqrt().item()
                        
                        vt_std = vt_valid.std().item()
                        vtar_std = vtar_valid.std().item()
                        
                        # Expected = sqrt(Var(Data) + E[Sigma^2])
                        expected_rms = (vtar_std**2 + sigma_rms**2)**0.5
                    # -------------------------------------------------

                # After computing H in training loop
                if global_step % 2000 == 0 and (fabric is None or fabric.is_global_zero):
                    with torch.no_grad():
                        # Check if H varies meaningfully across batch
                        H_std_per_dim = H.std(dim=0).mean()
                        H_mean_norm = H.mean(dim=0).norm()
                        print(f"[CONTEXT] H_std={H_std_per_dim:.3f} H_mean_norm={H_mean_norm:.3f}")
                        
                        # Check if conditioning makes a difference
                        V_hat_with_H = score_net(V_t, t_norm, H, mask)
                        V_hat_no_H = score_net(V_t, t_norm, torch.zeros_like(H), mask)
                        diff = (V_hat_with_H - V_hat_no_H).norm()
                        print(f"[CONTEXT] Conditional vs unconditional diff: {diff:.3f}")
                
                # DEBUG: CFG stats (first few steps)
                if DEBUG and epoch == 0 and global_step < 10:
                    n_dropped = drop_mask.sum().item()
                    print(f"[CFG] step={global_step} dropped={int(n_dropped)}/{batch_size_real} contexts")
                

                # DEBUG: CFG stats (first epoch only)
                if DEBUG and epoch == 0 and global_step < 10:
                   n_dropped = drop_mask.sum().item()
                #    print(f"[CFG] step={global_step} dropped={int(n_dropped)}/{batch_size_real} contexts")

                # === forward pass ===
                if use_edm:
                    sigma_flat = sigma_t.view(-1)
                    
                    # Phase 4: Get debug info periodically
                    return_debug = (global_step % 100 == 0) and (fabric is None or fabric.is_global_zero)

                    # =========================================================
                    # [DEBUG] IN-PLACE MODIFICATION TRAP
                    # =========================================================
                    with torch.no_grad():
                        vt_sum_before = V_t.sum().item()
                        vt_norm_before = V_t.norm().item()
                        if not is_sc:
                            tgt_sum_before = V_target.sum().item()
                    # =========================================================

                    # ========== SELF-CONDITIONING (STANDARD MODE) ==========
                    if self_cond_mode == 'standard' and score_net.self_conditioning:
                        # Standard two-pass self-conditioning (matches inference)
                        with torch.no_grad():
                            # First pass: no self-conditioning
                            x0_pred_0_result = score_net.forward_edm(
                                V_t, sigma_flat, H_train, mask, sigma_data,
                                self_cond=None, return_debug=False
                            )
                            if isinstance(x0_pred_0_result, tuple):
                                x0_pred_0_tmp = x0_pred_0_result[0]
                            else:
                                x0_pred_0_tmp = x0_pred_0_result
                        
                        # Second pass: use first pass output as self-conditioning
                        result = score_net.forward_edm(
                            V_t, sigma_flat, H_train, mask, sigma_data,
                            self_cond=x0_pred_0_tmp.detach(),  # stopgrad
                            return_debug=return_debug
                        )
                        
                        # Debug self-conditioning effectiveness
                        if global_step % 200 == 0 and (fabric is None or fabric.is_global_zero):
                            with torch.no_grad():
                                if isinstance(result, tuple):
                                    x0_pred_final = result[0]
                                else:
                                    x0_pred_final = result
                                
                                diff = (x0_pred_final - x0_pred_0_tmp).abs()
                                mask_f = mask.unsqueeze(-1).float()
                                diff_mean = (diff * mask_f).sum() / mask_f.sum()
                                x0_scale = (x0_pred_final.abs() * mask_f).sum() / mask_f.sum()
                                rel_diff = diff_mean / x0_scale.clamp(min=1e-6)
                                
                                print(f"\n[SELF-COND] step={global_step} mode={self_cond_mode}")
                                print(f"  Relative change from pass1 to pass2: {rel_diff.item()*100:.2f}%")

                    elif self_cond_mode == 'none':
                        # No self-conditioning
                        result = score_net.forward_edm(
                            V_t, sigma_flat, H_train, mask, sigma_data,
                            self_cond=None, return_debug=return_debug
                        )
                        
                    else:
                        # Fallback for old behavior (should not happen with new args)
                        result = score_net.forward_edm(
                            V_t, sigma_flat, H_train, mask, sigma_data,
                            self_cond=None, return_debug=return_debug
                        )

                                        
                    if isinstance(result, tuple):
                        x0_pred, debug_dict = result
                    else:
                        x0_pred = result
                        debug_dict = None

                    # --- DEBUG: Per-sigma scale analysis ---
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero) and not is_sc:
                        with torch.no_grad():
                            sigma_bins_debug = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, float('inf'))]
                            
                            print(f"\n[DEBUG-PRECOND] Step {global_step} - Per-sigma preconditioning analysis:")
                            for lo, hi in sigma_bins_debug:
                                in_bin = (sigma_flat >= lo) & (sigma_flat < hi)
                                if in_bin.any():
                                    # RMS of predictions and targets
                                    pred_bin = x0_pred[in_bin]
                                    tgt_bin = V_target[in_bin]
                                    mask_bin = mask[in_bin]
                                    
                                    mask_f = mask_bin.unsqueeze(-1).float()
                                    valid_count = mask_f.sum()
                                    
                                    rms_pred = (pred_bin.pow(2) * mask_f).sum().div(valid_count).sqrt().item()
                                    rms_tgt = (tgt_bin.pow(2) * mask_f).sum().div(valid_count).sqrt().item()
                                    scale_ratio = rms_pred / max(rms_tgt, 1e-8)
                                    
                                    print(f"  σ∈[{lo:.1f},{hi:.1f}): n={in_bin.sum().item()}, "
                                        f"rms_pred={rms_pred:.4f}, rms_tgt={rms_tgt:.4f}, "
                                        f"scale_ratio={scale_ratio:.3f}")


                    # --- DEBUG 3: Per-sigma edge analysis ---
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero) and not is_sc:
                        with torch.no_grad():
                            sigma_bins_debug = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, float('inf'))]
                            print(f"\n[DEBUG-EDGE] Step {global_step} - Per-sigma edge analysis:")
                            
                            for lo, hi in sigma_bins_debug:
                                in_bin = (sigma_flat >= lo) & (sigma_flat < hi)
                                if in_bin.any():
                                    # Get samples in this bin
                                    pred_bin = x0_pred[in_bin]
                                    tgt_bin = V_target[in_bin]
                                    mask_bin = mask[in_bin]
                                    
                                    # Compute edge distances
                                    D_pred = torch.cdist(pred_bin, pred_bin)
                                    D_tgt = torch.cdist(tgt_bin, tgt_bin)
                                    
                                    # Average over valid pairs
                                    mask_2d = mask_bin.unsqueeze(2) * mask_bin.unsqueeze(1)
                                    mask_2d = mask_2d.float()
                                    
                                    valid_pairs = mask_2d.sum()
                                    if valid_pairs > 0:
                                        mean_pred = (D_pred * mask_2d).sum() / valid_pairs
                                        mean_tgt = (D_tgt * mask_2d).sum() / valid_pairs
                                        ratio = (mean_pred / mean_tgt.clamp(min=1e-6)).item()
                                        print(f"  σ∈[{lo:.1f},{hi:.1f}): n={in_bin.sum().item()}, "
                                              f"D_pred={mean_pred.item():.4f}, D_tgt={mean_tgt.item():.4f}, "
                                              f"ratio={ratio:.3f}")

                    
                    # --- DEBUG 6: Self-conditioning effectiveness check ---
                    if global_step % 100 == 0 and use_self_cond and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            print(f"\n[DEBUG-SELFCOND] Step {global_step} - Self-cond effectiveness:")
                            
                            # Compare x0_pred with and without self-cond for same input
                            x0_no_sc = score_net.forward_edm(V_t, sigma_flat, H_train, mask, sigma_data, 
                                                             self_cond=None)
                            if isinstance(x0_no_sc, tuple):
                                x0_no_sc = x0_no_sc[0]
                            
                            # Difference caused by self-conditioning
                            diff = (x0_pred - x0_no_sc).abs()
                            mask_f = mask.unsqueeze(-1).float()
                            diff_mean = (diff * mask_f).sum() / mask_f.sum()
                            x0_scale = (x0_pred.abs() * mask_f).sum() / mask_f.sum()
                            rel_diff = diff_mean / x0_scale.clamp(min=1e-6)
                            
                            print(f"  Self-cond relative effect: {rel_diff.item()*100:.2f}%")
                            
                            # Per-sigma breakdown
                            for lo, hi in [(0.0, 0.3), (0.3, 1.0)]:
                                in_bin = (sigma_flat >= lo) & (sigma_flat < hi)
                                if in_bin.any():
                                    diff_bin = diff[in_bin]
                                    mask_bin = mask[in_bin].unsqueeze(-1).float()
                                    diff_bin_mean = (diff_bin * mask_bin).sum() / mask_bin.sum()
                                    print(f"  σ∈[{lo:.1f},{hi:.1f}): self-cond effect={diff_bin_mean.item():.4f}")

                    # === DEBUG-SELFCOND-QUALITY: Per σ-bin quality tracking ===
                    if global_step % 200 == 0 and use_self_cond and (fabric is None or fabric.is_global_zero) and not is_sc:
                        with torch.no_grad():
                            print(f"\n[DEBUG-SELFCOND-QUALITY] Step {global_step}:")
                            
                            # Helper: per-sample centered MSE (CORRECTED)
                            def _batch_centered_mse(A, B, m):
                                """Compute MSE after centering each sample individually."""
                                mf = m.unsqueeze(-1).float()  # (Bbin, N, 1)
                                denom = mf.sum(dim=1, keepdim=True).clamp(min=1)  # (Bbin, 1, 1)
                                A_c = A - (A * mf).sum(dim=1, keepdim=True) / denom
                                B_c = B - (B * mf).sum(dim=1, keepdim=True) / denom
                                # Per-sample MSE, then average
                                per_sample_mse = ((A_c - B_c).pow(2) * mf).sum(dim=(1,2)) / denom.squeeze(-1).squeeze(-1).clamp(min=1)
                                return per_sample_mse.mean()
                            
                            sigma_bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.5), (1.5, float('inf'))]
                            
                            for lo, hi in sigma_bins:
                                in_bin = (sigma_flat >= lo) & (sigma_flat < hi)
                                if not in_bin.any():
                                    continue
                                
                                n_in_bin = in_bin.sum().item()
                                tgt_bin = V_target[in_bin]
                                pred_bin = x0_pred[in_bin]
                                mask_bin = mask[in_bin]
                                
                                mse_pred = _batch_centered_mse(pred_bin, tgt_bin, mask_bin).item()
                                
                                print(f"  σ∈[{lo:.1f},{hi:.1f}): n={n_in_bin}, mse(pred→tgt)={mse_pred:.6f}")

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
                        result = score_net(V_t, t_norm.unsqueeze(1), H_train, mask, self_cond=None, return_dist_aux=(not is_sc))
                        if isinstance(result, tuple):
                            eps_pred, dist_aux = result
                        else:
                            eps_pred = result
                            dist_aux = None

                    V_hat = V_t - sigma_t * eps_pred
                    V_hat = V_hat * mask.unsqueeze(-1).float()
       
            # ===== SCORE LOSS (in fp32 for numerical stability) =====
            with torch.autocast(device_type='cuda', enabled=False):
                mask_fp32 = mask.float()
                if use_edm:
                    # EDM: Loss on denoised prediction
                    # EDM: Loss on denoised prediction
                    x0_pred_fp32 = x0_pred.float()
                    sigma_flat = sigma_t.view(-1).float()  # (B,)

                    # EDM weight: (σ² + σ_d²) / (σ · σ_d)²  (returns (B,))
                    w_raw = uet.edm_loss_weight(sigma_flat, sigma_data).float()

                    # --- Adaptive soft-cap (data-driven) instead of fixed 50 ---
                    # This prevents low-σ explosion while avoiding a hard plateau at a fixed constant.
                    # Use EMA of a high quantile as the cap so it is stable across steps.
                    q = 0.95
                    cap_now = torch.quantile(w_raw.detach(), q)

                    # Initialize EMA state if needed
                    if edm_debug_state.get('w_cap_ema', None) is None:
                        edm_debug_state['w_cap_ema'] = cap_now
                    else:
                        ema = 0.98
                        edm_debug_state['w_cap_ema'] = ema * edm_debug_state['w_cap_ema'] + (1.0 - ema) * cap_now

                    cap = edm_debug_state['w_cap_ema'].clamp_min(1e-6)
                    w = cap * torch.tanh(w_raw / cap)  # (B,)

                    # Choose target: ST uses V_target; SC must not require coords.
                    # If your SC path truly has V_target, keep it; otherwise revert to V_0.
                    target_x0 = V_target if (not is_sc) else V_0
                    target_fp32 = target_x0.float()

                    # Masked MSE (B, N) mean over latent dims
                    # Masked MSE (B, N) mean over latent dims
                    err2_node = (x0_pred_fp32 - target_fp32).pow(2).mean(dim=-1)  # (B, N)

                    # ========== NEW: Anchored training - mask score loss to unknown points ==========
                    mask_score = mask_fp32
                    if anchor_train and anchor_mask_score_loss and anchor_cond_mask is not None and not is_sc:
                        anchor_cond_mask_fp32 = anchor_cond_mask.float()
                        mask_score = mask_fp32 * (1.0 - anchor_cond_mask_fp32)
                        
                        # Debug: track anchor vs unknown MSE
                        if global_step % anchor_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                            with torch.no_grad():
                                den_valid = mask_fp32.sum(dim=1).clamp_min(1.0)
                                den_unknown = mask_score.sum(dim=1).clamp_min(1.0)
                                
                                # MSE on anchors only
                                mask_anchor = mask_fp32 * anchor_cond_mask_fp32
                                den_anchor = mask_anchor.sum(dim=1).clamp_min(1.0)
                                mse_anchor = (err2_node * mask_anchor).sum(dim=1) / den_anchor
                                
                                # MSE on unknown only
                                mse_unknown = (err2_node * mask_score).sum(dim=1) / den_unknown
                                
                                print(f"  [ANCHOR-SCORE] den_valid_mean={den_valid.mean():.1f} "
                                      f"den_unknown_mean={den_unknown.mean():.1f}")
                                print(f"  [ANCHOR-ERR] mse_anchor={mse_anchor.mean():.6f} "
                                      f"mse_unknown={mse_unknown.mean():.6f} "
                                      f"(anchor should be tiny)")

                    # --- IMPORTANT: per-sample normalization (avoid variable-n set size bias) ---
                    den = mask_score.sum(dim=1).clamp_min(1.0)                         # (B,)
                    err2_sample = (err2_node * mask_score).sum(dim=1) / den            # (B,)

                    # ==============================================================================
                    # A/B 2: SCORE LOSS WEIGHTING EXPERIMENTS
                    # ==============================================================================
                    # Two separate effects:
                    #   - WNORM: normalize by w.sum (removes global w scaling, key fix)
                    #   - HI_BOOST: boost high-noise samples (distribution shaping)
                    
                    # Compute optional gate_hi_score only if we need it (boost or Fx-hi)
                    gate_hi_score = None
                    if (EXP_SCORE_HI_BOOST or EXP_SCORE_FX_HI):
                        with torch.no_grad():
                            c_skip_hi = (sigma_data ** 2) / (sigma_flat ** 2 + sigma_data ** 2 + 1e-12)
                            noise_score_hi = -torch.log(c_skip_hi + 1e-12)  # (B,)
                            score_hi_gate.update(noise_score_hi)
                            gate_hi_score = score_hi_gate.gate(noise_score_hi, torch.ones_like(noise_score_hi))
                    
                    # Start from base weights
                    w_eff = w
                    
                    # Optional boost with data-driven readiness (CHANGE 2)
                    # Optional boost with data-driven readiness (CHANGE 2)
                    if EXP_SCORE_HI_BOOST:
                        # Update boost readiness from Fx scale at high-σ
                        with torch.no_grad():
                            fx_ratio = boost_state.get('last_fx_ratio_hi', None)
                            
                            if fx_ratio is not None and fx_ratio > 0:
                                # EMA update
                                boost_state['ema_fx_ratio_hi'] = (
                                    boost_state['ema_decay'] * boost_state['ema_fx_ratio_hi'] +
                                    (1 - boost_state['ema_decay']) * fx_ratio
                                )
                                
                                # Check stability using log error (scale-invariant)
                                fx_err = abs(math.log(boost_state['ema_fx_ratio_hi']))
                                is_stable = (fx_err < boost_state['stability_tol'])
                                
                                if is_stable:
                                    boost_state['stable_count'] += 1
                                else:
                                    boost_state['stable_count'] = 0
                                
                                # Trigger readiness
                                if (not boost_state['ready'] and 
                                    boost_state['stable_count'] >= boost_state['min_stable_checks']):
                                    boost_state['ready'] = True
                                    boost_state['start_step'] = global_step
                                    if fabric is None or fabric.is_global_zero:
                                        print(f"\n[BOOST-READY] Boost activated at step {global_step}!")
                                        print(f"  ema_fx_ratio_hi={boost_state['ema_fx_ratio_hi']:.4f}")
                        
                        # Compute boost ramp
                        if not boost_state['ready']:
                            ramp = 0.0  # Boost disabled until ready
                        else:
                            steps_since_ready = global_step - boost_state['start_step']
                            ramp = min(1.0, steps_since_ready / max(1, SCORE_HI_BOOST_RAMP))
                        
                        # Base boost
                        base_boost = 1.0 + ramp * (SCORE_HI_BOOST_FACTOR - 1.0) * gate_hi_score  # (B,)
                        
                        # ==================== CHANGE 7B: TAIL SAFETY CAP (Option 1 Change C) ====================
                        # Cap boost for extreme tail samples to prevent runaway
                        if TAIL_SAFETY_ENABLED:
                            sigma_tail_threshold = sigma_flat.quantile(TAIL_QUANTILE)
                            is_tail = (sigma_flat >= sigma_tail_threshold)  # (B,) bool
                            
                            # Cap boost for tail samples
                            tail_boost_cap = 1.0 + ramp * (TAIL_BOOST_CAP - 1.0) * gate_hi_score
                            boost = torch.where(is_tail, 
                                               torch.minimum(base_boost, tail_boost_cap),
                                               base_boost)
                        else:
                            boost = base_boost
                            is_tail = torch.zeros_like(sigma_flat, dtype=torch.bool)
                        
                        w_eff = w_eff * boost

                        
                        # Debug logging
                        # Debug logging
                        if (global_step % 200 == 0) and (fabric is None or fabric.is_global_zero):
                            thr_val = score_hi_gate.thr
                            thr_str = f"{thr_val:.3f}" if thr_val is not None else "warmup"
                            print(f"[BOOST-READY] step={global_step} ready={boost_state['ready']} "
                                  f"ema_fx={boost_state['ema_fx_ratio_hi']:.3f} "
                                  f"stable_count={boost_state['stable_count']} "
                                  f"ramp={ramp:.2f}")
                            print(f"[SCORE-HI-BOOST] step={global_step} ramp={ramp:.2f} "
                                  f"thr={thr_str} "
                                  f"hit={(gate_hi_score>0).float().mean().item():.2%} "
                                  f"boost_mean={boost.mean().item():.3f}")
                            
                            # CHANGE 7C: Tail safety debug
                            if TAIL_SAFETY_ENABLED:
                                n_tail = is_tail.sum().item()
                                if n_tail > 0:
                                    tail_boost_mean = boost[is_tail].mean().item()
                                    print(f"[TAIL-SAFETY] n_tail={n_tail} tail_boost_mean={tail_boost_mean:.3f} "
                                          f"(capped at {TAIL_BOOST_CAP:.1f}x vs {SCORE_HI_BOOST_FACTOR:.1f}x)")



                    
                    # Now choose how to aggregate score loss
                    if EXP_SCORE_WNORM:
                        # Weighted average of err2 (relative weights only, removes global w scaling)
                        L_score = (w_eff * err2_sample).sum() / w_eff.sum().clamp(min=1e-8)
                    else:
                        # Classic EDM objective estimate
                        L_score = (w_eff * err2_sample).mean()
                    
                    # Debug: Log WNORM effect
                    if (global_step % 200 == 0) and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            L_score_mean = (w_eff * err2_sample).mean().item()
                            L_score_wnorm = ((w_eff * err2_sample).sum() / w_eff.sum().clamp(min=1e-8)).item()
                            print(f"[SCORE-AGG] step={global_step} L_score_mean={L_score_mean:.4f} "
                                  f"L_score_wnorm={L_score_wnorm:.4f} ratio={L_score_mean/max(L_score_wnorm,1e-8):.2f} "
                                  f"w_sum={w_eff.sum().item():.1f} w_mean={w_eff.mean().item():.2f}")


                    
                    # ==============================================================================
                    # A/B 3: F_x-SPACE SUPERVISION FOR HIGH-NOISE SAMPLES
                    # ==============================================================================
                    # At high σ, x0_pred ≈ c_out * F_x (learned branch).
                    # Directly supervise F_x to ensure the learned branch produces correct scale.
                    
                    if EXP_SCORE_FX_HI and use_edm:
                        with torch.autocast(device_type='cuda', enabled=False):
                            # Get EDM scalars for this batch
                            sigma_flat_fx = sigma_t.view(-1).float()
                            c_skip_fx, c_out_fx, _, _ = uet.edm_precond(sigma_flat_fx, sigma_data)
                            c_skip_fx = c_skip_fx.view(-1, 1, 1)  # (B, 1, 1)
                            c_out_fx = c_out_fx.view(-1, 1, 1)    # (B, 1, 1)
                            
                            # Center the noisy input and predictions
                            V_c_fx, _ = uet.center_only(V_t.float(), mask)
                            V_c_fx = V_c_fx * mask.float().unsqueeze(-1)
                            
                            x0_pred_c_fx, _ = uet.center_only(x0_pred_fp32, mask)
                            x0_tgt_c_fx, _ = uet.center_only(target_fp32, mask)
                            x0_pred_c_fx = x0_pred_c_fx * mask.float().unsqueeze(-1)
                            x0_tgt_c_fx = x0_tgt_c_fx * mask.float().unsqueeze(-1)
                            
                            # Compute F_x: the learned branch output in EDM parameterization
                            # x0_pred = c_skip * V_c + c_out * F_x  =>  F_x = (x0_pred - c_skip * V_c) / c_out
                            eps_fx = 1e-8
                            F_pred = (x0_pred_c_fx - c_skip_fx * V_c_fx) / (c_out_fx + eps_fx)
                            F_tgt = (x0_tgt_c_fx - c_skip_fx * V_c_fx) / (c_out_fx + eps_fx)
                            
                            # Per-sample F_x MSE
                            diff_fx = (F_pred - F_tgt)
                            err2_fx_point = diff_fx.pow(2).mean(dim=-1)  # (B, N)
                            err2_fx_sample = (err2_fx_point * mask_fp32).sum(dim=1) / den  # (B,)
                            
                            # Match units to x0-space if desired (multiply by c_out^2)
                            if FX_USE_COUT2_MATCH:
                                cout2 = (c_out_fx.view(-1) ** 2).detach()  # (B,)
                                err2_fx_sample_u = err2_fx_sample * cout2
                            else:
                                err2_fx_sample_u = err2_fx_sample
                            
                            # Sanitize before gating
                            err2_fx_sample_u = torch.nan_to_num(err2_fx_sample_u, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Apply gate: only high-noise samples contribute
                            w_fx = w  # Reuse EDM weights
                            gate_fx = gate_hi_score
                            
                            num_fx = (w_fx * err2_fx_sample_u * gate_fx).sum()
                            den_fx = (w_fx * gate_fx).sum().clamp(min=1e-8)
                            L_fx_hi = num_fx / den_fx
                            
                            # Add to score loss
                            L_score = L_score + FX_HI_WEIGHT * L_fx_hi
                            
                            # Debug
                            if (global_step % 200 == 0) and (fabric is None or fabric.is_global_zero):
                                print(f"[SCORE-FX-HI] step={global_step} "
                                      f"hit={(gate_fx>0).float().mean().item():.2%} "
                                      f"L_fx_hi={L_fx_hi.item():.4f} FX_HI_WEIGHT={FX_HI_WEIGHT}")

                    # ==============================================================================
                    # DEBUG: PER-NOISE-BIN SCORE CONTRIBUTION
                    # ==============================================================================
                    # This tells us if high-σ is actually getting meaningful gradient.
                    # If HIGH has tiny (w*err2)_mean relative to LOW, the training is imbalanced.
                    
                    if (global_step % 200 == 0) and (fabric is None or fabric.is_global_zero) and use_edm:
                        with torch.no_grad():
                            # Compute noise_score locally (same formula as later in the code)
                            c_skip_debug = (sigma_data ** 2) / (sigma_flat ** 2 + sigma_data ** 2 + 1e-12)
                            ns = -torch.log(c_skip_debug + 1e-12)  # (B,) higher = noisier

                            # 3 bins: low/mid/high noise by quantiles
                            q1 = torch.quantile(ns, 0.33)
                            q2 = torch.quantile(ns, 0.66)
                            
                            print(f"\n[SCORE-BINS] Step {global_step} - Per-noise-level score contribution:")
                            for name, sel in [
                                ("LOW ", ns <= q1),
                                ("MID ", (ns > q1) & (ns <= q2)),
                                ("HIGH", ns > q2),
                            ]:
                                if sel.any():
                                    n_sel = sel.float().sum().item()
                                    w_sel = w[sel].mean().item()
                                    err2_sel = err2_sample[sel].mean().item()
                                    werr2_sel = (w[sel] * err2_sample[sel]).mean().item()
                                    
                                    # Also compute sigma stats for this bin
                                    sigma_sel = sigma_flat[sel]
                                    sigma_min_sel = sigma_sel.min().item()
                                    sigma_max_sel = sigma_sel.max().item()
                                    
                                    print(f"  {name}: n={n_sel:4.0f} σ=[{sigma_min_sel:.3f},{sigma_max_sel:.3f}] "
                                          f"w_mean={w_sel:6.2f} err2_mean={err2_sel:.4f} "
                                          f"(w*err2)_mean={werr2_sel:.4f}")
                            
                            # Warn if HIGH has much smaller contribution than LOW
                            sel_low = ns <= q1
                            sel_high = ns > q2
                            if sel_low.any() and sel_high.any():
                                werr2_low = (w[sel_low] * err2_sample[sel_low]).mean().item()
                                werr2_high = (w[sel_high] * err2_sample[sel_high]).mean().item()
                                ratio = werr2_high / max(werr2_low, 1e-8)
                                if ratio < 0.3:
                                    print(f"  ⚠️ HIGH/LOW gradient ratio = {ratio:.2f} (<0.3) - high-σ may be under-trained!")

                    # FIX 4: Check if L_score is the NaN source
                    if not torch.isfinite(L_score):
                        print(f"[NAN-SOURCE] L_score is NaN/Inf at step {global_step}")
                        print(f"  sigma range: [{sigma_t.min().item():.4f}, {sigma_t.max().item():.4f}]")
                        print(f"  x0_pred range: [{x0_pred.min().item():.4f}, {x0_pred.max().item():.4f}]")
                        print(f"  V_target range: [{V_target.min().item():.4f}, {V_target.max().item():.4f}]")
                        # L_score = torch.tensor(0.0, device=device)  # Replace with 0


                    # ========== PHASE 6: PER-SIGMA BIN ACCUMULATION ==========
                    # FIX: use sigma_bin_edges key, not sigma_bins (avoid re-init every step)
                    if edm_debug_state.get('sigma_bin_edges', None) is None:
                        n_bins = 8

                        # Build stable edges from a warmup buffer (first few steps), then freeze.
                        # This avoids computing quantiles from a single batch (too noisy).
                        edm_debug_state['log_sigma_buffer'] = []
                        edm_debug_state['sigma_bin_sum_err2'] = torch.zeros(n_bins)
                        edm_debug_state['sigma_bin_sum_w'] = torch.zeros(n_bins)
                        edm_debug_state['sigma_bin_sum_werr2'] = torch.zeros(n_bins)
                        edm_debug_state['sigma_bin_count'] = torch.zeros(n_bins)

                    # Append to warmup buffer until we have enough, then create edges once
                    if edm_debug_state.get('sigma_bin_edges', None) is None:
                        edm_debug_state['log_sigma_buffer'].append(sigma_flat.log().detach().cpu())

                        # Tune this buffer size if needed; 1024 gives stable quantiles.
                        if sum(x.numel() for x in edm_debug_state['log_sigma_buffer']) >= 1024:
                            log_sigma_all = torch.cat(edm_debug_state['log_sigma_buffer'], dim=0)
                            quantiles = torch.linspace(0, 1, n_bins + 1)
                            bin_edges = torch.quantile(log_sigma_all, quantiles)
                            edm_debug_state['sigma_bin_edges'] = bin_edges
                            edm_debug_state.pop('log_sigma_buffer', None)

                    # Accumulate per bin once edges exist
                    if edm_debug_state.get('sigma_bin_edges', None) is not None:
                        bin_edges = edm_debug_state['sigma_bin_edges']

                        log_sigma_batch = sigma_flat.log().detach().cpu()          # (B,)
                        err2_batch = err2_sample.detach().cpu()                    # (B,)
                        w_batch = w.detach().cpu()                                 # (B,)
                        werr2_batch = w_batch * err2_batch                         # (B,)

                        for b_idx in range(log_sigma_batch.numel()):
                            log_s = log_sigma_batch[b_idx]
                            bin_id = torch.searchsorted(bin_edges, log_s, right=False) - 1
                            bin_id = torch.clamp(bin_id, 0, n_bins - 1).item()

                            edm_debug_state['sigma_bin_sum_err2'][bin_id] += err2_batch[b_idx]
                            edm_debug_state['sigma_bin_sum_w'][bin_id] += w_batch[b_idx]
                            edm_debug_state['sigma_bin_sum_werr2'][bin_id] += werr2_batch[b_idx]
                            edm_debug_state['sigma_bin_count'][bin_id] += 1


                else:
                    sigma_t_fp32 = sigma_t.float()
                    eps_pred_fp32 = eps_pred.float()
                    eps_fp32 = eps.float()
                    mask_fp32 = mask.float()

                    # ===== CORRECT EDM WEIGHTING FOR NOISE PREDICTION =====
                    sigma_t_squeezed = sigma_t_fp32.squeeze(-1)  # (B, N) or (B,)
                    if sigma_t_squeezed.dim() == 1:
                        sigma_t_squeezed = sigma_t_squeezed.unsqueeze(-1)  # (B, 1)

                    
                    # 1. Compute a data-driven floor from the current batch
                    # This prevents weight explosion at very small sigmas (e.g. 0.002)
                    # We take the 20th percentile of sigmas in this batch as the "max weight" threshold
                    sigma_vals = sigma_t_squeezed.detach().view(-1) # Flatten to (B,)
                    
                    if sigma_vals.numel() > 1:
                        # Use 20th percentile as floor
                        sigma_floor = torch.quantile(sigma_vals, 0.2)
                    else:
                        # Fallback for batch_size=1
                        sigma_floor = sigma_vals.min()
                    
                    # 2. Compute weights: w = 1 / max(sigma, floor)^2
                    # This gives high weight to low noise, but caps it at the floor level
                    w = 1.0 / (sigma_t_squeezed ** 2).clamp(min=sigma_floor**2)
                    w = w.clamp(max=10.0)
                    
                    # Ensure broadcasting works: w needs to be (B, 1) to match err2 (B, N)
                    if w.dim() == 1:
                        w = w.view(-1, 1)
                    
                    # Diagnostic print (optional, can remove later)
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                        print(f"[WEIGHTS] sigma_floor={sigma_floor:.4f} | "
                              f"w_min={w.min().item():.2f} w_max={w.max().item():.2f}")
                    # =========================================================

                    # Compute loss
                    err2 = (eps_pred_fp32 - eps_fp32).pow(2).mean(dim=2)  # (B, N)

                    # --- [FIX 3] LOSS SCALE CHECK ---
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            # Mean MSE over valid nodes only
                            mse_valid = err2[mask.bool()].mean().item()
                            print(f"[LOSS CHECK] Raw MSE (valid nodes only): {mse_valid:.4f} (Should be ~1.0 if sigma_data correct)")
                    # --------------------------------

                    # Handle broadcasting
                    if w.shape[-1] == 1:
                        w = w.expand_as(err2)

                    L_score = (w * err2 * mask_fp32).sum() / mask_fp32.sum()

                    # ===== DIAGNOSTICS (ADD THIS) =====
                    # DIAGNOSTIC 3: V_t scale vs sigma
                    if global_step % 20 == 0 and (fabric is None or fabric.is_global_zero):
                        sigma_mean_val = sigma_t.mean().item()
                        v_t_std_val = V_t.std().item()
                        expected_std = (V_target.std().item()**2 + sigma_mean_val**2)**0.5
                        print(f"[V_T] sigma_mean={sigma_mean_val:.3f} V_t_std={v_t_std_val:.3f} "
                            f"expected≈{expected_std:.3f} ratio={v_t_std_val/expected_std:.3f}")
                        
                    # # After computing L_score in non-EDM branch
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            # Flatten but keep only VALID entries
                            # eps and eps_pred are (B, N, D)
                            # mask is (B, N) -> expand to (B, N, D)
                            m_bool = mask.bool().unsqueeze(-1).expand_as(eps_fp32)
                            
                            eps_valid = eps_fp32.masked_select(m_bool)
                            pred_valid = eps_pred_fp32.masked_select(m_bool)
                            
                            if eps_valid.numel() > 0:
                                # Pearson correlation on valid data only
                                mean_e = eps_valid.mean()
                                mean_p = pred_valid.mean()
                                num = ((eps_valid - mean_e) * (pred_valid - mean_p)).mean()
                                den = eps_valid.std() * pred_valid.std() + 1e-8
                                corr = num / den
                                
                                print(f"[NOISE DIAG] Masked Correlation: {corr.item():.4f}")
                                print(f"   eps_std_valid={eps_valid.std().item():.4f}")
                                print(f"   pred_std_valid={pred_valid.std().item():.4f}")
                    # ---------------------------------------------

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
            L_out_scale = torch.tensor(0.0, device=device)  # NEW: Learned-branch scale calibration
            L_gram_learn = torch.tensor(0.0, device=device)  # ACTION 4: Learned-branch geom emphasis
            L_knn_nca = torch.tensor(0.0, device=device)
            L_knn_scale = torch.tensor(0.0, device=device)  # NEW
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
            L_subspace = torch.tensor(0.0, device=device)
            L_gen_scale = torch.tensor(0.0, device=device)
            L_ctx_edge = torch.tensor(0.0, device=device)  # NEW: context invariance
            # Safe loss wrapper - apply to ALL losses before using them
            L_gram = safe_loss(L_gram, "L_gram", max_val=50.0, global_step=global_step)
            L_gram_scale = safe_loss(L_gram_scale, "L_gram_scale", max_val=50.0, global_step=global_step)
            L_out_scale = safe_loss(L_out_scale, "L_out_scale", max_val=50.0, global_step=global_step)
            L_gram_learn = safe_loss(L_gram_learn, "L_gram_learn", max_val=50.0, global_step=global_step)
            L_knn_nca = safe_loss(L_knn_nca, "L_knn_nca", max_val=50.0, global_step=global_step)
            L_heat = safe_loss(L_heat, "L_heat", max_val=50.0, global_step=global_step)
            L_sw_st = safe_loss(L_sw_st, "L_sw_st", max_val=50.0, global_step=global_step)
            L_sw_sc = safe_loss(L_sw_sc, "L_sw_sc", max_val=50.0, global_step=global_step)
            L_overlap = safe_loss(L_overlap, "L_overlap", max_val=50.0, global_step=global_step)
            L_ordinal_sc = safe_loss(L_ordinal_sc, "L_ordinal_sc", max_val=50.0, global_step=global_step)
            L_st_dist = safe_loss(L_st_dist, "L_st_dist", max_val=50.0, global_step=global_step)
            L_edm_tail = safe_loss(L_edm_tail, "L_edm_tail", max_val=50.0, global_step=global_step)
            L_gen_align = safe_loss(L_gen_align, "L_gen_align", max_val=50.0, global_step=global_step)
            L_dim = safe_loss(L_dim, "L_dim", max_val=50.0, global_step=global_step)
            L_triangle = safe_loss(L_triangle, "L_triangle", max_val=50.0, global_step=global_step)
            L_radial = safe_loss(L_radial, "L_radial", max_val=50.0, global_step=global_step)
            L_repel = safe_loss(L_repel, "L_repel", max_val=50.0, global_step=global_step)
            L_shape = safe_loss(L_shape, "L_shape", max_val=50.0, global_step=global_step)
            L_edge = safe_loss(L_edge, "L_edge", max_val=50.0, global_step=global_step)
            L_topo = safe_loss(L_topo, "L_topo", max_val=50.0, global_step=global_step)
            L_shape_spec = safe_loss(L_shape_spec, "L_shape_spec", max_val=50.0, global_step=global_step)
            L_subspace = safe_loss(L_subspace, "L_subspace", max_val=50.0, global_step=global_step)
            L_gen_scale = safe_loss(L_gen_scale, "L_gen_scale", max_val=50.0, global_step=global_step)

            # ==================== NEW GLOBAL GEOMETRY BLOCK ====================
            # ==================== NEW GLOBAL GEOMETRY BLOCK ====================
            # ChatGPT EDM8-style approach:
            #   V_hat_centered = raw centered prediction → SCALE supervision (knn_scale only)
            #   V_geom = centered + locally clamped → STRUCTURE supervision (Gram, Edge, NCA, etc.)
            #
            # WHY: Structure losses should focus on shape/neighborhood, not waste gradients on scale.
            #      Scale errors are corrected by knn_scale loss on raw (unclamped) tensor.
            #      Using same tensor for both creates "split-brain" supervision (Option 2 failure).
            
            with torch.autocast(device_type='cuda', enabled=False):
                V_hat_f32 = V_hat.float()
                m_bool = mask.bool()
                m_float = mask.float().unsqueeze(-1)
                
                # --- Step A: Compute V_hat_centered (raw centered, for knn_scale ONLY) ---
                valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean = (V_hat_f32 * m_float).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
                V_hat_centered = (V_hat_f32 - mean) * m_float  # (B,N,D), centered but NOT clamped
                
                # --- Step B: Compute V_geom (centered + locally clamped, for STRUCTURE losses) ---
                # Use kNN-based local scale correction (detached) if we have spatial kNN and this is ST
                knn_spatial_for_scale = batch.get('knn_spatial', None)
                
                if (not is_sc) and knn_spatial_for_scale is not None:
                    knn_spatial_for_scale = knn_spatial_for_scale.to(device)
                    
                    # --- ADAPTIVE CLAMP CAP (ChatGPT Change 3) ---
                    # Don't ramp max_log_correction while pin-rate is terrible (>60%).
                    # Only allow ramping once model starts learning proper scale.
                    clamp_warmup_steps = 2000
                    max_log_base = 0.25  # ~1.28x at start
                    max_log_final = 0.50  # ~1.65x at end
                                        
                    # Only allow ramp if pin-rate has improved below threshold
                    pin_rate_threshold = 0.60
                    if last_pin_rate <= pin_rate_threshold:
                        # Pin-rate is acceptable, allow normal ramping
                        clamp_progress = min(1.0, global_step / clamp_warmup_steps)
                        max_log_correction = max_log_base + clamp_progress * (max_log_final - max_log_base)
                    else:
                        # Pin-rate still high, keep max_log_correction at base (don't let clamp hide problem)
                        max_log_correction = max_log_base
                    
                    # Compute local scale correction from kNN edges (DETACHED)
                    # Now returns 4 values including log_s_unclamped
                    s_corr, ratio_raw, valid_scale, log_s_unclamped, edges_used, neighbor_frac = uet.compute_local_scale_correction(
                        V_pred=V_hat_centered,
                        V_tgt=V_target.float(),
                        mask=mask,
                        knn_indices=knn_spatial_for_scale,
                        k=15,
                        max_log_correction=max_log_correction,
                    )
                    
                    # Apply scale correction (detached) to get V_geom
                    V_geom = uet.apply_scale_correction(V_hat_centered, s_corr.detach(), mask)
                    
                    # --- STORE VARIABLES FOR KNN_SCALE PIN-UPWEIGHT (ChatGPT Change 2) ---
                    # Use log_s_unclamped directly for pinned detection (more precise than ratio_raw)
                    log_s_unclamped_for_knn = log_s_unclamped.detach()
                    valid_scale_for_knn = valid_scale
                    max_log_corr_for_knn = float(max_log_correction)
                    edges_used_for_knn = edges_used
                    neighbor_frac_for_knn = neighbor_frac

                    # ========== ANCHOR-AWARE STRUCTURE TENSORS ==========
                    # Clamp anchor points in structure tensors so they cannot move from structure losses
                    if anchor_train and anchor_geom_losses and anchor_cond_mask is not None and (not is_sc):
                        # Compute unknown mask
                        unknown_mask = mask & (~anchor_cond_mask.bool())
                        n_unknown = unknown_mask.sum(dim=1)
                        
                        # Only use anchor_geom if we have enough unknown points
                        use_anchor_geom = (n_unknown.median() >= anchor_geom_min_unknown) and anchor_cond_mask.any()
                        
                        if use_anchor_geom:
                            # Center the target in the same frame as V_geom for proper clamping
                            V_target_centered, _ = uet.center_only(V_target.float(), mask)
                            V_target_centered = V_target_centered * m_float
                            
                            # Create anchor-clamped versions of structure tensors
                            # V_geom_L: for Gram, Edge, NCA (structure losses)
                            V_geom_L = uet.clamp_anchors_for_loss(V_geom, V_target_centered, anchor_cond_mask, mask)
                            
                            # V_hat_centered_L: for knn_scale (raw scale loss needs clamping too)
                            V_hat_centered_L = uet.clamp_anchors_for_loss(V_hat_centered, V_target_centered, anchor_cond_mask, mask)
                            
                            # Debug logging
                            if global_step % anchor_geom_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                                with torch.no_grad():
                                    n_anchor_mean = anchor_cond_mask.float().sum(dim=1).mean().item()
                                    n_unknown_mean = unknown_mask.float().sum(dim=1).mean().item()
                                    
                                    # Verify anchor clamping worked (error should be ~0)
                                    anchor_mask_3d = anchor_cond_mask.unsqueeze(-1)
                                    max_anchor_err_geom = (V_geom_L - V_target_centered).abs()[anchor_mask_3d.expand_as(V_geom_L)].max().item() if anchor_cond_mask.any() else 0.0
                                    max_anchor_err_scale = (V_hat_centered_L - V_target_centered).abs()[anchor_mask_3d.expand_as(V_hat_centered_L)].max().item() if anchor_cond_mask.any() else 0.0
                                    
                                    print(f"\n{uet.ANCHOR_TAG} [GEOM] step={global_step}")
                                    print(f"  anchors_mean={n_anchor_mean:.1f} unknown_mean={n_unknown_mean:.1f} use_anchor_geom={use_anchor_geom}")
                                    print(f"  V_geom_L max_anchor_err={max_anchor_err_geom:.6f} (should be ~0)")
                                    print(f"  V_hat_centered_L max_anchor_err={max_anchor_err_scale:.6f} (should be ~0)")
                        else:
                            # Not enough unknown points, use original tensors
                            V_geom_L = V_geom
                            V_hat_centered_L = V_hat_centered
                            use_anchor_geom = False
                    else:
                        # No anchor training, use original tensors
                        V_geom_L = V_geom
                        V_hat_centered_L = V_hat_centered
                        use_anchor_geom = False

                    
                    # --- POST-CLAMP DIAGNOSTIC (ChatGPT updated) ---
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            # Compute ratio_post (after clamp) for comparison
                            _, ratio_post, _, _, _, _ = uet.compute_local_scale_correction(
                                V_pred=V_geom,
                                V_tgt=V_target.float(),
                                mask=mask,
                                knn_indices=knn_spatial_for_scale,
                                k=15,
                                max_log_correction=max_log_correction,
                            )
                            
                            n_valid_samples = valid_scale.sum().item()
                            
                            # Compute pin-rate from log_s_unclamped directly (more precise)
                            # Pinned = |log_s_unclamped| >= max_log_correction (within tolerance)
                            tol = 0.01
                            valid_log_s = log_s_unclamped[valid_scale]
                            if valid_log_s.numel() > 0:
                                pinned_frac = ((valid_log_s.abs() >= (max_log_correction - tol)).sum().float() / 
                                              max(n_valid_samples, 1)).item()
                            else:
                                pinned_frac = 0.0
                            
                            # Update stored pin_rate for adaptive clamp cap
                            last_pin_rate = pinned_frac
                            
                            print(f"\n[V_GEOM CLAMP] step={global_step} max_log_corr={max_log_correction:.3f}")
                            print(f"  Samples with valid scale: {int(n_valid_samples)}/{batch_size_real}")
                            print(f"  ratio_raw (V_hat_centered): p10={ratio_raw.quantile(0.1).item():.3f} "
                                  f"p50={ratio_raw.median().item():.3f} "
                                  f"p90={ratio_raw.quantile(0.9).item():.3f}")
                            print(f"  ratio_post (V_geom clamped): p10={ratio_post.quantile(0.1).item():.3f} "
                                  f"p50={ratio_post.median().item():.3f} "
                                  f"p90={ratio_post.quantile(0.9).item():.3f}")
                            print(f"  s_corr: p10={s_corr.quantile(0.1).item():.3f} "
                                  f"p50={s_corr.median().item():.3f} "
                                  f"p90={s_corr.quantile(0.9).item():.3f}")
                            print(f"  log_s_unclamped: p10={log_s_unclamped.quantile(0.1).item():.3f} "
                                  f"p50={log_s_unclamped.median().item():.3f} "
                                  f"p90={log_s_unclamped.quantile(0.9).item():.3f}")
                            print(f"  Clamp pin-rate: {pinned_frac:.1%} (threshold for ramp: <{pin_rate_threshold:.0%})")
                            
                            # Check if clamp is helping
                            improvement = (ratio_raw.median() - 1.0).abs() - (ratio_post.median() - 1.0).abs()
                            if improvement.item() > 0:
                                print(f"  ✓ Clamp is helping: pulled ratio {improvement.item():.3f} closer to 1.0")
                            else:
                                print(f"  ⚠ Clamp not helping much (improvement={improvement.item():.3f})")


                            # --- [CLAMP-COVERAGE] Edge coverage instrumentation ---
                            if valid_scale.any() and edges_used is not None:
                                v_mask = valid_scale
                                neighbor_frac_valid = neighbor_frac[v_mask]
                                edges_used_valid = edges_used[v_mask].float()
                                log_s_valid = log_s_unclamped[v_mask]
                                
                                # Coverage percentiles
                                if neighbor_frac_valid.numel() > 0:
                                    nf_p10 = torch.quantile(neighbor_frac_valid, 0.1).item()
                                    nf_p50 = torch.quantile(neighbor_frac_valid, 0.5).item()
                                    nf_p90 = torch.quantile(neighbor_frac_valid, 0.9).item()
                                    
                                    eu_p10 = torch.quantile(edges_used_valid, 0.1).item()
                                    eu_p50 = torch.quantile(edges_used_valid, 0.5).item()
                                    eu_p90 = torch.quantile(edges_used_valid, 0.9).item()
                                    
                                    print(f"  [CLAMP-COVERAGE] neighbor_frac p10={nf_p10:.3f} p50={nf_p50:.3f} p90={nf_p90:.3f}")
                                    print(f"  [CLAMP-COVERAGE] edges_used p10={eu_p10:.0f} p50={eu_p50:.0f} p90={eu_p90:.0f}")
                                    
                                    # Pin-rate split by coverage
                                    pin_mask_valid = (log_s_valid.abs() >= (max_log_correction - tol))
                                    
                                    low_cov = (neighbor_frac_valid < 0.40)
                                    high_cov = (neighbor_frac_valid >= 0.60)
                                    
                                    n_low = low_cov.sum().item()
                                    n_high = high_cov.sum().item()
                                    
                                    pin_rate_low = pin_mask_valid[low_cov].float().mean().item() if n_low > 0 else 0.0
                                    pin_rate_high = pin_mask_valid[high_cov].float().mean().item() if n_high > 0 else 0.0
                                    
                                    print(f"  [CLAMP-COVERAGE] pin_rate low(<0.40)={pin_rate_low:.1%} (n={n_low}), "
                                          f"high(>=0.60)={pin_rate_high:.1%} (n={n_high})")
                            
                            # --- kNN INDEXING VERIFICATION ---
                            if global_step % 100 == 0:
                                knn_max = knn_spatial_for_scale.max().item()
                                n_max_batch = mask.shape[1]
                                b_test = 0
                                m_test = mask[b_test].bool()
                                knn_test = knn_spatial_for_scale[b_test, m_test, :]
                                valid_frac = (knn_test >= 0).float().mean().item()
                                print(f"  [KNN-INDEX-CHECK] knn_max={knn_max}, N={n_max_batch}, "
                                      f"valid_neighbor_frac={valid_frac:.2%}")

                else:
                    # SC batches or no spatial kNN: V_geom = V_hat_centered (no clamp available)
                    V_geom = V_hat_centered
                    # No clamp computed - set variables to None for knn_scale block
                    log_s_unclamped_for_knn = None
                    valid_scale_for_knn = None
                    max_log_corr_for_knn = None
                    edges_used_for_knn = None
                    neighbor_frac_for_knn = None



            # --- PATCH 7: Low-rank subspace penalty ---
            if not is_sc and WEIGHTS.get('subspace', 0) > 0:
                L_subspace = uet.variance_outside_topk(V_geom_L, mask, k=2)
            else:
                L_subspace = torch.tensor(0.0, device=device)

            # ---------------------------------------------------------------
            # ---------------------------------------------------------------
            # ========== CONTEXT REPLACEMENT INVARIANCE LOSS ==========
            # Key insight: We want CORE geometry to be invariant to EXTRA points.
            # So we keep core tokens the same and only replace extras.
            # Loss uses CENTERED edge-length invariance (scale+gauge-safe).
            L_ctx_replace = torch.tensor(0.0, device=device)

            if (not is_sc and 
                effective_ctx_loss_weight > 0 and 
                'anchor_mask' in batch and 
                'knn_spatial' in batch and
                not anchor_train):

                
                # Get core mask (anchor_mask marks core vs extras regardless of anchor_train)
                core_mask = batch['anchor_mask'].to(device).bool() & mask
                knn_spatial = batch['knn_spatial'].to(device)
                
                # Check if we have both core and extra points
                n_core_per_sample = core_mask.sum(dim=1)
                extra_mask = mask & (~core_mask)
                n_extra_per_sample = extra_mask.sum(dim=1)
                
                # Require core >= 8 and extras > 0 for meaningful replacement
                valid_samples = (n_core_per_sample >= 8) & (n_extra_per_sample > 0)
                
                if valid_samples.any():
                    B_ctx = batch_size_real
                    n_core = int(n_core_per_sample.float().mean().item())
                    n_extra = int(n_extra_per_sample.float().mean().item())
                    
                    # Decide whether to apply ctx loss this step
                    apply_ctx = (torch.rand(1, device=device).item() < ctx_replace_p)
                    
                    if apply_ctx:
                        # ===== COMPUTE SNR GATING =====
                        sigma_flat_ctx = sigma_flat if use_edm else torch.exp(4.0 * t_norm)
                        sigma_sq = sigma_flat_ctx ** 2
                        snr_w = (sigma_data**2 / (sigma_sq + sigma_data**2)).detach().float()
                        if snr_w.dim() == 0:
                            snr_w = snr_w.unsqueeze(0)
                        
                        # Gate: only apply when SNR is high enough (low noise)
                        w_snr = (snr_w >= ctx_snr_thresh).float()
                        
                        # Warmup ramp
                        warmup_mult = min(1.0, global_step / max(ctx_warmup_steps, 1))
                        w_ctx = w_snr * warmup_mult
                        
                        # Only proceed if at least some samples pass SNR gate
                        if w_ctx.sum() > 0:
                            # ===== BUILD CORE-FIXED REPLACEMENT CONTEXT =====
                            # CRITICAL: Keep core tokens the same, only replace extras
                            
                            with torch.no_grad():
                                if ctx_replace_variant == 'permute':
                                    # Random permutation for donor selection (prefer derangement)
                                    perm = torch.randperm(B_ctx, device=device)
                                    
                                    # Try to fix any fixed points (best effort)
                                    fixed_points = (perm == torch.arange(B_ctx, device=device))
                                    n_fixed = fixed_points.sum().item()
                                    
                                    if n_fixed > 0 and B_ctx > 1:
                                        fixed_idx = fixed_points.nonzero(as_tuple=True)[0]
                                        for i in fixed_idx:
                                            swap_target = (i + 1) % B_ctx
                                            if not fixed_points[swap_target]:
                                                perm[i], perm[swap_target] = perm[swap_target].clone(), perm[i].clone()
                                    
                                    fixed_points_final = (perm == torch.arange(B_ctx, device=device))
                                    n_fixed_final = fixed_points_final.sum().item()
                                    perm_fixed_frac = n_fixed_final / max(B_ctx, 1)
                                    
                                    donor_idx = perm
                                    ctx_debug_info = f"permute fixed={n_fixed_final}/{B_ctx}"
                                    ctx_metric = perm_fixed_frac
                                    
                                elif ctx_replace_variant == 'hard':
                                    # Hard negative: most similar context as donor
                                    mask_f = mask.unsqueeze(-1).float()
                                    denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
                                    
                                    if H_train.dim() == 2:
                                        v_ctx = H_train
                                    elif H_train.dim() == 3:
                                        v_ctx = (H_train * mask_f).sum(dim=1) / denom.squeeze(-1)
                                    else:
                                        v_ctx = H_train.reshape(B_ctx, -1)
                                    
                                    v_ctx_norm = v_ctx / (v_ctx.norm(dim=-1, keepdim=True) + 1e-8)
                                    sim_matrix = v_ctx_norm @ v_ctx_norm.T
                                    sim_matrix.fill_diagonal_(-1.0)
                                    
                                    hard_idx = sim_matrix.argmax(dim=1)
                                    hard_sim = sim_matrix.max(dim=1)[0]
                                    hard_sim_mean = hard_sim.mean().item()
                                    
                                    donor_idx = hard_idx
                                    ctx_debug_info = f"hard sim={hard_sim_mean:.3f}"
                                    ctx_metric = hard_sim_mean
                                    
                                else:
                                    raise ValueError(f"Unknown ctx_replace_variant: {ctx_replace_variant}")
                                
                                # ===== CONSTRUCT REPLACEMENT Z_set: KEEP CORE, REPLACE EXTRAS =====
                                # Z_set shape: (B, N, D_z)
                                Z_set_rep = Z_set.clone()
                                
                                for b in range(B_ctx):
                                    donor_b = donor_idx[b].item()
                                    
                                    # Get extra masks for this sample and donor
                                    extra_b = extra_mask[b]       # (N,) bool - extras in sample b
                                    extra_donor = extra_mask[donor_b]  # (N,) bool - extras in donor
                                    
                                    n_extra_b = int(extra_b.sum().item())
                                    n_extra_donor = int(extra_donor.sum().item())
                                    
                                    if n_extra_b == 0 or n_extra_donor == 0:
                                        continue
                                    
                                    # Get extra indices
                                    extra_idx_b = extra_b.nonzero(as_tuple=True)[0]         # Where to put
                                    extra_idx_donor = extra_donor.nonzero(as_tuple=True)[0]  # Where to get from
                                    
                                    # How many extras to actually replace (min of both)
                                    n_replace = min(n_extra_b, n_extra_donor)
                                    
                                    # Random subset if donor has more extras than needed
                                    if n_extra_donor > n_replace:
                                        perm_donor = torch.randperm(n_extra_donor, device=device)[:n_replace]
                                        extra_idx_donor = extra_idx_donor[perm_donor]
                                    
                                    # Random subset if target has more extra slots than donor extras
                                    if n_extra_b > n_replace:
                                        perm_b = torch.randperm(n_extra_b, device=device)[:n_replace]
                                        extra_idx_b = extra_idx_b[perm_b]
                                    
                                    # Replace: Z_set_rep[b, extra positions] = Z_set[donor, donor extra positions]
                                    Z_set_rep[b, extra_idx_b] = Z_set[donor_b, extra_idx_donor]
                                
                                # Mask stays the same - same valid positions
                                mask_rep = mask
                            
                            # Encode replacement context (core same, extras different)
                            H_rep = context_encoder(Z_set_rep, mask_rep)
                            
                            # ===== FORWARD PASS WITH REPLACEMENT CONTEXT =====
                            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                                if self_cond_mode == 'standard' and score_net.self_conditioning:
                                    # Two-pass self-conditioning for replacement branch
                                    with torch.no_grad():
                                        x0_pred_rep_0_result = score_net.forward_edm(
                                            V_t, sigma_flat, H_rep, mask, sigma_data,
                                            self_cond=None, return_debug=False
                                        )
                                        if isinstance(x0_pred_rep_0_result, tuple):
                                            x0_pred_rep_0 = x0_pred_rep_0_result[0]
                                        else:
                                            x0_pred_rep_0 = x0_pred_rep_0_result
                                    
                                    x0_pred_rep_result = score_net.forward_edm(
                                        V_t, sigma_flat, H_rep, mask, sigma_data,
                                        self_cond=x0_pred_rep_0.detach(),
                                        return_debug=False
                                    )
                                else:
                                    x0_pred_rep_result = score_net.forward_edm(
                                        V_t, sigma_flat, H_rep, mask, sigma_data,
                                        self_cond=None, return_debug=False
                                    )
                                
                                if isinstance(x0_pred_rep_result, tuple):
                                    x0_pred_rep = x0_pred_rep_result[0]
                                else:
                                    x0_pred_rep = x0_pred_rep_result
                            
                            # ===== COMPUTE EDGE-BASED CONTEXT INVARIANCE LOSS =====
                            # UPDATED: Now uses SHAPE + SCALE decomposition
                            # - Shape loss: centered (scale-invariant), penalizes relative edge distortion
                            # - Scale loss: penalizes global scale drift (the main stitching failure mode!)
                            CTX_SCALE_WEIGHT = 1.0  # Weight for scale term relative to shape term

                            with torch.autocast(device_type='cuda', enabled=False):
                                x0_pred_full = x0_pred.float()
                                x0_pred_rep_f32 = x0_pred_rep.float()

                                # Get BOTH shape and scale losses
                                shape_losses, scale_losses, n_edges, ctx_debug_dict = uet.core_edge_log_invariance_loss(
                                    x0_pred_full, x0_pred_rep_f32,
                                    core_mask, knn_spatial, mask,
                                    eps=1e-8, huber_delta=0.1,
                                    scale_weight=CTX_SCALE_WEIGHT
                                )

                                # Combined loss = shape + scale_weight * scale
                                combined_losses = shape_losses + CTX_SCALE_WEIGHT * scale_losses

                                # Per-sample losses weighted by SNR gate
                                valid_loss_samples = (n_edges > 0) & (w_ctx > 0)

                                if valid_loss_samples.any():
                                    # Total ctx loss (shape + scale combined)
                                    L_ctx_replace = (w_ctx * combined_losses)[valid_loss_samples].sum() / w_ctx[valid_loss_samples].sum().clamp(min=1.0)

                                    # Track components separately for logging
                                    L_ctx_shape = (w_ctx * shape_losses)[valid_loss_samples].sum() / w_ctx[valid_loss_samples].sum().clamp(min=1.0)
                                    L_ctx_scale = (w_ctx * scale_losses)[valid_loss_samples].sum() / w_ctx[valid_loss_samples].sum().clamp(min=1.0)
                                else:
                                    L_ctx_replace = torch.tensor(0.0, device=device)
                                    L_ctx_shape = torch.tensor(0.0, device=device)
                                    L_ctx_scale = torch.tensor(0.0, device=device)

                            # ===== DEBUG LOGGING (ENHANCED) =====
                            if global_step % ctx_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                                with torch.no_grad():
                                    sigma_med = sigma_flat_ctx.median().item()
                                    snr_med = snr_w.median().item()
                                    n_on = int((w_ctx > 0).sum().item())
                                    n_edges_mean = n_edges[n_edges > 0].mean().item() if (n_edges > 0).any() else 0

                                    print(f"\n[CTX-REPL-v3] step={global_step} variant={ctx_replace_variant}")
                                    print(f"  {ctx_debug_info}")
                                    print(f"  p_apply={ctx_replace_p:.2f} applied=True")
                                    print(f"  core={n_core} extra={n_extra} (CORE-FIXED replacement)")
                                    print(f"  mean_core_edges={n_edges_mean:.1f}")

                                    print(f"\n[CTX-GATE] step={global_step}")
                                    print(f"  sigma_data={sigma_data:.5f} sigma_med={sigma_med:.4f}")
                                    print(f"  snr_med={snr_med:.4f} snr_thresh={ctx_snr_thresh:.2f}")
                                    print(f"  n_on={n_on}/{B_ctx} warmup={warmup_mult:.2f}")

                                    # ===== KEY NEW DIAGNOSTICS =====
                                    print(f"\n[CTX-LOSS-DECOMPOSED] step={global_step}")
                                    print(f"  L_shape={L_ctx_shape.item():.6f} (centered, scale-invariant)")
                                    print(f"  L_scale={L_ctx_scale.item():.6f} (scale drift penalty)")
                                    print(f"  L_total={L_ctx_replace.item():.6f} (shape + {CTX_SCALE_WEIGHT}*scale)")
                                    print(f"  λ_ctx={ctx_loss_weight} contribution={ctx_loss_weight * L_ctx_replace.item():.6f}")

                                    # Scale shift diagnostics (THE KEY METRIC)
                                    print(f"\n[CTX-SCALE-SHIFT] step={global_step}")
                                    print(f"  delta_scale_mean={ctx_debug_dict['delta_scale_mean']:.4f} (should be ~0 if no drift)")
                                    print(f"  delta_scale_abs_mean={ctx_debug_dict['delta_scale_abs_mean']:.4f} (magnitude of drift)")
                                    if 'delta_scale_abs_p90' in ctx_debug_dict:
                                        print(f"  delta_scale_abs_p90={ctx_debug_dict['delta_scale_abs_p90']:.4f}")
                                        print(f"  delta_scale_abs_max={ctx_debug_dict['delta_scale_abs_max']:.4f}")
                                    print(f"  raw_log_diff_mean={ctx_debug_dict['raw_log_diff_mean']:.4f} (pre-centering diff)")
                                    print(f"  shape_residual_mean={ctx_debug_dict['shape_residual_mean']:.6f} (post-centering residual)")

                                    # Interpretation help
                                    ds_abs = ctx_debug_dict['delta_scale_abs_mean']
                                    if ds_abs > 0.1:
                                        implied_scale = 2.718 ** ds_abs  # e^delta ≈ scale ratio
                                        print(f"  [!] SCALE DRIFT DETECTED: |delta|={ds_abs:.3f} => scale_ratio≈{implied_scale:.3f} or {1/implied_scale:.3f}")
                                    elif ds_abs > 0.05:
                                        print(f"  [~] Moderate scale drift: |delta|={ds_abs:.3f}")
                                    else:
                                        print(f"  [✓] Scale drift is small: |delta|={ds_abs:.3f}")

                                    # Additional per-sample diagnostics
                                    if valid_loss_samples.any():
                                        print(f"\n[CTX-LOSS-SAMPLES] step={global_step}")
                                        print(f"  shape_loss: mean={shape_losses[valid_loss_samples].mean().item():.6f} max={shape_losses[valid_loss_samples].max().item():.6f}")
                                        print(f"  scale_loss: mean={scale_losses[valid_loss_samples].mean().item():.6f} max={scale_losses[valid_loss_samples].max().item():.6f}")
                                        print(f"  combined:   mean={combined_losses[valid_loss_samples].mean().item():.6f} max={combined_losses[valid_loss_samples].max().item():.6f}")
                            
                            # Track for epoch averaging
                            ctx_replace_sum += L_ctx_replace.item()
                            ctx_apply_count += 1
                            ctx_snr_sum += snr_w.median().item()
                            
                            if ctx_replace_variant == 'hard':
                                ctx_hard_sim_sum += ctx_metric
                            elif ctx_replace_variant == 'permute':
                                ctx_perm_fixed_sum += ctx_metric
                        
                        else:
                            # SNR gate blocked all samples
                            if global_step % ctx_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                                print(f"[CTX-REPL-v3] step={global_step} SNR gate blocked all samples")
                    
                    else:
                        # Not applying this step (probability gate)
                        if global_step % ctx_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                            print(f"[CTX-REPL-v3] step={global_step} skipped (prob gate)")
                
                else:
                    # Not enough core/extra points
                    if global_step % ctx_debug_every == 0 and (fabric is None or fabric.is_global_zero):
                        print(f"[CTX-REPL-v3] step={global_step} skipped: insufficient core/extra")

            else:
                # Context loss disabled or wrong conditions
                if global_step % (ctx_debug_every * 5) == 0 and (fabric is None or fabric.is_global_zero):
                    reasons = []
                    if is_sc:
                        reasons.append("is_sc=True")
                    if ctx_loss_weight <= 0:
                        reasons.append("weight=0")
                    if 'anchor_mask' not in batch:
                        reasons.append("no_anchor_mask")
                    if 'knn_spatial' not in batch:
                        reasons.append("no_knn_spatial")
                    if anchor_train:
                        reasons.append("anchor_train=True (ctx_replace only in unanchored mode)")
                    print(f"[CTX-REPL-v3] step={global_step} disabled: {', '.join(reasons)}")


            if not is_sc:
                # ==================== COMPUTE V_TARGET_BATCH ONCE ====================
                # Needed by: edge loss, topo loss, shape loss, etc.
                # Only compute if at least one of these losses is active
                need_v_target = (WEIGHTS['edge'] > 0 or WEIGHTS['topo'] > 0 or 
                                WEIGHTS['shape_spec'] > 0 or WEIGHTS['edm_tail'] > 0 or
                                WEIGHTS['gen_align'] > 0)
                
                if need_v_target:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # PATCH 3 CONTINUED: V_target is already canonicalized above, just reuse it
                        V_target_batch = V_target.float()

                # ==================== UNIFIED SNR-BASED GATING (COMPUTE ONCE) ====================
                if not is_sc and 'knn_spatial' in batch:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # Get sigma values (works for both EDM and non-EDM)
                        sigma_vec = sigma_t.view(-1).float()  # (B,)
                        
                        # Compute robust edge scale from target kNN edges
                        knn_indices_batch = batch['knn_spatial'].to(device)
                        edge_scales = torch.zeros(batch_size_real, device=device)
                        
                        for b in range(batch_size_real):
                            m_b = mask[b]
                            n_valid = int(m_b.sum().item())
                            if n_valid < 2 or not need_v_target:
                                edge_scales[b] = 1.0  # Fallback
                                continue
                            
                            V_tgt_b = V_target_batch[b, m_b]
                            if V_tgt_b.shape[0] < 2:
                                edge_scales[b] = 1.0
                                continue
                            
                            knn_b = knn_indices_batch[b, m_b]  # (n_valid, k)
                            valid_neighbors = (knn_b >= 0) & (knn_b < n_valid)
                            
                            if not valid_neighbors.any():
                                edge_scales[b] = 1.0
                                continue
                            
                            # Compute edge lengths for all valid kNN pairs
                            k_size = knn_b.shape[1]
                            i_idx = torch.arange(n_valid, device=device).unsqueeze(1).expand(-1, k_size)
                            j_idx = knn_b
                            
                            valid_mask = valid_neighbors & (i_idx != j_idx) & (j_idx >= 0) & (j_idx < n_valid)
                            i_edges = i_idx[valid_mask]
                            j_edges = j_idx[valid_mask]
                            
                            if i_edges.numel() > 0:
                                edge_vecs = V_tgt_b[i_edges] - V_tgt_b[j_edges]
                                edge_lens = edge_vecs.norm(dim=-1)
                                edge_scales[b] = torch.quantile(edge_lens, 0.10).clamp_min(1e-6)
                            else:
                                edge_scales[b] = 1.0
                        
                        # Compute SNR: rho = (sigma * sqrt(D)) / edge_scale
                        rho = (sigma_vec * math.sqrt(D_latent)) / edge_scales.clamp_min(1e-6)  # (B,) unitless
                        
                        # Store for debug
                        if DEBUG and (global_step % LOG_EVERY == 0):
                            print(f"[SNR] edge_scale: min={edge_scales.min():.3f} median={edge_scales.median():.3f} max={edge_scales.max():.3f}")
                            print(f"[SNR] rho (noise/signal): min={rho.min():.3f} median={rho.median():.3f} max={rho.max():.3f}")
                else:
                    # Fallback: no SNR available (use old thresholds)
                    sigma_vec = sigma_t.view(-1).float()
                    rho = None

                # Conditional flag (same for all losses)
                cond_only = (drop_mask.view(-1) < 0.5).float()  # (B,) 1.0 when NOT dropped

                # ==============================================================================
                # COMPUTE NOISE METRIC FOR ADAPTIVE GATING
                # ==============================================================================
                # noise = -log(c_skip) where c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
                # Higher noise score = noisier sample (c_skip closer to 0)
                # This is dataset-independent because c_skip is normalized by sigma_data
                
                if use_edm:
                    with torch.no_grad():
                        sigma_flat_gate = sigma_t.view(-1)  # (B,)
                        c_skip_b = (sigma_data ** 2) / (sigma_flat_gate ** 2 + sigma_data ** 2 + 1e-12)
                        noise_score = -torch.log(c_skip_b + 1e-12)  # (B,) higher = noisier
                        
                        # Base gate from CFG dropout
                        base_gate = cond_only.float()  # (B,) 1.0 when conditioned
                        
                        # Update geometry gate controllers with eligible (conditioned) samples only
                        eligible = base_gate > 0.5
                        if eligible.any():
                            adaptive_gates["gram"].update(noise_score[eligible])
                            adaptive_gates["gram_scale"].update(noise_score[eligible])
                            adaptive_gates["edge"].update(noise_score[eligible])
                            adaptive_gates["nca"].update(noise_score[eligible])
                            adaptive_gates["learn_hi"].update(noise_score[eligible])
                        
                        # A/B 2: Update score_hi_gate on ALL samples (score loss applies to both cond/uncond)
                        # score_hi_gate.update(noise_score)
                        # gate_hi_score = score_hi_gate.gate(noise_score, torch.ones_like(noise_score))
                else:
                    # Fallback for non-EDM: use t_norm as noise proxy
                    with torch.no_grad():
                        noise_score = t_norm.view(-1)  # (B,)
                        base_gate = cond_only.float()
                        
                        eligible = base_gate > 0.5
                        if eligible.any():
                            for gate in adaptive_gates.values():
                                gate.update(noise_score[eligible])
                        
                        # A/B 2: Fallback gate for non-EDM
                        # score_hi_gate.update(noise_score)
                        # gate_hi_score = score_hi_gate.gate(noise_score, torch.ones_like(noise_score))

                # ==============================================================================
                # BUILD ADAPTIVE GATES FOR EACH LOSS
                # ==============================================================================
                
                # --- MINIMUM COVERAGE FLOOR (prevents Option 2 starvation) ---
                MIN_GEOMETRY_SAMPLES = 4  # Minimum samples per batch with geometry loss
                
                def ensure_minimum_coverage(gate: torch.Tensor, score: torch.Tensor, 
                                           min_count: int, prefer_low_score: bool = True,
                                           eligible_mask: torch.Tensor = None) -> torch.Tensor:
                    """
                    Ensure gate has at least min_count active samples.
                    If fewer pass threshold, force-include best candidates from eligible set.
                    
                    This prevents geometry gates from starving (Option 2 had 6.25% hit rate → Jaccard collapsed).
                    
                    Args:
                        gate: (B,) current gate values
                        score: (B,) score for ranking candidates (lower = preferred if prefer_low_score)
                        min_count: minimum active samples
                        prefer_low_score: if True, prefer low score when forcing (e.g., low noise for structure)
                                         if False, prefer low score too but for different meaning (e.g., low c_skip = high σ)
                        eligible_mask: (B,) bool mask of samples eligible for this loss
                    """
                    if eligible_mask is None:
                        eligible_mask = torch.ones(gate.shape[0], device=gate.device, dtype=torch.bool)
                    
                    # Only count eligible samples that are gated on
                    n_active = ((gate > 0) & eligible_mask).sum().item()
                    if n_active >= min_count:
                        return gate
                    
                    # Need to force-include (min_count - n_active) samples
                    n_needed = min_count - int(n_active)
                    
                    # Candidates: eligible samples not already gated
                    not_gated = (gate <= 0) & eligible_mask
                    
                    if not_gated.sum() == 0:
                        return gate  # No candidates available
                    
                    # Score for ranking (low score = preferred)
                    scores = score.clone()
                    scores[~not_gated] = float('inf')  # Exclude already gated or ineligible
                    
                    if not prefer_low_score:
                        # Invert so low score becomes high (for learn_hi: want high σ = low c_skip)
                        scores = -scores
                        scores[~not_gated] = float('inf')
                    
                    # Find best candidates
                    n_to_add = min(n_needed, int(not_gated.sum().item()))
                    _, best_idx = scores.topk(n_to_add, largest=False)
                    
                    # Create new gate with forced inclusions
                    new_gate = gate.clone()
                    new_gate[best_idx] = 1.0
                    
                    return new_gate
                
                # Eligibility: MUST include base_gate (ChatGPT A2 correction)
                # This prevents forcing uncond samples into geometry losses
                n_valid_per_sample = mask.sum(dim=1)  # (B,)
                eligible_for_geo = (n_valid_per_sample >= 16) & (base_gate > 0.5)
                
                if use_edm:
                    geo_gate_gram = adaptive_gates["gram"].gate(noise_score, base_gate)
                    geo_gate_gram_scale = adaptive_gates["gram_scale"].gate(noise_score, base_gate)
                    geo_gate_edge = adaptive_gates["edge"].gate(noise_score, base_gate)
                    geo_gate_nca = adaptive_gates["nca"].gate(noise_score, base_gate)
                    geo_gate_learn_hi = adaptive_gates["learn_hi"].gate(noise_score, base_gate)
                    
                    # Apply minimum coverage floor to each gate
                    # Structure losses: prefer low noise (clean signal)
                    geo_gate_gram = ensure_minimum_coverage(
                        geo_gate_gram, noise_score, MIN_GEOMETRY_SAMPLES, 
                        prefer_low_score=True, eligible_mask=eligible_for_geo)
                    geo_gate_gram_scale = ensure_minimum_coverage(
                        geo_gate_gram_scale, noise_score, MIN_GEOMETRY_SAMPLES,
                        prefer_low_score=True, eligible_mask=eligible_for_geo)
                    geo_gate_edge = ensure_minimum_coverage(
                        geo_gate_edge, noise_score, MIN_GEOMETRY_SAMPLES,
                        prefer_low_score=True, eligible_mask=eligible_for_geo)
                    geo_gate_nca = ensure_minimum_coverage(
                        geo_gate_nca, noise_score, MIN_GEOMETRY_SAMPLES,
                        prefer_low_score=True, eligible_mask=eligible_for_geo)
                    
                    # --- A3: learn_hi needs HIGHEST σ (lowest c_skip) when forcing ---
                    # Use c_skip_b as score, prefer_low_score=True means prefer low c_skip = high σ
                    geo_gate_learn_hi = ensure_minimum_coverage(
                        geo_gate_learn_hi, c_skip_b, MIN_GEOMETRY_SAMPLES,
                        prefer_low_score=True, eligible_mask=eligible_for_geo)
                else:
                    # Fallback: just use cond_only for all
                    geo_gate_gram = base_gate
                    geo_gate_gram_scale = base_gate
                    geo_gate_edge = base_gate
                    geo_gate_nca = base_gate
                    geo_gate_learn_hi = base_gate


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
                    # PATCH 2 (CORRECTED): Use rank-limited Gram from V_target_batch
                    # V_target_batch is already rank ≤ D_latent from factor_from_gram
                    # This ensures the target Gram is feasible in D_latent dimensions
                    # (G_target from batch is full-rank from 2D coords, creating impossible pressure)
                    Vt_c_for_gram, _ = uet.center_only(V_target_batch.float(), mask)
                    Gt = Vt_c_for_gram @ Vt_c_for_gram.transpose(1, 2)  # (B, N, N), rank ≤ D_latent
                    # NOTE: MM and P_off masks are applied downstream, no need to multiply here
                    B, N, _ = V_geom.shape
                    
                    # PATCH 2 DEBUG: Log Gram rank/trace comparison
                    if global_step % 500 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            tr_Gt_new = (Gt.diagonal(dim1=1, dim2=2) * mask.float()).sum(dim=1).mean()
                            tr_Gt_old = (G_target.float().diagonal(dim1=1, dim2=2) * mask.float()).sum(dim=1).mean()
                            print(f"[PATCH2-GRAM] tr(Gt_rank_limited)={tr_Gt_new.item():.4f} "
                                  f"tr(Gt_old_fullrank)={tr_Gt_old.item():.4f} "
                                  f"ratio={tr_Gt_new.item()/tr_Gt_old.item():.4f}")
                    
                    # Build predicted Gram *with* true scale
                    Gp_raw = V_geom_L @ V_geom_L.transpose(1, 2)          # (B,N,N)
                    
                    # --- PATCH 6: Diagonal distribution loss (per-point norm matching) ---
                    diag_p = torch.diagonal(Gp_raw, dim1=-2, dim2=-1)  # (B,N)
                    diag_t = torch.diagonal(Gt, dim1=-2, dim2=-1)      # (B,N)
                    
                    m_float_1d = m_bool.float()  # (B,N)
                    den_diag = (diag_t.pow(2) * m_float_1d).sum(dim=-1).clamp_min(1e-8)  # (B,)
                    diag_rel = ((diag_p - diag_t).pow(2) * m_float_1d).sum(dim=-1) / den_diag  # (B,)
                    # ---------------------------------------------------------------

                    
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
                    if (global_step % 25 == 0) and (fabric is None or fabric.is_global_zero):
                        tr_p_raw = (Gp_raw.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1)
                        tr_t_raw = (Gt.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1)
                        ratio_tr = (tr_p_raw.mean() / tr_t_raw.mean().clamp_min(1e-12) * 100).item()
                        print(f"[gram/trace] pred={tr_p_raw.mean().item():.1f} "
                            f"target={tr_t_raw.mean().item():.1f} ratio={ratio_tr:.1f}%")

                        # Consolidated Gram vs Distance check (same sample, same tensors)
                        with torch.no_grad():
                            idx = 0  # first sample
                            V_p = V_geom[idx]  # (N, D)
                            V_tgt_sample = V_target_batch[idx]  # (N, D)
                            m0 = m_bool[idx]  # (N,)
                            
                            # Gram trace for this sample
                            G_p_diag = (V_p * V_p).sum(dim=1)  # ||v_i||^2
                            G_t_diag = (V_tgt_sample * V_tgt_sample).sum(dim=1)
                            tr_p_sample = G_p_diag[m0].sum().item()
                            tr_t_sample = G_t_diag[m0].sum().item()
                            gram_ratio_sample = tr_p_sample / max(tr_t_sample, 1e-8)
                            
                            # Distance for this sample
                            D_p = torch.cdist(V_p[m0], V_p[m0])
                            D_t = torch.cdist(V_tgt_sample[m0], V_tgt_sample[m0])
                            n_valid = m0.sum().item()
                            mask_upper = torch.triu(torch.ones(n_valid, n_valid, device=D_p.device), diagonal=1).bool()
                            d_p_med = D_p[mask_upper].median().item()
                            d_t_med = D_t[mask_upper].median().item()
                            dist_ratio_sample = d_p_med / max(d_t_med, 1e-8)
                            
                            # Expected distance ratio from Gram
                            expected_dist_ratio = gram_ratio_sample ** 0.5
                            
                            print(f"[CONSISTENCY] sample=0 | gram_ratio={gram_ratio_sample:.3f} "
                                  f"dist_ratio={dist_ratio_sample:.3f} expected_dist={expected_dist_ratio:.3f} "
                                  f"error={abs(dist_ratio_sample - expected_dist_ratio):.3f}")
                        
                        # Store for epoch-end health check
                        debug_state['last_gram_trace_ratio'] = ratio_tr


                    # ===== RAW GRAM LOSS (SCALE-AWARE) =====
                    # Compute difference in raw space (keeps scale information)
                    diff_raw = (Gp_raw - Gt) * P_off  # (B, N, N)
                    pair_cnt = P_off.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)

                    # --- DEBUG 4: Raw Gram difference statistics ---
                    if global_step % 25 == 0 and (fabric is None or fabric.is_global_zero):
                        gp_off = Gp_raw[P_off.bool()]
                        gt_off = Gt[P_off.bool()]
                        if gp_off.numel() > 0 and gt_off.numel() > 0:
                            diff_vec = gp_off - gt_off
                            rel_frob = diff_vec.norm() / (gt_off.norm() + 1e-12)
                            cos_sim = float(F.cosine_similarity(gp_off, gt_off, dim=0).item()) if (gp_off.norm() > 0 and gt_off.norm() > 0) else float('nan')
                            scale_ratio = (gp_off.pow(2).mean().sqrt() / gt_off.pow(2).mean().sqrt().clamp_min(1e-12)).item()
                            print(f"[gram/raw] offdiag stats | "
                                  f"P(mean={gp_off.mean().item():.3e}, std={gp_off.std().item():.3e})  "
                                  f"T(mean={gt_off.mean().item():.3e}, std={gt_off.std().item():.3e})  "
                                  f"ΔF/TF={rel_frob:.3e}  cos={cos_sim:.3f}  scale_ratio={scale_ratio:.3f}")

                    # Numerator: mean squared error per pair in raw space
                    numerator = diff_raw.pow(2).sum(dim=(1, 2)) / pair_cnt  # (B,)

                    # Denominator: mean squared target energy per pair in raw space
                    t_energy_raw = (Gt.pow(2) * P_off).sum(dim=(1, 2)) / pair_cnt  # (B,)

                    # Floor: fraction of median target energy (data-driven stability)
                    delta = 0.05
                    t_energy_median = t_energy_raw.detach().median().clamp_min(1e-12)
                    denominator = t_energy_raw.clamp_min(delta * t_energy_median)

                    per_set_relative_loss = numerator / denominator  # (B,)

                    # Sigma compensation: geometry gradients scale with sigma
                    # --- FIX #5: Remove σ-weighting from shape loss ---
                    # The σ-weighting was causing scale pressure to vary with σ
                    # Keep it for the shape component but NOT for the scale component
                    sigma_vec = sigma_t.view(-1).float()  # (B,)

                    # Gate geometry to conditional + low noise samples
                    # PATCH 3A: Re-enable SNR gating for Gram loss
                    # With sigma_data≈0.17, sigma=0.5 is ~3x sigma_data, so use sigma<=0.4 as "low noise"
                    # PATCH 3A (CORRECTED): Re-enable SNR gating for Gram loss
                    # Starting thresholds - tune based on hit-rate debug below
                    # Target: ~30-60% hit rate for Gram (global structure)
                    if rho is not None:
                        # rho = sigma * sqrt(D) / edge_scale
                        # Start with rho <= 5.0 and calibrate
                        low_noise_gram = (rho <= 5.0)
                    else:
                        # Fallback: sigma <= 0.5 (moderate noise)
                        low_noise_gram = (sigma_vec <= 0.5)
                    # geo_gate = cond_only * low_noise_gram.float()
                    # Use adaptive gate (replaces hardcoded rho threshold)
                    geo_gate = geo_gate_gram
                    gate_sum = geo_gate.sum().clamp(min=1.0)
                    
                    # Sanitize per-sample losses BEFORE gating (NaN * 0 = NaN)
                    per_set_relative_loss = torch.nan_to_num(per_set_relative_loss, nan=0.0, posinf=0.0, neginf=0.0)
                    diag_rel = torch.nan_to_num(diag_rel, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    L_gram_offdiag = (per_set_relative_loss * geo_gate).sum() / gate_sum
                    L_gram_diag = (diag_rel * geo_gate).sum() / gate_sum
                    L_gram = L_gram_offdiag + 0.5 * L_gram_diag

                    
                    # PATCH 3 CALIBRATION DEBUG: Log gate hit-rates and rho distribution
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            hit_rate_gram = (geo_gate > 0).float().mean().item()
                            cond_rate = cond_only.float().mean().item()
                            low_noise_rate = low_noise_gram.float().mean().item()
                            
                            print(f"\n[PATCH3-GATE-CALIBRATION] Step {global_step}:")
                            print(f"  Gram gate: hit_rate={hit_rate_gram:.2%} "
                                  f"(cond_only={cond_rate:.2%}, low_noise={low_noise_rate:.2%})")
                            
                            if rho is not None:
                                rho_cpu = rho.detach().cpu()
                                print(f"  rho: p10={rho_cpu.quantile(0.1).item():.2f} "
                                      f"p50={rho_cpu.quantile(0.5).item():.2f} "
                                      f"p90={rho_cpu.quantile(0.9).item():.2f}")
                            else:
                                sig_cpu = sigma_vec.detach().cpu()
                                print(f"  sigma: p10={sig_cpu.quantile(0.1).item():.4f} "
                                      f"p50={sig_cpu.quantile(0.5).item():.4f} "
                                      f"p90={sig_cpu.quantile(0.9).item():.4f}")
                            
                            # Warn if hit rate is outside target range
                            if hit_rate_gram < 0.20:
                                print(f"  ⚠️ Gram hit rate too LOW (<20%) - consider relaxing threshold")
                            elif hit_rate_gram > 0.70:
                                print(f"  ⚠️ Gram hit rate HIGH (>70%) - consider tightening threshold")



                    # ==================== ACTION 4 (FINAL): TRUE LEARNED-BRANCH GEOMETRY ====================
                    # Fixes applied:
                    # 1. Compute Gram from V_out (not V_hat) - TRUE learned-branch supervision
                    # 2. Tight σ gate: c_skip < 0.05 corresponds to σ > 0.73 (the actual shrink regime)
                    # 3. Compensate c_out gradient damping with 1/c_out² (clamped)
                    # 4. Full-batch weighted (no boolean indexing) for DDP stability
                    
                    if use_edm and WEIGHTS.get('gram_learn', 0) > 0:
                        with torch.autocast(device_type='cuda', enabled=False):
                            # Get preconditioning scalars
                            sigma_flat_gl = sigma_t.view(-1).float()
                            c_skip_gl, c_out_gl, _, _ = uet.edm_precond(sigma_flat_gl, sigma_data)
                            c_skip_1d = c_skip_gl.view(-1)  # (B,)
                            c_out_1d = c_out_gl.view(-1).clamp(min=1e-6)  # (B,)
                            
                            # Centered noisy input (same as forward_edm)
                            V_c_gl, _ = uet.center_only(V_t, mask)  # (B, N, D)
                            
                            # LEARNED BRANCH OUTPUT: V_out = V_hat_centered - c_skip * V_c
                            V_out_gl = (V_geom - c_skip_gl * V_c_gl) * m_float  # (B, N, D)
                            
                            # CORRECT TARGET for learned branch: x0_centered - c_skip * x_c
                            V_tgt_c_gl, _ = uet.center_only(V_target, mask)
                            V_out_tgt_gl = (V_tgt_c_gl - c_skip_gl * V_c_gl) * m_float  # (B, N, D)
                            
                            # FIX 2: TIGHT HIGH-σ SELECTION (c_skip < 0.05 → σ > 0.73)
                            sel = (c_skip_1d < 0.05).float()  # (B,)
                            
                            # Smooth weight within selected regime
                            w_gl = sel * (1.0 - c_skip_1d).pow(2)  # (B,)
                            
                            # FIX 3: COMPENSATE c_out GRADIENT DAMPING
                            # At high σ, c_out ≈ 0.165, so 1/c_out² ≈ 36
                            # Clamp to avoid explosion at very high σ
                            inv_cout2 = (1.0 / c_out_1d.pow(2)).clamp(max=64.0)
                            w_gl = w_gl * inv_cout2  # (B,)
                            
                            # FIX 4: FULL-BATCH WEIGHTED (no boolean indexing)
                            # Compute Gram on V_out for ENTIRE batch
                            G_out_pred = V_out_gl @ V_out_gl.transpose(1, 2)  # (B, N, N)
                            G_out_tgt = V_out_tgt_gl @ V_out_tgt_gl.transpose(1, 2)  # (B, N, N)
                            
                            # Valid pair mask
                            MM_gl = (m_bool.unsqueeze(-1) & m_bool.unsqueeze(-2)).float()  # (B, N, N)
                            eye_gl = torch.eye(G_out_pred.shape[1], device=device).unsqueeze(0)
                            P_off_gl = MM_gl * (1.0 - eye_gl)  # Off-diagonal pairs
                            
                            # Per-sample off-diagonal loss
                            pair_cnt_gl = P_off_gl.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
                            num_gl = ((G_out_pred - G_out_tgt).pow(2) * P_off_gl).sum(dim=(1, 2)) / pair_cnt_gl
                            den_gl = (G_out_tgt.pow(2) * P_off_gl).sum(dim=(1, 2)) / pair_cnt_gl
                            
                            den_med_gl = den_gl.detach().median().clamp_min(1e-12)
                            den_gl = den_gl.clamp_min(0.05 * den_med_gl)
                            loss_off_gl = num_gl / den_gl  # (B,)
                            
                            # Per-sample diagonal loss
                            diag_p_gl = torch.diagonal(G_out_pred, dim1=-2, dim2=-1)  # (B, N)
                            diag_t_gl = torch.diagonal(G_out_tgt, dim1=-2, dim2=-1)   # (B, N)
                            m1_gl = m_bool.float()  # (B, N)
                            den_d_gl = (diag_t_gl.pow(2) * m1_gl).sum(dim=-1).clamp_min(1e-8)  # (B,)
                            loss_d_gl = ((diag_p_gl - diag_t_gl).pow(2) * m1_gl).sum(dim=-1) / den_d_gl  # (B,)
                            
                            # Combined per-sample loss
                            per_sample_gl = loss_off_gl + 0.5 * loss_d_gl  # (B,)

                            # Sanitize per-sample loss before gating
                            per_sample_gl = torch.nan_to_num(per_sample_gl, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Weighted average
                            wsum_gl = sel.sum().clamp_min(1.0)
                            L_gram_learn = (per_sample_gl * sel).sum() / wsum_gl
                            
                            
                            # Debug logging
                            if global_step % 100 == 0 and ((fabric is None) or (hasattr(fabric, 'is_global_zero') and fabric.is_global_zero)):
                                with torch.no_grad():
                                    n_selected = (sel > 0).sum().item()
                                    
                                    # Gram trace ratio for V_out (only selected samples)
                                    if n_selected > 0:
                                        tr_pred_gl = (G_out_pred.diagonal(dim1=1, dim2=2) * m1_gl).sum(dim=1)
                                        tr_tgt_gl = (G_out_tgt.diagonal(dim1=1, dim2=2) * m1_gl).sum(dim=1)
                                        tr_ratio_gl = (tr_pred_gl / tr_tgt_gl.clamp(min=1e-8))
                                        tr_ratio_selected = tr_ratio_gl[sel > 0].mean().item()
                                    else:
                                        tr_ratio_selected = float('nan')
                                    
                                    print(f"\n[ACTION4-FINAL] Step {global_step}:")
                                    print(f"  Selected {n_selected}/{len(sel)} samples (c_skip < 0.05, i.e. σ > ~0.73)")
                                    print(f"  V_out Gram trace ratio (selected): {tr_ratio_selected:.4f} (want → 1.0)")
                                    print(f"  Mean 1/c_out² (selected): {inv_cout2[sel > 0].mean().item() if n_selected > 0 else 0:.2f}")
                                    print(f"  L_gram_learn = {L_gram_learn.item():.6f}")
                    else:
                        L_gram_learn = torch.tensor(0.0, device=device)
                    # ==================== END ACTION 4 (FINAL) ====================


                    # --- [DEBUG] Log Gram Gate Hit-Rate ---
                    if global_step % 10 == 0:
                         hit_rate = geo_gate.float().mean().item()
                        #  print(f"[GATE-GRAM] step={global_step} hit_rate={hit_rate:.2%} "
                        #        f"({int(geo_gate.sum().item())}/{len(geo_gate)} samples)")
                         if hit_rate < 0.1:
                             print(f"   ⚠️ WARNING: Gram gate is killing >90% of samples! Check rho threshold.")

                    # Global vs local scale check
                    if global_step % 20 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            D_pred = torch.cdist(V_geom[0], V_geom[0])
                            D_tgt = torch.cdist(V_target_batch[0], V_target_batch[0])
                            m0 = m_bool[0]
                            valid = m0.unsqueeze(0) & m0.unsqueeze(1)
                            d_pred_all = D_pred[valid & ~torch.eye(N, dtype=torch.bool, device=D_pred.device)]
                            d_tgt_all = D_tgt[valid & ~torch.eye(N, dtype=torch.bool, device=D_tgt.device)]
                            if d_tgt_all.numel() > 0:
                                dist_scale = (d_pred_all.median() / d_tgt_all.median().clamp_min(1e-8)).item()
                                print(f"[DIST SCALE] global_dist_ratio={dist_scale:.3f} (should be ~1.0)")

                    # Sanity check: V_target_batch should be consistent with G_target
                    if (global_step % 2000 == 0) and (fabric is None or fabric.is_global_zero):
                        G_reconstructed = V_target_batch @ V_target_batch.transpose(1, 2)
                        G_recon_trace = (G_reconstructed.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1)
                        G_target_trace = (Gt.diagonal(dim1=1, dim2=2) * m_bool.float()).sum(dim=1)
                        trace_ratio = (G_recon_trace / G_target_trace.clamp_min(1e-8)).mean().item()
                        print(f"[SCALE SANITY] V_target→G vs G_target trace ratio={trace_ratio:.3f} (MUST be 1.0)")
                    
                    if global_step % LOG_EVERY == 0 and not is_sc:
                        with torch.no_grad():
                            n_gated = geo_gate.sum().item()
                            if n_gated > 0:
                                sigma_gated = sigma_vec[geo_gate.bool()]
                                        
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

                        # Use adaptive gate for gram_scale
                        gate_sum_scale = geo_gate_gram_scale.sum().clamp(min=1.0)
                        
                        log_ratio_sq = log_ratio ** 2
                        # Sanitize before gating
                        log_ratio_sq = torch.nan_to_num(log_ratio_sq, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        L_gram_scale = (log_ratio_sq * geo_gate_gram_scale).sum() / gate_sum_scale
                        # ==================== LEARNED-BRANCH SCALE CALIBRATION (FIXED) ====================
                        # FIX 1: Use CORRECT RESIDUAL TARGET for the learned branch
                        #   Correct: V_out_tgt = x0_c - c_skip * x_c (not just x0_c)
                        # FIX 2: Supervise in F_x SPACE to remove c_out gradient bottleneck
                        #   Instead of supervising V_out = c_out * F_x, supervise F_x = V_out / c_out
                        # FIX 3: Gate activates earlier (c_skip < 0.25) since shrink starts at σ ≈ 0.3-0.4
                        #
                        # Why this fixes the stuck ~0.6 ratio:
                        # - Old: gradients to F_x were scaled by c_out (~0.165), too weak to climb
                        # - New: gradients hit F_x directly with full strength
                        
                        if use_edm and WEIGHTS.get('out_scale', 0) > 0:
                            with torch.autocast(device_type='cuda', enabled=False):
                                sigma_flat_f = sigma_flat.float()
                                c_skip_b, c_out_b, c_in_b, _ = uet.edm_precond(sigma_flat_f, sigma_data)
                                # c_skip_b, c_out_b: (B, 1, 1)
                                
                                # Center noisy input and clean target in the SAME way as V_geom
                                # V_geom is already centered(x0_pred) from the global geometry block
                                x_c, _ = uet.center_only(V_t, mask)       # Centered noisy input
                                x0_c, _ = uet.center_only(V_target, mask)  # Centered clean target
                                
                                # V_geom = centered(x0_pred) is already computed above
                                x0_pred_c = V_geom  # (B, N, D)
                                
                                # CORRECT learned contribution (centered): c_out * F_x = x0_pred - c_skip * x_c
                                V_out_pred = (x0_pred_c - c_skip_b * x_c) * m_float
                                
                                # CORRECT target for learned contribution: x0 - c_skip * x_c
                                V_out_tgt = (x0_c - c_skip_b * x_c) * m_float
                                
                                # FIX 3: Gate on c_skip (data-driven), not σ directly
                                # Shrink starts at σ ≈ 0.3-0.4, which corresponds to c_skip ≈ 0.2-0.3
                                # Use earlier activation threshold
                                c_skip_1d = c_skip_b.view(-1)  # (B,)
                                hi = (c_skip_1d < 0.25)  # Select high-σ samples (where c_skip is small)
                                
                                if hi.any():
                                    # FIX 2: Convert to F_x space to remove c_out gradient bottleneck
                                    # F_x = V_out / c_out (this removes the c_out multiplier from gradients)
                                    c_out_hi = c_out_b[hi].clamp(min=1e-6)  # (Bhi, 1, 1), safe division
                                    
                                    Fx_pred = V_out_pred[hi] / c_out_hi  # (Bhi, N, D)
                                    Fx_tgt = V_out_tgt[hi] / c_out_hi    # (Bhi, N, D)
                                    
                                    # RMS per-sample (mask-aware)
                                    mf_hi = m_float[hi]  # (Bhi, N, 1)
                                    denom_pts = mf_hi.sum(dim=(1, 2)).clamp(min=1.0)  # (Bhi,)
                                    
                                    # Mean square per sample
                                    ms_pred = (Fx_pred.pow(2) * mf_hi).sum(dim=(1, 2)) / denom_pts  # (Bhi,)
                                    ms_tgt = (Fx_tgt.pow(2) * mf_hi).sum(dim=(1, 2)) / denom_pts    # (Bhi,)
                                    
                                    rms_pred = ms_pred.sqrt()
                                    rms_tgt = ms_tgt.sqrt()
                                    
                                    # Log-ratio loss in F_x space
                                    log_ratio_fx = torch.log(rms_pred + 1e-8) - torch.log(rms_tgt + 1e-8)
                                    
                                    # Smooth weighting within the selected hi set using (1 - c_skip)^2
                                    gate_w = (1.0 - c_skip_1d[hi]).pow(2)  # (Bhi,)
                                    L_out_scale = ((log_ratio_fx ** 2) * gate_w).sum() / gate_w.sum().clamp(min=1.0)
                                else:
                                    L_out_scale = torch.tensor(0.0, device=device)
                                
                                # Debug logging with new metrics
                                if global_step % 100 == 0 and ((fabric is None) or (hasattr(fabric, 'is_global_zero') and fabric.is_global_zero)):
                                    with torch.no_grad():
                                        # Compute Fx_pred/Fx_tgt for all samples (not just hi)
                                        c_out_all = c_out_b.clamp(min=1e-6)
                                        Fx_pred_all = V_out_pred / c_out_all
                                        Fx_tgt_all = V_out_tgt / c_out_all
                                        
                                        denom_all = m_float.sum(dim=(1, 2)).clamp(min=1.0)
                                        rms_fx_pred_all = ((Fx_pred_all.pow(2) * m_float).sum(dim=(1, 2)) / denom_all).sqrt()
                                        rms_fx_tgt_all = ((Fx_tgt_all.pow(2) * m_float).sum(dim=(1, 2)) / denom_all).sqrt()
                                        ratio_fx_all = rms_fx_pred_all / rms_fx_tgt_all.clamp(min=1e-8)

                                        # Store high-σ Fx ratio for boost readiness (CHANGE 2)
                                        hi_mask = (c_skip_1d < 0.25)
                                        if hi_mask.any():
                                            boost_state['last_fx_ratio_hi'] = ratio_fx_all[hi_mask].median().item()
           
                                        sigma_vals = sigma_flat_f
                                        sigma_bins_out = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.2), (1.2, 2.4), (2.4, float('inf'))]
                                        
                                        print(f"\n[DEBUG-OUT-SCALE-FX] Step {global_step} - F_x space scale analysis:")
                                        print(f"  {'σ-bin':>15} | {'n':>4} | {'Fx_pred/Fx_tgt':>14} | {'c_skip':>8} | {'selected':>8}")
                                        print(f"  {'-'*15} | {'-'*4} | {'-'*14} | {'-'*8} | {'-'*8}")
                                        
                                        for lo, hi_bound in sigma_bins_out:
                                            in_bin = (sigma_vals >= lo) & (sigma_vals < hi_bound)
                                            if in_bin.any():
                                                n_bin = in_bin.sum().item()
                                                fx_ratio_bin = ratio_fx_all[in_bin].mean().item()
                                                c_skip_bin = c_skip_1d[in_bin].mean().item()
                                                # Check if these samples are in the 'hi' selection
                                                n_selected = (in_bin & hi).sum().item()
                                                
                                                bin_label = f"[{lo:.2f}, {hi_bound:.2f})"
                                                print(f"  {bin_label:>15} | {n_bin:>4} | {fx_ratio_bin:>14.4f} | {c_skip_bin:>8.4f} | {n_selected:>8}")
                                        
                                        print(f"\n  L_out_scale = {L_out_scale.item():.6f}")
                                        print(f"  Gate threshold: c_skip < 0.25 (selected {hi.sum().item()}/{len(hi)} samples)")
                                        print(f"  TARGET: Fx_pred/Fx_tgt should approach 1.0 at high σ")
                        else:
                            L_out_scale = torch.tensor(0.0, device=device)


                    # --- DEBUG: Per-σ-bin Gram loss verification ---
                    if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                        with torch.no_grad():
                            sigma_bins_gram = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, float('inf'))]
                            print(f"\n[DEBUG-GRAM-BINS] Step {global_step} - Per-σ Gram analysis:")
                            
                            for lo, hi in sigma_bins_gram:
                                in_bin = (sigma_vec >= lo) & (sigma_vec < hi)
                                if in_bin.any():
                                    # Per-set loss in this bin
                                    loss_bin = per_set_relative_loss[in_bin].mean().item()
                                    
                                    # Trace ratio for this bin
                                    tr_p_bin_val = tr_p[in_bin].mean().item()
                                    tr_t_bin_val = tr_t[in_bin].mean().item()
                                    tr_ratio_bin = tr_p_bin_val / max(tr_t_bin_val, 1e-8)
                                    
                                    # Scale loss for this bin  
                                    scale_loss_bin = (log_ratio[in_bin] ** 2).mean().item()
                                    
                                    print(f"  σ∈[{lo:.1f},{hi:.1f}): n={in_bin.sum().item()}, "
                                        f"shape_loss={loss_bin:.4f}, "
                                        f"trace_ratio={tr_ratio_bin:.3f}, "
                                        f"scale_loss={scale_loss_bin:.4f}")

                # --- kNN NCA Loss (with float32 + autocast disabled) ---
                # PATCH 6 (COMPETITOR): Use anchor_mask for point weighting
                if WEIGHTS['knn_nca'] > 0 and (not is_sc):
                    with torch.autocast(device_type='cuda', enabled=False):
                        # Get anchor_mask from batch (if available)
                        if 'anchor_mask' in batch and compete_anchor_only:
                            anchor_mask_batch = batch['anchor_mask'].to(device)  # (B, N)
                            # point_weight = mask * anchor_mask (only anchors contribute to loss)
                            point_weight = mask.float() * anchor_mask_batch.float()
                        else:
                            point_weight = None  # All valid points contribute
                        
                        # Compute NCA loss with point weighting
                        L_knn_per = uet.knn_nca_loss(
                            V_geom_L,              # Clamped structure tensor
                            V_target.float(),    # Raw target
                            mask, 
                            k=15, 
                            temperature=tau_reference,
                            return_per_sample=True,
                            scale_compensate=True,
                            point_weight=point_weight,  # NEW: anchor-only weighting
                        )
                        
                        # Gate NCA (same as before)
                        gate_sum_nca = geo_gate_nca.sum().clamp(min=1.0)
                        
                        # Sanitize before gating
                        L_knn_per = torch.nan_to_num(L_knn_per, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        L_knn_nca = (L_knn_per * geo_gate_nca).sum() / gate_sum_nca

                        # Debug - ENHANCED with competitor info
                        if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                            hit_rate_nca = (geo_gate_nca > 0).float().mean().item()
                            
                            # Log anchor coverage if competitor training
                            if point_weight is not None:
                                anchor_rate = (point_weight > 0).float().sum(dim=1).mean().item()
                                total_pts = mask.float().sum(dim=1).mean().item()
                                print(f"  NCA gate: hit={hit_rate_nca:.2%} L={L_knn_nca.item():.4f} "
                                      f"anchors/total={anchor_rate:.1f}/{total_pts:.1f}")
                            else:
                                print(f"  NCA gate: hit_rate={hit_rate_nca:.2%} L_knn_nca={L_knn_nca.item():.4f}")

                else:
                    L_knn_nca = torch.tensor(0.0, device=device)

                # ==============================================================================
                # [COMPETITOR-DIAG] Training-time competitor diagnostics (ChatGPT requested)
                # ==============================================================================
                # ========== SPOT IDENTITY: Diagnostic updated to use spot_indices ==========
                if compete_train and (global_step % compete_diag_every == 0) and (fabric is None or fabric.is_global_zero):
                    if not is_sc and 'spot_indices' in batch and 'anchor_mask' in batch:
                        with torch.no_grad():
                            spot_indices = batch['spot_indices'].to(device)  # (B, N) within-slide
                            global_uid = batch['global_indices'].to(device)  # (B, N) int64 UIDs
                            anchor_mask_b = batch['anchor_mask'].to(device)  # (B, N)
                            
                            # Get slide's full GT kNN for coverage computation
                            # Use spot_indices to index into targets_dict (per-slide lookup)
                            coverage_k = 10

                            precision_k = 10
                            
                            coverage_scores = []
                            precision_scores = []
                            intruder_scores = []
                            
                            B_diag = mask.shape[0]
                            for b in range(min(B_diag, 4)):  # Sample up to 4 batches for speed
                                m_b = mask[b]
                                anchor_b = anchor_mask_b[b] & m_b
                                spot_idx_b = spot_indices[b]  # ✅ Within-slide indices for GT lookup
                                uid_b = global_uid[b]  # UIDs for cross-sample checks (if needed)
                                slide_id_b = batch['overlap_info'][b]['slide_id']  # Extract slide
                                
                                if anchor_b.sum() < 3:
                                    continue
                                
                                # Get V_geom for this sample
                                V_b = V_geom[b][m_b]  # (n_valid, D)
                                n_valid = V_b.shape[0]
                                
                                if n_valid < coverage_k + 1:
                                    continue
                                
                                # Compute predicted kNN within batch
                                # Compute predicted kNN within batch
                                D_pred = torch.cdist(V_b, V_b)
                                D_pred.fill_diagonal_(float('inf'))
                                _, pred_knn = torch.topk(D_pred, k=coverage_k, largest=False)
                                
                                # Map to global indices using uid_b (global UIDs)
                                local_to_global = uid_b[m_b]  # (n_valid,) - FIX: was global_idx_b
                                pred_knn_global = local_to_global[pred_knn]  # (n_valid, k)
                                
                                # For coverage: we need full slide GT neighbors
                                # This requires access to the dataset's targets
                                # For now, approximate with V_target kNN as "GT"
                                V_tgt_b = V_target[b][m_b]
                                D_tgt = torch.cdist(V_tgt_b, V_tgt_b)
                                D_tgt.fill_diagonal_(float('inf'))
                                _, tgt_knn = torch.topk(D_tgt, k=coverage_k, largest=False)
                                tgt_knn_global = local_to_global[tgt_knn]
                                
                                # Compute precision: fraction of pred neighbors that are in GT neighbors
                                anchor_local = anchor_b[m_b].nonzero(as_tuple=True)[0]
                                for a in anchor_local[:10]:  # Sample anchors
                                    pred_set = set(pred_knn_global[a].tolist())
                                    tgt_set = set(tgt_knn_global[a].tolist())
                                    
                                    if len(pred_set) > 0:
                                        precision = len(pred_set & tgt_set) / len(pred_set)
                                        precision_scores.append(precision)
                                        intruder_scores.append(1.0 - precision)
                            
                            if precision_scores:
                                mean_precision = sum(precision_scores) / len(precision_scores)
                                mean_intruder = sum(intruder_scores) / len(intruder_scores)
                                
                                print(f"\n[COMPETITOR-DIAG] step={global_step}")
                                print(f"  precision@{precision_k}={mean_precision:.3f} "
                                      f"intruder@{precision_k}={mean_intruder:.3f}")
                                print(f"  (lower intruder = better competitor robustness)")
                                
                                # Log compete_debug counters if available
                                if 'compete_debug' in batch and batch['compete_debug']:
                                    cd = batch['compete_debug'][0]  # First sample's debug info
                                    print(f"  miniset composition: core={cd.get('n_core', 'N/A')} "
                                          f"pos={cd.get('n_pos', 'N/A')} rand={cd.get('n_rand', 'N/A')} "
                                          f"hard={cd.get('n_hard', 'N/A')} total={cd.get('n_total', 'N/A')}")

                
                # --- NCA sanity debug ---
                # --- NCA sanity debug (ENHANCED) ---
                if global_step % 200 == 0 and (fabric is None or fabric.is_global_zero) and not is_sc and WEIGHTS['knn_nca'] > 0:
                    with torch.no_grad():
                        n_valid = mask.sum(dim=1).float()
                        baseline = torch.log((n_valid - 1).clamp(min=1)).mean().item()
                        
                        # --- FIX: Debug NCA scale matching ---
                        # V_pred_nca = x0_pred.float() if use_edm else V_hat.float()
                        # V_tgt_nca = V_target.float()

                        # --- FIX: Debug NCA scale matching (using structure tensors) ---
                        # CHANGE 4C: Match what NCA actually sees during training
                        V_pred_nca = V_geom            # Matches what NCA actually sees
                        V_tgt_nca = V_target.float()   # Matches what NCA actually sees

                        
                        # Compute median kNN squared distances for pred and target
                        D2_pred_list = []
                        D2_tgt_list = []
                        for b in range(min(4, V_pred_nca.shape[0])):
                            m_b = mask[b].bool()
                            n_b = m_b.sum().item()
                            if n_b < 16:
                                continue
                            
                            v_p = V_pred_nca[b, m_b]
                            v_t = V_tgt_nca[b, m_b]
                            
                            d2_p = torch.cdist(v_p, v_p).pow(2)
                            d2_t = torch.cdist(v_t, v_t).pow(2)
                            
                            # Get k=15 nearest (exclude self)
                            d2_p.fill_diagonal_(float('inf'))
                            d2_t.fill_diagonal_(float('inf'))
                            
                            knn_d2_p, _ = d2_p.topk(15, largest=False, dim=1)
                            knn_d2_t, _ = d2_t.topk(15, largest=False, dim=1)
                            
                            D2_pred_list.append(knn_d2_p.median().item())
                            D2_tgt_list.append(knn_d2_t.median().item())
                        
                        if D2_pred_list:
                            med_D2_pred = np.median(D2_pred_list)
                            med_D2_tgt = np.median(D2_tgt_list)
                            ratio = med_D2_pred / max(med_D2_tgt, 1e-8)
                            
                            print(f"[DEBUG-NCA-SCALE] tau_reference={tau_reference:.4f}")
                            print(f"[DEBUG-NCA-SCALE] median(D2_pred_knn)={med_D2_pred:.4f}, "
                                f"median(D2_tgt_knn)={med_D2_tgt:.4f}, ratio={ratio:.3f}")
                            
                            if ratio < 0.5 or ratio > 2.0:
                                print(f"  ⚠️ WARNING: D2 ratio far from 1.0 - scale mismatch!")
                        
                        print(f"[NCA] n_valid_mean={n_valid.mean().item():.1f} "
                            f"uniform_baseline≈{baseline:.3f} L_knn_nca={L_knn_nca.item():.3f}")

                # --- kNN Scale Loss (CHANGE 3 - EDGE-BASED) ---
                # --- kNN Scale Loss (CHANGE 5 - EDM8 style) ---
                # KEY: knn_scale supervises SCALE learning, so it MUST see RAW (unclamped) tensor
                # Structure losses see V_geom (clamped); knn_scale sees V_hat_centered (raw)
                # This is the ONLY loss that uses V_hat_centered
                if WEIGHTS.get('knn_scale', 0) > 0 and (not is_sc):
                    with torch.autocast(device_type='cuda', enabled=False):
                        # Use knn_spatial from batch (same edges as edge loss)
                        knn_spatial_batch = batch.get('knn_spatial', None)
                        if knn_spatial_batch is not None:
                            knn_spatial_batch = knn_spatial_batch.to(device)
                        
                        # CHANGE 5: Use V_hat_centered (RAW, NOT clamped V_geom)
                        # This is critical - knn_scale needs to see actual scale error to teach it
                        L_knn_scale_per = uet.knn_scale_loss(
                            V_hat_centered_L,  # RAW centered - this is the KEY difference
                            V_target.float(),
                            mask,
                            knn_indices=knn_spatial_batch,
                            k=15,
                            return_per_sample=True
                        )
                        
                        # CHANGE 5: OWN GATE with broad coverage (decoupled from NCA)
                        # knn_scale should apply to ALL ST samples, not be gated by noise level
                        # Only require enough valid points (no cond_only per Option 1 - broader coverage)
                        scale_gate = (n_valid_per_sample >= 16).float()
                        
                        # Apply minimum coverage floor (scale can learn broadly)
                        # Use noise_score but prefer_low_score=False means we accept any noise level
                        scale_gate = ensure_minimum_coverage(
                            scale_gate, noise_score, 
                            min_count=MIN_GEOMETRY_SAMPLES,
                            prefer_low_score=False,  # Scale loss can apply to any noise level
                            eligible_mask=(n_valid_per_sample >= 16)
                        )
                        
                        # ---- FORCE per-sample shapes (B,) BEFORE ANY USE ----
                        B = mask.shape[0]
                        if scale_gate.ndim > 1:
                            scale_gate = scale_gate.view(B, -1).mean(dim=1)
                        else:
                            scale_gate = scale_gate.float()

                        gate_sum_scale = scale_gate.sum().clamp(min=1.0)
                        
                        L_knn_scale_per = torch.nan_to_num(
                            L_knn_scale_per, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        
                        # --- ENHANCED PIN-AWARE UPWEIGHT (ChatGPT Change 2 + Change 4) ---
                        # Use log_s_unclamped directly for pinned detection (more precise)
                        # Increase alpha_max to 6.0 and make it conditional on pin-rate
                        if log_s_unclamped_for_knn is not None:
                            # Pinned = |log_s_unclamped| >= max_log_correction (within tolerance)
                            tol = 0.01
                            is_pinned_raw = (log_s_unclamped_for_knn.abs() >= (max_log_corr_for_knn - tol))
                            
                            # Also require valid_scale
                            if valid_scale_for_knn is not None:
                                is_pinned = (is_pinned_raw & valid_scale_for_knn).float()
                            else:
                                is_pinned = is_pinned_raw.float()
                            
                            # ---- FORCE per-sample shape (B,) ----
                            if is_pinned.ndim > 1:
                                is_pinned = is_pinned.view(B, -1).any(dim=1).float()
                            else:
                                is_pinned = is_pinned.float()
                            
                            # Compute current pin-rate for adaptive alpha
                            current_pin_rate = is_pinned.mean().item()
                            
                            # CHANGE 4: Higher alpha_max, conditional on pin-rate
                            # alpha_eff = alpha_max * ramp * clamp((pin_rate - 0.5) / 0.5, 0, 1)
                            # This keeps things stable if pin-rate improves
                            # alpha_max = 6.0  # Pinned samples can reach 7x weight
                            # alpha_ramp = min(1.0, global_step / 2000.0)
                            # pin_rate_factor = max(0.0, min(1.0, (current_pin_rate - 0.5) / 0.5))
                            # alpha = alpha_max * alpha_ramp * pin_rate_factor

                            # Option A: Remove pin_rate_factor gating entirely
                            # Plain ramp from 0 → alpha_max over warmup steps
                            alpha_max = 3.0  # Modest value (ChatGPT suggestion)
                            alpha_warmup = 2000.0
                            alpha = alpha_max * min(1.0, global_step / alpha_warmup)
                            pin_mult = 1.0 + alpha * is_pinned.float()

                            
                            # Minimum alpha to always give some pressure                            
                            pin_mult = 1.0 + alpha * is_pinned
                            
                            # Apply multiplier
                            L_knn_scale_per_weighted = L_knn_scale_per * pin_mult
                        else:
                            pin_mult = torch.ones_like(L_knn_scale_per)
                            is_pinned = torch.zeros_like(L_knn_scale_per)
                            alpha = 0.0
                            current_pin_rate = 0.0
                            L_knn_scale_per_weighted = L_knn_scale_per
                        
                        # --- CHANGE 5: PINNED-ONLY GLOBAL RMS SCALE TERM ---
                        # --- CHANGE 5: PINNED-ONLY GLOBAL RMS SCALE TERM ---
                        # Add small global scale supervision for pinned samples only
                        # This gives an additional gradient path that doesn't rely on kNN edges

                        # default to zero so it's always defined
                        L_rms_scale_per = torch.zeros_like(L_knn_scale_per_weighted)

                        if log_s_unclamped_for_knn is not None and is_pinned.sum() > 0:
                            V_pred_rms = V_hat_centered
                            V_tgt_rms = V_target.float()
                            B = mask.shape[0]

                            if V_pred_rms.ndim == 4 and V_pred_rms.shape[0] == V_pred_rms.shape[1]:
                                idx = torch.arange(B, device=V_pred_rms.device)
                                V_pred_rms = V_pred_rms[idx, idx]  # (B, N, D)

                            if V_tgt_rms.ndim == 4 and V_tgt_rms.shape[0] == V_tgt_rms.shape[1]:
                                idx = torch.arange(B, device=V_tgt_rms.device)
                                V_tgt_rms = V_tgt_rms[idx, idx]  # (B, N, D)

                            if V_pred_rms.ndim != 3:
                                raise RuntimeError(f"V_pred_rms shape unexpected: {V_pred_rms.shape}")
                            if V_tgt_rms.ndim != 3:
                                raise RuntimeError(f"V_tgt_rms shape unexpected: {V_tgt_rms.shape}")

                            m = mask.float().unsqueeze(-1)  # (B, N, 1)
                            denom = mask.sum(dim=1).clamp(min=1)  # (B,)

                            rms_pred = (V_pred_rms.pow(2) * m).sum(dim=2).sum(dim=1) / denom
                            rms_tgt  = (V_tgt_rms.pow(2) * m).sum(dim=2).sum(dim=1) / denom

                            if torch.is_tensor(eps):
                                eps_val = eps.item() if eps.numel() == 1 else eps.min().item()
                            else:
                                eps_val = float(eps)

                            rms_pred = rms_pred.sqrt().clamp(min=eps_val)
                            rms_tgt  = rms_tgt.sqrt().clamp(min=eps_val)

                            L_rms_scale_per = torch.log(rms_pred / rms_tgt).pow(2)

                            # Only apply to pinned samples
                            L_rms_scale_per = L_rms_scale_per * is_pinned * scale_gate

                        # Add to weighted knn_scale with small coefficient
                        rms_coef = 0.1
                        L_knn_scale_per_weighted = L_knn_scale_per_weighted + rms_coef * L_rms_scale_per

                        
                        # Recompute gate sum (safe)
                        gate_sum_scale = scale_gate.sum().clamp(min=1.0)

                        L_knn_scale = (L_knn_scale_per_weighted * scale_gate).sum() / gate_sum_scale
                        
                        # Extended debug with per-sigma breakdown
                        if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                            with torch.no_grad():
                                hit = (scale_gate > 0).sum().item()
                                print(f"\n[KNN-SCALE] step={global_step} gate_hit={hit}/{len(scale_gate)} "
                                    f"L_knn_scale={L_knn_scale.item():.6f}")
                                print(f"  (Using V_hat_centered = RAW, teaching scale correction)")
                                
                                # Per-sigma breakdown
                                sigma_vec_dbg = sigma_t.view(-1).float()
                                for lo, hi_bound in [(0.0, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 5.0)]:
                                    in_bin = (sigma_vec_dbg >= lo) & (sigma_vec_dbg < hi_bound) & (scale_gate > 0)
                                    if in_bin.any():
                                        loss_bin = L_knn_scale_per[in_bin].mean().item()
                                        print(f"    σ∈[{lo:.1f},{hi_bound:.1f}): L_knn_scale={loss_bin:.6f} n={in_bin.sum().item()}")
                                
                                # --- PIN-UPWEIGHT DEBUG ---
                                pinned_frac = is_pinned.mean().item()
                                mean_mult = pin_mult.mean().item()
                                print(f"  [PIN-UPWEIGHT] pinned_frac={pinned_frac:.1%} mean_mult={mean_mult:.3f} "
                                    f"alpha={alpha:.3f}")
                                
                                # Split loss by pinned vs unpinned
                                gated_mask = (scale_gate > 0)
                                pinned_and_gated = (is_pinned > 0) & gated_mask
                                unpinned_and_gated = (is_pinned == 0) & gated_mask
                                
                                if pinned_and_gated.any():
                                    L_pinned = L_knn_scale_per[pinned_and_gated].mean().item()
                                    print(f"    L_knn_scale (PINNED): {L_pinned:.6f} n={pinned_and_gated.sum().item()}")
                                if unpinned_and_gated.any():
                                    L_unpinned = L_knn_scale_per[unpinned_and_gated].mean().item()
                                    print(f"    L_knn_scale (unpinned): {L_unpinned:.6f} n={unpinned_and_gated.sum().item()}")
                                
                                # RMS scale debug
                                if log_s_unclamped_for_knn is not None and 'L_rms_scale_per' in dir():
                                    L_rms_pinned = (L_rms_scale_per[pinned_and_gated]).mean().item() if pinned_and_gated.any() else 0.0
                                    print(f"    L_rms_scale (pinned only): {L_rms_pinned:.6f}")
                else:
                    L_knn_scale = torch.tensor(0.0, device=device)



                # Low-noise gating - use actual drop_mask for consistency with CFG
                if WEIGHTS['edge'] > 0 and 'knn_spatial' in batch:
                    with torch.autocast(device_type='cuda', enabled=False):
                        # Use SPATIAL kNN for geometry, not expression kNN
                        knn_indices_batch = batch['knn_spatial'].to(device)
                        
                        # PATCH 3B (CORRECTED): Re-enable edge gating with stricter threshold
                        # Target: ~10-30% hit rate for edge (local features)
                        if rho is not None:
                            # Edges need cleaner signal - stricter than Gram
                            low_noise_edge = (rho <= 2.0)  # Start stricter, calibrate
                        else:
                            low_noise_edge = (sigma_vec <= 0.3)  # Stricter fallback
                        # geo_gate_edge = cond_only * low_noise_edge.float()
                        
                        # PATCH 3B DEBUG
                        if global_step % 100 == 0 and (fabric is None or fabric.is_global_zero):
                            hit_rate_edge = (geo_gate_edge > 0).float().mean().item()
                            print(f"  Edge gate: hit_rate={hit_rate_edge:.2%}")
                            if hit_rate_edge < 0.05:
                                print(f"  ⚠️ Edge hit rate very LOW (<5%) - relax threshold or check sigma sampling")

                        gate_sum_edge = geo_gate_edge.sum().clamp(min=1.0)

                        # --- [DEBUG] Log Edge Gate Hit-Rate ---
                        if global_step % 10 == 0 and (fabric is None or fabric.is_global_zero):
                             hit_rate_edge = geo_gate_edge.float().mean().item()
                             if hit_rate_edge < 0.1:
                                 print(f"   ⚠️ WARNING: Edge gate is killing >90% of samples! Relax rho or check sampling.")


                        # PATCH 5: Use log-ratio edge loss for multiplicative error
                        # Use adaptive gate for edge
                        gate_sum_edge = geo_gate_edge.sum().clamp(min=1.0)
                        
                        # L_edge_per = uet.edge_log_ratio_loss(
                        #     V_pred=x0_pred.float() if use_edm else V_hat.float(),
                        #     V_tgt=V_target.float(),
                        #     knn_idx=knn_indices_batch,
                        #     mask=mask
                        # )
                        # CHANGE 3: Use V_geom (clamped) for edge loss
                        # Edge is a STRUCTURE loss - it measures local neighborhood shape
                        # Using clamped V_geom removes systematic scale bias from gradients
                        # Scale learning happens via knn_scale on V_hat_centered
                        L_edge_per = uet.edge_log_ratio_loss(
                            V_pred=V_geom_L,  # Clamped structure tensor, NOT raw x0_pred
                            V_tgt=V_target.float(),
                            knn_idx=knn_indices_batch,
                            mask=mask
                        )

                        
                        # Sanitize before gating (NaN * 0 = NaN)
                        L_edge_per = torch.nan_to_num(L_edge_per, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        L_edge = (L_edge_per * geo_gate_edge).sum() / gate_sum_edge


                        # k-NN preservation check (identity, not just distance)
                        # =================================================================
                        # [DEBUG] 3-WAY KNN JACCARD DIAGNOSTIC & SCALE CHECKS
                        # =================================================================
                        if global_step % 25 == 0 and (fabric is None or fabric.is_global_zero):
                            with torch.no_grad():
                                # ✅ Use CLONED version
                                Tgt = V_target_orig

                                # --- A. COMPUTE SCALE & ERROR RMS ---
                                mask_f = mask_orig.unsqueeze(-1).float()
                                valid_elements = mask_f.sum() * Tgt.shape[-1]
                                
                                def get_rms(tensor):
                                    return (tensor.pow(2) * mask_f).sum().div(valid_elements + 1e-8).sqrt().item()

                                rms_tgt   = get_rms(Tgt)
                                rms_hat   = get_rms(V_hat)
                                rms_noise = get_rms(V_t - Tgt)     # Magnitude of added noise
                                rms_pred  = get_rms(V_hat - Tgt)   # Magnitude of prediction error

                                # Compute Median NN Distance on Target
                                median_nn = 0.0
                                for b_idx in range(Tgt.shape[0]):
                                    m_b = mask_orig[b_idx].bool()
                                    if m_b.sum() > 5:
                                        tgt_b = Tgt[b_idx][m_b]
                                        d_b = torch.cdist(tgt_b, tgt_b)
                                        d_b.fill_diagonal_(float('inf'))
                                        nn_dists = d_b.min(dim=1).values
                                        median_nn = nn_dists.median().item()
                                        break

                                # --- B. Helper Function ---
                                def run_jaccard_check(V_A, V_B, m_tensor, k=10, name=""):
                                    if V_A.dim() == 2: V_A = V_A.unsqueeze(0)
                                    if V_B.dim() == 2: V_B = V_B.unsqueeze(0)
                                    if m_tensor.dim() == 1: m_tensor = m_tensor.unsqueeze(0)

                                    if V_A.dim() != 3 or V_B.dim() != 3:
                                        return 0.0
                                    
                                    scores = []
                                    for b_idx in range(V_A.shape[0]):
                                        valid = m_tensor[b_idx].bool()
                                        n_pts = valid.sum().item()
                                        
                                        if n_pts < k + 2: continue
                                        
                                        try:
                                            va = V_A[b_idx][valid].float()
                                            vb = V_B[b_idx][valid].float()
                                        except IndexError:
                                            continue

                                        da = torch.cdist(va, va)
                                        db = torch.cdist(vb, vb)
                                        
                                        da.fill_diagonal_(float('inf'))
                                        db.fill_diagonal_(float('inf'))
                                        
                                        _, idx_a = da.topk(min(k, n_pts-1), largest=False)
                                        _, idx_b = db.topk(min(k, n_pts-1), largest=False)
                                        
                                        sample_scores = []
                                        ia_list = idx_a.cpu().tolist()
                                        ib_list = idx_b.cpu().tolist()
                                        
                                        for row_a, row_b in zip(ia_list, ib_list):
                                            sa, sb = set(row_a), set(row_b)
                                            intersect = len(sa & sb)
                                            union = len(sa | sb)
                                            sample_scores.append(intersect / max(union, 1))
                                        
                                        scores.append(sum(sample_scores) / len(sample_scores))
                                    
                                    return sum(scores) / len(scores) if scores else 0.0

                                # --- C. Compute Jaccard ---
                                k_check = 10
                                score_sanity = run_jaccard_check(Tgt, Tgt, mask_orig, k=k_check, name="Sanity")
                                score_noise = run_jaccard_check(V_t, Tgt, mask_orig, k=k_check, name="Input_Noise")
                                score_pred = run_jaccard_check(V_hat, Tgt, mask_orig, k=k_check, name="Prediction")
                                
                                # --- D. Print Report ---
                                sigma_val = sigma_t_orig.mean().item()
                                print(f"\n[KNN DIAGNOSTIC] step={global_step} (sigma_avg={sigma_val:.3f})")

                                # --- TRANSLATION NOISE DIAGNOSTIC ---
                                noise_diagnostics = []
                                sigma_list = []
                                for b_idx in range(Tgt.shape[0]):
                                    m_b = mask_orig[b_idx].bool()
                                    if m_b.sum() > 5:
                                        noise_b = (V_t[b_idx] - Tgt[b_idx])[m_b]
                                        
                                        noise_point_std = noise_b.std(dim=0).mean().item()
                                        noise_point_mean_norm = noise_b.mean(dim=0).norm().item()
                                        sig_b = float(sigma_t_orig[b_idx].view(-1)[0])
                                        
                                        noise_diagnostics.append((noise_point_std, noise_point_mean_norm, sig_b))
                                        sigma_list.append(sig_b)

                                if not noise_diagnostics:
                                    # Compute per-sample ratios (the TRUE test of noise correctness)
                                    ratios = [d[0] / max(d[2], 1e-8) for d in noise_diagnostics]
                                    ratio_med = torch.tensor(ratios).median().item()
                                    ratio_mean = torch.tensor(ratios).mean().item()
                                    ratio_std = torch.tensor(ratios).std().item()
                                    
                                    # Compute mean_norm ratios (normalized by sigma and sqrt(D/n))
                                    # For each sample: expected_mean_norm ≈ sigma * sqrt(D/n)
                                    D_latent = Tgt.shape[-1]  # Usually 16
                                    mean_norm_ratios = []
                                    for noise_std, mean_norm, sig in noise_diagnostics:
                                        # Find n for this sample (approximate from noise_diagnostics context)
                                        # The mean_norm should be ~ sig * sqrt(D/n), so ratio should be ~sqrt(D/n)
                                        # We normalize by sigma to get: mean_norm/sigma, which should be ~sqrt(D/n)
                                        normalized = mean_norm / max(sig, 1e-8)
                                        mean_norm_ratios.append(normalized)
                                    
                                    mean_norm_ratio_med = torch.tensor(mean_norm_ratios).median().item()
                                    
                                    sigma_rms = sigma_t_orig.pow(2).mean().sqrt().item()
                                    
                                    print(f"  --- Noise Structure Analysis ---")
                                    print(f"  sigma_rms: {sigma_rms:.4f}")
                                    print(f"  Per-sample noise_std/sigma: median={ratio_med:.3f} mean={ratio_mean:.3f} std={ratio_std:.3f}")
                                    print(f"  Per-sample mean_norm/sigma: median={mean_norm_ratio_med:.3f} (expected ~sqrt(D/n) ≈ 0.2-0.3)")
                                    
                                    # The TRUE test: per-sample ratio should be ~1.0
                                    if ratio_med < 0.7:
                                        print(f"  🔴 FAIL: Noise is TRANSLATION-LIKE (ratio={ratio_med:.3f})")
                                    elif ratio_med > 1.3:
                                        print(f"  🔴 FAIL: Noise is TOO LARGE (ratio={ratio_med:.3f})")
                                    elif ratio_std > 0.3:
                                        print(f"  ⚠️ WARNING: High variance in noise ratios (std={ratio_std:.3f})")
                                    elif mean_norm_ratio_med > 1.0:
                                        # mean_norm/sigma > 1.0 means the mean shift is larger than sigma itself
                                        # This would indicate systematic bias, not random noise
                                        print(f"  ⚠️ WARNING: Mean shift is large relative to sigma (ratio={mean_norm_ratio_med:.3f})")
                                    else:
                                        print(f"  ✅ PASS: Noise appears to be proper per-point iid")


                                print(f"  --- Scale & Error Stats ---")
                                print(f"  RMS(Tgt):   {rms_tgt:.4f} | Median NN(Tgt): {median_nn:.4f}")
                                print(f"  RMS(Noise): {rms_noise:.4f} (Input |V_t - Tgt|)")
                                print(f"  RMS(Pred):  {rms_hat:.4f} (Output Scale)")
                                print(f"  RMS(Error): {rms_pred:.4f} (Output |V_hat - Tgt|)")
                                
                                print(f"  --- Jaccard Scores (k={k_check}) ---")
                                print(f"  1. GT  vs GT   (Sanity): {score_sanity:.4f}")
                                print(f"  2. V_t vs GT   (Input):  {score_noise:.4f}")
                                print(f"  3. Pred vs GT  (Output): {score_pred:.4f}")
                                
                                if score_sanity < 0.99:
                                    print("  🔴 FAIL: Sanity check < 1.0. Indices misaligned!")
                                elif rms_hat < 0.1 * rms_tgt:
                                    print(f"  🔴 FAIL: Model COLLAPSED to zero (RMS ratio {rms_hat/rms_tgt:.3f})")
                                elif rms_pred > rms_noise:
                                    print(f"  ⚠️ WARNING: Destructive! Output error ({rms_pred:.3f}) > Input noise ({rms_noise:.3f})")
                                elif score_pred < score_noise:
                                    print("  ⚠️ WARNING: Jaccard WORSE than input. Geometry scrambled.")
                                elif score_pred > score_noise + 0.05:
                                    print(f"  ✅ PASS: Improving structure (+{score_pred - score_noise:.3f})")
                                else:
                                    print("  ⚪ NEUTRAL: Output ≈ Input.")
                                print("="*60)


                                # ✅ ADD THIS HERE:
                                print(f"\n  --- Per-Sigma Jaccard (bins) ---")
                                sigma_bins = [
                                    (0.0, 0.05),
                                    (0.05, 0.1),
                                    (0.1, 0.3),
                                    (0.3, 0.6),
                                    (0.6, 1.0),
                                    (1.0, 2.0),
                                    (2.0, 5.0)
                                ]

                                for bin_idx, (sigma_low, sigma_high) in enumerate(sigma_bins):
                                    mask_bin = (sigma_t_orig.squeeze() >= sigma_low) & (sigma_t_orig.squeeze() < sigma_high)
                                    if mask_bin.sum() > 0:
                                        # Get samples in this sigma range
                                        V_hat_bin = V_hat[mask_bin]
                                        V_tgt_bin = V_target_orig[mask_bin]
                                        mask_bin_expanded = mask_orig[mask_bin]
                                        
                                        # Compute Jaccard for this bin
                                        jaccard_bin = run_jaccard_check(V_hat_bin, V_tgt_bin, mask_bin_expanded, k=k_check)
                                        n_samples = mask_bin.sum().item()
                                        sigma_mean = sigma_t_orig.squeeze()[mask_bin].mean().item()
                                        
                                        print(f"  Bin {bin_idx} σ∈[{sigma_low:.2f},{sigma_high:.2f}] "
                                            f"(n={n_samples}, σ_avg={sigma_mean:.3f}): Jaccard={jaccard_bin:.3f}")


                        if global_step % 25 == 0 and (fabric is None or fabric.is_global_zero):
                            with torch.no_grad():
                                k_check = min(10, knn_indices_batch.shape[-1])
                                jaccard_scores = []
                                
                                for b_idx in range(min(2, B)):  # Check first 2 samples
                                    m_b = m_bool[b_idx]
                                    n_valid = m_b.sum().item()
                                    if n_valid < k_check + 1:
                                        continue
                                    
                                    V_p = V_geom[b_idx][m_b]  # (n_valid, D)
                                    # V_t = V_target_batch[b_idx][m_b]  # (n_valid, D)
                                    V_tgt_local = V_target_batch[b_idx][m_b]
                                    
                                    # Compute k-NN in pred and target space
                                    D_p = torch.cdist(V_p, V_p)
                                    # D_t = torch.cdist(V_t, V_t)
                                    D_t = torch.cdist(V_tgt_local, V_tgt_local)
                                    
                                    # Get k-NN indices (exclude self)
                                    knn_pred = D_p.topk(k_check + 1, largest=False).indices[:, 1:]  # (n_valid, k)
                                    knn_tgt = D_t.topk(k_check + 1, largest=False).indices[:, 1:]   # (n_valid, k)
                                    
                                    # Jaccard per point
                                    for i in range(n_valid):
                                        set_p = set(knn_pred[i].tolist())
                                        set_t = set(knn_tgt[i].tolist())
                                        jaccard = len(set_p & set_t) / len(set_p | set_t)
                                        jaccard_scores.append(jaccard)
                                
                                if jaccard_scores:
                                    jaccard_arr = torch.tensor(jaccard_scores)
                    
                        # --- DEBUG 4: Generator vs diffusion neighbor comparison ---
                        if global_step % 25 == 0 and (fabric is None or fabric.is_global_zero) and not is_sc:
                            with torch.no_grad():
                                print(f"\n[DEBUG-GEN-VS-DIFF] Step {global_step} - Generator vs Diffusion neighbor structure:")
                                
                                # Get generator output
                                V_gen_debug = generator(H_train, mask)
                                
                                # Compute k-NN Jaccard for both
                                k = 15
                                for name, V_test in [("Generator", V_gen_debug), ("Diffusion", x0_pred)]:
                                    # Sample a few points from batch
                                    b_idx = 0
                                    m = mask[b_idx]
                                    valid_idx = m.nonzero(as_tuple=True)[0]
                                    
                                    if len(valid_idx) > 20:
                                        v_test = V_test[b_idx:b_idx+1]
                                        v_tgt = V_target[b_idx:b_idx+1]
                                        m_batch = mask[b_idx:b_idx+1]
                                        
                                        jaccard = uet.knn_jaccard_soft(v_test, v_tgt, m_batch, k=k)
                                        print(f"  {name}: k={k} Jaccard={jaccard.item():.4f}")
                                
                                # Per-sigma breakdown for diffusion
                                for lo, hi in [(0.0, 0.3), (0.3, 1.0), (1.0, float('inf'))]:
                                    in_bin = (sigma_flat >= lo) & (sigma_flat < hi)
                                    if in_bin.any() and in_bin.sum() >= 2:
                                        jaccard_bin = uet.knn_jaccard_soft(
                                            x0_pred[in_bin][:2], V_target[in_bin][:2], 
                                            mask[in_bin][:2], k=k
                                        )
                                        print(f"  Diffusion σ∈[{lo:.1f},{hi:.1f}): Jaccard={jaccard_bin.item():.4f}")


                        # ========== COMPREHENSIVE GEOMETRY DIAGNOSTICS ==========
                        if (global_step % 25 == 0) and (not is_sc) and (fabric is None or fabric.is_global_zero):
                            with torch.no_grad():
                                print(f"\n{'='*80}")
                                print(f"[GEOMETRY DIAGNOSTIC] step={global_step}")
                                print(f"{'='*80}")
                                
                                # === 1. TENSOR CONSISTENCY CHECK ===
                                print(f"\n[1] TENSOR SOURCES:")
                                print(f"  V_hat (diffusion output): shape={V_hat.shape}")
                                print(f"  V_geom (centered V_hat):  shape={V_geom.shape}")
                                print(f"  V_target_batch (from G):  shape={V_target_batch.shape}")
                                print(f"  G_target (ground truth):  shape={G_target.shape}")
                                
                                # Check if batch['V_target'] exists and differs
                                if 'V_target' in batch:
                                    V_raw = batch['V_target'].to(device).float()
                                    print(f"  batch['V_target'] (raw): shape={V_raw.shape}")
                                    
                                    # Compare V_target_batch vs batch['V_target']
                                    # --- DEBUG 5: Check multiple sigma bins ---
                                    sigma_for_track = sigma_flat if use_edm else sigma_t.view(-1)
                                    debug_samples = get_one_sample_per_sigma_bin(sigma_for_track)
                                    
                                    for bin_name, b_idx in debug_samples[:2]:
                                        m_b = mask[b_idx].bool()
                                        n_valid = m_b.sum().item()
                                        sigma_val = sigma_for_track[b_idx].item()
                                        if n_valid > 5:
                                            print(f"  [{bin_name}] σ={sigma_val:.3f} sample={b_idx}:")
                                            v_from_gram = V_target_batch[b_idx, m_b]
                                            v_raw = V_raw[b_idx, m_b]
                                            
                                            # Center both for fair comparison
                                            v_from_gram_c = v_from_gram - v_from_gram.mean(dim=0)
                                            v_raw_c = v_raw - v_raw.mean(dim=0)
                                            
                                            diff_rms = (v_from_gram_c - v_raw_c).pow(2).mean().sqrt().item()
                                            scale_ratio = v_from_gram_c.pow(2).mean().sqrt() / v_raw_c.pow(2).mean().sqrt().clamp_min(1e-8)
                                            
                                            print(f"  V_target_batch vs batch['V_target'] (sample 0):")
                                            print(f"    RMS diff: {diff_rms:.6f}")
                                            print(f"    Scale ratio: {scale_ratio.item():.6f} (should be ~1.0)")
                                            
                                            if abs(scale_ratio.item() - 1.0) > 0.1:
                                                print(f"    ⚠️ WARNING: V_target sources differ by {abs(scale_ratio.item()-1.0)*100:.1f}%!")
                                
                                # === 2. GRAM MATRIX RECONSTRUCTION CHECK ===
                                print(f"\n[2] GRAM RECONSTRUCTION:")
                                # --- DEBUG 5: Check across sigma bins ---
                                sigma_for_gram = sigma_flat if use_edm else sigma_t.view(-1)
                                debug_samples = get_one_sample_per_sigma_bin(sigma_for_gram)
                                
                                for bin_name, b_idx in debug_samples[:2]:
                                    m_b = mask[b_idx].bool()
                                    sigma_val = sigma_for_gram[b_idx].item()
                                    print(f"  [{bin_name}] σ={sigma_val:.3f}:")
                                    n_valid = m_b.sum().item()
                                    
                                    if n_valid > 5:
                                        # Reconstruct Gram from V_target_batch
                                        v_tgt = V_target_batch[b_idx, m_b]
                                        G_reconstructed = v_tgt @ v_tgt.T
                                        G_true = G_target[b_idx, :n_valid, :n_valid].float()
                                        
                                        # Compare traces
                                        tr_recon = G_reconstructed.diag().sum().item()
                                        tr_true = G_true.diag().sum().item()
                                        tr_ratio = tr_recon / max(tr_true, 1e-8)
                                        
                                        # Compare off-diagonal
                                        eye_mask = ~torch.eye(n_valid, device=device).bool()
                                        g_recon_off = G_reconstructed[eye_mask]
                                        g_true_off = G_true[eye_mask]
                                        frob_err = (g_recon_off - g_true_off).norm() / g_true_off.norm().clamp_min(1e-8)
                                        
                                        print(f"  Sample 0 (n={n_valid}):")
                                        print(f"    Gram trace: recon={tr_recon:.3f} vs true={tr_true:.3f} ratio={tr_ratio:.6f}")
                                        print(f"    Offdiag Frob: {frob_err.item():.6f} (should be <0.01)")
                                        
                                        if abs(tr_ratio - 1.0) > 0.05:
                                            print(f"    🔴 FAIL: factor_from_gram scale mismatch!")
                                
                                # === 3. LOSS INPUT VERIFICATION ===
                                print(f"\n[3] LOSS INPUTS:")
                                
                                # Gram loss
                                if WEIGHTS['gram'] > 0:
                                    # --- DEBUG 5: Show gram loss for different sigma bins ---
                                    sigma_for_gram_loss = sigma_flat if use_edm else sigma_t.view(-1)
                                    debug_samples = get_one_sample_per_sigma_bin(sigma_for_gram_loss)
                                    
                                    for bin_name, b_idx in debug_samples[:2]:
                                        m_b = mask[b_idx].bool()
                                        sigma_val = sigma_for_gram_loss[b_idx].item()
                                        n_valid = m_b.sum().item()
                                        if n_valid > 5:
                                            v_hat_b = V_geom[b_idx, m_b]
                                            G_pred_b = v_hat_b @ v_hat_b.T
                                            G_tgt_b = G_target[b_idx, :n_valid, :n_valid].float()
                                            
                                            print(f"  GRAM LOSS (sample 0):")
                                            print(f"  GRAM LOSS [{bin_name}] σ={sigma_val:.3f} sample={b_idx}:")
                                            print(f"    Input: V_geom[{b_idx}, valid] -> Gp_raw")
                                            print(f"    Target: G_target[{b_idx}, :n, :n]")
                                            print(f"    Gram_pred trace: {G_pred_b.diag().sum().item():.3f}")
                                            print(f"    Gram_tgt trace:  {G_tgt_b.diag().sum().item():.3f}")
                                            print(f"    Ratio: {(G_pred_b.diag().sum() / G_tgt_b.diag().sum().clamp_min(1e-8)).item():.6f}")
                                
                                # Edge loss
                                if WEIGHTS['edge'] > 0 and 'knn_spatial' in batch:
                                    knn_idx = batch['knn_spatial'].to(device)
                                    # --- DEBUG 5: Check edge loss across sigma bins ---
                                    sigma_for_edge = sigma_flat if use_edm else sigma_t.view(-1)
                                    debug_samples = get_one_sample_per_sigma_bin(sigma_for_edge)
                                    
                                    for bin_name, b_idx in debug_samples[:2]:
                                        m_b = mask[b_idx].bool()
                                        sigma_val = sigma_for_edge[b_idx].item()
                                        n_valid = m_b.sum().item()
                                        
                                        if n_valid > 10:
                                            v_pred_b = V_geom[b_idx, m_b]
                                            v_tgt_b = V_target_batch[b_idx, m_b]
                                            knn_b = knn_idx[b_idx, m_b]
                                            
                                            # Compute sample edge lengths
                                            valid_edges = (knn_b >= 0) & (knn_b < n_valid)
                                            if valid_edges.sum() > 0:
                                                i_idx = torch.arange(n_valid, device=device).unsqueeze(1).expand_as(knn_b)
                                                i_valid = i_idx[valid_edges]
                                                j_valid = knn_b[valid_edges]
                                                
                                                d_pred = (v_pred_b[i_valid] - v_pred_b[j_valid]).norm(dim=-1)
                                                d_tgt = (v_tgt_b[i_valid] - v_tgt_b[j_valid]).norm(dim=-1)
                                                
                                                print(f"  EDGE LOSS (sample 0, {i_valid.shape[0]} edges):")
                                                print(f"  EDGE LOSS [{bin_name}] σ={sigma_val:.3f} sample={b_idx}:")
                                                print(f"    Input: V_geom[{b_idx}, valid]")
                                                print(f"    Target: V_target_batch[{b_idx}, valid]")
                                                print(f"    d_pred: min={d_pred.min():.4f} med={d_pred.median():.4f} max={d_pred.max():.4f}")
                                                print(f"    d_tgt:  min={d_tgt.min():.4f} med={d_tgt.median():.4f} max={d_tgt.max():.4f}")
                                                print(f"    Ratio (pred/tgt): {(d_pred.median() / d_tgt.median().clamp_min(1e-8)).item():.6f}")
                                
                                # === 4. GLOBAL SCALE CONSISTENCY ===
                                print(f"\n[4] GLOBAL SCALE:")
                                # --- DEBUG 5: Check scale across sigma bins ---
                                sigma_for_scale = sigma_flat if use_edm else sigma_t.view(-1)
                                debug_samples = get_one_sample_per_sigma_bin(sigma_for_scale)
                                
                                for bin_name, b_idx in debug_samples:
                                    m_b = mask[b_idx].bool()
                                    sigma_val = sigma_for_scale[b_idx].item()
                                    n_valid = m_b.sum().item()
                                    print(f"  [{bin_name}] σ={sigma_val:.3f} sample={b_idx}:")
                                    n_valid = m_b.sum().item()
                                    
                                    if n_valid > 5:
                                        v_hat_b = V_geom[b_idx, m_b]
                                        v_tgt_b = V_target_batch[b_idx, m_b]
                                        
                                        # RMS
                                        rms_hat = v_hat_b.pow(2).mean().sqrt().item()
                                        rms_tgt = v_tgt_b.pow(2).mean().sqrt().item()
                                        
                                        # Median pairwise distance
                                        D_hat = torch.cdist(v_hat_b, v_hat_b)
                                        D_tgt = torch.cdist(v_tgt_b, v_tgt_b)
                                        triu_mask = torch.triu(torch.ones_like(D_hat, dtype=torch.bool), diagonal=1)
                                        med_d_hat = D_hat[triu_mask].median().item()
                                        med_d_tgt = D_tgt[triu_mask].median().item()
                                        
                                        print(f"  Sample 0 (n={n_valid}):")
                                        print(f"    RMS: pred={rms_hat:.4f} tgt={rms_tgt:.4f} ratio={rms_hat/max(rms_tgt,1e-8):.6f}")
                                        print(f"    Median dist: pred={med_d_hat:.4f} tgt={med_d_tgt:.4f} ratio={med_d_hat/max(med_d_tgt,1e-8):.6f}")
                                        
                                        # These should match if losses are working
                                        gram_ratio = (rms_hat**2) / max(rms_tgt**2, 1e-8)
                                        dist_ratio = med_d_hat / max(med_d_tgt, 1e-8)
                                        expected_dist_from_gram = gram_ratio ** 0.5
                                        
                                        print(f"    Consistency: gram_ratio={gram_ratio:.6f} dist_ratio={dist_ratio:.6f}")
                                        print(f"    Expected dist from gram: {expected_dist_from_gram:.6f}")
                                        print(f"    Error: {abs(dist_ratio - expected_dist_from_gram):.6f} (should be <0.05)")
                                
                                print(f"{'='*80}\n")


                # ==================== TOPOLOGY LOSS ====================
                if WEIGHTS['topo'] > 0:
                    with torch.autocast(device_type='cuda', enabled=False):
                        topo_info = batch.get('topo_info', None)
                        # Low-noise gating (more aggressive for topology)
                        if rho is not None:
                            low_noise = (rho <= 0.3)  # Strictest: topology needs very clean signal
                        else:
                            low_noise = (t_norm.squeeze() < 0.4)  # Fallback
                        geo_gate_topo = (low_noise.float() * cond_only)
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
                
                        # Low-noise gating
                        # SNR-based gating (shape spectrum = global → moderate threshold)
                        if rho is not None:
                            low_noise = (rho <= 1.0)  # Same as gram (global structure)
                        else:
                            low_noise = (t_norm.squeeze() < 0.6)  # Fallback
                        geo_gate_shape = (low_noise.float() * cond_only)
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
                # PATCH 8: Scale-aware generator alignment losses
                L_gen_scale = torch.tensor(0.0, device=device)
                
                if WEIGHTS['gen_align'] > 0 and (not is_sc):
                    with torch.autocast(device_type='cuda', enabled=False):
                        # Center V_gen
                        V_gen_f32 = V_gen.float()
                        m_float_gen = mask.float().unsqueeze(-1)
                        valid_counts_gen = mask.sum(dim=1, keepdim=True).clamp(min=1)
                        mean_Vgen = (V_gen_f32 * m_float_gen).sum(dim=1, keepdim=True) / valid_counts_gen.unsqueeze(-1)
                        V_gen_centered = (V_gen_f32 - mean_Vgen) * m_float_gen
                        
                        # ==================== CHANGE 6: GENERATOR SCALE CLAMP (Option 1 Change B) ====================

                        # PATCH 8: Scale-aware generator alignment losses
                        # Use knn_spatial for generator scale clamp (same as diffusion)
                        knn_spatial_for_gen = batch.get('knn_spatial', None)
                        if knn_spatial_for_gen is not None:
                            knn_spatial_for_gen = knn_spatial_for_gen.to(device)
                            
                            # Compute scale correction for GENERATOR (separate from diffusion)
                            s_corr_gen, _, valid_scale_gen, _, _, _ = uet.compute_local_scale_correction(
                                V_pred=V_gen_centered,  # Generator output, NOT V_hat_centered
                                V_tgt=V_target.float(),
                                mask=mask,
                                knn_indices=knn_spatial_for_gen,
                                k=15,
                                max_log_correction=max_log_corr_for_knn if max_log_corr_for_knn is not None else 0.25,
                            )
                            
                            # Apply scale correction to generator
                            V_gen_geo = uet.apply_scale_correction(V_gen_centered, s_corr_gen.detach(), mask)
                        else:
                            V_gen_geo = V_gen_centered

                        # ========== ANCHOR-CLAMP GENERATOR TENSORS ==========
                        if anchor_train and anchor_geom_losses and anchor_cond_mask is not None and use_anchor_geom:
                            V_target_centered_gen, _ = uet.center_only(V_target.float(), mask)
                            V_target_centered_gen = V_target_centered_gen * mask.float().unsqueeze(-1)
                            V_gen_geo_L = uet.clamp_anchors_for_loss(V_gen_geo, V_target_centered_gen, anchor_cond_mask, mask)
                            V_gen_centered_L = uet.clamp_anchors_for_loss(V_gen_centered, V_target_centered_gen, anchor_cond_mask, mask)
                        else:
                            V_gen_geo_L = V_gen_geo
                            V_gen_centered_L = V_gen_centered
                        
                        # PATCH 8: rotation-only alignment (no scaling) + explicit scale loss
                        # Use anchor-clamped V_gen_geo_L for alignment
                        L_gen_align = uet.rigid_align_mse_no_scale(V_gen_geo_L, V_target_batch, mask)
                        # Use anchor-clamped V_gen_centered_L for scale loss
                        L_gen_scale = uet.rms_log_loss(V_gen_centered_L, V_target_batch, mask)
                        
                        # CHANGE 6 ADDITION: Local scale loss for generator (like knn_scale)
                        # This ensures generator learns correct local neighborhood distances
                        if knn_spatial_for_gen is not None:
                            L_gen_scale_local = uet.knn_scale_loss(
                                V_gen_centered_L,  # Anchor-clamped generator output

                                V_target_batch,
                                mask,
                                knn_indices=knn_spatial_for_gen,
                                k=15,
                                return_per_sample=False  # Scalar loss
                            )
                            # Combine global and local scale supervision
                            L_gen_scale = L_gen_scale + 0.5 * L_gen_scale_local
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


                # ============================================================================
                # DEBUG: RIGID TRANSFORM INVARIANCE TEST (ST batch only)
                # ============================================================================
                if DEBUG and (global_step % 200 == 0):
                    with torch.no_grad():
                        i_test = 0
                        n_test = int(n_list[i_test].item())
                        if n_test >= 3:
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

            # ==================== GEOMETRY SCALE DIAGNOSTICS ====================
            if (not is_sc) and (global_step % 50 == 0) and global_step < 2000:
                with torch.no_grad():
                    print(f"\n[GEO SCALE] step={global_step}")

                    # Print sigma info for samples we're diagnosing
                    if batch_size_real > 0:
                        sigma_vals = sigma_t[:min(2, batch_size_real), 0, 0]
                        t_norm_vals = t_norm[:min(2, batch_size_real)]
                        print(f"  Batch sigmas: {[f'{s.item():.3f}' for s in sigma_vals]}")
                        print(f"  Batch t_norms: {[f'{t.item():.3f}' for t in t_norm_vals]}")
                    
                    # 1) Edge distances from EdgeLengthLoss
                    if WEIGHTS['edge'] > 0 and 'knn_spatial' in batch:
                        knn_indices = batch['knn_spatial'].to(device)
                        
                        # Pick first valid sample
                        for b in range(batch_size_real):
                            n_valid = int(n_list[b].item())
                            if n_valid < 10:
                                continue
                                
                            # Get valid predicted coords (from V_hat after centering)
                            V_b = V_hat[b, :n_valid].float()
                            m = mask[b, :n_valid]
                            
                            # Center
                            V_centered = V_b - V_b.mean(dim=0, keepdim=True)
                            
                            # Get target coords
                            V_tgt = batch['V_target'][b, :n_valid].to(device).float()
                            V_tgt_centered = V_tgt - V_tgt.mean(dim=0, keepdim=True)
                            
                            # Compute edge distances (using kNN indices)
                            knn_b = knn_indices[b, :n_valid]  # (n_valid, k)
                            valid_edges = (knn_b >= 0) & (knn_b < n_valid)
                            
                            if valid_edges.sum() > 0:
                                # Predicted distances
                                i_idx = torch.arange(n_valid, device=device).unsqueeze(1).expand_as(knn_b)
                                j_idx = knn_b
                                
                                valid_mask = valid_edges
                                i_valid = i_idx[valid_mask]
                                j_valid = j_idx[valid_mask]
                                
                                d_pred = (V_centered[i_valid] - V_centered[j_valid]).norm(dim=-1)
                                d_tgt = (V_tgt_centered[i_valid] - V_tgt_centered[j_valid]).norm(dim=-1)

                                # Print sigma for this sample
                                sigma_b = sigma_t[b, 0, 0].item()
                                t_norm_b = t_norm[b].item()
                                print(f"  [EDGE b={b}] sigma={sigma_b:.3f} t_norm={t_norm_b:.3f}")
                                
                                print(f"  [EDGE b={b}] d_pred: min={d_pred.min():.6f} "
                                    f"p10={d_pred.quantile(0.1):.6f} med={d_pred.median():.6f}")
                                print(f"  [EDGE b={b}] d_tgt:  min={d_tgt.min():.6f} "
                                    f"p10={d_tgt.quantile(0.1):.6f} med={d_tgt.median():.6f}")
                            
                            break  # Only first sample
                    
                    # 2) Gram loss vector norms
                    if WEIGHTS['gram'] > 0:
                        for b in range(min(1, batch_size_real)):  # First sample only
                            n_valid = int(n_list[b].item())
                            if n_valid < 10:
                                continue
                            
                            # Get centered V_hat
                            V_b = V_hat[b, :n_valid].float()
                            V_centered = V_b - V_b.mean(dim=0, keepdim=True)
                            
                            # Vector norms
                            v_norms = V_centered.norm(dim=-1)
                            
                            print(f"  [GRAM b={b}] ||v_i||: min={v_norms.min():.6f} "
                                f"p10={v_norms.quantile(0.1):.6f} med={v_norms.median():.6f} "
                                f"max={v_norms.max():.6f}")
                            
                            # Also check Gram diagonal
                            G_raw = V_centered @ V_centered.T
                            diag = G_raw.diag()
                            print(f"  [GRAM b={b}] G_diag: min={diag.min():.6f} "
                                f"med={diag.median():.6f} max={diag.max():.6f}")
                            
                            break
                    
                    print()  # Newline for readability
            # ==================== END GEOMETRY SCALE DIAGNOSTICS ====================

            # ==============================================================================
            # ==============================================================================
            # GATE HEALTH DEBUG (every 200 steps)
            # ==============================================================================
            if global_step % 200 == 0 and (fabric is None or fabric.is_global_zero) and use_edm:
                with torch.no_grad():
                    def hit_rate(g): 
                        return (g > 0).float().mean().item() if g.numel() > 0 else 0.0
                    
                    def format_thr(gate):
                        """Format threshold, handling None during warmup."""
                        if gate.thr is None:
                            return "warmup"
                        return f"{gate.thr:.3f}"
                    
                    nq = noise_score.detach().cpu()
                    print(f"\n[ADAPTIVE-GATES] Step {global_step}")
                    print(f"  noise_score: p10={nq.quantile(0.1).item():.3f} "
                          f"p50={nq.quantile(0.5).item():.3f} "
                          f"p90={nq.quantile(0.9).item():.3f}")
                    
                    print(f"  gram:       thr={format_thr(adaptive_gates['gram']):>8} "
                          f"hit={hit_rate(geo_gate_gram):.1%} (target=50%)")
                    print(f"  gram_scale: thr={format_thr(adaptive_gates['gram_scale']):>8} "
                          f"hit={hit_rate(geo_gate_gram_scale):.1%} (target=60%)")
                    print(f"  edge:       thr={format_thr(adaptive_gates['edge']):>8} "
                          f"hit={hit_rate(geo_gate_edge):.1%} (target=15%)")
                    print(f"  nca:        thr={format_thr(adaptive_gates['nca']):>8} "
                          f"hit={hit_rate(geo_gate_nca):.1%} (target=15%)")
                    print(f"  learn_hi:   thr={format_thr(adaptive_gates['learn_hi']):>8} "
                          f"hit={hit_rate(geo_gate_learn_hi):.1%} (target=50%)")
                    
                    # Warn if hit rates are way off
                    gram_hit = hit_rate(geo_gate_gram)
                    edge_hit = hit_rate(geo_gate_edge)
                    if gram_hit < 0.30 or gram_hit > 0.70:
                        print(f"  ⚠️ Gram hit rate outside 30-70% range")
                    if edge_hit < 0.05 or edge_hit > 0.30:
                        print(f"  ⚠️ Edge hit rate outside 5-30% range")


            L_total = (WEIGHTS['score'] * score_multiplier * L_score +
                    WEIGHTS['gram'] * L_gram +
                    WEIGHTS['gram_scale'] * L_gram_scale +
                    WEIGHTS.get('out_scale', 0) * L_out_scale +
                    WEIGHTS.get('gram_learn', 0) * L_gram_learn +
                    WEIGHTS['heat'] * L_heat +
                    WEIGHTS['sw_st'] * L_sw_st +
                    WEIGHTS['sw_sc'] * L_sw_sc +
                    WEIGHTS['overlap'] * L_overlap +
                    WEIGHTS['ordinal_sc'] * L_ordinal_sc +
                    WEIGHTS['st_dist'] * L_st_dist +
                    WEIGHTS['edm_tail'] * L_edm_tail +
                    WEIGHTS['gen_align'] * L_gen_align +
                    WEIGHTS.get('gen_scale', 0) * L_gen_scale +  # PATCH 8
                    WEIGHTS['dim'] * L_dim +
                    WEIGHTS['triangle'] * L_triangle +
                    WEIGHTS['knn_nca'] * L_knn_nca +
                    WEIGHTS.get('knn_scale', 0) * L_knn_scale +
                    WEIGHTS['radial'] * L_radial +
                    WEIGHTS['repel'] * L_repel +
                    WEIGHTS['shape'] * L_shape +
                    WEIGHTS['edge'] * L_edge +
                    WEIGHTS['topo'] * L_topo +
                    WEIGHTS['shape_spec'] * L_shape_spec +
                    WEIGHTS.get('subspace', 0) * L_subspace +  # PATCH 7
                    WEIGHTS.get('ctx_edge', 0) * L_ctx_edge +
                    effective_ctx_loss_weight * L_ctx_replace)   # NEW: context invariance

            # Add SC dimension prior if this is an SC batch
            if is_sc:
                L_total = L_total + WEIGHTS['dim'] * L_dim_sc + WEIGHTS['triangle'] * L_triangle_sc

            # ==============================================================================
            # PAIRED OVERLAP TRAINING (Candidate 1)
            # ==============================================================================
            L_ov_shape_batch = torch.tensor(0.0, device=device)
            L_ov_scale_batch = torch.tensor(0.0, device=device)
            L_ov_kl_batch = torch.tensor(0.0, device=device)
            L_score_2_batch = torch.tensor(0.0, device=device)

            if train_pair_overlap and pair_batch_full is not None and not is_sc:
                ov_pair_batches += 1
                # Use the paired batch we already have (view1 was used for main losses above)
                view2 = pair_batch_full['view2']
                idx1_I = pair_batch_full['idx1_I'].to(device)
                idx2_I = pair_batch_full['idx2_I'].to(device)
                I_mask = pair_batch_full['I_mask'].to(device)
                I_sizes = pair_batch_full['I_sizes'].to(device)

                # Move view2 data to device
                Z_set_2 = view2['Z_set'].to(device)
                mask_2 = view2['mask'].to(device)
                V_target_2 = view2['V_target'].to(device)

                B_pair = Z_set.shape[0]  # Same batch size as view1 (which is now 'batch')

                # Use the SAME sigma that was used for view1 (already computed above as sigma_t/sigma_flat)
                sigma_pair = sigma_flat  # Reuse the sigma from main batch
                sigma_pair_3d = sigma_t  # Reuse the 3D version

                # SNR gate check
                # Per-sample gate: only apply overlap loss to samples with sigma <= threshold
                ov_keep = (sigma_pair <= overlap_sigma_thresh)  # (B,)
                if ov_keep.any():
                    idx_keep = ov_keep.nonzero(as_tuple=True)[0]

                    # Slice paired tensors to only the valid samples
                    Z_set_2 = Z_set_2[idx_keep]
                    mask_2 = mask_2[idx_keep]
                    V_target_2 = V_target_2[idx_keep]

                    idx1_I = idx1_I[idx_keep]
                    idx2_I = idx2_I[idx_keep]
                    I_mask = I_mask[idx_keep]
                    I_sizes = I_sizes[idx_keep]

                    # Also slice view1 tensors to match - use _ov suffix to avoid corrupting originals
                    # HYGIENE FIX: Don't overwrite mask/eps which may be used later in the iteration
                    x0_pred_1 = x0_pred[idx_keep]
                    mask_ov = mask[idx_keep]  # Was: mask = mask[idx_keep]
                    sigma_pair_ov = sigma_pair[idx_keep]  # Was: sigma_pair = sigma_pair[idx_keep]
                    sigma_pair_3d_ov = sigma_pair_3d[idx_keep]  # Was: sigma_pair_3d = sigma_pair_3d[idx_keep]
                    eps_ov = eps[idx_keep]  # Was: eps = eps[idx_keep]
 
                    # DEBUG: gate + sigma stats
                    if global_step % overlap_debug_every == 0:
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        print(f"\n[OVLP-GATE][rank={rank}] step={global_step} epoch={epoch}")
                        print(f"  ov_keep: {ov_keep.sum().item()}/{ov_keep.numel()} kept")
                        print(f"  sigma_pair: min={sigma_pair_ov.min().item():.4f}, "
                            f"median={sigma_pair_ov.median().item():.4f}, "
                            f"max={sigma_pair_ov.max().item():.4f}, "
                            f"thresh={overlap_sigma_thresh}")

                        # ============ [VTGT-PAIR] Target consistency check ============
                        with torch.no_grad():
                            V_tgt_1 = V_target[idx_keep]
                            m1_f = mask_ov.unsqueeze(-1).float()
                            m2_f = mask_2.unsqueeze(-1).float()
                            
                            rms_tgt_1 = (V_tgt_1.pow(2) * m1_f).sum() / m1_f.sum().clamp(min=1)
                            rms_tgt_1 = rms_tgt_1.sqrt().item()
                            rms_tgt_2 = (V_target_2.pow(2) * m2_f).sum() / m2_f.sum().clamp(min=1)
                            rms_tgt_2 = rms_tgt_2.sqrt().item()
                            
                            # Gram traces
                            G_tgt_1 = V_tgt_1 @ V_tgt_1.transpose(1, 2)
                            G_tgt_2 = V_target_2 @ V_target_2.transpose(1, 2)
                            tr_tgt_1 = (G_tgt_1.diagonal(dim1=1, dim2=2) * mask_ov.float()).sum(dim=1).mean().item()
                            tr_tgt_2 = (G_tgt_2.diagonal(dim1=1, dim2=2) * mask_2.float()).sum(dim=1).mean().item()
                            
                            print(f"\n[VTGT-PAIR] step={global_step}")
                            print(f"  rms(V_target_1)={rms_tgt_1:.4f} rms(V_target_2)={rms_tgt_2:.4f}")
                            print(f"  trace(G_tgt_1)={tr_tgt_1:.4f} trace(G_tgt_2)={tr_tgt_2:.4f}")
                            if rms_tgt_2 > 0:
                                print(f"  target_rms_ratio={rms_tgt_1/rms_tgt_2:.4f} (should be ~1.0)")


                    # Sample noise for view2
                    eps_2 = torch.randn_like(V_target_2)

                    # Apply coupled noise: copy view1 noise to view2 overlap positions
                    for b in range(Z_set_2.shape[0]):
                        I_valid = I_mask[b]
                        n_I = I_valid.sum().item()
                        if n_I > 0:
                            idx1 = idx1_I[b, I_valid]
                            idx2 = idx2_I[b, I_valid]
                            eps_2[b, idx2] = eps_ov[b, idx1]  # Use sliced eps_ov
 
                    # Add noise to view2 target
                    V_t_2 = V_target_2 + sigma_pair_3d_ov * eps_2  # Use sliced sigma_pair_3d_ov
                    V_t_2 = V_t_2 * mask_2.unsqueeze(-1).float()
 
                    # Encode view2 context
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        H_2 = context_encoder(Z_set_2, mask_2)
 
                        # Forward pass for view2
                        x0_pred_2_result = score_net.forward_edm(
                            V_t_2, sigma_pair_ov, H_2, mask_2, sigma_data,  # Use sliced sigma_pair_ov
                            self_cond=None, return_debug=False
                        )

                        if isinstance(x0_pred_2_result, tuple):
                            x0_pred_2 = x0_pred_2_result[0]
                        else:
                            x0_pred_2 = x0_pred_2_result

                        # ============================================================
                        # FIX: Add score loss for view2 to prevent scale drift
                        # Without this, view2 has no scale supervision - only overlap
                        # consistency. This allows view2 scale to drift arbitrarily,
                        # and the overlap scale loss then pulls view1 down with it.
                        # ============================================================
                        mask_2_f = mask_2.unsqueeze(-1).float()
                        score_error_2 = (x0_pred_2 - V_target_2) * mask_2_f
                        L_score_2 = score_error_2.pow(2).sum() / mask_2_f.sum().clamp(min=1.0)

                        # Add view2 score loss with same weight as main score loss
                        # This anchors view2's scale to its target
                        L_total = L_total + WEIGHTS.get('score', 1.0) * L_score_2
                        L_score_2_batch = L_score_2  # Store for epoch tracking

                        # Compute overlap losses
                        ov_loss_dict = compute_overlap_losses(
                            x0_pred_1, x0_pred_2,
                            idx1_I, idx2_I, I_mask,
                            mask_ov, mask_2,
                            kl_tau=overlap_kl_tau,
                        )

                        # Apply overlap losses if valid
                        if ov_loss_dict['valid_batch_count'] > 0:
                            L_ov_shape_batch = ov_loss_dict['L_ov_shape']
                            L_ov_scale_batch = ov_loss_dict['L_ov_scale']
                            L_ov_kl_batch = ov_loss_dict['L_ov_kl']

                            # Add to total loss
                            L_total = (
                                L_total
                                + overlap_loss_weight_shape * L_ov_shape_batch
                                + overlap_loss_weight_scale * L_ov_scale_batch
                                + overlap_loss_weight_kl * L_ov_kl_batch
                            )

                            # Track epoch stats
                            ov_apply_count += 1
                            ov_loss_sum['shape'] += float(L_ov_shape_batch.item())
                            ov_loss_sum['scale'] += float(L_ov_scale_batch.item())
                            ov_loss_sum['kl'] += float(L_ov_kl_batch.item())
                            ov_loss_sum['total'] += float(
                                L_ov_shape_batch.item()
                                + L_ov_scale_batch.item()
                                + L_ov_kl_batch.item()
                            )

                            # Debug stats
                            if ov_loss_dict['debug_info']['I_sizes']:
                                ov_I_sizes.extend(ov_loss_dict['debug_info']['I_sizes'])
                            if ov_loss_dict['debug_info']['jaccard_k10']:
                                ov_jaccard_k10.extend(ov_loss_dict['debug_info']['jaccard_k10'])

                            # ============ [OVLP-VS-GLOBAL] Overlap vs Global health ============
                            if global_step % overlap_debug_every == 0:
                                with torch.no_grad():
                                    # Get overlap Jaccard from the debug info
                                    ov_jacc = ov_loss_dict['debug_info'].get('jaccard_k10', [])
                                    ov_jacc_mean = np.mean(ov_jacc) if ov_jacc else 0.0
                                    
                                    # Get overlap scale ratio from loss dict
                                    ov_scale_r = ov_loss_dict.get('debug_scale_ratio', 1.0)
                                    
                                    # Compute GLOBAL Jaccard@10 on view1 (full set, not just overlap)
                                    global_jacc_sum = 0.0
                                    global_jacc_cnt = 0
                                    for b in range(min(4, x0_pred_1.shape[0])):
                                        m_b = mask_ov[b].bool()
                                        n_valid = int(m_b.sum().item())
                                        if n_valid < 15:
                                            continue
                                        
                                        pred_b = x0_pred_1[b, m_b]
                                        tgt_b = V_target[idx_keep][b, m_b]
                                        
                                        D_pred = torch.cdist(pred_b, pred_b)
                                        D_tgt = torch.cdist(tgt_b, tgt_b)
                                        
                                        k_j = min(10, n_valid - 1)
                                        _, knn_pred = D_pred.topk(k_j + 1, largest=False)
                                        _, knn_tgt = D_tgt.topk(k_j + 1, largest=False)
                                        knn_pred = knn_pred[:, 1:]
                                        knn_tgt = knn_tgt[:, 1:]
                                        
                                        for i in range(n_valid):
                                            set_pred = set(knn_pred[i].tolist())
                                            set_tgt = set(knn_tgt[i].tolist())
                                            inter = len(set_pred & set_tgt)
                                            union = len(set_pred | set_tgt)
                                            if union > 0:
                                                global_jacc_sum += inter / union
                                                global_jacc_cnt += 1
                                    
                                    global_jacc_mean = global_jacc_sum / max(global_jacc_cnt, 1)
                                    
                                    # Global scale ratio
                                    m1_f = mask_ov.unsqueeze(-1).float()
                                    rms_pred_g = (x0_pred_1.pow(2) * m1_f).sum() / m1_f.sum().clamp(min=1)
                                    rms_pred_g = rms_pred_g.sqrt().item()
                                    rms_tgt_g = (V_target[idx_keep].pow(2) * m1_f).sum() / m1_f.sum().clamp(min=1)
                                    rms_tgt_g = rms_tgt_g.sqrt().item()
                                    global_scale_r = rms_pred_g / max(rms_tgt_g, 1e-8)
                                    
                                    print(f"\n[OVLP-VS-GLOBAL] step={global_step}")
                                    print(f"  Overlap:  Jacc@10={ov_jacc_mean:.4f}")
                                    print(f"  Global:   Jacc@10={global_jacc_mean:.4f} scale_r={global_scale_r:.4f}")
                                    print(f"  Delta:    ov-global Jacc={ov_jacc_mean - global_jacc_mean:.4f}")
                                    if ov_jacc_mean > global_jacc_mean + 0.1:
                                        print(f"  WARNING: Overlap much better than global - possible degenerate solution!")

                        else:
                            ov_no_valid += 1

                        if global_step % overlap_debug_every == 0:
                            rank = dist.get_rank() if dist.is_initialized() else 0
                            print(f"  [rank={rank}] valid_batch_count={ov_loss_dict['valid_batch_count']}")
                            print(f"  [rank={rank}] L_score_2={L_score_2.item():.6f} (view2 scale anchor)")
                            if ov_loss_dict['debug_info']['I_sizes']:
                                print(f"  [rank={rank}] I_sizes: min={min(ov_loss_dict['debug_info']['I_sizes'])}, "
                                    f"mean={np.mean(ov_loss_dict['debug_info']['I_sizes']):.1f}")

                            # ============ [PAIR-SCALE] View1 vs View2 scale comparison ============
                            with torch.no_grad():
                                m1_f = mask_ov.unsqueeze(-1).float()
                                m2_f = mask_2.unsqueeze(-1).float()
                                
                                rms_pred_1 = (x0_pred_1.pow(2) * m1_f).sum() / m1_f.sum().clamp(min=1)
                                rms_pred_1 = rms_pred_1.sqrt().item()
                                rms_tgt_1 = (V_target[idx_keep].pow(2) * m1_f).sum() / m1_f.sum().clamp(min=1)
                                rms_tgt_1 = rms_tgt_1.sqrt().item()
                                scale_r_1 = rms_pred_1 / max(rms_tgt_1, 1e-8)
                                
                                rms_pred_2 = (x0_pred_2.pow(2) * m2_f).sum() / m2_f.sum().clamp(min=1)
                                rms_pred_2 = rms_pred_2.sqrt().item()
                                rms_tgt_2 = (V_target_2.pow(2) * m2_f).sum() / m2_f.sum().clamp(min=1)
                                rms_tgt_2 = rms_tgt_2.sqrt().item()
                                scale_r_2 = rms_pred_2 / max(rms_tgt_2, 1e-8)
                                
                                scale_gap = abs(np.log(scale_r_1 + 1e-8) - np.log(scale_r_2 + 1e-8))
                                
                                print(f"\n[PAIR-SCALE] step={global_step}")
                                print(f"  View1: rms_pred={rms_pred_1:.4f} rms_tgt={rms_tgt_1:.4f} scale_r={scale_r_1:.4f}")
                                print(f"  View2: rms_pred={rms_pred_2:.4f} rms_tgt={rms_tgt_2:.4f} scale_r={scale_r_2:.4f}")
                                print(f"  scale_gap (|log diff|)={scale_gap:.4f} (want < 0.1)")

                            # ============ [PAIR-LOSS] Loss component breakdown ============
                            L_score_1_val = L_score.item() if isinstance(L_score, torch.Tensor) else 0.0
                            L_score_2_val = L_score_2.item()
                            L_ov_total = L_ov_shape_batch.item() + L_ov_scale_batch.item() + L_ov_kl_batch.item()
                            w_score = WEIGHTS.get('score', 1.0)
                            w_ov = overlap_loss_weight_shape + overlap_loss_weight_scale + overlap_loss_weight_kl
                            
                            score_contrib = w_score * (L_score_1_val + L_score_2_val)
                            ov_contrib = w_ov * L_ov_total if L_ov_total > 0 else 0.0
                            ratio_ov_score = ov_contrib / max(score_contrib, 1e-8)
                            
                            print(f"\n[PAIR-LOSS] step={global_step}")
                            print(f"  L_score_1={L_score_1_val:.6f} L_score_2={L_score_2_val:.6f}")
                            print(f"  L_ov: shape={L_ov_shape_batch.item():.6f} scale={L_ov_scale_batch.item():.6f} kl={L_ov_kl_batch.item():.6f}")
                            print(f"  weighted_ratio (ov/score)={ratio_ov_score:.4f} (watch if >> 1 early)")

                            # ============ [BLOB-CHECK] Collapse detection ============
                            with torch.no_grad():
                                # View1 blob check
                                pred_1_valid = x0_pred_1[mask_ov.bool()]
                                if pred_1_valid.numel() > 0:
                                    var_per_dim_1 = pred_1_valid.var(dim=0).mean().item()
                                    # Sample pairwise distances (subsample if large)
                                    n_pts_1 = min(pred_1_valid.shape[0], 500)
                                    idx_sub = torch.randperm(pred_1_valid.shape[0])[:n_pts_1]
                                    d_pred_1 = torch.cdist(pred_1_valid[idx_sub], pred_1_valid[idx_sub])
                                    d_pred_1_med = d_pred_1[d_pred_1 > 0].median().item() if (d_pred_1 > 0).any() else 0.0
                                    
                                    tgt_1_valid = V_target[idx_keep][mask_ov.bool()]
                                    d_tgt_1 = torch.cdist(tgt_1_valid[idx_sub], tgt_1_valid[idx_sub])
                                    d_tgt_1_med = d_tgt_1[d_tgt_1 > 0].median().item() if (d_tgt_1 > 0).any() else 0.0
                                else:
                                    var_per_dim_1, d_pred_1_med, d_tgt_1_med = 0.0, 0.0, 0.0
                                
                                # View2 blob check
                                pred_2_valid = x0_pred_2[mask_2.bool()]
                                if pred_2_valid.numel() > 0:
                                    var_per_dim_2 = pred_2_valid.var(dim=0).mean().item()
                                    n_pts_2 = min(pred_2_valid.shape[0], 500)
                                    idx_sub_2 = torch.randperm(pred_2_valid.shape[0])[:n_pts_2]
                                    d_pred_2 = torch.cdist(pred_2_valid[idx_sub_2], pred_2_valid[idx_sub_2])
                                    d_pred_2_med = d_pred_2[d_pred_2 > 0].median().item() if (d_pred_2 > 0).any() else 0.0
                                    
                                    tgt_2_valid = V_target_2[mask_2.bool()]
                                    d_tgt_2 = torch.cdist(tgt_2_valid[idx_sub_2], tgt_2_valid[idx_sub_2])
                                    d_tgt_2_med = d_tgt_2[d_tgt_2 > 0].median().item() if (d_tgt_2 > 0).any() else 0.0
                                else:
                                    var_per_dim_2, d_pred_2_med, d_tgt_2_med = 0.0, 0.0, 0.0
                                
                                print(f"\n[BLOB-CHECK] step={global_step}")
                                print(f"  View1: var_per_dim={var_per_dim_1:.6f} d_pred_med={d_pred_1_med:.4f} d_tgt_med={d_tgt_1_med:.4f}")
                                print(f"  View2: var_per_dim={var_per_dim_2:.6f} d_pred_med={d_pred_2_med:.4f} d_tgt_med={d_tgt_2_med:.4f}")
                                if d_tgt_1_med > 0:
                                    print(f"  View1 dist_ratio={d_pred_1_med/d_tgt_1_med:.4f} (blob if << 1)")
                                if d_tgt_2_med > 0:
                                    print(f"  View2 dist_ratio={d_pred_2_med/d_tgt_2_med:.4f} (blob if << 1)")


                else:
                    ov_skipped_sigma += 1
                    if global_step % overlap_debug_every == 0:
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        print(f"[OVLP][rank={rank}] step={global_step} SKIPPED (sigma > {overlap_sigma_thresh})")
                        print(f"  [rank={rank}] sigma_pair: min={sigma_pair.min().item():.4f}, "
                            f"median={sigma_pair.median().item():.4f}, "
                            f"max={sigma_pair.max().item():.4f}")



            # Track overlap losses in epoch_losses
            epoch_losses['ov_shape'] = epoch_losses.get('ov_shape', 0.0) + L_ov_shape_batch.item()
            epoch_losses['ov_scale'] = epoch_losses.get('ov_scale', 0.0) + L_ov_scale_batch.item()
            epoch_losses['ov_kl'] = epoch_losses.get('ov_kl', 0.0) + L_ov_kl_batch.item()
            epoch_losses['score_2'] = epoch_losses.get('score_2', 0.0) + L_score_2_batch.item()  # View2 scale anchor
    
            
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

                # vgn_score = vhat_gn_from_eps(L_score, eps_pred, sigma_t)
                # Score gradient: use direct V_hat gradient for EDM, chain-rule for eps-prediction
                if use_edm:
                    # EDM: L_score is computed directly on x0_pred (= V_hat), so use vhat_gn
                    vgn_score = vhat_gn(L_score)
                else:
                    # Non-EDM: L_score is computed on eps_pred, need chain rule
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
                fabric.backward(L_total)
                
                # PATCH 6 (CORRECTED): Gradient health check - BEFORE clip/step
                nonfinite_detected = False
                if global_step < 20 or global_step % 50 == 0:  # Check often early
                    nonfinite_grads = 0
                    nonfinite_param_names = []
                    total_grad_norm_sq = 0.0
                    
                    # Check score_net
                    for name, p in score_net.named_parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():
                                nonfinite_grads += 1
                                nonfinite_param_names.append(f"score_net.{name}")
                                nonfinite_detected = True
                            else:
                                total_grad_norm_sq += p.grad.norm().item() ** 2
                    
                    # Check context_encoder
                    for name, p in context_encoder.named_parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():
                                nonfinite_grads += 1
                                nonfinite_param_names.append(f"context_encoder.{name}")
                                nonfinite_detected = True
                            else:
                                total_grad_norm_sq += p.grad.norm().item() ** 2
                    
                    total_grad_norm = math.sqrt(total_grad_norm_sq)
                    
                    if nonfinite_detected:
                        # Print which parameters have NaN
                        if fabric is None or fabric.is_global_zero:
                            print(f"\n[NAN-DEBUG] Step {global_step}: {nonfinite_grads} params with NaN grads")
                            print(f"  First 10 affected: {nonfinite_param_names[:10]}")
                            
                            # Also print loss values to see which might be causing it
                            print(f"  Loss values at this step:")
                            print(f"    L_score={L_score.item() if torch.isfinite(L_score) else 'NaN':.4f}")
                            print(f"    L_gram={L_gram.item() if torch.isfinite(L_gram) else 'NaN':.4f}")
                            print(f"    L_gram_scale={L_gram_scale.item() if torch.isfinite(L_gram_scale) else 'NaN':.4f}")
                            print(f"    L_edge={L_edge.item() if torch.isfinite(L_edge) else 'NaN':.4f}")
                            print(f"    L_knn_nca={L_knn_nca.item() if torch.isfinite(L_knn_nca) else 'NaN':.4f}")
                        
                        # Try to continue anyway with zeroed grads for affected params
                        for name, p in score_net.named_parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                p.grad.zero_()
                        for name, p in context_encoder.named_parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                p.grad.zero_()
                        
                        # DON'T skip - zero the bad grads and continue
                        # This allows training to proceed while we debug
                        print(f"  Zeroed NaN grads, continuing training...")

                
                torch.nn.utils.clip_grad_norm_(params, 10.0)
          
                # ========== PHASE 8: GRADIENT FLOW CHECKS ==========
                if global_step % 100 == 0 and fabric.is_global_zero:
                    print(f"\n[PHASE 8] Gradient Flow (step {global_step}):")
                    
                    # 8.1 Parameter grad norms by module
                    def module_grad_norm(module):
                        total = 0.0
                        for p in module.parameters():
                            if p.grad is not None:
                                total += p.grad.norm().item() ** 2
                        return math.sqrt(total)
                    
                    def module_param_norm(module):
                        total = 0.0
                        for p in module.parameters():
                            total += p.norm().item() ** 2
                        return math.sqrt(total)
                    
                    score_grad = module_grad_norm(score_net)
                    score_param = module_param_norm(score_net)
                    ctx_grad = module_grad_norm(context_encoder)
                    ctx_param = module_param_norm(context_encoder)
                    gen_grad = module_grad_norm(generator)
                    gen_param = module_param_norm(generator)
                    
                    print(f"  score_net: ||grad||={score_grad:.4e} ||param||={score_param:.4e} "
                          f"step_ratio={score_grad/score_param:.4e}")
                    print(f"  context_encoder: ||grad||={ctx_grad:.4e} ||param||={ctx_param:.4e} "
                          f"step_ratio={ctx_grad/ctx_param:.4e}")
                    print(f"  generator: ||grad||={gen_grad:.4e} ||param||={gen_param:.4e} "
                          f"step_ratio={gen_grad/gen_param:.4e}")
                    
                    if ctx_grad < score_grad * 0.01:
                        print(f"    ⚠️ WARNING: Context encoder grads are very small!")
                    
                    # 8.2 NaN/Inf sentinels
                    def check_finite(name, tensor):
                        if not torch.isfinite(tensor).all():
                            print(f"    ⚠️⚠️⚠️ NON-FINITE detected in {name}!")
                            return False
                        return True
                    
                    all_finite = True
                    all_finite &= check_finite("sigma", sigma_t)
                    all_finite &= check_finite("V_t", V_t)
                    all_finite &= check_finite("x0_pred", x0_pred)
                    all_finite &= check_finite("L_score", L_score)
                    
                    # Check a few representative grads
                    sample_params = [p for p in params if p.grad is not None][:5]
                    for i, p in enumerate(sample_params):
                        all_finite &= check_finite(f"grad_{i}", p.grad)
                    
                    # if not all_finite:
                    #     raise RuntimeError("NON-FINITE VALUES DETECTED - STOPPING")
                
                optimizer.step()
            else:
                # When not using Fabric, use manual GradScaler
                scaler.scale(L_total).backward()
                scaler.unscale_(optimizer)
                
                # PATCH 6 (CORRECTED): Gradient health check for non-Fabric path
                nonfinite_detected = False
                nonfinite_grads = 0
                for p in params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        nonfinite_grads += 1
                        nonfinite_detected = True
                
                if nonfinite_detected:
                    optimizer.zero_grad(set_to_none=True)
                    print(f"[PATCH6-NONFINITE] Step {global_step}: SKIPPING - nonfinite_grads={nonfinite_grads}")
                    scaler.update()  # Still update scaler
                    continue
                
                torch.nn.utils.clip_grad_norm_(params, 1000.0)
                scaler.step(optimizer)
                scaler.update()
            
            # ==================== EMA UPDATE ====================
            # IMPORTANT: Must happen AFTER optimizer.step() / scaler.step()
            ema_update(score_net_ema, score_net, ema_decay)
            ema_update(context_encoder_ema, context_encoder, ema_decay)
            # ====================================================

            # --- DEBUG 7: Gradient health check ---
            if global_step % 500 == 0 and (fabric is None or fabric.is_global_zero):
                print(f"\n[DEBUG-GRAD] Step {global_step} - Gradient health:")
                dead_grads = []
                exploding_grads = []
                for name, param in score_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm < 1e-8:
                            dead_grads.append((name, grad_norm))
                        elif grad_norm > 100:
                            exploding_grads.append((name, grad_norm))
                
                if dead_grads:
                    print(f"  ⚠️ DEAD GRADS ({len(dead_grads)}):")
                    for name, gn in dead_grads[:5]:  # Show first 5
                        print(f"    {name}: {gn:.2e}")
                
                if exploding_grads:
                    print(f"  ⚠️ EXPLODING GRADS ({len(exploding_grads)}):")
                    for name, gn in exploding_grads[:5]:
                        print(f"    {name}: {gn:.2e}")
                
                if not dead_grads and not exploding_grads:
                    print(f"  ✓ All gradients healthy")
                
                # Check generator gradients specifically
                gen_grad_norm = 0.0
                gen_param_count = 0
                for param in generator.parameters():
                    if param.grad is not None:
                        gen_grad_norm += param.grad.norm().item() ** 2
                        gen_param_count += 1
                gen_grad_norm = (gen_grad_norm ** 0.5) if gen_param_count > 0 else 0
                print(f"  Generator total grad norm: {gen_grad_norm:.4f}")

           
            # Log
            epoch_losses['total'] += L_total.item()
            epoch_losses['score'] += L_score.item()
            epoch_losses['st_dist'] += L_st_dist.item()
            epoch_losses['gram'] += L_gram.item()
            epoch_losses['gram_scale'] += L_gram_scale.item()
            epoch_losses['out_scale'] += L_out_scale.item()
            epoch_losses['knn_nca'] += L_knn_nca.item()
            epoch_losses['knn_scale'] += L_knn_scale.item()
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
            epoch_losses['gen_scale'] += L_gen_scale.item()
            epoch_losses['subspace'] += L_subspace.item()
            epoch_losses['gram_learn'] += L_gram_learn.item()  # FIX: Was missing!
            epoch_losses['ctx_edge'] += L_ctx_edge.item()  # NEW: context invariance
            # epoch_losses['ctx_replace'] += L_ctx_replace.item()



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
        # =====================================================================
        # FIXED-BATCH EVALUATION AT FIXED SIGMAS (once per epoch)
        # =====================================================================
        # Run fixed-batch eval at epochs 1,3,5,10 for early detection, then every 5 epochs
        early_eval_epochs = {0, 2, 4, 9}  # 0-indexed: epochs 1,3,5,10
        run_fixed_eval = (epoch % 5 == 0) or (epoch in early_eval_epochs)
        if (fabric is None or fabric.is_global_zero) and use_st and run_fixed_eval:
            fixed_eval_sigmas = [0.05, 0.15, 0.40, 0.70, 1.20, 2.40]

            
            fixed_batch_data = edm_debug_state.get('fixed_batch', None)
            
            if fixed_batch_data is not None:
                print(f"\n{'='*70}")
                print(f"[FIXED-BATCH EVAL] Epoch {epoch} - Evaluating at fixed sigmas")
                print(f"{'='*70}")
                
                # Set eval mode to disable dropout/stochastic layers
                was_training_score = score_net.training
                was_training_ctx = context_encoder.training
                score_net.eval()
                context_encoder.eval()
                
                with torch.no_grad():
                    Z_fixed = fixed_batch_data['Z_set'].to(device)
                    Z_fixed = apply_z_ln(Z_fixed, context_encoder)
                    mask_fixed = fixed_batch_data['mask'].to(device)
                    V_target_fixed = fixed_batch_data['V_target'].to(device)
                    G_target_fixed = fixed_batch_data['G_target'].to(device)
                    
                    B_fixed = Z_fixed.shape[0]
                    D_lat = score_net.D_latent if hasattr(score_net, 'D_latent') else \
                            score_net.module.D_latent if hasattr(score_net, 'module') else 16
                    
                    H_fixed = context_encoder(Z_fixed, mask_fixed)
                    
                    print(f"  {'σ':>8} | {'scale_r':>8} | {'trace_r':>8} | {'out/tgt':>8} | {'Jacc@10':>8} | {'SC':>6}")
                    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")

                    
                    # Use fixed seed for reproducible noise across epochs
                    rng_state = torch.get_rng_state()
                    
                    for sigma_val in fixed_eval_sigmas:
                        # Deterministic noise for this sigma
                        torch.manual_seed(42 + int(sigma_val * 1000))
                        
                        sigma_fixed = torch.full((B_fixed,), sigma_val, device=device)
                        sigma_fixed_3d = sigma_fixed.view(-1, 1, 1)
                        
                        eps_fixed = torch.randn_like(V_target_fixed)
                        V_t_fixed = V_target_fixed + sigma_fixed_3d * eps_fixed
                        V_t_fixed = V_t_fixed * mask_fixed.unsqueeze(-1).float()
                        
                        for sc_mode in ['no_sc', 'with_sc']:
                            if sc_mode == 'with_sc' and score_net.self_conditioning:
                                x0_pred_0_eval = score_net.forward_edm(
                                    V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
                                    sigma_data, self_cond=None
                                )
                                if isinstance(x0_pred_0_eval, tuple):
                                    x0_pred_0_eval = x0_pred_0_eval[0]
                                
                                x0_pred_eval = score_net.forward_edm(
                                    V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
                                    sigma_data, self_cond=x0_pred_0_eval
                                )
                            else:
                                x0_pred_eval = score_net.forward_edm(
                                    V_t_fixed, sigma_fixed, H_fixed, mask_fixed, 
                                    sigma_data, self_cond=None
                                )
                            
                            if isinstance(x0_pred_eval, tuple):
                                x0_pred_eval = x0_pred_eval[0]
                            
                            mask_f_eval = mask_fixed.unsqueeze(-1).float()
                            valid_count_eval = mask_f_eval.sum()
                            
                            # Scale ratio (centered)
                            V_pred_c, _ = uet.center_only(x0_pred_eval, mask_fixed)
                            V_tgt_c, _ = uet.center_only(V_target_fixed, mask_fixed)
                            rms_pred = (V_pred_c.pow(2) * mask_f_eval).sum().div(valid_count_eval).sqrt()
                            rms_tgt = (V_tgt_c.pow(2) * mask_f_eval).sum().div(valid_count_eval).sqrt()
                            scale_ratio = (rms_pred / rms_tgt.clamp(min=1e-8)).item()
                            
                            # Trace ratio
                            G_pred = V_pred_c @ V_pred_c.transpose(1, 2)
                            trace_pred = torch.diagonal(G_pred, dim1=-2, dim2=-1).sum(dim=-1)
                            trace_tgt = torch.diagonal(G_target_fixed, dim1=-2, dim2=-1).sum(dim=-1)
                            trace_ratio = (trace_pred / trace_tgt.clamp(min=1e-8)).mean().item()
                            
                            # Jaccard@10 (pure torch, no numpy)
                            jaccard_sum = 0.0
                            jaccard_count = 0
                            for b in range(min(4, B_fixed)):
                                m_b = mask_fixed[b].bool()
                                n_valid = int(m_b.sum().item())
                                if n_valid < 15:
                                    continue
                                
                                pred_b = x0_pred_eval[b, m_b]
                                tgt_b = V_target_fixed[b, m_b]
                                
                                D_pred_b = torch.cdist(pred_b, pred_b)
                                D_tgt_b = torch.cdist(tgt_b, tgt_b)
                                
                                k_j = min(10, n_valid - 1)
                                _, knn_pred = D_pred_b.topk(k_j + 1, largest=False)
                                _, knn_tgt = D_tgt_b.topk(k_j + 1, largest=False)
                                
                                knn_pred = knn_pred[:, 1:]  # Exclude self
                                knn_tgt = knn_tgt[:, 1:]
                                
                                for i in range(n_valid):
                                    set_pred = set(knn_pred[i].tolist())
                                    set_tgt = set(knn_tgt[i].tolist())
                                    inter = len(set_pred & set_tgt)
                                    union = len(set_pred | set_tgt)
                                    if union > 0:
                                        jaccard_sum += inter / union
                                        jaccard_count += 1
                            
                            jaccard_mean = jaccard_sum / max(jaccard_count, 1)


                            # Compute out/tgt (learned branch scale ratio)
                            # Compute out/tgt (learned branch scale ratio)
                            # Recompute EDM preconditioning for this sigma value
                            c_skip_fb, c_out_fb, _, _ = uet.edm_precond(sigma_fixed, sigma_data)
                            # c_skip_fb: (B, 1, 1)

                            # V_c (centered noisy input) - same as forward_edm does
                            V_c_fb, _ = uet.center_only(V_t_fixed, mask_fixed)
                            V_c_fb = V_c_fb * mask_f_eval

                            # Learned contribution: V_out = V_pred_centered - c_skip * V_c
                            # V_pred_c is already computed above
                            V_out_fb = V_pred_c - c_skip_fb * V_c_fb
                            V_out_fb = V_out_fb * mask_f_eval

                            # RMS of learned branch and target
                            rms_out_fb = (V_out_fb.pow(2) * mask_f_eval).sum().div(valid_count_eval).sqrt()
                            # rms_tgt is already computed above as rms_tgt from V_tgt_c
                            out_tgt_ratio = (rms_out_fb / rms_tgt.clamp(min=1e-8)).item()

                            sc_label = "SC" if sc_mode == 'with_sc' else "NO"
                            print(f"  {sigma_val:8.3f} | {scale_ratio:8.3f} | {trace_ratio:8.3f} | {out_tgt_ratio:8.3f} | {jaccard_mean:8.3f} | {sc_label:>6}")


                    
                    # Restore RNG state
                    torch.set_rng_state(rng_state)
                
                # Restore training mode
                if was_training_score:
                    score_net.train()
                if was_training_ctx:
                    context_encoder.train()
                
                print(f"{'='*70}\n")

                # =====================================================================
                # A/B TESTS (every 10 epochs to avoid too much output)
                # =====================================================================
                if (epoch % 5 == 0) or (epoch == 0):
                    run_ab_tests_fixed_batch(
                        score_net=score_net,
                        context_encoder=context_encoder,
                        fixed_batch_data=fixed_batch_data,
                        sigma_data=sigma_data,
                        fixed_eval_sigmas=fixed_eval_sigmas,
                        device=device,
                        epoch=epoch,
                    )
        # =====================================================================
        # PROBE EVALUATION (ChatGPT Hypothesis 4 - Multi-Sigma Discriminative Metrics)
        # =====================================================================
        if (fabric is None or fabric.is_global_zero) and use_st and probe_state.get('batch_captured', False):
            print(f"\n{'='*70}")
            print(f"[PROBE EVAL] Epoch {epoch+1} - Multi-Sigma Discriminative Metrics")
            print(f"{'='*70}")
            
            probe_results = run_probe_eval_multi_sigma(
                probe_state=probe_state,
                score_net=score_net,
                context_encoder=context_encoder,
                sigma_data=sigma_data,
                device=device,
                sigma_list=PROBE_SIGMAS,  # [0.2, 0.8, 1.5, 2.0, 2.5, 3.0]
                seed=PROBE_SEED,
                global_step=global_step,
                epoch=epoch,
                fabric=fabric,
            )
            
            print_probe_results_multi_sigma(probe_results, global_step, epoch)
            
            # Store key metrics in history for tracking
            if 'probe_knn10_s02' not in history['epoch_avg']:
                # Initialize tracking for each sigma
                for sig in PROBE_SIGMAS:
                    sig_tag = f"s{str(sig).replace('.', '')}"
                    history['epoch_avg'][f'probe_knn10_{sig_tag}'] = []
                    history['epoch_avg'][f'probe_nearmiss_{sig_tag}'] = []
            
            for sig in PROBE_SIGMAS:
                if sig in probe_results:
                    sig_tag = f"s{str(sig).replace('.', '')}"
                    history['epoch_avg'][f'probe_knn10_{sig_tag}'].append(
                        probe_results[sig].get('knn10_overlap', 0))
                    history['epoch_avg'][f'probe_nearmiss_{sig_tag}'].append(
                        probe_results[sig].get('nearmiss_ratio_p50', 0))
            
            print(f"{'='*70}\n")

        # =====================================================================
        # OPEN-LOOP PROBE SAMPLING (Check 1 - Trajectory Consistency)
        # =====================================================================
        if (fabric is None or fabric.is_global_zero) and use_st and probe_state.get('batch_captured', False):
            print(f"\n{'='*70}")
            print(f"[PROBE SAMPLE] Epoch {epoch+1} - Open-Loop Trajectory Test")
            print(f"{'='*70}")
            
            sample_results = run_probe_open_loop_sample(
                probe_state=probe_state,
                score_net=score_net,
                context_encoder=context_encoder,
                sigma_data=sigma_data,
                device=device,
                n_steps=PROBE_SAMPLE_STEPS,
                sigma_max=PROBE_SAMPLE_SIGMA_MAX,
                sigma_min=PROBE_SAMPLE_SIGMA_MIN,
                rho=PROBE_SAMPLE_RHO,
                seed=PROBE_SEED,
                global_step=global_step,
                epoch=epoch,
                trace_sigmas=PROBE_SAMPLE_TRACE_SIGMAS,
                fabric=fabric,
            )
            
            print_probe_sample_results(sample_results, global_step, epoch)
            
            # Store in history
            if 'probe_sample_knn10' not in history['epoch_avg']:
                history['epoch_avg']['probe_sample_knn10'] = []
                history['epoch_avg']['probe_sample_nearmiss'] = []
            
            history['epoch_avg']['probe_sample_knn10'].append(
                sample_results.get('knn10_overlap', 0))
            history['epoch_avg']['probe_sample_nearmiss'].append(
                sample_results.get('nearmiss_ratio_p50', 0))
            
            print(f"{'='*70}\n")


        # ========== END OF EPOCH: PHASE 6 + PHASE 7 REPORTING ==========
        if fabric is None or fabric.is_global_zero:
            # ========== PHASE 6: PER-SIGMA BIN REPORT ==========
            if edm_debug_state['sigma_bin_edges'] is not None:
                print(f"\n" + "="*70)
                print(f"[PHASE 6] Per-Sigma Bin Report (Epoch {epoch+1})")
                print("="*70)
                
                bin_edges = edm_debug_state['sigma_bin_edges']
                sum_err2 = edm_debug_state['sigma_bin_sum_err2']
                sum_w = edm_debug_state['sigma_bin_sum_w']
                sum_werr2 = edm_debug_state['sigma_bin_sum_werr2']
                counts = edm_debug_state['sigma_bin_count']
                
                print(f"{'Bin':>3} | {'Sigma Range':>20} | {'Count':>6} | {'Avg err2':>10} | {'Avg w':>10} | {'Avg w*err2':>12}")
                print("-"*70)
                
                for b_id in range(len(counts)):
                    if counts[b_id] > 0:
                        s_low = bin_edges[b_id].exp().item()
                        s_high = bin_edges[b_id+1].exp().item()
                        avg_err2 = sum_err2[b_id] / counts[b_id]
                        avg_w = sum_w[b_id] / counts[b_id]
                        avg_werr2 = sum_werr2[b_id] / counts[b_id]
                        
                        print(f"{b_id:3d} | [{s_low:8.4f}, {s_high:8.4f}] | {int(counts[b_id]):6d} | "
                              f"{avg_err2:10.6f} | {avg_w:10.4f} | {avg_werr2:12.6f}")
                
                print("="*70 + "\n")
                
                # Reset for next epoch
                edm_debug_state['sigma_bin_sum_err2'].zero_()
                edm_debug_state['sigma_bin_sum_w'].zero_()
                edm_debug_state['sigma_bin_sum_werr2'].zero_()
                edm_debug_state['sigma_bin_count'].zero_()
            
            # ========== PHASE 7: LEARNING CHECKS ==========
            if (epoch + 1) % 2 == 0 and edm_debug_state['fixed_batch'] is not None:
                print(f"\n" + "="*70)
                print(f"[PHASE 7] Learning Checks (Epoch {epoch+1})")
                print("="*70)
                
                with torch.no_grad():
                    # Load fixed batch
                    fb = edm_debug_state['fixed_batch']
                    Z_fb = fb['Z_set'].to(device)
                    mask_fb = fb['mask'].to(device)
                    V_target_fb = fb['V_target'].to(device)
                    
                    H_fb = context_encoder(Z_fb, mask_fb)
                    
                    # Test at 3 sigma levels (p20, p50, p80 of training distribution)
                    sigma_test_levels = [
                        sigma_refine_max * 0.2,  # Low noise
                        sigma_refine_max * 0.5,  # Medium
                        sigma_refine_max * 0.8,  # High
                    ]
                    
                    print("\n[PHASE 7.1] One-step Denoise Improvement:")
                    eps_seed = torch.randn_like(V_target_fb)
                    
                    for sigma_test in sigma_test_levels:
                        V_t_test = V_target_fb + sigma_test * eps_seed
                        V_t_test = V_t_test * mask_fb.unsqueeze(-1).float()
                        
                        sigma_batch = torch.full((V_t_test.shape[0],), sigma_test, device=device)
                        x0_pred_test = score_net.forward_edm(V_t_test, sigma_batch, H_fb, mask_fb, sigma_data, self_cond=None)
                        if isinstance(x0_pred_test, tuple):
                            x0_pred_test = x0_pred_test[0]
                        
                        # MSE: model vs baseline
                        err_model = (x0_pred_test - V_target_fb).pow(2).sum(dim=-1)  # (B, N)
                        err_baseline = (V_t_test - V_target_fb).pow(2).sum(dim=-1)
                        
                        mask_f = mask_fb.float()
                        mse_model = (err_model * mask_f).sum() / mask_f.sum()
                        mse_baseline = (err_baseline * mask_f).sum() / mask_f.sum()
                        
                        ratio = mse_model / (mse_baseline + 1e-8)
                        
                        print(f"  sigma={sigma_test:.4f}: mse_model={mse_model:.6f} "
                              f"mse_baseline={mse_baseline:.6f} ratio={ratio:.4f}")
                        
                        if ratio < 1.0:
                            print(f"    ✓ Model beats baseline")
                        else:
                            print(f"    ⚠️ Model worse than baseline!")
                    
                    # Compare to initial baseline from Phase 1
                    if edm_debug_state['baseline_mse'] is not None:
                        print(f"\n  Reference baseline MSE (epoch 0): {edm_debug_state['baseline_mse']:.6f}")
                
                print("="*70 + "\n")

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

            avg_ctx_replace = ctx_replace_sum / max(ctx_apply_count, 1)
            avg_ctx_snr = ctx_snr_sum / max(ctx_apply_count, 1)
            ctx_apply_rate = ctx_apply_count / max(n_batches, 1)

            if ctx_replace_variant == 'hard':
                avg_ctx_hard_sim = ctx_hard_sim_sum / max(ctx_apply_count, 1)
                avg_ctx_perm_fixed = 0.0
            elif ctx_replace_variant == 'permute':
                avg_ctx_perm_fixed = ctx_perm_fixed_sum / max(ctx_apply_count, 1)
                avg_ctx_hard_sim = 0.0
            else:
                avg_ctx_hard_sim = 0.0
                avg_ctx_perm_fixed = 0.0

            history['epoch_avg']['ctx_replace'].append(avg_ctx_replace)
            history['epoch_avg']['ctx_snr_med'].append(avg_ctx_snr)
            history['epoch_avg']['ctx_apply_rate'].append(ctx_apply_rate)
            history['epoch_avg']['ctx_hard_sim'].append(avg_ctx_hard_sim)
            history['epoch_avg']['ctx_perm_fixed'].append(avg_ctx_perm_fixed)

            # ========== OVERLAP CONSISTENCY LOSS EPOCH SUMMARY ==========
            if train_pair_overlap and ov_apply_count > 0:
                avg_ov_shape = ov_loss_sum['shape'] / ov_apply_count
                avg_ov_scale = ov_loss_sum['scale'] / ov_apply_count
                avg_ov_kl = ov_loss_sum['kl'] / ov_apply_count
                avg_ov_total = ov_loss_sum['total'] / ov_apply_count
                ov_apply_rate = ov_apply_count / max(n_batches, 1)

                # Track in history (safe init for partially populated dicts)
                history['epoch_avg'].setdefault('ov_shape', [])
                history['epoch_avg'].setdefault('ov_scale', [])
                history['epoch_avg'].setdefault('ov_kl', [])
                history['epoch_avg'].setdefault('ov_total', [])
                history['epoch_avg'].setdefault('ov_jaccard_k10', [])
                history['epoch_avg'].setdefault('ov_I_size_mean', [])


                history['epoch_avg']['ov_shape'].append(avg_ov_shape)
                history['epoch_avg']['ov_scale'].append(avg_ov_scale)
                history['epoch_avg']['ov_kl'].append(avg_ov_kl)
                history['epoch_avg']['ov_total'].append(avg_ov_total)

                avg_I_size = np.mean(ov_I_sizes) if ov_I_sizes else 0.0
                avg_jaccard = np.mean(ov_jaccard_k10) if ov_jaccard_k10 else 0.0
                history['epoch_avg']['ov_I_size_mean'].append(avg_I_size)
                history['epoch_avg']['ov_jaccard_k10'].append(avg_jaccard)

                print(f"\n[OVLP-EPOCH] Epoch {epoch+1} Overlap Loss Summary:")
                print(f"  L_ov_shape={avg_ov_shape:.6f}, L_ov_scale={avg_ov_scale:.6f}, "
                      f"L_ov_kl={avg_ov_kl:.6f}, L_ov_total={avg_ov_total:.6f}")
                print(f"  apply_rate={ov_apply_rate:.2%}, skipped_sigma={ov_skipped_sigma}")
                print(f"  avg_I_size={avg_I_size:.1f}, Jaccard@10={avg_jaccard:.4f}")
        
            # Print detailed losses every 5 epochs, simple summary otherwise
            if (epoch + 1) % 5 == 0:
                avg_gram_scale = epoch_losses['gram_scale'] / max(n_batches, 1)
                avg_out_scale = epoch_losses['out_scale'] / max(n_batches, 1)
                avg_gram_learn = epoch_losses['gram_learn'] / max(n_batches, 1)
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
                avg_gen_scale = epoch_losses['gen_scale'] / max(n_batches, 1)
                avg_subspace = epoch_losses['subspace'] / max(n_batches, 1)
                avg_ctx_edge = epoch_losses['ctx_edge'] / max(n_batches, 1)


                print(f"[Epoch {epoch+1}] DETAILED LOSSES:")
                print(f"  total={avg_total:.4f} | score={avg_score:.4f} | gram={avg_gram:.4f} | gram_scale={avg_gram_scale:.4f} | out_scale={avg_out_scale:.4f} | gram_learn={avg_gram_learn:.4f}") 
                print(f"  heat={avg_heat:.4f} | sw_st={avg_sw_st:.4f} | sw_sc={avg_sw_sc:.4f} | knn_nca={avg_knn_nca:.4f}")
                print(f"  overlap={avg_overlap:.4f} | ordinal_sc={avg_ordinal_sc:.4f} | st_dist={avg_st_dist:.4f}")
                print(f"  edm_tail={avg_edm_tail:.4f} | gen_align={avg_gen_align:.4f}")
                print(f"  dim={avg_dim:.4f} | triangle={avg_triangle:.4f} | radial={avg_radial:.4f}")
                print(f"  edge={avg_edge:.4f} | topo={avg_topo:.4f} | shape_spec={avg_shape_spec:.4f}")
                print(f"  gen_scale={avg_gen_scale:.4f} subspace={avg_subspace:.4f} ctx_edge={avg_ctx_edge:.4f}")



            else:
                print(f"[Epoch {epoch+1}] Avg Losses: score={avg_score:.4f}, gram={avg_gram:.4f}, total={avg_total:.4f}")
            
            if enable_early_stop and (epoch + 1) >= early_stop_min_epochs:
                # Use total weighted loss as validation metric
                # This includes all geometry regularizers (dim, triangle, radial)
                val_metric = avg_total

                
                # Check for improvement
                if val_metric < early_stop_best:
                    rel_improv = (early_stop_best - val_metric) / max(early_stop_best, 1e-8)
                    
                    if rel_improv > early_stop_threshold:
                        # Significant improvement
                        early_stop_best = val_metric
                        early_stop_no_improve = 0
                    else:
                        # Minor improvement, doesn't count
                        early_stop_no_improve += 1
                else:
                    # No improvement
                    early_stop_no_improve += 1
                
                # Check if we should stop
                if early_stop_no_improve >= early_stop_patience:
                    should_stop = True
                    early_stopped = True
                    early_stop_epoch = epoch + 1

            elif enable_early_stop and (epoch + 1) < early_stop_min_epochs:
                # Just track best, don't stop yet
                val_metric = avg_total
                if val_metric < early_stop_best:
                    early_stop_best = val_metric
        

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
            if key in ('gram', 'gram_scale', 'heat', 'sw_st', 'cone', 'edm_tail', 'gen_align', 'dim', 'triangle', 'radial', 'st_dist', 'edge', 'topo', 'shape_spec', 'subspace', 'gen_scale', 'ctx_edge'):
                return cnt_st
            if key in ('sw_sc', 'ordinal_sc'):
                return cnt_sc
            if key in ('overlap', 'ov_shape', 'ov_scale', 'ov_kl'):
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
            WEIGHTS['topo'] * epoch_losses['topo'] +
            WEIGHTS.get('gen_scale', 0) * epoch_losses['gen_scale'] +
            WEIGHTS.get('subspace', 0) * epoch_losses['subspace'] +
            WEIGHTS.get('ctx_edge', 0) * epoch_losses['ctx_edge']
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
        if fabric is not None:
            fabric.barrier()


        # --- save checkpoints only on rank-0 ---
        # --- save checkpoints only on rank-0 ---
        if (epoch + 1) % 100 == 0:
            if fabric is None or fabric.is_global_zero:
                ckpt = {
                    'epoch': epoch,
                    'context_encoder': context_encoder.state_dict(),
                    'score_net': score_net.state_dict(),
                    'generator': generator.state_dict(),
                    'context_encoder_ema': context_encoder_ema.state_dict(),
                    'score_net_ema': score_net_ema.state_dict(),
                    'ema_decay': ema_decay,
                    'optimizer': optimizer.state_dict(),
                    'history': history,
                    'sigma_data': sigma_data,
                    'sigma_min': sigma_min,
                    'sigma_max': sigma_max,
                }
                if encoder is not None:
                    encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
                    ckpt['encoder'] = encoder_to_save.state_dict()
                torch.save(ckpt, os.path.join(outf, f'ckpt_epoch_{epoch+1}.pt'))
    
    # Save final checkpoint after training loop
    if fabric is None or fabric.is_global_zero:
        ckpt_final = {
            'epoch': epoch,
            'context_encoder': context_encoder.state_dict(),
            'score_net': score_net.state_dict(),
            'generator': generator.state_dict(),
            'context_encoder_ema': context_encoder_ema.state_dict(),
            'score_net_ema': score_net_ema.state_dict(),
            'ema_decay': ema_decay,
            'optimizer': optimizer.state_dict(),
            'history': history,
            'sigma_data': sigma_data,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
        }
        if encoder is not None:
            encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
            ckpt_final['encoder'] = encoder_to_save.state_dict()
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

    history['sigma_data'] = sigma_data
    history['sigma_min'] = sigma_min
    history['sigma_max'] = sigma_max

    # Return EMA state dicts so GEMSModel can use them
    history['score_net_ema_state'] = score_net_ema.state_dict()
    history['context_encoder_ema_state'] = context_encoder_ema.state_dict()
    history['ema_decay'] = ema_decay

    # return history if _is_rank0() else None
    return history



# ==============================================================================
# STAGE D: SC INFERENCE (PATCH-BASED GLOBAL ALIGNMENT)
# ==============================================================================

# def sample_sc_edm_patchwise(
#     sc_gene_expr: torch.Tensor,
#     encoder: "SharedEncoder",
#     context_encoder: "SetEncoderContext",
#     score_net: "DiffusionScoreNet",
#     generator: "GeometryGenerator",
#     sigma_data: float,
#     target_st_p95: Optional[float] = None,
#     n_timesteps_sample: int = 160,
#     sigma_min: float = 0.01,
#     sigma_max: float = 5.0,
#     guidance_scale: float = 8.0,
#     eta: float = 0.0,
#     device: str = "cuda",
#     # patch / coverage knobs
#     patch_size: int = 384,
#     coverage_per_cell: float = 4.0,
#     # global alignment knobs
#     n_align_iters: int = 10,
#     # misc
#     return_coords: bool = True,
#     DEBUG_FLAG: bool = True,
#     DEBUG_EVERY: int = 10,
#     fixed_patch_graph: Optional[dict] = None,
#     coral_params: Optional[Dict] = None,
#     # --- DEBUG KNN ARGS ---
#     gt_coords: Optional[torch.Tensor] = None,
#     debug_knn: bool = False,
#     debug_max_patches: int = 20,
#     debug_k_list: Tuple[int, int] = (10, 20),
#     debug_global_subset: int = 4096,
#     debug_gap_k: int = 10,
#     # --- TWO-PASS MODE ---
#     two_pass: bool = False,
#     _pass_number: int = 1,  # Internal: which pass are we on (1 or 2)
#     _coords_from_pass1: Optional[torch.Tensor] = None,  # Internal: coords from pass 1 for rebuilding patches
# ) -> Dict[str, torch.Tensor]:
#     """
#     Stage D: Patch-based SC inference via global alignment.

#     1) Encode all SC cells -> Z.
#     2) Build overlapping SC patches using k-NN in Z-space.
#     3) For each patch:
#        - Run diffusion sampling on the patch (all points free),
#        - Canonicalize sample to match training invariances.
#     4) Align all patch samples into a single global embedding X via
#        alternating Procrustes (similarity transforms per patch).
#     5) Compute global EDM from X, optionally rescale to match ST scale.
#     6) Optionally run 2D MDS for visualization.
#     """

#     import numpy as np
#     import torch.nn.functional as F
#     from tqdm import tqdm
#     import utils_et as uet
#     from typing import List, Dict

#     import random


#     # ===================================================================
#     # DEBUG KNN HELPERS (only used if debug_knn=True and gt_coords provided)
#     # ===================================================================
#     def _knn_indices_dists(coords, k):
#         """Compute kNN indices and distances for a set of coords (n, d)."""
#         D = torch.cdist(coords, coords)
#         D.fill_diagonal_(float('inf'))
#         dists, idx = D.topk(k, largest=False, dim=1)
#         return idx, dists  # (n, k), (n, k)

#     def _knn_overlap_score(idx_a, idx_b):
#         """Compute per-point kNN overlap fraction between two index tensors."""
#         # idx_a, idx_b: (n, k)
#         n, k = idx_a.shape
#         overlaps = []
#         for i in range(n):
#             set_a = set(idx_a[i].tolist())
#             set_b = set(idx_b[i].tolist())
#             overlap = len(set_a & set_b) / k
#             overlaps.append(overlap)
#         return torch.tensor(overlaps)  # (n,)

#     def _nn_gap_stats(coords, k_gap):
#         """Compute NN gap statistics: gap = d[k+1] - d[k], gap_ratio = gap / d[k]."""
#         D = torch.cdist(coords, coords)
#         D.fill_diagonal_(float('inf'))
#         dists, _ = D.topk(k_gap + 1, largest=False, dim=1)  # (n, k+1)
#         d_k = dists[:, k_gap - 1]      # distance to k-th neighbor (0-indexed: k-1)
#         d_k1 = dists[:, k_gap]         # distance to (k+1)-th neighbor
#         gap = d_k1 - d_k
#         gap_ratio = gap / d_k.clamp(min=1e-8)
#         return gap, gap_ratio  # (n,), (n,)

#     def _local_edge_spearman(pred_coords, gt_coords, gt_knn_idx):
#         """
#         Compute Spearman correlation on GT kNN edges only.
#         pred_coords: (n, D_pred)
#         gt_coords: (n, D_gt)  
#         gt_knn_idx: (n, k) - GT kNN indices
#         Returns: Spearman correlation (scalar)
#         """
#         from scipy.stats import spearmanr
        
#         n, k = gt_knn_idx.shape
#         src = np.repeat(np.arange(n), k)
#         dst = gt_knn_idx.reshape(-1)
#         keep = (dst >= 0) & (dst < n) & (dst != src)
#         src, dst = src[keep], dst[keep]
        
#         if len(src) < 10:
#             return float('nan')
        
#         # Convert to numpy if needed
#         if torch.is_tensor(pred_coords):
#             pred_np = pred_coords.cpu().numpy()
#         else:
#             pred_np = pred_coords
#         if torch.is_tensor(gt_coords):
#             gt_np = gt_coords.cpu().numpy()
#         else:
#             gt_np = gt_coords
        
#         d_gt = np.linalg.norm(gt_np[src] - gt_np[dst], axis=1)
#         d_pr = np.linalg.norm(pred_np[src] - pred_np[dst], axis=1)
        
#         if np.std(d_gt) < 1e-12 or np.std(d_pr) < 1e-12:
#             return float('nan')
        
#         return spearmanr(d_gt, d_pr).correlation

#     def build_patches_from_coords(coords, n_cells, patch_size, coverage_per_cell, min_overlap=20):
#         """
#         Build patch graph using kNN in coordinate space (not expression space).
        
#         Args:
#             coords: (N, D) tensor of coordinates
#             n_cells: total number of cells
#             patch_size: target patch size
#             coverage_per_cell: target coverage
#             min_overlap: minimum overlap for connectivity
        
#         Returns:
#             patch_indices: List[LongTensor] - cell indices for each patch
#             memberships: List[List[int]] - patch indices for each cell
#         """
#         import math
        
#         coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
#         N = coords_np.shape[0]
        
#         # Build kNN index on coords
#         from sklearn.neighbors import NearestNeighbors
#         k_nbrs = min(patch_size, N - 1)
#         nbrs = NearestNeighbors(n_neighbors=k_nbrs, algorithm='ball_tree').fit(coords_np)
#         _, nbr_idx = nbrs.kneighbors(coords_np)  # (N, k_nbrs)
#         nbr_idx = torch.from_numpy(nbr_idx).long()
        
#         # Estimate number of patches needed
#         n_patches_est = int(math.ceil((coverage_per_cell * N) / patch_size))
        
#         # Sample random centers
#         centers = torch.randperm(N)[:n_patches_est].tolist()
        
#         patch_indices_list: List[torch.Tensor] = []
#         memberships_list: List[List[int]] = [[] for _ in range(N)]
        
#         # First pass: build patches around centers
#         for k, c in enumerate(centers):
#             S_k = nbr_idx[c, :patch_size]
#             S_k = torch.unique(S_k, sorted=False)
#             patch_indices_list.append(S_k)
#             for idx in S_k.tolist():
#                 memberships_list[idx].append(k)
        
#         # Ensure every cell appears in at least one patch
#         for i in range(N):
#             if len(memberships_list[i]) == 0:
#                 k = len(patch_indices_list)
#                 S_k = nbr_idx[i, :patch_size]
#                 S_k = torch.unique(S_k, sorted=False)
#                 patch_indices_list.append(S_k)
#                 memberships_list[i].append(k)
#                 for idx in S_k.tolist():
#                     if k not in memberships_list[idx]:
#                         memberships_list[idx].append(k)
        
#         # Check connectivity and add bridge patches if needed
#         K = len(patch_indices_list)
#         if K > 1:
#             # Build overlap graph
#             from collections import defaultdict
#             import networkx as nx
            
#             G = nx.Graph()
#             G.add_nodes_from(range(K))
            
#             for i in range(K):
#                 S_i = set(patch_indices_list[i].tolist())
#                 for j in range(i + 1, K):
#                     S_j = set(patch_indices_list[j].tolist())
#                     overlap = len(S_i & S_j)
#                     if overlap >= min_overlap:
#                         G.add_edge(i, j)
            
#             components = list(nx.connected_components(G))
            
#             # Add bridge patches if fragmented
#             while len(components) > 1:
#                 # Find boundary cells between components
#                 comp1 = list(components[0])
#                 comp2 = list(components[1])
                
#                 # Get all cells in each component
#                 cells1 = set()
#                 for p_idx in comp1:
#                     cells1.update(patch_indices_list[p_idx].tolist())
#                 cells2 = set()
#                 for p_idx in comp2:
#                     cells2.update(patch_indices_list[p_idx].tolist())
                
#                 # Find closest pair of cells between components
#                 cells1_list = list(cells1)
#                 cells2_list = list(cells2)
                
#                 coords1 = coords_np[cells1_list]
#                 coords2 = coords_np[cells2_list]
                
#                 from scipy.spatial.distance import cdist
#                 D_cross = cdist(coords1, coords2)
#                 min_idx = np.unravel_index(D_cross.argmin(), D_cross.shape)
#                 bridge_cell1 = cells1_list[min_idx[0]]
#                 bridge_cell2 = cells2_list[min_idx[1]]
                
#                 # Create bridge patch centered between them
#                 bridge_center = (coords_np[bridge_cell1] + coords_np[bridge_cell2]) / 2
#                 dists = np.linalg.norm(coords_np - bridge_center, axis=1)
#                 bridge_cells = np.argsort(dists)[:patch_size]
                
#                 k_new = len(patch_indices_list)
#                 S_bridge = torch.from_numpy(bridge_cells).long()
#                 patch_indices_list.append(S_bridge)
#                 for idx in S_bridge.tolist():
#                     if k_new not in memberships_list[idx]:
#                         memberships_list[idx].append(k_new)
                
#                 # Rebuild graph and check again
#                 K = len(patch_indices_list)
#                 G = nx.Graph()
#                 G.add_nodes_from(range(K))
                
#                 for i in range(K):
#                     S_i = set(patch_indices_list[i].tolist())
#                     for j in range(i + 1, K):
#                         S_j = set(patch_indices_list[j].tolist())
#                         overlap = len(S_i & S_j)
#                         if overlap >= min_overlap:
#                             G.add_edge(i, j)
                
#                 components = list(nx.connected_components(G))
                
#                 # Safety: don't infinite loop
#                 if len(patch_indices_list) > n_patches_est * 3:
#                     print(f"  [WARNING] Could not fully connect patch graph after adding bridges")
#                     break
        
#         return patch_indices_list, memberships_list


#     print(f"[PATCHWISE] FINAL IDK WHAT IS GOING ON Running on device={device}, starting inference...", flush=True)

#     encoder.eval()
#     context_encoder.eval()
#     score_net.eval()

#     n_sc = sc_gene_expr.shape[0]
#     D_latent = score_net.D_latent
#     patch_size = int(min(patch_size, n_sc))

#     if DEBUG_FLAG:
#         print("\n" + "=" * 72)
#         pass_label = f"[PASS{_pass_number}]" if two_pass else ""
#         print(f"STAGE D — PATCH-BASED GLOBAL SC INFERENCE {pass_label}")
#         print("=" * 72)
#         print(f"[cfg] n_sc={n_sc}  patch_size≈{patch_size}  coverage_per_cell={coverage_per_cell}")
#         print(f"[cfg] timesteps={n_timesteps_sample}  D_latent={D_latent}")
#         print(f"[cfg] sigma_min={sigma_min}  sigma_max={sigma_max}  guidance_scale={guidance_scale}")
#         print(f"[align] n_align_iters={n_align_iters}")
#         if target_st_p95 is not None:
#             print(f"[scale] target_st_p95={target_st_p95:.4f}")

#     # ------------------------------------------------------------------
#     # 1) Encode all SC cells into Z space
#     # ------------------------------------------------------------------
#     encode_bs = 1024
#     Z_chunks = []
#     for i in range(0, n_sc, encode_bs):
#         z = encoder(sc_gene_expr[i:i + encode_bs].to(device)).detach().cpu()
#         Z_chunks.append(z)
#     Z_all = torch.cat(Z_chunks, dim=0)          # (N, h)
#     del Z_chunks
#     if DEBUG_FLAG:
#         print(f"[ENC] Z_all shape={tuple(Z_all.shape)}")

#     # ------------------------------------------------------------------
#     # 2) Build k-NN index in Z-space for patch construction
#     # ------------------------------------------------------------------
#     K_nbrs = patch_size
#     nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs)  # (N, K_nbrs)

#     # ------------------------------------------------------------------
#     # 3) Define overlapping patches S_k (or reload from file)
#     # ------------------------------------------------------------------
#     pass_label = f"[PASS{_pass_number}]" if two_pass else ""
    
#     if fixed_patch_graph is not None:
#         # RELOAD existing patch graph - do NOT rebuild
#         patch_indices = [p.to(torch.long) for p in fixed_patch_graph["patch_indices"]]
#         memberships = fixed_patch_graph["memberships"]
#         K = len(patch_indices)
#         print(f"{pass_label}[PATCHWISE] Loaded fixed patch graph with {K} patches")
#     elif _pass_number == 2 and _coords_from_pass1 is not None:
#         # PASS 2: Build patches using geometry from pass 1
#         print(f"{pass_label}[PATCHWISE] Building patches from PASS1 geometry (not expression)...")
#         patch_indices, memberships = build_patches_from_coords(
#             coords=_coords_from_pass1,
#             n_cells=n_sc,
#             patch_size=patch_size,
#             coverage_per_cell=coverage_per_cell,
#             min_overlap=20
#         )
#         K = len(patch_indices)
#         print(f"{pass_label}[PATCHWISE] Built geometry-based patch graph with {K} patches")
#     else:
#         # PASS 1 (or single-pass): BUILD patch graph from expression kNN
#         n_patches_est = int(math.ceil((coverage_per_cell * n_sc) / patch_size))
#         centers = torch.randint(low=0, high=n_sc, size=(n_patches_est,), dtype=torch.long)

#         patch_indices: List[torch.Tensor] = []
#         memberships: List[List[int]] = [[] for _ in range(n_sc)]

#         # First pass: random patches around centers
#         for k, c in enumerate(centers.tolist()):
#             S_k = nbr_idx[c, :patch_size]
#             S_k = torch.unique(S_k, sorted=False)
#             patch_indices.append(S_k)
#             for idx in S_k.tolist():
#                 memberships[idx].append(k)

#         # Ensure every cell appears in at least one patch
#         for i in range(n_sc):
#             if len(memberships[i]) == 0:
#                 k = len(patch_indices)
#                 S_k = nbr_idx[i, :patch_size]
#                 S_k = torch.unique(S_k, sorted=False)
#                 patch_indices.append(S_k)
#                 memberships[i].append(k)
#                 for idx in S_k.tolist():
#                     memberships[idx].append(k)

#         K = len(patch_indices)
#         print(f"{pass_label}[PATCHWISE] Built new patch graph with {K} patches")

#         # ===================================================================
#         # DIAGNOSTIC A: PATCH GRAPH ANALYSIS
#         # ===================================================================
#         if DEBUG_FLAG:
#             print("\n" + "="*70)
#             print(f"DIAGNOSTIC A: PATCH GRAPH ANALYSIS {pass_label}")
#             print("="*70)
            
#             stats = compute_overlap_graph_stats(patch_indices, n_sc, min_overlap=20)
#             overlap_sizes = stats['overlap_sizes']
            
#             # A1: Overlap size distribution
#             print(f"\n[A1] Overlap Statistics:")
#             print(f"  Total patch pairs: {len(overlap_sizes)}")
#             print(f"  Mean overlap: {np.mean(overlap_sizes):.1f}")
#             print(f"  Median overlap: {np.median(overlap_sizes):.1f}")
#             print(f"  % pairs with ≥20: {100*np.mean(np.array(overlap_sizes) >= 20):.1f}%")
#             print(f"  % pairs with ≥50: {100*np.mean(np.array(overlap_sizes) >= 50):.1f}%")
#             print(f"  % pairs with ≥100: {100*np.mean(np.array(overlap_sizes) >= 100):.1f}%")
            
#             # Plot overlap distribution
#             import matplotlib.pyplot as plt
#             fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
#             axes[0].hist(overlap_sizes, bins=50, edgecolor='black', alpha=0.7)
#             axes[0].axvline(20, color='red', linestyle='--', label='Min overlap=20')
#             axes[0].set_xlabel('Overlap Size')
#             axes[0].set_ylabel('Count')
#             axes[0].set_title(f'Patch Overlap Distribution {pass_label}')
#             axes[0].legend()
#             axes[0].grid(alpha=0.3)
            
#             # A2: Connectivity
#             print(f"\n[A2] Patch Graph Connectivity (min_overlap=20):")
#             print(f"  # Connected components: {stats['n_components']}")
#             print(f"  Giant component size: {stats['giant_component_size']}/{K} ({stats['giant_component_frac']*100:.1f}%)")
            
#             # Plot component sizes
#             if stats['n_components'] > 1:
#                 print("  ⚠️  WARNING: Patch graph is fragmented!")
            
#             axes[1].bar(['Giant', 'Other'], 
#                     [stats['giant_component_size'], K - stats['giant_component_size']],
#                     color=['green', 'red'], alpha=0.7)
#             axes[1].set_ylabel('# Patches')
#             axes[1].set_title(f'Patch Graph Components (n={stats["n_components"]}) {pass_label}')
#             axes[1].grid(alpha=0.3, axis='y')
            
#             plt.tight_layout()
#             plt.show()
#             print("="*70 + "\n")


#     if DEBUG_FLAG:
#         cover_counts = torch.tensor([len(memberships[i]) for i in range(n_sc)], dtype=torch.float32)
#         patch_sizes = torch.tensor([len(S_k) for S_k in patch_indices], dtype=torch.float32)
#         print(f"[PATCH] final n_patches={K}")
#         print(f"[PATCH] per-cell coverage: "
#               f"min={cover_counts.min().item():.1f} "
#               f"p25={cover_counts.quantile(0.25).item():.1f} "
#               f"p50={cover_counts.quantile(0.50).item():.1f} "
#               f"p75={cover_counts.quantile(0.75).item():.1f} "
#               f"max={cover_counts.max().item():.1f}")
#         print(f"[PATCH] patch sizes: "
#               f"min={patch_sizes.min().item():.0f} "
#               f"p25={patch_sizes.quantile(0.25).item():.0f} "
#               f"p50={patch_sizes.quantile(0.50).item():.0f} "
#               f"p75={patch_sizes.quantile(0.75).item():.0f} "
#               f"max={patch_sizes.max().item():.0f}")
        
#     # ------------------------------------------------------------------
#     # DEBUG 1: Patch graph summary
#     # ------------------------------------------------------------------
#     if DEBUG_FLAG:
#         cover_counts = torch.tensor([len(memberships[i]) for i in range(n_sc)], dtype=torch.float32)
#         patch_sizes = torch.tensor([len(S_k) for S_k in patch_indices], dtype=torch.float32)

#         print(f"\n[PATCH] final n_patches={K}")
#         print(f"[PATCH] per-cell coverage stats: "
#               f"min={cover_counts.min().item():.1f} "
#               f"p25={cover_counts.quantile(0.25).item():.1f} "
#               f"p50={cover_counts.quantile(0.50).item():.1f} "
#               f"p75={cover_counts.quantile(0.75).item():.1f} "
#               f"max={cover_counts.max().item():.1f}")
#         print(f"[PATCH] cells with coverage==1: {(cover_counts==1).sum().item()} "
#               f"({(cover_counts==1).float().mean().item()*100:.1f}%)")
#         print(f"[PATCH] patch size stats: "
#               f"min={patch_sizes.min().item():.0f} "
#               f"p25={patch_sizes.quantile(0.25).item():.0f} "
#               f"p50={patch_sizes.quantile(0.50).item():.0f} "
#               f"p75={patch_sizes.quantile(0.75).item():.0f} "
#               f"max={patch_sizes.max().item():.0f}")
        
#         # Save patch graph for reproducibility testing
#         incidence = torch.zeros(n_sc, K, dtype=torch.bool)
#         for k_idx, S_k in enumerate(patch_indices):
#             incidence[S_k, k_idx] = True
        
#         torch.save(
#             {
#                 "patch_indices": [p.cpu() for p in patch_indices],
#                 "memberships": memberships,
#                 "incidence": incidence.cpu(),
#                 "n_sc": n_sc,
#                 "patch_size": patch_size,
#             },
#             f"debug_patch_graph_seed.pt",
#         )
#         print("[DEBUG] Saved patch graph to debug patch graph")


#     # ===================================================================
#     # DEBUG: Precompute GT global kNN for closure/coverage diagnostics
#     # ===================================================================
#     gt_knn_global = None
#     gt_subset_indices = None
#     if debug_knn and gt_coords is not None:
#         gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
#         N_gt = gt_coords_t.shape[0]
#         M = min(N_gt, debug_global_subset)
        
#         # Random subset for efficiency
#         gt_subset_indices = torch.randperm(N_gt)[:M]
#         gt_coords_subset = gt_coords_t[gt_subset_indices]
        
#         # Compute kNN on subset
#         k_closure = max(debug_k_list)
#         gt_knn_global, _ = _knn_indices_dists(gt_coords_subset, k_closure)
        
#         print(f"\n[DEBUG-KNN] Precomputed GT kNN on subset of {M} cells (k={k_closure})")



#     # EDM Karras sigma schedule - use refinement sigma range
#     # sigma_max should match training sigma_refine_max (typically 1.0 * sigma_data)
#     sigma_refine_max = min(sigma_max, 20.0 * sigma_data)  # Match training
#     sigmas = uet.edm_sigma_schedule(n_timesteps_sample, sigma_min, sigma_refine_max, rho=7.0, device=device)
    
#     if DEBUG_FLAG:
#         print(f"[SAMPLE] Refinement mode: sigma range [{sigma_min:.4f}, {sigma_refine_max:.4f}]")

#     # ------------------------------------------------------------------
#     # Sigma-dependent guidance schedule + CFG WNORM
#     # guidance_scale is the CEILING (max effective guidance at low sigma)
#     # ------------------------------------------------------------------
#     def _sigma_guidance_eff(sigma_val: float, gs_ceiling: float) -> float:
#         """
#         Compute effective guidance based on sigma level.
#         - sigma > 1.0:    guidance_eff in [0.0, 0.5] (linear ramp 0→0.5)
#         - sigma in [0.3, 1.0]: ramp from 0.5 to 1.0
#         - sigma < 0.3:    full guidance (gs_ceiling)
#         Returns multiplier in [0, gs_ceiling].
#         """
#         if sigma_val > 1.0:
#             # High noise: minimal guidance, ramp from 0 at sigma=5 to 0.5 at sigma=1
#             # Linear: at sigma=5 → 0.0, at sigma=1 → 0.5
#             frac = max(0.0, min(1.0, (5.0 - sigma_val) / 4.0))
#             return 0.5 * frac * gs_ceiling
#         elif sigma_val > 0.3:
#             # Medium noise: ramp from 0.5 to 1.0
#             # At sigma=1 → 0.5, at sigma=0.3 → 1.0
#             frac = (1.0 - sigma_val) / 0.7  # 0 at sigma=1, 1 at sigma=0.3
#             return (0.5 + 0.5 * frac) * gs_ceiling
#         else:
#             # Low noise: full guidance ceiling
#             return gs_ceiling

#     patch_coords: List[torch.Tensor] = []
#     # Collectors for PRE-STITCH diagnostics
#     prestitch_edge_spearman_list = []
#     prestitch_knn_overlap_list = {k_val: [] for k_val in debug_k_list}


#     if DEBUG_FLAG:
#         print("\n[STEP] Sampling local geometries for patches...")

#     with torch.no_grad():
#         for k in tqdm(range(K), desc="Sampling patches"):
#             S_k = patch_indices[k]
#             m_k = S_k.numel()
#             Z_k = Z_all[S_k].to(device)         # (m_k, h)

#             Z_k_batched = Z_k.unsqueeze(0)      # (1, m_k, h)
#             mask_k = torch.ones(1, m_k, dtype=torch.bool, device=device)
#             H_k = context_encoder(Z_k_batched, mask_k)
            
#             # APPLY CORAL TRANSFORMATION IF ENABLED
#             if coral_params is not None:
#                 from core_models_et_p3 import GEMSModel
#                 H_k = GEMSModel.apply_coral_transform(
#                     H_k,
#                     mu_sc=coral_params['mu_sc'],
#                     A=coral_params['A'],
#                     B=coral_params['B'],
#                     mu_st=coral_params['mu_st']
#                 )

#             # Start from generator proposal + noise (refinement mode)
#             V_gen = generator(H_k, mask_k)  # Generator proposal
#             V_t = V_gen + torch.randn_like(V_gen) * sigmas[0]
#             V_t = V_t * mask_k.unsqueeze(-1).float()  # Mask out invalid positions

            
#             # EDM Euler + Heun sampler
#             for i in range(len(sigmas) - 1):
#                 sigma = sigmas[i]
#                 sigma_next = sigmas[i + 1]
#                 sigma_b = sigma.view(1)  # (B=1,)

#                 # x0 predictions with CFG
#                 x0_c = score_net.forward_edm(V_t, sigma_b, H_k, mask_k, sigma_data, self_cond=None)

#                 if guidance_scale != 1.0:
#                     H_null = torch.zeros_like(H_k)
#                     x0_u = score_net.forward_edm(V_t, sigma_b, H_null, mask_k, sigma_data, self_cond=None)
#                     # x0 = x0_u + guidance_scale * (x0_c - x0_u)
#                     # Sigma-dependent effective guidance (guidance_scale is ceiling)
#                     sigma_f = float(sigma)
#                     guidance_eff = _sigma_guidance_eff(sigma_f, guidance_scale)

#                     # CFG WNORM: normalize diff to prevent blowing up locality
#                     diff = x0_c - x0_u
#                     diff_norm = diff.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)  # (B, 1, 1)
#                     x0_u_norm = x0_u.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)
#                     wnorm_scale = (x0_u_norm / diff_norm).clamp(max=1.0)  # clamp to avoid amplifying

#                     # Apply: x0 = x0_u + guidance_eff * wnorm_scale * diff
#                     x0 = x0_u + guidance_eff * wnorm_scale * diff
#                 else:
#                     x0 = x0_c

#                 # Debug on first patch
#                 if DEBUG_FLAG and k == 0 and i < 3:
#                     if guidance_scale != 1.0:
#                         du = x0_u.norm(dim=[1, 2]).mean().item()
#                         dc = x0_c.norm(dim=[1, 2]).mean().item()
#                         diff_mag = diff.norm(dim=[1, 2]).mean().item()
#                         wn_s = wnorm_scale.mean().item()
#                         print(f"  [PATCH0] i={i:3d} sigma={sigma_f:.4f} g_eff={guidance_eff:.3f} "
#                               f"||x0_u||={du:.3f} ||x0_c||={dc:.3f} ||diff||={diff_mag:.3f} wnorm={wn_s:.3f}")

#                 # Euler step
#                 d = (V_t - x0) / sigma.clamp_min(1e-8)
#                 V_euler = V_t + (sigma_next - sigma) * d

#                 # Heun corrector (skip if sigma_next==0)
#                 if sigma_next > 0:
#                     x0_next_c = score_net.forward_edm(V_euler, sigma_next.view(1), H_k, mask_k, sigma_data, self_cond=None)
#                     if guidance_scale != 1.0:
#                         x0_next_u = score_net.forward_edm(V_euler, sigma_next.view(1), H_null, mask_k, sigma_data, self_cond=None)
#                         # x0_next = x0_next_u + guidance_scale * (x0_next_c - x0_next_u)
#                         # Sigma-dependent effective guidance for Heun step (use sigma_next)
#                         sigma_next_f = float(sigma_next)
#                         guidance_eff_next = _sigma_guidance_eff(sigma_next_f, guidance_scale)

#                         # CFG WNORM for Heun step
#                         diff_next = x0_next_c - x0_next_u
#                         diff_next_norm = diff_next.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)
#                         x0_next_u_norm = x0_next_u.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)
#                         wnorm_scale_next = (x0_next_u_norm / diff_next_norm).clamp(max=1.0)

#                         x0_next = x0_next_u + guidance_eff_next * wnorm_scale_next * diff_next
#                     else:
#                         x0_next = x0_next_c

#                     d2 = (V_euler - x0_next) / sigma_next.clamp_min(1e-8)
#                     V_t = V_t + (sigma_next - sigma) * 0.5 * (d + d2)
#                 else:
#                     V_t = V_euler

#                 # Apply mask
#                 V_t = V_t * mask_k.unsqueeze(-1).float()

                
#                 # Optional stochastic noise (if eta > 0)
#                 if eta > 0 and sigma_next > 0:
#                     noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma**2, min=0))
#                     V_t = V_t + noise_scale * torch.randn_like(V_t)


#             # if DEBUG_FLAG and (k % max(1, K // 5) == 0):
#             #     rms = V_canon.pow(2).mean().sqrt().item()
#             # NEW: Only center, do NOT apply unit RMS (matches training)
#             V_final = V_t.squeeze(0)  # (m_k, D)
#             V_centered = V_final - V_final.mean(dim=0, keepdim=True)

#             # ===================================================================
#             # DIAGNOSTIC B: PATCH SAMPLING QUALITY (first 3 patches only)
#             # ===================================================================
#             if DEBUG_FLAG and k < 3:
#                 print(f"\n[DIAGNOSTIC B] Patch {k} sampling analysis:")
                
#                 # B1: Check if patch is isotropic (blob)
#                 cov = torch.cov(V_centered.float().T)
#                 eigs = torch.linalg.eigvalsh(cov)
#                 aniso_ratio = float(eigs.max() / (eigs.min() + 1e-8))
#                 print(f"  Anisotropy ratio: {aniso_ratio:.2f} (higher=more structured)")
                
#                 if aniso_ratio < 2.0:
#                     print(f"  ⚠️  WARNING: Patch {k} is very isotropic (blob-like)!")
#             patch_coords.append(V_centered.detach().cpu())


#             # ===================================================================
#             # [PATCH-KNN-VS-GT] PRE-STITCH: kNN overlap with GT (per-patch)
#             # ===================================================================
#             if debug_knn and gt_coords is not None and k < debug_max_patches:
#                 with torch.no_grad():
#                     gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
#                     gt_patch = gt_coords_t[S_k]  # (m_k, 2 or D)
#                     pred_patch = V_centered.float()  # (m_k, D_latent)
                    
#                     for k_val in debug_k_list:
#                         if m_k > k_val + 1:
#                             knn_pred, _ = _knn_indices_dists(pred_patch, k_val)
#                             knn_gt, _ = _knn_indices_dists(gt_patch, k_val)
#                             overlap = _knn_overlap_score(knn_pred, knn_gt)
                            
#                             # Collect for aggregation
#                             prestitch_knn_overlap_list[k_val].append(overlap.mean().item())
                            
#                             if k < 10:  # Only print first 5 patches
#                                 print(f"  [PATCH-KNN-VS-GT][PRE-STITCH] patch={k} n={m_k} "
#                                       f"k={k_val} mean={overlap.mean().item():.3f} "
#                                       f"p50={overlap.median().item():.3f}")


#             # ===================================================================
#             # [PATCH-KNN-CLOSURE]: How neighbor-closed is this patch vs global GT?
#             # ===================================================================
#             if debug_knn and gt_subset_indices is not None and k < debug_max_patches:
#                 with torch.no_grad():
#                     # Find overlap between patch and GT subset
#                     S_k_set = set(S_k.tolist())
#                     subset_set = set(gt_subset_indices.tolist())
#                     patch_in_subset = S_k_set & subset_set
                    
#                     if len(patch_in_subset) >= 20:
#                         # Map to subset-local indices
#                         subset_to_local = {int(g): i for i, g in enumerate(gt_subset_indices.tolist())}
#                         patch_local_in_subset = [subset_to_local[g] for g in patch_in_subset]
                        
#                         closure_fracs = []
#                         for k_val in [10, 20]:
#                             if k_val <= gt_knn_global.shape[1]:
#                                 for local_idx in patch_local_in_subset:
#                                     gt_neighbors = set(gt_knn_global[local_idx, :k_val].tolist())
#                                     neighbors_in_patch = gt_neighbors & set(patch_local_in_subset)
#                                     closure = len(neighbors_in_patch) / k_val
#                                     closure_fracs.append(closure)
                        
#                         if closure_fracs and k < 10:
#                             closure_t = torch.tensor(closure_fracs)
#                             print(f"  [PATCH-KNN-CLOSURE] patch={k} n_eval={len(patch_in_subset)} "
#                                   f"closure_mean={closure_t.mean().item():.3f} "
#                                   f"closure_p50={closure_t.median().item():.3f}")

#             # ===================================================================
#             # [PATCH-LOCAL-EDGE-CORR][PRE-STITCH]: Local edge Spearman within patch
#             # ===================================================================
#             if debug_knn and gt_coords is not None and k < debug_max_patches:
#                 with torch.no_grad():
#                     gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
#                     gt_patch = gt_coords_t[S_k].cpu().numpy()  # (m_k, 2)
#                     pred_patch = V_centered.cpu().numpy()       # (m_k, D_latent)
                    
#                     if m_k > 15:
#                         # Compute GT kNN within this patch
#                         from sklearn.neighbors import NearestNeighbors
#                         k_edge = min(20, m_k - 1)
#                         nbrs = NearestNeighbors(n_neighbors=k_edge+1, algorithm='ball_tree').fit(gt_patch)
#                         _, gt_knn_patch = nbrs.kneighbors(gt_patch)
#                         gt_knn_patch = gt_knn_patch[:, 1:]  # Remove self
                        
#                         # Compute local-edge Spearman
#                         edge_spearman = _local_edge_spearman(pred_patch, gt_patch, gt_knn_patch)
                        
#                         if not np.isnan(edge_spearman):
#                             prestitch_edge_spearman_list.append(edge_spearman)
                        
#                         if k < 10:  # Only print first 5 patches
#                             print(f"  [PATCH-LOCAL-EDGE-CORR][PRE-STITCH] patch={k} n={m_k} "
#                                   f"k_edge={k_edge} spearman={edge_spearman:.3f}")


#             # DEBUG 2: Per-patch sample diagnostics
#             if DEBUG_FLAG and (k < 5 or k % max(1, K // 5) == 0):
#                 rms = V_centered.pow(2).mean().sqrt().item()

#                 # Coord covariance eigs to see anisotropy / effective dimension
#                 cov_k = torch.cov(V_centered.float().T)
#                 eigs_k = torch.linalg.eigvalsh(cov_k)
#                 dim_eff = float((eigs_k.sum() ** 2) / (eigs_k ** 2).sum())
#                 aniso = float(eigs_k.max() / (eigs_k.min().clamp(min=1e-8)))

#                 print(f"[PATCH-SAMPLE] k={k}/{K} m_k={m_k} "
#                       f"rms={rms:.3f} dim_eff={dim_eff:.2f} aniso={aniso:.1f} "
#                       f"eigs_min={eigs_k.min().item():.3e} eigs_max={eigs_k.max().item():.3e}")

#             if DEBUG_FLAG and (k % max(1, K // 5) == 0):
#                 rms = V_centered.pow(2).mean().sqrt().item()
#                 print(f"  [PATCH {k}/{K}] RMS={rms:.3f} (centered, natural scale)")

#                 # mean_norm = V_canon.mean(dim=0).norm().item()
#                 mean_norm = V_centered.mean(dim=0).norm().item()
#                 print(f"[PATCH] k={k}/{K} m_k={m_k} "
#                       f"coords_rms={rms:.3f} center_norm={mean_norm:.3e}")

#             if 'cuda' in device:
#                 torch.cuda.empty_cache()

#     # ===================================================================
#     # [PRE-STITCH SUMMARY] Aggregated stats across patches
#     # ===================================================================
#     if debug_knn and gt_coords is not None:
#         print("\n" + "="*70)
#         print("[PRE-STITCH SUMMARY] Aggregated across patches")
#         print("="*70)
        
#         if prestitch_edge_spearman_list:
#             es = np.array(prestitch_edge_spearman_list)
#             print(f"  [PATCH-LOCAL-EDGE-CORR] n_patches={len(es)}")
#             print(f"    Spearman: mean={np.mean(es):.3f} p50={np.median(es):.3f} "
#                   f"p10={np.percentile(es,10):.3f} p90={np.percentile(es,90):.3f}")
            
#             if np.median(es) < 0.3:
#                 print("    → ⚠️ LOW: Patch generator produces locally wrong geometry")
#             elif np.median(es) > 0.5:
#                 print("    → ✓ DECENT: Patch outputs have reasonable local structure")
        
#         for k_val in debug_k_list:
#             if prestitch_knn_overlap_list[k_val]:
#                 ov = np.array(prestitch_knn_overlap_list[k_val])
#                 print(f"  [PATCH-KNN-VS-GT] k={k_val} n_patches={len(ov)}")
#                 print(f"    Overlap: mean={np.mean(ov):.3f} p50={np.median(ov):.3f} "
#                       f"p10={np.percentile(ov,10):.3f} p90={np.percentile(ov,90):.3f}")
        
#         print("="*70 + "\n")

#     # ------------------------------------------------------------------
#     # DEBUG: Save patch coords (AFTER all patches sampled, BEFORE alignment)
#     # ------------------------------------------------------------------
#     if DEBUG_FLAG:
#         all_rms = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
#         print(f"\n[PATCH-SAMPLE] Final RMS distribution across patches: "
#               f"min={all_rms.min().item():.3f} "
#               f"p25={all_rms.quantile(0.25).item():.3f} "
#               f"p50={all_rms.quantile(0.50).item():.3f} "
#               f"p75={all_rms.quantile(0.75).item():.3f} "
#               f"max={all_rms.max().item():.3f}")
        
#         torch.save(
#             {
#                 "patch_indices": [p.cpu() for p in patch_indices],
#                 "patch_coords": [pc.cpu() for pc in patch_coords],
#             },
#             f"debug_patch_coords_seed.pt",
#         )
#         print(f"[DEBUG] Saved patch coords to debug_patch_coords_seed.pt\n")



#     # ===================================================================
#     # DIAGNOSTIC B3: OVERLAP NEIGHBOR AGREEMENT (JACCARD)
#     # ===================================================================
#     if DEBUG_FLAG:
#         print("\n" + "="*70)
#         print("DIAGNOSTIC B3: OVERLAP NEIGHBOR AGREEMENT")
#         print("="*70)
        
#         overlap_jaccards = []
#         overlap_dist_corrs = []
        
#         # Sample 20 random overlapping patch pairs
#         overlap_pairs = []
#         for i in range(min(K, 50)):
#             for j in range(i+1, min(K, 50)):
#                 S_i = set(patch_indices[i].tolist())
#                 S_j = set(patch_indices[j].tolist())
#                 shared = S_i & S_j
#                 if len(shared) >= 30:  # Need enough for meaningful kNN
#                     overlap_pairs.append((i, j, list(shared)))
        
#         # Analyze up to 20 pairs
#         for i, j, shared_list in overlap_pairs[:20]:
#             # Get coords for shared cells in both patches

#             # Build position lookup dicts (O(1) lookup)
#             pos_i = {int(cid): p for p, cid in enumerate(patch_indices[i].tolist())}
#             pos_j = {int(cid): p for p, cid in enumerate(patch_indices[j].tolist())}

#             shared_idx_i = torch.tensor([pos_i[c] for c in shared_list], dtype=torch.long)
#             shared_idx_j = torch.tensor([pos_j[c] for c in shared_list], dtype=torch.long)
            
#             V_i_shared = patch_coords[i][shared_idx_i]
#             V_j_shared = patch_coords[j][shared_idx_j]
            
#             # Compute kNN Jaccard
#             knn_i = knn_sets(V_i_shared, k=min(10, len(shared_list)-1))
#             knn_j = knn_sets(V_j_shared, k=min(10, len(shared_list)-1))
#             jaccard = mean_jaccard(knn_i, knn_j)
#             overlap_jaccards.append(jaccard)
            
#             # Also compute distance correlation
#             D_i = torch.cdist(V_i_shared, V_i_shared).cpu().numpy()
#             D_j = torch.cdist(V_j_shared, V_j_shared).cpu().numpy()
#             triu = np.triu_indices(len(shared_list), k=1)
#             dist_corr = np.corrcoef(D_i[triu], D_j[triu])[0, 1]
#             overlap_dist_corrs.append(dist_corr)
        
#         if overlap_jaccards:
#             print(f"\n[B3] Analyzed {len(overlap_jaccards)} overlapping patch pairs:")
#             print(f"  kNN Jaccard: {np.mean(overlap_jaccards):.3f} ± {np.std(overlap_jaccards):.3f}")
#             print(f"  Distance corr: {np.mean(overlap_dist_corrs):.3f} ± {np.std(overlap_dist_corrs):.3f}")
            
#             if np.mean(overlap_jaccards) < 0.3:
#                 print("  ⚠️  LOW JACCARD: Patches disagree on neighborhoods → DIFFUSION PROBLEM")
#             elif np.mean(overlap_jaccards) > 0.6:
#                 print("  ✓ GOOD JACCARD: Patches agree on neighborhoods")
            
#             # Plot
#             import matplotlib.pyplot as plt
#             fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
#             axes[0].hist(overlap_jaccards, bins=20, edgecolor='black', alpha=0.7, color='blue')
#             axes[0].axvline(np.mean(overlap_jaccards), color='red', linestyle='--', 
#                            label=f'Mean={np.mean(overlap_jaccards):.3f}')
#             axes[0].set_xlabel('kNN Jaccard')
#             axes[0].set_ylabel('Count')
#             axes[0].set_title('Overlap Neighbor Agreement (Jaccard)')
#             axes[0].legend()
#             axes[0].grid(alpha=0.3)
            
#             axes[1].scatter(overlap_dist_corrs, overlap_jaccards, alpha=0.6)
#             axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
#             axes[1].axvline(0.8, color='red', linestyle='--', alpha=0.5)
#             axes[1].set_xlabel('Distance Correlation')
#             axes[1].set_ylabel('kNN Jaccard')
#             axes[1].set_title('Distance Corr vs Neighbor Agreement')
#             axes[1].grid(alpha=0.3)
            
#             plt.tight_layout()
#             plt.show()
        
#         print("="*70 + "\n")

#     # Compute median patch RMS as target scale for global alignment
#     patch_rms_list = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
#     rms_target = patch_rms_list.median().item()

#     if DEBUG_FLAG:
#         print(f"[ALIGN] Target RMS for global space: {rms_target:.3f} (median of patches)")


#     # DEBUG 3: RMS distribution across all patches
#     if DEBUG_FLAG:
#         all_rms = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
#         print(f"\n[PATCH-SAMPLE] RMS distribution across patches: "
#             f"min={all_rms.min().item():.3f} "
#             f"p25={all_rms.quantile(0.25).item():.3f} "
#             f"p50={all_rms.quantile(0.50).item():.3f} "
#             f"p75={all_rms.quantile(0.75).item():.3f} "
#             f"max={all_rms.max().item():.3f}")
        
#         # Save patch coords for reproducibility testing
#         torch.save(
#             {
#                 "patch_indices": [p.cpu() for p in patch_indices],
#                 "patch_coords": [pc.cpu() for pc in patch_coords],
#             },
#             "debug_patch_coords.pt",
#         )
#         print("[DEBUG] Saved patch coords to debug_patch_coords.pt\n")

#     # ===== DIAGNOSTIC: Patch-level geometry BEFORE any stitching =====
#     if DEBUG_FLAG and K > 1:
#         print("\n" + "="*60)
#         print("PATCH OVERLAP DIAGNOSTIC (pre-alignment)")
#         print("="*60)
        
#         # Find pairs of overlapping patches
#         overlap_corrs = []
        
#         for k1 in range(min(10, K)):  # Check first 10 patches
#             S_k1 = set(patch_indices[k1].cpu().tolist())
#             V_k1 = patch_coords[k1].cpu()  # (m_k1, D)
            
#             for k2 in range(k1+1, min(k1+5, K)):  # Check next 4 patches
#                 S_k2 = set(patch_indices[k2].cpu().tolist())
                
#                 # Find shared cells
#                 shared = S_k1 & S_k2
#                 if len(shared) < 20:  # Need enough overlap
#                     continue
                
#                 shared_list = sorted(list(shared))
                
#                 # Get positions in each patch
#                 S_k1_list = patch_indices[k1].cpu().tolist()
#                 S_k2_list = patch_indices[k2].cpu().tolist()
                
#                 pos_k1 = [S_k1_list.index(s) for s in shared_list]
#                 pos_k2 = [S_k2_list.index(s) for s in shared_list]
                
#                 # Extract shared cell coords from each patch
#                 V_shared_k1 = V_k1[pos_k1]  # (n_shared, D)
#                 V_shared_k2 = patch_coords[k2].cpu()[pos_k2]
                
#                 # Compute pairwise distances within shared cells
#                 D_k1 = torch.cdist(V_shared_k1, V_shared_k1).numpy()
#                 D_k2 = torch.cdist(V_shared_k2, V_shared_k2).numpy()
                
#                 # Compare distances (upper triangle)
#                 tri = np.triu_indices(len(shared_list), k=1)
#                 if len(tri[0]) > 10:
#                     from scipy.stats import pearsonr
#                     corr = pearsonr(D_k1[tri], D_k2[tri])[0]
#                     overlap_corrs.append(corr)
        
#         if overlap_corrs:
#             overlap_corrs = np.array(overlap_corrs)
#             print(f"[OVERLAP] Checked {len(overlap_corrs)} patch pairs")
#             print(f"[OVERLAP] Distance correlation in shared cells:")
#             print(f"  min={overlap_corrs.min():.3f} "
#                 f"p25={np.percentile(overlap_corrs, 25):.3f} "
#                 f"median={np.median(overlap_corrs):.3f} "
#                 f"p75={np.percentile(overlap_corrs, 75):.3f} "
#                 f"max={overlap_corrs.max():.3f}")
            
#             if np.median(overlap_corrs) < 0.7:
#                 print("\n⚠️  WARNING: Low overlap consistency!")
#                 print("    Patches disagree about shared cells' neighborhoods")
#                 print("    → Problem is in DIFFUSION MODEL (Stage C)")
#             elif np.median(overlap_corrs) > 0.85:
#                 print("\n✓ Good overlap consistency")
#                 print("    Patches agree about shared cells")
#                 print("    → If final result is bad, problem is in STITCHING")
#             else:
#                 print("\n⚠️  Moderate overlap consistency")
#                 print("    Some disagreement between patches")
        
#         print("="*60 + "\n")

#     # DEBUG: Patch-level correlation to GT (if GT is available)
#     if DEBUG_FLAG and hasattr(sc_gene_expr, 'gt_coords'):
#         from scipy.spatial.distance import cdist
#         from scipy.stats import pearsonr
        
#         gt_coords = sc_gene_expr.gt_coords  # Assume passed in somehow
#         patch_local_corrs = []
        
#         for k in range(K):
#             S_k_np = patch_indices[k].cpu().numpy()
#             V_k = patch_coords[k].cpu().numpy()  # (m_k, D_latent)
#             gt_k = gt_coords[S_k_np]  # (m_k, 2)
            
#             D_pred = cdist(V_k, V_k)
#             D_gt = cdist(gt_k, gt_k)
            
#             tri = np.triu_indices(len(S_k_np), k=1)
#             if len(tri[0]) > 0:
#                 r = pearsonr(D_pred[tri], D_gt[tri])[0]
#                 patch_local_corrs.append(r)
        
#         patch_local_corrs = np.array(patch_local_corrs)
#         print(f"\n[PATCH-LOCAL] Pearson vs GT: "
#               f"min={patch_local_corrs.min():.3f} "
#               f"p25={np.percentile(patch_local_corrs, 25):.3f} "
#               f"p50={np.median(patch_local_corrs):.3f} "
#               f"p75={np.percentile(patch_local_corrs, 75):.3f} "
#               f"max={patch_local_corrs.max():.3f}")

#     # ------------------------------------------------------------------
#     # 5) Global alignment: alternately solve patch transforms and X
#     # ------------------------------------------------------------------
#     if DEBUG_FLAG:
#         print("\n[ALIGN] Starting global alignment...")

#     # 5.1 Initialize X by averaging patch coords per cell
#     X_global = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
#     W_global = torch.zeros(n_sc, 1, dtype=torch.float32, device=device)

#     if DEBUG_FLAG:
#         print("\n[ALIGN] Stitching with centrality weighting...")

#     for k in range(K):
#         # Global indices for this patch
#         S_k = patch_indices[k].to(device)              # (m_k,)
#         V_k = patch_coords[k].to(device)               # (m_k, D)

#         # --- centrality weights in patch coordinates ---
#         # center of patch in its local frame
#         center_k = V_k.mean(dim=0, keepdim=True)       # (1, D)
#         # distance of each point from center
#         dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
#         max_d = dists.max().clamp_min(1e-6)
#         # linear taper: 1.0 at center, ~0.2 at edge, clamped at 0.01
#         weights_k = 1.0 - (dists / (max_d * 1.2))
#         weights_k = weights_k.clamp(min=0.01)          # (m_k, 1)
#         # -----------------------------------------------

#         # accumulate weighted coords and total weight
#         X_global.index_add_(0, S_k, V_k * weights_k)   # (N,D) += (m_k,D)
#         W_global.index_add_(0, S_k, weights_k)         # (N,1) += (m_k,1)

#     # normalize by total weight where seen
#     mask_seen = W_global.squeeze(-1) > 0
#     X_global[mask_seen] /= W_global[mask_seen]

#     # global recentering
#     X_global = X_global - X_global.mean(dim=0, keepdim=True)
#     rms_init = X_global.pow(2).mean().sqrt().item()

#     # Rescale to match median patch RMS (data-driven)
#     scale_factor = (rms_target / (rms_init + 1e-8))
#     scale_factor = torch.clamp(torch.tensor(scale_factor), 0.25, 4.0).item()  # Safety bounds
#     X_global = X_global * scale_factor

#     rms_final = X_global.pow(2).mean().sqrt().item()
#     if DEBUG_FLAG:
#         print(f"[ALIGN] Init X_global: rms_raw={rms_init:.3f} "
#             f"→ rescaled to {rms_final:.3f} (target={rms_target:.3f}, scale={scale_factor:.3f})")

#     # 5.2 Alternating Procrustes alignment (SIMPLIFIED: fixed scale, single iteration)
#     s_global = 1.0
    
#     print(f"\n[ALIGN] Using FIXED global scale s_global={s_global} (no dynamic scaling)")
#     print(f"[ALIGN] Running simplified alignment with {n_align_iters} iteration(s)...")
    
#     for it in range(n_align_iters):
#         if DEBUG_FLAG:
#             print(f"\n[ALIGN] Iteration {it + 1}/{n_align_iters}")

#         R_list: List[torch.Tensor] = []
#         t_list: List[torch.Tensor] = []
#         s_list: List[torch.Tensor] = []
        
#         # For global alignment loss tracking
#         per_patch_mse = []

#         # ======================================================================
#         # Step A: Compute rotations (NO scale accumulation)
#         # ======================================================================
#         for k in range(K):
#             S_k = patch_indices[k]
#             V_k = patch_coords[k].to(X_global.device)   # (m_k, D)
#             X_k = X_global[S_k]                         # (m_k, D)
#             m_k = V_k.shape[0]

#             # Centrality weights (same as Step B initialization)
#             center_k = V_k.mean(dim=0, keepdim=True)       # (1, D)
#             dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
#             max_d = dists.max().clamp_min(1e-6)
#             weights_k = 1.0 - (dists / (max_d * 1.2))
#             weights_k = weights_k.clamp(min=0.01)          # (m_k, 1)

#             # Weighted centroids
#             w_sum = weights_k.sum()
#             mu_X = (weights_k * X_k).sum(dim=0, keepdim=True) / w_sum
#             mu_V = (weights_k * V_k).sum(dim=0, keepdim=True) / w_sum
            
#             # Center
#             Xc = X_k - mu_X
#             Vc = V_k - mu_V

#             # Apply sqrt weights for proper weighted Procrustes
#             w_sqrt = weights_k.sqrt()
#             Xc_w = Xc * w_sqrt
#             Vc_w = Vc * w_sqrt

#             # Weighted cross-covariance
#             C = Xc_w.T @ Vc_w
            
#             # SVD for rotation
#             U, S_vals, Vh = torch.linalg.svd(C, full_matrices=False)
#             R_k = U @ Vh
#             if torch.det(R_k) < 0:
#                 U[:, -1] *= -1
#                 R_k = U @ Vh

            # # Compute per-patch scale (no clamp initially - let's see natural values)
            # numer = S_vals.sum()
            # denom = (Vc_w ** 2).sum().clamp_min(1e-8)
            # s_k_raw = numer / denom

            # # Gentle safety clamp (wide range to allow data-driven values)
            # s_k = s_k_raw.clamp(0.3, 3.0)

            # R_list.append(R_k)
            # s_list.append(s_k)

#         # ======================================================================
#         # NO GLOBAL SCALE RECOMPUTATION - use fixed s_global = 1.0
#         # ======================================================================
#         if DEBUG_FLAG and it == 0:
#             # print(f"[ALIGN] Using FIXED s_global={s_global} (not recomputed from patches)")
#             s_tensor = torch.stack(s_list)
#             print(f"[ALIGN] per-patch s_k: "
#                   f"min={s_tensor.min().item():.3f} "
#                   f"p25={s_tensor.quantile(0.25).item():.3f} "
#                   f"p50={s_tensor.quantile(0.50).item():.3f} "
#                   f"p75={s_tensor.quantile(0.75).item():.3f} "
#                   f"max={s_tensor.max().item():.3f}")
            
#             # Show how many are hitting clamps
#             n_clamp_low = (s_tensor < 0.31).sum().item()
#             n_clamp_high = (s_tensor > 2.99).sum().item()
#             if n_clamp_low > 0 or n_clamp_high > 0:
#                 print(f"[ALIGN] WARNING: {n_clamp_low} patches hit lower clamp, "
#                     f"{n_clamp_high} hit upper clamp - consider adjusting bounds")

#         # ======================================================================
#         # Step A (cont.): Compute translations and track alignment error
#         # ======================================================================
#         for k in range(K):
#             S_k = patch_indices[k]
#             V_k = patch_coords[k].to(X_global.device)
#             X_k = X_global[S_k]
#             R_k = R_list[k]
#             s_k = s_list[k]

#             # Recompute centrality weights (or cache from above)
#             center_k = V_k.mean(dim=0, keepdim=True)
#             dists = torch.norm(V_k - center_k, dim=1, keepdim=True)
#             max_d = dists.max().clamp_min(1e-6)
#             weights_k = 1.0 - (dists / (max_d * 1.2))
#             weights_k = weights_k.clamp(min=0.01)

#             w_sum = weights_k.sum()
#             mu_X = (weights_k * X_k).sum(dim=0, keepdim=True) / w_sum
#             mu_V = (weights_k * V_k).sum(dim=0, keepdim=True) / w_sum

#             # Translation using this patch's scale
#             t_k = (mu_X - s_k * (mu_V @ R_k.T)).squeeze(0)
#             t_list.append(t_k)

#             # Track patch alignment error with current X_global
#             X_hat_k = s_k * (V_k @ R_k.T) + t_k  # (m_k, D)
#             sqerr = (X_hat_k - X_k).pow(2).sum(dim=1)  # (m_k,)
#             patch_mse = sqerr.mean().item()
#             per_patch_mse.append(patch_mse)

#         if DEBUG_FLAG:
#             per_patch_mse_t = torch.tensor(per_patch_mse)
#             print(f"[ALIGN] per-patch mse: "
#                 f"p10={per_patch_mse_t.quantile(0.10).item():.4e} "
#                 f"p50={per_patch_mse_t.quantile(0.50).item():.4e} "
#                 f"p90={per_patch_mse_t.quantile(0.90).item():.4e}")

#             # DEBUG 5: Transform magnitudes
#             R_norms = []
#             t_norms = []
#             for k_t in range(K):
#                 R_k_t = R_list[k_t]
#                 t_k_t = t_list[k_t]
#                 # how far from identity?
#                 R_norms.append((R_k_t - torch.eye(D_latent, device=R_k_t.device)).pow(2).sum().sqrt().item())
#                 t_norms.append(t_k_t.norm().item())
#             R_norms_t = torch.tensor(R_norms)
#             t_norms_t = torch.tensor(t_norms)
#             print(f"[ALIGN-TRANSFORMS] iter={it+1} "
#                 f"R_dev_from_I: p50={R_norms_t.quantile(0.5).item():.3f} "
#                 f"p90={R_norms_t.quantile(0.9).item():.3f} "
#                 f"t_norm: p50={t_norms_t.quantile(0.5).item():.3f} "
#                 f"p90={t_norms_t.quantile(0.9).item():.3f}")
            
#         # ===================================================================
#         # DIAGNOSTIC C: STITCHING QUALITY
#         # ===================================================================
#         if DEBUG_FLAG and it == 0:  # Only on first iteration
#             print(f"\n[DIAGNOSTIC C] Stitching Analysis (Iteration {it+1}):")
            
#             # C2: Per-cell multi-patch disagreement
#             sum_x = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
#             sum_x2 = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
#             count = torch.zeros(n_sc, 1, dtype=torch.float32, device=device)
            
#             for k_idx in range(K):
#                 S_k = patch_indices[k_idx].to(device)
#                 V_k = patch_coords[k_idx].to(device)
#                 R_k = R_list[k_idx].to(device)
#                 t_k = t_list[k_idx].to(device)
#                 s_k = s_list[k_idx].to(device)
                
#                 X_hat = s_k * (V_k @ R_k.T) + t_k
                
#                 sum_x.index_add_(0, S_k, X_hat)
#                 sum_x2.index_add_(0, S_k, X_hat**2)
#                 count.index_add_(0, S_k, torch.ones(len(S_k), 1, device=device))
            
#             mean = sum_x / count.clamp_min(1)
#             var = (sum_x2 / count.clamp_min(1)) - mean**2
#             cell_var = var.mean(dim=1).cpu().numpy()
            
#             print(f"  Per-cell disagreement:")
#             print(f"    p50: {np.median(cell_var):.4f}")
#             print(f"    p90: {np.percentile(cell_var, 90):.4f}")
#             print(f"    p95: {np.percentile(cell_var, 95):.4f}")
            
#             if np.median(cell_var) > 0.5:
#                 print("  ⚠️  HIGH VARIANCE: Patches strongly disagree → STITCHING PROBLEM")
            
#             # Plot
#             import matplotlib.pyplot as plt
#             plt.figure(figsize=(10, 4))
#             plt.hist(cell_var, bins=50, edgecolor='black', alpha=0.7, color='purple')
#             plt.axvline(np.median(cell_var), color='red', linestyle='--', 
#                        label=f'Median={np.median(cell_var):.3f}')
#             plt.xlabel('Per-cell Variance')
#             plt.ylabel('Count')
#             plt.title('Multi-Patch Coordinate Disagreement')
#             plt.legend()
#             plt.grid(alpha=0.3)
#             plt.tight_layout()
#             plt.show()

#         # ======================================================================
#         # Step B: update X from all patch transforms (centrality-weighted)
#         # ======================================================================
#         device_X = X_global.device

#         new_X = torch.zeros_like(X_global)
#         W_X = torch.zeros(n_sc, 1, dtype=torch.float32, device=device_X)

#         for k in range(K):
#             S_k = patch_indices[k].to(device_X)          # (m_k,)
#             V_k = patch_coords[k].to(device_X)           # (m_k, D)
#             R_k = R_list[k].to(device_X)                 # (D, D)
#             t_k = t_list[k].to(device_X)                 # (D,)
#             s_k = s_list[k].to(device_X)                 # scalar

#             # Centrality weights in local patch coordinates
#             center_k = V_k.mean(dim=0, keepdim=True)     # (1, D)
#             dists = torch.norm(V_k - center_k, dim=1, keepdim=True)   # (m_k, 1)
#             max_d = dists.max().clamp_min(1e-6)
#             weights_k = 1.0 - (dists / (max_d * 1.2))
#             weights_k = weights_k.clamp(min=0.01)        # (m_k, 1)

#             # Transformed patch in global frame using this patch's scale
#             X_hat_k = s_k * (V_k @ R_k.T) + t_k     # (m_k, D)

#             # Weighted accumulation
#             new_X.index_add_(0, S_k, X_hat_k * weights_k)
#             W_X.index_add_(0, S_k, weights_k)

#         # Finish Step B: normalize and recenter
#         mask_seen2 = W_X.squeeze(-1) > 0
#         new_X[mask_seen2] /= W_X[mask_seen2]
#         # Cells never hit by any patch: keep previous
#         new_X[~mask_seen2] = X_global[~mask_seen2]


#         new_X = new_X - new_X.mean(dim=0, keepdim=True)
        
#         # Enforce target scale at every iteration (data-driven)
#         rms_current = new_X.pow(2).mean().sqrt()
#         scale_correction = rms_target / (rms_current + 1e-8)
#         new_X = new_X * scale_correction
        
#         X_global = new_X

#         if DEBUG_FLAG:
#             rms_new = new_X.pow(2).mean().sqrt().item()
#             print(f"[new ALIGN] iter={it + 1} coords_rms={rms_new:.3f} (global scale)")

#         # DEBUG 4: Patch consistency after alignment iteration
#         patch_fit_errs = []
#         for k_check in range(K):
#             S_k_check = patch_indices[k_check].to(device_X)
#             V_k_check = patch_coords[k_check].to(device_X)
#             X_k_check = X_global[S_k_check]

#             # Compare distances within patch before/after stitching
#             D_V = torch.cdist(V_k_check, V_k_check)
#             D_X = torch.cdist(X_k_check, X_k_check)
#             # normalize by RMS to ignore global scale
#             D_V = D_V / (D_V.pow(2).mean().sqrt().clamp_min(1e-6))
#             D_X = D_X / (D_X.pow(2).mean().sqrt().clamp_min(1e-6))
#             err = (D_V - D_X).abs().mean().item()
#             patch_fit_errs.append(err)

#         patch_fit_errs_t = torch.tensor(patch_fit_errs)
#         print(f"[ALIGN-CHECK] iter={it+1} patch dist mismatch: "
#               f"p10={patch_fit_errs_t.quantile(0.10).item():.4e} "
#               f"p50={patch_fit_errs_t.quantile(0.50).item():.4e} "
#               f"p90={patch_fit_errs_t.quantile(0.90).item():.4e} "
#               f"max={patch_fit_errs_t.max().item():.4e}")

#     # ===================================================================
#     # [PATCH-KNN-VS-GT] POST-STITCH: Compare final global coords to GT
#     # ===================================================================
#     if debug_knn and gt_coords is not None:
#         print("\n" + "="*70)
#         print("[PATCH-KNN-VS-GT] POST-STITCH ANALYSIS")
#         print("="*70)
        
#         with torch.no_grad():
#             gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
#             # Use subset for efficiency
#             M = min(n_sc, debug_global_subset)
#             subset_idx = torch.randperm(n_sc)[:M]
            
#             X_subset = X_global[subset_idx].float()
#             gt_subset = gt_coords_t[subset_idx]
            
#             # Compute coverage per cell
#             cover_counts = torch.tensor([len(memberships[i]) for i in subset_idx.tolist()], 
#                                         dtype=torch.float32, device=device)
            
#             for cov_thresh, label in [(2, "coverage>=2"), (1, "coverage==1")]:
#                 if cov_thresh == 1:
#                     mask = (cover_counts == 1)
#                 else:
#                     mask = (cover_counts >= cov_thresh)
                
#                 n_in_split = mask.sum().item()
#                 if n_in_split < 50:
#                     print(f"  [{label}] n={n_in_split} - too few for analysis")
#                     continue
                
#                 X_split = X_subset[mask]
#                 gt_split = gt_subset[mask]
                
#                 for k_val in debug_k_list:
#                     if n_in_split > k_val + 1:
#                         knn_pred, _ = _knn_indices_dists(X_split, k_val)
#                         knn_gt, _ = _knn_indices_dists(gt_split, k_val)
#                         overlap = _knn_overlap_score(knn_pred, knn_gt)
                        
#                         print(f"  [PATCH-KNN-VS-GT][POST-STITCH] {label} n={n_in_split} "
#                               f"k={k_val} mean={overlap.mean().item():.3f} "
#                               f"p50={overlap.median().item():.3f}")

#     # ===================================================================
#     # [NN-GAP] Analysis: Why Spearman looks fine but kNN is bad
#     # ===================================================================
#     if debug_knn and gt_coords is not None:
#         print("\n" + "="*70)
#         print("[NN-GAP] ANALYSIS")
#         print("="*70)
        
#         with torch.no_grad():
#             gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
#             # Use subset
#             M = min(n_sc, debug_global_subset)
#             subset_idx = torch.randperm(n_sc)[:M]
            
#             X_subset = X_global[subset_idx].float()
#             gt_subset = gt_coords_t[subset_idx]
            
#             # Compute NN-gap for GT and Pred
#             gap_gt, gap_ratio_gt = _nn_gap_stats(gt_subset, debug_gap_k)
#             gap_pred, gap_ratio_pred = _nn_gap_stats(X_subset, debug_gap_k)
            
#             print(f"  [NN-GAP] GT gap_ratio({debug_gap_k}->{debug_gap_k+1}): "
#                   f"p10={gap_ratio_gt.quantile(0.1).item():.4f} "
#                   f"p50={gap_ratio_gt.quantile(0.5).item():.4f} "
#                   f"p90={gap_ratio_gt.quantile(0.9).item():.4f}")
            
#             print(f"  [NN-GAP] PR gap_ratio({debug_gap_k}->{debug_gap_k+1}): "
#                   f"p10={gap_ratio_pred.quantile(0.1).item():.4f} "
#                   f"p50={gap_ratio_pred.quantile(0.5).item():.4f} "
#                   f"p90={gap_ratio_pred.quantile(0.9).item():.4f}")
            
#             # Interpretation
#             if gap_ratio_gt.median() < 0.05:
#                 print("  ⚠️ GT has very small gap_ratio → kNN@k is inherently fragile (near-ties)")
#             if gap_ratio_pred.median() < gap_ratio_gt.median() * 0.5:
#                 print("  ⚠️ Pred gap_ratio << GT → predicted space is 'flattened' (scrambles neighbors)")
        
#         print("="*70 + "\n")

#     # ===================================================================
#     # [LOCAL-DENSITY-RATIO] Compare local radius in pred vs GT
#     # ===================================================================
#     if debug_knn and gt_coords is not None:
#         print("\n" + "="*70)
#         print("[LOCAL-DENSITY-RATIO] ANALYSIS")
#         print("="*70)
        
#         with torch.no_grad():
#             gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
#             # Use subset
#             M = min(n_sc, debug_global_subset)
#             subset_idx = torch.randperm(n_sc)[:M]
            
#             X_subset = X_global[subset_idx].float()
#             gt_subset = gt_coords_t[subset_idx]
            
#             # Compute distance to k-th neighbor in both spaces
#             for k_density in [10, 20]:
#                 # GT: distance to k-th neighbor
#                 D_gt = torch.cdist(gt_subset, gt_subset)
#                 D_gt.fill_diagonal_(float('inf'))
#                 d_gt_sorted, _ = D_gt.topk(k_density, largest=False, dim=1)
#                 r_gt_k = d_gt_sorted[:, -1]  # distance to k-th neighbor
                
#                 # Pred: distance to k-th neighbor
#                 D_pr = torch.cdist(X_subset, X_subset)
#                 D_pr.fill_diagonal_(float('inf'))
#                 d_pr_sorted, _ = D_pr.topk(k_density, largest=False, dim=1)
#                 r_pr_k = d_pr_sorted[:, -1]
                
#                 # Ratio
#                 ratio = r_pr_k / r_gt_k.clamp(min=1e-8)
                
#                 print(f"  [LOCAL-DENSITY-RATIO] k={k_density}")
#                 print(f"    r_gt({k_density}): p50={r_gt_k.median().item():.4f}")
#                 print(f"    r_pr({k_density}): p50={r_pr_k.median().item():.4f}")
#                 print(f"    ratio r_pr/r_gt: p10={ratio.quantile(0.1).item():.2f} "
#                       f"p50={ratio.median().item():.2f} "
#                       f"p90={ratio.quantile(0.9).item():.2f}")
            
#             # Interpretation
#             ratio_median = ratio.median().item()
#             ratio_spread = ratio.quantile(0.9).item() - ratio.quantile(0.1).item()
            
#             if ratio_spread > 2.0:
#                 print("  → ⚠️ HIGH SPREAD: Local density/scale varies wildly → kills local-edge correlation")
#             if ratio_median > 2.0 or ratio_median < 0.5:
#                 print(f"  → ⚠️ SHIFTED: Predicted local scale is {'expanded' if ratio_median > 1 else 'compressed'}")
        
#         print("="*70 + "\n")


#     # ------------------------------------------------------------------
#     # 6) Compute EDM and optional ST-scale alignment
#     # ------------------------------------------------------------------
#     if DEBUG_FLAG:
#         print("\n[GLOBAL] Computing EDM from global coordinates...")

#     X_full = X_global
#     X_full = X_full - X_full.mean(dim=0, keepdim=True)

#     Xd = X_full.to(device)
#     D = torch.cdist(Xd, Xd)
#     D_edm = uet.edm_project(D).detach().cpu()

#     if target_st_p95 is not None:
#         N = D_edm.shape[0]
#         iu_s, ju_s = torch.triu_indices(N, N, 1, device=D_edm.device)
#         D_vec = D_edm[iu_s, ju_s]
#         current_p95 = torch.quantile(D_vec, 0.95).clamp_min(1e-6)
#         scale_factor = (target_st_p95 / current_p95).clamp(0.5, 4.0)
#         D_edm = D_edm * scale_factor
#         if DEBUG_FLAG:
#             print(f"[SCALE] current_p95={current_p95:.3f} "
#                   f"target_p95={target_st_p95:.3f} "
#                   f"scale={scale_factor:.3f}")

#     result: Dict[str, torch.Tensor] = {"D_edm": D_edm}

#     # ------------------------------------------------------------------
#     # DEBUG: Save final EDM
#     # ------------------------------------------------------------------
#     if DEBUG_FLAG:
#         torch.save(
#             {
#                 "D_edm": D_edm.cpu(),
#             },
#             f"debug_final_edm_seed.pt",
#         )
#         print(f"[DEBUG] Saved final EDM to debug_final_edm_seed.pt")

#     # ------------------------------------------------------------------
#     # 7) Debug stats and optional 2D coords
#     # ------------------------------------------------------------------
#     if DEBUG_FLAG:
#         N = D_edm.shape[0]
#         total_pairs = N * (N - 1) // 2
#         print(f"\n[GLOBAL] N={N} (total_pairs={total_pairs})")

#         MAX_SAMPLES = 1_000_000
#         if total_pairs <= MAX_SAMPLES:
#             iu_all, ju_all = torch.triu_indices(N, N, 1, device=D_edm.device)
#             D_sample = D_edm[iu_all, ju_all].float()
#         else:
#             k = MAX_SAMPLES
#             i = torch.randint(0, N, (int(k * 1.3),), device=D_edm.device)
#             j = torch.randint(0, N, (int(k * 1.3),), device=D_edm.device)
#             keep = i < j
#             i = i[keep][:k]
#             j = j[keep][:k]
#             D_sample = D_edm[i, j].float()
#             print(f"[GLOBAL] (sampled {len(D_sample)} pairs for stats)")

#         print(f"[GLOBAL] dist: "
#               f"p50={D_sample.quantile(0.50):.3f} "
#               f"p90={D_sample.quantile(0.90):.3f} "
#               f"p99={D_sample.quantile(0.99):.3f} "
#               f"max={D_sample.max():.3f}")

#         coords_rms = X_full.pow(2).mean().sqrt().item()
#         print(f"[GLOBAL] coords_rms={coords_rms:.3f}")

#         cov = torch.cov(X_full.float().T)
#         eigs = torch.linalg.eigvalsh(cov)
#         ratio = float(eigs.max() / (eigs.min().clamp(min=1e-8)))
#         print(f"[GLOBAL] coord_cov eigs: "
#               f"min={eigs.min():.3e} "
#               f"max={eigs.max():.3e} "
#               f"ratio={ratio:.1f}")

#     if return_coords:
#         n = D_edm.shape[0]
#         Jn = torch.eye(n) - torch.ones(n, n) / n
#         B = -0.5 * (Jn @ (D_edm**2) @ Jn)
#         coords = uet.classical_mds(B.to(device), d_out=2).detach().cpu()
#         coords_canon = uet.canonicalize_coords(coords).detach().cpu()
#         result["coords"] = coords
#         result["coords_canon"] = coords_canon

#     # Cleanup GPU tensors to allow process exit
#     # Cleanup GPU tensors to allow process exit
#     del Xd, D, X_full
#     # Don't delete X_global yet if we need it for two-pass
#     if not (two_pass and _pass_number == 1):
#         del Z_all, patch_indices, memberships, patch_coords

    
#     if "cuda" in device:
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()
#     gc.collect()

#     if DEBUG_FLAG:
#         pass_label = f"[PASS{_pass_number}]" if two_pass else ""
#         print("=" * 72)
#         print(f"STAGE D (PATCH-BASED) COMPLETE {pass_label}")
#         print("=" * 72 + "\n")

#     # ===================================================================
#     # TWO-PASS MODE: If this is pass 1 and two_pass=True, run pass 2
#     # ===================================================================
#     if two_pass and _pass_number == 1:
#         print("\n" + "=" * 72)
#         print("[TWO-PASS] Pass 1 complete. Rebuilding patches from predicted geometry...")
#         print("=" * 72 + "\n")
        
#         # Get coordinates from pass 1 (use X_global before cleanup)
#         # We need to NOT delete X_global yet
#         coords_pass1 = X_global.detach().cpu()
        
#         # Run pass 2 with geometry-based patches
#         result_pass2 = sample_sc_edm_patchwise(
#             sc_gene_expr=sc_gene_expr,
#             encoder=encoder,
#             context_encoder=context_encoder,
#             score_net=score_net,
#             generator=generator,
#             sigma_data=sigma_data,
#             target_st_p95=target_st_p95,
#             n_timesteps_sample=n_timesteps_sample,
#             sigma_min=sigma_min,
#             sigma_max=sigma_max,
#             guidance_scale=guidance_scale,
#             eta=eta,
#             device=device,
#             patch_size=patch_size,
#             coverage_per_cell=coverage_per_cell,
#             n_align_iters=n_align_iters,
#             return_coords=return_coords,
#             DEBUG_FLAG=DEBUG_FLAG,
#             DEBUG_EVERY=DEBUG_EVERY,
#             fixed_patch_graph=None,  # Don't use fixed graph - build from geometry
#             coral_params=coral_params,
#             gt_coords=gt_coords,
#             debug_knn=debug_knn,
#             debug_max_patches=debug_max_patches,
#             debug_k_list=debug_k_list,
#             debug_global_subset=debug_global_subset,
#             debug_gap_k=debug_gap_k,
#             two_pass=True,
#             _pass_number=2,
#             _coords_from_pass1=coords_pass1,
#         )
        
#         # Return pass 2 results
#         return result_pass2

#     return result



def sample_sc_edm_patchwise(
    sc_gene_expr: torch.Tensor,
    encoder: "SharedEncoder",
    context_encoder: "SetEncoderContext",
    score_net: "DiffusionScoreNet",
    generator: "GeometryGenerator",
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
    coral_params: Optional[Dict] = None,
    # --- DEBUG KNN ARGS ---
    gt_coords: Optional[torch.Tensor] = None,
    debug_knn: bool = False,
    debug_max_patches: int = 20,
    debug_k_list: Tuple[int, int] = (10, 20),
    debug_global_subset: int = 4096,
    debug_gap_k: int = 10,
    # --- ANCHOR-CHANNEL DIAGNOSTIC ---
    anchor_channel_sensitivity_diag: bool = False,
    # --- TWO-PASS MODE ---
    two_pass: bool = False,
    _pass_number: int = 1,  # Internal: which pass are we on (1 or 2)
    _coords_from_pass1: Optional[torch.Tensor] = None,  # Internal: coords from pass 1 for rebuilding patches
    # --- ST-STYLE STOCHASTIC PATCH SAMPLING ---
    pool_mult: float = 4.0,
    stochastic_tau: float = 1.0,
    tau_mode: str = "adaptive_median",
    ensure_connected: bool = True,
    # --- MERGE MODE (Test 2 ablation) ---
    merge_mode: str = "mean",  # "mean", "median", "geomedian", "best_patch"
    # --- ALIGNMENT CONSTRAINTS ---
    align_freeze_scale: bool = False,  # If True, force s_k = 1 (rigid Procrustes)
    align_scale_clamp: Tuple[float, float] = (0.8, 1.2),  # Clamp scale to this range
    # --- POST-STITCH LOCAL DENSITY REFINEMENT ---
    local_refine: bool = False,
    local_refine_steps: int = 100,
    local_refine_lr: float = 0.01,
    local_refine_anchor_weight: float = 0.1,
    # --- DGSO: Distance-Graph Stitch Optimization ---
    # --- DGSO-v2: Distance-Graph Stitch Optimization ---
    enable_dgso: bool = False,
    dgso_k_edge: int = 15,
    dgso_iters: int = 1000,
    dgso_lr: float = 1e-2,
    dgso_batch_size: int = 100000,
    dgso_huber_delta: float = 0.1,
    dgso_anchor_lambda: float = 1.0,
    dgso_log_every: int = 100,
    # --- DGSO-v2 NEW PARAMS ---
    dgso_m_min: int = 3,
    dgso_tau_spread: float = 0.30,
    dgso_spread_penalty_alpha: float = 10.0,
    dgso_dist_band: Tuple[float, float] = (0.05, 0.95),
    dgso_radius_lambda: float = 0.1,
    dgso_two_phase: bool = True,
    dgso_phase1_iters: int = 200,
    dgso_phase1_anchor_mult: float = 10.0,
    coldstart_diag: bool = False,
    # --- NEW DEBUG FLAGS (no behavior change) ---
    debug_oracle_gt_stitch: bool = False,
    debug_incremental_stitch_curve: bool = False,
    debug_overlap_postcheck: bool = False,
    debug_cycle_closure: bool = False,
    debug_scale_compression: bool = False,
    # --- DEBUG/ABLATION FLAGS ---
    debug_gen_vs_noise: bool = False,
    ablate_use_generator_init: bool = False,
    ablate_use_pure_noise_init: bool = False,

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

    # ===================================================================
    # DEBUG KNN HELPERS (only used if debug_knn=True and gt_coords provided)
    # ===================================================================
    def _knn_indices_dists(coords, k):
        """Compute kNN indices and distances for a set of coords (n, d)."""
        D = torch.cdist(coords, coords)
        D.fill_diagonal_(float('inf'))
        dists, idx = D.topk(k, largest=False, dim=1)
        return idx, dists  # (n, k), (n, k)


    def _knn_overlap_score(idx_a, idx_b):
        """Compute per-point kNN overlap fraction between two index tensors."""
        # idx_a, idx_b: (n, k)
        n, k = idx_a.shape
        overlaps = []
        for i in range(n):
            set_a = set(idx_a[i].tolist())
            set_b = set(idx_b[i].tolist())
            overlap = len(set_a & set_b) / k
            overlaps.append(overlap)
        return torch.tensor(overlaps)  # (n,)

    def _forward_edm_self_cond(
        score_net,
        x_t,
        sigma_b,
        H_ctx,
        mask_k,
        sigma_data,
        center_mask=None,
    ):
        x0_pred_0 = score_net.forward_edm(
            x_t,
            sigma_b,
            H_ctx,
            mask_k,
            sigma_data,
            self_cond=None,
            center_mask=center_mask,
        )
        x0_pred = score_net.forward_edm(
            x_t,
            sigma_b,
            H_ctx,
            mask_k,
            sigma_data,
            self_cond=x0_pred_0.detach(),
            center_mask=center_mask,
        )
        return x0_pred

    def _nn_gap_stats(coords, k_gap):
        """Compute NN gap statistics: gap = d[k+1] - d[k], gap_ratio = gap / d[k]."""
        D = torch.cdist(coords, coords)
        D.fill_diagonal_(float('inf'))
        dists, _ = D.topk(k_gap + 1, largest=False, dim=1)  # (n, k+1)
        d_k = dists[:, k_gap - 1]      # distance to k-th neighbor (0-indexed: k-1)
        d_k1 = dists[:, k_gap]         # distance to (k+1)-th neighbor
        gap = d_k1 - d_k
        gap_ratio = gap / d_k.clamp(min=1e-8)
        return gap, gap_ratio  # (n,), (n,)


    def _local_edge_spearman(pred_coords, gt_coords, gt_knn_idx):
        """
        Compute Spearman correlation on GT kNN edges only.
        pred_coords: (n, D_pred)
        gt_coords: (n, D_gt)  
        gt_knn_idx: (n, k) - GT kNN indices
        Returns: Spearman correlation (scalar)
        """
        from scipy.stats import spearmanr

        
        n, k = gt_knn_idx.shape
        src = np.repeat(np.arange(n), k)
        dst = gt_knn_idx.reshape(-1)
        keep = (dst >= 0) & (dst < n) & (dst != src)
        src, dst = src[keep], dst[keep]
        
        if len(src) < 10:
            return float('nan')
        
        # Convert to numpy if needed
        if torch.is_tensor(pred_coords):
            pred_np = pred_coords.cpu().numpy()
        else:
            pred_np = pred_coords
        if torch.is_tensor(gt_coords):
            gt_np = gt_coords.cpu().numpy()
        else:
            gt_np = gt_coords
        
        d_gt = np.linalg.norm(gt_np[src] - gt_np[dst], axis=1)
        d_pr = np.linalg.norm(pred_np[src] - pred_np[dst], axis=1)
        
        if np.std(d_gt) < 1e-12 or np.std(d_pr) < 1e-12:
            return float('nan')
        
        return spearmanr(d_gt, d_pr).correlation


    def build_patches_from_coords(coords, n_cells, patch_size, coverage_per_cell, min_overlap=20):
        """
        Build patch graph using kNN in coordinate space (not expression space).
        
        Args:
            coords: (N, D) tensor of coordinates
            n_cells: total number of cells
            patch_size: target patch size
            coverage_per_cell: target coverage
            min_overlap: minimum overlap for connectivity
        
        Returns:
            patch_indices: List[LongTensor] - cell indices for each patch
            memberships: List[List[int]] - patch indices for each cell
        """
        
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        N = coords_np.shape[0]
        
        # Build kNN index on coords
        from sklearn.neighbors import NearestNeighbors
        k_nbrs = min(patch_size, N - 1)
        nbrs = NearestNeighbors(n_neighbors=k_nbrs, algorithm='ball_tree').fit(coords_np)
        _, nbr_idx = nbrs.kneighbors(coords_np)  # (N, k_nbrs)
        nbr_idx = torch.from_numpy(nbr_idx).long()
        
        # Estimate number of patches needed
        n_patches_est = int(math.ceil((coverage_per_cell * N) / patch_size))
        
        # Sample random centers
        centers = torch.randperm(N)[:n_patches_est].tolist()
        
        patch_indices_list: List[torch.Tensor] = []
        memberships_list: List[List[int]] = [[] for _ in range(N)]
        
        # First pass: build patches around centers
        for k, c in enumerate(centers):
            S_k = nbr_idx[c, :patch_size]
            S_k = torch.unique(S_k, sorted=False)
            patch_indices_list.append(S_k)
            for idx in S_k.tolist():
                memberships_list[idx].append(k)
        
        # Ensure every cell appears in at least one patch
        for i in range(N):
            if len(memberships_list[i]) == 0:
                k = len(patch_indices_list)
                S_k = nbr_idx[i, :patch_size]
                S_k = torch.unique(S_k, sorted=False)
                patch_indices_list.append(S_k)
                memberships_list[i].append(k)
                for idx in S_k.tolist():
                    if k not in memberships_list[idx]:
                        memberships_list[idx].append(k)
        
        # Check connectivity and add bridge patches if needed
        K = len(patch_indices_list)
        if K > 1:
            # Build overlap graph
            from collections import defaultdict
            import networkx as nx
            
            G = nx.Graph()
            G.add_nodes_from(range(K))
            
            for i in range(K):
                S_i = set(patch_indices_list[i].tolist())
                for j in range(i + 1, K):
                    S_j = set(patch_indices_list[j].tolist())
                    overlap = len(S_i & S_j)
                    if overlap >= min_overlap:
                        G.add_edge(i, j)
            
            components = list(nx.connected_components(G))
            
            # Add bridge patches if fragmented
            while len(components) > 1:
                # Find boundary cells between components
                comp1 = list(components[0])
                comp2 = list(components[1])
                
                # Get all cells in each component
                cells1 = set()
                for p_idx in comp1:
                    cells1.update(patch_indices_list[p_idx].tolist())
                cells2 = set()
                for p_idx in comp2:
                    cells2.update(patch_indices_list[p_idx].tolist())
                
                # Find closest pair of cells between components
                cells1_list = list(cells1)
                cells2_list = list(cells2)
                
                coords1 = coords_np[cells1_list]
                coords2 = coords_np[cells2_list]
                
                from scipy.spatial.distance import cdist
                D_cross = cdist(coords1, coords2)
                min_idx = np.unravel_index(D_cross.argmin(), D_cross.shape)
                bridge_cell1 = cells1_list[min_idx[0]]
                bridge_cell2 = cells2_list[min_idx[1]]
                
                # Create bridge patch centered between them
                bridge_center = (coords_np[bridge_cell1] + coords_np[bridge_cell2]) / 2
                dists = np.linalg.norm(coords_np - bridge_center, axis=1)
                bridge_cells = np.argsort(dists)[:patch_size]
                
                k_new = len(patch_indices_list)
                S_bridge = torch.from_numpy(bridge_cells).long()
                patch_indices_list.append(S_bridge)
                for idx in S_bridge.tolist():
                    if k_new not in memberships_list[idx]:
                        memberships_list[idx].append(k_new)
                
                # Rebuild graph and check again
                K = len(patch_indices_list)
                G = nx.Graph()
                G.add_nodes_from(range(K))
                
                for i in range(K):
                    S_i = set(patch_indices_list[i].tolist())
                    for j in range(i + 1, K):
                        S_j = set(patch_indices_list[j].tolist())
                        overlap = len(S_i & S_j)
                        if overlap >= min_overlap:
                            G.add_edge(i, j)
                
                components = list(nx.connected_components(G))
                
                # Safety: don't infinite loop
                if len(patch_indices_list) > n_patches_est * 3:
                    print(f"  [WARNING] Could not fully connect patch graph after adding bridges")
                    break
        
        return patch_indices_list, memberships_list


    # ===================================================================
    # ST-STYLE STOCHASTIC PATCH SAMPLER (mirrors STSetDataset core logic)
    # ===================================================================
    sampler_rank_stats = []  # Collect rank stats for Test 3 diagnostic
    
    def st_style_patch_sampler(
        center_idx: int,
        nbr_idx: torch.Tensor,  # (N, K_nbrs)
        Z_all: torch.Tensor,    # (N, h) - CPU tensor
        patch_size: int,
        pool_mult: float,
        stochastic_tau: float,
        tau_mode: str = "adaptive_median",
        track_ranks: bool = True,
    ) -> torch.Tensor:
        """
        ST-style stochastic patch sampling in embedding space.
        
        Returns:
            S_k: LongTensor of shape (~patch_size,) with unique indices
        """
        N = Z_all.shape[0]
        
        # Get candidate neighbors from precomputed index
        cand = nbr_idx[center_idx].clone()  # (K_nbrs,)
        
        # Remove center from candidates if present
        mask_not_center = cand != center_idx
        cand = cand[mask_not_center]
        
        if cand.numel() == 0:
            return torch.tensor([center_idx], dtype=torch.long)
        
        # Compute distances from center to all candidates in embedding space
        Z_center = Z_all[center_idx]  # (h,)
        Z_cand = Z_all[cand]  # (n_cand, h)
        dists = torch.norm(Z_cand - Z_center.unsqueeze(0), dim=1)  # (n_cand,)
        
        # Sort by distance
        sort_order = torch.argsort(dists)
        cand_sorted = cand[sort_order]
        dists_sorted = dists[sort_order]
        
        # Determine pool size and needed neighbors
        n_neighbors_needed = patch_size - 1
        K_pool = min(cand_sorted.numel(), int(pool_mult * patch_size))
        K_pool = max(K_pool, n_neighbors_needed)
        K_pool = min(K_pool, cand_sorted.numel())
        
        # Take the pool (sorted by distance)
        pool = cand_sorted[:K_pool]
        pool_d = dists_sorted[:K_pool]
        
        # Compute adaptive tau
        if pool_d.numel() == 0:
            tau_eff = stochastic_tau
        elif tau_mode == "adaptive_median":
            tau_eff = stochastic_tau * pool_d.median().clamp(min=1e-8)
        elif tau_mode == "adaptive_kth":
            kth_idx = min(n_neighbors_needed - 1, pool_d.numel() - 1)
            tau_eff = stochastic_tau * pool_d[kth_idx].clamp(min=1e-8)
        elif tau_mode == "adaptive_mean":
            tau_eff = stochastic_tau * pool_d.mean().clamp(min=1e-8)
        else:
            tau_eff = stochastic_tau
        
        # Sample neighbors with distance-decayed weights
        if pool.numel() <= n_neighbors_needed:
            neighbors = pool
            sampled_ranks = torch.arange(pool.numel())  # All ranks used
        else:
            weights = torch.softmax(-pool_d / tau_eff, dim=0)
            sampled_idx = torch.multinomial(weights, min(n_neighbors_needed, pool.numel()), replacement=False)
            neighbors = pool[sampled_idx]
            sampled_ranks = sampled_idx  # These are the ranks (0-indexed in sorted order)
        
        # Track rank statistics for Test 3
        if track_ranks and sampled_ranks.numel() > 0:
            sampler_rank_stats.append({
                'mean_rank': sampled_ranks.float().mean().item(),
                'max_rank': sampled_ranks.max().item(),
                'p90_rank': sampled_ranks.float().quantile(0.9).item() if sampled_ranks.numel() > 1 else sampled_ranks.float().mean().item(),
                'frac_beyond_patchsize': (sampled_ranks >= patch_size).float().mean().item(),
                'pool_size': K_pool,
                'tau_eff': tau_eff.item() if torch.is_tensor(tau_eff) else tau_eff,
            })
        
        # Combine center + sampled neighbors
        indices_core = torch.cat([
            torch.tensor([center_idx], dtype=torch.long),
            neighbors
        ])
        
        # Fill any missing with random non-included indices
        if indices_core.numel() < patch_size:
            missing = patch_size - indices_core.numel()
            all_idx = torch.arange(N, dtype=torch.long)
            mask_extra = ~torch.isin(all_idx, indices_core)
            extra_pool = all_idx[mask_extra]
            if extra_pool.numel() > 0:
                perm = torch.randperm(extra_pool.numel())
                add = extra_pool[perm[:min(missing, extra_pool.numel())]]
                indices_core = torch.cat([indices_core, add])
        
        # Shuffle to avoid positional bias
        indices_core = indices_core[torch.randperm(indices_core.numel())]
        indices_core = torch.unique(indices_core, sorted=False)
        
        return indices_core


    def ensure_patch_connectivity(
        patch_indices: List[torch.Tensor],
        memberships: List[List[int]],
        Z_all: torch.Tensor,
        n_sc: int,
        patch_size: int,
        min_overlap: int,
        pool_mult: float,
        stochastic_tau: float,
        tau_mode: str,
        nbr_idx: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Ensure patch graph is connected by adding bridge patches between components.
        Uses embedding space distances for bridging.
        """
        import networkx as nx
        
        K = len(patch_indices)
        if K <= 1:
            return patch_indices, memberships
        
        # Build overlap graph
        G = nx.Graph()
        G.add_nodes_from(range(K))
        
        for i in range(K):
            S_i = set(patch_indices[i].tolist())
            for j in range(i + 1, K):
                S_j = set(patch_indices[j].tolist())
                overlap = len(S_i & S_j)
                if overlap >= min_overlap:
                    G.add_edge(i, j)
        
        components = list(nx.connected_components(G))
        max_bridge_attempts = K  # Safety limit
        bridge_count = 0
        
        while len(components) > 1 and bridge_count < max_bridge_attempts:
            # Get cells in each of the first two components
            comp1_patches = list(components[0])
            comp2_patches = list(components[1])
            
            cells1 = set()
            for p_idx in comp1_patches:
                cells1.update(patch_indices[p_idx].tolist())
            cells2 = set()
            for p_idx in comp2_patches:
                cells2.update(patch_indices[p_idx].tolist())
            
            cells1_list = list(cells1)
            cells2_list = list(cells2)
            
            # Find closest pair of cells between components (in embedding space)
            Z1 = Z_all[cells1_list]  # (n1, h)
            Z2 = Z_all[cells2_list]  # (n2, h)
            D_cross = torch.cdist(Z1, Z2)  # (n1, n2)
            min_idx = D_cross.argmin()
            min_i, min_j = min_idx // D_cross.shape[1], min_idx % D_cross.shape[1]
            bridge_cell1 = cells1_list[min_i.item() if torch.is_tensor(min_i) else min_i]
            bridge_cell2 = cells2_list[min_j.item() if torch.is_tensor(min_j) else min_j]
            
            # Create bridge patch centered at the midpoint cell
            # Use the cell from comp1 as center (arbitrary choice)
            bridge_center = bridge_cell1
            
            # Build bridge patch using ST-style sampler
            S_bridge = st_style_patch_sampler(
                center_idx=bridge_center,
                nbr_idx=nbr_idx,
                Z_all=Z_all,
                patch_size=patch_size,
                pool_mult=pool_mult,
                stochastic_tau=stochastic_tau,
                tau_mode=tau_mode,
            )
            
            # Add bridge patch
            k_new = len(patch_indices)
            patch_indices.append(S_bridge)
            for idx in S_bridge.tolist():
                if k_new not in memberships[idx]:
                    memberships[idx].append(k_new)
            
            bridge_count += 1
            
            # Rebuild graph and check connectivity
            K = len(patch_indices)
            G = nx.Graph()
            G.add_nodes_from(range(K))
            
            for i in range(K):
                S_i = set(patch_indices[i].tolist())
                for j in range(i + 1, K):
                    S_j = set(patch_indices[j].tolist())
                    overlap = len(S_i & S_j)
                    if overlap >= min_overlap:
                        G.add_edge(i, j)
            
            components = list(nx.connected_components(G))
        
        if len(components) > 1:
            print(f"  [WARNING] Patch graph still has {len(components)} components after {bridge_count} bridge attempts")
        elif bridge_count > 0:
            print(f"  [BRIDGE] Added {bridge_count} bridge patches to ensure connectivity")
        
        return patch_indices, memberships


    print(f"[PATCHWISE] MADE CHANGES TO HOW WE DO THE PATCHES Running on device={device}, starting inference...", flush=True)


    encoder.eval()
    context_encoder.eval()
    score_net.eval()


    n_sc = sc_gene_expr.shape[0]
    D_latent = score_net.D_latent
    patch_size = int(min(patch_size, n_sc))


    if DEBUG_FLAG:
        print("\n" + "=" * 72)
        pass_label = f"[PASS{_pass_number}]" if two_pass else ""
        print(f"STAGE D — PATCH-BASED GLOBAL SC INFERENCE {pass_label}")
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

    # ===================================================================
    # [GLOBAL-KNN-STAGE] Prepare fixed subset for stagewise kNN tracking
    # ===================================================================
    global_knn_stage_subset = None
    global_knn_stage_results = {}
    
    if debug_knn and gt_coords is not None:
        gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
        M_stage = min(n_sc, 2000)  # Fixed subset size
        global_knn_stage_subset = torch.randperm(n_sc)[:M_stage]
        global_knn_stage_gt = gt_coords_t[global_knn_stage_subset]
        
        if DEBUG_FLAG:
            print(f"[GLOBAL-KNN-STAGE] Prepared fixed subset of {M_stage} cells for stagewise tracking")


    # ------------------------------------------------------------------
    # 2) Build k-NN index in Z-space for patch construction
    # ------------------------------------------------------------------
    # K_nbrs = patch_size
    # nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs)  # (N, K_nbrs)

    # ------------------------------------------------------------------
    # 2) Build k-NN index in Z-space for patch construction
    # ------------------------------------------------------------------
    # Increase K_nbrs to allow stochastic sampling from a larger pool
    K_nbrs = min(n_sc - 1, max(patch_size, int(pool_mult * patch_size)))
    nbr_idx = uet.build_topk_index(Z_all, K=K_nbrs)  # (N, K_nbrs)
    
    if DEBUG_FLAG:
        print(f"[PATCH] K_nbrs={K_nbrs} (pool_mult={pool_mult}, patch_size={patch_size})")



    # ------------------------------------------------------------------
    # 3) Define overlapping patches S_k (or reload from file)
    # ------------------------------------------------------------------
    pass_label = f"[PASS{_pass_number}]" if two_pass else ""
    
    if fixed_patch_graph is not None:
        # RELOAD existing patch graph - do NOT rebuild
        patch_indices = [p.to(torch.long) for p in fixed_patch_graph["patch_indices"]]
        memberships = fixed_patch_graph["memberships"]
        K = len(patch_indices)
        print(f"{pass_label}[PATCHWISE] Loaded fixed patch graph with {K} patches")
    elif _pass_number == 2 and _coords_from_pass1 is not None:
        # PASS 2: Build patches using geometry from pass 1
        print(f"{pass_label}[PATCHWISE] Building patches from PASS1 geometry (not expression)...")
        patch_indices, memberships = build_patches_from_coords(
            coords=_coords_from_pass1,
            n_cells=n_sc,
            patch_size=patch_size,
            coverage_per_cell=coverage_per_cell,
            min_overlap=20
        )
        K = len(patch_indices)
        print(f"{pass_label}[PATCHWISE] Built geometry-based patch graph with {K} patches")
    else:
        # PASS 1 (or single-pass): BUILD patch graph using ST-style stochastic sampling
        n_patches_est = int(math.ceil((coverage_per_cell * n_sc) / patch_size))
        centers = torch.randint(low=0, high=n_sc, size=(n_patches_est,), dtype=torch.long)

        patch_indices: List[torch.Tensor] = []
        memberships: List[List[int]] = [[] for _ in range(n_sc)]

        if DEBUG_FLAG:
            print(f"{pass_label}[PATCHWISE] Using ST-style stochastic patch sampling")
            print(f"  pool_mult={pool_mult}, stochastic_tau={stochastic_tau}, tau_mode={tau_mode}")

        # First pass: ST-style stochastic patches around centers
        for k, c in enumerate(centers.tolist()):
            S_k = st_style_patch_sampler(
                center_idx=c,
                nbr_idx=nbr_idx,
                Z_all=Z_all,
                patch_size=patch_size,
                pool_mult=pool_mult,
                stochastic_tau=stochastic_tau,
                tau_mode=tau_mode,
            )
            patch_indices.append(S_k)
            for idx in S_k.tolist():
                memberships[idx].append(k)

        # Ensure every cell appears in at least one patch
        for i in range(n_sc):
            if len(memberships[i]) == 0:
                k = len(patch_indices)
                S_k = st_style_patch_sampler(
                    center_idx=i,
                    nbr_idx=nbr_idx,
                    Z_all=Z_all,
                    patch_size=patch_size,
                    pool_mult=pool_mult,
                    stochastic_tau=stochastic_tau,
                    tau_mode=tau_mode,
                )
                patch_indices.append(S_k)
                memberships[i].append(k)
                for idx in S_k.tolist():
                    if k not in memberships[idx]:
                        memberships[idx].append(k)

        K = len(patch_indices)
        print(f"{pass_label}[PATCHWISE] Built {K} patches using ST-style stochastic sampling")

    # ===================================================================
    # [TEST1-COVER] GT NEIGHBOR CO-OCCURRENCE UPPER BOUND
    # ===================================================================
    # Goal: Measure whether patching scheme allows recovering GT kNN.
    # If median cover@k is low, global kNN cannot exceed that even with
    # perfect stitching - the true neighbors were never co-sampled.
    # ===================================================================
    # if debug_knn and gt_coords is not None:
    #     print("\n" + "="*70)
    #     print("[TEST1-COVER] GT NEIGHBOR CO-OCCURRENCE UPPER BOUND")
    #     print("="*70)
        
    #     with torch.no_grad():
    #         gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
    #         # Use subset for efficiency
    #         n_eval = min(n_sc, debug_global_subset)
    #         eval_indices = torch.randperm(n_sc)[:n_eval].tolist()
            
    #         # Build GT kNN for eval cells
    #         gt_subset = gt_coords_t[eval_indices]
    #         D_gt = torch.cdist(gt_subset, gt_subset)
    #         D_gt.fill_diagonal_(float('inf'))
            
    #         # For each k in k_list, compute GT neighbors
    #         max_k = max(debug_k_list)
    #         _, gt_knn_idx = D_gt.topk(max_k, largest=False, dim=1)  # (n_eval, max_k)
            
    #         # Map eval_indices to their positions
    #         eval_idx_to_pos = {gid: pos for pos, gid in enumerate(eval_indices)}
            
    #         # Build co-occurrence set U(i) for each eval cell
    #         # U(i) = union of all cells appearing in any patch with i
    #         cover_scores = {k_val: [] for k_val in debug_k_list}
    #         union_sizes = []
            
    #         for pos_i, gid_i in enumerate(eval_indices):
    #             # Get all patches containing cell gid_i
    #             patches_containing_i = memberships[gid_i]
                
    #             # Build U(i)
    #             U_i = set()
    #             for p_idx in patches_containing_i:
    #                 U_i.update(patch_indices[p_idx].tolist())
    #             U_i.discard(gid_i)  # Remove self
                
    #             union_sizes.append(len(U_i))
                
    #             # For each k, compute cover score
    #             for k_val in debug_k_list:
    #                 # GT neighbors of i (in eval-local indices)
    #                 gt_neighbors_local = gt_knn_idx[pos_i, :k_val].tolist()
    #                 # Map back to global indices
    #                 gt_neighbors_global = [eval_indices[loc] for loc in gt_neighbors_local]
                    
    #                 # How many GT neighbors are in U(i)?
    #                 n_covered = len(set(gt_neighbors_global) & U_i)
    #                 cover_scores[k_val].append(n_covered / k_val)
            
    #         # Print results with proper tags
    #         print(f"[TEST1-COVER] n_cells_evaluated={n_eval}")
            
    #         union_sizes_t = torch.tensor(union_sizes, dtype=torch.float32)
    #         print(f"[TEST1-COVER] union_size |U(i)|: min={union_sizes_t.min().item():.0f} "
    #               f"p50={union_sizes_t.median().item():.0f} "
    #               f"max={union_sizes_t.max().item():.0f}")
            
    #         cover_counts_t = torch.tensor([len(memberships[i]) for i in eval_indices], dtype=torch.float32)
    #         print(f"[TEST1-COVER] coverage_count: min={cover_counts_t.min().item():.0f} "
    #               f"p50={cover_counts_t.median().item():.0f} "
    #               f"max={cover_counts_t.max().item():.0f}")
            
    #         for k_val in debug_k_list:
    #             scores = torch.tensor(cover_scores[k_val])
    #             p10 = scores.quantile(0.1).item()
    #             p50 = scores.median().item()
    #             p90 = scores.quantile(0.9).item()
    #             mean_score = scores.mean().item()
                
    #             print(f"[TEST1-COVER] k={k_val}: mean={mean_score:.3f}, p10={p10:.3f}, p50={p50:.3f}, p90={p90:.3f}")
            
    #         # Interpretation
    #         k_main = debug_k_list[0]  # Usually 10
    #         main_cover = torch.tensor(cover_scores[k_main]).median().item()
            
    #         if main_cover < 0.5:
    #             print(f"\n[TEST1-COVER] ⚠️ WARNING: Median cover@{k_main} = {main_cover:.2f} < 0.5")
    #             print(f"    → HARD STRUCTURAL CEILING: Many GT neighbors never co-sampled!")
    #             print(f"    → Global kNN overlap CANNOT exceed ~{main_cover:.2f} no matter how good stitching is")
    #             print(f"    → ACTION: Increase patch_size, coverage_per_cell, or change patch construction")
    #         elif main_cover < 0.8:
    #             print(f"\n[TEST1-COVER] ⚠️ MODERATE: Median cover@{k_main} = {main_cover:.2f}")
    #             print(f"    → Some ceiling from patch design, but room for improvement via stitching")
    #         else:
    #             print(f"\n[TEST1-COVER] ✓ GOOD: Median cover@{k_main} = {main_cover:.2f} >= 0.8")
    #             print(f"    → Patch design allows most GT neighbors to be co-sampled")
    #             print(f"    → If final kNN is still low, problem is model/inference, not patch design")
        
    #     print("="*70 + "\n")


    # Ensure connectivity via bridging (if enabled)
    if ensure_connected:
        patch_indices, memberships = ensure_patch_connectivity(
            patch_indices=patch_indices,
            memberships=memberships,
            Z_all=Z_all,
            n_sc=n_sc,
            patch_size=patch_size,
            min_overlap=20,
            pool_mult=pool_mult,
            stochastic_tau=stochastic_tau,
            tau_mode=tau_mode,
            nbr_idx=nbr_idx,
        )
        K = len(patch_indices)


    # ===================================================================
    # [TEST1-COVER] GT NEIGHBOR CO-OCCURRENCE (DIRECT CO-SAMPLING) DIAGNOSTIC
    # ===================================================================
    # For each eval cell i:
    #   - compute GT kNN in the FULL dataset
    #   - compute U(i) = union of all cells co-sampled with i in any patch
    #   - cover@k(i) = |N_k^GT(i) ∩ U(i)| / k
    #
    # Note: This is a "direct evidence" diagnostic. If it's low, your patch graph
    # rarely co-samples true neighbors, so global hard-kNN will typically cap out
    # unless indirect constraints recover it.
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "=" * 70)
        print("[TEST1-COVER] GT NEIGHBOR CO-OCCURRENCE (DIRECT CO-SAMPLING)")
        print("=" * 70)

        with torch.no_grad():
            # ---- config ----
            K_list = list(debug_k_list) if isinstance(debug_k_list, (list, tuple)) else [int(debug_k_list)]
            kmax = max(K_list)

            # eval size
            M_eval = int(min(n_sc, debug_global_subset)) if 'debug_global_subset' in locals() else int(min(n_sc, 838))

            # stable eval subset if available
            if global_knn_stage_subset is not None:
                eval_idx = global_knn_stage_subset[:M_eval].to(device)
            else:
                eval_idx = torch.randperm(n_sc, device=device)[:M_eval]

            eval_idx_cpu = eval_idx.detach().cpu().tolist()

            # ---- GT coords on device ----
            gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            N_gt = gt_coords_t.shape[0]
            if N_gt != n_sc:
                print(f"[TEST1-COVER] WARNING: gt_coords has N={N_gt} but n_sc={n_sc}. Using min(N) indexing assumptions.")

            # ---- compute GT kNN for eval cells against FULL GT ----
            gt_eval = gt_coords_t[eval_idx]  # (M_eval, 2)

            # cdist (M_eval x N_gt)
            D = torch.cdist(gt_eval, gt_coords_t)  # (M_eval, N_gt)

            # mask self-distance: row r corresponds to global id eval_idx[r]
            D[torch.arange(M_eval, device=device), eval_idx] = float('inf')

            _, gt_knn_full = D.topk(kmax, largest=False, dim=1)  # (M_eval, kmax) global indices

            # ---- build fast patch lists on CPU (once) ----
            patch_lists = [p.detach().cpu().tolist() for p in patch_indices]

            cover_scores = {k: [] for k in K_list}
            union_sizes = []
            n_patches_each = []

            for row, gid in enumerate(eval_idx_cpu):
                # union of co-sampled cells with gid
                Ui = set()
                for pk in memberships[gid]:
                    Ui.update(patch_lists[pk])
                Ui.discard(gid)

                union_sizes.append(len(Ui))
                n_patches_each.append(len(memberships[gid]))

                nbrs = gt_knn_full[row].detach().cpu().tolist()  # length kmax
                for k in K_list:
                    n_cov = 0
                    for nb in nbrs[:k]:
                        if nb in Ui:
                            n_cov += 1
                    cover_scores[k].append(n_cov / float(k))

            # ---- summarize ----
            union_t = torch.tensor(union_sizes, dtype=torch.float32)
            covcount_t = torch.tensor(n_patches_each, dtype=torch.float32)

            print(f"[TEST1-COVER] n_eval={M_eval}  N_gt={N_gt}  patch_size={patch_size}  coverage_per_cell={coverage_per_cell}")
            print(f"[TEST1-COVER] |U(i)| union size: min={union_t.min().item():.0f} p50={union_t.median().item():.0f} p90={union_t.quantile(0.9).item():.0f} max={union_t.max().item():.0f}")
            print(f"[TEST1-COVER] #patches containing i: min={covcount_t.min().item():.0f} p50={covcount_t.median().item():.0f} p90={covcount_t.quantile(0.9).item():.0f} max={covcount_t.max().item():.0f}")

            for k in K_list:
                s = torch.tensor(cover_scores[k], dtype=torch.float32)
                print(f"[TEST1-COVER] cover@{k}: mean={s.mean().item():.3f} p10={s.quantile(0.1).item():.3f} p50={s.median().item():.3f} p90={s.quantile(0.9).item():.3f}")

        print("=" * 70 + "\n")


        # ===================================================================
        # DIAGNOSTIC A: PATCH GRAPH ANALYSIS
        # ===================================================================
        if DEBUG_FLAG:
            print("\n" + "="*70)
            print(f"DIAGNOSTIC A: PATCH GRAPH ANALYSIS {pass_label}")
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
            axes[0].set_title(f'Patch Overlap Distribution {pass_label}')
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
            axes[1].set_title(f'Patch Graph Components (n={stats["n_components"]}) {pass_label}')
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




    # ===================================================================
    # DEBUG: Precompute GT global kNN for closure/coverage diagnostics
    # ===================================================================
    gt_knn_global = None
    gt_subset_indices = None
    if debug_knn and gt_coords is not None:
        gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
        N_gt = gt_coords_t.shape[0]
        M = min(N_gt, debug_global_subset)
        
        # Random subset for efficiency
        gt_subset_indices = torch.randperm(N_gt)[:M]
        gt_coords_subset = gt_coords_t[gt_subset_indices]
        
        # Compute kNN on subset
        k_closure = max(debug_k_list)
        gt_knn_global, _ = _knn_indices_dists(gt_coords_subset, k_closure)
        
        print(f"\n[DEBUG-KNN] Precomputed GT kNN on subset of {M} cells (k={k_closure})")


    # EDM Karras sigma schedule - use refinement sigma range
    # sigma_max should match training sigma_refine_max (typically 1.0 * sigma_data)
    sigma_refine_max = min(sigma_max, 20.0 * sigma_data)  # Match training
    sigmas = uet.edm_sigma_schedule(n_timesteps_sample, sigma_min, sigma_refine_max, rho=7.0, device=device)
    
    if DEBUG_FLAG:
        print(f"[SAMPLE] Refinement mode: sigma range [{sigma_min:.4f}, {sigma_refine_max:.4f}]")


    # ------------------------------------------------------------------
    # Sigma-dependent guidance schedule + CFG WNORM
    # guidance_scale is the CEILING (max effective guidance at low sigma)
    # ------------------------------------------------------------------
    def _sigma_guidance_eff(sigma_val: float, gs_ceiling: float) -> float:
        """
        Compute effective guidance based on sigma level.
        - sigma > 1.0:    guidance_eff in [0.0, 0.5] (linear ramp 0→0.5)
        - sigma in [0.3, 1.0]: ramp from 0.5 to 1.0
        - sigma < 0.3:    full guidance (gs_ceiling)
        Returns multiplier in [0, gs_ceiling].
        """
        if sigma_val > 1.0:
            # High noise: minimal guidance, ramp from 0 at sigma=5 to 0.5 at sigma=1
            # Linear: at sigma=5 → 0.0, at sigma=1 → 0.5
            frac = max(0.0, min(1.0, (5.0 - sigma_val) / 4.0))
            return 0.5 * frac * gs_ceiling
        elif sigma_val > 0.3:
            # Medium noise: ramp from 0.5 to 1.0
            # At sigma=1 → 0.5, at sigma=0.3 → 1.0
            frac = (1.0 - sigma_val) / 0.7  # 0 at sigma=1, 1 at sigma=0.3
            return (0.5 + 0.5 * frac) * gs_ceiling
        else:
            # Low noise: full guidance ceiling
            return gs_ceiling


    patch_coords: List[torch.Tensor] = []
    # Collectors for PRE-STITCH diagnostics
    prestitch_edge_spearman_list = []
    prestitch_knn_overlap_list = {k_val: [] for k_val in debug_k_list}

    if DEBUG_FLAG:
        print("\n[STEP] Sampling local geometries for patches...")
    
    # ===================================================================
    # ORIGINAL NON-ANCHORED MODE (existing code below)
    # ===================================================================
    # ========== Check if context_encoder expects anchor channel ==========
    expected_in_ctx = getattr(context_encoder, "input_dim", None)
    if expected_in_ctx is None and hasattr(context_encoder, 'input_proj'):
        expected_in_ctx = context_encoder.input_proj.in_features
    
    with torch.no_grad():
        for k in tqdm(range(K), desc="Sampling patches"):
            S_k = patch_indices[k]

            m_k = S_k.numel()
            Z_k = Z_all[S_k].to(device)         # (m_k, h)

            Z_k_batched = Z_k.unsqueeze(0)      # (1, m_k, h)
            mask_k = torch.ones(1, m_k, dtype=torch.bool, device=device)
            
            # ========== Append zero anchor channel if context_encoder expects it ==========
            if expected_in_ctx is not None and expected_in_ctx == Z_k_batched.shape[-1] + 1:
                zeros_anchor = torch.zeros(1, m_k, 1, device=device, dtype=Z_k_batched.dtype)
                Z_k_batched = torch.cat([Z_k_batched, zeros_anchor], dim=-1)
                if k == 0:
                    print(f"[SAMPLE] Appended zero anchor channel, Z_k shape={Z_k_batched.shape}")
            
            # ========== Apply Z layer normalization (matches training) ==========
            Z_k_batched = apply_z_ln(Z_k_batched, context_encoder)
            
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


            # Start from generator proposal + noise (refinement mode)
            # Start from generator proposal + noise (refinement mode)
            V_gen = generator(H_k, mask_k)  # Generator proposal

            # Ablations (init only)
            if ablate_use_generator_init:
                if DEBUG_FLAG:
                    print("[ABLATE] ablate_use_generator_init=True → V_gen set to 0")
                V_gen = torch.zeros_like(V_gen)

            noise = torch.randn_like(V_gen) * sigmas[0]

            if ablate_use_pure_noise_init:
                if DEBUG_FLAG:
                    print("[ABLATE] ablate_use_pure_noise_init=True → init = pure noise")
                V_t = noise
            else:
                V_t = V_gen + noise

            V_t = V_t * mask_k.unsqueeze(-1).float()  # Mask out invalid positions

            # Debug prints: generator vs noise
            if debug_gen_vs_noise and DEBUG_FLAG:
                rms_gen = V_gen.pow(2).mean().sqrt().item()
                rms_noise = noise.pow(2).mean().sqrt().item()
                rms_vt = V_t.pow(2).mean().sqrt().item()
                ratio = rms_gen / (rms_noise + 1e-8)
                init_mode = "init+noise" if not ablate_use_pure_noise_init else "pure_noise"
                print(f"[GEN-VS-NOISE] mode={init_mode} rms_gen={rms_gen:.4f} rms_noise={rms_noise:.4f} rms_vt={rms_vt:.4f} ratio={ratio:.3f}")

            
            # EDM Euler + Heun sampler
            for i in range(len(sigmas) - 1):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                sigma_b = sigma.view(1)  # (B=1,)


                # x0 predictions with CFG
                # x0 predictions with CFG + SELF-CONDITIONING
                x0_c = _forward_edm_self_cond(score_net, V_t, sigma_b, H_k, mask_k, sigma_data)

                if debug_gen_vs_noise and DEBUG_FLAG and i == len(sigmas) // 2:
                    rms_x0 = x0_c.pow(2).mean().sqrt().item()
                    rms_vt_mid = V_t.pow(2).mean().sqrt().item()
                    print(f"[GEN-VS-NOISE] mid-step i={i} rms_x0={rms_x0:.4f} rms_vt={rms_vt_mid:.4f}")


                if guidance_scale != 1.0:
                    H_null = torch.zeros_like(H_k)
                    x0_u = _forward_edm_self_cond(score_net, V_t, sigma_b, H_null, mask_k, sigma_data)

                    # x0 = x0_u + guidance_scale * (x0_c - x0_u)
                    # Sigma-dependent effective guidance (guidance_scale is ceiling)
                    sigma_f = float(sigma)
                    guidance_eff = _sigma_guidance_eff(sigma_f, guidance_scale)


                    # CFG WNORM: normalize diff to prevent blowing up locality
                    diff = x0_c - x0_u
                    diff_norm = diff.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)  # (B, 1, 1)
                    x0_u_norm = x0_u.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)
                    wnorm_scale = (x0_u_norm / diff_norm).clamp(max=1.0)  # clamp to avoid amplifying


                    # Apply: x0 = x0_u + guidance_eff * wnorm_scale * diff
                    x0 = x0_u + guidance_eff * wnorm_scale * diff
                else:
                    x0 = x0_c


                # Debug on first patch
                if DEBUG_FLAG and k == 0 and i < 3:
                    if guidance_scale != 1.0:
                        du = x0_u.norm(dim=[1, 2]).mean().item()
                        dc = x0_c.norm(dim=[1, 2]).mean().item()
                        diff_mag = diff.norm(dim=[1, 2]).mean().item()
                        wn_s = wnorm_scale.mean().item()
                        print(f"  [PATCH0] i={i:3d} sigma={sigma_f:.4f} g_eff={guidance_eff:.3f} "
                              f"||x0_u||={du:.3f} ||x0_c||={dc:.3f} ||diff||={diff_mag:.3f} wnorm={wn_s:.3f}")


                # Euler step
                d = (V_t - x0) / sigma.clamp_min(1e-8)
                V_euler = V_t + (sigma_next - sigma) * d


                # Heun corrector (skip if sigma_next==0)
                if sigma_next > 0:
                    x0_next_c = _forward_edm_self_cond(score_net, V_euler, sigma_next.view(1), H_k, mask_k, sigma_data)
                    if guidance_scale != 1.0:
                        x0_next_u = _forward_edm_self_cond(score_net, V_euler, sigma_next.view(1), H_null, mask_k, sigma_data)

                        # x0_next = x0_next_u + guidance_scale * (x0_next_c - x0_next_u)
                        # Sigma-dependent effective guidance for Heun step (use sigma_next)
                        sigma_next_f = float(sigma_next)
                        guidance_eff_next = _sigma_guidance_eff(sigma_next_f, guidance_scale)


                        # CFG WNORM for Heun step
                        diff_next = x0_next_c - x0_next_u
                        diff_next_norm = diff_next.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)
                        x0_next_u_norm = x0_next_u.norm(dim=[1, 2], keepdim=True).clamp(min=1e-8)
                        wnorm_scale_next = (x0_next_u_norm / diff_next_norm).clamp(max=1.0)


                        x0_next = x0_next_u + guidance_eff_next * wnorm_scale_next * diff_next
                    else:
                        x0_next = x0_next_c


                    d2 = (V_euler - x0_next) / sigma_next.clamp_min(1e-8)
                    V_t = V_t + (sigma_next - sigma) * 0.5 * (d + d2)
                else:
                    V_t = V_euler


                # Apply mask
                V_t = V_t * mask_k.unsqueeze(-1).float()


                
                # Optional stochastic noise (if eta > 0)
                if eta > 0 and sigma_next > 0:
                    noise_scale = eta * torch.sqrt(torch.clamp(sigma_next**2 - sigma**2, min=0))
                    V_t = V_t + noise_scale * torch.randn_like(V_t)




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




            # ===================================================================
            # [PATCH-KNN-VS-GT] PRE-STITCH: kNN overlap with GT (per-patch)
            # ===================================================================
            if debug_knn and gt_coords is not None and k < debug_max_patches:
                with torch.no_grad():
                    gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
                    gt_patch = gt_coords_t[S_k]  # (m_k, 2 or D)
                    pred_patch = V_centered.float()  # (m_k, D_latent)
                    
                    for k_val in debug_k_list:
                        if m_k > k_val + 1:
                            knn_pred, _ = _knn_indices_dists(pred_patch, k_val)
                            knn_gt, _ = _knn_indices_dists(gt_patch, k_val)
                            overlap = _knn_overlap_score(knn_pred, knn_gt)
                            
                            # Collect for aggregation
                            prestitch_knn_overlap_list[k_val].append(overlap.mean().item())
                            
                            if k < 10:  # Only print first 5 patches
                                print(f"  [PATCH-KNN-VS-GT][PRE-STITCH] patch={k} n={m_k} "
                                      f"k={k_val} mean={overlap.mean().item():.3f} "
                                      f"p50={overlap.median().item():.3f}")




            # ===================================================================
            # [PATCH-KNN-CLOSURE]: How neighbor-closed is this patch vs global GT?
            # ===================================================================
            if debug_knn and gt_subset_indices is not None and k < debug_max_patches:
                with torch.no_grad():
                    # Find overlap between patch and GT subset
                    S_k_set = set(S_k.tolist())
                    subset_set = set(gt_subset_indices.tolist())
                    patch_in_subset = S_k_set & subset_set
                    
                    if len(patch_in_subset) >= 20:
                        # Map to subset-local indices
                        subset_to_local = {int(g): i for i, g in enumerate(gt_subset_indices.tolist())}
                        patch_local_in_subset = [subset_to_local[g] for g in patch_in_subset]
                        
                        closure_fracs = []
                        for k_val in [10, 20]:
                            if k_val <= gt_knn_global.shape[1]:
                                for local_idx in patch_local_in_subset:
                                    gt_neighbors = set(gt_knn_global[local_idx, :k_val].tolist())
                                    neighbors_in_patch = gt_neighbors & set(patch_local_in_subset)
                                    closure = len(neighbors_in_patch) / k_val
                                    closure_fracs.append(closure)
                        
                        if closure_fracs and k < 10:
                            closure_t = torch.tensor(closure_fracs)
                            print(f"  [PATCH-KNN-CLOSURE] patch={k} n_eval={len(patch_in_subset)} "
                                  f"closure_mean={closure_t.mean().item():.3f} "
                                  f"closure_p50={closure_t.median().item():.3f}")


            # ===================================================================
            # [PATCH-LOCAL-EDGE-CORR][PRE-STITCH]: Local edge Spearman within patch
            # ===================================================================
            if debug_knn and gt_coords is not None and k < debug_max_patches:
                with torch.no_grad():
                    gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
                    gt_patch = gt_coords_t[S_k].cpu().numpy()  # (m_k, 2)
                    pred_patch = V_centered.cpu().numpy()       # (m_k, D_latent)
                    
                    if m_k > 15:
                        # Compute GT kNN within this patch
                        from sklearn.neighbors import NearestNeighbors
                        k_edge = min(20, m_k - 1)
                        nbrs = NearestNeighbors(n_neighbors=k_edge+1, algorithm='ball_tree').fit(gt_patch)
                        _, gt_knn_patch = nbrs.kneighbors(gt_patch)
                        gt_knn_patch = gt_knn_patch[:, 1:]  # Remove self
                        
                        # Compute local-edge Spearman
                        edge_spearman = _local_edge_spearman(pred_patch, gt_patch, gt_knn_patch)
                        
                        if not np.isnan(edge_spearman):
                            prestitch_edge_spearman_list.append(edge_spearman)
                        
                        if k < 10:  # Only print first 5 patches
                            print(f"  [PATCH-LOCAL-EDGE-CORR][PRE-STITCH] patch={k} n={m_k} "
                                  f"k_edge={k_edge} spearman={edge_spearman:.3f}")




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

    
    # ===================================================================
    # ORIGINAL NON-ANCHORED MODE (existing code below)
    # ===================================================================



    # ===================================================================
    # [PRE-STITCH SUMMARY] Aggregated stats across patches
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[PRE-STITCH SUMMARY] Aggregated across patches")
        print("="*70)
        
        if prestitch_edge_spearman_list:
            es = np.array(prestitch_edge_spearman_list)
            print(f"  [PATCH-LOCAL-EDGE-CORR] n_patches={len(es)}")
            print(f"    Spearman: mean={np.mean(es):.3f} p50={np.median(es):.3f} "
                  f"p10={np.percentile(es,10):.3f} p90={np.percentile(es,90):.3f}")
            
            if np.median(es) < 0.3:
                print("    → ⚠️ LOW: Patch generator produces locally wrong geometry")
            elif np.median(es) > 0.5:
                print("    → ✓ DECENT: Patch outputs have reasonable local structure")
        
        for k_val in debug_k_list:
            if prestitch_knn_overlap_list[k_val]:
                ov = np.array(prestitch_knn_overlap_list[k_val])
                print(f"  [PATCH-KNN-VS-GT] k={k_val} n_patches={len(ov)}")
                print(f"    Overlap: mean={np.mean(ov):.3f} p50={np.median(ov):.3f} "
                      f"p10={np.percentile(ov,10):.3f} p90={np.percentile(ov,90):.3f}")
        
        print("="*70 + "\n")


    # ===================================================================
    # [TEST3-STABILITY] NEIGHBORHOOD STABILITY ACROSS PATCHES (Context Dependence)
    # ===================================================================
    # Goal: Test if generator's local neighbor ordering changes when different
    # cells are present (context dependence). Low stability means the generator
    # produces different neighborhoods for the same cell depending on context.
    # ===================================================================
    if debug_knn and gt_coords is not None and K > 1 and len(patch_coords) > 0:
        print("\n" + "="*70)
        print("[TEST3-STABILITY] NEIGHBORHOOD STABILITY ACROSS PATCHES")
        print("="*70)
        
        with torch.no_grad():
            k_base = 10  # k for neighbor comparison
            pairs_tested = 0
            cells_tested = set()
            jaccard_list = []
            overlap_list = []
            
            # Find cells appearing in >=2 patches
            multi_patch_cells = [i for i in range(n_sc) if len(memberships[i]) >= 2]
            
            # Sample up to 500 cells for efficiency
            sample_cells = multi_patch_cells[:min(500, len(multi_patch_cells))]
            
            for cell_i in sample_cells:
                patches_with_i = memberships[cell_i]
                if len(patches_with_i) < 2:
                    continue
                
                # Test up to 3 random pairs of patches for this cell
                import itertools
                patch_pairs = list(itertools.combinations(patches_with_i, 2))[:3]
                
                for p_idx, q_idx in patch_pairs:
                    # Get cell sets for each patch
                    S_p = set(patch_indices[p_idx].tolist())
                    S_q = set(patch_indices[q_idx].tolist())
                    
                    # Intersection (cells in both patches, excluding cell_i)
                    S_intersect = (S_p & S_q) - {cell_i}
                    
                    if len(S_intersect) < k_base + 5:
                        continue  # Not enough shared cells
                    
                    S_intersect_list = list(S_intersect)
                    
                    # Build position lookup for each patch
                    pos_p = {int(cid): idx for idx, cid in enumerate(patch_indices[p_idx].tolist())}
                    pos_q = {int(cid): idx for idx, cid in enumerate(patch_indices[q_idx].tolist())}
                    
                    # Position of cell_i in each patch
                    local_i_p = pos_p[cell_i]
                    local_i_q = pos_q[cell_i]
                    
                    # Get coords for cell_i and intersection cells in patch p
                    V_p = patch_coords[p_idx]  # (m_p, D_latent)
                    coord_i_p = V_p[local_i_p]  # (D_latent,)
                    
                    # Coords for intersection cells in patch p
                    local_intersect_p = [pos_p[c] for c in S_intersect_list]
                    coords_intersect_p = V_p[local_intersect_p]  # (n_intersect, D_latent)
                    
                    # Distances from cell_i to intersection cells in patch p
                    dists_p = torch.norm(coords_intersect_p - coord_i_p.unsqueeze(0), dim=1)
                    _, topk_p = dists_p.topk(k_base, largest=False)
                    neighbors_p = set([S_intersect_list[idx] for idx in topk_p.tolist()])
                    
                    # Same for patch q
                    V_q = patch_coords[q_idx]
                    coord_i_q = V_q[local_i_q]
                    local_intersect_q = [pos_q[c] for c in S_intersect_list]
                    coords_intersect_q = V_q[local_intersect_q]
                    dists_q = torch.norm(coords_intersect_q - coord_i_q.unsqueeze(0), dim=1)
                    _, topk_q = dists_q.topk(k_base, largest=False)
                    neighbors_q = set([S_intersect_list[idx] for idx in topk_q.tolist()])
                    
                    # Compute Jaccard and overlap
                    intersection = neighbors_p & neighbors_q
                    union = neighbors_p | neighbors_q
                    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                    overlap_frac = len(intersection) / k_base
                    
                    jaccard_list.append(jaccard)
                    overlap_list.append(overlap_frac)
                    pairs_tested += 1
                    cells_tested.add(cell_i)
            
            if jaccard_list:
                jaccard_t = torch.tensor(jaccard_list)
                overlap_t = torch.tensor(overlap_list)
                
                print(f"[TEST3-STABILITY] pairs_tested={pairs_tested} cells_tested={len(cells_tested)}")
                print(f"[TEST3-STABILITY] Jaccard@{k_base}: "
                      f"p10={jaccard_t.quantile(0.1).item():.3f} "
                      f"p50={jaccard_t.median().item():.3f} "
                      f"p90={jaccard_t.quantile(0.9).item():.3f}")
                print(f"[TEST3-STABILITY] overlap@{k_base}: "
                      f"p10={overlap_t.quantile(0.1).item():.3f} "
                      f"p50={overlap_t.median().item():.3f} "
                      f"p90={overlap_t.quantile(0.9).item():.3f}")
                
                # Interpretation
                median_jaccard = jaccard_t.median().item()
                if median_jaccard < 0.4:
                    print(f"\n[TEST3-STABILITY] ⚠️ LOW STABILITY: p50 Jaccard@{k_base} = {median_jaccard:.2f} < 0.4")
                    print(f"    → Generator's local ordering is CONTEXT-DEPENDENT")
                    print(f"    → Same cell gets different neighbors depending on which cells are present")
                    print(f"    → Stitching CANNOT fix this - it's a diffusion model limitation")
                elif median_jaccard < 0.6:
                    print(f"\n[TEST3-STABILITY] MODERATE: p50 Jaccard@{k_base} = {median_jaccard:.2f}")
                    print(f"    → Some context dependence, but may still stitch reasonably")
                else:
                    print(f"\n[TEST3-STABILITY] ✓ GOOD: p50 Jaccard@{k_base} = {median_jaccard:.2f} >= 0.6")
                    print(f"    → Generator is fairly consistent across contexts")
                    print(f"    → If final kNN is low, problem is likely in stitching")
            else:
                print(f"[TEST3-STABILITY] ⚠️ Could not test any pairs (not enough overlap)")
        
        print("="*70 + "\n")

    # ------------------------------------------------------------------
    # DEBUG: Save patch coords (AFTER all patches sampled, BEFORE alignment)
    # ------------------------------------------------------------------
    if DEBUG_FLAG and len(patch_coords) > 0:
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
    if DEBUG_FLAG and len(patch_coords) > 0:
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
            if i >= len(patch_coords) or j >= len(patch_coords):
                continue

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
    if len(patch_coords) > 0:
        patch_rms_list = torch.tensor([pc.pow(2).mean().sqrt().item() for pc in patch_coords])
        rms_target = patch_rms_list.median().item()
    else:
        rms_target = 1.0  # Default value when patch_coords is empty


    if DEBUG_FLAG and len(patch_coords) > 0:
        print(f"[ALIGN] Target RMS for global space: {rms_target:.3f} (median of patches)")




    # DEBUG 3: RMS distribution across all patches
    if DEBUG_FLAG and len(patch_coords) > 0:
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
    if DEBUG_FLAG and K > 1 and len(patch_coords) > 0:
        print("\n" + "="*60)
        print("PATCH OVERLAP DIAGNOSTIC (pre-alignment)")
        print("="*60)

        # Find pairs of overlapping patches
        overlap_corrs = []

        for k1 in range(min(10, K)):  # Check first 10 patches
            if k1 >= len(patch_coords):
                continue
            S_k1 = set(patch_indices[k1].cpu().tolist())
            V_k1 = patch_coords[k1].cpu()  # (m_k1, D)

            for k2 in range(k1+1, min(k1+5, K)):  # Check next 4 patches
                if k2 >= len(patch_coords):
                    continue
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
    if DEBUG_FLAG and hasattr(sc_gene_expr, 'gt_coords') and len(patch_coords) > 0:
        from scipy.spatial.distance import cdist
        from scipy.stats import pearsonr

        gt_coords = sc_gene_expr.gt_coords  # Assume passed in somehow
        patch_local_corrs = []

        for k in range(K):
            if k >= len(patch_coords):
                continue
            S_k_np = patch_indices[k].cpu().numpy()
            V_k = patch_coords[k].cpu().numpy()  # (m_k, D_latent)
            gt_k = gt_coords[S_k_np]  # (m_k, 2)

            D_pred = cdist(V_k, V_k)
            D_gt = cdist(gt_k, gt_k)

            tri = np.triu_indices(len(S_k_np), k=1)
            if len(tri[0]) > 0:
                r = pearsonr(D_pred[tri], D_gt[tri])[0]
                patch_local_corrs.append(r)

        if len(patch_local_corrs) > 0:
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


    if DEBUG_FLAG and len(patch_coords) > 0:
        print("\n[ALIGN] Stitching with centrality weighting...")


    for k in range(K):
        if k >= len(patch_coords):
            continue
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
    if DEBUG_FLAG and len(patch_coords) > 0:
        print(f"[ALIGN] Init X_global: rms_raw={rms_init:.3f} "
            f"→ rescaled to {rms_final:.3f} (target={rms_target:.3f}, scale={scale_factor:.3f})")

    # ===================================================================
    # [GLOBAL-KNN-STAGE] Stage 1: After initial weighted mean (before PGSO)
    # ===================================================================
    if debug_knn and gt_coords is not None and global_knn_stage_subset is not None:
        with torch.no_grad():
            X_init_subset = X_global[global_knn_stage_subset, :2].float()  # Use 2D
            
            init_knn_scores = {}
            for k_val in debug_k_list:
                if len(global_knn_stage_subset) > k_val + 1:
                    knn_pred, _ = _knn_indices_dists(X_init_subset, k_val)
                    knn_gt, _ = _knn_indices_dists(global_knn_stage_gt, k_val)
                    overlap = _knn_overlap_score(knn_pred, knn_gt)
                    init_knn_scores[k_val] = overlap.mean().item()
            
            global_knn_stage_results['init'] = init_knn_scores
            
            if DEBUG_FLAG:
                knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in init_knn_scores.items()])
                print(f"[GLOBAL-KNN-STAGE] init (pre-PGSO): {knn_str}")

    # ===================================================================
    # [TEST1] POST-ALIGN PRE-MERGE KNN DIAGNOSTIC HELPER
    # ===================================================================
    def compute_postalign_premerge_knn(
        patch_coords_aligned: List[torch.Tensor],  # List of (m_k, D) aligned coords per patch
        patch_indices: List[torch.Tensor],         # List of global indices per patch
        gt_coords: torch.Tensor,                   # (N, 2) ground truth
        k_list: Tuple[int, ...] = (10, 20),
    ) -> Dict:
        """
        Compute kNN overlap for each patch AFTER alignment but BEFORE merging.
        This pinpoints whether alignment or merge causes kNN degradation.
        """
        results = {k: [] for k in k_list}
        
        for patch_idx, (coords_aligned, indices) in enumerate(zip(patch_coords_aligned, patch_indices)):
            if coords_aligned is None or len(indices) < max(k_list) + 1:
                continue
            
            # Get GT coords for this patch
            gt_patch = gt_coords[indices]
            
            # Compute kNN in predicted coords
            D_pred = torch.cdist(coords_aligned, coords_aligned)
            D_pred.fill_diagonal_(float('inf'))
            
            # Compute kNN in GT coords
            D_gt = torch.cdist(gt_patch, gt_patch)
            D_gt.fill_diagonal_(float('inf'))
            
            for k in k_list:
                if len(indices) < k + 1:
                    continue
                _, knn_pred = D_pred.topk(k, largest=False, dim=1)
                _, knn_gt = D_gt.topk(k, largest=False, dim=1)
                
                # Compute overlap
                overlaps = []
                for i in range(len(indices)):
                    set_pred = set(knn_pred[i].tolist())
                    set_gt = set(knn_gt[i].tolist())
                    overlaps.append(len(set_pred & set_gt) / k)
                
                results[k].append(np.mean(overlaps))
        
        return results



    # ===================================================================
    # 5.2 PGSO: Pose-Graph Stitching via Global Least-Squares Solve
    # ===================================================================
    # Key insight: Instead of iteratively aligning patches to a drifting global
    # reference, we compute pairwise transforms on overlaps and solve a global
    # least-squares system that enforces loop closure constraints.
    # ===================================================================

    print(f"\n[PGSO] Starting Pose-Graph Stitching with Global Least-Squares Solve...")

    # ===================================================================
    # PGSO-A: Build overlap graph and compute pairwise Procrustes transforms
    # ===================================================================
    MIN_OVERLAP_FOR_EDGE = 30  # Minimum shared cells to consider an edge

    def weighted_procrustes_2d(X_src, X_tgt, weights=None):
        """
        Compute similarity transform (R, s, t) mapping X_src -> X_tgt.
        Convention: X_tgt ≈ s * (X_src @ R.T) + t  (row-vector convention)

        Returns: R (2,2), s (scalar), t (2,), residual (scalar)
        """
        n = X_src.shape[0]
        if weights is None:
            weights = torch.ones(n, 1, device=X_src.device)
        else:
            weights = weights.view(-1, 1)

        w_sum = weights.sum()
        mu_src = (weights * X_src).sum(dim=0, keepdim=True) / w_sum
        mu_tgt = (weights * X_tgt).sum(dim=0, keepdim=True) / w_sum

        Xc = X_src - mu_src
        Yc = X_tgt - mu_tgt

        w_sqrt = weights.sqrt()
        Xc_w = Xc * w_sqrt
        Yc_w = Yc * w_sqrt

        # Cross-covariance: C = Y.T @ X (for row-vector convention)
        C = Yc_w.T @ Xc_w  # (D, D)

        U, S_vals, Vh = torch.linalg.svd(C, full_matrices=False)
        R = U @ Vh

        # Ensure proper rotation (det = +1)
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vh

        # Scale: s = trace(S) / ||Xc_w||^2
        numer = S_vals.sum()
        denom = (Xc_w ** 2).sum().clamp_min(1e-8)
        s = numer / denom

        # Translation: t = mu_tgt - s * (mu_src @ R.T)
        t = (mu_tgt - s * (mu_src @ R.T)).squeeze(0)

        # Residual
        X_hat = s * (X_src @ R.T) + t
        residual = ((X_hat - X_tgt) ** 2).sum(dim=1).sqrt().mean()

        return R, s, t, residual

    # Project all patches to 2D via global PCA
    # ===================================================================
    # ===================================================================
    # Project all patches to 2D via GLOBAL PCA
    # (Per-patch PCA caused rotation degeneracy collapse - reverted)
    # ===================================================================
    patch_coords_2d = []
    patch_eig_ratios_2d = []
    pca_basis = None
    eigs = None

    if len(patch_coords) > 0:
        print(f"[PGSO-A] Projecting patches to 2D via global PCA...")

        # Gather all patch coords for PCA
        all_coords_for_pca = []
        for k in range(K):
            if k < len(patch_coords):
                all_coords_for_pca.append(patch_coords[k])

        if len(all_coords_for_pca) > 0:
            all_coords_cat = torch.cat(all_coords_for_pca, dim=0)  # (sum_m_k, D_latent)

            # Compute PCA basis (top 2 components)
            all_coords_centered = all_coords_cat - all_coords_cat.mean(dim=0, keepdim=True)
            cov = all_coords_centered.T @ all_coords_centered / all_coords_centered.shape[0]
            eigs, vecs = torch.linalg.eigh(cov)
            # eigh returns in ascending order, we want descending
            pca_basis = vecs[:, -2:].flip(dims=[1])  # (D_latent, 2) - top 2 eigenvectors

            # Project each patch to 2D
            for k in range(K):
                if k >= len(patch_coords):
                    continue
                V_k = patch_coords[k]  # (m_k, D_latent)
                V_k_centered = V_k - V_k.mean(dim=0, keepdim=True)
                V_k_2d = V_k_centered @ pca_basis  # (m_k, 2)
                patch_coords_2d.append(V_k_2d)

                # Compute anisotropy in 2D for this patch (used for rotation confidence)
                cov_2d = V_k_2d.T @ V_k_2d / V_k_2d.shape[0]
                eigs_2d = torch.linalg.eigvalsh(cov_2d)
                eig_ratio_2d = (eigs_2d.max() / eigs_2d.min().clamp(min=1e-8)).item()
                patch_eig_ratios_2d.append(eig_ratio_2d)

            if DEBUG_FLAG and eigs is not None:
                eig_ratio = eigs[-1] / eigs[-2] if eigs[-2] > 1e-8 else float('inf')
                print(f"[PGSO-A] Global PCA eigenvalues: top2=[{eigs[-1].item():.4f}, {eigs[-2].item():.4f}], ratio={eig_ratio:.2f}")

                # Report per-patch 2D anisotropy (for rotation confidence diagnostic)
                if len(patch_eig_ratios_2d) > 0:
                    eig_ratios_2d_t = torch.tensor(patch_eig_ratios_2d)
                    print(f"[PGSO-A] Per-patch 2D anisotropy (eig_max/eig_min):")
                    print(f"    p25={eig_ratios_2d_t.quantile(0.25).item():.1f} "
                          f"p50={eig_ratios_2d_t.median().item():.1f} "
                          f"p75={eig_ratios_2d_t.quantile(0.75).item():.1f} "
                          f"max={eig_ratios_2d_t.max().item():.1f}")
                    if eig_ratios_2d_t.median() > 10:
                        print(f"    ⚠️ HIGH ANISOTROPY: Patches are near-1D in 2D space → rotation is ill-conditioned")
    else:
        print(f"[PGSO-A] Skipping PCA projection (no patch_coords available)")

    # ===================================================================
    # DEBUG: ORACLE GT STITCH / INCREMENTAL CURVE / SCALE COMPRESSION (pre-PGSO)
    # ===================================================================
    if DEBUG_FLAG and debug_knn and gt_coords is not None:
        gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()

        # ---------- CHANGE B: Oracle GT stitch ----------
        if debug_oracle_gt_stitch and len(patch_coords_2d) > 0 and global_knn_stage_subset is not None:
            print("\n" + "=" * 70)
            print("[ORACLE-GT-STITCH] Similarity-to-GT per patch + GT-frame merge")
            print("=" * 70)

            def rigid_procrustes_2d(X_src, X_tgt, weights=None):
                n = X_src.shape[0]
                if weights is None:
                    weights = torch.ones(n, 1, device=X_src.device)
                else:
                    weights = weights.view(-1, 1)

                w_sum = weights.sum()
                mu_src = (weights * X_src).sum(dim=0, keepdim=True) / w_sum
                mu_tgt = (weights * X_tgt).sum(dim=0, keepdim=True) / w_sum

                Xc = X_src - mu_src
                Yc = X_tgt - mu_tgt

                w_sqrt = weights.sqrt()
                Xc_w = Xc * w_sqrt
                Yc_w = Yc * w_sqrt

                C = Yc_w.T @ Xc_w
                U, S_vals, Vh = torch.linalg.svd(C, full_matrices=False)
                R = U @ Vh
                if torch.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vh

                s = torch.tensor(1.0, device=X_src.device)
                t = (mu_tgt - (mu_src @ R.T)).squeeze(0)
                X_hat = (X_src @ R.T) + t
                residual = ((X_hat - X_tgt) ** 2).sum(dim=1).sqrt().mean()
                return R, s, t, residual

            X_oracle = torch.zeros(n_sc, 2, device=device)
            W_oracle = torch.zeros(n_sc, 1, device=device)

            n_test = min(K, debug_max_patches, len(patch_coords_2d))
            for k in range(n_test):
                if k >= len(patch_coords_2d):
                    continue
                S_k = patch_indices[k]
                m_k = S_k.numel()
                if m_k < 5:
                    continue

                pred_k = patch_coords_2d[k].float().to(device)  # (m_k, 2)
                gt_k = gt_coords_t[S_k]  # (m_k, 2)

                R_sim, s_sim, t_sim, res_sim = weighted_procrustes_2d(pred_k, gt_k)
                R_r, s_r, t_r, res_r = rigid_procrustes_2d(pred_k, gt_k)

                # rotation angle in degrees
                theta = torch.atan2(R_sim[1, 0], R_sim[0, 0]).item() * 180.0 / np.pi

                print(f"  [ORACLE] patch={k:03d} n={m_k:4d} "
                      f"res_sim={res_sim.item():.4f} res_rigid={res_r.item():.4f} "
                      f"s={s_sim.item():.3f} rot_deg={theta:+.1f}")

                # Ensure indices are on the same device as X_oracle / pred_gt
                S_k = patch_indices[k].to(device)

                # Apply similarity-to-GT and merge in GT frame
                pred_gt = s_sim * (pred_k @ R_sim.T) + t_sim  # (m_k, 2)

                # centrality weights (same logic as normal merge)
                center_k = pred_gt.mean(dim=0, keepdim=True)
                dists = torch.norm(pred_gt - center_k, dim=1, keepdim=True)
                max_d = dists.max().clamp_min(1e-6)
                weights_k = 1.0 - (dists / (max_d * 1.2))
                weights_k = weights_k.clamp(min=0.01)

                X_oracle.index_add_(0, S_k, pred_gt * weights_k)
                W_oracle.index_add_(0, S_k, weights_k)


            mask_seen = W_oracle.squeeze(-1) > 0
            X_oracle[mask_seen] /= W_oracle[mask_seen]
            X_oracle = X_oracle - X_oracle.mean(dim=0, keepdim=True)

            # rescale to rms_target for fair kNN comparison
            rms_oracle = X_oracle.pow(2).mean().sqrt().item()
            scale_oracle = (rms_target / (rms_oracle + 1e-8))
            X_oracle = X_oracle * scale_oracle

            # kNN on same global subset
            X_oracle_subset = X_oracle[global_knn_stage_subset, :2].float()
            oracle_knn_scores = {}
            for k_val in debug_k_list:
                if len(global_knn_stage_subset) > k_val + 1:
                    knn_pred, _ = _knn_indices_dists(X_oracle_subset, k_val)
                    knn_gt, _ = _knn_indices_dists(global_knn_stage_gt, k_val)
                    overlap = _knn_overlap_score(knn_pred, knn_gt)
                    oracle_knn_scores[k_val] = overlap.mean().item()

            knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in oracle_knn_scores.items()])
            print(f"[ORACLE-GT-STITCH] oracle merge: {knn_str}")

            print("[ORACLE-GT-STITCH] Interpretation:")
            print("  - If oracle kNN is low → patch geometry is the bottleneck.")
            print("  - If oracle kNN is high but normal is low → stitching/merge is the bottleneck.")
            print("=" * 70 + "\n")

        # ---------- CHANGE D: Incremental stitch curve ----------
        if debug_incremental_stitch_curve and len(patch_coords_2d) > 0 and global_knn_stage_subset is not None:
            print("\n" + "=" * 70)
            print("[INCR-STITCH] Incremental stitch curve (kNN vs #patches)")
            print("=" * 70)

            # Build overlap graph (simple)
            min_overlap_curve = 20
            neighbors = {k: [] for k in range(K)}
            for i in range(K):
                Si = set(patch_indices[i].tolist())
                for j in range(i + 1, K):
                    Sj = set(patch_indices[j].tolist())
                    if len(Si & Sj) >= min_overlap_curve:
                        neighbors[i].append(j)
                        neighbors[j].append(i)

            # Pick a starting patch: lowest anisotropy if available
            if len(patch_eig_ratios_2d) == K:
                start = int(np.argmin(np.array(patch_eig_ratios_2d)))
            else:
                start = 0

            # BFS order if connected; fallback to 0..K-1
            order = []
            seen = set([start])
            queue = [start]
            while queue:
                u = queue.pop(0)
                order.append(u)
                for v in neighbors.get(u, []):
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
            if len(order) < K:
                order = list(range(K))

            prev_knn = None
            biggest_drop = (None, 0.0)

            for t in range(1, K + 1):
                X_partial = torch.zeros(n_sc, 2, device=device)
                W_partial = torch.zeros(n_sc, 1, device=device)

                for idx in order[:t]:
                    if idx >= len(patch_coords_2d):
                        continue
                    S_k = patch_indices[idx]
                    V_k = patch_coords_2d[idx].float().to(device)

                    center_k = V_k.mean(dim=0, keepdim=True)
                    dists = torch.norm(V_k - center_k, dim=1, keepdim=True)
                    max_d = dists.max().clamp_min(1e-6)
                    weights_k = 1.0 - (dists / (max_d * 1.2))
                    weights_k = weights_k.clamp(min=0.01)

                    # Ensure indices are on the same device as X_partial / V_k
                    S_k = patch_indices[idx].to(device)

                    X_partial.index_add_(0, S_k, V_k * weights_k)
                    W_partial.index_add_(0, S_k, weights_k)


                mask_seen = W_partial.squeeze(-1) > 0
                X_partial[mask_seen] /= W_partial[mask_seen]
                X_partial = X_partial - X_partial.mean(dim=0, keepdim=True)

                # rescale to rms_target
                rms_partial = X_partial.pow(2).mean().sqrt().item()
                X_partial = X_partial * (rms_target / (rms_partial + 1e-8))

                covered = int(mask_seen.sum().item())
                X_sub = X_partial[global_knn_stage_subset, :2].float()
                knn_scores = {}
                for k_val in debug_k_list:
                    if len(global_knn_stage_subset) > k_val + 1:
                        knn_pred, _ = _knn_indices_dists(X_sub, k_val)
                        knn_gt, _ = _knn_indices_dists(global_knn_stage_gt, k_val)
                        overlap = _knn_overlap_score(knn_pred, knn_gt)
                        knn_scores[k_val] = overlap.mean().item()

                knn10 = knn_scores.get(debug_k_list[0], float('nan'))
                knn20 = knn_scores.get(debug_k_list[1], float('nan'))
                print(f"[INCR-STITCH] t={t:03d} covered={covered:5d} kNN@{debug_k_list[0]}={knn10:.3f} kNN@{debug_k_list[1]}={knn20:.3f}")

                if prev_knn is not None and not np.isnan(knn10):
                    drop = prev_knn - knn10
                    if drop > biggest_drop[1]:
                        biggest_drop = (t, drop)
                prev_knn = knn10

            if biggest_drop[0] is not None:
                print(f"[INCR-STITCH] Biggest drop at t={biggest_drop[0]}: ΔkNN@{debug_k_list[0]}={biggest_drop[1]:.3f}")
            print("=" * 70 + "\n")

        # ---------- CHANGE E (patch-level): scale compression ----------
        if debug_scale_compression and len(patch_coords_2d) > 0:
            scale_ratios = []
            k_scale = 10
            n_test = min(K, debug_max_patches, len(patch_coords_2d))

            for k in range(n_test):
                if k >= len(patch_coords_2d):
                    continue
                S_k = patch_indices[k]
                S_k = patch_indices[k].to(device)

                m_k = S_k.numel()
                if m_k <= k_scale + 1:
                    continue

                pred_k = patch_coords_2d[k].float().to(device)
                gt_k = gt_coords_t[S_k]

                _, d_pred = _knn_indices_dists(pred_k, k_scale)
                _, d_gt = _knn_indices_dists(gt_k, k_scale)

                pred_scale = torch.median(d_pred[:, -1]).item()
                gt_scale = torch.median(d_gt[:, -1]).item()
                if gt_scale > 1e-8:
                    scale_ratios.append(pred_scale / gt_scale)

            if scale_ratios:
                sr = np.array(scale_ratios)
                print(f"[SCALE-COMP] patch scale ratio (pred/gt): "
                      f"p10={np.percentile(sr,10):.3f} p50={np.median(sr):.3f} p90={np.percentile(sr,90):.3f}")

    # ===================================================================
    # [PATCH-KNN-VS-GT-2D] Patch kNN vs GT in PCA-projected 2D space
    # ===================================================================
    # Goal: Check if neighborhood info is lost at the 32D→2D projection step.
    # If kNN is already ~0.36/0.50 in 2D (same as final), then PGSO cannot
    # improve it - the problem is the 2D readout, not stitching.
    # ===================================================================
    if debug_knn and gt_coords is not None and len(patch_coords_2d) > 0 and len(patch_coords) > 0:
        print("\n" + "="*70)
        print("[PATCH-KNN-VS-GT-2D] PATCH kNN IN PCA-PROJECTED 2D SPACE")
        print("="*70)

        with torch.no_grad():
            gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()

            patch_knn_2d_overlap = {k_val: [] for k_val in debug_k_list}

            # Also compute 32D kNN for comparison (same patches)
            patch_knn_32d_overlap = {k_val: [] for k_val in debug_k_list}

            n_patches_to_test = min(K, debug_max_patches, len(patch_coords_2d), len(patch_coords))

            for k in range(n_patches_to_test):
                if k >= len(patch_coords_2d) or k >= len(patch_coords):
                    continue
                S_k = patch_indices[k]
                S_k = patch_indices[k].to(device)

                m_k = S_k.numel()

                if m_k < max(debug_k_list) + 5:
                    continue

                # Get GT coords for this patch
                gt_patch = gt_coords_t[S_k]  # (m_k, 2)

                # Get 2D coords (PCA-projected)
                pred_patch_2d = patch_coords_2d[k].float().to(device)  # (m_k, 2)

                # Get 32D coords (original)
                pred_patch_32d = patch_coords[k].float().to(device)  # (m_k, D_latent)

                for k_val in debug_k_list:
                    if m_k > k_val + 1:
                        # kNN in 2D projected space
                        knn_pred_2d, _ = _knn_indices_dists(pred_patch_2d, k_val)
                        knn_gt, _ = _knn_indices_dists(gt_patch, k_val)
                        overlap_2d = _knn_overlap_score(knn_pred_2d, knn_gt)
                        patch_knn_2d_overlap[k_val].append(overlap_2d.mean().item())

                        # kNN in 32D space (for comparison)
                        knn_pred_32d, _ = _knn_indices_dists(pred_patch_32d, k_val)
                        overlap_32d = _knn_overlap_score(knn_pred_32d, knn_gt)
                        patch_knn_32d_overlap[k_val].append(overlap_32d.mean().item())

            # Print results
            print(f"[PATCH-KNN-VS-GT-2D] n_patches_tested={n_patches_to_test}")
            print()
            print("Comparing kNN overlap in 32D (original) vs 2D (PCA-projected):")
            print()

            for k_val in debug_k_list:
                if patch_knn_2d_overlap[k_val] and patch_knn_32d_overlap[k_val]:
                    ov_2d = torch.tensor(patch_knn_2d_overlap[k_val])
                    ov_32d = torch.tensor(patch_knn_32d_overlap[k_val])

                    print(f"[PATCH-KNN-VS-GT-2D] k={k_val}: "
                          f"mean={ov_2d.mean().item():.3f} "
                          f"p10={ov_2d.quantile(0.1).item():.3f} "
                          f"p50={ov_2d.median().item():.3f} "
                          f"p90={ov_2d.quantile(0.9).item():.3f}")

                    print(f"[PATCH-KNN-VS-GT-32D] k={k_val}: "
                          f"mean={ov_32d.mean().item():.3f} "
                          f"p10={ov_32d.quantile(0.1).item():.3f} "
                          f"p50={ov_32d.median().item():.3f} "
                          f"p90={ov_32d.quantile(0.9).item():.3f}")

                    # Delta
                    delta = ov_32d.mean().item() - ov_2d.mean().item()
                    print(f"[PATCH-KNN-VS-GT] k={k_val} delta(32D-2D)={delta:+.3f}")
                    print()

            # Interpretation
            k_main = debug_k_list[0]
            if patch_knn_2d_overlap[k_main]:
                mean_2d = torch.tensor(patch_knn_2d_overlap[k_main]).mean().item()
                mean_32d = torch.tensor(patch_knn_32d_overlap[k_main]).mean().item()

                delta_main = mean_32d - mean_2d

                if delta_main > 0.08:
                    print(f"[PATCH-KNN-VS-GT-2D] ⚠️ SIGNIFICANT DROP: 32D→2D loses {delta_main:.2f} kNN overlap")
                    print(f"    → PCA projection is discarding neighborhood-relevant dimensions")
                    print(f"    → PGSO cannot recover this; need better 2D readout (e.g., ST-trained linear)")
                elif delta_main > 0.03:
                    print(f"[PATCH-KNN-VS-GT-2D] MODERATE DROP: 32D→2D loses {delta_main:.2f} kNN overlap")
                    print(f"    → Some info lost in projection, but may not be dominant issue")
                else:
                    print(f"[PATCH-KNN-VS-GT-2D] ✓ MINIMAL DROP: 32D and 2D give similar kNN ({delta_main:+.2f})")
                    print(f"    → Projection is not the bottleneck")

                # Check if 2D is already at final level
                # (We'll compare with global final kNN in Test B)
                print()
                print(f"[PATCH-KNN-VS-GT-2D] 2D patch kNN@{k_main} = {mean_2d:.3f}")
                print(f"    → Compare this to [GLOBAL-KNN-STAGE] and final kNN to see if PGSO helps or hurts")

        print("="*70 + "\n")


    # Build overlap graph edges
    edges = []  # List of (a, b, R_ab, s_ab, t_ab, weight, n_overlap)

    if len(patch_coords_2d) > 0:
        print(f"[PGSO-A] Computing pairwise Procrustes on overlapping patches...")

        for a in range(K):
            if a >= len(patch_coords_2d):
                continue
            S_a = set(patch_indices[a].tolist())
            V_a_2d = patch_coords_2d[a]

            for b in range(a + 1, K):
                if b >= len(patch_coords_2d):
                    continue
                S_b = set(patch_indices[b].tolist())
                shared = S_a & S_b

                if len(shared) < MIN_OVERLAP_FOR_EDGE:
                    continue

                shared_list = list(shared)

                # Get positions in each patch
                pos_a = {int(cid): p for p, cid in enumerate(patch_indices[a].tolist())}
                pos_b = {int(cid): p for p, cid in enumerate(patch_indices[b].tolist())}

                idx_a = torch.tensor([pos_a[c] for c in shared_list], dtype=torch.long)
                idx_b = torch.tensor([pos_b[c] for c in shared_list], dtype=torch.long)

                V_shared_a = V_a_2d[idx_a]  # (n_shared, 2)
                V_shared_b = patch_coords_2d[b][idx_b]  # (n_shared, 2)

                # Compute Procrustes: V_b ≈ s_ab * (V_a @ R_ab.T) + t_ab
                R_ab, s_ab, t_ab, residual = weighted_procrustes_2d(V_shared_a, V_shared_b)

                # Weight by overlap size and inverse residual SQUARED (stronger penalty)
                # This downweights bad edges much more aggressively
                weight = len(shared) / (residual.item() ** 2 + 1e-4)

                edges.append({
                    'a': a, 'b': b,
                    'R_ab': R_ab, 's_ab': s_ab, 't_ab': t_ab,
                    'weight': weight, 'n_overlap': len(shared),
                    'residual': residual.item()
                })
    else:
        print(f"[PGSO-A] Skipping pairwise Procrustes (no patch_coords_2d available)")

    # ===================================================================
    # [EDGE-MODEL] DIAGNOSTIC 6: Similarity vs Affine fit comparison
    # ===================================================================
    # Goal: Check if similarity transform is too restrictive for overlaps.
    # If affine dramatically reduces residuals, per-patch relationship has
    # shear/anisotropic scaling that Sim(2) cannot capture.
    # ===================================================================
    if DEBUG_FLAG and len(edges) > 0:
        print("\n" + "="*70)
        print("[EDGE-MODEL] DIAGNOSTIC 6: SIMILARITY vs AFFINE FIT")
        print("="*70)
        
        def weighted_affine_2d(X_src, X_tgt, weights=None):
            """
            Compute affine transform (A, t) mapping X_src -> X_tgt.
            Convention: X_tgt ≈ X_src @ A.T + t  (row-vector convention)
            A is 2x2 (can have shear/anisotropic scaling), t is 2D translation.
            
            Returns: A (2,2), t (2,), residual (scalar)
            """
            n = X_src.shape[0]
            if weights is None:
                weights = torch.ones(n, 1, device=X_src.device)
            else:
                weights = weights.view(-1, 1)
            
            w_sum = weights.sum()
            mu_src = (weights * X_src).sum(dim=0, keepdim=True) / w_sum
            mu_tgt = (weights * X_tgt).sum(dim=0, keepdim=True) / w_sum
            
            Xc = X_src - mu_src
            Yc = X_tgt - mu_tgt
            
            w_sqrt = weights.sqrt()
            Xc_w = Xc * w_sqrt
            Yc_w = Yc * w_sqrt
            
            # Solve for A: Yc ≈ Xc @ A.T  =>  Yc.T ≈ A @ Xc.T
            # Using least squares: A = (Yc.T @ Xc) @ inv(Xc.T @ Xc)
            XtX = Xc_w.T @ Xc_w  # (2, 2)
            YtX = Yc_w.T @ Xc_w  # (2, 2)
            
            # Regularize to avoid singular matrix
            XtX_reg = XtX + 1e-6 * torch.eye(2, device=X_src.device)
            A = YtX @ torch.linalg.inv(XtX_reg)  # (2, 2)
            
            # Translation: t = mu_tgt - mu_src @ A.T
            t = (mu_tgt - mu_src @ A.T).squeeze(0)
            
            # Residual
            X_hat = X_src @ A.T + t
            residual = ((X_hat - X_tgt) ** 2).sum(dim=1).sqrt().mean()
            
            return A, t, residual
        
        residual_sim_list = []
        residual_aff_list = []
        delta_list = []
        shear_list = []  # Measure how "non-similarity" the affine is
        
        # Sample edges for analysis
        edges_to_test = edges[:min(100, len(edges))]
        
        for e in edges_to_test:
            a, b = e['a'], e['b']
            if a >= len(patch_coords_2d) or b >= len(patch_coords_2d):
                continue
            shared_list = list(set(patch_indices[a].tolist()) & set(patch_indices[b].tolist()))

            if len(shared_list) < 20:
                continue

            # Get positions in each patch
            pos_a = {int(cid): p for p, cid in enumerate(patch_indices[a].tolist())}
            pos_b = {int(cid): p for p, cid in enumerate(patch_indices[b].tolist())}

            idx_a = torch.tensor([pos_a[c] for c in shared_list], dtype=torch.long)
            idx_b = torch.tensor([pos_b[c] for c in shared_list], dtype=torch.long)

            V_shared_a = patch_coords_2d[a][idx_a]  # (n_shared, 2)
            V_shared_b = patch_coords_2d[b][idx_b]  # (n_shared, 2)
            
            # Similarity fit (already have this from edge computation)
            residual_sim = e['residual']
            residual_sim_list.append(residual_sim)
            
            # Affine fit
            A_aff, t_aff, residual_aff = weighted_affine_2d(V_shared_a, V_shared_b)
            residual_aff_list.append(residual_aff.item())
            
            # Delta
            delta = residual_sim - residual_aff.item()
            delta_list.append(delta)
            
            # Measure "shear" in affine transform
            # For similarity: A = s * R, so A.T @ A = s^2 * I
            # Shear = deviation from this pattern
            AtA = A_aff.T @ A_aff
            # Eigenvalues of AtA: for pure similarity they should be equal
            eigs_AtA = torch.linalg.eigvalsh(AtA)
            shear_ratio = (eigs_AtA.max() / eigs_AtA.min().clamp(min=1e-8)).item()
            shear_list.append(shear_ratio)
        
        if residual_sim_list:
            res_sim = torch.tensor(residual_sim_list)
            res_aff = torch.tensor(residual_aff_list)
            delta_t = torch.tensor(delta_list)
            shear_t = torch.tensor(shear_list)
            
            print(f"[EDGE-MODEL] n_edges_tested={len(residual_sim_list)}")
            print(f"[EDGE-MODEL] residual_sim: "
                  f"p10={res_sim.quantile(0.1).item():.4f} "
                  f"p50={res_sim.median().item():.4f} "
                  f"p90={res_sim.quantile(0.9).item():.4f}")
            print(f"[EDGE-MODEL] residual_aff: "
                  f"p10={res_aff.quantile(0.1).item():.4f} "
                  f"p50={res_aff.median().item():.4f} "
                  f"p90={res_aff.quantile(0.9).item():.4f}")
            print(f"[EDGE-MODEL] delta(sim-aff): "
                  f"p10={delta_t.quantile(0.1).item():.4f} "
                  f"p50={delta_t.median().item():.4f} "
                  f"p90={delta_t.quantile(0.9).item():.4f}")
            print(f"[EDGE-MODEL] affine_shear_ratio (1.0=pure similarity): "
                  f"p10={shear_t.quantile(0.1).item():.2f} "
                  f"p50={shear_t.median().item():.2f} "
                  f"p90={shear_t.quantile(0.9).item():.2f}")
            
            # Interpretation
            median_delta = delta_t.median().item()
            median_sim = res_sim.median().item()
            improvement_pct = (median_delta / median_sim * 100) if median_sim > 0 else 0
            
            if improvement_pct > 30:
                print(f"\n[EDGE-MODEL] ⚠️ AFFINE HELPS SIGNIFICANTLY: {improvement_pct:.0f}% residual reduction")
                print(f"    → Per-patch relationship has shear/anisotropic scaling")
                print(f"    → Sim(2) PGSO cannot capture this; consider affine transforms")
            elif improvement_pct > 15:
                print(f"\n[EDGE-MODEL] MODERATE: Affine reduces residual by {improvement_pct:.0f}%")
                print(f"    → Some non-similarity distortion, but Sim(2) may still work")
            else:
                print(f"\n[EDGE-MODEL] ✓ SIMILARITY IS ADEQUATE: Affine only reduces residual by {improvement_pct:.0f}%")
                print(f"    → Issue is not transform model; problem is elsewhere")
            
            if shear_t.median() > 1.5:
                print(f"    → Affine transforms have significant shear (ratio p50={shear_t.median().item():.2f})")
        
        print("="*70 + "\n")


    n_edges_raw = len(edges)
    
    # ===================================================================
    # [PGSO-A-FILTER] Hard filter: keep only edges with residual <= p50
    # This removes the worst half of edges to improve LS solve quality
    # ===================================================================
    if n_edges_raw > 0:
        residuals_all = torch.tensor([e['residual'] for e in edges])
        residual_thresh = residuals_all.median().item()
        
        # Filter edges but ensure we keep enough for connectivity
        edges_filtered = [e for e in edges if e['residual'] <= residual_thresh]
        
        # Check connectivity after filtering
        import networkx as nx
        G_check = nx.Graph()
        G_check.add_nodes_from(range(K))
        for e in edges_filtered:
            G_check.add_edge(e['a'], e['b'])
        n_components = nx.number_connected_components(G_check)
        
        if n_components == 1:
            # Graph still connected, use filtered edges
            edges = edges_filtered
            if DEBUG_FLAG:
                print(f"[PGSO-A-FILTER] Filtered {n_edges_raw} → {len(edges)} edges "
                      f"(residual <= {residual_thresh:.4f})")
        else:
            # Filtering disconnects graph, fall back to softer filter (p75)
            residual_thresh_soft = residuals_all.quantile(0.75).item()
            edges_filtered_soft = [e for e in edges if e['residual'] <= residual_thresh_soft]
            
            G_check2 = nx.Graph()
            G_check2.add_nodes_from(range(K))
            for e in edges_filtered_soft:
                G_check2.add_edge(e['a'], e['b'])
            
            if nx.number_connected_components(G_check2) == 1:
                edges = edges_filtered_soft
                if DEBUG_FLAG:
                    print(f"[PGSO-A-FILTER] Soft filter {n_edges_raw} → {len(edges)} edges "
                          f"(residual <= {residual_thresh_soft:.4f}, p50 disconnected)")
            else:
                if DEBUG_FLAG:
                    print(f"[PGSO-A-FILTER] No filtering (would disconnect graph)")
    
    n_edges = len(edges)
    print(f"[PGSO-A] Using {n_edges} overlap edges (min_overlap={MIN_OVERLAP_FOR_EDGE})")

    # ===================================================================
    # DEBUG: Cycle-closure error on overlap graph
    # ===================================================================
    if DEBUG_FLAG and debug_cycle_closure and n_edges > 0:
        print("\n" + "=" * 70)
        print("[CYCLE-CLOSURE] Pose-graph consistency (Sim(2) loops)")
        print("=" * 70)

        # Build quick lookup for transforms
        edge_map = {}
        for e in edges:
            edge_map[(e['a'], e['b'])] = (e['R_ab'], e['s_ab'], e['t_ab'])

        def invert_sim2(R, s, t):
            s_inv = 1.0 / s
            R_inv = R.T
            t_inv = (-t @ R) * s_inv
            return R_inv, s_inv, t_inv

        def get_sim2(a, b):
            if (a, b) in edge_map:
                return edge_map[(a, b)]
            if (b, a) in edge_map:
                R_ab, s_ab, t_ab = edge_map[(b, a)]
                return invert_sim2(R_ab, s_ab, t_ab)
            return None

        def compose_sim2(R1, s1, t1, R2, s2, t2):
            # X -> s1*(X@R1.T)+t1, then s2*(X@R2.T)+t2
            R = R2 @ R1
            s = s2 * s1
            t = s2 * (t1 @ R2.T) + t2
            return R, s, t

        # Sample triangles
        import random
        nodes = list(range(K))
        triangles = []
        max_tri = min(100, n_edges)
        tries = 0
        while len(triangles) < max_tri and tries < 1000:
            a = random.choice(nodes)
            b = random.choice(nodes)
            c = random.choice(nodes)
            if a == b or b == c or a == c:
                tries += 1
                continue
            if get_sim2(a, b) and get_sim2(b, c) and get_sim2(c, a):
                triangles.append((a, b, c))
            tries += 1

        rot_errs = []
        logscale_errs = []
        trans_errs = []

        for (a, b, c) in triangles:
            Rab, sab, tab = get_sim2(a, b)
            Rbc, sbc, tbc = get_sim2(b, c)
            Rca, sca, tca = get_sim2(c, a)

            R1, s1, t1 = compose_sim2(Rab, sab, tab, Rbc, sbc, tbc)
            R_loop, s_loop, t_loop = compose_sim2(R1, s1, t1, Rca, sca, tca)

            theta = np.arctan2(R_loop[1, 0].item(), R_loop[0, 0].item()) * 180.0 / np.pi
            rot_errs.append(theta)
            logscale_errs.append(np.log(max(s_loop.item(), 1e-8)))
            trans_errs.append(torch.norm(t_loop).item())

        if rot_errs:
            print(f"[CYCLE-CLOSURE] rot_deg: p50={np.median(np.abs(rot_errs)):.2f} "
                  f"p90={np.percentile(np.abs(rot_errs),90):.2f}")
            print(f"[CYCLE-CLOSURE] log-scale: p50={np.median(np.abs(logscale_errs)):.4f} "
                  f"p90={np.percentile(np.abs(logscale_errs),90):.4f}")
            print(f"[CYCLE-CLOSURE] trans_norm: p50={np.median(trans_errs):.4f} "
                  f"p90={np.percentile(trans_errs,90):.4f}")
        else:
            print("[CYCLE-CLOSURE] No triangles found (graph too sparse).")

        print("=" * 70 + "\n")

    if DEBUG_FLAG and n_edges > 0:
        residuals = [e['residual'] for e in edges]
        overlaps = [e['n_overlap'] for e in edges]
        print(f"[PGSO-A] Edge residuals: min={min(residuals):.4f} p50={np.median(residuals):.4f} max={max(residuals):.4f}")
        print(f"[PGSO-A] Edge overlaps: min={min(overlaps)} p50={np.median(overlaps):.0f} max={max(overlaps)}")

    # ===================================================================
    # PGSO-B1: Global least-squares solve for rotations and log-scales
    # ===================================================================
    # 
    # Math derivation (from ChatGPT):
    # Edge a→b means: V_b ≈ s_ab * (V_a @ R_ab.T) + t_ab  (Procrustes output)
    # Global pose: X = s_k * (V_k @ R_k.T) + t_k
    # 
    # Loop closure constraint: G_a = G_b ∘ T_ab
    # This gives:
    #   θ_a - θ_b = θ_ab  (NOT θ_b - θ_a = θ_ab)
    #   ℓ_a - ℓ_b = ℓ_ab  (where ℓ = log(s))
    #   t_a - t_b = s_b * (t_ab @ R_b.T)
    #
    # With incidence A[e,a]=-1, A[e,b]=+1, we encode: θ_b - θ_a = RHS
    # So we need RHS = -θ_ab to get: θ_b - θ_a = -θ_ab ⟺ θ_a - θ_b = θ_ab
    # ===================================================================
    print(f"\n[PGSO-B1] Solving global rotations and scales...")

    if n_edges == 0:
        print("[PGSO-B1] WARNING: No overlap edges! Falling back to identity transforms.")
        theta_global = torch.zeros(K)
        ell_global = torch.zeros(K)
    else:
        # Convert each edge rotation to angle
        theta_edges = []
        ell_edges = []
        edge_weights = []
        edge_pairs = []

        # ===================================================================
        # Compute rotation confidence per edge based on overlap geometry
        # When overlap is near-1D, rotation is ill-conditioned and should
        # be downweighted to prevent collapse
        # ===================================================================
        ROTATION_CONFIDENCE_THRESHOLD = 5.0  # anisotropy ratio above which rotation is unreliable
        
        for e in edges:
            R_ab = e['R_ab']
            s_ab = e['s_ab']
            a, b = e['a'], e['b']

            # Extract angle: theta = atan2(R[1,0], R[0,0])
            theta_ab = torch.atan2(R_ab[1, 0], R_ab[0, 0]).item()
            ell_ab = torch.log(s_ab).item()

            theta_edges.append(theta_ab)
            ell_edges.append(ell_ab)
            edge_pairs.append((e['a'], e['b']))
            
            # Base weight from overlap size and residual
            base_weight = e['weight']
            
            # Compute rotation confidence from overlap anisotropy
            # Use the more anisotropic of the two patches (conservative)
            aniso_a = patch_eig_ratios_2d[a] if a < len(patch_eig_ratios_2d) else 1.0
            aniso_b = patch_eig_ratios_2d[b] if b < len(patch_eig_ratios_2d) else 1.0
            max_aniso = max(aniso_a, aniso_b)
            
            # Rotation confidence: 1.0 when isotropic, drops toward 0 when highly anisotropic
            # c = 1 / max_aniso, then clip(c / c_threshold, 0, 1)
            rot_confidence = min(1.0, ROTATION_CONFIDENCE_THRESHOLD / max_aniso)
            
            edge_weights.append(base_weight)
            
            # Store rotation confidence separately (will be used for theta system only)
            if not hasattr(e, 'rot_confidence'):
                e['rot_confidence'] = rot_confidence


        theta_edges = torch.tensor(theta_edges)
        ell_edges = torch.tensor(ell_edges)
        edge_weights = torch.tensor(edge_weights)

        # Build incidence matrix: A[e, k] = +1 if k=b, -1 if k=a for edge e=(a,b)
        # This encodes: θ_b - θ_a = RHS
        A = torch.zeros(n_edges, K)
        for e_idx, (a, b) in enumerate(edge_pairs):
            A[e_idx, a] = -1.0
            A[e_idx, b] = 1.0

        # Weight the system
        # W = torch.diag(edge_weights.sqrt())
        # A_w = W @ A

        # ===================================================================
        # Separate weights for rotation vs scale systems
        # Rotation uses confidence-weighted to handle near-1D patches
        # ===================================================================
        
        # Collect rotation confidences
        rot_confidences = torch.tensor([e.get('rot_confidence', 1.0) for e in edges])
        
        # Theta weights: base_weight * rot_confidence
        theta_weights = edge_weights * rot_confidences
        
        # Scale weights: just base_weight (scale is still meaningful in 1D)
        scale_weights = edge_weights
        
        if DEBUG_FLAG:
            print(f"[PGSO-B1] Rotation confidence: p25={rot_confidences.quantile(0.25).item():.3f} "
                  f"p50={rot_confidences.median().item():.3f} "
                  f"p75={rot_confidences.quantile(0.75).item():.3f}")
            if rot_confidences.median() < 0.5:
                print(f"    ⚠️ LOW ROTATION CONFIDENCE: Most edges have unreliable rotation estimates")
        
        # Weight matrices
        W_theta = torch.diag(theta_weights.sqrt())
        W_scale = torch.diag(scale_weights.sqrt())
        
        A_w_theta = W_theta @ A
        A_w_scale = W_scale @ A


        # Anchor first patch: theta_0 = 0, ell_0 = 0
        # Remove first column from A, solve for remaining K-1 variables
        A_reduced_theta = A_w_theta[:, 1:]
        A_reduced_scale = A_w_scale[:, 1:]

        # ===================================================================
        # FIX: Negate RHS to match correct constraint direction
        # Correct constraint: θ_a - θ_b = θ_ab  ⟺  θ_b - θ_a = -θ_ab
        # Since A encodes (θ_b - θ_a), we need RHS = -θ_ab
        # ===================================================================
        b_theta = W_theta @ (-theta_edges)  # NEGATED, rotation-confidence weighted
        theta_reduced, _, _, _ = torch.linalg.lstsq(A_reduced_theta, b_theta.unsqueeze(1))
        theta_global = torch.cat([torch.zeros(1), theta_reduced.squeeze(1)])

        # Same fix for log-scales (uses scale weights, not rotation weights)
        b_ell = W_scale @ (-ell_edges)  # NEGATED, normal weights
        ell_reduced, _, _, _ = torch.linalg.lstsq(A_reduced_scale, b_ell.unsqueeze(1))
        ell_global = torch.cat([torch.zeros(1), ell_reduced.squeeze(1)])


        # ===================================================================
        # DEBUG: Check B1 residuals (should be small if correct)
        # ===================================================================
        if DEBUG_FLAG:
            # Predicted edge measurements from solved global poses
            theta_pred = []
            ell_pred = []
            for e_idx, (a, b) in enumerate(edge_pairs):
                # Constraint was: θ_a - θ_b = θ_ab
                theta_pred.append((theta_global[a] - theta_global[b]).item())
                ell_pred.append((ell_global[a] - ell_global[b]).item())
            theta_pred = torch.tensor(theta_pred)
            ell_pred = torch.tensor(ell_pred)
            
            # Residuals
            theta_resid = theta_pred - theta_edges
            # Wrap angle residuals to [-π, π]
            theta_resid = torch.atan2(torch.sin(theta_resid), torch.cos(theta_resid))
            ell_resid = ell_pred - ell_edges
            
            print(f"[PGSO-B1-CHECK] θ residual (deg): p50={theta_resid.abs().median().item()*180/np.pi:.2f} "
                  f"p90={theta_resid.abs().quantile(0.9).item()*180/np.pi:.2f}")
            print(f"[PGSO-B1-CHECK] ℓ residual: p50={ell_resid.abs().median().item():.4f} "
                  f"p90={ell_resid.abs().quantile(0.9).item():.4f}")


        # ===================================================================
        # OPTIONAL: 2-iteration reweighted LS with wrapped angle residuals
        # Uses rotation-confidence weighting for theta
        # ===================================================================
        for refine_iter in range(2):
            # Compute residuals with current solution
            theta_resid_list = []
            ell_resid_list = []
            for e_idx, (a, b) in enumerate(edge_pairs):
                # Predicted edge: θ_a - θ_b
                theta_pred = theta_global[a] - theta_global[b]
                theta_meas = theta_edges[e_idx]
                # Wrap residual to [-π, π]
                resid_theta = theta_pred - theta_meas
                resid_theta = torch.atan2(torch.sin(torch.tensor(resid_theta)), 
                                          torch.cos(torch.tensor(resid_theta))).item()
                theta_resid_list.append(resid_theta)
                
                # Log-scale residual (no wrapping needed)
                ell_pred = ell_global[a] - ell_global[b]
                ell_resid_list.append((ell_pred - ell_edges[e_idx]).item())
            
            theta_resid = torch.tensor(theta_resid_list)
            ell_resid = torch.tensor(ell_resid_list)
            
            # Reweight: downweight high-residual edges
            resid_mag = theta_resid.abs() + 0.5 * ell_resid.abs()
            resid_scale = resid_mag.median().clamp(min=0.01)
            reweight_factor = 1.0 / (1.0 + (resid_mag / resid_scale) ** 2)
            
            # Apply reweighting to both systems (keep rotation confidence for theta)
            new_theta_weights = theta_weights * reweight_factor
            new_scale_weights = scale_weights * reweight_factor
            
            W_theta = torch.diag(new_theta_weights.sqrt())
            W_scale = torch.diag(new_scale_weights.sqrt())
            A_w_theta = W_theta @ A
            A_w_scale = W_scale @ A
            A_reduced_theta = A_w_theta[:, 1:]
            A_reduced_scale = A_w_scale[:, 1:]
            
            # Re-solve with new weights (NEGATED RHS)
            b_theta = W_theta @ (-theta_edges)
            theta_reduced, _, _, _ = torch.linalg.lstsq(A_reduced_theta, b_theta.unsqueeze(1))
            theta_global = torch.cat([torch.zeros(1), theta_reduced.squeeze(1)])
            
            b_ell = W_scale @ (-ell_edges)
            ell_reduced, _, _, _ = torch.linalg.lstsq(A_reduced_scale, b_ell.unsqueeze(1))
            ell_global = torch.cat([torch.zeros(1), ell_reduced.squeeze(1)])
            
            if DEBUG_FLAG:
                print(f"[PGSO-B1] Refine iter {refine_iter+1}: "
                      f"θ_resid_p50={theta_resid.abs().median().item()*180/np.pi:.2f}° "
                      f"ℓ_resid_p50={ell_resid.abs().median().item():.4f}")


        # Center log-scales around 0 (median scale = 1)
        ell_global = ell_global - ell_global.median()

        # Soft clamp scales to prevent extreme drift
        SCALE_CLAMP_MIN = 0.85
        SCALE_CLAMP_MAX = 1.15
        ell_global = ell_global.clamp(np.log(SCALE_CLAMP_MIN), np.log(SCALE_CLAMP_MAX))


    # Convert back to R, s
    s_global_list = torch.exp(ell_global)
    R_global_list = []
    for k in range(K):
        theta_k = theta_global[k]
        c, s_rot = torch.cos(theta_k), torch.sin(theta_k)
        R_k = torch.tensor([[c, -s_rot], [s_rot, c]])
        R_global_list.append(R_k)

    if DEBUG_FLAG:
        print(f"[PGSO-B1] Global angles (deg): min={theta_global.min().item()*180/np.pi:.1f} "
              f"p50={theta_global.median().item()*180/np.pi:.1f} "
              f"max={theta_global.max().item()*180/np.pi:.1f}")
        print(f"[PGSO-B1] Global scales: min={s_global_list.min().item():.3f} "
              f"p50={s_global_list.median().item():.3f} "
              f"max={s_global_list.max().item():.3f}")

    # ===================================================================
    # PGSO-B2: Global least-squares solve for translations
    # ===================================================================
    print(f"\n[PGSO-B2] Solving global translations...")

    if n_edges == 0:
        print("[PGSO-B2] WARNING: No overlap edges! Setting translations to 0.")
        t_global_list = [torch.zeros(2) for _ in range(K)]
    else:
        # Build translation constraints
        # Constraint: t_a - t_b = s_b * (t_ab @ R_b.T)
        # This is a 2K x 2K system (t is 2D for each patch)

        # Build block system: for each edge (a,b), we have 2 equations
        A_t = torch.zeros(n_edges * 2, K * 2)
        b_t = torch.zeros(n_edges * 2)

        for e_idx, e in enumerate(edges):
            a, b = e['a'], e['b']
            R_b = R_global_list[b]
            s_b = s_global_list[b]
            t_ab = e['t_ab']  # (2,)

            # RHS: s_b * (t_ab @ R_b.T)
            rhs = s_b * (t_ab @ R_b.T)

            # LHS: t_a - t_b
            # Equations for x-component
            A_t[e_idx * 2, a * 2] = 1.0      # t_a[0]
            A_t[e_idx * 2, b * 2] = -1.0     # -t_b[0]
            b_t[e_idx * 2] = rhs[0].item()

            # Equations for y-component
            A_t[e_idx * 2 + 1, a * 2 + 1] = 1.0   # t_a[1]
            A_t[e_idx * 2 + 1, b * 2 + 1] = -1.0  # -t_b[1]
            b_t[e_idx * 2 + 1] = rhs[1].item()

        # Weight the system
        W_t = torch.zeros(n_edges * 2, n_edges * 2)
        for e_idx, e in enumerate(edges):
            w = np.sqrt(e['weight'])
            W_t[e_idx * 2, e_idx * 2] = w
            W_t[e_idx * 2 + 1, e_idx * 2 + 1] = w

        A_t_w = W_t @ A_t
        b_t_w = W_t @ b_t

        # Anchor first patch: t_0 = 0
        # Remove columns 0 and 1 (x and y of patch 0)
        A_t_reduced = A_t_w[:, 2:]

        # Solve
        t_reduced, _, _, _ = torch.linalg.lstsq(A_t_reduced, b_t_w.unsqueeze(1))
        t_flat = torch.cat([torch.zeros(2), t_reduced.squeeze(1)])

        t_global_list = [t_flat[k*2:(k+1)*2] for k in range(K)]

    if DEBUG_FLAG:
        t_norms = torch.tensor([t.norm().item() for t in t_global_list])
        print(f"[PGSO-B2] Translation norms: min={t_norms.min().item():.3f} "
              f"p50={t_norms.median().item():.3f} "
              f"max={t_norms.max().item():.3f}")
        
        # ===================================================================
        # DEBUG: Check B2 residuals (should be small if B1+B2 are consistent)
        # ===================================================================
        if n_edges > 0:
            t_resid_norms = []
            for e in edges[:min(50, len(edges))]:
                a, b = e['a'], e['b']
                R_b = R_global_list[b]
                s_b = s_global_list[b]
                t_ab = e['t_ab']
                
                # Expected: t_a - t_b = s_b * (t_ab @ R_b.T)
                lhs = t_global_list[a] - t_global_list[b]
                rhs = s_b * (t_ab @ R_b.T)
                resid = (lhs - rhs).norm().item()
                t_resid_norms.append(resid)
            
            t_resid_norms = torch.tensor(t_resid_norms)
            print(f"[PGSO-B2-CHECK] Translation residual norms: "
                  f"p50={t_resid_norms.median().item():.4f} "
                  f"p90={t_resid_norms.quantile(0.9).item():.4f}")


    # ===================================================================
    # PGSO-C: Apply global transforms to each patch (in 2D space)
    # ===================================================================
    patch_coords_transformed = []

    if len(patch_coords_2d) > 0:
        print(f"\n[PGSO-C] Applying global transforms to patches...")

        for k in range(K):
            if k >= len(patch_coords_2d) or k >= len(R_global_list) or k >= len(s_global_list) or k >= len(t_global_list):
                continue
            V_k_2d = patch_coords_2d[k]  # (m_k, 2)
            R_k = R_global_list[k]
            s_k = s_global_list[k]
            t_k = t_global_list[k]

            # Transform: X_hat_k = s_k * (V_k @ R_k.T) + t_k
            X_hat_k = s_k * (V_k_2d @ R_k.T) + t_k
            patch_coords_transformed.append(X_hat_k)
    else:
        print(f"\n[PGSO-C] Skipping global transforms (no patch_coords_2d available)")


    # ===================================================================
    # PGSO-CHECK: Verify transform consistency
    # ===================================================================
    if DEBUG_FLAG and n_edges > 0:
        print(f"\n[PGSO-CHECK] Verifying transform consistency on overlap edges...")

        dist_mismatches = []
        coord_mismatches = []  # NEW: Direct coordinate mismatch after transform
        
        for e in edges[:min(50, len(edges))]:  # Check first 50 edges
            a, b = e['a'], e['b']
            if a >= len(patch_coords_transformed) or b >= len(patch_coords_transformed):
                continue
            shared_list = list(set(patch_indices[a].tolist()) & set(patch_indices[b].tolist()))

            pos_a = {int(cid): p for p, cid in enumerate(patch_indices[a].tolist())}
            pos_b = {int(cid): p for p, cid in enumerate(patch_indices[b].tolist())}

            idx_a = torch.tensor([pos_a[c] for c in shared_list], dtype=torch.long)
            idx_b = torch.tensor([pos_b[c] for c in shared_list], dtype=torch.long)

            X_a = patch_coords_transformed[a][idx_a]
            X_b = patch_coords_transformed[b][idx_b]

            # NEW: Direct coordinate mismatch (should be small after correct PGSO)
            coord_diff = (X_a - X_b).norm(dim=1)
            coord_mismatches.append(coord_diff.mean().item())

            # Distance matrices
            D_a = torch.cdist(X_a, X_a)
            D_b = torch.cdist(X_b, X_b)
            D_a_norm = D_a / (D_a.pow(2).mean().sqrt().clamp_min(1e-6))
            D_b_norm = D_b / (D_b.pow(2).mean().sqrt().clamp_min(1e-6))

            mismatch = (D_a_norm - D_b_norm).abs().mean().item()
            dist_mismatches.append(mismatch)

        print(f"[PGSO-CHECK] Coord mismatch (should be small): "
              f"p50={np.median(coord_mismatches):.4f} "
              f"p90={np.percentile(coord_mismatches, 90):.4f}")
        print(f"[PGSO-CHECK] Patch dist mismatch: p10={np.percentile(dist_mismatches, 10):.4f} "
              f"p50={np.median(dist_mismatches):.4f} "
              f"p90={np.percentile(dist_mismatches, 90):.4f}")
        
        # Interpretation
        # Normalize coord mismatch by RMS for meaningful threshold
        coord_mismatch_norm = np.median(coord_mismatches) / rms_target
        print(f"  [PGSO-CHECK] Coord mismatch / RMS = {coord_mismatch_norm:.3f}")
        
        if coord_mismatch_norm > 0.5:
            print(f"    ⚠️ WARNING: Large coord mismatch (>{50}% of RMS) → alignment failed")
        elif coord_mismatch_norm > 0.2:
            print(f"    ⚠️ MODERATE: Coord mismatch is {coord_mismatch_norm*100:.0f}% of RMS")
        else:
            print(f"    ✓ Coord mismatch is small (<20% of RMS) → alignment is working")



        # ===================================================================
        # [PGSO-CHECK-TRANSFORM] Verify predicted vs measured transforms
        # From G_a ≈ G_b ∘ T_ab, we can compute: T_ab_pred = G_b^-1 ∘ G_a
        # Compare to measured T_ab from Procrustes
        # ===================================================================
        print(f"\n[PGSO-CHECK-TRANSFORM] Comparing predicted vs measured edge transforms...")
        
        scale_errors = []
        rot_errors = []
        trans_errors = []
        
        for e in edges[:min(30, len(edges))]:
            a, b = e['a'], e['b']
            
            # Measured from Procrustes
            R_ab_meas = e['R_ab']
            s_ab_meas = e['s_ab']
            t_ab_meas = e['t_ab']
            
            # Predicted from global poses: T_ab = G_b^-1 ∘ G_a
            # G_k(x) = s_k * (x @ R_k.T) + t_k
            # G_b^-1(y) = (1/s_b) * ((y - t_b) @ R_b)
            # T_ab_pred = G_b^-1 ∘ G_a
            #   s_pred = s_a / s_b
            #   R_pred = R_b.T @ R_a  (since R^-1 = R.T for rotation)
            #   t_pred = (1/s_b) * ((t_a - t_b) @ R_b)
            
            R_a, R_b = R_global_list[a], R_global_list[b]
            s_a, s_b = s_global_list[a], s_global_list[b]
            t_a, t_b = t_global_list[a], t_global_list[b]
            
            s_ab_pred = s_a / s_b
            R_ab_pred = R_b.T @ R_a
            t_ab_pred = (1.0 / s_b) * ((t_a - t_b) @ R_b)
            
            # Errors
            scale_err = abs(torch.log(s_ab_pred) - torch.log(s_ab_meas)).item()
            
            # Rotation error (wrapped)
            theta_pred = torch.atan2(R_ab_pred[1, 0], R_ab_pred[0, 0])
            theta_meas = torch.atan2(R_ab_meas[1, 0], R_ab_meas[0, 0])
            rot_err = torch.atan2(torch.sin(theta_pred - theta_meas), 
                                  torch.cos(theta_pred - theta_meas)).abs().item()
            
            trans_err = (t_ab_pred - t_ab_meas).norm().item()
            
            scale_errors.append(scale_err)
            rot_errors.append(rot_err)
            trans_errors.append(trans_err)
        
        scale_errors = torch.tensor(scale_errors)
        rot_errors = torch.tensor(rot_errors)
        trans_errors = torch.tensor(trans_errors)
        
        print(f"  [PGSO-CHECK-TRANSFORM] Scale error (log): "
              f"p50={scale_errors.median().item():.4f} "
              f"p90={scale_errors.quantile(0.9).item():.4f}")
        print(f"  [PGSO-CHECK-TRANSFORM] Rotation error (rad): "
              f"p50={rot_errors.median().item():.4f} ({rot_errors.median().item()*180/np.pi:.1f}°) "
              f"p90={rot_errors.quantile(0.9).item():.4f} ({rot_errors.quantile(0.9).item()*180/np.pi:.1f}°)")
        print(f"  [PGSO-CHECK-TRANSFORM] Translation error: "
              f"p50={trans_errors.median().item():.4f} "
              f"p90={trans_errors.quantile(0.9).item():.4f}")



    # ===================================================================
    # DEBUG: Post-PGSO overlap sanity decomposition
    # ===================================================================
    if DEBUG_FLAG and debug_overlap_postcheck and len(patch_coords_transformed) > 0 and len(edges) > 0:
        print("\n" + "=" * 70)
        print("[POSTCHECK] Overlap mismatch after PGSO transforms")
        print("=" * 70)

        mismatch_norms = []
        proc_residuals = []

        edges_to_check = edges[:min(100, len(edges))]
        for e in edges_to_check:
            a, b = e['a'], e['b']
            if a >= len(patch_coords_transformed) or b >= len(patch_coords_transformed):
                continue

            Sa = set(patch_indices[a].tolist())
            Sb = set(patch_indices[b].tolist())
            shared = list(Sa & Sb)
            if len(shared) < 10:
                continue

            pos_a = {int(cid): p for p, cid in enumerate(patch_indices[a].tolist())}
            pos_b = {int(cid): p for p, cid in enumerate(patch_indices[b].tolist())}

            idx_a = torch.tensor([pos_a[c] for c in shared], dtype=torch.long)
            idx_b = torch.tensor([pos_b[c] for c in shared], dtype=torch.long)

            Xa = patch_coords_transformed[a][idx_a].to(device)
            Xb = patch_coords_transformed[b][idx_b].to(device)


            mismatch_raw = torch.norm(Xa - Xb, dim=1).mean().item()
            mismatch_norm = mismatch_raw / (rms_target + 1e-8)
            mismatch_norms.append(mismatch_norm)

            # Procrustes residual AFTER transform
            _, _, _, res = weighted_procrustes_2d(Xa, Xb)
            proc_residuals.append(res.item())

        if mismatch_norms:
            mm = np.array(mismatch_norms)
            pr = np.array(proc_residuals)

            print(f"[POSTCHECK] mismatch_norm: p10={np.percentile(mm,10):.3f} "
                  f"p50={np.median(mm):.3f} p90={np.percentile(mm,90):.3f}")
            print(f"[POSTCHECK] procrustes_residual: p10={np.percentile(pr,10):.4f} "
                  f"p50={np.median(pr):.4f} p90={np.percentile(pr,90):.4f}")

            if len(mm) > 2:
                corr = np.corrcoef(mm, pr)[0, 1]
                print(f"[POSTCHECK] corr(mismatch_norm, procrustes_residual)={corr:.3f}")

            print("[POSTCHECK] Interpretation:")
            print("  - mismatch large, procrustes small → likely indexing/bookkeeping error")
            print("  - both large → overlap geometry is non‑rigid inconsistent")
        else:
            print("[POSTCHECK] No valid overlap edges to check.")

        print("=" * 70 + "\n")

    # ===================================================================
    # PGSO-D: Merge transformed patches (simple weighted mean)
    # ===================================================================
    print(f"\n[PGSO-D] Merging patches with weighted mean...")

    # Work in 2D space (the transformed coordinates)
    X_global = torch.zeros(n_sc, 2, dtype=torch.float32, device=device)
    W_global = torch.zeros(n_sc, 1, dtype=torch.float32, device=device)

    for k in range(K):
        if k >= len(patch_coords_transformed) or k >= len(patch_coords_2d):
            continue
        S_k = patch_indices[k].to(device)
        X_hat_k = patch_coords_transformed[k].to(device)
        V_k_2d = patch_coords_2d[k].to(device)

        # Centrality weights
        center_k = V_k_2d.mean(dim=0, keepdim=True)
        dists = torch.norm(V_k_2d - center_k, dim=1, keepdim=True)
        max_d = dists.max().clamp_min(1e-6)
        weights_k = 1.0 - (dists / (max_d * 1.2))
        weights_k = weights_k.clamp(min=0.01)

        X_global.index_add_(0, S_k, X_hat_k * weights_k)
        W_global.index_add_(0, S_k, weights_k)

    # Normalize
    mask_seen = W_global.squeeze(-1) > 0
    X_global[mask_seen] /= W_global[mask_seen]

    # Recenter
    X_global = X_global - X_global.mean(dim=0, keepdim=True)

    # Rescale to target RMS
    rms_current = X_global.pow(2).mean().sqrt()
    scale_correction = rms_target / (rms_current + 1e-8)
    X_global = X_global * scale_correction

    if DEBUG_FLAG:
        coverage_counts = torch.tensor([len(memberships[i]) for i in range(n_sc)], dtype=torch.float32)
        rms_final = X_global.pow(2).mean().sqrt().item()
        print(f"[PGSO-D] Final X_global: shape={tuple(X_global.shape)}, rms={rms_final:.3f}")
        print(f"[PGSO-D] Coverage: min={coverage_counts.min():.0f} p50={coverage_counts.median():.0f} max={coverage_counts.max():.0f}")

    # ===================================================================
    # [GLOBAL-KNN-STAGE] Stage 2: After PGSO merge (before refinement)
    # ===================================================================
    if debug_knn and gt_coords is not None and global_knn_stage_subset is not None:
        with torch.no_grad():
            X_pgso_subset = X_global[global_knn_stage_subset, :2].float()  # Use 2D
            
            pgso_knn_scores = {}
            for k_val in debug_k_list:
                if len(global_knn_stage_subset) > k_val + 1:
                    knn_pred, _ = _knn_indices_dists(X_pgso_subset, k_val)
                    knn_gt, _ = _knn_indices_dists(global_knn_stage_gt, k_val)
                    overlap = _knn_overlap_score(knn_pred, knn_gt)
                    pgso_knn_scores[k_val] = overlap.mean().item()
            
            global_knn_stage_results['pgso'] = pgso_knn_scores
            
            if DEBUG_FLAG:
                knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in pgso_knn_scores.items()])
                print(f"[GLOBAL-KNN-STAGE] pgso (post-merge, pre-refine): {knn_str}")


    # ===================================================================
    # [MERGE-AB] DIAGNOSTIC 5: Merge strategy A/B comparison
    # ===================================================================
    # Goal: Test if merge-induced compression is the issue.
    # Compare: weighted mean (baseline), pick-best, median
    # If pick-best/median improves kNN and r_ratio, problem is averaging.
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[MERGE-AB] DIAGNOSTIC 5: MERGE STRATEGY COMPARISON")
        print("="*70)
        
        with torch.no_grad():
            gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
            # ---------------------------------------------------------------
            # Merge variant 1: WEIGHTED MEAN (baseline - already computed as X_global)
            # ---------------------------------------------------------------
            X_mean = X_global[:, :2].clone()  # Use 2D coords
            
            # ---------------------------------------------------------------
            # Merge variant 2: PICK-BEST (take coord from patch with max centrality)
            # ---------------------------------------------------------------
            X_pickbest = torch.zeros(n_sc, 2, dtype=torch.float32, device=device)
            best_weight = torch.zeros(n_sc, dtype=torch.float32, device=device)
            
            for k in range(K):
                if k >= len(patch_coords_transformed) or k >= len(patch_coords_2d):
                    continue
                S_k = patch_indices[k].to(device)
                X_hat_k = patch_coords_transformed[k].to(device)
                V_k_2d = patch_coords_2d[k].to(device)

                # Centrality weights
                center_k = V_k_2d.mean(dim=0, keepdim=True)
                dists = torch.norm(V_k_2d - center_k, dim=1)
                max_d = dists.max().clamp_min(1e-6)
                weights_k = 1.0 - (dists / (max_d * 1.2))
                weights_k = weights_k.clamp(min=0.01)

                # For each cell, keep coord if this patch has higher weight
                for local_idx, global_idx in enumerate(S_k.tolist()):
                    w = weights_k[local_idx].item()
                    if w > best_weight[global_idx].item():
                        best_weight[global_idx] = w
                        X_pickbest[global_idx] = X_hat_k[local_idx]
            
            # Recenter and rescale pickbest
            X_pickbest = X_pickbest - X_pickbest.mean(dim=0, keepdim=True)
            rms_pb = X_pickbest.pow(2).mean().sqrt()
            X_pickbest = X_pickbest * (rms_target / (rms_pb + 1e-8))
            
            # ---------------------------------------------------------------
            # Merge variant 3: COORDINATE-WISE MEDIAN
            # ---------------------------------------------------------------
            X_median = torch.zeros(n_sc, 2, dtype=torch.float32, device=device)
            
            for cell_i in range(n_sc):
                patches_with_i = memberships[cell_i]
                if len(patches_with_i) == 0:
                    continue
                
                coords_i = []
                for p_idx in patches_with_i:
                    if p_idx >= len(patch_coords_transformed):
                        continue
                    patch_cell_list = patch_indices[p_idx].tolist()
                    if cell_i in patch_cell_list:
                        local_pos = patch_cell_list.index(cell_i)
                        coord_from_patch = patch_coords_transformed[p_idx][local_pos]
                        coords_i.append(coord_from_patch)
                
                if coords_i:
                    coords_i_stack = torch.stack(coords_i).to(device)  # (n_patches, 2)
                    # Coordinate-wise median
                    X_median[cell_i] = coords_i_stack.median(dim=0).values
            
            # Recenter and rescale median
            X_median = X_median - X_median.mean(dim=0, keepdim=True)
            rms_med = X_median.pow(2).mean().sqrt()
            X_median = X_median * (rms_target / (rms_med + 1e-8))
            
            # ---------------------------------------------------------------
            # Compute metrics for each merge variant
            # ---------------------------------------------------------------
            merge_variants = {
                'mean': X_mean,
                'pickbest': X_pickbest,
                'median': X_median,
            }
            
            # Use subset for efficiency (device-consistent indexing)
            M_merge = min(n_sc, debug_global_subset)

            # pick a device that matches the merge tensors (X_mean etc.)
            dev_merge = next(iter(merge_variants.values())).device

            subset_idx = torch.randperm(n_sc, device=dev_merge)[:M_merge]

            gt_subset = gt_coords_t.to(dev_merge)[subset_idx]

            for name, X_var in merge_variants.items():
                X_subset = X_var[subset_idx].float()
      
                # kNN overlap
                knn_scores = {}
                for k_val in debug_k_list:
                    if M_merge > k_val + 1:
                        knn_pred, _ = _knn_indices_dists(X_subset, k_val)
                        knn_gt, _ = _knn_indices_dists(gt_subset, k_val)
                        overlap = _knn_overlap_score(knn_pred, knn_gt)
                        knn_scores[k_val] = overlap.mean().item()
                
                # Local density ratio (r_pr/r_gt at k=10)
                k_density = 10
                D_gt = torch.cdist(gt_subset, gt_subset)
                D_gt.fill_diagonal_(float('inf'))
                d_gt_sorted, _ = D_gt.topk(k_density, largest=False, dim=1)
                r_gt_k = d_gt_sorted[:, -1]
                
                D_pr = torch.cdist(X_subset, X_subset)
                D_pr.fill_diagonal_(float('inf'))
                d_pr_sorted, _ = D_pr.topk(k_density, largest=False, dim=1)
                r_pr_k = d_pr_sorted[:, -1]
                
                ratio = r_pr_k / r_gt_k.clamp(min=1e-8)
                r_ratio_p50 = ratio.median().item()
                
                knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in knn_scores.items()])
                print(f"[MERGE-AB] {name}_merge: {knn_str} r_ratio_p50={r_ratio_p50:.3f}")
            
            # Interpretation
            print()
            mean_knn10 = knn_scores.get(debug_k_list[0], 0) if 'mean' in merge_variants else 0
            
            # Re-extract metrics for comparison (need to recompute for mean)
            X_mean_sub = merge_variants['mean'][subset_idx].float()
            D_pr_mean = torch.cdist(X_mean_sub, X_mean_sub)
            D_pr_mean.fill_diagonal_(float('inf'))
            d_pr_sorted_mean, _ = D_pr_mean.topk(k_density, largest=False, dim=1)
            r_pr_k_mean = d_pr_sorted_mean[:, -1]
            ratio_mean = r_pr_k_mean / r_gt_k.clamp(min=1e-8)
            r_ratio_mean = ratio_mean.median().item()
            
            X_pb_sub = merge_variants['pickbest'][subset_idx].float()
            D_pr_pb = torch.cdist(X_pb_sub, X_pb_sub)
            D_pr_pb.fill_diagonal_(float('inf'))
            d_pr_sorted_pb, _ = D_pr_pb.topk(k_density, largest=False, dim=1)
            r_pr_k_pb = d_pr_sorted_pb[:, -1]
            ratio_pb = r_pr_k_pb / r_gt_k.clamp(min=1e-8)
            r_ratio_pb = ratio_pb.median().item()
            
            # Compute kNN for mean and pickbest
            knn_mean_10 = 0
            knn_pb_10 = 0
            k_main = debug_k_list[0]
            if M_merge > k_main + 1:
                knn_pred_mean, _ = _knn_indices_dists(X_mean_sub, k_main)
                knn_pred_pb, _ = _knn_indices_dists(X_pb_sub, k_main)
                knn_gt_sub, _ = _knn_indices_dists(gt_subset, k_main)
                knn_mean_10 = _knn_overlap_score(knn_pred_mean, knn_gt_sub).mean().item()
                knn_pb_10 = _knn_overlap_score(knn_pred_pb, knn_gt_sub).mean().item()
            
            delta_knn = knn_pb_10 - knn_mean_10
            delta_r = r_ratio_pb - r_ratio_mean
            
            if delta_r > 0.1 and delta_knn > 0.02:
                print(f"[MERGE-AB] ⚠️ PICK-BEST HELPS: r_ratio +{delta_r:.2f}, kNN@{k_main} +{delta_knn:.3f}")
                print(f"    → Averaging inconsistent estimates is compressing local structure")
                print(f"    → Consider: pick-best merge, or post-merge decompression correction")
            elif delta_r > 0.05:
                print(f"[MERGE-AB] MODERATE: Pick-best improves r_ratio by {delta_r:.2f}")
                print(f"    → Some compression from averaging, but may not be main issue")
            else:
                print(f"[MERGE-AB] ✓ MERGE STRATEGY HAS MINIMAL EFFECT")
                print(f"    → Compression is already baked into per-patch geometry")
                print(f"    → Problem is upstream: diffusion output, not merge")
        
        print("="*70 + "\n")


    # ===================================================================
    # [TEST2-SCATTER] STITCH NOISE PER CELL (Post-transform, Pre-merge)
    # ===================================================================
    # Goal: For cells appearing in multiple patches, measure how much their
    # predicted coordinates disagree AFTER transforms but BEFORE merge.
    # Large scatter = stitching disagreement is the culprit.
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[TEST2-SCATTER] STITCH NOISE PER CELL (Post-transform, Pre-merge)")
        print("="*70)
        
        with torch.no_grad():
            # Find cells with coverage >= 3
            high_cov_cells = [i for i in range(n_sc) if len(memberships[i]) >= 3]
            
            if len(high_cov_cells) >= 50:
                scatter_list = []
                cov_list = []
                
                # Sample for efficiency
                sample_cells = high_cov_cells[:min(1000, len(high_cov_cells))]
                
                for cell_i in sample_cells:
                    patches_with_i = memberships[cell_i]
                    n_patches_i = len(patches_with_i)
                    
                    # Collect coords for cell_i from each patch (post-transform)
                    coords_i = []
                    for p_idx in patches_with_i:
                        if p_idx >= len(patch_coords_transformed):
                            continue
                        # Find position of cell_i in patch p_idx
                        patch_cell_list = patch_indices[p_idx].tolist()
                        if cell_i in patch_cell_list:
                            local_pos = patch_cell_list.index(cell_i)
                            coord_from_patch = patch_coords_transformed[p_idx][local_pos]
                            coords_i.append(coord_from_patch)
                    
                    if len(coords_i) >= 3:
                        coords_i_stack = torch.stack(coords_i)  # (n_patches_i, 2)
                        mean_coord = coords_i_stack.mean(dim=0)
                        distances_to_mean = torch.norm(coords_i_stack - mean_coord.unsqueeze(0), dim=1)
                        scatter_i = distances_to_mean.median().item()
                        
                        scatter_list.append(scatter_i)
                        cov_list.append(n_patches_i)
                
                if scatter_list:
                    scatter_t = torch.tensor(scatter_list)
                    cov_t = torch.tensor(cov_list, dtype=torch.float32)
                    
                    print(f"[TEST2-SCATTER] n_cells_used={len(scatter_list)}")
                    print(f"[TEST2-SCATTER] scatter: "
                          f"p10={scatter_t.quantile(0.1).item():.4f} "
                          f"p50={scatter_t.median().item():.4f} "
                          f"p90={scatter_t.quantile(0.9).item():.4f}")
                    
                    # Scatter by coverage bucket
                    for cov_thresh in [3, 5]:
                        mask = cov_t >= cov_thresh
                        if mask.sum() >= 10:
                            scatter_cov = scatter_t[mask]
                            print(f"[TEST2-SCATTER] scatter_by_coverage: cov>={cov_thresh} "
                                  f"n={mask.sum().item()} p50={scatter_cov.median().item():.4f}")
                    
                    # Scatter as fraction of RMS
                    scatter_norm = scatter_t.median().item() / rms_target
                    print(f"[TEST2-SCATTER] scatter / RMS = {scatter_norm:.3f}")
                    
                    # Interpretation
                    if scatter_norm < 0.05:
                        print(f"\n[TEST2-SCATTER] ✓ SMALL: Scatter < 5% of RMS")
                        print(f"    → Stitching disagreement is NOT the main issue")
                        print(f"    → If kNN is still bad, problem is model's global neighbor ordering")
                    elif scatter_norm < 0.15:
                        print(f"\n[TEST2-SCATTER] MODERATE: Scatter = {scatter_norm*100:.0f}% of RMS")
                        print(f"    → Some stitching noise, may be contributing to kNN error")
                    else:
                        print(f"\n[TEST2-SCATTER] ⚠️ LARGE: Scatter = {scatter_norm*100:.0f}% of RMS")
                        print(f"    → Stitching disagreement is significant")
                        print(f"    → Patch alignment may be failing")
                    
                    # Optional: Correlation with kNN error
                    # (We'd need to compute per-cell kNN error here, which is expensive)
                    # Skipping for now to keep it simple
            else:
                print(f"[TEST2-SCATTER] ⚠️ Only {len(high_cov_cells)} cells with coverage>=3 (need 50+)")
        
        print("="*70 + "\n")

    # ===================================================================
    # [PGSO-REFINE] Post-PGSO refinement: FULL SIMILARITY (bundle adjustment)
    # 
    # Rationale: Edge-based pose graph can be consistent but still fail
    # when Procrustes residuals are large. Direct optimization of overlap
    # point agreement (bundle adjustment) minimizes the actual coord mismatch.
    #
    # Alternating minimization:
    #   (A) Patch→global Procrustes: fit (R_k, s_k, t_k) per patch
    #   (B) Global merge: weighted average of all patch predictions
    #
    # Guardrail: For near-1D patches (high anisotropy), freeze rotation
    # to prevent instability.
    # ===================================================================
    # pgso_refine_iters = int(n_align_iters) if n_align_iters > 1 else 0

    # [DEBUG-ARGS] Ensure n_align_iters is always an int (Fix B from ChatGPT)
    n_align_iters = int(n_align_iters)
    pgso_refine_iters = n_align_iters if n_align_iters > 1 else 0
    
    if DEBUG_FLAG:
        print(f"[DEBUG-ARGS] n_align_iters={n_align_iters} (int), pgso_refine_iters={pgso_refine_iters}")

    
    # Anisotropy threshold: if patch eig_ratio > this, freeze rotation
    ANISO_FREEZE_ROTATION_THRESH = 50.0
    ANISO_FREEZE_SCALE_THRESH = 200.0  # Even more extreme: also freeze scale
    
    if pgso_refine_iters > 0 and DEBUG_FLAG:
        print(f"\n[PGSO-REFINE] Running {pgso_refine_iters} FULL SIMILARITY refinement iterations...")
        print(f"    (Bundle adjustment: optimizing overlap point agreement directly)")
        print(f"    Guardrail: freeze rotation if patch anisotropy > {ANISO_FREEZE_ROTATION_THRESH}")
    
    # Track metrics across iterations
    refine_coord_mismatches = []
    refine_procrustes_residuals = []
    refine_knn_scores = {k: [] for k in debug_k_list} if debug_knn else {}
    test4_subset_indices = None  # Will be set on first iteration

    
    for refine_iter in range(pgso_refine_iters):
        # ===============================================================
        # Step A: Patch→global Procrustes (full similarity, with guardrails)
        # ===============================================================
        new_patch_coords_transformed = []
        iter_residuals = []
        n_frozen_rot = 0
        n_frozen_scale = 0
        
        for k in range(K):
            if k >= len(patch_coords_2d):
                continue
            S_k = patch_indices[k]
            S_k = patch_indices[k].to(device)

            V_k_2d = patch_coords_2d[k]  # Patch coords in 2D
            X_k_global = X_global[S_k, :2].cpu()  # Current global coords for this patch's cells

            # Compute centrality weights
            center_k = V_k_2d.mean(dim=0, keepdim=True)
            dists = torch.norm(V_k_2d - center_k, dim=1)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)

            # Get patch anisotropy in 2D (already computed earlier)
            aniso_k = patch_eig_ratios_2d[k] if k < len(patch_eig_ratios_2d) else 1.0

            # Decide what to freeze based on anisotropy
            freeze_rotation = (aniso_k > ANISO_FREEZE_ROTATION_THRESH)
            freeze_scale = (aniso_k > ANISO_FREEZE_SCALE_THRESH)

            if freeze_rotation:
                n_frozen_rot += 1
            if freeze_scale:
                n_frozen_scale += 1

            if freeze_rotation:
                # Keep rotation from PGSO (or identity if not available)
                R_k_fixed = R_global_list[k] if k < len(R_global_list) else torch.eye(2)

                if freeze_scale:
                    # Only solve translation
                    s_k_fixed = s_global_list[k] if k < len(s_global_list) else torch.tensor(1.0)
                    V_k_rotscaled = s_k_fixed * (V_k_2d @ R_k_fixed.T)
                    diff = X_k_global - V_k_rotscaled
                    w = weights_k.unsqueeze(1)
                    t_k_new = (w * diff).sum(dim=0) / w.sum()
                    R_k, s_k, t_k = R_k_fixed, s_k_fixed, t_k_new
                    residual_k = ((s_k * (V_k_2d @ R_k.T) + t_k - X_k_global) ** 2).sum(dim=1).sqrt().mean()
                else:
                    # Solve scale + translation with fixed rotation
                    # Apply fixed rotation first
                    V_k_rotated = V_k_2d @ R_k_fixed.T

                    # Weighted Procrustes for scale+translation only
                    # X_k_global ≈ s * V_k_rotated + t
                    w = weights_k.unsqueeze(1)
                    w_sum = w.sum()
                    mu_src = (w * V_k_rotated).sum(dim=0, keepdim=True) / w_sum
                    mu_tgt = (w * X_k_global).sum(dim=0, keepdim=True) / w_sum

                    Vc = V_k_rotated - mu_src
                    Yc = X_k_global - mu_tgt

                    # Scale: s = sum(w * Yc · Vc) / sum(w * Vc · Vc)
                    numer = (w * Yc * Vc).sum()
                    denom = (w * Vc * Vc).sum().clamp_min(1e-8)
                    s_k = numer / denom
                    s_k = s_k.clamp(0.5, 2.0)  # Safety clamp

                    # Translation
                    t_k = (mu_tgt - s_k * mu_src).squeeze(0)
                    R_k = R_k_fixed

                    residual_k = ((s_k * V_k_rotated + t_k - X_k_global) ** 2).sum(dim=1).sqrt().mean()
            else:
                # Full similarity Procrustes
                R_k, s_k, t_k, residual_k = weighted_procrustes_2d(
                    V_k_2d, X_k_global, weights_k
                )

            iter_residuals.append(residual_k.item() if torch.is_tensor(residual_k) else residual_k)

            # Apply transform
            X_hat_k = s_k * (V_k_2d @ R_k.T) + t_k
            new_patch_coords_transformed.append(X_hat_k)

            # Update global lists for next iteration
            if k < len(R_global_list):
                R_global_list[k] = R_k
            if k < len(s_global_list):
                s_global_list[k] = s_k if torch.is_tensor(s_k) else torch.tensor(s_k)
            if k < len(t_global_list):
                t_global_list[k] = t_k
        
        # ===============================================================
        # Step B: Global merge (weighted average of all patch predictions)
        # ===============================================================
        X_global_new = torch.zeros(n_sc, 2, dtype=torch.float32, device=device)
        W_global_new = torch.zeros(n_sc, 1, dtype=torch.float32, device=device)
        
        for k in range(K):
            if k >= len(new_patch_coords_transformed) or k >= len(patch_coords_2d):
                continue
            S_k = patch_indices[k].to(device)
            X_hat_k = new_patch_coords_transformed[k].to(device)
            V_k_2d = patch_coords_2d[k].to(device)

            # Centrality weights
            center_k = V_k_2d.mean(dim=0, keepdim=True)
            dists = torch.norm(V_k_2d - center_k, dim=1, keepdim=True)
            max_d = dists.max().clamp_min(1e-6)
            weights_k = 1.0 - (dists / (max_d * 1.2))
            weights_k = weights_k.clamp(min=0.01)

            X_global_new.index_add_(0, S_k, X_hat_k * weights_k)
            W_global_new.index_add_(0, S_k, weights_k)
        
        mask_seen = W_global_new.squeeze(-1) > 0
        X_global_new[mask_seen] /= W_global_new[mask_seen]
        X_global_new = X_global_new - X_global_new.mean(dim=0, keepdim=True)
        
        # Rescale to target RMS
        rms_new = X_global_new.pow(2).mean().sqrt()
        scale_correction = rms_target / (rms_new + 1e-8)
        X_global_new = X_global_new * scale_correction
        
        # Update X_global
        X_global[:, :2] = X_global_new
        
        # Update transformed coords for next iteration
        patch_coords_transformed = new_patch_coords_transformed
        
        # ===============================================================
        # Compute overlap coord mismatch for this iteration
        # ===============================================================
        iter_coord_mismatches = []
        for e in edges[:min(30, len(edges))]:
            a, b = e['a'], e['b']
            if a >= len(patch_coords_transformed) or b >= len(patch_coords_transformed):
                continue
            shared_list = list(set(patch_indices[a].tolist()) & set(patch_indices[b].tolist()))
            if len(shared_list) < 10:
                continue

            pos_a = {int(cid): p for p, cid in enumerate(patch_indices[a].tolist())}
            pos_b = {int(cid): p for p, cid in enumerate(patch_indices[b].tolist())}

            idx_a = torch.tensor([pos_a[c] for c in shared_list], dtype=torch.long)
            idx_b = torch.tensor([pos_b[c] for c in shared_list], dtype=torch.long)

            X_a = patch_coords_transformed[a][idx_a]
            X_b = patch_coords_transformed[b][idx_b]

            coord_diff = (X_a - X_b).norm(dim=1).mean().item()
            iter_coord_mismatches.append(coord_diff)
        
        if iter_coord_mismatches:
            coord_mismatch_p50 = np.median(iter_coord_mismatches)
            coord_mismatch_norm = coord_mismatch_p50 / rms_target
            refine_coord_mismatches.append(coord_mismatch_norm)
        
        iter_residuals_t = torch.tensor(iter_residuals)
        refine_procrustes_residuals.append(iter_residuals_t.median().item())
        
        # ===============================================================
        # [TEST4-REFINE] Track kNN vs GT each refinement iteration
        # ===============================================================
        iter_knn_scores = {}
        if debug_knn and gt_coords is not None:
            with torch.no_grad():
                gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
                
                # Use fixed subset for consistency across iterations
                M_test4 = min(n_sc, 2000)
                if refine_iter == 0:
                    # Store subset indices for reuse
                    test4_subset_indices = torch.randperm(n_sc)[:M_test4]
                
                X_test4 = X_global[test4_subset_indices, :2].float()  # Use only 2D coords (Fix A)
                gt_test4 = gt_coords_t[test4_subset_indices]
                
                for k_val in debug_k_list:
                    if M_test4 > k_val + 1:
                        knn_pred, _ = _knn_indices_dists(X_test4, k_val)
                        knn_gt, _ = _knn_indices_dists(gt_test4, k_val)
                        overlap = _knn_overlap_score(knn_pred, knn_gt)
                        iter_knn_scores[k_val] = overlap.mean().item()
        
        # ===============================================================
        # Print progress
        # ===============================================================
        if DEBUG_FLAG and (refine_iter < 5 or refine_iter == pgso_refine_iters - 1 or (refine_iter + 1) % 5 == 0):
            coord_str = f"{coord_mismatch_norm:.3f}" if iter_coord_mismatches else "N/A"
            knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in iter_knn_scores.items()]) if iter_knn_scores else ""
            print(f"  [TEST4-REFINE] iter={refine_iter} coord_mismatch/RMS={coord_str} {knn_str}")
            print(f"  [PGSO-REFINE] iter={refine_iter}: "
                  f"Procrustes_resid_p50={iter_residuals_t.median().item():.4f} "
                  f"coord_mismatch/RMS={coord_str} "
                  f"frozen_rot={n_frozen_rot}/{K} frozen_scale={n_frozen_scale}/{K}")

        # Collect kNN scores for summary
        if debug_knn and iter_knn_scores:
            for k_val, score in iter_knn_scores.items():
                refine_knn_scores[k_val].append(score)



    # ===================================================================
    # ===================================================================
    # [DGSO-v2] Distance-Graph Stitch Optimization (Consensus + Filtering + Trust Region)
    #
    # Key improvements over v1:
    # 1. Use POST-TRANSFORM coords (patch_coords_transformed) for distance measurement
    # 2. Compute disagreement score per pair and filter high-disagreement edges
    # 3. Weight edges by trust (penalize high spread)
    # 4. Full trust region anchor (ALL points, not subset)
    # 5. Anti-collapse local radius term
    # 6. Early stopping fail-safes
    #
    # Objective:
    # L(X) = sum_{(i,j) in E_keep} w_ij * rho(|X_i - X_j| - d_bar_ij)
    #      + lambda_all * (1/N) * sum_i |X_i - X0_i|^2
    #      + lambda_rad * (1/N) * sum_i rho(r_i(X) - r_bar_i)
    # ===================================================================
    if enable_dgso:
        print("\n" + "="*70)
        print("[DGSO-v2] DISTANCE-GRAPH STITCH OPTIMIZATION")
        print("="*70)
        
        # ---------------------------------------------------------------
        # STEP 0: Extract kNN edges using POST-TRANSFORM coords
        # (This fixes the scale inconsistency issue from v1)
        # ---------------------------------------------------------------
        print(f"\n[DGSO-V2-MEAS] Extracting kNN edges from {min(K, debug_max_patches)} patches, k_edge={dgso_k_edge}...")
        print(f"[DGSO-V2-MEAS] Using POST-TRANSFORM coordinates (scale-consistent)")
        
        from collections import defaultdict
        
        # Store edge measurements: edge_measurements[(i,j)] = list of (distance, weight)
        edge_measurements = defaultdict(list)
        n_patches_used = min(K, debug_max_patches)
        n_edges_directed = 0
        
        # For comparison logging
        meas_compare_samples = []
        
        for k in range(n_patches_used):
            if k >= len(patch_coords_transformed):
                continue
            S_k = patch_indices[k]
            S_k = patch_indices[k].to(device)

            # USE TRANSFORMED COORDS (post-PGSO) instead of raw patch_coords_2d
            V_k_transformed = patch_coords_transformed[k].to(device)  # (m_k, 2)
            m_k = V_k_transformed.shape[0]

            if m_k < dgso_k_edge + 1:
                continue

            # Compute centrality weights for this patch
            center_k = V_k_transformed.mean(dim=0, keepdim=True)
            dists_to_center = torch.norm(V_k_transformed - center_k, dim=1)
            max_d = dists_to_center.max().clamp_min(1e-6)
            centrality_k = 1.0 - (dists_to_center / (max_d * 1.2))
            centrality_k = centrality_k.clamp(min=0.01)

            # Compute pairwise distances within patch (POST-TRANSFORM)
            D_patch = torch.cdist(V_k_transformed, V_k_transformed)  # (m_k, m_k)
            D_patch.fill_diagonal_(float('inf'))

            # Get kNN for each point in patch
            _, knn_local = D_patch.topk(dgso_k_edge, largest=False, dim=1)  # (m_k, k_edge)

            # Also compute raw distances for comparison (first patch only)
            if k == 0 and DEBUG_FLAG and k < len(patch_coords_2d):
                V_k_raw = patch_coords_2d[k].to(device)
                D_patch_raw = torch.cdist(V_k_raw, V_k_raw)
            
            # Extract edges
            S_k_list = S_k.tolist()
            for local_i in range(m_k):
                global_i = S_k_list[local_i]
                c_i = centrality_k[local_i].item()
                
                for knn_j in range(dgso_k_edge):
                    local_j = knn_local[local_i, knn_j].item()
                    global_j = S_k_list[local_j]
                    
                    if global_i == global_j:
                        continue
                    
                    # Store as undirected edge (min, max ordering)
                    edge_key = (min(global_i, global_j), max(global_i, global_j))
                    d_ij = D_patch[local_i, local_j].item()
                    c_j = centrality_k[local_j].item()
                    w_ij = c_i * c_j  # Edge weight = product of centralities
                    
                    edge_measurements[edge_key].append((d_ij, w_ij))
                    n_edges_directed += 1
                    
                    # Sample for comparison (first 5 edges of first patch)
                    if k == 0 and len(meas_compare_samples) < 5 and DEBUG_FLAG:
                        d_raw = D_patch_raw[local_i, local_j].item()
                        meas_compare_samples.append((edge_key, d_ij, d_raw))
        
        n_pairs_unique = len(edge_measurements)
        
        # Compute stats on measurements per pair
        meas_counts = [len(v) for v in edge_measurements.values()]
        meas_counts_t = torch.tensor(meas_counts, dtype=torch.float32)
        
        print(f"[DGSO-V2-MEAS] n_patches_used={n_patches_used} k_edge={dgso_k_edge}")
        print(f"[DGSO-V2-MEAS] n_edges_directed={n_edges_directed} n_pairs_unique={n_pairs_unique}")
        print(f"[DGSO-V2-MEAS] measurements_per_pair: p50={meas_counts_t.median().item():.0f} "
              f"p90={meas_counts_t.quantile(0.9).item():.0f} max={meas_counts_t.max().item():.0f}")
        
        # Debug: compare raw vs transformed distances
        if DEBUG_FLAG and meas_compare_samples:
            print(f"\n[DGSO-V2-MEAS-COMPARE] Raw vs Transformed distances (sample):")
            for (ei, ej), d_trans, d_raw in meas_compare_samples:
                ratio = d_trans / (d_raw + 1e-8)
                print(f"  pair({ei},{ej}): d_transformed={d_trans:.4f} d_raw={d_raw:.4f} ratio={ratio:.3f}")
        
        # ---------------------------------------------------------------
        # STEP 1: Build consensus distance AND disagreement score per pair
        # ---------------------------------------------------------------
        print(f"\n[DGSO-V2-DISPERSION] Computing consensus and disagreement per edge...")
        
        edge_stats_list = []  # List of (i, j, d_bar, w_sum, M, rel_spread, q10, q50, q90)
        
        for (i, j), measurements in edge_measurements.items():
            M = len(measurements)
            if M == 0:
                continue
            
            dists = torch.tensor([m[0] for m in measurements])
            weights = torch.tensor([m[1] for m in measurements])
            
            # Weighted median (consensus distance)
            sort_idx = torch.argsort(dists)
            dists_sorted = dists[sort_idx]
            weights_sorted = weights[sort_idx]
            
            cum_w = weights_sorted.cumsum(dim=0)
            total_w = cum_w[-1]
            median_idx = (cum_w >= total_w / 2).nonzero(as_tuple=True)[0][0].item()
            d_bar = dists_sorted[median_idx].item()
            
            # Compute weighted quantiles for disagreement
            q10_idx = (cum_w >= total_w * 0.1).nonzero(as_tuple=True)[0]
            q90_idx = (cum_w >= total_w * 0.9).nonzero(as_tuple=True)[0]
            
            q10 = dists_sorted[q10_idx[0].item()].item() if len(q10_idx) > 0 else dists_sorted[0].item()
            q50 = d_bar
            q90 = dists_sorted[q90_idx[0].item()].item() if len(q90_idx) > 0 else dists_sorted[-1].item()
            
            # Relative spread = (q90 - q10) / (q50 + eps)
            rel_spread = (q90 - q10) / (q50 + 1e-8)
            
            # Sum of weights
            w_sum = weights.sum().item()
            
            edge_stats_list.append((i, j, d_bar, w_sum, M, rel_spread, q10, q50, q90))
        
        # Convert to tensors for analysis
        rel_spreads = torch.tensor([e[5] for e in edge_stats_list])
        meas_counts_full = torch.tensor([e[4] for e in edge_stats_list], dtype=torch.float32)
        d_bars = torch.tensor([e[2] for e in edge_stats_list])
        
        print(f"[DGSO-V2-DISPERSION] rel_spread: p10={rel_spreads.quantile(0.1).item():.3f} "
              f"p50={rel_spreads.median().item():.3f} p90={rel_spreads.quantile(0.9).item():.3f}")
        print(f"[DGSO-V2-DISPERSION] M (meas count): p50={meas_counts_full.median().item():.0f} "
              f"p90={meas_counts_full.quantile(0.9).item():.0f} max={meas_counts_full.max().item():.0f}")
        
        # Print worst edges by rel_spread
        if DEBUG_FLAG:
            sorted_by_spread = sorted(edge_stats_list, key=lambda x: x[5], reverse=True)
            print(f"\n[DGSO-V2-DISPERSION-WORST] Top 10 worst edges by rel_spread:")
            for idx, (ei, ej, d_bar, w_sum, M, rel_sp, q10, q50, q90) in enumerate(sorted_by_spread[:10]):
                print(f"  pair({ei},{ej}): M={M} q10={q10:.4f} q50={q50:.4f} q90={q90:.4f} rel_spread={rel_sp:.3f}")
        
        # ---------------------------------------------------------------
        # STEP 2: Filter edges hard
        # ---------------------------------------------------------------
        print(f"\n[DGSO-V2-FILTER] Filtering edges (M_min={dgso_m_min}, tau_spread={dgso_tau_spread})...")
        
        # Compute distance band thresholds
        d_band_low = d_bars.quantile(dgso_dist_band[0]).item()
        d_band_high = d_bars.quantile(dgso_dist_band[1]).item()
        
        n_removed_m = 0
        n_removed_spread = 0
        n_removed_dist = 0
        edge_list_filtered = []
        
        for (i, j, d_bar, w_sum, M, rel_spread, q10, q50, q90) in edge_stats_list:
            # Filter 1: Require minimum measurements
            if M < dgso_m_min:
                n_removed_m += 1
                continue
            
            # Filter 2: Require low disagreement
            if rel_spread > dgso_tau_spread:
                n_removed_spread += 1
                continue
            
            # Filter 3: Distance within band (avoid pathological tiny/huge edges)
            if d_bar < d_band_low or d_bar > d_band_high:
                n_removed_dist += 1
                continue
            
            edge_list_filtered.append((i, j, d_bar, w_sum, M, rel_spread))
        
        n_edges_before = len(edge_stats_list)
        n_edges_after = len(edge_list_filtered)
        
        print(f"[DGSO-V2-FILTER] before={n_edges_before} after={n_edges_after}")
        print(f"[DGSO-V2-FILTER-BREAKDOWN] removed by M<{dgso_m_min}: {n_removed_m}")
        print(f"[DGSO-V2-FILTER-BREAKDOWN] removed by rel_spread>{dgso_tau_spread}: {n_removed_spread}")
        print(f"[DGSO-V2-FILTER-BREAKDOWN] removed by distance_out_of_band: {n_removed_dist}")
        
        if n_edges_after > 0:
            kept_d_bars = torch.tensor([e[2] for e in edge_list_filtered])
            kept_spreads = torch.tensor([e[5] for e in edge_list_filtered])
            kept_M = torch.tensor([e[4] for e in edge_list_filtered], dtype=torch.float32)
            
            print(f"[DGSO-V2-FILTER-KEPT-STATS] d_bar: p10={kept_d_bars.quantile(0.1).item():.4f} "
                  f"p50={kept_d_bars.median().item():.4f} p90={kept_d_bars.quantile(0.9).item():.4f}")
            print(f"[DGSO-V2-FILTER-KEPT-STATS] rel_spread: p10={kept_spreads.quantile(0.1).item():.3f} "
                  f"p50={kept_spreads.median().item():.3f} p90={kept_spreads.quantile(0.9).item():.3f}")
            print(f"[DGSO-V2-FILTER-KEPT-STATS] M: p10={kept_M.quantile(0.1).item():.0f} "
                  f"p50={kept_M.median().item():.0f} p90={kept_M.quantile(0.9).item():.0f}")
        
        # ---------------------------------------------------------------
        # STEP 3: Compute trust-weighted edge weights
        # w_final = w_sum * exp(-alpha * rel_spread^2)
        # ---------------------------------------------------------------
        print(f"\n[DGSO-V2-WEIGHTS] Computing trust-weighted edge weights (alpha={dgso_spread_penalty_alpha})...")
        
        edge_list_final = []
        for (i, j, d_bar, w_sum, M, rel_spread) in edge_list_filtered:
            # Trust penalty: edges with high spread get downweighted
            trust_factor = torch.exp(torch.tensor(-dgso_spread_penalty_alpha * rel_spread**2)).item()
            w_final = w_sum * trust_factor
            edge_list_final.append((i, j, d_bar, w_final))
        
        n_edges_final = len(edge_list_final)
        
        if n_edges_final > 0:
            # Convert to tensors
            edge_src = torch.tensor([e[0] for e in edge_list_final], dtype=torch.long, device=device)
            edge_dst = torch.tensor([e[1] for e in edge_list_final], dtype=torch.long, device=device)
            edge_target_d = torch.tensor([e[2] for e in edge_list_final], dtype=torch.float32, device=device)
            edge_weights = torch.tensor([e[3] for e in edge_list_final], dtype=torch.float32, device=device)
            
            # Normalize weights
            edge_weights = edge_weights / edge_weights.sum()
            
            w_raw = torch.tensor([e[3] for e in edge_list_filtered], dtype=torch.float32)
            w_final_t = torch.tensor([e[3] for e in edge_list_final], dtype=torch.float32)
            
            print(f"[DGSO-V2-WEIGHTS] w_raw: p10={w_raw.quantile(0.1).item():.4f} "
                  f"p50={w_raw.median().item():.4f} p90={w_raw.quantile(0.9).item():.4f}")
            print(f"[DGSO-V2-WEIGHTS] w_final (normalized): p10={edge_weights.quantile(0.1).item():.6f} "
                  f"p50={edge_weights.median().item():.6f} p90={edge_weights.quantile(0.9).item():.6f}")
            
            # Check correlation (should be somewhat negative if spread penalty works)
            if len(kept_spreads) > 10:
                from scipy.stats import pearsonr
                corr, _ = pearsonr(kept_spreads.numpy(), w_final_t.numpy())
                print(f"[DGSO-V2-WEIGHTS-CORR] corr(w_final, rel_spread)={corr:.3f}")
        
        # ---------------------------------------------------------------
        # STEP 4: Build neighbor sets for radius term
        # ---------------------------------------------------------------
        if n_edges_final > 100 and dgso_radius_lambda > 0:
            print(f"\n[DGSO-V2-RADIUS] Building neighbor sets for anti-collapse term...")
            
            # Build adjacency from edges
            from collections import defaultdict
            neighbors = defaultdict(list)
            neighbor_dists = defaultdict(list)
            neighbor_weights = defaultdict(list)
            
            for idx, (i, j, d_bar, w_final) in enumerate(edge_list_final):
                neighbors[i].append(j)
                neighbors[j].append(i)
                neighbor_dists[i].append(d_bar)
                neighbor_dists[j].append(d_bar)
                neighbor_weights[i].append(w_final)
                neighbor_weights[j].append(w_final)
            
            # Compute target radius per node (weighted median of neighbor distances)
            target_radius = torch.zeros(n_sc, device=device)
            has_neighbors = torch.zeros(n_sc, dtype=torch.bool, device=device)
            
            for node_i in neighbors:
                dists_i = torch.tensor(neighbor_dists[node_i])
                weights_i = torch.tensor(neighbor_weights[node_i])
                
                if len(dists_i) > 0:
                    # Weighted median
                    sort_idx = torch.argsort(dists_i)
                    dists_sorted = dists_i[sort_idx]
                    weights_sorted = weights_i[sort_idx]
                    cum_w = weights_sorted.cumsum(dim=0)
                    total_w = cum_w[-1]
                    median_idx = (cum_w >= total_w / 2).nonzero(as_tuple=True)[0][0].item()
                    target_radius[node_i] = dists_sorted[median_idx].item()
                    has_neighbors[node_i] = True
            
            n_with_neighbors = has_neighbors.sum().item()
            print(f"[DGSO-V2-RADIUS] Nodes with neighbors: {n_with_neighbors}/{n_sc}")
            print(f"[DGSO-V2-RADIUS] target_radius: p10={target_radius[has_neighbors].quantile(0.1).item():.4f} "
                  f"p50={target_radius[has_neighbors].median().item():.4f} "
                  f"p90={target_radius[has_neighbors].quantile(0.9).item():.4f}")
        else:
            dgso_radius_lambda = 0  # Disable if not enough edges
            has_neighbors = None
        
        # ---------------------------------------------------------------
        # STEP 5: Optimization with full trust region + two-phase schedule
        # ---------------------------------------------------------------
        if n_edges_final < 100:
            print(f"[DGSO-v2] WARNING: Only {n_edges_final} edges after filtering, skipping optimization")
        else:
            print(f"\n[DGSO-V2-OPT] Starting optimization with {n_edges_final} trusted edges...")
            print(f"[DGSO-V2-OPT] iters={dgso_iters} lr={dgso_lr} batch_size={dgso_batch_size}")
            print(f"[DGSO-V2-OPT] huber_delta={dgso_huber_delta} anchor_lambda={dgso_anchor_lambda}")
            print(f"[DGSO-V2-OPT] radius_lambda={dgso_radius_lambda}")
            if dgso_two_phase:
                print(f"[DGSO-V2-OPT] TWO-PHASE: phase1={dgso_phase1_iters} iters @ {dgso_phase1_anchor_mult}x anchor")
            
            # Initialize X from current global (2D) - this is X^(0)
            X_dgso = X_global[:, :2].clone().detach().to(device).requires_grad_(True)
            X_init = X_global[:, :2].clone().detach().to(device)  # Trust region anchor (ALL points)
            
            optimizer = torch.optim.Adam([X_dgso], lr=dgso_lr)
            
            # Huber loss function
            def huber_loss(r, delta):
                abs_r = torch.abs(r)
                return torch.where(
                    abs_r <= delta,
                    0.5 * r**2,
                    delta * (abs_r - 0.5 * delta)
                )
            
            # For tracking
            dgso_knn_history = {k: [] for k in debug_k_list} if debug_knn else {}
            dgso_loss_history = []
            dgso_knn_at_iter0 = None
            
            # Fixed subset for kNN tracking
            if debug_knn and gt_coords is not None:
                dgso_knn_subset = global_knn_stage_subset if global_knn_stage_subset is not None else torch.randperm(n_sc)[:min(n_sc, debug_global_subset)]
                dgso_knn_gt = gt_coords[dgso_knn_subset].float().to(device)
            
            # Track displacement
            max_displacement_threshold = 0.2 * rms_target
            
            for iter_idx in range(dgso_iters):
                optimizer.zero_grad()
                
                # Two-phase anchor schedule
                if dgso_two_phase and iter_idx < dgso_phase1_iters:
                    current_anchor_lambda = dgso_anchor_lambda * dgso_phase1_anchor_mult
                else:
                    current_anchor_lambda = dgso_anchor_lambda
                
                # Mini-batch sampling
                if n_edges_final <= dgso_batch_size:
                    batch_idx = torch.arange(n_edges_final, device=device)
                else:
                    batch_idx = torch.randperm(n_edges_final, device=device)[:dgso_batch_size]
                
                # Get batch edges
                src_batch = edge_src[batch_idx]
                dst_batch = edge_dst[batch_idx]
                d_target_batch = edge_target_d[batch_idx]
                w_batch = edge_weights[batch_idx]
                
                # Compute predicted distances
                d_pred_batch = torch.norm(X_dgso[src_batch] - X_dgso[dst_batch], dim=1)
                
                # Residuals
                r_batch = d_pred_batch - d_target_batch
                
                # Weighted Huber loss (edge stress)
                huber_vals = huber_loss(r_batch, dgso_huber_delta)
                loss_edge = (w_batch * huber_vals).sum() / w_batch.sum()
                
                # FULL trust region anchor (ALL points, not subset!)
                displacement = X_dgso - X_init
                loss_anchor = (displacement**2).mean()
                
                # Anti-collapse radius term
                loss_radius = torch.tensor(0.0, device=device)
                if dgso_radius_lambda > 0 and has_neighbors is not None:
                    # Compute current radius per node
                    with torch.no_grad():
                        nodes_with_nbrs = has_neighbors.nonzero(as_tuple=True)[0]
                        if len(nodes_with_nbrs) > 0:
                            radius_residuals = []
                            for node_i in nodes_with_nbrs[:1000].tolist():  # Sample for efficiency
                                nbrs_i = neighbors[node_i]
                                if len(nbrs_i) > 0:
                                    nbrs_tensor = torch.tensor(nbrs_i, device=device)
                                    dists_to_nbrs = torch.norm(X_dgso[node_i] - X_dgso[nbrs_tensor], dim=1)
                                    r_i_current = dists_to_nbrs.median()
                                    r_i_target = target_radius[node_i]
                                    radius_residuals.append((r_i_current - r_i_target).abs())
                            
                            if radius_residuals:
                                loss_radius = torch.stack(radius_residuals).mean()
                
                # Total loss
                loss = loss_edge + current_anchor_lambda * loss_anchor + dgso_radius_lambda * loss_radius
                
                loss.backward()
                optimizer.step()
                
                # Gauge-fix: recenter and rescale
                with torch.no_grad():
                    X_dgso.data = X_dgso.data - X_dgso.data.mean(dim=0, keepdim=True)
                    rms_current = X_dgso.data.pow(2).mean().sqrt()
                    X_dgso.data = X_dgso.data * (rms_target / (rms_current + 1e-8))
                
                # Logging
                if iter_idx % dgso_log_every == 0 or iter_idx == dgso_iters - 1:
                    dgso_loss_history.append(loss.item())
                    
                    # Compute shape stats and displacement
                    with torch.no_grad():
                        cov_dgso = torch.cov(X_dgso.T)
                        eigs_dgso = torch.linalg.eigvalsh(cov_dgso)
                        eig_ratio_dgso = (eigs_dgso.max() / eigs_dgso.min().clamp(min=1e-8)).item()
                        rms_dgso = X_dgso.pow(2).mean().sqrt().item()
                        
                        # RMS displacement from X0
                        displacement_rms = ((X_dgso - X_init)**2).mean().sqrt().item()
                        
                        # Edge residual stats
                        sample_size = min(10000, n_edges_final)
                        sample_idx = torch.randperm(n_edges_final, device=device)[:sample_size]
                        d_pred_sample = torch.norm(X_dgso[edge_src[sample_idx]] - X_dgso[edge_dst[sample_idx]], dim=1)
                        resid_sample = torch.abs(d_pred_sample - edge_target_d[sample_idx])
                    
                    phase_str = "P1" if dgso_two_phase and iter_idx < dgso_phase1_iters else "P2"
                    print(f"[DGSO-V2-OPT] [{phase_str}] iter={iter_idx}: loss={loss.item():.4f} "
                          f"loss_edge={loss_edge.item():.4f} loss_anchor={loss_anchor.item():.4f}")
                    print(f"[DGSO-V2-OPT]   rms={rms_dgso:.4f} eig_ratio={eig_ratio_dgso:.1f} ||X-X0||_RMS={displacement_rms:.4f}")
                    print(f"[DGSO-V2-OPT]   edge_abs_resid: p50={resid_sample.median().item():.4f} "
                          f"p90={resid_sample.quantile(0.9).item():.4f}")
                    
                    # kNN tracking
                    if debug_knn and gt_coords is not None:
                        with torch.no_grad():
                            X_dgso_subset = X_dgso[dgso_knn_subset]
                            
                            for k_val in debug_k_list:
                                if len(dgso_knn_subset) > k_val + 1:
                                    knn_pred, _ = _knn_indices_dists(X_dgso_subset, k_val)
                                    knn_gt, _ = _knn_indices_dists(dgso_knn_gt, k_val)
                                    overlap = _knn_overlap_score(knn_pred, knn_gt)
                                    dgso_knn_history[k_val].append(overlap.mean().item())
                            
                            knn_str = " ".join([f"kNN@{k}={dgso_knn_history[k][-1]:.3f}" for k in debug_k_list if dgso_knn_history[k]])
                            print(f"[DGSO-V2-KNN] iter={iter_idx}: {knn_str}")
                            
                            # Store iter 0 kNN for fail-safe
                            if iter_idx == 0:
                                dgso_knn_at_iter0 = {k: dgso_knn_history[k][-1] for k in debug_k_list if dgso_knn_history[k]}
                    
                    # ---------------------------------------------------------------
                    # FAIL-SAFES (Step 6)
                    # ---------------------------------------------------------------
                    
                    # Check 1: eig_ratio explosion (shape collapse)
                    if eig_ratio_dgso > 200:
                        print(f"[DGSO-V2-STOP] eig_ratio={eig_ratio_dgso:.1f} > 200, stopping early")
                        break
                    
                    # Check 2: displacement exceeded threshold
                    if displacement_rms > max_displacement_threshold:
                        print(f"[DGSO-V2-STOP] ||X-X0||_RMS={displacement_rms:.4f} > {max_displacement_threshold:.4f}, stopping early")
                        break
                    
                    # Check 3: kNN dropped too much from iter 0 (if GT available)
                    if debug_knn and gt_coords is not None and dgso_knn_at_iter0 is not None and iter_idx > 0:
                        k_main = debug_k_list[0]
                        if k_main in dgso_knn_history and len(dgso_knn_history[k_main]) > 0:
                            current_knn = dgso_knn_history[k_main][-1]
                            iter0_knn = dgso_knn_at_iter0.get(k_main, current_knn)
                            knn_drop = iter0_knn - current_knn
                            if knn_drop > 0.05:
                                print(f"[DGSO-V2-STOP] kNN@{k_main} dropped by {knn_drop:.3f} (from {iter0_knn:.3f} to {current_knn:.3f}), stopping early")
                                break
                
                # Check for no improvement (convergence)
                if iter_idx > 300 and len(dgso_loss_history) > 3:
                    recent_losses = dgso_loss_history[-3:]
                    if abs(recent_losses[-1] - recent_losses[0]) < 1e-6:
                        print(f"[DGSO-V2-NOIMPROVE] Loss plateaued, stopping early at iter={iter_idx}")
                        break
            
            # Update X_global with DGSO result
            with torch.no_grad():
                X_global[:, :2] = X_dgso.detach()
            
            print(f"\n[DGSO-V2-OPT] Optimization complete")
            
            # ---------------------------------------------------------------
            # Update stagewise tracking
            # ---------------------------------------------------------------
            if debug_knn and gt_coords is not None and global_knn_stage_subset is not None:
                with torch.no_grad():
                    X_dgso_subset = X_global[global_knn_stage_subset, :2].float()
                    
                    dgso_knn_scores = {}
                    for k_val in debug_k_list:
                        if len(global_knn_stage_subset) > k_val + 1:
                            knn_pred, _ = _knn_indices_dists(X_dgso_subset, k_val)
                            knn_gt, _ = _knn_indices_dists(global_knn_stage_gt, k_val)
                            overlap = _knn_overlap_score(knn_pred, knn_gt)
                            dgso_knn_scores[k_val] = overlap.mean().item()
                    
                    global_knn_stage_results['dgso'] = dgso_knn_scores
                    
                    knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in dgso_knn_scores.items()])
                    print(f"[GLOBAL-KNN-STAGE] dgso-v2 (post-distance-refine): {knn_str}")
                    
                    # Compute deltas
                    if 'pgso' in global_knn_stage_results:
                        k_main = debug_k_list[0]
                        pgso_main = global_knn_stage_results['pgso'].get(k_main, 0)
                        dgso_main = dgso_knn_scores.get(k_main, 0)
                        delta_pgso_dgso = dgso_main - pgso_main
                        
                        init_main = global_knn_stage_results.get('init', {}).get(k_main, 0)
                        delta_total = dgso_main - init_main
                        
                        print(f"[GLOBAL-KNN-STAGE] Δ(pgso→dgso-v2) kNN@{k_main}={delta_pgso_dgso:+.3f}")
                        print(f"[GLOBAL-KNN-STAGE] Δ(total) kNN@{k_main}={delta_total:+.3f}")
                        
                        if delta_pgso_dgso > 0.02:
                            print(f"[GLOBAL-KNN-STAGE] ✓ DGSO-v2 HELPS: Improved kNN by {delta_pgso_dgso:.3f}")
                        elif delta_pgso_dgso < -0.02:
                            print(f"[GLOBAL-KNN-STAGE] ⚠️ DGSO-v2 HURTS: Decreased kNN by {abs(delta_pgso_dgso):.3f}")
                            print(f"    → Consider: tighter filtering (lower tau_spread) or higher anchor_lambda")
                        else:
                            print(f"[GLOBAL-KNN-STAGE] → DGSO-v2 has minimal effect (Δ={delta_pgso_dgso:+.3f})")
                            print(f"    → Stitching may not be the main bottleneck")
        
        print("="*70 + "\n")
    # ===================================================================
    # [TEST2-SCATTER-POSTREFINE] Scatter AFTER refinement
    # ===================================================================
    # Goal: Check if refinement reduced per-cell scatter.
    # If scatter unchanged, mismatch is non-rigid / not capturable by Sim(2).
    # ===================================================================
    if pgso_refine_iters > 0 and debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[TEST2-SCATTER-POSTREFINE] SCATTER AFTER REFINEMENT")
        print("="*70)
        
        with torch.no_grad():
            # Find cells with coverage >= 3
            high_cov_cells = [i for i in range(n_sc) if len(memberships[i]) >= 3]
            
            if len(high_cov_cells) >= 50:
                scatter_list_post = []
                
                # Sample for efficiency
                sample_cells = high_cov_cells[:min(1000, len(high_cov_cells))]
                
                for cell_i in sample_cells:
                    patches_with_i = memberships[cell_i]
                    
                    # Collect coords for cell_i from each patch (post-refine transforms)
                    coords_i = []
                    for p_idx in patches_with_i:
                        if p_idx >= len(patch_coords_transformed):
                            continue
                        patch_cell_list = patch_indices[p_idx].tolist()
                        if cell_i in patch_cell_list:
                            local_pos = patch_cell_list.index(cell_i)
                            # Use the updated patch_coords_transformed from refinement
                            coord_from_patch = patch_coords_transformed[p_idx][local_pos]
                            coords_i.append(coord_from_patch)
                    
                    if len(coords_i) >= 3:
                        coords_i_stack = torch.stack(coords_i)  # (n_patches_i, 2)
                        mean_coord = coords_i_stack.mean(dim=0)
                        distances_to_mean = torch.norm(coords_i_stack - mean_coord.unsqueeze(0), dim=1)
                        scatter_i = distances_to_mean.median().item()
                        scatter_list_post.append(scatter_i)
                
                if scatter_list_post:
                    scatter_post_t = torch.tensor(scatter_list_post)
                    
                    print(f"[TEST2-SCATTER-POSTREFINE] n_cells_used={len(scatter_list_post)}")
                    print(f"[TEST2-SCATTER-POSTREFINE] scatter: "
                          f"p10={scatter_post_t.quantile(0.1).item():.4f} "
                          f"p50={scatter_post_t.median().item():.4f} "
                          f"p90={scatter_post_t.quantile(0.9).item():.4f}")
                    
                    scatter_norm_post = scatter_post_t.median().item() / rms_target
                    print(f"[TEST2-SCATTER-POSTREFINE] scatter / RMS = {scatter_norm_post:.3f}")
                    
                    # Compare to pre-refine scatter (if available)
                    # The pre-refine scatter was stored in the original TEST2-SCATTER
                    # We need to compare
                    print(f"\n[TEST2-SCATTER-POSTREFINE] Compare to pre-refine:")
                    print(f"    → Check [TEST2-SCATTER] above for pre-refine value")
                    
                    if scatter_norm_post < 0.1:
                        print(f"    ✓ Post-refine scatter is small (<10% RMS)")
                    else:
                        print(f"    ⚠️ Post-refine scatter still significant ({scatter_norm_post*100:.0f}% RMS)")
                        print(f"       → Refinement did NOT eliminate per-cell disagreement")
                        print(f"       → Mismatch may be non-rigid (need affine or local warps)")
            else:
                print(f"[TEST2-SCATTER-POSTREFINE] ⚠️ Not enough high-coverage cells")
        
        print("="*70 + "\n")

    # ===================================================================
    # [GLOBAL-KNN-STAGE] Stage 3: After refinement (final)
    # ===================================================================
    if debug_knn and gt_coords is not None and global_knn_stage_subset is not None:
        with torch.no_grad():
            X_refine_subset = X_global[global_knn_stage_subset, :2].float()  # Use 2D
            
            stage3_knn_scores = {}
            for k_val in debug_k_list:
                if len(global_knn_stage_subset) > k_val + 1:
                    knn_pred, _ = _knn_indices_dists(X_refine_subset, k_val)
                    knn_gt, _ = _knn_indices_dists(global_knn_stage_gt, k_val)
                    overlap = _knn_overlap_score(knn_pred, knn_gt)
                    stage3_knn_scores[k_val] = overlap.mean().item()
            
            global_knn_stage_results['refine'] = stage3_knn_scores
            
            if DEBUG_FLAG:
                knn_str = " ".join([f"kNN@{k}={v:.3f}" for k, v in stage3_knn_scores.items()])
                print(f"[GLOBAL-KNN-STAGE] refine (post-refine): {knn_str}")

        
        # ===============================================================
        # [GLOBAL-KNN-STAGE] SUMMARY: Compare all stages
        # ===============================================================
        print("\n" + "="*70)
        print("[GLOBAL-KNN-STAGE] STAGEWISE kNN SUMMARY")
        print("="*70)
        
        for k_val in debug_k_list:
            init_score = global_knn_stage_results.get('init', {}).get(k_val, float('nan'))
            pgso_score = global_knn_stage_results.get('pgso', {}).get(k_val, float('nan'))
            refine_score = global_knn_stage_results.get('refine', {}).get(k_val, float('nan'))
            
            delta_pgso = pgso_score - init_score
            delta_refine = refine_score - pgso_score
            delta_total = refine_score - init_score
            
            print(f"[GLOBAL-KNN-STAGE] k={k_val}:")
            print(f"    init={init_score:.3f} → pgso={pgso_score:.3f} → refine={refine_score:.3f}")
            print(f"    Δ(init→pgso)={delta_pgso:+.3f}, Δ(pgso→refine)={delta_refine:+.3f}, Δ(total)={delta_total:+.3f}")
        
        # Interpretation
        k_main = debug_k_list[0]
        init_main = global_knn_stage_results.get('init', {}).get(k_main, 0)
        pgso_main = global_knn_stage_results.get('pgso', {}).get(k_main, 0)
        refine_main = global_knn_stage_results.get('refine', {}).get(k_main, 0)
        
        print()
        
        # Check if refinement was actually run
        if pgso_refine_iters == 0:
            print(f"[GLOBAL-KNN-STAGE] Note: n_align_iters<=1, so no refinement was run (refine=pgso)")
            refine_main = pgso_main  # They're the same
        
        if abs(pgso_main - init_main) < 0.02 and abs(refine_main - pgso_main) < 0.02:
            print(f"[GLOBAL-KNN-STAGE] ⚠️ kNN@{k_main} is FLAT across all stages ({init_main:.3f}→{refine_main:.3f})")
            print(f"    → Neither PGSO nor refinement improves kNN vs GT")
            print(f"    → The problem is UPSTREAM: either 32D→2D projection or generator output")
            print(f"    → Check [PATCH-KNN-VS-GT-2D] to see if projection is the issue")
        elif init_main < 0.40 and abs(refine_main - init_main) < 0.03:
            print(f"[GLOBAL-KNN-STAGE] ⚠️ kNN@{k_main} starts low ({init_main:.3f}) and stays low")
            print(f"    → Initial embedding already has wrong neighborhood structure")
            print(f"    → PGSO and refinement cannot recover what was never there")
        elif pgso_main < init_main - 0.03:
            print(f"[GLOBAL-KNN-STAGE] ⚠️ PGSO HURTS: kNN@{k_main} drops from {init_main:.3f} to {pgso_main:.3f}")
            print(f"    → Pose-graph stitching is damaging neighborhood structure")
        elif refine_main > pgso_main + 0.03:
            print(f"[GLOBAL-KNN-STAGE] ✓ REFINE HELPS: kNN@{k_main} improves from {pgso_main:.3f} to {refine_main:.3f}")
        else:
            print(f"[GLOBAL-KNN-STAGE] Stages have minimal effect on kNN@{k_main}")
            print(f"    → Issue is likely in the 2D readout or generator geometry")
        
        print("="*70 + "\n")

 
    # ===============================================================
    # Post-refinement diagnostics
    # ===============================================================
    if pgso_refine_iters > 0 and DEBUG_FLAG:
        # [TEST4-REFINE-SUMMARY] Did refinement improve kNN vs GT?
        if debug_knn and refine_knn_scores:
            print(f"\n[TEST4-REFINE-SUMMARY] kNN vs GT across refinement:")
            for k_val in debug_k_list:
                if k_val in refine_knn_scores and len(refine_knn_scores[k_val]) >= 2:
                    scores = refine_knn_scores[k_val]
                    delta = scores[-1] - scores[0]
                    best_iter = int(np.argmax(scores))
                    print(f"  [TEST4-REFINE-SUMMARY] kNN@{k_val}: "
                          f"{scores[0]:.3f} → {scores[-1]:.3f} (Δ={delta:+.3f})")
                    print(f"    best_iter_by_knn{k_val}={best_iter}, best_score={max(scores):.3f}")
                    
                    if delta > 0.02:
                        print(f"    ✓ Refinement IMPROVED kNN@{k_val}")
                    elif delta < -0.02:
                        print(f"    ⚠️ Refinement HURT kNN@{k_val}")
                    else:
                        print(f"    → Refinement had minimal effect on kNN@{k_val}")
            
            # Compare coord mismatch improvement vs kNN improvement
            if len(refine_coord_mismatches) >= 2 and any(len(v) >= 2 for v in refine_knn_scores.values()):
                delta_coord = refine_coord_mismatches[0] - refine_coord_mismatches[-1]
                k_main = debug_k_list[0]
                delta_knn = refine_knn_scores[k_main][-1] - refine_knn_scores[k_main][0] if k_main in refine_knn_scores else 0
                
                print(f"\n  [TEST4-REFINE-SUMMARY] delta_coord_mismatch={delta_coord:.3f}")
                if delta_coord > 0.05 and delta_knn < 0.01:
                    print(f"    ⚠️ DIAGNOSTIC: Coord mismatch improved but kNN didn't")
                    print(f"       → Refinement is enforcing overlap agreement, NOT recovering true neighbors")
                    print(f"       → The ceiling is from patching/model, not stitching")
        
        # Check if solution collapsed (existing code follows)

    if pgso_refine_iters > 0 and DEBUG_FLAG:
        # Check if solution collapsed
        cov_check = torch.cov(X_global[:, :2].T)
        eigs_check = torch.linalg.eigvalsh(cov_check)
        eig_ratio_check = (eigs_check.max() / eigs_check.min().clamp(min=1e-8)).item()
        
        print(f"\n[PGSO-REFINE] SUMMARY:")
        print(f"  Final RMS={X_global[:,:2].pow(2).mean().sqrt().item():.3f} "
              f"eig_ratio={eig_ratio_check:.1f}")
        
        if len(refine_coord_mismatches) >= 2:
            improvement = refine_coord_mismatches[0] - refine_coord_mismatches[-1]
            print(f"  Coord mismatch/RMS: {refine_coord_mismatches[0]:.3f} → {refine_coord_mismatches[-1]:.3f} "
                  f"(Δ={improvement:.3f})")
            if improvement > 0.05:
                print(f"    ✓ Refinement improved overlap agreement")
            elif improvement < -0.05:
                print(f"    ⚠️ WARNING: Refinement made overlap agreement worse!")
            else:
                print(f"    → Refinement had minimal effect on overlap agreement")
        
        if len(refine_procrustes_residuals) >= 2:
            print(f"  Procrustes residual p50: {refine_procrustes_residuals[0]:.4f} → {refine_procrustes_residuals[-1]:.4f}")
        
        if eig_ratio_check > 100:
            print(f"    ⚠️ WARNING: Solution may be collapsing (eig_ratio > 100)")
        elif eig_ratio_check > 20:
            print(f"    ⚠️ CAUTION: Solution is becoming anisotropic (eig_ratio={eig_ratio_check:.1f})")
        else:
            print(f"    ✓ Solution shape is reasonable (eig_ratio={eig_ratio_check:.1f})")



    # Expand X_global back to D_latent dimensions for compatibility with downstream code
    # We'll use the 2D coords for now since they're what we actually computed
    # Pad with zeros for dimensions 3..D_latent
    if D_latent > 2:
        X_global_full = torch.zeros(n_sc, D_latent, dtype=torch.float32, device=device)
        X_global_full[:, :2] = X_global
        X_global = X_global_full

    # Store R_list, s_list, t_list for diagnostics compatibility
    R_list = [R.to(device) if R.shape[0] == D_latent else torch.eye(D_latent, device=device) for R in R_global_list]
    s_list = [s.to(device) if torch.is_tensor(s) else torch.tensor(s, device=device) for s in s_global_list]
    t_list = [t.to(device) if t.shape[0] == D_latent else torch.zeros(D_latent, device=device) for t in t_global_list]

    # ===================================================================
    # POST-PGSO DIAGNOSTICS
    # ===================================================================

    # ===================================================================
    # [PATCH-KNN-VS-GT] POST-STITCH: Compare final global coords to GT
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[PATCH-KNN-VS-GT] POST-STITCH ANALYSIS")
        print("="*70)
        
        with torch.no_grad():
            gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
            # Use subset for efficiency
            # M = min(n_sc, debug_global_subset)
            # subset_idx = torch.randperm(n_sc)[:M]

            M = min(n_sc, debug_global_subset)
            subset_idx = torch.randperm(n_sc, device=device)[:M]

            
            X_subset = X_global[subset_idx].float()
            gt_subset = gt_coords_t[subset_idx]
            
            # Compute coverage per cell
            cover_counts = torch.tensor([len(memberships[i]) for i in subset_idx.tolist()], 
                                        dtype=torch.float32, device=device)
            
            for cov_thresh, label in [(2, "coverage>=2"), (1, "coverage==1")]:
                if cov_thresh == 1:
                    mask = (cover_counts == 1)
                else:
                    mask = (cover_counts >= cov_thresh)
                
                n_in_split = mask.sum().item()
                if n_in_split < 50:
                    print(f"  [{label}] n={n_in_split} - too few for analysis")
                    continue
                
                X_split = X_subset[mask]
                gt_split = gt_subset[mask]
                
                for k_val in debug_k_list:
                    if n_in_split > k_val + 1:
                        knn_pred, _ = _knn_indices_dists(X_split, k_val)
                        knn_gt, _ = _knn_indices_dists(gt_split, k_val)
                        overlap = _knn_overlap_score(knn_pred, knn_gt)
                        
                        print(f"  [PATCH-KNN-VS-GT][POST-STITCH] {label} n={n_in_split} "
                              f"k={k_val} mean={overlap.mean().item():.3f} "
                              f"p50={overlap.median().item():.3f}")


    # ===================================================================
    # [NN-GAP] Analysis: Why Spearman looks fine but kNN is bad
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[NN-GAP] ANALYSIS")
        print("="*70)
        
        with torch.no_grad():
            gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
            # Use subset
            M = min(n_sc, debug_global_subset)
            subset_idx = torch.randperm(n_sc)[:M]
            
            X_subset = X_global[subset_idx].float()
            gt_subset = gt_coords_t[subset_idx]
            
            # Compute NN-gap for GT and Pred
            gap_gt, gap_ratio_gt = _nn_gap_stats(gt_subset, debug_gap_k)
            gap_pred, gap_ratio_pred = _nn_gap_stats(X_subset, debug_gap_k)
            
            print(f"  [NN-GAP] GT gap_ratio({debug_gap_k}->{debug_gap_k+1}): "
                  f"p10={gap_ratio_gt.quantile(0.1).item():.4f} "
                  f"p50={gap_ratio_gt.quantile(0.5).item():.4f} "
                  f"p90={gap_ratio_gt.quantile(0.9).item():.4f}")
            
            print(f"  [NN-GAP] PR gap_ratio({debug_gap_k}->{debug_gap_k+1}): "
                  f"p10={gap_ratio_pred.quantile(0.1).item():.4f} "
                  f"p50={gap_ratio_pred.quantile(0.5).item():.4f} "
                  f"p90={gap_ratio_pred.quantile(0.9).item():.4f}")
            
            # Interpretation
            if gap_ratio_gt.median() < 0.05:
                print("  ⚠️ GT has very small gap_ratio → kNN@k is inherently fragile (near-ties)")
            if gap_ratio_pred.median() < gap_ratio_gt.median() * 0.5:
                print("  ⚠️ Pred gap_ratio << GT → predicted space is 'flattened' (scrambles neighbors)")
        
        print("="*70 + "\n")


    # ===================================================================
    # [LOCAL-DENSITY-RATIO] Compare local radius in pred vs GT
    # ===================================================================
    if debug_knn and gt_coords is not None:
        print("\n" + "="*70)
        print("[LOCAL-DENSITY-RATIO] ANALYSIS")
        print("="*70)
        
        with torch.no_grad():
            gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
            
            # Use subset
            M = min(n_sc, debug_global_subset)
            subset_idx = torch.randperm(n_sc)[:M]
            
            X_subset = X_global[subset_idx].float()
            gt_subset = gt_coords_t[subset_idx]
            
            # Compute distance to k-th neighbor in both spaces
            for k_density in [10, 20]:
                # GT: distance to k-th neighbor
                D_gt = torch.cdist(gt_subset, gt_subset)
                D_gt.fill_diagonal_(float('inf'))
                d_gt_sorted, _ = D_gt.topk(k_density, largest=False, dim=1)
                r_gt_k = d_gt_sorted[:, -1]  # distance to k-th neighbor
                
                # Pred: distance to k-th neighbor
                D_pr = torch.cdist(X_subset, X_subset)
                D_pr.fill_diagonal_(float('inf'))
                d_pr_sorted, _ = D_pr.topk(k_density, largest=False, dim=1)
                r_pr_k = d_pr_sorted[:, -1]
                
                # Ratio
                ratio = r_pr_k / r_gt_k.clamp(min=1e-8)
                
                print(f"  [LOCAL-DENSITY-RATIO] k={k_density}")
                print(f"    r_gt({k_density}): p50={r_gt_k.median().item():.4f}")
                print(f"    r_pr({k_density}): p50={r_pr_k.median().item():.4f}")
                print(f"    ratio r_pr/r_gt: p10={ratio.quantile(0.1).item():.2f} "
                      f"p50={ratio.median().item():.2f} "
                      f"p90={ratio.quantile(0.9).item():.2f}")
            
            # Interpretation
            ratio_median = ratio.median().item()
            ratio_spread = ratio.quantile(0.9).item() - ratio.quantile(0.1).item()
            
            if ratio_spread > 2.0:
                print("  → ⚠️ HIGH SPREAD: Local density/scale varies wildly → kills local-edge correlation")
            if ratio_median > 2.0 or ratio_median < 0.5:
                print(f"  → ⚠️ SHIFTED: Predicted local scale is {'expanded' if ratio_median > 1 else 'compressed'}")
        
        print("="*70 + "\n")




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


        # [DEBUG-SHAPE-2D] Compute stats on ACTUAL 2D coords, not padded D_latent tensor
        # (Fix A from ChatGPT: avoid misleading RMS/covariance on zero-padded dims)
        X_2d = X_full[:, :2]  # Extract actual 2D coordinates
        coords_rms_2d = X_2d.pow(2).mean().sqrt().item()
        print(f"[DEBUG-SHAPE-2D] coords_rms (2D only)={coords_rms_2d:.3f}")
        
        cov_2d = torch.cov(X_2d.float().T)
        eigs_2d = torch.linalg.eigvalsh(cov_2d)
        ratio_2d = float(eigs_2d.max() / (eigs_2d.min().clamp(min=1e-8)))
        print(f"[DEBUG-SHAPE-2D] coord_cov eigs (2D): "
              f"min={eigs_2d.min():.3e} "
              f"max={eigs_2d.max():.3e} "
              f"ratio={ratio_2d:.1f}")
        
        # Also print full D_latent stats for reference (but mark clearly)
        coords_rms_full = X_full.pow(2).mean().sqrt().item()
        print(f"[GLOBAL] coords_rms (full D={X_full.shape[1]}, mostly zeros)={coords_rms_full:.3f}")



    if return_coords:
        n = D_edm.shape[0]
        Jn = torch.eye(n) - torch.ones(n, n) / n
        B = -0.5 * (Jn @ (D_edm**2) @ Jn)
        coords = uet.classical_mds(B.to(device), d_out=2).detach().cpu()
        coords_canon = uet.canonicalize_coords(coords).detach().cpu()
        result["coords"] = coords
        result["coords_canon"] = coords_canon


    # # Cleanup GPU tensors to allow process exit
    # # Cleanup GPU tensors to allow process exit
    # del Xd, D, X_full
    # # Don't delete X_global yet if we need it for two-pass
    # if not (two_pass and _pass_number == 1):
    #     del Z_all, patch_indices, memberships, patch_coords

    # ===================================================================
    # [LOCAL_REFINE] Post-stitch local density refinement
    # ===================================================================
    if local_refine:
        if DEBUG_FLAG:
            print(f"\n[LOCAL_REFINE] Running local density refinement...")
            print(f"  [LOCAL_REFINE] steps={local_refine_steps}, lr={local_refine_lr}, anchor_weight={local_refine_anchor_weight}")
        
        # Enable gradients for refinement (even if called from torch.no_grad() context)
        with torch.enable_grad():
            X_refined = X_global.clone().detach().to(device).requires_grad_(True)
            X_anchor = X_global.clone().detach().to(device)
            
            # Build embedding kNN graph for local structure
            k_local = min(20, n_sc - 1)
            Z_all_device = Z_all.to(device)
            D_Z = torch.cdist(Z_all_device, Z_all_device)
            D_Z.fill_diagonal_(float('inf'))
            _, knn_idx = D_Z.topk(k_local, largest=False, dim=1)  # (n_sc, k_local)
            
            optimizer = torch.optim.Adam([X_refined], lr=local_refine_lr)
            
            # Vectorized version for efficiency (avoid Python loop over all cells)
            # Precompute target distances for all edges
            src_idx = torch.arange(n_sc, device=device).unsqueeze(1).expand(-1, k_local).reshape(-1)
            dst_idx = knn_idx.reshape(-1)
            
            # Target distances in embedding space (fixed)
            d_Z_edges = torch.norm(Z_all_device[src_idx] - Z_all_device[dst_idx], dim=1)  # (n_sc * k_local,)
            
            # Reshape for per-cell normalization
            d_Z_per_cell = d_Z_edges.reshape(n_sc, k_local)
            d_Z_means = d_Z_per_cell.mean(dim=1, keepdim=True).clamp(min=1e-8)
            d_Z_normalized = (d_Z_per_cell / d_Z_means).reshape(-1)
            
            for step in range(local_refine_steps):
                optimizer.zero_grad()
                
                # Predicted distances (changes each step)
                d_X_edges = torch.norm(X_refined[src_idx] - X_refined[dst_idx], dim=1)
                
                # Per-cell normalization for scale-free loss
                d_X_per_cell = d_X_edges.reshape(n_sc, k_local)
                d_X_means = d_X_per_cell.mean(dim=1, keepdim=True).clamp(min=1e-8)
                d_X_normalized = (d_X_per_cell / d_X_means).reshape(-1)
                
                # Loss: match normalized distance patterns
                loss_local = ((d_X_normalized - d_Z_normalized) ** 2).mean()
                
                # Anchor loss (preserve global structure)
                loss_anchor = ((X_refined - X_anchor) ** 2).mean()
                
                # Total loss
                loss = loss_local + local_refine_anchor_weight * loss_anchor
                
                loss.backward()
                optimizer.step()
                
                if DEBUG_FLAG and (step % 20 == 0 or step == local_refine_steps - 1):
                    print(f"    [LOCAL_REFINE] step={step}: loss_local={loss_local.item():.4f} loss_anchor={loss_anchor.item():.4f}")
            
            # Use refined coordinates
            X_global = X_refined.detach()
        
        if DEBUG_FLAG:
            print(f"  [LOCAL_REFINE] ✓ Refinement complete, recomputing EDM...")
        
        # Recompute EDM with refined coordinates
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
        
        result["D_edm"] = D_edm
        
        # Also update coords if return_coords is True
        if return_coords:
            n = D_edm.shape[0]
            Jn = torch.eye(n) - torch.ones(n, n) / n
            B = -0.5 * (Jn @ (D_edm**2) @ Jn)
            coords = uet.classical_mds(B.to(device), d_out=2).detach().cpu()
            coords_canon = uet.canonicalize_coords(coords).detach().cpu()
            result["coords"] = coords
            result["coords_canon"] = coords_canon

    
    # ===================================================================
    # DEBUG: Global scale compression (final coords)
    # ===================================================================
    if DEBUG_FLAG and debug_scale_compression and debug_knn and gt_coords is not None and global_knn_stage_subset is not None:
        gt_coords_t = gt_coords.float().to(device) if not gt_coords.is_cuda else gt_coords.float()
        k_scale = 10

        Xg = X_global[global_knn_stage_subset, :2].float()
        Gg = gt_coords_t[global_knn_stage_subset, :2].float()

        if Xg.shape[0] > k_scale + 1:
            _, d_pred = _knn_indices_dists(Xg, k_scale)
            _, d_gt = _knn_indices_dists(Gg, k_scale)

            pred_scale = torch.median(d_pred[:, -1]).item()
            gt_scale = torch.median(d_gt[:, -1]).item()
            ratio = pred_scale / max(gt_scale, 1e-8)

            print(f"[SCALE-COMP] global scale ratio (pred/gt) on subset: {ratio:.3f}")

    if "cuda" in device:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


    if DEBUG_FLAG:
        pass_label = f"[PASS{_pass_number}]" if two_pass else ""
        print("=" * 72)
        print(f"STAGE D (PATCH-BASED) COMPLETE {pass_label}")
        print("=" * 72 + "\n")


    # ===================================================================
    # TWO-PASS MODE: If this is pass 1 and two_pass=True, run pass 2
    # ===================================================================
    if two_pass and _pass_number == 1:
        print("\n" + "=" * 72)
        print("[TWO-PASS] Pass 1 complete. Rebuilding patches from predicted geometry...")
        print("=" * 72 + "\n")
        
        # Get coordinates from pass 1 (use X_global before cleanup)
        # We need to NOT delete X_global yet
        coords_pass1 = X_global.detach().cpu()
        
        # Run pass 2 with geometry-based patches
        result_pass2 = sample_sc_edm_patchwise(
            sc_gene_expr=sc_gene_expr,
            encoder=encoder,
            context_encoder=context_encoder,
            score_net=score_net,
            generator=generator,
            sigma_data=sigma_data,
            target_st_p95=target_st_p95,
            n_timesteps_sample=n_timesteps_sample,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            guidance_scale=guidance_scale,
            eta=eta,
            device=device,
            patch_size=patch_size,
            coverage_per_cell=coverage_per_cell,
            n_align_iters=n_align_iters,
            return_coords=return_coords,
            DEBUG_FLAG=DEBUG_FLAG,
            DEBUG_EVERY=DEBUG_EVERY,
            fixed_patch_graph=None,  # Don't use fixed graph - build from geometry
            coral_params=coral_params,
            gt_coords=gt_coords,
            debug_knn=debug_knn,
            debug_max_patches=debug_max_patches,
            debug_k_list=debug_k_list,
            debug_global_subset=debug_global_subset,
            debug_gap_k=debug_gap_k,
            two_pass=True,
            _pass_number=2,
            _coords_from_pass1=coords_pass1,
        )
        
        # Return pass 2 results
        return result_pass2


    return result