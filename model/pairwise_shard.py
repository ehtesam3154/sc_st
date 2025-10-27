'''
optional O(N^2 pairwise operation sharding for DDP
splits gram matrix, EDM, and heat kernel computations across GPUs
'''

import torch
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any
import math 

def is_sharding_beneficial(N: int, world_size: int, threshold: int=512) -> bool:
    '''
    check if sharding should be enabled
    args:
        N: mini-set size
        world_size: number of gpus
        thresold: minimum N to trigger sharding
    returns:
        true if sharding should be used
    '''
    return world_size > 1 and N >= threshold

def get_block_assignments(N: int, block_size: int, rank: int, world_size: int) -> list:
    '''
    compute which upper-triangular blocks this rank should process
    
    args:
        N: matrix dimension
        block_size: block size for tiling
        rank: current rank
        world_size: total ranks

    returns:
        list of (i_start, i_end, j_start, j_end) tuples
    '''

    n_blocks = (N + block_size -1) // block_size 
    assignments = []

    block_id = 0

    for i in range(n_blocks):
        for j in range(i, n_blocks): #upper triangle only
            if block_id % world_size == rank:
                i_start = i * block_size
                i_end = min((i+1) * block_size, N)
                j_start = j * block_size
                j_end = min((j+1) * block_size, N)
                assignments.append((i_start, i_end, j_start, j_end))
            block_id += 1

    return assignments

def sharded_gram_loss(
        V: torch.Tensor,
        G_target: torch.Tensor,
        mask: torch.Tensor,
        rank: int, 
        world_size: int,
        block_size: int=512,
        threshold: int=512
) -> torch.Tensor:
    '''
    compute frobenius norm loss on Gram matrix with sharding.
    
    Args:
        V: (batch, n, D) latent factors
        G_target: (batch, n, n) target Gram matrix
        mask: (batch, n) boolean mask
        rank: current rank
        world_size: number of ranks
        block_size: block size for tiling
        threshold: minimum n to enable sharding
        
    Returns:
        scalar loss tensor
    '''

    batch_size, n, D = V.shape

    #fallback to single GPU if not beneficial
    if not is_sharding_beneficial(n, world_size, threshold):
        import utils_et as uet
        loss_fn = uet.FrobeniusGramLoss()
        return loss_fn(V, G_target, mask)
    
    #compute gram: G = V @ V^T
    G_pred = torch.bmm(V, V.transpose(1,2))

    #get block assignments for this rank
    assignments = get_block_assignments(n, block_size, rank, world_size)

    #compute local contribution
    local_sum = torch.tensor(0.0, device=V.device)
    local_count = torch.tensor(0.0, device=V.device)

    for i_start, i_end, j_start, j_end in assignments:
        #extract blocks
        g_pred_block = G_pred[:, i_start:i_end, j_start:j_end]
        g_target_block = G_target[:, i_start:i_end, j_start:j_end]
        mask_i = mask[:, i_start:i_end].unsqueeze(2)
        mask_j = mask[:, j_start:j_end].unsqueeze(1)
        block_mask = (mask_i * mask_j).float()

        #squared error
        diff = (g_pred_block - g_target_block) * block_mask
        local_sum += (diff ** 2).sum()
        local_count += block_mask.sum()

        #mirror contribution (if not diagonal block)
        if i_start != j_start:
            local_sum += (diff ** 2).sum()
            local_count += block_mask.sum()

    #all reduce
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    loss = local_sum / (local_count + 1e-8)
    return loss

# Heat kernel sharding is more complex - leave as TODO or use unsharded version
def sharded_heat_kernel_loss(
    V: torch.Tensor,
    L_info: Dict[str, Any],
    mask: torch.Tensor,
    rank: int,
    world_size: int,
    **kwargs
) -> torch.Tensor:
    """
    Heat kernel loss - keep unsharded for now due to complexity.
    """
    # Heat kernel involves eigendecomposition which is hard to shard efficiently
    # Just use the original implementation
    import utils_et as uet
    loss_fn = uet.HeatKernelLoss(
        use_hutchinson=True,
        num_probes=8,
        chebyshev_degree=10,
        knn_k=8,
        t_list=(0.5, 1.0),
        laplacian='sym'
    )
    return loss_fn(V, L_info, mask)

def sharded_edm_loss(
        V: torch.Tensor,
        D_target: torch.Tensor,
        mask: torch.Tensor,
        rank: int,
        world_size: int,
        block_size: int=512,
        threshold: int=512
):
    '''
    compute EDM loss with sharding.
    
    similar to sharded_gram_loss but for distance matrices
    '''

    batch_size, n, D = V.shape

    if not is_sharding_beneficial(n, world_size, threshold):
        #fallback: compute full EDM
        G = torch.bmm(V, V.transpose(1, 2))
        diag = torch.diagonal(G, dim1=1, dim2=2).unsqueeze(2)
        D_pred = torch.sqrt(torch.clam(diag + diag.transpose(1, 2) - 2* G, min=0))

        diff = (D_pred - D_target) * (mask.unsqueeze(2) * mask.unsqueeze(1)).float()
        loss = (diff ** 2).sum() / ((mask.unsqueeze(2) * mask.unsqueeze(1)).float().sum() + 1e-8)

        return loss
    
    #precompute squared norms
    V_norm_sq = (V ** 2).sum(dim=-1) #(batch, n)

    #block-wise computation
    assignments = get_block_assignments(n, block_size, rank, world_size)

    local_sum = torch.tensor(0.0, device=V.device)
    local_count = torch.tensor(0.0, device=V.device)

    for i_start, i_end, j_start, j_end in assignments:
        V_i = V[:, i_start: i_end, :]
        V_j = V[:, j_start: j_end, :]

        #D_ij = sqrt(||v_i||^2 + ||v_j||^2 - 2<v_i, v_j>)
        norm_i = V_norm_sq[:, i_start: i_end].unsqueeze(2)
        norm_j = V_norm_sq[:, j_start: j_end].unsqueeze(1)
        dot_ij = torch.bmm(V_i, V_j.transpose(1,2))

        D_pred_block = torch.sqrt(torch.clamp(norm_i + norm_j - 2 * dot_ij, min=0))
        D_target_block = D_target[:, i_start: i_end, j_start:j_end]

        mask_i = mask[:, i_start:i_end].unsqueeze(2)
        mask_j = mask[:, j_start: j_end].unsqueeze(1)
        block_mask = (mask_i * mask_j).float()

        diff = (D_pred_block - D_target_block) * block_mask
        local_sum += (diff ** 2).sum()
        local_count += block_mask.sum()

        #mirror (if off-diagonal)
        if i_start != j_start:
            local_sum += (diff ** 2).sum()
            local_count += block_mask.sum()

    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    loss = local_sum / (local_count + 1e-8)
    return loss








