'''
distributed training utils for pytorch ddp
minimal, self-contained helpers for multi-gpu training
'''

import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional
import random
import numpy as np
from tqdm import tqdm

def init_dist() -> Tuple[int, int, int, torch.device]:
    '''
    initialize distributed process group for DDP
    returns:
        (rank, local_rank, world-size, device)
    '''

    #check if launched via torchrun
    if 'RANK' not in os.environ:
        #single gpu fallback
        print('not launched via torchrun = failing back to single-gpu mode')
        return 0, 0, 1, torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    
    #DDP mode
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    #validate gpu availability
    n_gpus = torch.cuda.device_count()
    if world_size > n_gpus:
        raise RuntimeError(
            f'requested {world_size} processes but only {n_gpus} GPUs available'
            f'set CUDA_VISIBLE_DEVICES or reduce --nproc_per_node'
        )
    
    #init process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    return rank, local_rank, world_size, device

def cleanup_dist():
    '''
    clean up distributed process group
    '''
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    '''
    check if current process is rank 0
    '''
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank() -> int:
    '''
    get current process rank (0 if not distributed)
    '''
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    '''
    get world size (1 if not distributed)
    '''
    return dist.get_world_size() if dist.is_initialized() else 1

def rank_print(*args, **kwargs):
    '''print onlu on rank 0'''
    if is_main_process():
        print(*args, **kwargs)

def seed_all(base_seed: int, rank: int=0):
    '''
    set random seeds for reproducibility, with rank-dependent offset
    
    args:
        base_seed: base seed value
        rank: process rank (for deterministic per-rank variation)
    '''

    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #deterministic ops (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def rank_tqdm(iterable, desc: str = "", disable: bool = False, **kwargs):
    '''
    create a rank-aware tqdm progress bar
    
    Args:
        iterable: iterable to wrap
    '''
    rank = get_rank()
    world_size = get_world_size()

    if world_size > 1:
        desc = f'rank {rank} | {desc}' if desc else f'Rank {rank}'
        kwargs.setdefault('position', rank)
        kwargs.setdefault('leave', False)

    return tqdm(iterable, desc=desc, disable=disable, **kwargs)

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    '''
    all - reduce a tensor and compute eman across ranks
    
    args:
        tensor: tensor to reduce (must on gpu)
    returns:
        mean-reduced tensor
    '''

    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor

def broadcast_object(obj, src: int=0):
    '''
    broadcast a python object from src rank to all ranks
    args:
        obj: object to broadcast (only used on src rank)
        src: source rank
        
    returns:
        broadcasted object on all ranks
    '''

    if not dist.is_initialized():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

def gather_objects(obj, dst: int=0):
    '''
    gather python objects from all ranks to dst rank
    
    args:
        OBJ: OBJECT TO GATHER FROM THIS RANK
        dst: destination rank

    returns:
        list of objects (on dst rank) or None (on other ranks)
    '''
    if not dist.is_initialized():
        return [obj]
    
    world_size = get_world_size()
    rank = get_rank()

    if rank == dst:
        gather_list = [None] * world_size
        dist.gather_object(obj, gather_list, dst=dst)
        return gather_list
    else:
        dist.gather_object(obj, dst=dst)
        return None
    
def all_gather_objects(obj):
    '''
    gather python objects from all ranks to all ranks
    
    args:
        obj: object to gather from this rank
        
    returns:
        list of objects from all ranks
    '''
    if not dist.is_initialized():
        return [obj]
    
    world_size = get_world_size()
    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, obj)
    return gather_list



    






