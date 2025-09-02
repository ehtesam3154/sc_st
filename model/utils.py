import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import time
import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from typing import Optional, Dict, Tuple, List
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize
import ot


# =====================================================
# PART X: Graph-VAE Components
# =====================================================

class GraphVAEEncoder(nn.Module):
    """
    Graph encoder that learns latent representations from ST spot graphs.
    ⚠️ Do **not** touch `train_encoder`; its aligned embeddings are the sole conditioning signal throughout.
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Two GraphConv layers as specified
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP to output μ and log σ² FOR EACH NODE (not graph-level)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        x: node features (aligned embeddings E(X_st)) - shape (n_nodes, input_dim)
        edge_index: graph edges from K-NN adjacency
        edge_weight: optional edge weights
        batch: not used since we want node-level representations
        
        Returns:
        mu: (n_nodes, latent_dim)
        logvar: (n_nodes, latent_dim)
        """
        # Two GraphConv layers
        h = torch.relu(self.conv1(x, edge_index, edge_weight))
        h = torch.relu(self.conv2(h, edge_index, edge_weight))
        
        # NO GLOBAL POOLING - we want node-level representations
        # Output μ and log σ² for each node
        mu = self.mu_head(h)        # Shape: (n_nodes, latent_dim)
        logvar = self.logvar_head(h)  # Shape: (n_nodes, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick - works element-wise"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class GraphVAEDecoder(nn.Module):
    """
    Graph decoder that outputs 2D coordinates from latent z ONLY.
    Features are NOT passed to force geometry into z.
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Decoder takes ONLY latent z (no conditioning)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output 2D coordinates
        )
        
    def forward(self, z):
        """
        z: latent vectors (batch_size, latent_dim) ONLY
        """
        coords = self.decoder(z)
        return coords

def precompute_knn_edges(coords, k=30, device='cuda'):
    """
    Helper function to precompute K-NN edges for torch-geometric style layers.
    Uses existing graph construction utilities where possible.
    """
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = coords
        
    # Use existing construct_graph_spatial function
    from sklearn.neighbors import kneighbors_graph
    
    # Build KNN graph
    knn_graph = kneighbors_graph(
        coords_np, 
        n_neighbors=k, 
        mode='connectivity', 
        include_self=False
    )
    
    # Convert to torch-geometric format
    from torch_geometric.utils import from_scipy_sparse_matrix
    edge_index, edge_weight = from_scipy_sparse_matrix(knn_graph)
    
    # CRITICAL FIX: Ensure correct dtypes
    edge_index = edge_index.long().to(device)      # Edge indices should be long
    edge_weight = edge_weight.float().to(device)   # Edge weights should be float32
    
    return edge_index, edge_weight

class LatentDenoiser(nn.Module):
    """
    Latent-space denoiser identical to current MLP/U-Net stack but for latent dim=32.
    ⚠️ Do **not** touch `train_encoder`; its aligned embeddings are the sole conditioning signal throughout.
    """
    def __init__(self, latent_dim=32, condition_dim=128, hidden_dim=256, n_blocks=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Time embedding (reuse existing SinusoidalEmbedding)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition encoder (for aligned embeddings)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising blocks (similar to existing hierarchical blocks)
        self.denoising_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(n_blocks)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    
    def forward(self, z_noisy, t, condition):
        """
        z_noisy: noisy latent vectors (batch_size, latent_dim)
        t: timestep (batch_size,) - NOW 1D instead of 2D
        condition: aligned embeddings E(X) (batch_size, condition_dim)
        """
        batch_size = z_noisy.size(0)
        # ENSURE inputs are 2D
        if z_noisy.dim() > 2:
            z_noisy = z_noisy.squeeze()
        if condition.dim() > 2:
            condition = condition.squeeze()
            
        # Handle 1D timestep input
        if t.dim() == 1:
            t = t.unsqueeze(1)  # Make it (batch_size, 1)

        t = t.view(batch_size, 1)
        
        # Encode inputs
        z_enc = self.latent_encoder(z_noisy)
        t_enc = self.time_embed(t)
        c_enc = self.condition_encoder(condition)
        
        # Combine features
        h = z_enc + t_enc + c_enc
        
        # Apply denoising blocks
        for block in self.denoising_blocks:
            h = h + block(h)  # Residual connections
            
        # Output predicted noise
        noise_pred = self.output_head(h)
        return noise_pred

# =====================================================
# PART 1: Advanced Network Components
# =====================================================

class FeatureNet(nn.Module):
    def __init__(self, n_genes, n_embedding=[512, 256, 128], dp=0):
        super(FeatureNet, self).__init__()

        self.fc1 = nn.Linear(n_genes, n_embedding[0])
        self.bn1 = nn.LayerNorm(n_embedding[0])
        self.fc2 = nn.Linear(n_embedding[0], n_embedding[1])
        self.bn2 = nn.LayerNorm(n_embedding[1])
        self.fc3 = nn.Linear(n_embedding[1], n_embedding[2])
        
        self.dp = nn.Dropout(dp)
        
    def forward(self, x, isdp=False):
        if isdp:
            x = self.dp(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: (batch_size, 1) or (batch_size,)
        Returns: (batch_size, dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # Make (batch_size, 1)
        
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb.unsqueeze(0)  # (batch_size, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (batch_size, dim)
        
        if emb.size(1) != self.dim:
            # Handle odd dimensions
            emb = emb[:, :self.dim]
            
        return emb

import torch.optim as optim   
from geomloss import SamplesLoss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class CellTypeEmbedding(nn.Module):
    """Learned embeddings for cell types"""
    def __init__(self, num_cell_types, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_cell_types, embedding_dim)
        
    def forward(self, cell_type_indices):
        return self.embedding(cell_type_indices)

class UncertaintyHead(nn.Module):
    """Predicts coordinate uncertainty"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Uncertainty for x and y
        )
        
    def forward(self, x):
        return F.softplus(self.net(x)) + 0.01  # Ensure positive uncertainty

class PhysicsInformedLayer(nn.Module):
    """Incorporates cell non-overlap constraints"""
    def __init__(self, feature_dim):
        super().__init__()
        self.radius_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.repulsion_strength = nn.Parameter(torch.tensor(0.1))
        
    def compute_repulsion_gradient(self, coords, radii, cell_types=None):
        """Compute repulsion forces between cells"""
        batch_size = coords.shape[0]
        
        # Compute pairwise distances
        distances = torch.cdist(coords, coords, p=2)
        
        # Compute sum of radii for each pair
        radii_sum = radii + radii.T
        
        # Compute overlap (positive when cells overlap)
        overlap = F.relu(radii_sum - distances + 1e-6)
        
        # Mask out self-interactions
        mask = (1 - torch.eye(batch_size, device=coords.device))
        overlap = overlap * mask
        
        # Compute repulsion forces
        coord_diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (B, B, 2)
        distances_safe = distances + 1e-6  # Avoid division by zero
        
        # Normalize direction vectors
        directions = coord_diff / distances_safe.unsqueeze(-1)
        
        # Apply stronger repulsion for same cell types (optional)
        if cell_types is not None:
            same_type_mask = (cell_types.unsqueeze(1) == cell_types.unsqueeze(0)).float()
            repulsion_weight = 1.0 + 0.5 * same_type_mask  # 50% stronger for same type
        else:
            # repulsion_weight = 1.0
            batch_size = coords.shape[0]
            repulsion_weight = torch.ones(batch_size, batch_size, device=coords.device)
            
        # Compute repulsion magnitude
        repulsion_magnitude = overlap.unsqueeze(-1) * repulsion_weight.unsqueeze(-1)
        
        # Sum repulsion forces from all other cells
        repulsion_forces = (repulsion_magnitude * directions * mask.unsqueeze(-1)).sum(dim=1)
        
        return repulsion_forces
        
    def forward(self, coords, features, cell_types=None):
        # Predict cell radii based on features
        radii = self.radius_predictor(features).squeeze(-1) * 0.01  # Scale to reasonable size
        
        # Compute repulsion gradient
        repulsion_grad = self.compute_repulsion_gradient(coords, radii, cell_types)
        
        return repulsion_grad * self.repulsion_strength, radii
    
def construct_graph_torch(X, k, mode='connectivity', metric = 'minkowski', p=2, device='cuda'):
    '''construct knn graph with torch and gpu
    args:
        X: input data containing features (torch tensor)
        k: number of neighbors for each data point
        mode: 'connectivity' or 'distance'
        metric: distance metric (now euclidean supported for gpu knn)
        p: param for minkowski (not used if metric is euclidean)
    
    Returns:
        knn graph as a pytorch sparse tensor (coo format) or dense tensor depending on mode     
    '''

    assert mode in ['connectivity', 'distance'], "mode must be 'connectivity' or 'distance'."
    assert metric == 'euclidean', "for gpu knn, only 'euclidean' metric is currently supported in this implementation"

    if mode == 'connectivity':
        include_self = True
        mode_knn = 'connectivity'
    else:
        include_self = False
        mode_knn = 'distance'

    n_samples = X.shape[0]
    knn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto')

    if device == 'cuda' and torch.cuda.is_available():
        X_cpu = X.cpu().numpy()
    else:
        X_cpu = X.numpy()

    knn.fit(X_cpu)
    knn_graph_cpu = kneighbors_graph(knn, k, mode=mode_knn, include_self=include_self, metric=metric) #scipy sparse matrix on cpu
    knn_graph_coo = knn_graph_cpu.tocoo()

    if mode == 'connectivity':
        knn_graph = torch.sparse_coo_tensor(torch.LongTensor([knn_graph_coo.row, knn_graph_coo.col]),
                                            torch.FloatTensor(knn_graph_coo.data),
                                            size = knn_graph_coo.shape).to(device)
    elif mode == 'distance':
        knn_graph_dense = torch.tensor(knn_graph_cpu.toarray(), dtype=torch.float32, device=device) #move to gpu as dense tensor
        knn_graph = knn_graph_dense
    
    return knn_graph
    
def distances_cal_torch(graph, type_aware=None, aware_power =2, device='cuda'):
    '''
    calculate distance matrix from graph using dijkstra's algo
    args:
        graph: knn graph (pytorch sparse or dense tensor)
        type_aware: not implemented in this torch version for simplicity
        aware_power: same ^^
        device (str): 'cpu' or 'cuda' device to use
    Returns:
        distance matrix as a torch tensor
    '''

    if isinstance(graph, torch.Tensor) and graph.is_sparse:
        graph_cpu_csr = csr_matrix(graph.cpu().to_dense().numpy())
    elif isinstance(graph, torch.Tensor) and not graph.is_sparse:
        graph_cpu_csr = csr_matrix(graph.cpu().numpy())
    else:
        graph_cpu_csr = csr_matrix(graph) #assume scipy sparse matrix if not torch tensor

    shortestPath_cpu = dijkstra(csgraph = graph_cpu_csr, directed=False, return_predecessors=False) #dijkstra on cpu
    shortestPath = torch.tensor(shortestPath_cpu, dtype=torch.float32, device=device)

    # the_max = torch.nanmax(shortestPath[shortestPath != float('inf')])
    # shortestPath[shortestPath > the_max] = the_max

    #mask out infinite distances
    mask = shortestPath != float('inf')
    if mask.any():
        the_max = torch.max(shortestPath[mask])
        shortestPath[~mask] = the_max #replace inf with max value
    else:
        the_max = 1.0 #fallback if all are inf (should not happen in connected graphs)

    original_max_distance = the_max.item()
    C_dis = shortestPath / the_max
    # C_dis = shortestPath
    # C_dis -= torch.mean(C_dis)
    return C_dis, original_max_distance

def calculate_D_sc_torch(X_sc, k_neighbors=10, graph_mode='connectivity', device='cpu'):
    '''calculate distance matrix from graph using dijkstra's algo
    args:
        graph: knn graph (torch sparse or dense tensor)
        type_aware: not implemented
        aware_power: same ^^
        
    returns:
        distanced matrix as torch tensor'''
    
    if not isinstance(X_sc, torch.Tensor):
        raise TypeError('Input X_sc must be a pytorch tensor')
    
    if device == 'cuda' and torch.cuda.is_available():
        X_sc = X_sc.cuda(device=device)
    else:
        X_sc = X_sc.cpu()
        device= 'cpu'

    print(f'using device: {device}')
    print(f'constructing knn graph...')
    # X_normalized = normalize(X_sc.cpu().numpy(), norm='l2') #normalize on cpu for sklearn knn
    X_normalized = X_sc
    X_normalized_torch = torch.tensor(X_normalized, dtype=torch.float32, device=device)

    Xgraph = construct_graph_torch(X_normalized_torch, k=k_neighbors, mode=graph_mode, metric='euclidean', device=device)

    print('calculating distances from graph....')
    D_sc, sc_max_distance = distances_cal_torch(Xgraph, device=device)

    print('D_sc calculation complete')
    
    return D_sc, sc_max_distance


def construct_graph_spatial(location_array, k, mode='distance', metric='euclidean', p=2):
    '''construct KNN graph based on spatial coordinates
    args:
        location_array: spatial coordinates of spots (n-spots * 2)
        k: number of neighbors for each spot
        mode: 'connectivity' or 'distance'
        metric: distance metric for knn (p=2 is euclidean)
        p: param for minkowski if connectivity
        
    returns:
        scipy.sparse.csr_matrix: knn graph in csr format
    '''

    assert mode in ['connectivity', 'distance'], "mode must be 'connectivity' or 'distance'"
    if mode == 'connectivity':
        include_self = True
    else:
        include_self = False
    
    c_graph = kneighbors_graph(location_array, k, mode=mode, metric=metric, include_self=include_self, p=p)
    return c_graph

def distances_cal_spatial(graph, spot_ids=None, spot_types=None, aware_power=2):
    '''calculate spatial distance matrix from knn graph
    args:
        graph (scipy.sparse.csr_matrix): knn graph
        spot_ids (list, optional): list of spot ids corresponding to the rows/cols of the graph. required if type_aware is used
        spot_types (pd.Series, optinal): pandas series of spot types for type aware distance adjustment. required if type_aware is used
        aware_power (int): power for type-aware distance adjustment
        
    returns:
        sptial distance matrix'''
    shortestPath = dijkstra(csgraph = csr_matrix(graph), directed=False, return_predecessors=False)
    shortestPath = np.nan_to_num(shortestPath, nan=np.inf) #handle potential inf valyes after dijkstra

    if spot_types is not None and spot_ids is not None:
        shortestPath_df = pd.DataFrame(shortestPath, index=spot_ids, columns=spot_ids)
        shortestPath_df['id1'] = shortestPath_df.index
        shortestPath_melted = shortestPath_df.melt(id_vars=['id1'], var_name='id2', value_name='value')

        type_aware_df = pd.DataFrame({'spot': spot_ids, 'spot_type': spot_types}, index=spot_ids)
        meta1 = type_aware_df.copy()
        meta1.columns = ['id1', 'type1']
        meta2 = type_aware_df.copy()
        meta2.columns = ['id2', 'type2']

        shortestPath_melted = pd.merge(shortestPath_melted, meta1, on='id1', how='left')
        shortestPath_melted = pd.merge(shortestPath_melted, meta2, on='id2', how='left')

        shortestPath_melted['same_type'] = shortestPath_melted['type1'] == shortestPath_melted['type2']
        shortestPath_melted.loc[(~shortestPath_melted.smae_type), 'value'] = shortestPath_melted.loc[(~shortestPath_melted.same_type),
                                                                                                     'value'] * aware_power
        shortestPath_melted.drop(['type1', 'type2', 'same_type'], axis=1, inplace=True)
        shortestPath_pivot = shortestPath_melted.pivot(index='id1', columns='id2', values='value')

        order = spot_ids
        shortestPath = shortestPath_pivot[order].loc[order].values
    else:
        shortestPath = np.asarray(shortestPath) #ensure it's a numpy array

    #mask out infinite distances
    mask = shortestPath != float('inf')
    if mask.any():
        the_max = np.max(shortestPath[mask])
        shortestPath[~mask] = the_max #replace inf with max value
    else:
        the_max = 1.0 #fallback if all are inf (should not happen in connected graphs)

    #store original max distance for scale reference
    original_max_distance = the_max
    C_dis = shortestPath / the_max
    # C_dis = shortestPath
    # C_dis -= np.mean(C_dis)

    return C_dis, original_max_distance

def calculate_D_st_from_coords(spatial_coords, X_st=None, k_neighbors=10, graph_mode='distance', aware_st=False, 
                               spot_types=None, aware_power_st=2, spot_ids=None):
    '''calculates the spatial distance matrix D_st for spatial transcriptomics data directly from coordinates and optional spot types
    args:
        spatial_coords: spatial coordinates of spots (n_spots * 2)
        X_st: St gene expression data (not used for D_st calculation itself)
        k_neighbors: number of neighbors for knn graph
        graph_mode: 'connectivity or 'distance' for knn graph
        aware_st: whether to use type-aware distance adjustment
        spot_types: pandas series of spot types for type-aware adjustment
        aware_power_st: power for type-aware distance adjustment
        spot_ids: list or index of spot ids, required if spot_ids is provided
        
    returns:
        np.ndarray: spatial disance matrix D_st'''
    
    if isinstance(spatial_coords, pd.DataFrame):
        location_array = spatial_coords.values
        if spot_ids is None:
            spot_ids = spatial_coords.index.tolist() #use index of dataframe if available
    elif isinstance(spatial_coords, np.ndarray):
        location_array = spatial_coords
        if spot_ids is None:
            spot_ids = list(range(location_array.shape[0])) #generate default ids if not provided

    else:
        raise TypeError('spatial_coords must be a pandas dataframe or a numpy array')
    
    print(f'constructing {graph_mode} graph for ST data with k={k_neighbors}.....')
    Xgraph_st = construct_graph_spatial(location_array, k=k_neighbors, mode=graph_mode)
    
    if aware_st:
        if spot_types is None or spot_ids is None:
            raise ValueError('spot_types and spot_ids must be provided when aware_st=True')
        if not isinstance(spot_types, pd.Series):
            spot_types = pd.Series(spot_types, idnex=spot_ids) 
        print('applying type aware distance adjustment for ST data')
        print(f'aware power for ST: {aware_power_st}')
    else:
        spot_types = None 

    print(f'calculating spatial distances.....')
    D_st, st_max_distance = distances_cal_spatial(Xgraph_st, spot_ids=spot_ids, spot_types=spot_types, aware_power=aware_power_st)

    print('D_st calculation complete')
    return D_st, st_max_distance


def calculate_D_st_euclidean(spatial_coords):
    """
    Calculate Euclidean distance matrix for ST spots.
    
    Args:
        spatial_coords: (m_spots, 2) spatial coordinates
        
    Returns:
        D_st_euclid: (m_spots, m_spots) normalized Euclidean distance matrix
    """
    from scipy.spatial.distance import pdist, squareform
    
    if isinstance(spatial_coords, pd.DataFrame):
        coords_array = spatial_coords.values
    elif isinstance(spatial_coords, np.ndarray):
        coords_array = spatial_coords
    else:
        coords_array = np.array(spatial_coords)
    
    # Compute pairwise Euclidean distances
    D_euclid = squareform(pdist(coords_array, metric='euclidean'))
    
    # Normalize to [0,1]
    max_dist = D_euclid.max()
    if max_dist > 0:
        D_euclid = D_euclid / max_dist
    
    return D_euclid.astype(np.float32)