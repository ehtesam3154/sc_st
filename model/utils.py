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

class SlideAdversarialHead(nn.Module):
    """
    Adversarial head that tries to predict slide ID from latent z.
    The main model tries to fool this head (slide-invariant representations).
    """
    def __init__(self, latent_dim, num_slides, hidden_dim=128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_slides),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, z):
        return self.discriminator(z)


class CrossSlideConsistencyLoss(nn.Module):
    """
    Ensures global structure is the same across slides.
    Uses descriptors like radial histograms and angular power spectra.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, coords_list, slide_labels):
        """
        coords_list: list of coordinate tensors from different slides
        slide_labels: which slide each coordinate set comes from
        """
        if len(coords_list) < 2:
            return torch.tensor(0.0, device=coords_list[0].device)
        
        # Compute descriptors for each coordinate set
        descriptors = []
        for coords in coords_list:
            desc = self._compute_descriptors(coords)
            descriptors.append(desc)
        
        # Penalize variance across slides
        descriptors_tensor = torch.stack(descriptors, dim=0)
        variance = torch.var(descriptors_tensor, dim=0).mean()
        
        return variance
        
    # def _compute_descriptors(self, coords):
    #     """
    #     Compute slide-level descriptors (radial histogram, etc.)
    #     """
    #     with torch.no_grad():
    #         center = coords.mean(dim=0)
    #         centered_coords = coords - center
            
    #         # Radial distances
    #         radii = torch.norm(centered_coords, dim=1)
            
    #         # Simple descriptors
    #         descriptors = torch.tensor([
    #             radii.mean(),           # average radius
    #             radii.std(),            # radius spread
    #             torch.max(radii),       # max radius
    #             (radii < radii.median()).float().mean(),  # proportion in inner half
    #         ], device=coords.device)
            
    #     return descriptors 

    def _compute_descriptors(self, coords):
        """
        Compute slide-level descriptors that should be invariant across slides.
        """
        with torch.no_grad():
            center = coords.mean(dim=0)
            centered_coords = coords - center
            
            # 1. Radial histogram (20 bins)
            radii = torch.norm(centered_coords, dim=1)
            max_r = radii.max() + 1e-8
            radii_norm = radii / max_r
            
            # Create histogram
            n_bins = 20
            hist = torch.zeros(n_bins, device=coords.device)
            bin_edges = torch.linspace(0, 1, n_bins + 1, device=coords.device)
            for i in range(n_bins):
                mask = (radii_norm >= bin_edges[i]) & (radii_norm < bin_edges[i+1])
                hist[i] = mask.float().mean()
            
            # 2. Angular power spectrum (first 5 frequencies)
            angles = torch.atan2(centered_coords[:, 1], centered_coords[:, 0])
            n_angular_bins = 72  # 5-degree bins
            angle_hist = torch.zeros(n_angular_bins, device=coords.device)
            angle_edges = torch.linspace(-torch.pi, torch.pi, n_angular_bins + 1, device=coords.device)
            for i in range(n_angular_bins):
                mask = (angles >= angle_edges[i]) & (angles < angle_edges[i+1])
                angle_hist[i] = mask.float().sum()
            
            # FFT of angular histogram (take first 5 frequencies magnitude)
            angle_hist = angle_hist - angle_hist.mean()  # Center
            fft = torch.fft.fft(angle_hist)
            power_spectrum = torch.abs(fft[:5])
            
            # 3. Collision rate (fraction of cells too close)
            if coords.shape[0] > 1:
                distances = torch.cdist(coords, coords)
                distances.fill_diagonal_(float('inf'))
                min_distances = distances.min(dim=1).values
                collision_rate = (min_distances < 0.01).float().mean().unsqueeze(0)
            else:
                collision_rate = torch.zeros(1, device=coords.device)
            
            # Combine all descriptors
            descriptors = torch.cat([
                hist,                    # 20 values
                power_spectrum,          # 5 values
                collision_rate,          # 1 value
            ])
            
        return descriptors 

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

        #FiLM head
        self.film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
    
    def forward(self, z_noisy, t, condition):
        """
        z_noisy: noisy latent vectors (batch_size, latent_dim) 
        t: timestep (batch_size, 1) - normalized
        condition: dict with conditioning information OR legacy tensor
        """
        if isinstance(condition, dict):
            # New structured masking mode
            batch_size = z_noisy.size(0)
            
            # Ensure inputs are 2D
            if z_noisy.dim() > 2:
                z_noisy = z_noisy.squeeze()
            if t.dim() == 1:
                t = t.unsqueeze(1)
            
            # Extract conditioning components
            h_shared = condition['h_shared']
            z_clean = condition['z_clean'] 
            eps_prev = condition.get('eps_prev', torch.zeros_like(z_noisy))
            mask_vec = condition.get('mask_vec')
            
            # Get dimensions
            hidden_dim = self.latent_encoder[0].out_features  # 256
            h_shared_dim = h_shared.shape[-1]  # embedding dim
            z_clean_dim = z_clean.shape[-1]    # K (clean dimensions)
            eps_prev_dim = eps_prev.shape[-1]  # latent_dim
            
            # Create conditioning MLPs with explicit dimensions
            if not hasattr(self, 'cond_h_mlp'):
                self.cond_h_mlp = nn.Sequential(
                    nn.Linear(h_shared_dim, hidden_dim),
                    nn.ReLU()
                ).to(z_noisy.device)
                
            if not hasattr(self, 'cond_zc_mlp'):
                self.cond_zc_mlp = nn.Sequential(
                    nn.Linear(z_clean_dim, hidden_dim), 
                    nn.ReLU()
                ).to(z_noisy.device)
                
            if not hasattr(self, 'cond_ep_mlp'):
                self.cond_ep_mlp = nn.Sequential(
                    nn.Linear(eps_prev_dim, hidden_dim),
                    nn.ReLU()
                ).to(z_noisy.device)
            
            # Apply conditioning MLPs
            c_h = self.cond_h_mlp(h_shared)      
            c_zc = self.cond_zc_mlp(z_clean)       
            c_ep = self.cond_ep_mlp(eps_prev)    
            c_m = 0
            
            if mask_vec is not None:
                mask_vec_dim = mask_vec.shape[-1]
                if not hasattr(self, 'cond_m_mlp'):
                    self.cond_m_mlp = nn.Sequential(
                        nn.Linear(mask_vec_dim, hidden_dim),
                        nn.ReLU()
                    ).to(z_noisy.device)
                c_m = self.cond_m_mlp(mask_vec)
            
            # Encode latent input
            z_enc = self.latent_encoder(z_noisy)  
            t_enc = self.time_embed(t)            
            
            # Combine features 
            h = z_enc + t_enc + c_h + c_zc + c_ep + c_m
            
        else:
            # Legacy mode - keep backward compatibility
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
            # h = z_enc + t_enc + c_enc

            h = z_enc + t_enc 
            gamma_beta = self.film(c_enc)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            for block in self.denoising_blocks:
                h = h + block(h)
                h = gamma * h + beta
            
            # # Apply denoising blocks
            # for block in self.denoising_blocks:
            #     h = h + block(h)  # Residual connections
                
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

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

class STDensityPrior:
    """
    Smooth density over pooled ST coordinates using Gaussian KDE.
    E_ST(y) = -log p_ST(y) up to a constant. Autograd will flow through y.
    """
    def __init__(self, st_coords_np, device="cuda", bandwidth=None):
        # st_coords_np: (N,2) numpy
        self.device = device
        self.X = torch.as_tensor(st_coords_np, dtype=torch.float32, device=device)  # (N,2)
        # bandwidth from median 2-NN distance if not provided
        if bandwidth is None:
            nbrs = NearestNeighbors(n_neighbors=2).fit(st_coords_np)
            d2, _ = nbrs.kneighbors(st_coords_np)
            spot_pitch = np.median(d2[:, 1])
            bandwidth = 1.5 * spot_pitch
        self.h2 = float(bandwidth)**2

    def energy(self, y):
        """
        y: (B,2) tensor requiring grad on same device
        returns scalar E_ST(y) = - mean log density
        """
        # pairwise distances to all ST spots (N~4k; B<=512 → 2M distances, OK on GPU)
        diff = y.unsqueeze(1) - self.X.unsqueeze(0)       # (B,N,2)
        d2   = (diff*diff).sum(-1)                        # (B,N)
        w    = torch.exp(-0.5 * d2 / self.h2)             # (B,N)
        rho  = w.sum(dim=1) + 1e-12                       # (B,)
        E    = (-torch.log(rho)).mean()                   # scalar
        return E


# ---------- (A) Minimal convex-hull + half-space penalty (pure Torch) ----------
import torch
import numpy as np

def _convex_hull_monotone_chain(xy_np):
    """Return hull vertices (CCW) using Andrew's monotone chain. xy_np: (N,2) float32."""
    P = np.asarray(xy_np, dtype=np.float32)
    P = P[np.lexsort((P[:,1], P[:,0]))]  # sort by x, then y
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in P:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in P[::-1]:
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)  # CCW, no repeat
    return hull

def _hull_normals_ccw(hull_np):
    """
    For a CCW convex polygon with vertices v_i (M,2), return outward unit normals N (M,2)
    and offsets b (M,) such that inside satisfies (N_i · y) - b_i <= 0 for all i.
    """
    V = hull_np
    M = V.shape[0]
    N = np.zeros_like(V)
    b = np.zeros((M,), dtype=np.float32)
    for i in range(M):
        v0 = V[i]
        v1 = V[(i+1)%M]
        e  = v1 - v0                      # edge direction
        n  = np.array([ e[1], -e[0] ])    # outward normal for CCW (rotate -90°)
        n  = n / (np.linalg.norm(n) + 1e-12)
        N[i] = n
        b[i] = (n * v0).sum()
    return N, b

def make_hull_penalty(st_coords_np, device):
    """Precompute convex hull planes; return a Torch function E_hull(y)->scalar."""
    hull = _convex_hull_monotone_chain(st_coords_np)       # (M,2)
    N, b = _hull_normals_ccw(hull)                         # (M,2), (M,)
    N_t = torch.as_tensor(N, dtype=torch.float32, device=device)  # (M,2)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=device)  # (M,)
    def E_hull(y):
        """
        y: (B,2) tensor with grad. For convex hull, signed violation per edge:
        s_i(y) = (n_i · y) - b_i ; inside → all s_i<=0. Penalty = (max_i ReLU(s_i))^2 mean.
        """
        S = y @ N_t.T - b_t[None, :]          # (B,M)
        viol = torch.relu(S).max(dim=1).values  # (B,)
        return (viol ** 2).mean()
    return E_hull


class ConditionAdapter(nn.Module):
    '''maps whitened embeddings to vae latent space (mu like hint)'''
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, c):
        return self.mlp(c)
    

# utils.py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def precompute_knn_edges_from_features(
    X,
    k: int = 30,
    device: str = "cuda",
    mode: str = "connectivity",   # "connectivity" or "distance"
    metric: str = "cosine",       # "cosine" or "euclidean"
    weight: str = "binary",       # "binary" | "inv" | "gaussian"
    symmetrize: bool = True,
    include_self: bool = False,
    sigma: float = None,          # for gaussian weighting; if None, auto from kth dist median
    eps: float = 1e-8,
):
    """
    Build kNN edges from *expression* features (torch or numpy) and return PyG-style (edge_index, edge_weight).
    - mode="connectivity": unweighted kNN (edge_weight=1) unless 'weight' overrides
    - mode="distance": we still return an edge list + weights; distances only used to compute weights

    Recommended defaults for expression: metric="cosine", mode="connectivity", weight="binary"
    """
    # to numpy
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)
    N = X_np.shape[0]

    # fit kNN (k+1 if include_self to drop self later)
    n_neighbors = k + 1 if include_self else k
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    nbrs.fit(X_np)
    dists, idxs = nbrs.kneighbors(X_np)  # shapes (N, n_neighbors)

    # drop self if requested
    if include_self:
        idxs = idxs[:, 1:]
        dists = dists[:, 1:]

    # build edge list
    src = np.repeat(np.arange(N), k)
    dst = idxs.reshape(-1)
    dist_flat = dists.reshape(-1)

    # weights
    if weight == "binary":
        w = np.ones_like(dist_flat, dtype=np.float32)
    elif weight == "inv":
        w = (1.0 / (dist_flat + eps)).astype(np.float32)
    elif weight == "gaussian":
        if sigma is None:
            # robust sigma from median k-th neighbor distance per node
            kth = dists[:, -1]
            sigma = float(np.median(kth) + 1e-8)
        w = np.exp(-(dist_flat**2) / (2.0 * sigma**2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown weight='{weight}'")

    # symmetrize (mutual kNN) if requested
    if symmetrize:
        # add reverse edges and max weights
        src_sym = np.concatenate([src, dst], axis=0)
        dst_sym = np.concatenate([dst, src], axis=0)
        w_sym = np.concatenate([w, w], axis=0)
        # coalesce duplicates by taking max weight
        # build a dict key=(i,j)
        key = src_sym.astype(np.int64) * N + dst_sym.astype(np.int64)
        order = np.argsort(key)
        key_sorted = key[order]; src_sorted = src_sym[order]; dst_sorted = dst_sym[order]; w_sorted = w_sym[order]
        uniq_idx = np.concatenate([[0], np.where(np.diff(key_sorted) != 0)[0] + 1, [len(key_sorted)]])
        src_co, dst_co, w_co = [], [], []
        for a, b in zip(uniq_idx[:-1], uniq_idx[1:]):
            src_co.append(src_sorted[a])
            dst_co.append(dst_sorted[a])
            w_co.append(float(w_sorted[a:b].max()))
        src = np.array(src_co, dtype=np.int64)
        dst = np.array(dst_co, dtype=np.int64)
        w = np.array(w_co, dtype=np.float32)

    # to torch (PyG style)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(w, dtype=torch.float32, device=device)
    return edge_index, edge_weight
