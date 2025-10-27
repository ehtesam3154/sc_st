import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
import os
import time
import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from typing import Optional, Dict, Tuple, List
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from model.utils import *
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import math


class AdvancedHierarchicalDiffusion(nn.Module):
    def __init__(
        self,
        st_gene_expr,
        st_coords,
        sc_gene_expr,
        cell_types_sc=None,  # Cell type labels for SC data
        transport_plan=None,  # Optimal transport plan from domain alignment
        D_st=None,
        D_induced=None,
        n_genes=None,
        # n_embedding=128,
        n_embedding=[512, 256, 128],
        coord_space_diameter=200,
        st_max_distance=None,
        sc_max_distance=None,
        sigma=3.0,
        alpha=0.9,
        mmdbatch=0.1,
        batch_size=64,
        device='cuda',
        lr_e=0.0001,
        lr_d=0.0002,
        n_timesteps=1000,
        n_denoising_blocks=6,
        hidden_dim=512,
        num_heads=8,
        num_hierarchical_scales=3,
        dp=0.1,
        outf='output'
    ):
        super().__init__()

        self.diffusion_losses = {
            'total': [],
            'diffusion': [],
            'struct': [],
            'physics': [],
            'uncertainty': [],
            'epochs': []
        }

        # Loss tracking for Graph-VAE training
        self.vae_losses = {
            'total': [],
            'reconstruction': [],
            'kl': [],
            'epochs': []
        }
        
        # Loss tracking for Latent Diffusion training  
        self.latent_diffusion_losses = {
            'total': [],
            'diffusion': [],
            'struct': [],
            'epochs': []
        }
        
        # Keep encoder losses separate (if you want to track them)
        self.encoder_losses = {
            'total': [],
            'pred': [],
            'circle': [],
            'mmd': [],
            'epochs': []
        }
        
        self.device = device
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.sigma = sigma
        self.alpha = alpha
        self.mmdbatch = mmdbatch
        self.n_embedding = n_embedding

        #slide style affine head
        self.affine_A = nn.Parameter(torch.eye(2, device=self.device))
        self.affine_b = nn.Parameter(torch.zeros(1, 2, device=self.device))
        self.affine_reg = 1e-4

        # Create output directory
        self.outf = outf
        if not os.path.exists(outf):
            os.makedirs(outf)
        
        # Store data
        self.st_gene_expr = torch.tensor(st_gene_expr, dtype=torch.float32).to(device)
        self.st_coords = torch.tensor(st_coords, dtype=torch.float32).to(device)
        self.sc_gene_expr = torch.tensor(sc_gene_expr, dtype=torch.float32).to(device)

        
        # Temperature regularization for geometric attention
        self.temp_weight_decay = 1e-4
        
        # Store transport plan if provided
        self.transport_plan = torch.tensor(transport_plan, dtype=torch.float32).to(device) if transport_plan is not None else None
        
        # Process cell types
        if cell_types_sc is not None:
            # Convert cell type strings to indices
            unique_cell_types = np.unique(cell_types_sc)
            self.cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}
            self.num_cell_types = len(unique_cell_types)
            cell_type_indices = [self.cell_type_to_idx[ct] for ct in cell_types_sc]
            self.sc_cell_types = torch.tensor(cell_type_indices, dtype=torch.long).to(device)
        else:
            self.sc_cell_types = None
            self.num_cell_types = 0
            
        # Store distance matrices
        self.D_st = torch.tensor(D_st, dtype=torch.float32).to(device) if D_st is not None else None
        self.D_induced = torch.tensor(D_induced, dtype=torch.float32).to(device) if D_induced is not None else None

        # If D_st is not provided, calculate it from spatial coordinates
        if self.D_st is None:
            print("D_st not provided, calculating from spatial coordinates...")
            if isinstance(st_coords, torch.Tensor):
                st_coords_np = st_coords.cpu().numpy()
            else:
                st_coords_np = st_coords
            
            D_st_np, st_max_distance = calculate_D_st_from_coords(
                spatial_coords=st_coords_np, 
                k_neighbors=50, 
                graph_mode="distance"
            )
            self.D_st = torch.tensor(D_st_np, dtype=torch.float32).to(device)
            self.st_max_distance = st_max_distance
            print(f"D_st calculated, shape: {self.D_st.shape}")


        print(f"Final matrices - D_st: {self.D_st.shape if self.D_st is not None else None}, "
            f"D_induced: {self.D_induced.shape if self.D_induced is not None else None}")
        
        # Normalize coordinates
        self.st_coords_norm, self.coords_center, self.coords_radius = self.normalize_coordinates_isotropic(self.st_coords)        
        print(f"\n=== NORMALIZED ST Coordinates ===")
        print(f"  X range: [{self.st_coords_norm[:, 0].min():.3f}, {self.st_coords_norm[:, 0].max():.3f}]")
        print(f"  Y range: [{self.st_coords_norm[:, 1].min():.3f}, {self.st_coords_norm[:, 1].max():.3f}]")
        print(f"  Center: {self.coords_center}")
        print(f"  Max radius: {self.coords_radius:.2f}")

        # self.st_coords_norm = self.st_coords

        # Model parameters
        self.n_genes = n_genes or st_gene_expr.shape[1]
        
        # ========== FEATURE ENCODER ==========
        self.netE = self.build_feature_encoder(self.n_genes, n_embedding, dp)

        self.train_log = os.path.join(outf, 'train.log')

        
        # ========== CELL TYPE EMBEDDING ==========

        use_cell_types = (cell_types_sc is not None)  # Check if SC data has cell types
        self.use_cell_types = use_cell_types

        if self.num_cell_types > 0:
            self.cell_type_embedding = CellTypeEmbedding(self.num_cell_types, n_embedding[-1] // 2)
            total_feature_dim = n_embedding[-1] + n_embedding[-1] // 2
        else:
            self.cell_type_embedding = None
            total_feature_dim = n_embedding[-1]
            
        # ========== HIERARCHICAL DIFFUSION COMPONENTS ==========
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate encoder
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Feature projection (includes cell type if available)
        self.feat_proj = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ========== GRAPH-VAE COMPONENTS (REPLACING HIERARCHICAL DIFFUSION) ==========        
        # Graph-VAE parameters
        self.latent_dim = 32  # As specified in instructions
        
        # Graph-VAE Encoder (learns latent representations from ST graphs)
        self.graph_vae_encoder = GraphVAEEncoder(
            input_dim=n_embedding[-1],  # Aligned embedding dimension
            hidden_dim=128,             # GraphConv hidden dimension  
            latent_dim=self.latent_dim
        ).to(device)

        self.graph_vae_decoder = GraphVAEDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=128  # Remove condition_dim
        ).to(device)

        #adapter: embedding to latent
        self.cond_adapter = ConditionAdapter(n_embedding[-1], self.latent_dim).to(device)
        
        # Latent Denoiser (replaces hierarchical_blocks)
        self.latent_denoiser = LatentDenoiser(
            latent_dim=self.latent_dim,
            condition_dim=n_embedding[-1] + self.latent_dim, #to accomodate cond adapater
            hidden_dim=hidden_dim,
            n_blocks=n_denoising_blocks
        ).to(device)
        
        # ========== HIERARCHICAL DENOISING BLOCKS ==========
        self.hierarchical_blocks = nn.ModuleList([
            HierarchicalDiffusionBlock(hidden_dim, num_hierarchical_scales)
            for _ in range(n_denoising_blocks)
        ])    

        # ========== PHYSICS-INFORMED COMPONENTS ==========
        self.physics_layer = PhysicsInformedLayer(hidden_dim)
        
        # ========== UNCERTAINTY QUANTIFICATION ==========
        self.uncertainty_head = UncertaintyHead(hidden_dim)
        
        # ========== OPTIMAL TRANSPORT GUIDANCE ==========
        if self.transport_plan is not None:
            self.ot_guidance_strength = nn.Parameter(torch.tensor(0.1))
            
        # ========== OUTPUT LAYERS ==========
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Create noise schedule
        # self.noise_schedule = self.create_noise_schedule()
        self.noise_schedule = self.build_noise_schedule(self.n_timesteps)

        self.guidance_scale = 2.0
        
        # Optimizers
        self.setup_optimizers(lr_e, lr_d)
        
        # MMD Loss for domain alignment
        self.mmd_loss = MMDLoss()

        #embeddong condition whitening (fit on ST and freeze as buffers)
        with torch.no_grad():
            F_st = self.netE(self.st_gene_expr).float()
            mu_c = F_st.mean(0, keepdim=True)
            C = torch.cov(F_st.T) + 1e-4 * torch.eye(F_st.shape[1], device=F_st.device)
            L = torch.linalg.cholesky(C)
            Linv = torch.linalg.inv(L)

        self.register_buffer('cond_mu', mu_c)
        self.register_buffer('cond_Linv', Linv)

        def _whiten_cond(self, c):
            return (c- self.cond_mu) @ self.cond_Linv
        
        self._whiten_cond = _whiten_cond.__get__(self, type(self))

        # Move entire model to device
        self.to(self.device)

    def build_noise_schedule(self, T, beta_start=1e-4, beta_end=2e-2):
        """Rebuild noise schedule when T changes"""
        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'posterior_variance': posterior_variance
        }

    import torch

    def _build_noise_schedule_linear(self, T:int, beta_start:float, beta_end:float):
        betas  = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return {
            'betas': betas, 'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'posterior_variance': posterior_variance
        }

    def _build_noise_schedule_cosine(self, T:int, s:float=0.008):
        steps = torch.arange(T+1, device=self.device, dtype=torch.float32) / T
        f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-5, max=1.0)  # ᾱ(0)=1
        betas = (1 - (alpha_bar[1:] / alpha_bar[:-1])).clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return {
            'betas': betas, 'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'posterior_variance': posterior_variance
        }


    # def update_noise_schedule(self):
    #     """Update noise schedule when n_timesteps changes"""
    #     self.noise_schedule = self.build_noise_schedule(self.n_timesteps)
    #     print(f"Updated noise schedule for T={self.n_timesteps}")
    #     print(f"alpha_bar[0]={self.noise_schedule['alphas_cumprod'][0]:.6f}, alpha_bar[-1]={self.noise_schedule['alphas_cumprod'][-1]:.6f}")

    def update_noise_schedule(self):
        T = int(self.n_timesteps)  # single source of truth
        if self.noise_schedule_mode == 'cosine':
            self.noise_schedule = self._build_noise_schedule_cosine(T)
        else:
            # pick beta_end so ᾱ_T hits your target with *this* T
            total = -math.log(self.noise_target_alpha_end)
            beta_end = max(2*total/T - self.beta_start, self.beta_start + 1e-6)
            self.noise_schedule = self._build_noise_schedule_linear(T, self.beta_start, beta_end)
        print(f"[noise] mode={self.noise_schedule_mode}, T={T}, "
            f"alpha_bar[-1]={self.noise_schedule['alphas_cumprod'][-1].item():.5f}")


    def setup_spatial_sampling(self):
        if hasattr(self, 'st_coords_norm'):
            self.spatial_sampler = SpatialBatchSampler(
                coordinates=self.st_coords_norm.cpu().numpy(),
                batch_size=self.batch_size
            )
        else:
            self.spatial_sampler = None

    def get_spatial_batch(self):
        """Get spatially contiguous batch for training"""
        if self.spatial_sampler is not None:
            return self.spatial_sampler.sample_spatial_batch()
        else:
            # Fallback to random sampling
            return torch.randperm(len(self.st_coords_norm))[:self.batch_size]
        
    def _evaluate_sigma_quality(self, st_embeddings, k=10):
        """Evaluate how well encoder embeddings preserve spatial k-NN structure"""
        with torch.no_grad():
            # Get k-NN from encoder similarity
            netpred = st_embeddings.mm(st_embeddings.t())
            pred_knn = self._get_knn_from_similarity(netpred, k=k)
            
            # Get k-NN from physical coordinates  
            phys_knn = self._get_knn_from_coords(self.st_coords_norm, k=k)
            
            # Compute overlap
            overlap = (pred_knn == phys_knn).float().mean().item()
            return overlap

    def _get_knn_from_similarity(self, similarity_matrix, k=10):
        """Extract top-k neighbors from similarity matrix"""
        # Get top-k indices for each node
        _, topk_indices = torch.topk(similarity_matrix, k=k+1, dim=1)  # +1 to exclude self
        topk_indices = topk_indices[:, 1:]  # Remove self-connections
        return topk_indices

    def _get_knn_from_coords(self, coords, k=10):
        """Extract top-k spatial neighbors from coordinates"""
        # Compute pairwise distances
        distances = torch.cdist(coords, coords)
        # Get top-k closest (smallest distances)
        _, topk_indices = torch.topk(distances, k=k+1, dim=1, largest=False)  # +1 for self
        topk_indices = topk_indices[:, 1:]  # Remove self-connections  
        return topk_indices
        
    def normalize_coordinates_isotropic(self, coords):
        """Normalize coordinates isotropically to [-1, 1]"""
        center = coords.mean(dim=0)
        centered_coords = coords - center
        max_dist = torch.max(torch.norm(centered_coords, dim=1))
        normalized_coords = centered_coords / (max_dist + 1e-8)
        return normalized_coords, center, max_dist
        

    def build_feature_encoder(self, n_genes, n_embedding, dp):
        """Build the feature encoder network"""
        return FeatureNet(n_genes, n_embedding=n_embedding, dp=dp).to(self.device)
        
    def create_noise_schedule(self):
        """Create the noise schedule for diffusion"""
        betas = torch.linspace(0.0001, 0.02, self.n_timesteps, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod)
        }
        
    def setup_optimizers(self, lr_e, lr_d):
        """Setup optimizers and schedulers"""
        # Encoder optimizer
        self.optimizer_E = torch.optim.AdamW(self.netE.parameters(), lr=0.002)               
        self.scheduler_E = lr_scheduler.StepLR(self.optimizer_E, step_size=200, gamma=0.5) 

        # MMD Loss
        self.mmd_fn = MMDLoss()   
        
        # Diffusion model optimizer
        diff_params = []
        diff_params.extend(self.time_embed.parameters())
        diff_params.extend(self.coord_encoder.parameters())
        diff_params.extend(self.feat_proj.parameters())
        diff_params.extend(self.hierarchical_blocks.parameters())
        # diff_params.extend(self.geometric_attention_blocks.parameters())
        diff_params.extend(self.physics_layer.parameters())
        diff_params.extend(self.uncertainty_head.parameters())
        diff_params.extend(self.noise_predictor.parameters())
        
        if self.cell_type_embedding is not None:
            diff_params.extend(self.cell_type_embedding.parameters())
            
        if self.transport_plan is not None:
            diff_params.append(self.ot_guidance_strength)
            
        self.optimizer_diff = torch.optim.Adam(diff_params, lr=lr_d, betas=(0.9, 0.999))
        self.scheduler_diff = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_diff, T_0=500)
        
    def add_noise(self, coords, t, noise_schedule):
        """Add noise to coordinates according to the diffusion schedule"""
        noise = torch.randn_like(coords)
        sqrt_alphas_cumprod_t = noise_schedule['sqrt_alphas_cumprod'][t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = noise_schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1)
        
        noisy_coords = sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_coords, noise
        
        
    def forward_diffusion(self, noisy_coords, t, features, cell_types=None):
        """Forward pass through the advanced diffusion model"""
        batch_size = noisy_coords.shape[0]
        
        # Encode inputs
        time_emb = self.time_embed(t)
        coord_emb = self.coord_encoder(noisy_coords)
        
        # Process features with optional cell type
        if cell_types is not None and self.cell_type_embedding is not None:
            cell_type_emb = self.cell_type_embedding(cell_types)
            combined_features = torch.cat([features, cell_type_emb], dim=-1)
        else:
            #when no cell types, pad with zeros to match expected input size
            if self.cell_type_embedding is not None:
                #create zero padding for cell type embedding
                cell_type_dim = self.n_embedding[-1] // 2
                zero_padding = torch.zeros(batch_size, cell_type_dim, device=features.device)
                combined_features = torch.cat([features, zero_padding], dim=-1)
            else:
                combined_features = features
            # combined_features = features
            
        feat_emb = self.feat_proj(combined_features)
        
        # Combine embeddings
        h = coord_emb + time_emb + feat_emb
        
        # Process through hierarchical blocks with geometric attention
        for i, block in enumerate(self.hierarchical_blocks):
            h = block(h, t)
                
        # Predict noise
        noise_pred = self.noise_predictor(h)
        
        # Compute physics-informed correction
        physics_correction, cell_radii = self.physics_layer(noisy_coords, h, cell_types)
        
        # Compute uncertainty
        uncertainty = self.uncertainty_head(h)
        
        # Apply corrections based on timestep (less physics at high noise)
        t_factor = (1 - t).unsqueeze(-1) #shape: (natch_size, 1)
        noise_pred = noise_pred + t_factor * physics_correction * 0.1
        
        return noise_pred, uncertainty, cell_radii
        
    def train_encoder(self, n_epochs=1000, ratio_start=0, ratio_end=1.0):
        """Train the STEM encoder to align ST and SC data"""
        print("Training STEM encoder...")
        
        # Log training start
        with open(self.train_log, 'a') as f:
            localtime = time.asctime(time.localtime(time.time()))
            f.write(f"{localtime} - Starting STEM encoder training\n")
            f.write(f"n_epochs={n_epochs}, ratio_start={ratio_start}, ratio_end={ratio_end}\n")
        
        # Calculate spatial adjacency matrix
        # if self.sigma == 0:
        #     nettrue = torch.eye(self.st_coords.shape[0], device=self.device)
        # else:
        #     nettrue = torch.tensor(scipy.spatial.distance.cdist(
        #         self.st_coords.cpu().numpy(), 
        #         self.st_coords.cpu().numpy()
        #     ), device=self.device).to(torch.float32)
            
        #     sigma = self.sigma
        #     nettrue = torch.exp(-nettrue**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
        #     nettrue = F.normalize(nettrue, p=1, dim=1)

        # Auto-calculate sigma from normalized coordinates
        with torch.no_grad():
            # Use normalized coords for everything
            coords_for_rbf = self.st_coords_norm
            D = torch.cdist(coords_for_rbf, coords_for_rbf)
            
            # Get k-NN distances (k=15 is robust)
            k = 15
            kth_distances = torch.kthvalue(D, k+1, dim=1).values  # k+1 because diagonal is 0
            
            # Use median as default sigma
            auto_sigma = torch.median(kth_distances).item()
            
            # Apply a scaling factor for Gaussian width (0.8 = slightly tighter, 1.0 = exact median)
            self.sigma = auto_sigma * 0.8
            
            print(f"Auto-calculated sigma = {self.sigma:.4f} (based on normalized coordinates)")
            print(f"k-NN distance stats: min={kth_distances.min():.4f}, "
                f"median={kth_distances.median():.4f}, max={kth_distances.max():.4f}")
            
            # Calculate spatial adjacency matrix with normalized coords and auto sigma
            if self.sigma == 0:
                nettrue = torch.eye(self.st_coords_norm.shape[0], device=self.device)
            else:
                distances = torch.cdist(coords_for_rbf, coords_for_rbf)
                nettrue = torch.exp(-distances**2/(2*self.sigma**2))/(np.sqrt(2*np.pi)*self.sigma)
                nettrue = F.normalize(nettrue, p=1, dim=1)

        # Log the auto-calculated sigma
        with open(self.train_log, 'a') as f:
            f.write(f"Auto-calculated sigma={self.sigma:.4f} from normalized coordinates\n")
        
        # Training loop
        for epoch in range(n_epochs):
            # Schedule for circle loss weight
            ratio = ratio_start + (ratio_end - ratio_start) * min(epoch / (n_epochs * 0.8), 1.0)
            
            # Forward pass ST data
            e_seq_st = self.netE(self.st_gene_expr, True)
            
            # Sample from SC data due to large size
            sc_idx = torch.randint(0, self.sc_gene_expr.shape[0], (min(self.batch_size, self.mmdbatch),), device=self.device)
            sc_batch = self.sc_gene_expr[sc_idx]
            e_seq_sc = self.netE(sc_batch, False)
            
            # Calculate losses
            self.optimizer_E.zero_grad()
            
            # Prediction loss (equivalent to netpred in STEM)
            netpred = e_seq_st.mm(e_seq_st.t())
            loss_E_pred = F.cross_entropy(netpred, nettrue, reduction='mean')
            
            # Mapping matrices
            st2sc = F.softmax(e_seq_st.mm(e_seq_sc.t()), dim=1)
            sc2st = F.softmax(e_seq_sc.mm(e_seq_st.t()), dim=1)
            
            # Circle loss
            st2st = torch.log(st2sc.mm(sc2st) + 1e-7)
            loss_E_circle = F.kl_div(st2st, nettrue, reduction='none').sum(1).mean()
            
            # MMD loss
            # ranidx = torch.randint(0, e_seq_sc.shape[0], (min(self.mmdbatch, e_seq_sc.shape[0]),), device=self.device)
            # loss_E_mmd = self.mmd_fn(e_seq_st, e_seq_sc[ranidx])

            st_ranidx = torch.randint(0, e_seq_st.shape[0], (min(self.mmdbatch, e_seq_st.shape[0]),), device=self.device)
            sc_ranidx = torch.randint(0, e_seq_sc.shape[0], (min(self.mmdbatch, e_seq_sc.shape[0]),), device=self.device)
            loss_E_mmd = self.mmd_fn(e_seq_st[st_ranidx], e_seq_sc[sc_ranidx])

            # st_idx = torch.randint(0, self.st_gene_expr.shape[0], (self.mmdbatch,), device=self.device)
            # sc_idx = torch.randint(0, self.sc_gene_expr.shape[0], (self.mmdbatch,), device=self.device)

            # e_seq_st = self.netE(self.st_gene_expr[st_idx], True)
            # e_seq_sc = self.netE(self.sc_gene_expr[sc_idx], False)

            # loss_E_mmd = self.mmd_fn(e_seq_st, e_seq_sc)
            
            # Total loss
            loss_E = loss_E_pred + self.alpha * loss_E_mmd + ratio * loss_E_circle
            
            # Backward and optimize
            loss_E.backward()
            self.optimizer_E.step()
            self.scheduler_E.step()
            
            # Log progress
            if epoch % 200 == 0:
                log_msg = (f"Encoder epoch {epoch}/{n_epochs}, "
                          f"Loss_E: {loss_E.item():.6f}, "
                          f"Loss_E_pred: {loss_E_pred.item():.6f}, "
                          f"Loss_E_circle: {loss_E_circle.item():.6f}, "
                          f"Loss_E_mmd: {loss_E_mmd.item():.6f}, "
                          f"Ratio: {ratio:.4f}")
                
                print(log_msg)
                with open(self.train_log, 'a') as f:
                    f.write(log_msg + '\n')
                
                # Save checkpoint
                if epoch % 500 == 0:
                    torch.save({
                        'epoch': epoch,
                        'netE_state_dict': self.netE.state_dict(),
                        'optimizer_state_dict': self.optimizer_E.state_dict(),
                        'scheduler_state_dict': self.scheduler_E.state_dict(),
                    }, os.path.join(self.outf, f'encoder_checkpoint_epoch_{epoch}.pt'))
    

        print("\n" + "="*50)
        print("EVALUATING SIGMA QUALITY")
        print("="*50)
        
        # Evaluate current sigma
        with torch.no_grad():
            self.netE.eval()
            st_embeddings = self.netE(self.st_gene_expr, True)  # Get final ST embeddings
            current_overlap = self._evaluate_sigma_quality(st_embeddings, k=10)
            print(f"Current sigma ({self.sigma:.4f}) -> kNN overlap = {current_overlap:.3f}")
        
        # Test different sigma values to find optimal
        print("\nTesting different sigma values...")
        sigma_candidates = [
            self.sigma * 0.5,   # Half current
            self.sigma * 0.75,  # 3/4 current  
            self.sigma,         # Current (baseline)
            self.sigma * 1.25,  # 5/4 current
            self.sigma * 1.5,   # 1.5x current
            self.sigma * 2.0,    # Double current
            self.sigma * 2.5,   # Double current
            self.sigma * 3.0,    # Double current
            self.sigma * 4.0    # Double current

        ]
        
        overlaps = []
        for test_sigma in sigma_candidates:
            # Recompute adjacency with test sigma
            if test_sigma == 0:
                test_nettrue = torch.eye(self.st_coords.shape[0], device=self.device)
            else:
                distances = torch.tensor(scipy.spatial.distance.cdist(
                    self.st_coords.cpu().numpy(), 
                    self.st_coords.cpu().numpy()
                ), device=self.device).to(torch.float32)
                
                test_nettrue = torch.exp(-distances**2/(2*test_sigma**2))/(np.sqrt(2*np.pi)*test_sigma)
                test_nettrue = F.normalize(test_nettrue, p=1, dim=1)
            
            # Quick test: how well does current encoder match this adjacency?
            with torch.no_grad():
                netpred = st_embeddings.mm(st_embeddings.t())
                pred_knn = self._get_knn_from_similarity(netpred, k=15)
                true_knn = self._get_knn_from_similarity(test_nettrue, k=15)
                overlap = (pred_knn == true_knn).float().mean().item()
                overlaps.append(overlap)
                
            print(f"  sigma = {test_sigma:.4f} -> overlap = {overlap:.5f}")
        
        # Find best sigma
        best_idx = np.argmax(overlaps)
        best_sigma = sigma_candidates[best_idx]
        best_overlap = overlaps[best_idx]

        # print(overlaps)
        
        print(f"\nBest sigma: {best_sigma:.4f} (overlap = {best_overlap:.5f})")
        if best_sigma != self.sigma:
            print(f"⚠️  Consider using sigma = {best_sigma:.4f} instead of {self.sigma:.4f}")
            print(f"   Improvement: {best_overlap:.3f} vs {current_overlap:.3f} (+{(best_overlap-current_overlap)*100:.1f}%)")
        else:
            print("✅ Current sigma is optimal!")
        
        print("="*50)
        # ===================================
        
        # Save final encoder
        torch.save({
            'netE_state_dict': self.netE.state_dict(),
        }, os.path.join(self.outf, 'final_encoder.pt'))
        
        print("Encoder training complete!")

    def train_graph_vae(self, epochs=800, lr=1e-3, warmup_epochs=320,  # 40% of epochs
                    lambda_cov_max=0.3, angle_loss_weight=0.2, 
                    radius_loss_weight=1.0, angle_warmup_epochs=0, 
                    beta_final=1e-3):  # Small β_final to prevent blow-up
        """
        Train the Graph-VAE with:
        1) normalized KL (no free-bits, mean over batch and dims)
        2) covariance warm-up (fix axes)  
        3) geometry-anchored angle/radius loss (fix gauge deterministically)
        4) NO reconstruction loss, NO gradient loss (as requested)
        """
        print("Training Graph-VAE with normalized KL and polar losses...")
        
        # Freeze encoder
        self.netE.eval()
        for p in self.netE.parameters():
            p.requires_grad = False

        # Build ST graph
        adj_idx, adj_w = precompute_knn_edges(self.st_coords_norm, k=30, device=self.device)

        # Precompute aligned features (ST only for training)
        with torch.no_grad():
            st_features_aligned = self.netE(self.st_gene_expr).float()

        # Precompute canonical frame and angle/radius targets for ST
        print('Computing canonical angular frame for geometry anchoring....')
        c, a_theta, R, theta_true, r_true = self._compute_canonical_frame(self.st_coords_norm)

        # Precompute true covariance of ST spots
        with torch.no_grad():
            centered = self.st_coords_norm - self.st_coords_norm.mean(0, keepdim=True)
            self.cov_true = (centered.T @ centered) / (centered.shape[0] - 1)

        # Optimizer + scheduler
        # vae_params = list(self.graph_vae_encoder.parameters()) + list(self.graph_vae_decoder.parameters()+ [self.affine_A, self.affine_b])
        vae_params = (
            list(self.graph_vae_encoder.parameters())
        + list(self.graph_vae_decoder.parameters())
        + [self.affine_A, self.affine_b]
        )
        optimizer = torch.optim.Adam(vae_params, lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # --- Neighborhood target P over ST (row-stochastic), matches your encoder's sigma ---
        with torch.no_grad():
            X = self.st_coords_norm                    # (N, 2) normalized coords you already use
            D2_true = torch.cdist(X, X).pow(2)         # (N, N)
            # temperature τ for decoder space: tie to your sigma; good start τ = max(self.sigma, 1e-3)
            tau = max(float(self.sigma), 1e-3)
            P = torch.softmax(-D2_true / (2.0 * tau * tau), dim=1)  # (N, N), row-stochastic


        # Initialize loss logs
        for key in ('total','reconstruction','kl','kl_raw','cov','lambda_cov','angle','radius','beta','epochs'):
            self.vae_losses.setdefault(key, [])

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute loss weights with warmups
            lambda_cov = lambda_cov_max * min(epoch+1, warmup_epochs) / warmup_epochs
            w_theta = 0.0 if epoch < angle_warmup_epochs else angle_loss_weight
            w_radius = 0.0 if epoch < angle_warmup_epochs else radius_loss_weight
            
            # KL warm-up: β from 0 to β_final (small value)
            beta = beta_final * min(epoch+1, warmup_epochs) / warmup_epochs

            # Forward pass for ST spots only
            mu_st, logvar_st = self.graph_vae_encoder(st_features_aligned, adj_idx, adj_w)
            z_st = self.graph_vae_encoder.reparameterize(mu_st, logvar_st)
            
            # Decoder takes ONLY z (no features)
            coords_pred_st = self.graph_vae_decoder(z_st)

            #map cannonical to slide coords (for loss only)
            coords_for_loss = coords_pred_st @ self.affine_A.T + self.affine_b
            
            #edge distance reconstruction 
            i, j = adj_idx[0], adj_idx[1]
            dist_true = (self.st_coords_norm[i] - self.st_coords_norm[j]).pow(2).sum(1).sqrt()
            # dist_pred = (coords_pred_st[i] - coords_pred_st[j]).pow(2).sum(1).sqrt()
            dist_pred = (coords_for_loss[i] - coords_for_loss[j]).pow(2).sum(1).sqrt()

            # --- Predicted neighborhoods Q from decoded coords ---
            D2_pred = torch.cdist(coords_pred_st, coords_pred_st).pow(2)   # (N, N)
            logQ = F.log_softmax(-D2_pred / (2.0 * tau * tau), dim=1)      # log row-stochastic
            L_neigh = F.kl_div(logQ, P, reduction='batchmean')             # KL(P || Q)


            L_edges = F.smooth_l1_loss(dist_pred, dist_true)
            lambda_edges = 2.0
            L_recon = lambda_edges * L_edges

            # 2) Normalized KL divergence (Option A - no free-bits)
            # KL per dim (positive values)
            kl_per_dim = -0.5 * (1 + logvar_st - mu_st.pow(2) - logvar_st.exp())  # [N, latent_dim]
            
            # Mean over batch AND latent dims (this prevents blow-up)
            KL_raw = kl_per_dim.mean()  # O(1) scale, not O(latent_dim)
            L_KL = beta * KL_raw

            # 3) Covariance alignment (ST spots only, warm-up)
            # coords_pred_centered = coords_pred_st - coords_pred_st.mean(0, keepdim=True)
            # cov_pred = (coords_pred_centered.T @ coords_pred_centered) / (coords_pred_centered.shape[0] - 1)
            # L_cov = F.mse_loss(cov_pred, self.cov_true)

            coords_centered = coords_for_loss - coords_for_loss.mean(0, keepdim=True)
            cov_pred = (coords_centered.T @ coords_centered) / (coords_centered.shape[0] - 1)
            L_cov = F.mse_loss(cov_pred, self.cov_true)


            # 4) NO Gradient anchoring (as requested)
            # L_grad = 0.0

            # 5) Geometry-anchored angle/radius losses
            # Compute predicted angles/radii using the SAME canonical frame (c, a_theta, R)

            # v_hat = coords_pred_st - c
            v_hat = coords_for_loss - c 
            cross = a_theta[0] * v_hat[:, 1] - a_theta[1] * v_hat[:, 0]
            dot = a_theta[0] * v_hat[:, 0] + a_theta[1] * v_hat[:, 1]
            theta_hat = torch.atan2(cross, dot)
            r_hat = v_hat.norm(dim=1) / (R + 1e-8)

            # Circular angle loss: 1 - cos(delta)
            delta = theta_hat - theta_true
            L_angle_raw = (1.0 - torch.cos(delta)).mean()
            L_angle = w_theta * L_angle_raw

            # Radius loss
            L_radius_raw = F.mse_loss(r_hat, r_true)
            L_radius = w_radius * L_radius_raw

            # 6) Total loss
            total_loss = (
                L_recon +           
                L_KL +              # Small, normalized
                lambda_cov * L_cov +
                L_angle + 
                L_radius
            )


            # Regularize A near rotation+isotropic scale and small translation
            AtA = self.affine_A.T @ self.affine_A
            L_aff = ((AtA - torch.eye(2, device=self.device))**2).mean() + (self.affine_b**2).mean()
            total_loss = total_loss + self.affine_reg * L_aff


            lambda_edges = 0.0        # turn OFF raw (x,y) edge distances (imprints hull)
            lambda_cov   = 0.0        # turn OFF global covariance/centroid match (absolute frame)
            lambda_angle = 0.0        # turn OFF polar angle target (absolute frame)
            lambda_radius= 0.0        # turn OFF radius target (absolute frame)
            lambda_neigh = 1.0        # primary geometric term

            total_loss = (
                lambda_neigh * L_neigh
                + beta * L_KL
            )

            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Log losses
            self.vae_losses['total'].append(total_loss.item())
            self.vae_losses['reconstruction'].append(L_recon.item())
            self.vae_losses['kl'].append(L_KL.item())
            self.vae_losses['kl_raw'].append(KL_raw.item())
            self.vae_losses['cov'].append(L_cov.item())
            self.vae_losses['lambda_cov'].append(lambda_cov)
            self.vae_losses['angle'].append(L_angle.item())
            self.vae_losses['radius'].append(L_radius.item())
            self.vae_losses['beta'].append(beta)
            self.vae_losses['epochs'].append(epoch)

            if epoch % 100 == 0 or epoch == epochs-1:
                # Compute interpretable angle error in degrees
                with torch.no_grad():
                    mean_angle_err_deg = torch.mean(torch.abs(torch.rad2deg(torch.atan2(torch.sin(delta), torch.cos(delta))))).item()
                
                print(f"Epoch {epoch+1}/{epochs}  "
                    f"Loss={total_loss:.4f}  "
                    f"L_recon={L_recon:.4f}  "
                    f"L_KL={L_KL:.6f}(β={beta:.6f})  "
                    f"KL_raw={KL_raw:.4f}  "
                    f"L_cov={L_cov:.4f}  "
                    f"L_angle={L_angle:.4f}  "
                    f"L_radius={L_radius:.4f}  "
                    f"AngleErr={mean_angle_err_deg:.2f}°")

        print("Graph-VAE training complete.")
        
        # Diagnostic: Check if decoder uses z meaningfully
        print("\n=== DECODER DEPENDENCY DIAGNOSTIC ===")
        with torch.no_grad():
            # Fix features, vary z
            mu_fixed = mu_st[:5]
            logvar_fixed = logvar_st[:5]
            
            coords_samples = []
            for _ in range(5):
                z_sample = self.graph_vae_encoder.reparameterize(mu_fixed, logvar_fixed)
                coords_sample = self.graph_vae_decoder(z_sample)
                coords_samples.append(coords_sample)
            
            coords_stack = torch.stack(coords_samples)  # (5, 5, 2)
            coord_variance = coords_stack.var(dim=0).mean().item()  # Average variance across samples
            
            print(f"Coordinate variance from z sampling: {coord_variance:.6f}")
            if coord_variance < 1e-4:
                print("⚠️  WARNING: Low variance suggests potential posterior collapse!")
            else:
                print("Good: Decoder shows dependency on z")
        print("=" * 50)


    def _compute_canonical_frame(self, X_st):
        '''compute canonical angular frame from ST coordinates for geometry anchoring
        
        returns:
            c: centroid (2, )
            a_theta: reference direction vector (2, )
            R: max radius (scalar)
            theta_true: true_angles (N, )
            r_true: true normalized radii (N,)
        '''
        #centroid
        c = X_st.mean(dim=0)

        #find farthest point to define reference direction
        d = torch.linalg.norm(X_st - c, dim=1)
        A = torch.argmax(d).item()
        a_theta = (X_st[A] - c)
        R = d.max().clamp_min(1e-8)

        #compute true angles and radii for all points
        v = X_st - c
        cross = a_theta[0] * v[:, 1] - a_theta[1] * v[:, 0]
        dot = a_theta[0] * v[:, 0] + a_theta[1] * v[:, 1]
        theta_true = torch.atan2(cross, dot)
        r_true = (v.norm(dim=1) / R)

        print(f"Canonical frame: center=({c[0]:.3f}, {c[1]:.3f}), "
            f"ref_dir=({a_theta[0]:.3f}, {a_theta[1]:.3f}), max_radius={R:.3f}")
        
        return c, a_theta, R, theta_true, r_true
    
    def _compute_spatial_pc1(self):
        """
        Compute first spatial principal component from continuous SVGs for anchoring
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from sklearn.decomposition import TruncatedSVD
        
        # Step 1: Filter to continuous genes (5%-95% expression)
        st_expr = self.st_gene_expr.cpu().numpy()
        nonzero_frac = (st_expr > 0).mean(0)
        mask = (nonzero_frac >= 0.05) & (nonzero_frac <= 0.95)
        expr_cont = st_expr[:, mask]
        
        print(f"Filtered to {expr_cont.shape[1]} continuous genes from {st_expr.shape[1]} total")
        
        if expr_cont.shape[1] < 10:
            print("Warning: Too few continuous genes, using all genes")
            expr_cont = st_expr
        
        # Step 2: Smooth expression and compute PCA
        expr_smooth = gaussian_filter(expr_cont, sigma=(1, 0))  # smooth over spots only
        
        # Compute first PC
        svd = TruncatedSVD(n_components=1, random_state=42)
        pc1 = svd.fit_transform(expr_smooth).flatten()
        
        print(f"PC-1 explains {svd.explained_variance_ratio_[0]:.3f} of spatial variance")
        
        return torch.tensor(pc1, device=self.device, dtype=torch.float32)

    def fine_tune_decoder_boundary(self,
                                epochs=12,
                                batch_size=1024,
                                lambda_hull=5.0,
                                outlier_sigma=0.5,
                                shuffle=True):
        """
        Fine-tunes ONLY the Graph-VAE decoder so decoded coords stay inside the ST hull.
        - shared encoder: FROZEN
        - graph VAE encoder: FROZEN
        - diffusion: untouched
        """
        device = self.device

        # 1) Freeze everything except decoder
        self.netE.eval()
        for p in self.netE.parameters(): 
            p.requires_grad_(False)
        for p in self.graph_vae_encoder.parameters(): 
            p.requires_grad_(False)

        # VERY IMPORTANT: ensure decoder is on device and trainable
        self.graph_vae_decoder.to(device)
        for p in self.graph_vae_decoder.parameters():
            p.requires_grad_(True)
        self.graph_vae_decoder.train()

        opt_dec = torch.optim.Adam(self.graph_vae_decoder.parameters(), lr=1e-4)

        # 2) ST latents (mu) and targets (coords) for recon term
        with torch.no_grad():
            h_st = self.netE(self.st_gene_expr).float()                   # (N_st, d_embed)
            # use coords for edges if you like; not needed for FT here
            st_adj_idx, st_adj_w = precompute_knn_edges(h_st, k=30, device=device)
            mu_st, _ = self.graph_vae_encoder(h_st, st_adj_idx, st_adj_w) # (N_st, d_z)
            Y_st     = self.st_coords_norm                                # (N_st, 2)

        ds = torch.utils.data.TensorDataset(mu_st.detach(), Y_st.detach())
        batch_size = min(batch_size, len(mu_st))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        # 3) Precompute convex-hull penalty
        E_hull = make_hull_penalty(self.st_coords_norm.detach().cpu().numpy(), device)

        # 4) Training loop (ensure grad is enabled here)
        torch.set_grad_enabled(True)
        for epoch in range(epochs):
            running = []
            for mu_batch, y_true in dl:
                mu_batch = mu_batch.to(device)          # (B, d_z), no grad needed on inputs
                y_true   = y_true.to(device)            # (B, 2)

                # a) standard recon on in-support latents
                y_hat = self.graph_vae_decoder(mu_batch)        # requires grad w.r.t. decoder params
                loss_recon = ((y_hat - y_true) ** 2).mean()

                # b) small "outlier" latents to shape hull
                z_out = mu_batch + outlier_sigma * torch.randn_like(mu_batch)
                y_out = self.graph_vae_decoder(z_out)           # requires grad
                loss_hull = E_hull(y_out)                       # tensor on 'device'

                loss = loss_recon + lambda_hull * loss_hull

                opt_dec.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.graph_vae_decoder.parameters(), 1.0)
                opt_dec.step()

                running.append([loss_recon.item(), loss_hull.item(), loss.item()])

            rec, hul, tot = (torch.tensor(running).mean(dim=0)).tolist()
            print(f"[decoder FT] epoch {epoch+1:02d}/{epochs}  recon={rec:.4f}  hull={hul:.4f}  total={tot:.4f}")

        self.graph_vae_decoder.eval()
        print("Decoder fine-tune done (encoder + diffusion unchanged).")


    def train_diffusion_latent(
        self,
        n_epochs: int = 400,
        p_drop_max: float = 0.05,
        struct_warmup_epochs: int = None,
        posterior_temp_floor: float = 0.30,
        save_every: int = 500,
    ):
        """
        Train a conditional DDPM prior over Graph-VAE latents.
        Adds (i) condition dropout and (ii) structure-aware loss by decoding x0
        and matching local edge distances in coordinate space.
        """
        print("Training latent-space diffusion model...")

        # Freeze encoder + VAE (encoder, *and* decoder) during diffusion training
        self.netE.eval()
        self.graph_vae_encoder.eval()
        self.graph_vae_decoder.eval()
        for p in self.netE.parameters():
            p.requires_grad = False
        for p in self.graph_vae_encoder.parameters():
            p.requires_grad = False
        for p in self.graph_vae_decoder.parameters():
            p.requires_grad = False

        self.noise_schedule_mode = 'linear'
        self.noise_target_alpha_end = 0.08
        self.beta_start = 1e-4
        self.sampling_start_alpha = 0.08

        # Ensure noise schedule is current
        self.update_noise_schedule()
        self.latent_denoiser.train()

        # Precompute ST latents (μ, logσ²) and ST aligned features (condition)
        print("Computing fixed ST latents...")
        with torch.no_grad():
            st_features_aligned = self.netE(self.st_gene_expr).float()             # (N, d_embed)
            st_adj_idx_full, _ = precompute_knn_edges(self.st_coords_norm, k=30, device=self.device)  # not used directly (we re-make per batch)
            st_mu, st_logvar = self.graph_vae_encoder(st_features_aligned, st_adj_idx_full, None)     # (N, D), (N, D)

        #quick adapter fit: embedding to latent
        opt_adapt = torch.optim.AdamW(self.cond_adapter.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(5):
            idx_adapt = torch.randperm(st_mu.size(0), device=self.device)
            c_b = self._whiten_cond(st_features_aligned[idx_adapt])
            z_tgt = st_mu[idx_adapt].detach()
            loss_ad = F.mse_loss(self.cond_adapter(c_b), z_tgt)
            opt_adapt.zero_grad()
            loss_ad.backward()
            opt_adapt.step()

        # Optimizer / scheduler / EMA for the denoiser
        optimizer = torch.optim.AdamW(self.latent_denoiser.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
        from torch.optim.swa_utils import AveragedModel
        ema = AveragedModel(self.latent_denoiser, avg_fn=None)

        # Warmups
        if struct_warmup_epochs is None:
            struct_warmup_epochs = int(0.7 * n_epochs)  # 0 → 2.0 over first ~40% epochs

        # Optional: learned null condition (better than zeros)
        if not hasattr(self, "null_cond"):
            with torch.no_grad():
                null_c = self._whiten_cond(st_features_aligned.mean(0, keepdim=True))
                null_z = self.cond_adapter(null_c)
                null_full = torch.cat([null_c, null_z], dim=1)
            self.null_cond = nn.Parameter(null_full)
            #     null_init = st_features_aligned.mean(0, keepdim=True)
            # self.null_cond = nn.Parameter(null_init)  # trainable

        # Train
        for epoch in range(n_epochs):
            # Batch indices
            B = min(self.batch_size, st_mu.shape[0])
            idx = torch.randperm(st_mu.shape[0], device=self.device)[:B]
            batch_mu      = st_mu[idx]                      # (B, D)
            batch_logvar  = st_logvar[idx]                  # (B, D)
            batch_h       = st_features_aligned[idx]        # (B, d_embed)
            coords_true_b = self.st_coords_norm[idx]        # (B, 2)

            # Posterior sampling for z0 with a small temperature floor
            eps0 = torch.randn_like(batch_mu)
            posterior_std = torch.sqrt(torch.exp(batch_logvar) + posterior_temp_floor**2)
            z0  = batch_mu + posterior_std * eps0           # (B, D)

            # Sample timesteps
            t = torch.randint(0, self.n_timesteps, (B,), device=self.device)
            alpha_bar_t = self.noise_schedule['alphas_cumprod'][t].view(-1, 1)

            # Forward noising
            eps = torch.randn_like(z0)
            z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * eps

            # ----- condition dropout (linear warmup) -----
            # p_drop = p_drop_max * min(epoch / max(1, struct_warmup_epochs), 1.0)
            # drop_mask = (torch.rand(B, 1, device=self.device) >= p_drop).float()
            # cond = torch.where(drop_mask.bool(), batch_h, self.null_cond.expand_as(batch_h))
            c_b = self._whiten_cond(batch_h)                        # whitened embeddings
            z_hint = self.cond_adapter(c_b).detach()                # embeddings->latent guess
            cond_full = torch.cat([c_b, z_hint], dim=1)           

            # ----- condition dropout (linear warmup) -----
            p_drop = p_drop_max * min(epoch / max(1, struct_warmup_epochs), 1.0)
            drop_mask = (torch.rand(B, 1, device=self.device) >= p_drop).float()
            cond = drop_mask * cond_full + (1.0 - drop_mask) * self.null_cond.expand_as(cond_full)

            # Predict noise
            t_norm = (t.float() / max(self.n_timesteps - 1, 1)).unsqueeze(1)       # (B,1)
            eps_pred = self.latent_denoiser(z_t, t_norm, cond)

            # Diffusion (simple MSE)
            loss_diffusion = F.mse_loss(eps_pred, eps)

            # ---- structure-aware loss (decode x0, match edge distances) ----
            with torch.no_grad():
                # kNN edges for THIS batch (keeps indices local to [0..B-1])
                batch_edge_idx, _ = precompute_knn_edges(coords_true_b.detach(), k=30, device=self.device)
            i, j = batch_edge_idx[0], batch_edge_idx[1]

            # # x0 prediction and decoding
            sqrt_ab  = torch.sqrt(alpha_bar_t)
            sqrt_1m  = torch.sqrt(1.0 - alpha_bar_t)
            z0_pred  = (z_t - sqrt_1m * eps_pred) / (sqrt_ab + 1e-8)               # DDPM x0
            coords_pred = self.graph_vae_decoder(z0_pred)                           # (B, 2)
            tau = max(float(self.sigma), 1e-3)

            # # Edge distances
            # dist_true = (coords_true_b[i] - coords_true_b[j]).pow(2).sum(1).sqrt()
            # dist_pred = (coords_pred[i]     - coords_pred[j]).pow(2).sum(1).sqrt()
            # L_struct  = F.smooth_l1_loss(dist_pred, dist_true)

            # Build P_batch on the corresponding ST coords (same τ)
            Xb = coords_true_b                                # use self.st_coords_norm[idx] or cached X[idx]
            D2_true_b = torch.cdist(Xb, Xb).pow(2)
            P_b = torch.softmax(-D2_true_b / (2.0 * tau * tau), dim=1)

            # Predicted neighborhoods on decoded coords
            D2_pred_b = torch.cdist(coords_pred, coords_pred).pow(2)
            logQ_b = F.log_softmax(-D2_pred_b / (2.0 * tau * tau), dim=1)

            L_struct = F.kl_div(logQ_b, P_b, reduction='batchmean')



            # x0 prediction and decoding
            # sqrt_ab  = torch.sqrt(alpha_bar_t)
            # sqrt_1m  = torch.sqrt(1.0 - alpha_bar_t)
            # z0_pred  = (z_t - sqrt_1m * eps_pred) / (sqrt_ab + 1e-8)
            # coords_pred = self.graph_vae_decoder(z0_pred)

            # # Apply affine for ST structure loss only
            # coords_for_loss = coords_pred @ self.affine_A.T + self.affine_b

            # # Edge distances
            # dist_true = (coords_true_b[i] - coords_true_b[j]).pow(2).sum(1).sqrt()
            # dist_pred = (coords_for_loss[i] - coords_for_loss[j]).pow(2).sum(1).sqrt()
            # L_struct  = F.smooth_l1_loss(dist_pred, dist_true)


            # Warm up structure weight: 0 → 2.0
            lambda_struct = 1 * min(epoch / max(1, struct_warmup_epochs), 1.0)

            total_loss = loss_diffusion + lambda_struct * L_struct

            # Bookkeeping
            self.latent_diffusion_losses.setdefault('total', []).append(total_loss.item())
            self.latent_diffusion_losses.setdefault('diffusion', []).append(loss_diffusion.item())
            self.latent_diffusion_losses.setdefault('struct', []).append(L_struct.item())
            self.latent_diffusion_losses.setdefault('lambda_struct', []).append(lambda_struct)
            self.latent_diffusion_losses.setdefault('epochs', []).append(epoch)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.latent_denoiser.parameters(), 1.0)
            optimizer.step()
            ema.update_parameters(self.latent_denoiser)
            scheduler.step()

            # Logs / checkpoints
            if (epoch % save_every) == 0:
                msg = (f"[latent-diff] epoch {epoch}/{n_epochs}  "
                    f"loss={total_loss.item():.6f}  "
                    f"diff={loss_diffusion.item():.6f}  "
                    f"struct={L_struct.item():.6f}  "
                    f"λ_struct={lambda_struct:.3f}  p_drop={p_drop:.3f}")
                print(msg)
                with open(self.train_log, 'a') as f:
                    f.write(msg + "\n")
                torch.save({
                    'epoch': epoch,
                    'latent_denoiser_state_dict': self.latent_denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(self.outf, f'latent_diffusion_checkpoint_epoch_{epoch}.pt'))

        # Save final weights
        torch.save({'latent_denoiser_state_dict': self.latent_denoiser.state_dict()},
                os.path.join(self.outf, 'final_latent_diffusion.pt'))
        torch.save({'latent_denoiser_state_dict': ema.module.state_dict()},
                os.path.join(self.outf, 'final_latent_diffusion_ema.pt'))
        self.latent_denoiser_ema = ema.module
        print("Latent diffusion training complete!")

                        

    def train(self, encoder_epochs=1000, vae_epochs=800, diffusion_epochs=400, **kwargs):
        """
        Combined training pipeline: encoder → graph_vae → diffusion_latent
        ⚠️ Do **not** touch `train_encoder`; its aligned embeddings are the sole conditioning signal throughout.
        """
        print("Starting Graph-VAE + Latent Diffusion training pipeline...")
        
        # Stage 1: Train encoder (DO NOT MODIFY - keep existing train_encoder)
        print("Stage 1: Training domain alignment encoder...")
        self.train_encoder(n_epochs=encoder_epochs)
        
        # Stage 2: Train Graph-VAE
        print("Stage 2: Training Graph-VAE...")
        self.train_graph_vae(epochs=vae_epochs)
        
        # Stage 3: Train latent diffusion
        print("Stage 3: Training latent diffusion...")
        self.train_diffusion_latent(n_epochs=diffusion_epochs, **kwargs)
        
        print("Complete training pipeline finished!")

    
    def sample_sc_coordinates_batched(self, batch_size: int = 512, return_normalized=True):
        """
        Minimal, stable sampler:
        amortized z0 (from GraphVAEEncoder on SC embedding) →
        forward-noise to T → plain reverse DDPM (conditioned on h_sc) → decode once.
        """
        n_total = len(self.sc_gene_expr)
        print(f"[sampler] SC cells={n_total}; batch={batch_size}; plain reverse diffusion")

        # Eval modes
        self.netE.eval()
        self.graph_vae_encoder.eval()
        self.graph_vae_decoder.eval()
        denoiser = getattr(self, "latent_denoiser_ema", self.latent_denoiser).eval()

        # Noise schedule up to date
        self.update_noise_schedule()
        T = self.n_timesteps - 1
        alpha_bar_T = self.noise_schedule['alphas_cumprod'][T]  # scalar tensor

        all_coords = []
        n_batches = (n_total + batch_size - 1) // batch_size

        with torch.no_grad():
            for b in range(n_batches):
                s = b * batch_size
                e = min(s + batch_size, n_total)
                X_sc = self.sc_gene_expr[s:e]                      # (B, G)

                # 1) shared embedding (condition)
                h_sc = self.netE(X_sc).float()                     # (B, d_embed)
                B = h_sc.shape[0]

                # 2) amortized z0 from GraphVAEEncoder on SC embedding graph
                #    (kNN on embedding, not raw expression)
                sc_adj_idx, sc_adj_w = precompute_knn_edges(h_sc, k=30, device=self.device)
                mu_sc, logvar_sc = self.graph_vae_encoder(h_sc, sc_adj_idx, sc_adj_w)
                z0 = mu_sc                                          # deterministic init helps geometry

                # 3) forward-noise z0 to T (match training terminal distribution)
                eps_T = torch.randn_like(z0)
                z_t = alpha_bar_T.sqrt() * z0 + (1.0 - alpha_bar_T).sqrt() * eps_T

                # 4) plain reverse diffusion (no guidance, no extra tricks)
                for t in reversed(range(self.n_timesteps)):
                    # normalized timestep
                    t_norm = (torch.full((B, 1), t, device=self.device) /
                            max(self.n_timesteps - 1, 1))

                    # predict noise with condition
                    eps_pred = denoiser(z_t, t_norm, h_sc)

                    # DDPM update
                    alpha_t        = self.noise_schedule['alphas'][t]
                    alpha_bar_t    = self.noise_schedule['alphas_cumprod'][t]
                    beta_t         = self.noise_schedule['betas'][t]
                    if t > 0:
                        noise = torch.randn_like(z_t)
                    else:
                        noise = 0

                    z_t = (1.0 / alpha_t.sqrt()) * (
                            z_t - ((1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()) * eps_pred
                        ) + beta_t.sqrt() * noise

                # 5) decode once
                coords = self.graph_vae_decoder(z_t)                # (B, 2)
                all_coords.append(coords.detach().cpu())

        final_coords = torch.cat(all_coords, dim=0)                 # (N, 2)
        print("[sampler] done.")

        if return_normalized:
            return final_coords
        else:
            return self.denormalize_coordinates(final_coords)
        # return final_coords.numpy()

    
    from tqdm import tqdm


    def sample_sc_coordinates_pure_diffusion(self, batch_size=512, guidance_scale=1.0, return_normalized=True):
        """
        Pure diffusion sampling: Start from noise → denoise → decode.
        No Graph VAE used for SC inference at all.
        """
        n_total = len(self.sc_gene_expr)
        print(f"Sampling {n_total} SC coordinates using pure latent diffusion...")
        
        # Set models to eval mode
        self.netE.eval()
        self.graph_vae_decoder.eval()  # Only for final decoding
        self.latent_denoiser.eval()
        self.update_noise_schedule()
        
        all_coords = []
        n_batches = (n_total + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches), desc="Sampling batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_total)
                batch_sc_expr = self.sc_gene_expr[start_idx:end_idx]
                
                # Get SC conditioning (gene expression embeddings only)
                h_sc = self.netE(batch_sc_expr).float()  # (B, d_embed)
                B = h_sc.size(0)
                
                # Start from PURE NOISE in latent space (no Graph VAE)
                z_t = torch.randn(B, self.latent_dim, device=self.device)
                
                # Reverse diffusion with classifier-free guidance
                for t in reversed(range(self.n_timesteps)):
                    t_norm = torch.full((B, 1), t / max(self.n_timesteps - 1, 1), device=self.device)
                    
                    # Classifier-free guidance
                    eps_c = self.latent_denoiser(z_t, t_norm, h_sc)          # Conditioned
                    eps_u = self.latent_denoiser(z_t, t_norm, torch.zeros_like(h_sc))  # Unconditioned
                    eps_pred = (1 + guidance_scale) * eps_c - guidance_scale * eps_u
                    
                    # DDPM reverse step
                    alpha_t = self.noise_schedule['alphas'][t]
                    alpha_bar_t = self.noise_schedule['alphas_cumprod'][t]
                    
                    mu = (1 / torch.sqrt(alpha_t)) * (
                        z_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred
                    )
                    
                    if t > 0:
                        sigma_t = torch.sqrt(self.noise_schedule['posterior_variance'][t])
                        z_t = mu + sigma_t * torch.randn_like(z_t)
                    else:
                        z_t = mu
                
                # Decode final latent to coordinates (decoder only)
                batch_coords = self.graph_vae_decoder(z_t)
                all_coords.append(batch_coords.cpu())
                
                torch.cuda.empty_cache()
        
        final_coords = torch.cat(all_coords, dim=0)
        print(f"Pure diffusion sampling complete! Generated {final_coords.shape[0]} coordinates.")
        if return_normalized:
            return final_coords
        else:
            return self.denormalize_coordinates(final_coords)
        
    def sample_sc_coordinates(
        self,
        batch_size=512,
        guidance_scale=2.0,
        k_nn=15, margin=0.05,
        lambda_trip=0.05, lambda_rep=0.08, lambda_ang=0.01,lambda_rad = 0.1,
        gamma=0.05,
        repulsion_sigma=None,
        lambda_stprior = 0.10,
        prior_last_steps = 50,
        temp_sigma=1.0,          # optional diversity on reverse noise (1.0 = off)
        denoiser_chunk=None,     # e.g., 256 if denoiser is big; None = full batch
        return_normalized = False
    ):
        """
        Îµ-prediction sampler with geometry guidance and tight memory use.
        Uses the denoising logic from `sample_sc_coordinates_batched`.
        """
        n_total = len(self.sc_gene_expr)
        # print(f"Sampling {n_total} SC coordinates (geom_guidance={geometry_guidance}, k={k_nn})")
        # eval + schedule
        self.netE.eval(); self.graph_vae_encoder.eval()
        self.graph_vae_decoder.eval(); 
        # self.latent_denoiser.eval()
        denoiser = getattr(self, 'latent_denoiser_ema', self.latent_denoiser)
        denoiser.eval()
        self.update_noise_schedule()

        # st_prior = None
        st_prior = STDensityPrior(self.st_coords_norm.detach().cpu().numpy(), device=self.device)

        # ---- repulsion scale from ST once (cheap; subsample if ST is huge) ----
        if repulsion_sigma is None:
            with torch.no_grad():
                n_sub = min(1024, self.st_coords_norm.shape[0])
                idx = torch.randperm(self.st_coords_norm.shape[0], device=self.device)[:n_sub]
                st_sub = self.st_coords_norm[idx]
                D = torch.cdist(st_sub, st_sub)                              # (n_sub, n_sub)
                # 5-NN distance (k=6 incl. self)
                kth = torch.kthvalue(D, 6, dim=1).values
                repulsion_sigma = (kth.median().item() / 2.5) or 0.1
            # free temp tensor
            del D

        # optional micro-batched denoiser to cap peak activation memory
        def _denoise_eps(z, tnorm, cond):
            if denoiser_chunk is None:
                # return self.latent_denoiser(z, tnorm, cond)
                return denoiser(z, tnorm, cond)
            outs = []
            for i in range(0, z.size(0), denoiser_chunk):
                outs.append(denoiser(z[i:i+denoiser_chunk], tnorm[i:i+denoiser_chunk], cond[i:i+denoiser_chunk]))
            return torch.cat(outs, dim=0)
        
        # choose a start step t_start so alpha_bar[t_start] ~ target (not near 0)
        target_alpha = 0.08  # 0.05–0.15 is typically stable
        alpha_bar = self.noise_schedule['alphas_cumprod']   # (T,)
        t_start = int(torch.argmin(torch.abs(alpha_bar - target_alpha)).item())
        alpha_bar_start = alpha_bar[t_start]

        all_coords = []
        n_batches = (n_total + batch_size - 1) // batch_size

        with torch.no_grad():  # keep denoiser & DDPM math gradient-free
            # for b in range(n_batches):
            for b in tqdm(range(n_batches), desc="Sampling batches", unit="batch"):
                s = b * batch_size
                e = min(s + batch_size, n_total)
                batch_sc_expr = self.sc_gene_expr[s:e]

                # conditioning embedding
                h_sc = self.netE(batch_sc_expr).float()           # (B, d_embed)
                B = h_sc.size(0)

                # --- kNN once per batch (embedding space) ---
                D_h = torch.cdist(h_sc, h_sc)                     # (B, B) ~ 1 MB when B=512
                nn_idx = D_h.topk(k=k_nn+1, largest=False).indices[:, 1:]    # (B, k)
                far_idx = D_h.topk(k=1, largest=True).indices.squeeze(1)     # (B,)

                # --- baseline for tiny angle anchor (SC kNN graph; cheap) ---
                sc_adj_idx, sc_adj_w = precompute_knn_edges(h_sc, k=30, device=self.device)
                mu_sc, logvar_sc = self.graph_vae_encoder(h_sc, sc_adj_idx, sc_adj_w)  # (B, d_z)

                #whiten + adapter
                h_sc = self._whiten_cond(h_sc)
                z_hint = self.cond_adapter(h_sc)

                #reparam to get a posterior sample
                z_0 = mu_sc + torch.randn_like(mu_sc) * torch.exp(0.5 * logvar_sc)
                alpha_blend = 0.3 
                z_0 = (1.0 - alpha_blend) * z_0 + alpha_blend * z_hint 
                y_base = self.graph_vae_decoder(mu_sc)                         # (B, 2)

                #diffuse forward to the last timestep T so we start at the same noise level used during training
                # T = self.n_timesteps - 1
                # alpha_bar_T = self.noise_schedule['alphas_cumprod'][T]
                # z_t = (
                #     torch.sqrt(alpha_bar_T) * z_0 +
                #     torch.sqrt(1.0 - alpha_bar_T) * torch.randn_like(z_0)
                # )

                      # 3) forward-noise to the chosen start noise level (alpha_bar_start)
                z_t = (
                    torch.sqrt(alpha_bar_start) * z_0 +
                    torch.sqrt(1.0 - alpha_bar_start) * torch.randn_like(z_0)
                )

                # -------- reverse diffusion (Îµ-pred, classifier-free) --------
                # for t in reversed(range(self.n_timesteps)):
                for t in reversed(range(t_start + 1)):
                    # normalized timestep
                    t_norm = torch.full((B, 1), t / max(self.n_timesteps - 1, 1), device=self.device)
                    cond_full = torch.cat([h_sc, z_hint], dim=1)

                    # Îµ-pred with CF guidance (denoiser in inference mode â†’ tiny memory)
                    with torch.inference_mode():
                        eps_c = _denoise_eps(z_t, t_norm, cond_full) #changed from h_sc to cond_full, comment for later revision
                        eps_u = _denoise_eps(z_t, t_norm, torch.zeros_like(cond_full))
                        eps_pred = (1.0 + guidance_scale) * eps_c - guidance_scale * eps_u
                        # eps_pred = eps_u + guidance_scale * (eps_c - eps_u)

                    # DDPM reverse mean (same as your old code)
                    alpha_t = self.noise_schedule['alphas'][t]
                    alpha_bar_t = self.noise_schedule['alphas_cumprod'][t]
                    mu = (1.0 / torch.sqrt(alpha_t)) * (
                        z_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
                    )

                    # add noise with posterior variance (optionally temperature-scaled)
                    if t > 0:
                        sigma_post = torch.sqrt(self.noise_schedule['posterior_variance'][t])
                        z_t = mu + sigma_post * temp_sigma * torch.randn_like(z_t)
                    else:
                        z_t = mu

                # decode final latent â†’ coords (z only)
                batch_coords = self.graph_vae_decoder(z_t)
                all_coords.append(batch_coords.cpu())

                # free short-lived tensors
                del D_h, nn_idx, far_idx, sc_adj_idx, sc_adj_w, mu_sc, y_base, z_t
                torch.cuda.empty_cache()

        coords_final = torch.cat(all_coords, dim=0)
        print(f"Generated {coords_final.shape[0]} SC coordinates.")

        if return_normalized:
            return coords_final
        else:
            return self.denormalize_coordinates(coords_final)

    def denormalize_coordinates(self, normalized_coords):
        '''convert normalized coords back to original scale'''
        if isinstance(normalized_coords, torch.Tensor):
            coords_radius = self.coords_radius.to(normalized_coords.device)
            coords_center = self.coords_center.to(normalized_coords.device)
            original_coords = normalized_coords * coords_radius + coords_center
            return original_coords
        else:
            coords_radius = self.coords_radius.cpu().numpy()
            coords_center = self.coords_center.cpu().numpy()
            original_coords = normalized_coords * coords_radius + coords_center
            return original_coords


    def _compute_geometry_metrics(self, sc_coords, h_sc):
        """
        Compute geometry preservation metrics for evaluation.
        """
        with torch.no_grad():
            metrics = {}
            
            # 1. kNN agreement between embedding and coordinates
            k_values = [5, 10]
            for k in k_values:
                # Embedding kNN
                D_h = torch.cdist(h_sc, h_sc)
                emb_knn = D_h.topk(k=k+1, largest=False).indices[:,1:]  # Exclude self
                
                # Coordinate kNN  
                D_coord = torch.cdist(sc_coords, sc_coords)
                coord_knn = D_coord.topk(k=k+1, largest=False).indices[:,1:]
                
                # Jaccard similarity
                intersection = (emb_knn.unsqueeze(2) == coord_knn.unsqueeze(1)).any(dim=2).sum(dim=1)
                jaccard = intersection.float() / (2 * k - intersection.float())
                metrics[f'knn_jaccard_k{k}'] = jaccard.mean().item()
            
            # 2. Collision detection (minimum distances)
            min_distances = D_coord.fill_diagonal_(float('inf')).min(dim=1).values
            metrics['min_distance_mean'] = min_distances.mean().item()
            metrics['min_distance_std'] = min_distances.std().item()
            metrics['collision_rate'] = (min_distances < 0.01).float().mean().item()
            
            # 3. Radial distribution compared to ST
            sc_radii = torch.norm(sc_coords, dim=1)
            st_radii = torch.norm(self.st_coords_norm, dim=1)
            
            # KS test statistic (approximation)
            sc_sorted = torch.sort(sc_radii).values
            st_sorted = torch.sort(st_radii).values
            
            # Interpolate to same length for comparison
            from scipy import stats
            ks_stat, ks_pvalue = stats.ks_2samp(
                sc_sorted.cpu().numpy(), 
                st_sorted.cpu().numpy()
            )
            metrics['radial_ks_statistic'] = ks_stat
            metrics['radial_ks_pvalue'] = ks_pvalue
            
            return metrics

    # Add this to your diffusion model class
    def evaluate_geometry_preservation(self, sc_coords, print_results=True):
        """
        Evaluate how well the generated coordinates preserve geometric structure.
        """
        with torch.no_grad():
            sc_coords = sc_coords.to(self.device)
            h_sc = self.netE(self.sc_gene_expr).float()
            metrics = self._compute_geometry_metrics(sc_coords, h_sc)
            
            if print_results:
                print("\n=== Geometry Preservation Metrics ===")
                print(f"kNN Jaccard (k=5):  {metrics['knn_jaccard_k5']:.4f}")
                print(f"kNN Jaccard (k=10): {metrics['knn_jaccard_k10']:.4f}")
                print(f"Min distance (mean ± std): {metrics['min_distance_mean']:.4f} ± {metrics['min_distance_std']:.4f}")
                print(f"Collision rate (< 0.01): {metrics['collision_rate']:.4f}")
                print(f"Radial KS statistic: {metrics['radial_ks_statistic']:.4f} (p={metrics['radial_ks_pvalue']:.4f})")
                
            return metrics


    def plot_training_losses(self):
        """Plot training losses for Graph-VAE + Latent Diffusion pipeline"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Determine how many subplots we need
        n_plots = 0
        if len(self.vae_losses['epochs']) > 0:
            n_plots += 2  # VAE losses and VAE smoothed
        if len(self.latent_diffusion_losses['epochs']) > 0:
            n_plots += 2  # Latent diffusion losses and smoothed
        
        if n_plots == 0:
            print("No training losses to plot.")
            return
        
        # Create figure with appropriate number of subplots
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes = [axes] if n_plots == 2 else axes
        elif n_plots == 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Graph-VAE losses
        if len(self.vae_losses['epochs']) > 0:
            epochs_vae = np.array(self.vae_losses['epochs'])
            ax = axes[plot_idx]
            
            ax.plot(epochs_vae, self.vae_losses['total'], 'b-', label='Total VAE Loss', linewidth=2)
            ax.plot(epochs_vae, self.vae_losses['reconstruction'], 'g-', label='Reconstruction Loss', linewidth=2)
            ax.plot(epochs_vae, np.array(self.vae_losses['kl']) * 0.01, 'r--', label='KL Loss (×0.01)', alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Graph-VAE Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plot_idx += 1
            
            # Plot 2: Graph-VAE smoothed
            if len(self.vae_losses['total']) > 1:
                ax = axes[plot_idx]
                window = min(50, len(self.vae_losses['total']) // 10)
                if window > 1:
                    smoothed = np.convolve(self.vae_losses['total'], 
                                        np.ones(window)/window, mode='valid')
                    smooth_epochs = epochs_vae[window-1:]
                    ax.plot(epochs_vae, self.vae_losses['total'], 'lightblue', alpha=0.5, label='Raw')
                    ax.plot(smooth_epochs, smoothed, 'blue', linewidth=2, label=f'Smoothed (window={window})')
                else:
                    ax.plot(epochs_vae, self.vae_losses['total'], 'blue', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Graph-VAE Total Loss (Smoothed)')
                if window > 1:
                    ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                plot_idx += 1
        
        # Plot 3: Latent Diffusion losses
        if len(self.latent_diffusion_losses['epochs']) > 0:
            epochs_diff = np.array(self.latent_diffusion_losses['epochs'])
            ax = axes[plot_idx]
            
            ax.plot(epochs_diff, self.latent_diffusion_losses['total'], 'b-', label='Total Loss', linewidth=2)
            ax.plot(epochs_diff, self.latent_diffusion_losses['diffusion'], 'k-', label='Diffusion Loss', linewidth=2)
            
            # Only plot struct loss if it's non-zero
            struct_losses = np.array(self.latent_diffusion_losses['struct'])
            if np.any(struct_losses > 0):
                ax.plot(epochs_diff, struct_losses, 'r--', label='Structure Loss', alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Latent Diffusion Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plot_idx += 1
            
            # Plot 4: Latent Diffusion smoothed
            if len(self.latent_diffusion_losses['total']) > 1:
                ax = axes[plot_idx]
                window = min(50, len(self.latent_diffusion_losses['total']) // 10)
                if window > 1:
                    smoothed = np.convolve(self.latent_diffusion_losses['total'],
                                        np.ones(window)/window, mode='valid')
                    smooth_epochs = epochs_diff[window-1:]
                    ax.plot(epochs_diff, self.latent_diffusion_losses['total'], 'lightcoral', alpha=0.5, label='Raw')
                    ax.plot(smooth_epochs, smoothed, 'red', linewidth=2, label=f'Smoothed (window={window})')
                else:
                    ax.plot(epochs_diff, self.latent_diffusion_losses['total'], 'red', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Latent Diffusion Total Loss (Smoothed)')
                if window > 1:
                    ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        # Print final loss values
        print("\n=== Training Loss Summary ===")
        
        if len(self.vae_losses['total']) > 0:
            print(f"Graph-VAE - Initial Loss: {self.vae_losses['total'][0]:.6f}")
            print(f"Graph-VAE - Final Loss: {self.vae_losses['total'][-1]:.6f}")
            print(f"Graph-VAE - Loss Reduction: {(1 - self.vae_losses['total'][-1]/self.vae_losses['total'][0])*100:.2f}%")
            print(f"Graph-VAE - Final Reconstruction Loss: {self.vae_losses['reconstruction'][-1]:.6f}")
            print(f"Graph-VAE - Final KL Loss: {self.vae_losses['kl'][-1]:.6f}")
        
        if len(self.latent_diffusion_losses['total']) > 0:
            print(f"Latent Diffusion - Initial Loss: {self.latent_diffusion_losses['total'][0]:.6f}")
            print(f"Latent Diffusion - Final Loss: {self.latent_diffusion_losses['total'][-1]:.6f}")
            print(f"Latent Diffusion - Loss Reduction: {(1 - self.latent_diffusion_losses['total'][-1]/self.latent_diffusion_losses['total'][0])*100:.2f}%")
            print(f"Latent Diffusion - Final Diffusion Loss: {self.latent_diffusion_losses['diffusion'][-1]:.6f}")
            if np.any(np.array(self.latent_diffusion_losses['struct']) > 0):
                print(f"Latent Diffusion - Final Structure Loss: {self.latent_diffusion_losses['struct'][-1]:.6f}")


class HierarchicalDiffusionBlock(nn.Module):
    """Multi-scale diffusion block for coarse-to-fine generation"""
    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # Coarse-level predictor (for clusters/regions)
        self.coarse_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Fine-level predictor (for individual cells)
        self.fine_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),  # Takes both coarse and fine features
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Scale mixing weights
        self.scale_mixer = nn.Sequential(
            nn.Linear(1, 64),  # Takes timestep
            nn.ReLU(),
            nn.Linear(64, num_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, t, coarse_context=None):
        # Determine scale weights based on timestep
        scale_weights = self.scale_mixer(t.unsqueeze(-1))
        
        # Coarse prediction
        coarse_pred = self.coarse_net(x)
        
        # Fine prediction (conditioned on coarse if available)
        if coarse_context is not None:
            fine_input = torch.cat([x, coarse_context], dim=-1)
        else:
            fine_input = torch.cat([x, coarse_pred], dim=-1)
        fine_pred = self.fine_net(fine_input)
        
        # Mix scales based on timestep
        output = scale_weights[:, 0:1] * coarse_pred + scale_weights[:, 1:2] * fine_pred
        
        return output  
    

class SpatialBatchSampler:
    """Sample spatially contiguous batches for geometric attention"""
    
    def __init__(self, coordinates, batch_size, k_neighbors=None):
        """
        coordinates: (N, 2) array of spatial coordinates
        batch_size: size of each batch
        k_neighbors: number of neighbors to precompute (default: batch_size)
        """
        self.coordinates = coordinates
        self.batch_size = batch_size
        self.k_neighbors = k_neighbors or min(batch_size, len(coordinates))
        
        # Precompute nearest neighbors
        self.nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors, 
            algorithm='kd_tree'
        ).fit(coordinates)
        
    def sample_spatial_batch(self):
        """Sample a spatially contiguous batch"""
        # Pick random center point
        center_idx = np.random.randint(len(self.coordinates))
        
        # Get k nearest neighbors
        distances, indices = self.nbrs.kneighbors(
            self.coordinates[center_idx:center_idx+1], 
            return_distance=True
        )
        
        # Return indices as torch tensor
        batch_indices = torch.tensor(indices.flatten()[:self.batch_size], dtype=torch.long)
        return batch_indices
    
class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        tmp = 0
        for x in kernel_val:
            tmp += x
        return tmp

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def analyze_sc_st_patterns(model):
    """Run the complete analysis"""
    
    print("Analyzing SC vs ST expression patterns...")
    
    # Main comparison plot
    common_genes = model.compare_sc_st_expression_patterns(n_genes=20)
    
    # Detailed gene-by-gene analysis
    print(f"\nDetailed analysis for top {len(common_genes)} variable genes...")
    model.plot_detailed_gene_comparison(common_genes, n_genes=10)
    
    # Print some statistics
    print("\nExpression Statistics:")
    print(f"SC data shape: {model.sc_gene_expr.shape}")
    print(f"ST data shape: {model.st_gene_expr.shape}")
    
    with torch.no_grad():
        sc_mean = model.sc_gene_expr.mean(0)
        st_mean = model.st_gene_expr.mean(0)
        
        print(f"SC mean expression: {sc_mean.mean():.3f} ± {sc_mean.std():.3f}")
        print(f"ST mean expression: {st_mean.mean():.3f} ± {st_mean.std():.3f}")