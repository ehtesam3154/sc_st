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

# Import from project knowledge Set-Transformer components
from modules import MAB, SAB, ISAB, PMA
import utils_et as uet

# ==============================================================================
# STAGE C: SET-EQUIVARIANT CONTEXT ENCODER
# ==============================================================================

class SetEncoderContext(nn.Module):
    """
    Permutation-equivariant context encoder using Set Transformer.
    
    Takes set of embeddings Z_set and produces context H.
    Uses ISAB blocks for O(mn) complexity.
    """
    
    def __init__(
        self,
        h_dim: int = 128,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 3,
        isab_m: int = 64,
        ln: bool = True
    ):
        """
        Args:
            h_dim: input embedding dimension
            c_dim: output context dimension
            n_heads: number of attention heads
            n_blocks: number of ISAB blocks
            isab_m: number of inducing points in ISAB
            ln: use layer normalization
        """
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim
        
        # Input projection
        self.input_proj = nn.Linear(h_dim, c_dim)
        
        # Stack of ISAB blocks
        self.isab_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])
    
    def forward(self, Z_set: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_set: (batch, n, h_dim) set of embeddings
            mask: (batch, n) boolean mask (True = valid)
            
        Returns:
            H: (batch, n, c_dim) context features
        """
        batch_size, n, _ = Z_set.shape
        
        # Project to context dimension
        H = self.input_proj(Z_set)  # (batch, n, c_dim)
        
        # Apply ISAB blocks
        for isab in self.isab_blocks:
            # Reshape for ISAB: (batch*n, c_dim) → process → (batch, n, c_dim)
            H_flat = H.view(batch_size * n, self.c_dim)
            H_flat = isab(H_flat.unsqueeze(1)).squeeze(1)  # ISAB expects (B, 1, D)
            H = H_flat.view(batch_size, n, self.c_dim)
            
            # Apply mask
            H = H * mask.unsqueeze(-1).float()
        
        return H


# ==============================================================================
# STAGE C: METRIC SET GENERATOR Φθ
# ==============================================================================

class MetricSetGenerator(nn.Module):
    """
    Generator that produces V ∈ ℝ^{n×D} from context H.
    
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
            X_flat = X.view(batch_size * n, self.c_dim)
            X_flat = isab(X_flat.unsqueeze(1)).squeeze(1)
            X = X_flat.view(batch_size, n, self.c_dim)
            X = X * mask.unsqueeze(-1).float()
        
        # MLP head
        V = self.head(X)  # (batch, n, D_latent)
        
        # Row-mean centering (translation neutrality)
        V_centered = V - V.mean(dim=1, keepdim=True)
        
        # Apply mask
        V_centered = V_centered * mask.unsqueeze(-1).float()
        
        return V_centered
    
# ==============================================================================
# STAGE C: DIFFUSION SCORE NETWORK sψ
# ==============================================================================

class DiffusionScoreNet(nn.Module):
    """
    Conditional denoiser for V_t → ε̂.
    
    Set-equivariant architecture with time embedding.
    VE SDE: σ_min=0.01, σ_max=50
    """
    
    def __init__(
        self,
        D_latent: int = 16,
        c_dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        isab_m: int = 64,
        time_emb_dim: int = 128,
        ln: bool = True
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
        
        # Time embedding (Fourier features)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, c_dim),
            nn.SiLU(),
            nn.Linear(c_dim, c_dim)
        )
        
        # Input projection: V_t + H → combined features
        self.input_proj = nn.Linear(D_latent + c_dim, c_dim)
        
        # Denoising blocks (ISAB)
        self.denoise_blocks = nn.ModuleList([
            ISAB(c_dim, c_dim, n_heads, isab_m, ln=ln)
            for _ in range(n_blocks)
        ])
        
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
    
    def forward(
        self,
        V_t: torch.Tensor,
        t: torch.Tensor,
        H: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise ε̂ from noisy V_t.
        
        Args:
            V_t: (batch, n, D_latent) noisy factor
            t: (batch, 1) normalized time
            H: (batch, n, c_dim) context
            mask: (batch, n) boolean mask
            
        Returns:
            eps_pred: (batch, n, D_latent) predicted noise
        """
        batch_size, n, _ = V_t.shape
        
        # Time embedding
        t_emb = self.get_time_embedding(t)  # (batch, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (batch, c_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, n, -1)  # (batch, n, c_dim)
        
        # Combine V_t and H
        V_H = torch.cat([V_t, H], dim=-1)  # (batch, n, D_latent + c_dim)
        X = self.input_proj(V_H)  # (batch, n, c_dim)
        
        # Add time embedding
        X = X + t_emb
        
        # Apply denoising blocks
        for block in self.denoise_blocks:
            X_flat = X.view(batch_size * n, self.c_dim)
            X_flat = block(X_flat.unsqueeze(1)).squeeze(1)
            X = X_flat.view(batch_size, n, self.c_dim)
            X = X * mask.unsqueeze(-1).float()
        
        # Predict noise
        eps_pred = self.output_head(X)  # (batch, n, D_latent)
        
        # Apply mask
        eps_pred = eps_pred * mask.unsqueeze(-1).float()
        
        return eps_pred
    
# ==============================================================================
# STAGE C: TRAINING FUNCTION
# ==============================================================================

def train_stageC_diffusion_generator(
    context_encoder: SetEncoderContext,
    generator: MetricSetGenerator,
    score_net: DiffusionScoreNet,
    dataset: torch.utils.data.Dataset,
    n_epochs: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-4,
    n_timesteps: int = 1000,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    loss_weights: Dict[str, float] = {'alpha': 0.1, 'beta': 1.0, 'gamma': 0.25, 'eta': 0.5},
    device: str = 'cuda',
    outf: str = 'output'
):
    """
    Train Stage C: diffusion generator on mini-sets.
    
    Losses:
    1. Score loss (VE DSM)
    2. Frobenius Gram match
    3. Heat kernel trace matching
    4. Distance histogram SW
    5. Ordinal triplet loss
    
    Args:
        context_encoder: SetEncoderContext
        generator: MetricSetGenerator (not used in diffusion training)
        score_net: DiffusionScoreNet
        dataset: STSetDataset
        n_epochs: number of epochs
        batch_size: batch size
        lr: learning rate
        n_timesteps: number of diffusion timesteps
        sigma_min, sigma_max: VE SDE noise levels
        loss_weights: {'alpha', 'beta', 'gamma', 'eta'}
        device: torch device
        outf: output directory
    """
    context_encoder = context_encoder.to(device).train()
    score_net = score_net.to(device).train()
    
    # Optimizer
    params = list(context_encoder.parameters()) + list(score_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # VE SDE noise schedule
    sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), n_timesteps, device=device))
    
    # Loss modules
    loss_gram = uet.FrobeniusGramLoss()
    loss_heat = uet.HeatKernelLoss()
    loss_sw = uet.SlicedWassersteinLoss1D()
    loss_triplet = uet.OrdinalTripletLoss()
    
    # DataLoader
    from torch.utils.data import DataLoader
    from core_models_et_p1 import collate_minisets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=collate_minisets, num_workers=0)
    
    print(f"Training Stage C for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        epoch_losses = {'score': [], 'gram': [], 'heat': [], 'sw': [], 'triplet': [], 'total': []}
        
        for batch in dataloader:
            Z_set = batch['Z_set'].to(device)
            V_target = batch['V_target'].to(device)
            G_target = batch['G_target'].to(device)
            D_target = batch['D_target'].to(device)
            H_target = batch['H_target'].to(device)
            mask = batch['mask'].to(device)
            
            batch_size_real = Z_set.shape[0]
            n_max = Z_set.shape[1]
            
            # 1. Context encoding
            H = context_encoder(Z_set, mask)  # (batch, n, c_dim)
            
            # 2. Sample timestep and noise
            t_idx = torch.randint(0, n_timesteps, (batch_size_real,), device=device)
            t_norm = t_idx.float() / (n_timesteps - 1)  # Normalize to [0, 1]
            t_norm = t_norm.unsqueeze(1)  # (batch, 1)
            
            sigma_t = sigmas[t_idx].view(batch_size_real, 1, 1)  # (batch, 1, 1)
            
            # 3. Add noise to V_target
            eps = torch.randn_like(V_target)
            V_t = V_target + sigma_t * eps
            
            # Center V_t (important!)
            V_t = V_t - V_t.mean(dim=1, keepdim=True)
            
            # 4. Predict noise
            eps_pred = score_net(V_t, t_norm, H, mask)
            
            # 5. Score loss (masked MSE)
            mask_expanded = mask.unsqueeze(-1).float()
            L_score = ((eps - eps_pred) ** 2 * mask_expanded).sum() / mask_expanded.sum()
            
            # 6. Predict clean sample for auxiliary losses
            V_hat = V_t - sigma_t * eps_pred
            V_hat = V_hat - V_hat.mean(dim=1, keepdim=True)  # Recenter
            
            # Compute G_hat, D_hat
            G_hat_list = []
            D_hat_list = []
            for i in range(batch_size_real):
                n_valid = mask[i].sum().item()
                V_valid = V_hat[i, :n_valid]
                G_valid = V_valid @ V_valid.t()
                
                # Distances from Gram
                diag = torch.diag(G_valid).unsqueeze(1)
                D_valid = torch.sqrt(torch.clamp(diag + diag.t() - 2 * G_valid, min=0))
                
                # Pad back to n_max
                G_padded = torch.zeros(n_max, n_max, device=device)
                D_padded = torch.zeros(n_max, n_max, device=device)
                G_padded[:n_valid, :n_valid] = G_valid
                D_padded[:n_valid, :n_valid] = D_valid
                
                G_hat_list.append(G_padded)
                D_hat_list.append(D_padded)
            
            G_hat = torch.stack(G_hat_list, dim=0)
            D_hat = torch.stack(D_hat_list, dim=0)
            
            # 7. Auxiliary losses
            # a) Frobenius Gram
            L_gram = 0
            for i in range(batch_size_real):
                L_gram += loss_gram(G_hat[i], G_target[i], mask[i])
            L_gram = L_gram / batch_size_real
            
            # b) Heat kernel (simplified: just on first batch item for speed)
            L_heat = torch.tensor(0.0, device=device)
            if epoch % 5 == 0:  # Compute every 5 epochs to save time
                for i in range(min(1, batch_size_real)):
                    n_valid = mask[i].sum().item()
                    if n_valid > 10:
                        # Build Laplacian from D_hat
                        D_valid = D_hat[i, :n_valid, :n_valid]
                        y_hat_reconstructed = uet.classical_mds(
                            -0.5 * (torch.eye(n_valid, device=device) - 1/n_valid) @ (D_valid**2) @ 
                            (torch.eye(n_valid, device=device) - 1/n_valid),
                            d_out=2
                        )
                        edge_index, edge_weight = uet.build_knn_graph(y_hat_reconstructed, k=20, device=device)
                        L_hat = uet.compute_graph_laplacian(edge_index, edge_weight, n_valid)
                        
                        L_target_i = batch['L_info'][i]['L']
                        L_heat += loss_heat(L_hat, L_target_i, None)
            
            # c) Distance histogram SW
            L_sw = 0
            for i in range(batch_size_real):
                n_valid = mask[i].sum().item()
                if n_valid > 10:
                    D_valid = D_hat[i, :n_valid, :n_valid]
                    d_95 = torch.quantile(D_valid[torch.triu(torch.ones_like(D_valid), diagonal=1).bool()], 0.95)
                    bins = torch.linspace(0, d_95, H_target.shape[1], device=device)
                    H_hat = uet.compute_distance_hist(D_valid, bins)
                    L_sw += loss_sw(H_hat, H_target[i])
            L_sw = L_sw / batch_size_real
            
            # d) Ordinal triplet
            L_ord = 0
            for i in range(batch_size_real):
                triplets = batch['triplets'][i]
                if len(triplets) > 0:
                    n_valid = mask[i].sum().item()
                    # Filter triplets to valid indices
                    valid_triplets = triplets[triplets.max(dim=1).values < n_valid]
                    if len(valid_triplets) > 0:
                        D_valid = D_hat[i, :n_valid, :n_valid]
                        median_dist = torch.median(D_valid[torch.triu(torch.ones_like(D_valid), diagonal=1).bool()])
                        margin = 0.05 * median_dist
                        L_ord += loss_triplet(D_valid, valid_triplets, margin)
            L_ord = L_ord / batch_size_real
            
            # 8. Total loss
            alpha, beta, gamma, eta = loss_weights['alpha'], loss_weights['beta'], loss_weights['gamma'], loss_weights['eta']
            L_total = L_score + alpha * L_gram + beta * L_heat + gamma * L_sw + eta * L_ord
            
            # 9. Backward
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            # Track losses
            epoch_losses['score'].append(L_score.item())
            epoch_losses['gram'].append(L_gram.item())
            epoch_losses['heat'].append(L_heat.item())
            epoch_losses['sw'].append(L_sw.item())
            epoch_losses['triplet'].append(L_ord.item())
            epoch_losses['total'].append(L_total.item())
        
        scheduler.step()
        
        # Logging
        if epoch % 100 == 0:
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"Epoch {epoch}/{n_epochs} | "
                  f"Total: {avg_losses['total']:.4f} | "
                  f"Score: {avg_losses['score']:.4f} | "
                  f"Gram: {avg_losses['gram']:.4f} | "
                  f"Heat: {avg_losses['heat']:.4f} | "
                  f"SW: {avg_losses['sw']:.4f} | "
                  f"Triplet: {avg_losses['triplet']:.4f}")
        
        # Save checkpoint
        if epoch % 500 == 0:
            os.makedirs(outf, exist_ok=True)
            torch.save({
                'context_encoder': context_encoder.state_dict(),
                'score_net': score_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(outf, f'stageC_checkpoint_epoch_{epoch}.pt'))
    
    # Save final
    torch.save({
        'context_encoder': context_encoder.state_dict(),
        'score_net': score_net.state_dict()
    }, os.path.join(outf, 'stageC_final.pt'))
    
    print("Stage C training complete!")

# ==============================================================================
# STAGE D: SC INFERENCE
# ==============================================================================

def sample_sc_edm(
    sc_gene_expr: torch.Tensor,
    encoder: nn.Module,
    context_encoder: SetEncoderContext,
    score_net: DiffusionScoreNet,
    n_samples: int = 1,
    n_timesteps_sample: int = 250,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    return_coords: bool = True,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Sample EDM (and optionally coordinates) for SC data.
    
    Args:
        sc_gene_expr: (n_sc, n_genes) SC gene expression
        encoder: frozen SharedEncoder
        context_encoder: trained SetEncoderContext
        score_net: trained DiffusionScoreNet
        n_samples: number of samples to generate
        n_timesteps_sample: number of reverse diffusion steps
        sigma_min, sigma_max: VE SDE noise levels
        return_coords: whether to compute coordinates via MDS
        device: torch device
        
    Returns:
        dict with 'D_edm', optionally 'coords', 'coords_canon'
    """
    encoder.eval()
    context_encoder.eval()
    score_net.eval()
    
    n_sc = sc_gene_expr.shape[0]
    sc_gene_expr = sc_gene_expr.to(device)
    
    # 1. Encode SC data
    with torch.no_grad():
        Z_sc = encoder(sc_gene_expr)  # (n_sc, h_dim)
    
    # 2. Context encoding
    Z_sc_batch = Z_sc.unsqueeze(0)  # (1, n_sc, h_dim)
    mask = torch.ones(1, n_sc, dtype=torch.bool, device=device)
    H = context_encoder(Z_sc_batch, mask)  # (1, n_sc, c_dim)
    
    # 3. Reverse SDE sampling
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), n_timesteps_sample, device=device))
    
    # Initialize V_T from noise
    D_latent = score_net.D_latent
    V_t = torch.randn(1, n_sc, D_latent, device=device) * sigma_max
    V_t = V_t - V_t.mean(dim=1, keepdim=True)  # Center
    
    # Reverse diffusion
    print(f"Running reverse diffusion for {n_timesteps_sample} steps...")
    for t_idx in reversed(range(n_timesteps_sample)):
        t_norm = torch.tensor([[t_idx / (n_timesteps_sample - 1)]], device=device)
        sigma_t = sigmas[t_idx]
        
        # Predict noise
        eps_pred = score_net(V_t, t_norm, H, mask)
        
        # DDPM-style update (simplified)
        if t_idx > 0:
            sigma_prev = sigmas[t_idx - 1]
            alpha = sigma_prev / sigma_t
            V_t = alpha * (V_t - (1 - alpha) * eps_pred) + torch.randn_like(V_t) * (sigma_prev * (1 - alpha)).sqrt()
        else:
            V_t = V_t - eps_pred * sigma_t
        
        # Recenter
        V_t = V_t - V_t.mean(dim=1, keepdim=True)
    
    # 4. Final V_0
    V_0 = V_t.squeeze(0)  # (n_sc, D_latent)
    
    # 5. Compute Gram and distances
    G = V_0 @ V_0.t()  # (n_sc, n_sc)
    diag = torch.diag(G).unsqueeze(1)
    D = torch.sqrt(torch.clamp(diag + diag.t() - 2 * G, min=0))
    
    # 6. EDM projection
    D_edm = uet.edm_project(D)
    
    result = {'D_edm': D_edm.cpu()}
    
    # 7. Optional: compute coordinates via MDS
    if return_coords:
        print("Computing coordinates via classical MDS...")
        n = D_edm.shape[0]
        J = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
        B = -0.5 * J @ (D_edm ** 2) @ J
        coords = uet.classical_mds(B, d_out=2)
        
        # Canonicalize
        coords_canon = uet.canonicalize_coords(coords)
        
        result['coords'] = coords.cpu()
        result['coords_canon'] = coords_canon.cpu()
    
    print("SC inference complete!")
    return result
