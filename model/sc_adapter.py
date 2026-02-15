#!/usr/bin/env python3
"""
SC-Only Adapter for Domain Alignment
=====================================
ST-anchored alignment: freeze the trained ST encoder trunk and train
a small SC-only adapter/projection to match SC embeddings to the
frozen ST embedding distribution.

Key properties:
  - ST overlap@20 remains UNCHANGED (encoder is frozen)
  - Domain alignment improves via CORAL/MMD on adapted SC vs frozen ST
  - No adversarial GRL (which destroyed spatial structure)

Architecture:
  Frozen encoder → z_st (128-d)  [no gradient]
  Frozen encoder → z_sc (128-d)  [no gradient] → SC adapter (128→128) → z_sc_adapted

Training:
  L = coral_weight * CORAL(z_st, z_sc_adapted)
    + mmd_weight * MMD(z_st, z_sc_adapted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Optional, Dict, List

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_utils import (
    coral_loss, mmd_rbf_loss, set_seed,
    precompute_spatial_nce_structures, compute_knn_locality_metrics,
)
from core_models_et_p1 import SharedEncoder


class SCAdapter(nn.Module):
    """
    Small SC-only adapter: maps frozen SC embeddings to adapted space
    that matches the ST embedding distribution.

    Options:
      - 'linear': single linear layer (128 -> 128)
      - 'mlp':    2-layer MLP with LayerNorm+ReLU (128 -> 128 -> 128)
      - 'affine': per-dimension scale + shift: z' = a * z + b  (cannot scramble neighborhoods)
    """
    def __init__(self, embed_dim: int = 128, mode: str = 'mlp', dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == 'linear':
            self.adapter = nn.Linear(embed_dim, embed_dim)
        elif mode == 'mlp':
            self.adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        elif mode == 'affine':
            self.scale = nn.Parameter(torch.ones(embed_dim))
            self.shift = nn.Parameter(torch.zeros(embed_dim))
        else:
            raise ValueError(f"Unknown adapter mode: {mode}")

        # Initialize close to identity (linear/mlp only; affine is already identity)
        if mode in ('linear', 'mlp'):
            self._init_near_identity()

    def _init_near_identity(self):
        """Initialize weights so adapter starts close to identity mapping."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == 'affine':
            return z * self.scale + self.shift
        return self.adapter(z)


def train_sc_adapter(
    encoder: nn.Module,
    st_gene_expr: torch.Tensor,
    sc_gene_expr: torch.Tensor,
    st_coords: torch.Tensor,
    slide_ids: torch.Tensor,
    # Adapter config
    adapter_mode: str = 'mlp',
    adapter_dropout: float = 0.1,
    init_closed_form: bool = False,
    # Training config
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    # Loss weights
    coral_weight: float = 10.0,
    mmd_weight: float = 10.0,
    identity_weight: float = 0.0,
    variance_match_weight: float = 0.0,
    # Diagnostics
    log_every: int = 50,
    device: str = 'cuda',
    seed: int = 42,
    outf: str = './sc_adapter_output',
    # Physical kNN for overlap monitoring
    phys_knn_idx: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Train SC adapter with frozen encoder.

    Args:
        encoder:       Pre-trained SharedEncoder (will be frozen)
        st_gene_expr:  (n_st, n_genes) ST expression
        sc_gene_expr:  (n_sc, n_genes) SC expression
        st_coords:     (n_st, 2) ST spatial coordinates
        slide_ids:     (n_st,) slide IDs
        adapter_mode:  'linear', 'mlp', or 'affine'
        adapter_dropout: dropout for MLP adapter
        init_closed_form: if True, initialize adapter analytically:
                          - affine: per-dim mean/var matching (diagonal only)
                          - linear: whitening+coloring (full covariance matching)
                            This is the optimal transport map between Gaussians.
        n_epochs:      training epochs
        batch_size:    mini-batch size
        lr:            learning rate for adapter
        weight_decay:  L2 regularization
        coral_weight:  weight for CORAL loss
        mmd_weight:    weight for MMD loss
        variance_match_weight: weight for per-dim variance matching regularizer
                       L_var = Σ_d (log σ_adapted_d - log σ_target_d)^2
        log_every:     log metrics every N epochs
        device:        cuda/cpu
        seed:          random seed
        outf:          output directory
        phys_knn_idx:  (n_st, k) precomputed physical kNN for overlap monitoring

    Returns:
        dict with:
          'adapter': trained SCAdapter
          'history': training history dict
          'results': final metrics
    """
    set_seed(seed)
    os.makedirs(outf, exist_ok=True)

    n_st = st_gene_expr.shape[0]
    n_sc = sc_gene_expr.shape[0]
    embed_dim = encoder.n_embedding[-1]  # 128

    print("=" * 70)
    print("SC ADAPTER TRAINING (ST-Anchored Alignment)")
    print("=" * 70)
    print(f"  Encoder: FROZEN (no gradient)")
    print(f"  Adapter: {adapter_mode} ({embed_dim} -> {embed_dim})")
    print(f"  ST: {n_st} spots, SC: {n_sc} cells")
    print(f"  Epochs: {n_epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"  CORAL weight: {coral_weight}, MMD weight: {mmd_weight}, Identity weight: {identity_weight}")
    print(f"  Variance match weight: {variance_match_weight}")
    if init_closed_form and adapter_mode == 'affine':
        print(f"  Init: CLOSED-FORM (per-dim mean/var matching)")
    if init_closed_form and adapter_mode == 'linear':
        print(f"  Init: CLOSED-FORM (whitening+coloring, full covariance matching)")
    print(f"  Output: {outf}")

    # === Freeze encoder ===
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # === Pre-compute all embeddings (frozen) ===
    print(f"\n  Pre-computing frozen embeddings...")
    with torch.no_grad():
        z_st_parts = []
        for i in range(0, n_st, 512):
            z_st_parts.append(encoder(st_gene_expr[i:i + 512]))
        z_st_frozen = torch.cat(z_st_parts, dim=0)  # (n_st, 128)

        z_sc_parts = []
        for i in range(0, n_sc, 512):
            z_sc_parts.append(encoder(sc_gene_expr[i:i + 512]))
        z_sc_frozen = torch.cat(z_sc_parts, dim=0)  # (n_sc, 128)

    print(f"  z_st_frozen: {z_st_frozen.shape}, z_sc_frozen: {z_sc_frozen.shape}")
    print(f"  z_st norms: mean={z_st_frozen.norm(dim=1).mean():.3f}, std={z_st_frozen.norm(dim=1).std():.3f}")
    print(f"  z_sc norms: mean={z_sc_frozen.norm(dim=1).mean():.3f}, std={z_sc_frozen.norm(dim=1).std():.3f}")

    # === Build adapter ===
    adapter = SCAdapter(embed_dim=embed_dim, mode=adapter_mode, dropout=adapter_dropout).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter parameters: {n_params}")

    # === Closed-form init ===
    if init_closed_form and adapter_mode == 'affine':
        with torch.no_grad():
            mu_sc = z_sc_frozen.mean(dim=0)
            mu_st = z_st_frozen.mean(dim=0)
            std_sc = z_sc_frozen.std(dim=0).clamp(min=1e-8)
            std_st = z_st_frozen.std(dim=0)
            a = std_st / std_sc
            b = mu_st - a * mu_sc
            adapter.scale.copy_(a)
            adapter.shift.copy_(b)
        print(f"  Closed-form init: scale mean={a.mean():.4f}, shift mean={b.mean():.4f}")
        print(f"  Closed-form init: scale range=[{a.min():.4f}, {a.max():.4f}]")

    if init_closed_form and adapter_mode == 'linear':
        # Whitening+coloring: the optimal transport map between two Gaussians.
        # z' = C_ST^{1/2} @ C_SC^{-1/2} @ (z - mu_SC) + mu_ST
        # This matches the FULL covariance matrix (not just per-dim variance),
        # rotating SC's isotropic variance into ST's anisotropic PCA directions.
        with torch.no_grad():
            mu_sc = z_sc_frozen.mean(dim=0)  # (128,)
            mu_st = z_st_frozen.mean(dim=0)  # (128,)

            C_sc = torch.cov(z_sc_frozen.T)  # (128, 128)
            C_st = torch.cov(z_st_frozen.T)  # (128, 128)

            # Eigendecomposition for matrix square roots
            # C_SC^{-1/2} = U_SC @ diag(1/sqrt(S_SC)) @ U_SC^T
            S_sc, U_sc = torch.linalg.eigh(C_sc)
            S_st, U_st = torch.linalg.eigh(C_st)

            # Clamp small eigenvalues for numerical stability
            eps = 1e-6
            S_sc = S_sc.clamp(min=eps)
            S_st = S_st.clamp(min=eps)

            # C_SC^{-1/2}
            whiten = U_sc @ torch.diag(1.0 / S_sc.sqrt()) @ U_sc.T
            # C_ST^{1/2}
            color = U_st @ torch.diag(S_st.sqrt()) @ U_st.T

            # W = color @ whiten, b = mu_ST - W @ mu_SC
            W = color @ whiten   # (128, 128)
            b = mu_st - W @ mu_sc  # (128,)

            adapter.adapter.weight.copy_(W)
            adapter.adapter.bias.copy_(b)

            # Diagnostics: condition number tells us if transform is well-behaved
            cond = (S_sc.max() / S_sc.min()).item()
            print(f"  Closed-form init (whitening+coloring):")
            print(f"    SC eigenvalues: min={S_sc.min():.6f}, max={S_sc.max():.4f}")
            print(f"    ST eigenvalues: min={S_st.min():.6f}, max={S_st.max():.4f}")
            print(f"    SC covariance condition number: {cond:.1f}")
            print(f"    W matrix: norm={W.norm():.4f}, diag_mean={W.diag().mean():.4f}")

    # === Optimizer ===
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # === Pre-adapter metrics ===
    print(f"\n  --- Pre-Adapter Metrics ---")
    _print_domain_metrics(z_st_frozen, z_sc_frozen, st_coords, slide_ids, phys_knn_idx, encoder, st_gene_expr, prefix="  Before")

    # === Training history ===
    history = {
        'epoch': [], 'loss_total': [], 'loss_coral': [], 'loss_mmd': [],
        'loss_identity': [], 'loss_variance': [],
        'st_overlap_at_20': [], 'domain_centroid_dist': [],
        'lr': [],
    }

    # === Training loop ===
    print(f"\n  --- Training ---")
    best_loss = float('inf')
    best_adapter_state = None

    for epoch in range(n_epochs):
        adapter.train()

        # Sample balanced batches
        st_idx = torch.randperm(n_st, device=device)[:batch_size]
        sc_idx = torch.randperm(n_sc, device=device)[:batch_size]

        z_st_batch = z_st_frozen[st_idx]     # frozen, no grad
        z_sc_batch = z_sc_frozen[sc_idx]     # frozen, no grad

        # Adapt SC
        z_sc_adapted = adapter(z_sc_batch)   # grad flows through adapter only

        # === Losses ===
        loss = torch.tensor(0.0, device=device)

        # CORAL
        if coral_weight > 0:
            l_coral = coral_loss(z_st_batch, z_sc_adapted)
            loss = loss + coral_weight * l_coral
        else:
            l_coral = torch.tensor(0.0)

        # MMD
        if mmd_weight > 0:
            l_mmd = mmd_rbf_loss(z_st_batch, z_sc_adapted)
            loss = loss + mmd_weight * l_mmd
        else:
            l_mmd = torch.tensor(0.0)

        # Identity regularizer: g(z_st) ≈ z_st (prevents topology warping)
        if identity_weight > 0:
            z_st_through = adapter(z_st_batch)
            l_id = F.mse_loss(z_st_through, z_st_batch)
            loss = loss + identity_weight * l_id
        else:
            l_id = torch.tensor(0.0)

        # Variance matching: per-dim log-variance penalty
        # L_var = mean_d (log σ_adapted_d - log σ_ST_d)^2
        if variance_match_weight > 0:
            sc_var = z_sc_adapted.var(dim=0).clamp(min=1e-8)
            st_var = z_st_batch.var(dim=0).clamp(min=1e-8)
            l_var = (torch.log(sc_var) - torch.log(st_var)).pow(2).mean()
            loss = loss + variance_match_weight * l_var
        else:
            l_var = torch.tensor(0.0)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # === Logging ===
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            adapter.eval()
            with torch.no_grad():
                z_sc_all_adapted = adapter(z_sc_frozen)

                # Centroid distance
                mu_st = z_st_frozen.mean(dim=0)
                mu_sc = z_sc_all_adapted.mean(dim=0)
                centroid_dist = (mu_st - mu_sc).norm().item()

                # Full CORAL/MMD on all data
                full_coral = coral_loss(z_st_frozen, z_sc_all_adapted).item()
                full_mmd = mmd_rbf_loss(
                    z_st_frozen[torch.randperm(n_st)[:min(1000, n_st)]],
                    z_sc_all_adapted[torch.randperm(n_sc)[:min(1000, n_sc)]]
                ).item()

                # ST overlap@20 (should remain unchanged since encoder is frozen)
                st_ov = 0.0
                if phys_knn_idx is not None:
                    # ST embeddings are frozen, so overlap@20 is constant
                    # But we log it to confirm
                    ov_metrics = compute_knn_locality_metrics(
                        model=encoder, st_gene_expr=st_gene_expr,
                        st_coords=st_coords, slide_ids=slide_ids,
                        phys_knn_idx=phys_knn_idx, k=20, n_sample=300,
                    )
                    st_ov = ov_metrics['overlap_mean']

            current_lr = scheduler.get_last_lr()[0]
            id_str = f" L_id={l_id.item():.6f}" if identity_weight > 0 else ""
            var_str = f" L_var={l_var.item():.4f}" if variance_match_weight > 0 else ""

            # Per-dim spread ratio: adapted SC std vs ST std
            sc_ad_std = z_sc_all_adapted.std(dim=0).mean().item()
            st_std = z_st_frozen.std(dim=0).mean().item()
            spread_ratio = sc_ad_std / max(st_std, 1e-8)

            print(f"  e{epoch:4d} | loss={loss.item():.4f} coral={l_coral.item():.4f} mmd={l_mmd.item():.4f}{id_str}{var_str} | "
                  f"centroid_dist={centroid_dist:.4f} spread_ratio={spread_ratio:.3f} | st_overlap@20={st_ov:.4f} | lr={current_lr:.2e}")

            history['epoch'].append(epoch)
            history['loss_total'].append(loss.item())
            history['loss_coral'].append(full_coral)
            history['loss_mmd'].append(full_mmd)
            history['loss_identity'].append(l_id.item() if identity_weight > 0 else 0.0)
            history['loss_variance'].append(l_var.item() if variance_match_weight > 0 else 0.0)
            history['st_overlap_at_20'].append(st_ov)
            history['domain_centroid_dist'].append(centroid_dist)
            history['lr'].append(current_lr)

            # Best checkpoint (lowest CORAL+MMD)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_adapter_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}

    # === Restore best adapter ===
    if best_adapter_state is not None:
        adapter.load_state_dict(best_adapter_state)
        adapter.to(device)
    adapter.eval()

    # === Post-adapter metrics ===
    print(f"\n  --- Post-Adapter Metrics ---")
    with torch.no_grad():
        z_sc_final = adapter(z_sc_frozen)

    _print_domain_metrics(z_st_frozen, z_sc_final, st_coords, slide_ids, phys_knn_idx, encoder, st_gene_expr, prefix="  After")

    # === Domain classifier accuracy ===
    print(f"\n  --- Domain Classifier (ST vs adapted SC) ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    Z_before = torch.cat([z_st_frozen.cpu(), z_sc_frozen.cpu()], dim=0)
    Z_after = torch.cat([z_st_frozen.cpu(), z_sc_final.cpu()], dim=0)
    y_domain = np.concatenate([np.zeros(n_st), np.ones(n_sc)])

    Z_before_norm = F.normalize(Z_before, dim=1).numpy()
    Z_after_norm = F.normalize(Z_after, dim=1).numpy()

    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_before = cross_val_score(clf, Z_before_norm, y_domain, cv=cv, scoring='balanced_accuracy').mean()
    acc_after = cross_val_score(clf, Z_after_norm, y_domain, cv=cv, scoring='balanced_accuracy').mean()

    print(f"  Domain acc BEFORE adapter: {acc_before:.4f}")
    print(f"  Domain acc AFTER adapter:  {acc_after:.4f}")
    print(f"  Change: {acc_before:.4f} -> {acc_after:.4f} ({'improved' if acc_after < acc_before else 'unchanged/worse'})")

    # === Save ===
    torch.save(adapter.state_dict(), os.path.join(outf, 'sc_adapter.pt'))
    torch.save(encoder.state_dict(), os.path.join(outf, 'encoder_frozen.pt'))

    with open(os.path.join(outf, 'adapter_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Saved adapter to: {outf}/sc_adapter.pt")
    print(f"  Saved history to: {outf}/adapter_history.json")

    # === Final results ===
    final_results = {
        'st_overlap_at_20': history['st_overlap_at_20'][-1] if history['st_overlap_at_20'] else 0.0,
        'domain_acc_before': acc_before,
        'domain_acc_after': acc_after,
        'final_coral': history['loss_coral'][-1] if history['loss_coral'] else 0.0,
        'final_mmd': history['loss_mmd'][-1] if history['loss_mmd'] else 0.0,
        'final_centroid_dist': history['domain_centroid_dist'][-1] if history['domain_centroid_dist'] else 0.0,
    }

    print(f"\n" + "=" * 70)
    print(f"SC ADAPTER TRAINING COMPLETE")
    print(f"=" * 70)
    print(f"  ST overlap@20:          {final_results['st_overlap_at_20']:.4f} (should be ~unchanged)")
    print(f"  Domain acc (before):    {final_results['domain_acc_before']:.4f}")
    print(f"  Domain acc (after):     {final_results['domain_acc_after']:.4f}")
    print(f"  CORAL loss (final):     {final_results['final_coral']:.4f}")
    print(f"  MMD loss (final):       {final_results['final_mmd']:.4f}")
    print(f"  Centroid dist (final):  {final_results['final_centroid_dist']:.4f}")
    print(f"=" * 70)

    return {
        'adapter': adapter,
        'history': history,
        'results': final_results,
        'z_st_frozen': z_st_frozen,
        'z_sc_adapted': z_sc_final,
    }


def _print_domain_metrics(z_st, z_sc, st_coords, slide_ids, phys_knn_idx, encoder, st_gene_expr, prefix=""):
    """Print domain alignment metrics."""
    mu_st = z_st.mean(dim=0)
    mu_sc = z_sc.mean(dim=0)
    centroid_dist = (mu_st - mu_sc).norm().item()

    coral = coral_loss(z_st, z_sc).item()

    n_sub = min(1000, z_st.shape[0], z_sc.shape[0])
    mmd = mmd_rbf_loss(
        z_st[torch.randperm(z_st.shape[0])[:n_sub]],
        z_sc[torch.randperm(z_sc.shape[0])[:n_sub]]
    ).item()

    print(f"{prefix}: centroid_dist={centroid_dist:.4f}, CORAL={coral:.4f}, MMD={mmd:.4f}")
    print(f"{prefix}: z_st norms: mean={z_st.norm(dim=1).mean():.3f}, "
          f"z_sc norms: mean={z_sc.norm(dim=1).mean():.3f}")


def get_adapted_embeddings(
    encoder: nn.Module,
    adapter: nn.Module,
    gene_expr: torch.Tensor,
    is_sc: bool = True,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Get embeddings for new data, applying adapter for SC data.

    Args:
        encoder:   Frozen SharedEncoder
        adapter:   Trained SCAdapter (only applied if is_sc=True)
        gene_expr: (n, n_genes) expression data
        is_sc:     Whether this is SC data (apply adapter) or ST data (skip adapter)
        device:    cuda/cpu

    Returns:
        z: (n, 128) embeddings
    """
    encoder.eval()
    adapter.eval()

    with torch.no_grad():
        z_parts = []
        for i in range(0, gene_expr.shape[0], 512):
            z_batch = encoder(gene_expr[i:i + 512])
            if is_sc:
                z_batch = adapter(z_batch)
            z_parts.append(z_batch)
        return torch.cat(z_parts, dim=0)