from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

# Detailed sanity checks for ST/SC subset creation.
# Safe to import anywhere; prints only from rank-0 if you pass is_global_zero=True.

import torch, numpy as np, math, json, os, textwrap, time
from typing import Dict, Any, List, Optional

def _safe(name, fn, default=None):
    try:
        return fn()
    except Exception as e:
        return f"{name}: ERROR: {e}" if default is None else default

def _edm_psd_violations(D, max_m=128):
    """Return mean negative eigenvalue over random subminisets (≈ EDM cone check)."""
    n = D.size(0)
    if n < 4:
        return 0.0
    m = min(max_m, n)
    idx = torch.randperm(n, device=D.device)[:m]
    Dm = D[idx][:, idx]
    J = torch.eye(m, device=D.device) - torch.ones(m, m, device=D.device)/m
    B = -0.5 * (J @ (Dm**2) @ J)
    eigs = torch.linalg.eigvalsh(B)
    return float(torch.relu(-eigs).mean().item())

# ------------------------------- utilities -----------------------------------

def _fmt_head(title: str) -> str:
    bar = "─" * max(10, len(title) + 2)
    return f"\n┌{bar}┐\n│ {title} │\n└{bar}┘"

def _summ(vs: List[float]) -> Dict[str, float]:
    if not vs: 
        return dict(count=0, min=float("nan"), max=float("nan"), mean=float("nan"))
    arr = np.array(vs, dtype=float)
    return dict(count=len(vs), min=float(arr.min()), p25=float(np.percentile(arr,25)),
                median=float(np.percentile(arr,50)), p75=float(np.percentile(arr,75)),
                max=float(arr.max()), mean=float(arr.mean()), std=float(arr.std()+1e-12))

def _safe_q(vals: torch.Tensor, q: float, cap: int = 250_000) -> float:
    """Quantile on a sample; avoids allocating huge upper-tri vectors."""
    if vals.numel() == 0:
        return float("nan")
    if vals.numel() > cap:
        idx = torch.randint(0, vals.numel(), (cap,), device=vals.device)
        vals = vals.view(-1)[idx]
    return float(torch.quantile(vals, q).item())

def _edm_psd_hinge(D: torch.Tensor, max_m: int = 96) -> float:
    """Mean negative eigenvalue of -1/2 J D^2 J on a random sub-miniset."""
    n = D.shape[0]
    if n < 4: 
        return 0.0
    m = min(max_m, n)
    sel = torch.randperm(n, device=D.device)[:m]
    Dm = D.index_select(0, sel).index_select(1, sel)
    J = torch.eye(m, device=D.device) - torch.ones(m, m, device=D.device)/m
    B = -0.5 * (J @ (Dm**2) @ J)
    eigs = torch.linalg.eigvalsh(B)
    return float(torch.relu(-eigs).mean().item())

def _upper_triangle_sample(D: torch.Tensor, cap_pairs: int = 1_000_000) -> torch.Tensor:
    """Return ~cap_pairs randomly sampled upper-triangular distances."""
    n = D.shape[0]
    if n < 2: 
        return torch.empty(0, device=D.device)
    m = int(cap_pairs * 1.3)
    i = torch.randint(0, n, (m,), device=D.device)
    j = torch.randint(0, n, (m,), device=D.device)
    keep = i < j
    i, j = i[keep], j[keep]
    if i.numel() > cap_pairs:
        i, j = i[:cap_pairs], j[:cap_pairs]
    return D[i, j]

def _rank0_print(is_global_zero: bool, *args, **kwargs):
    if is_global_zero:
        print(*args, **kwargs)

# ------------------------------ ST checks ------------------------------------

@torch.no_grad()
def check_st_batch(
    batch: Dict[str, Any], 
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Expects: Z_set (B,N,h), mask (B,N), n (B,), G_target (B,N,N), optional L_info list
    """
    rep: Dict[str, Any] = {}
    Z, M, n_list = batch["Z_set"], batch["mask"], batch["n"]
    B, N, h = Z.shape
    rep["B"] = int(B); rep["N_max"] = int(N); rep["h_dim"] = int(h)

    # mask vs n
    valid_counts = [int(M[i].sum().item()) for i in range(B)]
    rep["valid_counts"] = valid_counts
    rep["n_equals_masksum_all"] = bool(all(int(n_list[i].item()) == valid_counts[i] for i in range(B)))

    # Gram target integrity
    Gt = batch.get("G_target", None)
    if Gt is not None:
        Gt = Gt.to(device).float()
        sym_err = (Gt - Gt.transpose(1,2)).abs().amax(dim=(1,2))
        diag = Gt.diagonal(dim1=1, dim2=2)

        # rep["gram_sym_maxerr"] = float(sym_err.max().item())
        # rep["gram_diag_min"] = float(diag.amin().item())

        # NEW (CORRECT - only valid regions):
        rep["gram_sym_maxerr"] = float(sym_err.max().item())

        # Only check diagonal in valid regions
        diag_mins = []
        for i in range(B):
            n_i = int(n_list[i].item())
            diag_i = torch.diag(Gt[i, :n_i, :n_i])
            diag_mins.append(diag_i.min().item())
        rep["gram_diag_min"] = float(min(diag_mins))

        # One example: factorize → recompute distances → EDM hinge
        i = 0
        n_i = int(n_list[i].item())
        
        # Gi = Gt[i, :n_i, :n_i]
        # # numerically robust factorization
        # Gi = Gi + 1e-6 * torch.eye(n_i, device=device)
        # try:
        #     Vi = torch.linalg.cholesky(Gi, upper=False)
        # except RuntimeError:
        #     # fallback to eigh if cholesky fails
        #     w, U = torch.linalg.eigh(Gi)
        #     w = torch.clamp(w, min=1e-12).sqrt()
        #     Vi = (U * w.unsqueeze(0)).T
        # Di = torch.cdist(Vi, Vi)

        Gi = Gt[i, :n_i, :n_i]
        # Compute distances from Gram using CORRECT formula: d_ij^2 = G_ii + G_jj - 2*G_ij
        G_diag = torch.diag(Gi).unsqueeze(0)  # (1, n_i)
        D_squared = G_diag.T + G_diag - 2 * Gi  # (n_i, n_i)
        Di = torch.sqrt(torch.clamp(D_squared, min=0))

        rep["edm_psd_hinge_mean"] = _edm_psd_hinge(Di)

        # Distance scale (approx percentiles on sample)
        d_samp = _upper_triangle_sample(Di)
        rep["D_p50_p90_p95"] = dict(p50=_safe_q(d_samp,0.50), p90=_safe_q(d_samp,0.90), p95=_safe_q(d_samp,0.95))
        rep["edm_psd_hinge_mean"] = _edm_psd_hinge(Di)



    # Laplacian (if provided in L_info)
    Linfo = batch.get("L_info", None)
    if isinstance(Linfo, list) and Linfo and ("L" in Linfo[0]):
        L = Linfo[0]["L"]
        L = L.to_dense() if L.layout != torch.strided else L
        L = L.to(device).float()
        rep["L_sym_err_max"] = float((L - L.T).abs().max().item())
        rep["L_row_sum_max_abs"] = float(L.sum(dim=1).abs().max().item())

    return rep

# ------------------------------ SC checks ------------------------------------

@torch.no_grad()
def check_sc_batch(
    batch: Dict[str, Any], 
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Expects: Z_set (B,N,h), mask (B,N), n (B,), sc_global_indices (B,N) >=0 on valid,
             pair_idxA (P,), pair_idxB (P,), shared_A_idx/shared_B_idx (P,K), optional is_landmark (B,N)
    """
    rep: Dict[str, Any] = {}
    Z, M, n_list = batch["Z_set"], batch["mask"], batch["n"]
    B, N, h = Z.shape
    rep["B"] = int(B); rep["N_max"] = int(N); rep["h_dim"] = int(h)

    valid_counts = [int(M[i].sum().item()) for i in range(B)]
    rep["valid_counts"] = valid_counts
    rep["n_equals_masksum_all"] = bool(all(int(n_list[i].item()) == valid_counts[i] for i in range(B)))

    GIDX = batch.get("sc_global_indices", None)
    rep["has_global_idx"] = bool(GIDX is not None)
    if GIDX is not None:
        g_nonneg = (GIDX[M].min().item() >= 0)
        rep["global_idx_nonneg"] = bool(g_nonneg)

        # no duplicates within a set (optional)
        dup_fracs = []
        for i in range(B):
            n_i = int(n_list[i].item())
            gi = GIDX[i, :n_i].view(-1).cpu().numpy()
            dup = 1.0 - (len(np.unique(gi)) / max(1,len(gi)))
            dup_fracs.append(dup)
        rep["dup_frac_per_set"] = _summ(dup_fracs)

    # overlap plumbing
    pairA, pairB = batch.get("pair_idxA"), batch.get("pair_idxB")
    SA, SB = batch.get("shared_A_idx"), batch.get("shared_B_idx")
    if (pairA is not None) and (pairB is not None) and (SA is not None) and (SB is not None):
        P, Kmax = SA.shape
        rep["overlap_P"] = int(P); rep["overlap_Kmax"] = int(Kmax)

        Ks = []
        z_consistency = []
        zero_pairs = 0
        for p in range(P):
            i, j = int(pairA[p]), int(pairB[p])
            n_i, n_j = int(n_list[i].item()), int(n_list[j].item())
            gi = GIDX[i, :n_i]; gj = GIDX[j, :n_j]

            ai, bi = SA[p].clamp_min(0), SB[p].clamp_min(0)
            mA, mB = SA[p] >= 0, SB[p] >= 0
            gi_sel = gi[ai[mA]]
            gj_sel = gj[bi[mB]]

            # intersection size
            common = torch.tensor(np.intersect1d(gi_sel.cpu().numpy(), gj_sel.cpu().numpy()), device=device)
            Ks.append(int(common.numel()))
            if common.numel() == 0:
                zero_pairs += 1
                continue

            # consistency in Z-set for shared cells (should be near-identical)
            # find local positions of 'common' within the selected arrays:
            # build map gid -> first position
            def idx_map(gids: torch.Tensor) -> Dict[int,int]:
                m = {}
                for pos, g in enumerate(gids.view(-1).tolist()):
                    if g not in m:
                        m[g] = pos
                return m
            mapA, mapB = idx_map(gi_sel), idx_map(gj_sel)

            posA = torch.tensor([mapA[int(g.item())] for g in common], device=device)
            posB = torch.tensor([mapB[int(g.item())] for g in common], device=device)

            # Zi = Z[i][mA][posA]
            # Zj = Z[j][mB][posB]

            Zi = Z[i][ai[mA]][posA]
            Zj = Z[j][bi[mB]][posB]
            z_consistency.append(float((Zi - Zj).norm(dim=1).mean().item()))

        rep["overlap_K_stats"] = _summ(Ks)
        rep["overlap_zero_frac"] = (zero_pairs / max(1,P))
        rep["overlap_Z_consistency_mean||Δ||"] = float(np.mean(z_consistency)) if z_consistency else float("nan")

    # landmarks (if present)
    is_lmk = batch.get("is_landmark", None)
    if is_lmk is not None:
        cnts = ((is_lmk & M).sum(dim=1)).float().cpu().tolist()
        rep["landmarks_per_set"] = _summ(cnts)

    return rep

# ----------------------------- driver routines --------------------------------

@torch.no_grad()
def sample_dataloader_and_report(
    st_loader = None,
    sc_loader = None,
    batches: int = 3,
    device: str = "cuda",
    is_global_zero: bool = True,
    save_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pull up to `batches` from each loader, compute diagnostics, print a rich report,
    and (optionally) save JSON.
    """
    t0 = time.time()
    out: Dict[str, Any] = dict(st=[], sc=[])
    if st_loader is not None:
        it = iter(st_loader)
        if st_loader is not None:
            it = iter(st_loader)
            for b in range(batches):
                batch = next(it, None)
                if batch is None:
                    break

                # Paired ST batches: unwrap view1 for standard checks
                if isinstance(batch, dict) and batch.get("is_pair_batch", False):
                    batch = batch["view1"]

                out["st"].append(check_st_batch(batch, device=device))
    if sc_loader is not None:
        it = iter(sc_loader)
        for b in range(batches):
            batch = next(it, None)
            if batch is None: break
            out["sc"].append(check_sc_batch(batch, device=device))
    t1 = time.time()

    # console pretty-print
    _rank0_print(is_global_zero, _fmt_head("ST MINI-SETS (structure targets)"))
    if not out["st"]:
        _rank0_print(is_global_zero, "  (no ST batches sampled)")
    for i, rep in enumerate(out["st"]):
        _rank0_print(is_global_zero, f"\n[ST batch {i}]  B={rep['B']}  N_max={rep['N_max']}  h={rep['h_dim']}")
        _rank0_print(is_global_zero, f"  valid_counts: {rep['valid_counts']}")
        _rank0_print(is_global_zero, f"  n == mask.sum for all sets?  {rep['n_equals_masksum_all']}")
        if "gram_sym_maxerr" in rep:
            _rank0_print(is_global_zero, f"  Gram symmetry max |Δ|: {rep['gram_sym_maxerr']:.3e}")
            _rank0_print(is_global_zero, f"  Gram diag min:     {rep['gram_diag_min']:.6f}")
            p = rep.get("D_p50_p90_p95", {})
            _rank0_print(is_global_zero, f"  Dist pctls (recon from Gram): p50={p.get('p50',float('nan')):.3f}, "
                                         f"p90={p.get('p90',float('nan')):.3f}, p95={p.get('p95',float('nan')):.3f}")
            _rank0_print(is_global_zero, f"  EDM PSD hinge (↓ better): {rep['edm_psd_hinge_mean']:.4f}")
        if "L_sym_err_max" in rep:
            _rank0_print(is_global_zero, f"  Laplacian symmetry max |Δ|: {rep['L_sym_err_max']:.3e}")
            _rank0_print(is_global_zero, f"  Laplacian row-sum max |Σ|:  {rep['L_row_sum_max_abs']:.3e}")

    _rank0_print(is_global_zero, _fmt_head("SC MINI-SETS (single-cell subsets)"))
    if not out["sc"]:
        _rank0_print(is_global_zero, "  (no SC batches sampled)")
    for i, rep in enumerate(out["sc"]):
        _rank0_print(is_global_zero, f"\n[SC batch {i}]  B={rep['B']}  N_max={rep['N_max']}  h={rep['h_dim']}")
        _rank0_print(is_global_zero, f"  valid_counts: {rep['valid_counts']}")
        _rank0_print(is_global_zero, f"  n == mask.sum for all sets?  {rep['n_equals_masksum_all']}")
        _rank0_print(is_global_zero, f"  has_global_idx: {rep.get('has_global_idx', False)}")
        if rep.get("has_global_idx", False):
            _rank0_print(is_global_zero, f"  global_idx_nonneg: {rep.get('global_idx_nonneg', False)}")
            dup = rep.get("dup_frac_per_set", {})
            _rank0_print(is_global_zero, f"  duplicate fraction per set (should be 0): "
                                         f"min={dup.get('min',float('nan')):.3f}, "
                                         f"p50={dup.get('median',float('nan')):.3f}, "
                                         f"max={dup.get('max',float('nan')):.3f}")
        if "overlap_P" in rep:
            Kstats = rep["overlap_K_stats"]
            _rank0_print(is_global_zero, f"  overlap pairs P={rep['overlap_P']}  Kmax={rep['overlap_Kmax']}")
            _rank0_print(is_global_zero, f"  overlap K (shared cells): "
                                         f"min={Kstats.get('min',float('nan'))}, "
                                         f"p25={Kstats.get('p25',float('nan')):.1f}, "
                                         f"p50={Kstats.get('median',float('nan')):.1f}, "
                                         f"p75={Kstats.get('p75',float('nan')):.1f}, "
                                         f"max={Kstats.get('max',float('nan'))}")
            _rank0_print(is_global_zero, f"  overlap zero-fraction: {rep['overlap_zero_frac']:.3f}")
            _rank0_print(is_global_zero, f"  Z consistency on shared cells (||Δ||, ↓ better): "
                                         f"{rep['overlap_Z_consistency_mean||Δ||']:.6f}")
        if "landmarks_per_set" in rep:
            Lm = rep["landmarks_per_set"]
            _rank0_print(is_global_zero, f"  landmarks per set: "
                                         f"min={Lm.get('min',float('nan')):.0f}, "
                                         f"median={Lm.get('median',float('nan')):.0f}, "
                                         f"max={Lm.get('max',float('nan')):.0f}")
    _rank0_print(is_global_zero, f"\n[debug] minisets check finished in {t1 - t0:.2f}s")

    if save_json_path:
        try:
            os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        except Exception:
            pass
        with open(save_json_path, "w") as f:
            json.dump(out, f, indent=2)
        _rank0_print(is_global_zero, f"[debug] wrote JSON: {save_json_path}")

    return out
