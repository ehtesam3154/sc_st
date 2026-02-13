#!/usr/bin/env python3
"""
Bug Verification & Metric Consistency Checks
=============================================
Sections:
  B1) Bug #1 verification: coordinate mismatch after subsampling
  B2) Bug #2 verification: subsampling destroys spatial resolution
  C)  Per-slide metric consistency: overlap@20, p25/median physical distance

Usage (from notebook or standalone):
    %run verify_bugs_and_metrics.py

Requires:
    - st_expr, st_coords, slide_ids, inf_expr (from notebook cell-1)
    - n_genes, ns, ST_PATHS, common_genes (from notebook cell-0/cell-1)
    - Trained encoder checkpoint at CKPT_DIR
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_utils import precompute_spatial_nce_structures, set_seed
from core_models_et_p1 import SharedEncoder


def run_all_checks(
    st_expr: torch.Tensor,
    st_coords: torch.Tensor,
    slide_ids: torch.Tensor,
    inf_expr: torch.Tensor,
    n_genes: int,
    ns: list,
    slide_names: list,
    ckpt_path: str,
    device: str = 'cuda',
    seed: int = 42,
):
    """Run all verification checks and return results dict."""

    results = {}

    # =========================================================================
    # B1) Bug #1 Verification: Coordinate Mismatch After Subsampling
    # =========================================================================
    print("=" * 70)
    print("B1) BUG #1 VERIFICATION: Coordinate Mismatch After Subsampling")
    print("=" * 70)

    n_st = st_expr.shape[0]
    n_sc = inf_expr.shape[0]

    print(f"\n  Full ST: {n_st} spots, SC: {n_sc} spots")
    print(f"  Condition for subsampling: n_st > n_sc => {n_st} > {n_sc} => {n_st > n_sc}")

    if n_st > n_sc:
        print(f"\n  --- Simulating the PRE-FIX code path ---")
        print(f"  Before subsampling:")
        print(f"    st_coords.shape = {st_coords.shape}")
        print(f"    slide_ids.shape = {slide_ids.shape}")

        # st_coords_norm was set BEFORE subsampling at line 273
        st_coords_norm_original = st_coords.clone()

        # Simulate subsampling (what line 427-434 did pre-fix)
        set_seed(seed)
        subsample_idx = torch.randperm(n_st, device=device)[:n_sc]
        st_gene_expr_sub = st_expr[subsample_idx]
        st_coords_sub = st_coords[subsample_idx]
        slide_ids_sub = slide_ids[subsample_idx]

        print(f"\n  After subsampling (n_st={n_st} -> {n_sc}):")
        print(f"    st_coords_sub.shape = {st_coords_sub.shape}")
        print(f"    slide_ids_sub.shape = {slide_ids_sub.shape}")
        print(f"    st_coords_norm_original.shape = {st_coords_norm_original.shape} (BUG: still full size!)")

        # === THE BUG: precompute called with wrong coords ===
        print(f"\n  --- THE BUG (pre-fix line ~688): ---")
        print(f"    precompute_spatial_nce_structures(")
        print(f"        st_coords=st_coords_norm_original,  # 3972 rows!")
        print(f"        slide_ids=slide_ids_sub,             # 1011 rows!")
        print(f"    )")

        # Demonstrate the mismatch with 20 random anchors
        print(f"\n  === UNIT TEST: 20 random anchors ===")
        rng = np.random.RandomState(seed)
        test_anchors = rng.choice(n_sc, 20, replace=False)

        print(f"\n  Anchor | subsamp_idx[a] | coords_buggy (orig[a]) | coords_correct (orig[sub[a]])")
        print(f"  " + "-" * 85)

        mismatches = 0
        for a in test_anchors:
            orig_global = subsample_idx[a].item()
            coord_buggy = st_coords_norm_original[a].cpu().numpy()       # Bug: reads index a from original
            coord_correct = st_coords_norm_original[orig_global].cpu().numpy()  # Correct: reads subsampled spot
            match = np.allclose(coord_buggy, coord_correct, atol=1e-5)
            if not match:
                mismatches += 1
            print(f"  {a:6d} | {orig_global:14d} | ({coord_buggy[0]:8.4f}, {coord_buggy[1]:8.4f}) | "
                  f"({coord_correct[0]:8.4f}, {coord_correct[1]:8.4f}) | {'MATCH' if match else 'MISMATCH'}")

        print(f"\n  Mismatches: {mismatches}/20")
        if mismatches > 0:
            print(f"  CONFIRMED: Bug #1 was real. Index {a} in the subsampled array refers to")
            print(f"  original spot {subsample_idx[a].item()}, but the buggy code read coordinates")
            print(f"  of original spot {a} (a completely different physical location).")
        else:
            print(f"  NOTE: No mismatches found (would occur if subsample_idx happened to be sorted)")

        # Now show precomputed pos_idx BEFORE and AFTER fix
        print(f"\n  --- pos_idx comparison: buggy vs correct ---")

        # Buggy: precompute with wrong coords
        print(f"  Computing BUGGY spatial structures (wrong coords)...")
        buggy_data = precompute_spatial_nce_structures(
            st_coords=st_coords_norm_original[:n_sc],  # first n_sc rows of original (what bug reads)
            st_gene_expr=st_gene_expr_sub,
            slide_ids=slide_ids_sub,
            k_phys=20, far_mult=4.0, n_hard=20, device=device,
        )

        # Correct: precompute with subsampled coords
        print(f"  Computing CORRECT spatial structures (subsampled coords)...")
        correct_data = precompute_spatial_nce_structures(
            st_coords=st_coords_sub,  # correct subsampled coords
            st_gene_expr=st_gene_expr_sub,
            slide_ids=slide_ids_sub,
            k_phys=20, far_mult=4.0, n_hard=20, device=device,
        )

        # Compare first 5 entries of pos_idx for test anchors
        print(f"\n  Anchor | pos_idx_buggy[:5]                  | pos_idx_correct[:5]")
        print(f"  " + "-" * 75)
        n_pos_diff = 0
        for a in test_anchors[:10]:
            buggy_pos = buggy_data['pos_idx'][a, :5].cpu().tolist()
            correct_pos = correct_data['pos_idx'][a, :5].cpu().tolist()
            same = (buggy_pos == correct_pos)
            if not same:
                n_pos_diff += 1
            print(f"  {a:6d} | {str(buggy_pos):40s} | {str(correct_pos):40s} | {'SAME' if same else 'DIFFERENT'}")

        print(f"\n  pos_idx differences: {n_pos_diff}/10 anchors have different neighbors")
        print(f"  r_pos buggy: {buggy_data['r_pos']}")
        print(f"  r_pos correct: {correct_data['r_pos']}")

        results['bug1_confirmed'] = mismatches > 0
        results['bug1_pos_idx_differ'] = n_pos_diff > 0

    else:
        print(f"  n_st ({n_st}) <= n_sc ({n_sc}): ST subsampling would NOT have occurred.")
        print(f"  Bug #1 is not applicable to this data configuration.")
        results['bug1_confirmed'] = False
        results['bug1_pos_idx_differ'] = False

    # =========================================================================
    # B2) Bug #2 Verification: Subsampling Destroys Spatial Resolution
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("B2) BUG #2 VERIFICATION: Subsampling Destroys Spatial Resolution")
    print("=" * 70)

    # Compute r_pos for FULL ST
    print(f"\n  Computing r_pos for FULL ST ({n_st} spots)...")
    full_data = precompute_spatial_nce_structures(
        st_coords=st_coords,
        st_gene_expr=st_expr,
        slide_ids=slide_ids,
        k_phys=20, far_mult=4.0, n_hard=20, device=device,
    )
    print(f"  r_pos (full, per slide): {full_data['r_pos']}")
    r_pos_full_mean = np.mean(full_data['r_pos'])
    print(f"  r_pos (full, mean): {r_pos_full_mean:.4f}")

    # Compute r_pos for SUBSAMPLED ST (if applicable)
    if n_st > n_sc:
        set_seed(seed)
        subsample_idx = torch.randperm(n_st, device=device)[:n_sc]
        st_coords_sub = st_coords[subsample_idx]
        st_expr_sub = st_expr[subsample_idx]
        slide_ids_sub = slide_ids[subsample_idx]

        print(f"\n  Computing r_pos for SUBSAMPLED ST ({n_sc} spots)...")
        sub_data = precompute_spatial_nce_structures(
            st_coords=st_coords_sub,
            st_gene_expr=st_expr_sub,
            slide_ids=slide_ids_sub,
            k_phys=20, far_mult=4.0, n_hard=20, device=device,
        )
        print(f"  r_pos (subsampled, per slide): {sub_data['r_pos']}")
        r_pos_sub_mean = np.mean(sub_data['r_pos'])
        print(f"  r_pos (subsampled, mean): {r_pos_sub_mean:.4f}")

        ratio = r_pos_sub_mean / r_pos_full_mean
        expected_ratio = np.sqrt(n_st / n_sc)
        print(f"\n  Resolution degradation:")
        print(f"    r_pos ratio (sub/full): {ratio:.2f}x")
        print(f"    Expected ratio sqrt({n_st}/{n_sc}): {expected_ratio:.2f}x")
        print(f"    {'CONFIRMED' if ratio > 1.3 else 'MARGINAL'}: subsampling {'does' if ratio > 1.3 else 'may not'} "
              f"significantly degrade spatial resolution")

        results['r_pos_full'] = full_data['r_pos']
        results['r_pos_sub'] = sub_data['r_pos']
        results['r_pos_ratio'] = ratio
    else:
        print(f"  Subsampling not applicable (n_st <= n_sc)")
        results['r_pos_full'] = full_data['r_pos']
        results['r_pos_sub'] = None
        results['r_pos_ratio'] = None

    print(f"\n  NOTE: The '0.70 overlap' reported in training was measured on the")
    print(f"  subsampled-only universe ({n_sc} spots, r_pos={results.get('r_pos_sub', 'N/A')}).")
    print(f"  It should NOT be referenced as a full-slide result.")

    # =========================================================================
    # C) Metric Consistency Check: Per-Slide overlap@20 + Physical Distances
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("C) METRIC CONSISTENCY: Per-Slide overlap@20 + Physical Distances")
    print("=" * 70)

    # Load trained model
    print(f"\n  Loading trained encoder from: {ckpt_path}")
    encoder = SharedEncoder(n_genes=n_genes, n_embedding=[512, 256, 128], dropout=0.1)
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder.to(device).eval()

    # Compute all embeddings
    with torch.no_grad():
        z_parts = []
        for i in range(0, n_st, 512):
            z_parts.append(encoder(st_expr[i:i + 512]))
        z_all = torch.cat(z_parts, dim=0)  # (n_st, 128)

    # Precomputed physical kNN
    pos_idx = full_data['pos_idx']  # (n_st, 20)

    K = 20
    unique_slides = torch.unique(slide_ids)

    print(f"\n  {'Slide':<12s} | {'n_spots':>7s} | {'overlap@20':>10s} | {'p25_phys_dist':>13s} | "
          f"{'median_phys_dist':>16s} | {'p75_phys_dist':>13s} | {'r_pos':>8s}")
    print(f"  " + "-" * 95)

    results['per_slide'] = {}
    all_overlaps = []
    all_phys_dists = []

    for si, sid in enumerate(unique_slides):
        s_mask = (slide_ids == sid)
        s_idx = torch.where(s_mask)[0]
        n_s = s_idx.shape[0]

        z_slide = z_all[s_idx]  # (n_s, 128)

        # Embedding kNN within this slide
        dists_emb = torch.cdist(z_slide, z_slide)
        dists_emb.fill_diagonal_(float('inf'))
        _, emb_knn_local = torch.topk(dists_emb, k=K, dim=1, largest=False)
        emb_knn_global = s_idx[emb_knn_local]  # (n_s, K)

        # Physical kNN (precomputed)
        phys_knn = pos_idx[s_idx]  # (n_s, K)

        # Physical distances of embedding kNN neighbors
        coords_slide = st_coords[s_idx]
        phys_dists = []
        overlaps = []

        for i in range(n_s):
            anchor_coord = st_coords[s_idx[i]]
            emb_neighbors = emb_knn_global[i]
            neighbor_coords = st_coords[emb_neighbors]
            dists = torch.norm(neighbor_coords - anchor_coord, dim=1)
            phys_dists.extend(dists.cpu().tolist())

            # Overlap
            emb_set = set(emb_knn_global[i].cpu().tolist())
            phys_set = set(phys_knn[i].cpu().tolist())
            phys_set.discard(-1)
            if len(phys_set) > 0:
                ov = len(emb_set & phys_set) / len(phys_set)
                overlaps.append(ov)

        phys_dists_arr = np.array(phys_dists)
        overlap_mean = np.mean(overlaps) if overlaps else 0.0
        p25 = np.percentile(phys_dists_arr, 25)
        p50 = np.median(phys_dists_arr)
        p75 = np.percentile(phys_dists_arr, 75)
        r_pos_slide = full_data['r_pos'][si] if si < len(full_data['r_pos']) else float('nan')

        sname = slide_names[si] if si < len(slide_names) else f"slide_{sid.item()}"
        print(f"  {sname:<12s} | {n_s:>7d} | {overlap_mean:>10.4f} | {p25:>13.4f} | "
              f"{p50:>16.4f} | {p75:>13.4f} | {r_pos_slide:>8.4f}")

        results['per_slide'][sname] = {
            'n_spots': n_s,
            'overlap_at_20': overlap_mean,
            'p25_phys_dist': p25,
            'median_phys_dist': p50,
            'p75_phys_dist': p75,
            'r_pos': r_pos_slide,
        }
        all_overlaps.extend(overlaps)
        all_phys_dists.extend(phys_dists)

    # Overall
    all_phys_dists_arr = np.array(all_phys_dists)
    overall_overlap = np.mean(all_overlaps)
    overall_p25 = np.percentile(all_phys_dists_arr, 25)
    overall_p50 = np.median(all_phys_dists_arr)
    overall_p75 = np.percentile(all_phys_dists_arr, 75)

    print(f"  " + "-" * 95)
    print(f"  {'OVERALL':<12s} | {n_st:>7d} | {overall_overlap:>10.4f} | {overall_p25:>13.4f} | "
          f"{overall_p50:>16.4f} | {overall_p75:>13.4f} | {'':>8s}")

    results['overall'] = {
        'overlap_at_20': overall_overlap,
        'p25_phys_dist': overall_p25,
        'median_phys_dist': overall_p50,
        'p75_phys_dist': overall_p75,
    }

    # Compute random baseline for context
    print(f"\n  --- Random Baseline (random encoder) ---")
    set_seed(seed)
    encoder_rand = SharedEncoder(n_genes=n_genes, n_embedding=[512, 256, 128], dropout=0.1)
    encoder_rand.to(device).eval()

    with torch.no_grad():
        z_parts_r = []
        for i in range(0, n_st, 512):
            z_parts_r.append(encoder_rand(st_expr[i:i + 512]))
        z_rand = torch.cat(z_parts_r, dim=0)

    rand_overlaps = []
    rand_phys_dists = []
    for si, sid in enumerate(unique_slides):
        s_mask = (slide_ids == sid)
        s_idx = torch.where(s_mask)[0]
        n_s = s_idx.shape[0]

        z_slide_r = z_rand[s_idx]
        dists_r = torch.cdist(z_slide_r, z_slide_r)
        dists_r.fill_diagonal_(float('inf'))
        _, rknn_local = torch.topk(dists_r, k=K, dim=1, largest=False)
        rknn_global = s_idx[rknn_local]

        phys_knn = pos_idx[s_idx]
        for i in range(n_s):
            emb_set = set(rknn_global[i].cpu().tolist())
            phys_set = set(phys_knn[i].cpu().tolist())
            phys_set.discard(-1)
            if len(phys_set) > 0:
                rand_overlaps.append(len(emb_set & phys_set) / len(phys_set))

            anchor_coord = st_coords[s_idx[i]]
            neighbor_coords = st_coords[rknn_global[i]]
            dists = torch.norm(neighbor_coords - anchor_coord, dim=1)
            rand_phys_dists.extend(dists.cpu().tolist())

    rand_phys_arr = np.array(rand_phys_dists)
    print(f"  Random overlap@20:       {np.mean(rand_overlaps):.4f}")
    print(f"  Random median phys dist: {np.median(rand_phys_arr):.4f}")
    print(f"  Random p25 phys dist:    {np.percentile(rand_phys_arr, 25):.4f}")

    results['random_baseline'] = {
        'overlap_at_20': np.mean(rand_overlaps),
        'median_phys_dist': np.median(rand_phys_arr),
        'p25_phys_dist': np.percentile(rand_phys_arr, 25),
    }

    # Domain classifier accuracy (before adapter, for reference)
    print(f"\n  --- Domain Separability (ST vs SC) ---")
    with torch.no_grad():
        z_st = z_all.cpu()
        z_sc_parts = []
        for i in range(0, inf_expr.shape[0], 512):
            z_sc_parts.append(encoder(inf_expr[i:i + 512]))
        z_sc = torch.cat(z_sc_parts, dim=0).cpu()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    Z_combined = torch.cat([z_st, z_sc], dim=0)
    Z_combined_norm = F.normalize(Z_combined, dim=1).numpy()
    y_domain = np.concatenate([np.zeros(z_st.shape[0]), np.ones(z_sc.shape[0])])

    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, Z_combined_norm, y_domain, cv=cv, scoring='balanced_accuracy')
    print(f"  Domain classifier (ST vs SC) balanced acc: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  (1.0 = perfectly separable, 0.5 = mixed)")

    results['domain_acc_before_adapter'] = cv_scores.mean()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\n  A) Summary corrections:")
    print(f"     - Architecture: genes -> 512 (LayerNorm+ReLU+Dropout) -> 256 (LayerNorm+ReLU+Dropout) -> 128 (Linear)")
    print(f"     - Gene count: {n_genes} (common genes across all liver slides)")
    print(f"     - Purpose: shared conditioning embedding for diffusion (NOT deconvolution)")

    print(f"\n  B1) Bug #1 (coord mismatch): {'CONFIRMED' if results['bug1_confirmed'] else 'NOT APPLICABLE (n_st <= n_sc)'}")
    if results['bug1_confirmed']:
        print(f"       - st_coords_norm pointed to {n_st}-row original array after subsampling")
        print(f"       - precompute received mismatched shapes: coords={n_st} vs slide_ids={n_sc}")
        print(f"       - pos_idx computed with WRONG physical coordinates")

    print(f"\n  B2) Bug #2 (resolution loss): r_pos ratio = {results.get('r_pos_ratio', 'N/A')}")
    if results.get('r_pos_ratio') and results['r_pos_ratio'] > 1.3:
        print(f"       - Full r_pos:  {results['r_pos_full']}")
        print(f"       - Sub r_pos:   {results['r_pos_sub']}")
        print(f"       - CONFIRMED: subsampling degrades spatial resolution by ~{results['r_pos_ratio']:.1f}x")

    print(f"\n  C) Metric consistency (trained model):")
    print(f"     - Overall overlap@20:       {results['overall']['overlap_at_20']:.4f}")
    print(f"     - Overall median phys dist: {results['overall']['median_phys_dist']:.4f}")
    print(f"     - Overall p25 phys dist:    {results['overall']['p25_phys_dist']:.4f}")
    print(f"     - Random overlap@20:        {results['random_baseline']['overlap_at_20']:.4f}")
    print(f"     - Random median phys dist:  {results['random_baseline']['median_phys_dist']:.4f}")
    print(f"     - Domain classifier acc:    {results['domain_acc_before_adapter']:.4f}")

    print("=" * 70)

    return results


if __name__ == '__main__':
    print("This script should be run from the notebook with pre-loaded data.")
    print("Use: results = run_all_checks(st_expr, st_coords, slide_ids, inf_expr, ...)")
