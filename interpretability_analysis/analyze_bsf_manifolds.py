import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root and interpretability_analysis to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.sam3_rcnn_v2 import build_sam3_fasterrcnn
from train_saes import get_data_splits, ImagePathDataset
from overcomplete import TopKSAE
from custom_saes import (
    RATopKSAE,
    OMPSAE,
    MpSAE as MPSAE,
    BSF,
    VanillaBSF,
    GrassmannianBSF,
    plot_block_concepts_manifold
)


def safe_corr(x, y):
    """Computes Pearson correlation safely with zero-variance protection."""
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def find_annotations_file(annotations_file, dataset_dir):
    """Auto-searches for RIVA annotations (CSV/JSON) if not explicitly provided."""
    if annotations_file and os.path.exists(annotations_file):
        return annotations_file

    search_dirs = []
    if dataset_dir and os.path.isdir(dataset_dir):
        p = Path(dataset_dir)
        search_dirs.extend([
            p,
            p / "annotations",
            p / "annotations" / "annotations",
            p.parent,
            p.parent / "annotations",
            p.parent / "annotations" / "annotations",
        ])

    search_dirs.extend([
        Path("./RIVA/annotations/annotations"),
        Path("./RIVA/annotations"),
        Path("./annotations"),
        Path("./data/annotations"),
        Path("./data"),
    ])

    for d in search_dirs:
        if d.is_dir():
            for name in ["val.csv", "train.csv", "annotations.csv", "annotations.json"]:
                f = d / name
                if f.is_file():
                    return str(f)
            for f in list(d.glob("*.csv")) + list(d.glob("*.json")):
                if "sample" not in f.name.lower():
                    return str(f)
    return None


def extract_features_and_proxies(sam3_model, sae_model, dataloader, device):
    """
    Extracts SAM3 FPN features, SAE block coordinates / norms, foreground masks,
    and continuous cytological proxies for every token patch across the dataset.
    """
    sam3_model.eval()
    sae_model.eval()

    is_bsf = hasattr(sae_model, "encode_blocks") and callable(sae_model.encode_blocks)

    activations = {}
    def hook_fn(module, input, output):
        activations["1"] = output["1"].detach()

    handle = sam3_model.backbone.register_forward_hook(hook_fn)

    # Sobel kernels for spatial gradient magnitude
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3) / 8.0

    all_images = []
    all_z_norm = []
    all_z_blocks = []
    all_valid_mask = []
    all_od = []
    all_hue = []
    all_grad = []
    all_img_indices = []
    all_cy = []
    all_cx = []
    all_r_grid = []
    all_c_grid = []

    H_feat, W_feat = None, None
    img_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features & proxies"):
            batch_dev = batch.to(device)
            B = batch_dev.shape[0]
            all_images.append(batch.cpu())

            # 1. Forward backbone to trigger hook
            _ = sam3_model.backbone(batch_dev)
            feat = activations["1"]
            B, C, H_feat, W_feat = feat.shape
            P = H_feat * W_feat

            feat_flat = feat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

            # 2. SAE Forward
            if is_bsf:
                z_blk = sae_model.encode_blocks(feat_flat)  # (B*P, n_groups, group_size)
                z_norm = z_blk.norm(dim=-1)                 # (B*P, n_groups)
                all_z_blocks.append(z_blk.cpu())
            else:
                _, z, _ = sae_model(feat_flat)
                z_norm = z
            all_z_norm.append(z_norm.cpu())

            # 3. Define valid non-padded slide tokens (valid_mask):
            # Token mean_rgb > 0.08 AND token is not on the 1-token outer perimeter (1 <= r < H-1, 1 <= c < W-1)
            mean_rgb = batch_dev.mean(dim=1, keepdim=True)
            mean_rgb_pooled = F.adaptive_avg_pool2d(mean_rgb, (H_feat, W_feat)).squeeze(1).view(-1)

            r_coords = torch.arange(H_feat, device=device)
            c_coords = torch.arange(W_feat, device=device)
            grid_r, grid_c = torch.meshgrid(r_coords, c_coords, indexing='ij')
            inner_mask = (grid_r >= 1) & (grid_r < H_feat - 1) & (grid_c >= 1) & (grid_c < W_feat - 1)
            inner_mask_batch = inner_mask.reshape(-1).repeat(B)

            valid_mask = (mean_rgb_pooled > 0.08) & inner_mask_batch
            all_valid_mask.append(valid_mask.cpu())

            # 4. Proxy 1: Optical density (mean inverse grayscale)
            gray = 0.2989 * batch_dev[:, 0:1] + 0.5870 * batch_dev[:, 1:2] + 0.1140 * batch_dev[:, 2:3]
            inv_gray = 1.0 - gray
            od_pooled = F.adaptive_avg_pool2d(inv_gray, (H_feat, W_feat)).squeeze(1).view(-1)
            all_od.append(od_pooled.cpu())

            # 5. Proxy 2: Color hue ratio (R - B) / (R + B + 1e-6)
            r_ch = batch_dev[:, 0:1]
            b_ch = batch_dev[:, 2:3]
            hue = (r_ch - b_ch) / (r_ch + b_ch + 1e-6)
            hue_pooled = F.adaptive_avg_pool2d(hue, (H_feat, W_feat)).squeeze(1).view(-1)
            all_hue.append(hue_pooled.cpu())

            # 6. Proxy 3: Gradient energy (mean Sobel magnitude)
            gx = F.conv2d(gray, sobel_x, padding=1)
            gy = F.conv2d(gray, sobel_y, padding=1)
            grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            grad_pooled = F.adaptive_avg_pool2d(grad_mag, (H_feat, W_feat)).squeeze(1).view(-1)
            all_grad.append(grad_pooled.cpu())

            # 7. Spatial pixel centers on 1008x1008 image & grid coordinates
            stride = 1008.0 / H_feat
            y_coords = ((torch.arange(H_feat, device=device) + 0.5) * stride).long()
            x_coords = ((torch.arange(W_feat, device=device) + 0.5) * stride).long()
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_y_flat = grid_y.reshape(-1)
            grid_x_flat = grid_x.reshape(-1)

            batch_img_idx = (torch.arange(B, device=device).unsqueeze(1).repeat(1, P).view(-1) + img_counter)
            batch_cy = grid_y_flat.repeat(B)
            batch_cx = grid_x_flat.repeat(B)

            all_img_indices.append(batch_img_idx.cpu())
            all_cy.append(batch_cy.cpu())
            all_cx.append(batch_cx.cpu())
            all_r_grid.append(grid_r.reshape(-1).repeat(B).cpu())
            all_c_grid.append(grid_c.reshape(-1).repeat(B).cpu())

            img_counter += B

    handle.remove()

    images_tensor = torch.cat(all_images, dim=0)       # (N_img, 3, 1008, 1008)
    z_norm_tensor = torch.cat(all_z_norm, dim=0)       # (N_tokens, G)
    z_blocks_tensor = torch.cat(all_z_blocks, dim=0) if is_bsf else None
    valid_mask_tensor = torch.cat(all_valid_mask, dim=0) # (N_tokens,)
    od_tensor = torch.cat(all_od, dim=0)               # (N_tokens,)
    hue_tensor = torch.cat(all_hue, dim=0)             # (N_tokens,)
    grad_tensor = torch.cat(all_grad, dim=0)           # (N_tokens,)
    img_idx_tensor = torch.cat(all_img_indices, dim=0) # (N_tokens,)
    cy_tensor = torch.cat(all_cy, dim=0)               # (N_tokens,)
    cx_tensor = torch.cat(all_cx, dim=0)               # (N_tokens,)
    r_grid_tensor = torch.cat(all_r_grid, dim=0)       # (N_tokens,)
    c_grid_tensor = torch.cat(all_c_grid, dim=0)       # (N_tokens,)

    # Cytological masks within valid_mask:
    valid_mask_np = valid_mask_tensor.numpy()
    od_np = od_tensor.numpy()
    grad_np = grad_tensor.numpy()

    if valid_mask_np.any():
        od_thresh = float(np.percentile(od_np[valid_mask_np], 65.0))
        grad_thresh = float(np.percentile(grad_np[valid_mask_np], 65.0))
        cell_fg_mask_np = valid_mask_np & ((od_np > od_thresh) | (grad_np > grad_thresh))
    else:
        cell_fg_mask_np = np.zeros_like(valid_mask_np, dtype=bool)

    slide_bg_mask_np = valid_mask_np & (~cell_fg_mask_np)

    cell_fg_mask_tensor = torch.from_numpy(cell_fg_mask_np)
    slide_bg_mask_tensor = torch.from_numpy(slide_bg_mask_np)

    print(f"[INFO] Tokens total: {len(valid_mask_np)}, Valid slide tokens: {int(valid_mask_np.sum())} ({valid_mask_np.mean()*100:.1f}%)")
    print(f"[INFO] Cellular foreground: {int(cell_fg_mask_np.sum())} ({cell_fg_mask_np.mean()*100:.1f}%), Slide background: {int(slide_bg_mask_np.sum())} ({slide_bg_mask_np.mean()*100:.1f}%)")

    return {
        "images": images_tensor,
        "z_norm": z_norm_tensor,
        "z_blocks": z_blocks_tensor,
        "valid_mask": valid_mask_tensor,
        "cell_fg_mask": cell_fg_mask_tensor,
        "slide_bg_mask": slide_bg_mask_tensor,
        "fg_mask": cell_fg_mask_tensor,
        "optical_density": od_tensor,
        "color_hue": hue_tensor,
        "gradient_energy": grad_tensor,
        "img_indices": img_idx_tensor,
        "cy": cy_tensor,
        "cx": cx_tensor,
        "r_grid": r_grid_tensor,
        "c_grid": c_grid_tensor,
        "grid_size": H_feat,
        "is_bsf": is_bsf
    }


def rank_and_save_concepts(data, sae_type, save_dir):
    """
    Ranks learned concepts/blocks on valid slide tokens according to sparsity,
    foreground selectivity, effective rank, and correlation with cytological proxies.
    Saves full ranking table to CSV and returns top recommended concept IDs.
    """
    z_norm = data["z_norm"].numpy()
    valid_mask = data["valid_mask"].numpy()
    cell_fg_mask = data["cell_fg_mask"].numpy()
    slide_bg_mask = data["slide_bg_mask"].numpy()
    is_bsf = data["is_bsf"]
    z_blocks = data["z_blocks"].numpy() if is_bsf else None

    od = data["optical_density"].numpy()
    hue = data["color_hue"].numpy()
    grad = data["gradient_energy"].numpy()

    num_concepts = z_norm.shape[1]
    N_valid = int(valid_mask.sum())

    records = []

    for g in range(num_concepts):
        act = z_norm[:, g]
        act_mask = (act > 1e-6)
        act_valid = act_mask & valid_mask
        n_act_valid = int(act_valid.sum())
        active_freq = float(n_act_valid / max(1, N_valid))

        fg_mean = float(act[cell_fg_mask].mean()) if cell_fg_mask.any() else 0.0
        bg_mean = float(act[slide_bg_mask].mean()) if slide_bg_mask.any() else 0.0
        global_mean = float(act[valid_mask].mean()) + 1e-4 if valid_mask.any() else 1e-4
        foreground_selectivity = (fg_mean + 0.05 * global_mean) / (bg_mean + 0.05 * global_mean)
        max_act = float(act[valid_mask].max()) if valid_mask.any() else 0.0

        # Effective rank via singular values
        if is_bsf and n_act_valid >= 4:
            Z_act = z_blocks[act_valid, g, :]
            s = np.linalg.svd(Z_act, compute_uv=False)
            if len(s) < 4:
                s = np.pad(s, (0, 4 - len(s)), constant_values=1e-8)
            s_sum = s.sum()
            if s_sum > 1e-8:
                s_norm = s / s_sum
                erank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-8))))
                erank = float(np.clip(erank, 1.0, 4.0))
            else:
                erank = 1.0
        else:
            erank = 1.0

        # Correlations with cytological proxies computed over valid_mask
        if n_act_valid >= 3:
            if is_bsf:
                Z_act = z_blocks[act_valid, g, :]
                Z_cent = Z_act - Z_act.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(Z_cent, full_matrices=False)
                proj_pc1 = Z_cent @ Vt[0].T
                ref_feature = proj_pc1
            else:
                ref_feature = act[act_valid]

            corr_optical_density = safe_corr(ref_feature, od[act_valid])
            corr_color_hue = safe_corr(ref_feature, hue[act_valid])
            corr_gradient_energy = safe_corr(ref_feature, grad[act_valid])
        else:
            corr_optical_density, corr_color_hue, corr_gradient_energy = 0.0, 0.0, 0.0

        # Cytological salience alignment and ranking
        pos_od = max(0.0, corr_optical_density)
        pos_grad = max(0.0, corr_gradient_energy)
        cyto_alignment = 1.5 * pos_od + 1.5 * pos_grad + 0.5 * abs(corr_color_hue)

        if is_bsf:
            meets_criteria = bool((0.003 <= active_freq <= 0.25) and (erank >= 3.0) and (foreground_selectivity > 1.2) and (pos_od > 0.10 or pos_grad > 0.15))
            composite_score = float(erank * np.log1p(foreground_selectivity) * (1.0 + cyto_alignment))
        else:
            meets_criteria = bool((0.003 <= active_freq <= 0.25) and (foreground_selectivity > 1.2) and (pos_od > 0.10 or pos_grad > 0.15))
            composite_score = float(np.log1p(fg_mean * 10.0) * np.log1p(foreground_selectivity) * (1.0 + cyto_alignment))

        records.append({
            "concept_id": g,
            "active_freq": active_freq,
            "foreground_selectivity": foreground_selectivity,
            "mean_activation": fg_mean,
            "max_activation": max_act,
            "erank": erank,
            "corr_optical_density": corr_optical_density,
            "corr_color_hue": corr_color_hue,
            "corr_gradient_energy": corr_gradient_energy,
            "composite_score": composite_score,
            "meets_criteria": meets_criteria
        })

    df = pd.DataFrame(records)

    df_meet = df[df["meets_criteria"]].sort_values(by="composite_score", ascending=False)
    df_other = df[~df["meets_criteria"]].sort_values(by="composite_score", ascending=False)
    df_ranked = pd.concat([df_meet, df_other]).reset_index(drop=True)
    df_ranked["rank"] = df_ranked.index + 1

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"concept_ranking_{sae_type}.csv")
    df_ranked.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved full concept ranking to: {csv_path}")

    # Display Top 15 recommended concepts
    print(f"\n{'='*95}")
    print(f"Top 15 Recommended Concepts for {sae_type}:")
    print(f"{'Rank':<5} {'ID':<6} {'Score':<9} {'ActFreq':<9} {'FgSelect':<10} {'ERank':<7} {'CorrOD':<9} {'CorrHue':<9} {'CorrGrad':<9} {'PassFilter'}")
    print(f"{'-'*95}")
    for _, row in df_ranked.head(15).iterrows():
        print(f"{int(row['rank']):<5} {int(row['concept_id']):<6} {row['composite_score']:<9.4f} "
              f"{row['active_freq']:<9.4f} {row['foreground_selectivity']:<10.2f} {row['erank']:<7.2f} "
              f"{row['corr_optical_density']:<+9.3f} {row['corr_color_hue']:<+9.3f} {row['corr_gradient_energy']:<+9.3f} "
              f"{str(bool(row['meets_criteria']))}")
    print(f"{'='*95}\n")

    return df_ranked


def visualize_subspace_traversals(data, sae_model, target_concepts, sae_type, n_quantiles, patch_radius, save_dir):
    """
    Performs 4D Orthonormal Subspace Axis Traversal for specified BSF concept blocks.
    Isolates each principal axis while penalizing off-axis deviations and crops 96x96
    cytological patches around activating tokens.
    """
    is_bsf = data["is_bsf"]
    if not is_bsf:
        print(f"\n[INFO] Model {sae_type} is a 1D SAE. Subspace traversal is specific to Block-Sparse Featurizers.")
        return

    images_np = data["images"].permute(0, 2, 3, 1).numpy().clip(0, 1)
    z_norm_np = data["z_norm"].numpy()
    z_blocks_np = data["z_blocks"].numpy()
    valid_mask_np = data["valid_mask"].numpy()

    od_np = data["optical_density"].numpy()
    hue_np = data["color_hue"].numpy()
    grad_np = data["gradient_energy"].numpy()

    token_img_idx = data["img_indices"].numpy()
    token_cy = data["cy"].numpy()
    token_cx = data["cx"].numpy()

    H_img, W_img = images_np.shape[1], images_np.shape[2]
    pr = patch_radius
    quantiles = np.linspace(5, 95, n_quantiles)

    for g in target_concepts:
        # Candidate tokens must activate and be in valid_mask
        act_cand_mask = (z_norm_np[:, g] > 1e-6) & valid_mask_np
        if act_cand_mask.sum() < 4:
            act_cand_mask = (z_norm_np[:, g] > 1e-6)
        if act_cand_mask.sum() == 0:
            print(f"Warning: Concept {g} has no active tokens. Skipping traversal.")
            continue

        cand_indices = np.where(act_cand_mask)[0]

        # Require the 96x96 crop window around (cy, cx) to have min_rgb_row_mean > 0.05
        # and min_rgb_col_mean > 0.05 (eliminates black aspect-ratio padding bars)
        clean_indices = []
        for idx in cand_indices:
            img_idx = token_img_idx[idx]
            cy = int(np.clip(token_cy[idx], pr, H_img - pr))
            cx = int(np.clip(token_cx[idx], pr, W_img - pr))
            crop_cand = images_np[img_idx, cy - pr : cy + pr, cx - pr : cx + pr]
            min_row = crop_cand.mean(axis=(1, 2)).min()
            min_col = crop_cand.mean(axis=(0, 2)).min()
            if min_row > 0.05 and min_col > 0.05:
                clean_indices.append(idx)

        if len(clean_indices) >= 4:
            act_indices = np.array(clean_indices, dtype=np.int64)
        else:
            act_indices = cand_indices  # fallback if too few crops pass filter

        norms_g = z_norm_np[act_indices, g]
        p50 = np.percentile(norms_g, 50.0)
        subset_indices = act_indices[norms_g >= p50]
        if len(subset_indices) < 8:
            subset_indices = act_indices

        Z_act_g = z_blocks_np[subset_indices, g, :]  # (N_act, 4)
        mean_z = Z_act_g.mean(axis=0, keepdims=True)
        Z_centered = Z_act_g - mean_z

        # PCA via SVD
        U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        coords_pca = Z_centered @ Vt.T  # (N_act, 4)
        var_explained = (S ** 2) / np.maximum(np.sum(S ** 2), 1e-12) * 100.0

        fig, axes = plt.subplots(4, n_quantiles, figsize=(2.6 * n_quantiles, 10.5))
        if n_quantiles == 1:
            axes = axes.reshape(4, 1)

        for k in range(4):
            r_od = safe_corr(coords_pca[:, k], od_np[subset_indices])
            r_hue = safe_corr(coords_pca[:, k], hue_np[subset_indices])
            r_grad = safe_corr(coords_pca[:, k], grad_np[subset_indices])

            proxy_candidates = [("Optical Dens.", r_od), ("Color Hue", r_hue), ("Grad. Energy", r_grad)]
            top_proxy_name, top_r = max(proxy_candidates, key=lambda x: abs(x[1]))

            axes[k, 0].set_ylabel(
                f"Axis {k+1} ({var_explained[k]:.1f}% var)\n{top_proxy_name} (r={top_r:+.2f})",
                fontsize=9, fontweight='bold'
            )

            other_axes = [o for o in range(4) if o != k]
            other_norm = np.linalg.norm(coords_pca[:, other_axes], axis=1)

            for j, q in enumerate(quantiles):
                q_val = np.percentile(coords_pca[:, k], q)
                dist = np.abs(coords_pca[:, k] - q_val) + 0.25 * other_norm
                best_sub_idx = np.argmin(dist)
                global_token_idx = subset_indices[best_sub_idx]

                img_idx = token_img_idx[global_token_idx]
                orig_cy = token_cy[global_token_idx]
                orig_cx = token_cx[global_token_idx]

                # Clamp (cy, cx) so [cy - pr, cy + pr] and [cx - pr, cx + pr] always lie strictly within [0, H_img] and [0, W_img]
                cy = int(np.clip(orig_cy, pr, H_img - pr))
                cx = int(np.clip(orig_cx, pr, W_img - pr))

                img_np = images_np[img_idx]
                crop = img_np[cy - pr : cy + pr, cx - pr : cx + pr]

                ax = axes[k, j]
                ax.imshow(crop)

                # Subtle circle at actual token center relative to clamped crop
                center_y = float(orig_cy - (cy - pr))
                center_x = float(orig_cx - (cx - pr))
                circle = patches.Circle((center_x, center_y), radius=3.5, edgecolor='#00ffcc', facecolor='none', linewidth=1.2, alpha=0.9)
                ax.add_patch(circle)
                ax.set_xticks([])
                ax.set_yticks([])

                actual_coord = coords_pca[best_sub_idx, k]
                ax.set_title(f"P{int(round(q))}: {actual_coord:+.2f}", fontsize=8)

        fig.suptitle(f"{sae_type} Concept {g} - 4D Orthonormal Subspace Axis Traversal", fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        out_traversal = os.path.join(save_dir, f"{sae_type}_concept_{g}_axis_traversal.png")
        plt.savefig(out_traversal, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved 4D subspace axis traversal to: {out_traversal}")

    # Plot 3D manifolds using plot_block_concepts_manifold
    if len(target_concepts) > 0:
        try:
            manifold_3d_path = os.path.join(save_dir, f"{sae_type}_top_concepts_3d_manifolds.png")
            plot_block_concepts_manifold(
                z_blocks=z_blocks_np,
                atoms=sae_model.atoms().cpu().numpy(),
                images=images_np,
                concepts=target_concepts,
                grid_size=data["grid_size"],
                n_img=min(5, len(images_np)),
                save_path=manifold_3d_path
            )
        except Exception as e:
            print(f"Error generating plot_block_concepts_manifold: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze continuous morphological manifolds and 4D subspaces in BSFs.")
    parser.add_argument("--sae_type", type=str, default="GrassmannianBSF",
                        choices=["GrassmannianBSF", "VanillaBSF", "OMPSAE", "MPSAE", "TopKSAE", "RATopKSAE"],
                        help="SAE architecture to evaluate")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to trained SAE checkpoint (defaults to ./interpretability_analysis/checkpoints/{sae_type}_sae.pt)")
    parser.add_argument("--weight_checkpoint", type=str, default="./weights/rcnn_best_anchors/best_checkpoint.pth",
                        help="Path to SAM3 Faster R-CNN weights checkpoint")
    parser.add_argument("--base_image_dir", type=str, default="",
                        help="Path to dataset image directory (auto-discovered if empty)")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Root path to the RIVA dataset")
    parser.add_argument("--annotations_file", type=str, default="",
                        help="Path to RIVA annotations file (CSV/JSON). Gracefully optional if not found.")
    parser.add_argument("--k", type=int, default=16, help="Target sparsity (L0 / k) used during training")
    parser.add_argument("--expansion_factor", type=float, default=4.0, help="Expansion factor used during training")
    parser.add_argument("--block_size", type=int, default=4, help="Block size for Block-Sparse Featurizers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on number of images to evaluate (0 = all)")
    parser.add_argument("--top_n_concepts", type=int, default=5, help="Number of top manifold blocks to visualize")
    parser.add_argument("--concept_ids", type=int, nargs="*", default=None,
                        help="Specific concept IDs to analyze (overrides automatic ranking)")
    parser.add_argument("--n_quantiles", type=int, default=6,
                        help="Number of quantiles from P5 to P95 along each of the 4 principal axes")
    parser.add_argument("--patch_radius", type=int, default=48,
                        help="Patch radius around activating token center (crops 2*r x 2*r)")
    parser.add_argument("--save_dir", type=str, default="./interpretability_analysis/manifold_analysis",
                        help="Directory to save analysis outputs and figures")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load SAM3 Backbone
    print("Loading SAM3 Faster R-CNN...")
    sam3_model = build_sam3_fasterrcnn(trainable_backbone=False)

    ckpt_to_load = args.weight_checkpoint
    if not (ckpt_to_load and os.path.exists(ckpt_to_load)):
        weight_dir = Path("./weights")
        weight_files = list(weight_dir.rglob("*.pt")) + list(weight_dir.rglob("*.pth"))
        ckpt_to_load = str(weight_files[0]) if weight_files else None

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"Found SAM3 Faster R-CNN weights: {ckpt_to_load}. Loading...")
        state_dict = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        sam3_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: No weights found at {args.weight_checkpoint}. Using initialized weights.")

    sam3_model.to(device)
    sam3_model.eval()

    # 2. Resolve Dataset
    candidate_dirs = [
        args.base_image_dir,
        os.path.join(args.dataset_path, "images/images") if args.dataset_path else None,
        os.path.join(args.dataset_path, "images") if args.dataset_path else None,
        "/local_data/RIVA/images/images",
        "/local_data/RIVA/images",
        "./RIVA/images/images",
        "./data/RIVA/images/images",
        "./images/images"
    ]

    resolved_base_dir = None
    train_paths, val_paths, test_paths = [], [], []

    for candidate in candidate_dirs:
        if candidate and os.path.isdir(candidate):
            tr, va, te = get_data_splits(candidate)
            if len(tr) > 0 or len(va) > 0 or len(te) > 0:
                resolved_base_dir = candidate
                train_paths, val_paths, test_paths = tr, va, te
                break

    if not resolved_base_dir:
        raise FileNotFoundError(
            "Could not find dataset images in default locations. "
            "Please provide --base_image_dir <path> containing 'train', 'val', or 'test' folders."
        )

    print(f"Using dataset from: {resolved_base_dir}")
    eval_paths = test_paths if len(test_paths) > 0 else (val_paths if len(val_paths) > 0 else train_paths)
    if args.max_images > 0:
        eval_paths = eval_paths[:args.max_images]
    print(f"Evaluating {len(eval_paths)} images")

    # Check for optional annotations file
    ann_file = find_annotations_file(args.annotations_file, resolved_base_dir)
    if ann_file:
        print(f"Found annotations file: {ann_file}")
    else:
        print("No annotations file found (optional). Proceeding with unsupervised cytological proxies.")

    dataset = ImagePathDataset(eval_paths)
    num_workers = 0 if os.name == "nt" else 4
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 3. Instantiate and load SAE
    input_dim = 256
    hidden_dim = int(input_dim * args.expansion_factor)

    loaded_points = None
    if args.sae_type == "RATopKSAE":
        points_path = "./interpretability_analysis/checkpoints/ratopk_points.pt"
        if os.path.exists(points_path):
            print(f"Loading RATopK anchor points from {points_path}")
            loaded_points = torch.load(points_path, map_location=device, weights_only=False)
        else:
            feat_path = "./interpretability_analysis/features/train_features.pt"
            if os.path.exists(feat_path):
                print(f"Sampling RATopK anchor points from {feat_path}")
                feats = torch.load(feat_path, map_location=device, weights_only=False)
                step = max(1, len(feats) // 1024)
                loaded_points = feats[::step][:1024].clone()
            else:
                loaded_points = torch.randn(1024, input_dim, device=device)

    sae_classes = {
        "TopKSAE": lambda d, h, k: TopKSAE(d, h, top_k=k),
        "RATopKSAE": lambda d, h, k: RATopKSAE(d, h, points=loaded_points.to(device), top_k=k),
        "MPSAE": lambda d, h, k: MPSAE(d, h, k=k),
        "OMPSAE": lambda d, h, k: OMPSAE(d, h, k=k),
        "VanillaBSF": lambda d, h, k: VanillaBSF(d, n_groups=max(1, h // args.block_size), group_size=args.block_size, l0=k),
        "GrassmannianBSF": lambda d, h, k: GrassmannianBSF(d, n_groups=max(1, h // args.block_size), group_size=args.block_size, l0=k),
    }

    sae_model = sae_classes[args.sae_type](input_dim, hidden_dim, args.k)
    checkpoint_path = args.checkpoint if args.checkpoint else f"./interpretability_analysis/checkpoints/{args.sae_type}_sae.pt"
    if os.path.exists(checkpoint_path):
        print(f"Found SAE checkpoint: {checkpoint_path}. Loading...")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sae_model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized SAE.")

    sae_model.to(device)
    sae_model.eval()

    # 4. Feature and proxy extraction
    data = extract_features_and_proxies(sam3_model, sae_model, dataloader, device)

    # 5. Concept ranking and CSV export
    df_ranked = rank_and_save_concepts(data, args.sae_type, args.save_dir)

    # 6. Select concept IDs for 4D subspace traversal
    if args.concept_ids is not None and len(args.concept_ids) > 0:
        target_concepts = args.concept_ids
        print(f"Analyzing user-specified concept IDs: {target_concepts}")
    else:
        target_concepts = df_ranked["concept_id"].iloc[:args.top_n_concepts].tolist()
        print(f"Analyzing top {len(target_concepts)} automatically ranked concept IDs: {target_concepts}")

    # 7. Subspace axis traversal and 3D manifold visualization
    visualize_subspace_traversals(
        data=data,
        sae_model=sae_model,
        target_concepts=target_concepts,
        sae_type=args.sae_type,
        n_quantiles=args.n_quantiles,
        patch_radius=args.patch_radius,
        save_dir=args.save_dir
    )

    print("\n[SUCCESS] Manifold analysis complete.")


if __name__ == "__main__":
    main()
