#!/usr/bin/env python3
"""
Block-Sparse Featurizer (BSF) Paper Figure 9 Reproduction Script.

Paper: "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds"
       (arXiv:2606.25234, Fel et al.)
Figure 9: "A block carries a concept together with its intrinsic geometry"

This script reproduces Figure 9 on the RIVA PAP smear test dataset:
- Left: 2x5 grid of 10 top-activating test set cellular regions (256x256 crops)
  with luminance-preserving smooth RGB manifold overlay.
- Right: 3D concept geometry cloud (u_g), tilted floor plane with 2D shadow projection,
  and the closest 1D directional SAE atom arrow (e.g. from OMPSAE/TopKSAE b=1 baseline).

Produces:
1. Individual high-res figure per block: {save_dir}/{sae_type}_block_C{g}_fig9_panel.png
2. Combined 6-block master publication figure: {save_dir}/{sae_type}_Figure9_6blocks_grid.png
"""

import os
import sys
import argparse
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TVF
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Add project root and interpretability_analysis to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.sam3_rcnn_v2 import (
    build_sam3_fasterrcnn,
    sam3_resize_longest_side_and_pad_square,
)
from interpretability_analysis.train_saes import get_data_splits, ImagePathDataset
from overcomplete import TopKSAE
from custom_saes import (
    RATopKSAE,
    OMPSAE,
    MpSAE as MPSAE,
    BSF,
    VanillaBSF,
    GrassmannianBSF,
)

# Biological descriptions for verified selective blocks
CONCEPT_DESCRIPTIONS: Dict[int, str] = {
    139: "Dysplasia Trajectory",
    92: "PMN / Inflammatory Exudate",
    75: "Epithelial-Inflammatory Transition",
    214: "Continuous Epithelial Manifold",
    89: "Continuous Epithelial Manifold",
    12: "Endocervical Honeycomb Sheets",
}


# ==============================================================================
# 1. Dataset & Image Utilities
# ==============================================================================

def find_image_directory(base_image_dir: str = "") -> str:
    """Auto-discovers the RIVA images root directory."""
    if base_image_dir and os.path.isdir(base_image_dir):
        return str(Path(base_image_dir).resolve())

    search_dirs = [
        "./RIVA/images/images",
        "./RIVA/images",
        "./RIVA final/images/images",
        "./RIVA final/images",
        "./data/RIVA/images/images",
        "./data/images",
        "/local_data/RIVA/images/images",
        "/local_data/RIVA/images",
        "./RIVA",
    ]

    for d in search_dirs:
        p = Path(d)
        if p.is_dir() and ((p / "test").is_dir() or (p / "val").is_dir() or (p / "train").is_dir()):
            return str(p.resolve())

    if os.path.isdir("./RIVA"):
        return str(Path("./RIVA").resolve())

    raise FileNotFoundError("Could not auto-discover RIVA image directory. Please specify --base_image_dir.")


class TestImagesWithMetadataDataset(Dataset):
    """Loads images, applies SAM3 square resize/pad to target_size (default 1008x1008),
    and returns both the tensor and the file path."""
    def __init__(self, image_paths: List[str], target_size: int = 1008):
        self.image_paths = image_paths
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img_t = TVF.to_tensor(img)
        img_padded, _, _ = sam3_resize_longest_side_and_pad_square(img_t, target_size=self.target_size)
        return img_padded, path


# ==============================================================================
# 2. Geometry & Colorization Utilities (Goodfire BSF Paper Specification)
# ==============================================================================

def pca_fit(c: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean and top-k principal axes of contribution cloud c (N, d).
    
    Returns:
        mean: shape (d,)
        comps: shape (k, d)
    """
    mean = c.mean(axis=0)
    centered = c - mean
    # SVD on centered data: centered = U @ diag(S) @ V
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:k]
    return mean, comps


def radial_clip(proj: np.ndarray, pct: float = 98.0) -> np.ndarray:
    """Radial outlier clipping scaling points beyond percentile radius back to boundary."""
    norm = np.linalg.norm(proj, axis=1)
    if len(norm) == 0:
        return proj
    cap = float(np.percentile(norm, pct))
    s = np.minimum(1.0, cap / np.maximum(norm, 1e-12))
    return proj * s[:, None]


def make_colorize(proj: np.ndarray, saturation: float = 1.35) -> Callable[[np.ndarray], np.ndarray]:
    """Maps 3D principal coordinates into RGB hues based on radial direction.
    
    Ensures direction in 3D PCA space maps vividly to Red, Green, and Blue channels.
    """
    def f(p: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(p, axis=-1, keepdims=True)
        unit = p / np.maximum(n, 1e-8)
        rgb = 0.5 + 0.5 * unit * saturation
        return np.clip(rgb, 0.0, 1.0)
    return f


def compute_erank(Z_act: np.ndarray) -> float:
    """Computes effective rank of active block activations via singular values entropy."""
    if len(Z_act) < 4:
        return 1.0
    s = np.linalg.svd(Z_act, compute_uv=False)
    if len(s) < 4:
        s = np.pad(s, (0, 4 - len(s)), constant_values=1e-8)
    s_sum = s.sum()
    if s_sum > 1e-8:
        s_norm = s / s_sum
        erank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-8))))
        return float(np.clip(erank, 1.0, 4.0))
    return 1.0


# ==============================================================================
# 3. Model Loading & 1D SAE Baseline Direction Matching
# ==============================================================================

def load_sam3_model(weight_checkpoint: str, device: torch.device) -> nn.Module:
    """Instantiates SAM3 Faster R-CNN and loads checkpoint if available."""
    print("[INFO] Building SAM3 Faster R-CNN backbone...")
    sam3_model = build_sam3_fasterrcnn(trainable_backbone=False)

    ckpt_to_load = weight_checkpoint
    if not (ckpt_to_load and os.path.exists(ckpt_to_load)):
        weight_dir = Path("./weights")
        weight_files = list(weight_dir.rglob("*.pt")) + list(weight_dir.rglob("*.pth"))
        ckpt_to_load = str(weight_files[0]) if weight_files else None

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"[INFO] Found Faster R-CNN checkpoint: {ckpt_to_load}. Loading...")
        state_dict = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        sam3_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARNING] No Faster R-CNN weights found at {weight_checkpoint}. Using default weights.")

    sam3_model.to(device)
    sam3_model.eval()
    return sam3_model


def build_and_load_bsf_model(
    sae_type: str,
    checkpoint_path: str,
    device: torch.device,
    input_dim: int = 256,
    expansion_factor: float = 4.0,
    k: int = 16,
    block_size: int = 4
) -> Tuple[BSF, int]:
    """Instantiates GrassmannianBSF or VanillaBSF and loads trained weights."""
    hidden_dim = int(input_dim * expansion_factor)
    n_groups = max(1, hidden_dim // block_size)

    if sae_type == "GrassmannianBSF":
        model = GrassmannianBSF(input_dim, n_groups=n_groups, group_size=block_size, l0=k)
    elif sae_type == "VanillaBSF":
        model = VanillaBSF(input_dim, n_groups=n_groups, group_size=block_size, l0=k)
    else:
        raise ValueError(f"Unsupported BSF type: {sae_type}. Expected GrassmannianBSF or VanillaBSF.")

    ckpt = checkpoint_path
    if not ckpt:
        candidates = [
            f"./interpretability_analysis/checkpoints/{sae_type}_sae.pt",
            f"./interpretability_analysis/checkpoints/{sae_type}.pt",
            f"./checkpoints/{sae_type}_sae.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt = c
                break

    if ckpt and os.path.exists(ckpt):
        print(f"[INFO] Loading BSF checkpoint from: {ckpt}")
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st, strict=False)
    else:
        print(f"[WARNING] BSF checkpoint '{ckpt}' not found. Using initialized model.")

    model.to(device)
    model.eval()
    return model, n_groups


def load_directional_sae_dictionary(
    directional_sae_type: str,
    directional_sae_checkpoint: str,
    device: torch.device,
    input_dim: int = 256,
    expansion_factor: float = 4.0,
    k: int = 16
) -> Tuple[np.ndarray, str]:
    """Loads 1D SAE dictionary W_1D (1024, 256) normalized to unit L2 norm."""
    hidden_dim = int(input_dim * expansion_factor)

    resolved_ckpt = directional_sae_checkpoint
    if not resolved_ckpt or not os.path.exists(resolved_ckpt):
        candidates = [
            f"./interpretability_analysis/checkpoints/{directional_sae_type}_sae.pt",
            f"./interpretability_analysis/checkpoints/{directional_sae_type}.pt",
            f"./interpretability_analysis/checkpoints/OMPSAE_sae.pt",
            f"./interpretability_analysis/checkpoints/TopKSAE_sae.pt",
            f"./interpretability_analysis/checkpoints/MPSAE_sae.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_ckpt = c
                break

    resolved_type = directional_sae_type
    if resolved_ckpt and os.path.exists(resolved_ckpt):
        fname = os.path.basename(resolved_ckpt).lower()
        if "omp" in fname:
            resolved_type = "OMPSAE"
        elif "topk" in fname:
            resolved_type = "TopKSAE"
        elif "mp" in fname:
            resolved_type = "MPSAE"

    # Instantiate model
    if resolved_type == "TopKSAE":
        model = TopKSAE(input_dim, hidden_dim, top_k=k)
    elif resolved_type == "OMPSAE":
        model = OMPSAE(input_dim, hidden_dim, k=k)
    elif resolved_type == "MPSAE":
        model = MPSAE(input_dim, hidden_dim, k=k)
    else:
        model = TopKSAE(input_dim, hidden_dim, top_k=k)

    if resolved_ckpt and os.path.exists(resolved_ckpt):
        print(f"[INFO] Loading directional 1D SAE from: {resolved_ckpt} ({resolved_type})")
        st = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st, strict=False)
    else:
        print(f"[WARNING] 1D SAE checkpoint not found. Using initialized {resolved_type} dictionary.")

    # Extract dictionary matrix (hidden_dim, input_dim)
    if hasattr(model, "get_dictionary") and callable(model.get_dictionary):
        W = model.get_dictionary().detach().cpu().numpy()
    elif hasattr(model, "decoder_atoms") and callable(model.decoder_atoms):
        W = model.decoder_atoms().detach().cpu().numpy()
    elif hasattr(model, "W_dec"):
        W = model.W_dec.detach().cpu().numpy()
    else:
        W = np.random.randn(hidden_dim, input_dim)

    # Normalize rows to unit L2 norm
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_normed = W / np.maximum(row_norms, 1e-8)
    return W_normed.astype(np.float32), resolved_type


def match_closest_1d_sae_atom(
    D_g: np.ndarray,
    W_1D: np.ndarray,
    comps: np.ndarray,
    mean: np.ndarray,
    u_g: np.ndarray
) -> Tuple[int, np.ndarray, float]:
    """Matches block g's subspace D_g with the closest 1D SAE atom.
    
    1. For block g's subspace D_g in R^(4 x 256), project every 1D atom onto span(D_g):
       proj = W_1D @ D_g.T @ D_g
       proj_norm_j = || proj[j] ||_2
       j* = argmax_j proj_norm_j
    2. In-subspace vector: v_1D = proj[j*]
    3. Project onto 3D PCA axes: d_arrow = v_1D @ comps.T
    4. Sign alignment: if dot(d_arrow, mean @ comps.T) < 0: d_arrow = -d_arrow
    5. Scale length to 1.15x the 95th percentile radius of u_g.
    """
    # D_g shape: (group_size, d) -> (4, 256)
    # W_1D shape: (1024, 256)
    # Projection onto span(D_g): proj = W_1D @ D_g.T @ D_g
    proj = (W_1D @ D_g.T) @ D_g
    proj_norms = np.linalg.norm(proj, axis=1)
    j_star = int(np.argmax(proj_norms))
    max_proj_norm = float(proj_norms[j_star])

    v_1D = proj[j_star]
    d_arrow = v_1D @ comps.T  # (3,)

    # Sign alignment with mean offset
    offset = mean @ comps.T
    if np.linalg.norm(offset) > 1e-6:
        if np.dot(d_arrow, offset) < 0:
            d_arrow = -d_arrow
    else:
        if np.dot(d_arrow, u_g.mean(axis=0)) < 0:
            d_arrow = -d_arrow

    # Scale length to 1.15 * 95th percentile radius of u_g
    r_95 = float(np.percentile(np.linalg.norm(u_g, axis=1), 95.0))
    arrow_norm = np.linalg.norm(d_arrow)
    if arrow_norm > 1e-8:
        d_arrow = (d_arrow / arrow_norm) * (1.15 * r_95)

    return j_star, d_arrow, max_proj_norm


# ==============================================================================
# 4. Test Set Feature & Token Extraction
# ==============================================================================

def extract_test_features(
    sam3_model: nn.Module,
    bsf_model: BSF,
    test_paths: List[str],
    device: torch.device,
    batch_size: int = 4,
    target_size: int = 1008
) -> Dict:
    """Runs inference across the test dataset to extract FPN feature activations,
    BSF latent block coordinates, foreground token masks, and token coordinates."""
    dataset = TestImagesWithMetadataDataset(test_paths, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    activations = {}
    def hook_fn(module, input, output):
        activations["1"] = output["1"].detach()

    handle = sam3_model.backbone.register_forward_hook(hook_fn)

    all_images = []
    all_z_blocks = []  # List of tensors of shape (B, H_feat, W_feat, n_groups, group_size)
    all_heat = []      # List of tensors of shape (B, H_feat, W_feat, n_groups)
    all_valid = []     # List of boolean masks of shape (B, H_feat, W_feat)

    H_feat, W_feat = None, None

    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(dataloader, desc="Extracting Test Features"):
            batch_dev = batch_imgs.to(device)
            B = batch_dev.shape[0]
            all_images.append(batch_imgs.cpu())

            _ = sam3_model.backbone(batch_dev)
            feat = activations["1"]
            B, C, H_feat, W_feat = feat.shape
            P = H_feat * W_feat

            feat_flat = feat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
            z_blk = bsf_model.encode_blocks(feat_flat)  # (B*P, n_groups, group_size)
            z_norm = z_blk.norm(dim=-1)                 # (B*P, n_groups)

            n_groups = z_blk.shape[1]
            group_size = z_blk.shape[2]

            z_blk_spatial = z_blk.view(B, H_feat, W_feat, n_groups, group_size).cpu()
            z_norm_spatial = z_norm.view(B, H_feat, W_feat, n_groups).cpu()

            all_z_blocks.append(z_blk_spatial)
            all_heat.append(z_norm_spatial)

            # Valid non-padded slide tokens:
            # mean_rgb > 0.08 and not on the 1-token outer perimeter (1 <= r < H-1, 1 <= c < W-1)
            mean_rgb = batch_dev.mean(dim=1, keepdim=True)
            mean_rgb_pooled = F.adaptive_avg_pool2d(mean_rgb, (H_feat, W_feat)).squeeze(1)

            r_coords = torch.arange(H_feat, device=device)
            c_coords = torch.arange(W_feat, device=device)
            grid_r, grid_c = torch.meshgrid(r_coords, c_coords, indexing='ij')
            inner_mask = (grid_r >= 1) & (grid_r < H_feat - 1) & (grid_c >= 1) & (grid_c < W_feat - 1)

            valid_mask = (mean_rgb_pooled > 0.08) & inner_mask.unsqueeze(0)
            all_valid.append(valid_mask.cpu())

    handle.remove()

    images_tensor = torch.cat(all_images, dim=0)     # (N_img, 3, 1008, 1008)
    z_blocks_tensor = torch.cat(all_z_blocks, dim=0) # (N_img, H_feat, W_feat, n_groups, group_size)
    heat_tensor = torch.cat(all_heat, dim=0)         # (N_img, H_feat, W_feat, n_groups)
    valid_tensor = torch.cat(all_valid, dim=0)       # (N_img, H_feat, W_feat)

    print(f"[INFO] Test features extracted: {len(test_paths)} images, grid {H_feat}x{W_feat}, "
          f"valid tokens: {int(valid_tensor.sum())} / {valid_tensor.numel()} ({valid_tensor.float().mean()*100:.1f}%)")

    return {
        "images": images_tensor,
        "z_blocks": z_blocks_tensor,
        "heat": heat_tensor,
        "valid": valid_tensor,
        "H_feat": H_feat,
        "W_feat": W_feat,
        "image_paths": test_paths,
        "target_size": target_size,
    }


# ==============================================================================
# 5. Top-10 Cellular Region ROI Cropping & Luminance-Preserving Blend
# ==============================================================================

def find_top_activating_regions(
    heat: torch.Tensor,
    valid: torch.Tensor,
    g: int,
    n_regions: int = 10,
    min_pixel_dist: float = 128.0,
    crop_size: int = 256,
    target_size: int = 1008
) -> List[Dict]:
    """Finds top 10 highest-activating distinct cellular regions for concept block g.
    
    Enforces min_pixel_dist separation between peaks from the same image and clamps
    the crop window strictly inside [0, target_size] so there are never black border bars.
    """
    N_img, H_feat, W_feat, n_groups = heat.shape
    stride = target_size / float(H_feat)

    # Flatten tokens and filter by valid mask and non-zero activation
    heat_g = heat[..., g].numpy()       # (N_img, H_feat, W_feat)
    valid_np = valid.numpy()            # (N_img, H_feat, W_feat)

    candidate_coords = []
    for img_idx in range(N_img):
        active_r, active_c = np.where((heat_g[img_idx] > 1e-5) & valid_np[img_idx])
        for r, c in zip(active_r, active_c):
            val = heat_g[img_idx, r, c]
            y_center = (r + 0.5) * stride
            x_center = (c + 0.5) * stride
            candidate_coords.append((val, img_idx, r, c, y_center, x_center))

    # Sort descending by activation value
    candidate_coords.sort(key=lambda item: item[0], reverse=True)

    selected = []
    half_crop = crop_size // 2

    for val, img_idx, r, c, y_c, x_c in candidate_coords:
        # Check minimum pixel distance with already selected regions in the same image
        conflict = False
        for s in selected:
            if s["img_idx"] == img_idx:
                dist = np.sqrt((y_c - s["y_center"]) ** 2 + (x_c - s["x_center"]) ** 2)
                if dist < min_pixel_dist:
                    conflict = True
                    break
        if conflict:
            continue

        # Clamp crop strictly within [0, target_size]
        y_min = int(np.clip(int(round(y_c - half_crop)), 0, target_size - crop_size))
        y_max = y_min + crop_size
        x_min = int(np.clip(int(round(x_c - half_crop)), 0, target_size - crop_size))
        x_max = x_min + crop_size

        selected.append({
            "val": float(val),
            "img_idx": img_idx,
            "r": int(r),
            "c": int(c),
            "y_center": float(y_c),
            "x_center": float(x_c),
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
        })

        if len(selected) >= n_regions:
            break

    # If fewer than n_regions passed distance filter, loosen distance requirement
    if len(selected) < n_regions and len(candidate_coords) > len(selected):
        for val, img_idx, r, c, y_c, x_c in candidate_coords:
            already_in = any(s["img_idx"] == img_idx and s["r"] == r and s["c"] == c for s in selected)
            if already_in:
                continue
            y_min = int(np.clip(int(round(y_c - half_crop)), 0, target_size - crop_size))
            y_max = y_min + crop_size
            x_min = int(np.clip(int(round(x_c - half_crop)), 0, target_size - crop_size))
            x_max = x_min + crop_size

            selected.append({
                "val": float(val),
                "img_idx": img_idx,
                "r": int(r),
                "c": int(c),
                "y_center": float(y_c),
                "x_center": float(x_c),
                "y_min": y_min,
                "y_max": y_max,
                "x_min": x_min,
                "x_max": x_max,
            })
            if len(selected) >= n_regions:
                break

    return selected


def compute_blended_crop(
    crop_rgb: np.ndarray,
    rgb_up: np.ndarray,
    norm_up: np.ndarray
) -> np.ndarray:
    """Luminance-preserving smooth overlay blending matching Paper Figure 9:
    
    1. Normalize alpha per crop: alpha = np.clip((norm_up / (p98 + 1e-8)) ** 0.85, 0, 1)
       Zero out values below 15% of peak so unactivated background stays clean.
    2. Convert RGB microscopy patch to clean luminance:
       gray = 0.299 * R + 0.587 * G + 0.114 * B
    3. Modulate manifold RGB hue by cell's grayscale luminance:
       tinted_cell = np.clip(rgb_up * (0.35 + 0.85 * gray), 0, 1)
    4. Blend with underlying cellular texture:
       blended = (1.0 - 0.78 * alpha) * crop_rgb + (0.78 * alpha) * tinted_cell
    """
    p98 = float(np.percentile(norm_up, 98.0)) if norm_up.size > 0 else 1.0
    alpha = np.clip((norm_up / (p98 + 1e-8)) ** 0.85, 0.0, 1.0)
    alpha[alpha < 0.15] = 0.0

    gray = (0.299 * crop_rgb[..., 0] + 0.587 * crop_rgb[..., 1] + 0.114 * crop_rgb[..., 2])[..., None]
    tinted_cell = np.clip(rgb_up * (0.35 + 0.85 * gray), 0.0, 1.0)
    blended_patch = (1.0 - 0.78 * alpha[..., None]) * crop_rgb + (0.78 * alpha[..., None]) * tinted_cell
    return np.clip(blended_patch, 0.0, 1.0)


# ==============================================================================
# 6. Plotting Utilities (3D Cloud + Shadow + 1D Arrow + 2x5 Thumbnail Grid)
# ==============================================================================

def plot_3d_concept_manifold(
    ax3d: Axes3D,
    u_g: np.ndarray,
    colorize: Callable,
    d_arrow: np.ndarray,
    erank: float,
    block_g: int,
    sae_1d_name: str,
    j_star: int
):
    """Draws 3D concept geometry cloud u_g, tilted floor plane with 2D shadow,
    and closest 1D SAE baseline atom arrow."""
    N_act = len(u_g)
    if N_act > 2500:
        sub_idx = np.random.choice(N_act, 2500, replace=False)
        u_sub = u_g[sub_idx]
    else:
        u_sub = u_g

    # Point cloud colors and norms
    rgb_sub = colorize(u_sub)
    norm_sub = np.linalg.norm(u_sub, axis=1)
    norm_p98 = float(np.percentile(norm_sub, 98.0)) if norm_sub.size > 0 else 1.0
    alpha_sub = np.clip(0.35 + 0.65 * (norm_sub / (norm_p98 + 1e-8)), 0.20, 0.95)
    rgba_sub = np.column_stack([rgb_sub, alpha_sub])

    # 1. 3D scatter
    ax3d.scatter(
        u_sub[:, 0], u_sub[:, 1], u_sub[:, 2],
        c=rgba_sub, s=14.0, edgecolors='none', depthshade=True
    )

    # 2. Tilted Floor Plane & 2D Shadow Projection
    z_min_val = float(u_g[:, 2].min())
    z_max_val = float(u_g[:, 2].max())
    z_span = max(z_max_val - z_min_val, 1e-4)
    z_floor = float(np.percentile(u_g[:, 2], 1.0) - 0.25 * z_span)

    x_span = max(float(u_g[:, 0].max() - u_g[:, 0].min()), 1e-4)
    y_span = max(float(u_g[:, 1].max() - u_g[:, 1].min()), 1e-4)
    x_min, x_max = u_g[:, 0].min() - 0.15 * x_span, u_g[:, 0].max() + 0.15 * x_span
    y_min, y_max = u_g[:, 1].min() - 0.15 * y_span, u_g[:, 1].max() + 0.15 * y_span

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = np.full_like(xx, z_floor)
    ax3d.plot_surface(xx, yy, zz, color='#EAEAEA', alpha=0.45, shade=False)

    # 2D Shadow on the floor plane
    ax3d.scatter(
        u_sub[:, 0], u_sub[:, 1], np.full_like(u_sub[:, 2], z_floor),
        c='#999999', alpha=0.18, s=1.8, edgecolors='none'
    )

    # 3. Closest 1D SAE Atom Arrow
    p_start = -0.30 * d_arrow
    p_end = 1.0 * d_arrow
    ax3d.plot(
        [p_start[0], p_end[0]],
        [p_start[1], p_end[1]],
        [p_start[2], p_end[2]],
        color='black', linewidth=2.4, zorder=12
    )

    # Arrow tip
    tip_start = p_end - 0.22 * d_arrow
    tip_vec = 0.22 * d_arrow
    ax3d.quiver(
        tip_start[0], tip_start[1], tip_start[2],
        tip_vec[0], tip_vec[1], tip_vec[2],
        color='black', linewidth=2.4, arrow_length_ratio=0.35, zorder=14
    )

    # Label near 1D atom arrow tip
    ax3d.text(
        p_end[0] * 1.06, p_end[1] * 1.06, p_end[2] * 1.06,
        f"1D SAE C{j_star}", color='black', fontsize=8.5, fontweight='bold', zorder=16
    )

    # Camera angle and aesthetics
    ax3d.view_init(elev=22, azim=-55)
    ax3d.set_axis_off()

    desc = CONCEPT_DESCRIPTIONS.get(block_g, "")
    desc_str = f" - {desc}" if desc else ""
    title_text = f"Block C{block_g}{desc_str}\n(erank={erank:.2f}, match: {sae_1d_name} C{j_star})"
    ax3d.set_title(title_text, fontsize=10.5, fontweight='bold', pad=2)


def render_block_panel_to_subgridspec(
    fig: plt.Figure,
    gs_parent: gridspec.SubplotSpec,
    block_info: Dict,
    crop_size: int = 256
):
    """Renders one block's Figure 9 panel into a SubplotSpec:
    - Left: 2x5 grid of blended 256x256 test crops with thin black frame.
    - Right: 3D manifold geometry cloud + floor shadow + 1D arrow.
    """
    # Create 1 row x 2 columns sub-gridspec: Left (crops), Right (3D)
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_parent, width_ratios=[1.38, 1.0], wspace=0.12
    )

    # Left: 2 rows x 5 columns thumbnail grid
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 5, subplot_spec=gs[0], wspace=0.03, hspace=0.03
    )

    blended_crops = block_info["blended_crops"]
    for idx in range(10):
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(gs_left[row, col])

        if idx < len(blended_crops):
            ax.imshow(blended_crops[idx])
        else:
            # Clean dark placeholder if fewer than 10 active crops found
            ax.imshow(np.ones((crop_size, crop_size, 3), dtype=np.float32) * 0.95)

        ax.set_xticks([])
        ax.set_yticks([])
        # Thin crisp black border around each crop
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
            spine.set_visible(True)

    # Right: 3D manifold scatter
    ax3d = fig.add_subplot(gs[1], projection='3d')
    plot_3d_concept_manifold(
        ax3d=ax3d,
        u_g=block_info["u_g"],
        colorize=block_info["colorize"],
        d_arrow=block_info["d_arrow"],
        erank=block_info["erank"],
        block_g=block_info["block_g"],
        sae_1d_name=block_info["sae_1d_name"],
        j_star=block_info["j_star"]
    )


# ==============================================================================
# 7. Processing Pipeline for Each Selected Concept Block
# ==============================================================================

def process_block(
    g: int,
    test_features: Dict,
    D_g: np.ndarray,
    W_1D: np.ndarray,
    sae_1d_name: str,
    crop_size: int = 256
) -> Optional[Dict]:
    """Computes all manifold coordinates, PCA projection, 10 blended test crops,
    and 1D SAE directional arrow alignment for concept block g."""
    images_tensor = test_features["images"]
    z_blocks_tensor = test_features["z_blocks"]
    heat_tensor = test_features["heat"]
    valid_tensor = test_features["valid"]
    H_feat = test_features["H_feat"]
    W_feat = test_features["W_feat"]
    target_size = test_features["target_size"]

    N_img = len(images_tensor)

    # 1. Gather active foreground tokens across test set
    # heat[:, :, :, g] > 1e-5 and valid
    heat_g = heat_tensor[..., g]
    active_mask = (heat_g > 1e-5) & valid_tensor

    active_indices = torch.nonzero(active_mask, as_tuple=False)
    if len(active_indices) < 8:
        print(f"[WARNING] Block C{g} has too few active test tokens ({len(active_indices)}). Loosening threshold...")
        active_mask = (heat_g > 1e-7) & valid_tensor
        active_indices = torch.nonzero(active_mask, as_tuple=False)

    if len(active_indices) == 0:
        print(f"[ERROR] Block C{g} is inactive across the entire test set.")
        return None

    # Extract z_blocks for active tokens: shape (N_act, group_size)
    z_act = z_blocks_tensor[active_mask][:, g, :].numpy()  # (N_act, group_size)
    erank = compute_erank(z_act)

    # Contribution vectors in feature space: c = z_act @ D_g -> (N_act, d)
    c = z_act @ D_g  # (N_act, 256)

    # Fit 3D PCA on c
    mean, comps = pca_fit(c, k=3)  # comps: (3, d)
    u_g = (c - mean) @ comps.T     # (N_act, 3)
    u_g = radial_clip(u_g, pct=98.0)
    colorize = make_colorize(u_g, saturation=1.35)

    # 2. Match closest 1D SAE baseline atom
    j_star, d_arrow, proj_norm = match_closest_1d_sae_atom(
        D_g=D_g, W_1D=W_1D, comps=comps, mean=mean, u_g=u_g
    )

    # 3. Find top 10 highest-activating distinct cellular regions
    top_regions = find_top_activating_regions(
        heat=heat_tensor,
        valid=valid_tensor,
        g=g,
        n_regions=10,
        min_pixel_dist=128.0,
        crop_size=crop_size,
        target_size=target_size
    )

    # Precompute per-image upsampled hue and norm maps to blend crops cleanly
    # Cache per unique image index
    unique_img_indices = sorted(list(set(r["img_idx"] for r in top_regions)))
    img_overlays = {}

    for img_idx in unique_img_indices:
        img_np = images_tensor[img_idx].permute(1, 2, 0).numpy()  # (1008, 1008, 3)
        z_img = z_blocks_tensor[img_idx, ..., g, :].numpy()       # (H_feat, W_feat, group_size)

        c_img = z_img @ D_g                                       # (H_feat, W_feat, d)
        u_img = (c_img - mean) @ comps.T                          # (H_feat, W_feat, 3)
        rgb_img = colorize(u_img)                                 # (H_feat, W_feat, 3)
        norm_img = np.linalg.norm(c_img, axis=-1)                 # (H_feat, W_feat)

        # Smooth bicubic upsample to target_size (1008x1008)
        rgb_t = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float()
        norm_t = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()

        rgb_up_full = F.interpolate(
            rgb_t, size=(target_size, target_size), mode='bicubic', align_corners=False
        ).squeeze(0).permute(1, 2, 0).numpy()

        norm_up_full = F.interpolate(
            norm_t, size=(target_size, target_size), mode='bicubic', align_corners=False
        ).squeeze(0).squeeze(0).numpy()

        img_overlays[img_idx] = {
            "img_rgb": img_np,
            "rgb_up": np.clip(rgb_up_full, 0.0, 1.0),
            "norm_up": np.maximum(norm_up_full, 0.0),
        }

    # Extract 10 blended crops
    blended_crops = []
    crop_metadata = []

    for reg in top_regions:
        img_idx = reg["img_idx"]
        y_min, y_max = reg["y_min"], reg["y_max"]
        x_min, x_max = reg["x_min"], reg["x_max"]

        cache = img_overlays[img_idx]
        crop_rgb = cache["img_rgb"][y_min:y_max, x_min:x_max]
        crop_rgb_up = cache["rgb_up"][y_min:y_max, x_min:x_max]
        crop_norm_up = cache["norm_up"][y_min:y_max, x_min:x_max]

        blended = compute_blended_crop(crop_rgb, crop_rgb_up, crop_norm_up)
        blended_crops.append(blended)
        crop_metadata.append(reg)

    print(f"[INFO] Block C{g:3d}: erank={erank:.2f}, match={sae_1d_name} C{j_star:4d} "
          f"(proj={proj_norm:.3f}), extracted {len(blended_crops)} cellular crops.")

    return {
        "block_g": g,
        "erank": erank,
        "u_g": u_g,
        "colorize": colorize,
        "d_arrow": d_arrow,
        "j_star": j_star,
        "sae_1d_name": sae_1d_name,
        "proj_norm": proj_norm,
        "blended_crops": blended_crops,
        "crop_metadata": crop_metadata,
    }


# ==============================================================================
# 8. Main CLI & Figure Rendering Orchestration
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Goodfire BSF Paper Figure 9 on RIVA PAP Smear Test Dataset"
    )
    parser.add_argument("--sae_type", type=str, default="GrassmannianBSF",
                        choices=["GrassmannianBSF", "VanillaBSF"],
                        help="Block-Sparse Featurizer architecture")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to trained BSF checkpoint (.pt)")
    parser.add_argument("--directional_sae_type", type=str, default="OMPSAE",
                        choices=["OMPSAE", "TopKSAE", "MPSAE"],
                        help="Directional 1D SAE architecture used for 1D atom arrow")
    parser.add_argument("--directional_sae_checkpoint", type=str, default="",
                        help="Path to trained 1D SAE checkpoint (.pt)")
    parser.add_argument("--weight_checkpoint", type=str,
                        default="./weights/rcnn_best_anchors/best_checkpoint.pth",
                        help="Path to SAM3 Faster R-CNN checkpoint (.pth)")
    parser.add_argument("--base_image_dir", type=str, default="",
                        help="Base directory containing RIVA dataset images (auto-discovered if empty)")
    parser.add_argument("--concept_ids", type=int, nargs="+",
                        default=[139, 92, 75, 214, 89, 12],
                        help="Verified concept block IDs to visualize in Figure 9")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Pixel size of each cellular ROI crop (default: 256x256)")
    parser.add_argument("--k", type=int, default=16,
                        help="Sparsity top-k parameter for BSF")
    parser.add_argument("--expansion_factor", type=float, default=4.0,
                        help="Hidden dimension expansion factor (e.g. 4.0 -> 1024)")
    parser.add_argument("--block_size", type=int, default=4,
                        help="Group size per block (default: 4)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size")
    parser.add_argument("--save_dir", type=str,
                        default="./interpretability_analysis/paper_fig9_visualizations",
                        help="Directory to save generated Figure 9 visualizations")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output image resolution DPI (default: 300)")
    parser.add_argument("--max_test_images", type=int, default=0,
                        help="Optional limit on number of test images to process (0 = all)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve RIVA image directory and test split
    image_dir = find_image_directory(args.base_image_dir)
    print(f"[INFO] Using RIVA image directory: {image_dir}")
    _, _, test_paths = get_data_splits(image_dir)

    if not test_paths:
        test_folder = Path(image_dir) / "test"
        if test_folder.is_dir():
            test_paths = [str(p) for ext in ('*.jpg', '*.png') for p in test_folder.glob(ext)]
        if not test_paths:
            raise FileNotFoundError(f"No test images found in {image_dir}.")

    if args.max_test_images > 0:
        test_paths = test_paths[:args.max_test_images]

    print(f"[INFO] Test dataset size: {len(test_paths)} images")

    # 2. Load SAM3 Faster R-CNN backbone
    sam3_model = load_sam3_model(args.weight_checkpoint, device)

    # 3. Load BSF model
    input_dim = 256
    bsf_model, n_groups = build_and_load_bsf_model(
        sae_type=args.sae_type,
        checkpoint_path=args.checkpoint,
        device=device,
        input_dim=input_dim,
        expansion_factor=args.expansion_factor,
        k=args.k,
        block_size=args.block_size
    )

    # 4. Load 1D Directional SAE dictionary for baseline arrow comparison
    W_1D, sae_1d_name = load_directional_sae_dictionary(
        directional_sae_type=args.directional_sae_type,
        directional_sae_checkpoint=args.directional_sae_checkpoint,
        device=device,
        input_dim=input_dim,
        expansion_factor=args.expansion_factor,
        k=args.k
    )
    print(f"[INFO] 1D SAE dictionary loaded: {W_1D.shape} ({sae_1d_name})")

    # 5. Extract decoder atoms for blocks
    # atoms shape: (n_groups, group_size, d) -> (256, 4, 256)
    atoms_tensor = bsf_model.atoms().detach().cpu().numpy()

    # 6. Extract test features & activations
    test_features = extract_test_features(
        sam3_model=sam3_model,
        bsf_model=bsf_model,
        test_paths=test_paths,
        device=device,
        batch_size=args.batch_size,
        target_size=1008
    )

    # 7. Process each concept block
    processed_blocks = []
    summary_records = []

    for g in args.concept_ids:
        if g < 0 or g >= n_groups:
            print(f"[WARNING] Concept ID {g} out of range [0, {n_groups - 1}]. Skipping.")
            continue

        D_g = atoms_tensor[g]  # (group_size, d)
        block_info = process_block(
            g=g,
            test_features=test_features,
            D_g=D_g,
            W_1D=W_1D,
            sae_1d_name=sae_1d_name,
            crop_size=args.crop_size
        )

        if block_info is None:
            continue

        processed_blocks.append(block_info)
        summary_records.append({
            "block_id": g,
            "description": CONCEPT_DESCRIPTIONS.get(g, "Biological Concept"),
            "erank": block_info["erank"],
            "matched_1d_sae": sae_1d_name,
            "matched_1d_atom": block_info["j_star"],
            "projection_norm": block_info["proj_norm"],
            "n_crops": len(block_info["blended_crops"]),
        })

        # Save Individual High-Res Panel per Block
        single_fig_path = save_dir / f"{args.sae_type}_block_C{g}_fig9_panel.png"
        fig_single = plt.figure(figsize=(16, 6.5), dpi=args.dpi)
        gs_single = gridspec.GridSpec(1, 1, figure=fig_single, left=0.04, right=0.96, top=0.92, bottom=0.06)

        render_block_panel_to_subgridspec(fig_single, gs_single[0], block_info, crop_size=args.crop_size)

        fig_single.suptitle(
            f"BSF Manifold Geometry & Cellular Grounding - Block C{g} "
            f"({CONCEPT_DESCRIPTIONS.get(g, '')})",
            fontsize=13, fontweight='bold', y=0.98
        )
        fig_single.savefig(str(single_fig_path), dpi=args.dpi, bbox_inches='tight')
        plt.close(fig_single)
        print(f"[SAVED] Individual panel saved to: {single_fig_path}")

    # 8. Render Combined Publication Master Figure (Exact 3 rows x 2 columns Paper Layout)
    n_total = len(processed_blocks)
    if n_total > 0:
        if n_total <= 2:
            n_rows = 1
            n_cols = n_total
            figsize = (12 * n_cols, 6.5)
        elif n_total <= 4:
            n_rows = 2
            n_cols = 2
            figsize = (24, 11)
        else:
            n_rows = 3
            n_cols = 2
            figsize = (24, 16)

        combined_fig_path = save_dir / f"{args.sae_type}_Figure9_6blocks_grid.png"

        # Dimensions: scalable width and height for 300 DPI publication crispness
        fig_comb = plt.figure(figsize=figsize, dpi=args.dpi)

        # Master GridSpec
        gs_master = gridspec.GridSpec(
            n_rows, n_cols, figure=fig_comb,
            left=0.03, right=0.97, top=0.91, bottom=0.04,
            wspace=0.10, hspace=0.22
        )

        for idx in range(min(n_total, n_rows * n_cols)):
            block_info = processed_blocks[idx]
            r_idx = idx // n_cols
            c_idx = idx % n_cols
            render_block_panel_to_subgridspec(
                fig=fig_comb,
                gs_parent=gs_master[r_idx, c_idx],
                block_info=block_info,
                crop_size=args.crop_size
            )

        # Publication Caption-Style Header
        fig_comb.suptitle(
            "Figure 9: A block carries a concept together with its intrinsic geometry (RIVA PAP Smear Test Set)\n"
            f"Architecture: {args.sae_type} (b=4, k={args.k})  |  Baseline: Closest Directional 1D Atom from {sae_1d_name} (b=1)\n"
            "Left: Top 10 cellular regions (256x256 crops) with luminance-preserving manifold RGB overlay  |  "
            "Right: 3D concept geometry cloud u_g, floor shadow projection, and 1D SAE baseline arrow",
            fontsize=13.5, fontweight='bold', y=0.97, ha='center'
        )

        fig_comb.savefig(str(combined_fig_path), dpi=args.dpi, bbox_inches='tight')
        plt.close(fig_comb)
        print(f"[SAVED] Combined 6-block master publication figure saved to: {combined_fig_path}")

    # 9. Save JSON Summary
    summary_path = save_dir / "paper_fig9_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "sae_type": args.sae_type,
            "directional_sae_type": sae_1d_name,
            "n_test_images": len(test_paths),
            "concept_blocks": summary_records,
        }, f, indent=2)
    print(f"[SAVED] Summary metrics saved to: {summary_path}")
    print("[SUCCESS] BSF Paper Figure 9 visualization completed successfully!")


if __name__ == "__main__":
    main()
