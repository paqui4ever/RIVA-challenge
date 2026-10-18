"""
Bethesda Concept Attribution Pipeline
======================================
Links learned Sparse Autoencoder (SAE) and Block-Sparse Featurizer (BSF) concepts
directly to ground-truth Bethesda clinical cell annotations in the RIVA PAP smear dataset.

Computes:
1. Ground-truth cell coordinate mapping & 3x3 FPN spatial max-pooling.
2. Background / exudate token sampling (BG_EXUDATE).
3. Class-conditional mean activations, fire rates, selectivity ratios, and AUROCs
   for each Bethesda class (NILM, ASC-US, LSIL, HSIL, SCC, ASC-H, ENDOCERVICAL, INFL)
   and grouped categories (HIGH_GRADE, LOW_GRADE, NORMAL).
4. Cell-vs-Background enrichment ratio separating epithelial concepts from exudates.
5. 4D Subspace Linear Probing for BSFs (GrassmannianBSF & VanillaBSF).
6. Publication-quality clustered heatmaps, top-cell crop grids, and dysplasia manifold projections.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TVF
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root and interpretability_analysis to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square
from train_saes import get_data_splits
from overcomplete import TopKSAE
from custom_saes import (
    RATopKSAE,
    OMPSAE,
    MpSAE as MPSAE,
    BSF,
    VanillaBSF,
    GrassmannianBSF,
)

# Standardized Bethesda Class Names & Mappings
BETHESDA_ALIASES = {
    "NILM": "NILM",
    "ASCUS": "ASC-US",
    "ASC-US": "ASC-US",
    "LSIL": "LSIL",
    "HSIL": "HSIL",
    "SCC": "SCC",
    "ASCH": "ASC-H",
    "ASC-H": "ASC-H",
    "ENDO": "ENDOCERVICAL",
    "ENDOCERVICAL": "ENDOCERVICAL",
    "INFL": "INFL",
    "INFLAMMATORY": "INFL",
}

CLASS_ID_TO_CANONICAL = {
    0: "NILM",
    1: "ENDOCERVICAL",
    2: "INFL",
    3: "ASC-US",
    4: "LSIL",
    5: "HSIL",
    6: "ASC-H",
    7: "SCC",
}

PRIMARY_CLINICAL_CLASSES = [
    "NILM",
    "ASC-US",
    "LSIL",
    "HSIL",
    "SCC",
    "ASC-H",
    "ENDOCERVICAL",
    "INFL",
]

HIGH_GRADE_SET = {"HSIL", "SCC", "ASC-H"}
LOW_GRADE_SET = {"LSIL", "ASC-US"}
NORMAL_SET = {"NILM"}

BETHESDA_COLORS = {
    "NILM": "#2ca02c",         # Forest Green
    "ASC-US": "#d4b106",       # Mustard Gold
    "LSIL": "#ff7f0e",         # Orange
    "HSIL": "#d62728",         # Red
    "SCC": "#7b1fa2",          # Deep Purple
    "ASC-H": "#e377c2",        # Magenta-Pink
    "ENDOCERVICAL": "#17becf", # Cyan-Teal
    "INFL": "#7f7f7f",         # Slate Gray
    "BG_EXUDATE": "#8c564b",   # Brownish Slate
    "HIGH_GRADE": "#d62728",
    "LOW_GRADE": "#ff7f0e",
    "NORMAL": "#2ca02c",
}


def safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Pearson correlation safely with zero-variance protection."""
    if len(x) < 3:
        return 0.0
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def compute_auroc_safe(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates AUROC safely, returning 0.5 if single-class or flat."""
    n_pos = int(np.sum(y_true_binary == 1))
    n_neg = int(np.sum(y_true_binary == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true_binary, y_score))
    except Exception:
        # Fallback rank-based Wilcoxon-Mann-Whitney statistic
        ranks = np.argsort(np.argsort(y_score)) + 1
        pos_ranks = ranks[y_true_binary == 1]
        u = float(np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0)
        return float(np.clip(u / (n_pos * n_neg), 0.0, 1.0))


# ==============================================================================
# Annotation & Dataset Auto-Discovery
# ==============================================================================

def find_annotations_files(annotations_path: str = "", base_dir: str = "") -> List[str]:
    """Auto-discovers ground-truth annotation CSV/JSON files in the repository."""
    found_files = []
    if annotations_path:
        p = Path(annotations_path)
        if p.is_file():
            return [str(p)]
        if p.is_dir():
            found_files.extend([str(f) for f in p.glob("*.csv") if "sample" not in f.name.lower()])
            if found_files:
                return found_files

    search_roots = []
    if base_dir and os.path.exists(base_dir):
        b = Path(base_dir)
        search_roots.extend([b, b.parent, b.parent.parent])

    search_roots.extend([
        Path("./RIVA/annotations/annotations"),
        Path("./RIVA/annotations"),
        Path("./RIVA final/annotations/annotations"),
        Path("./RIVA final/annotations"),
        Path("./data/RIVA/annotations"),
        Path("./data/annotations"),
        Path("./annotations"),
        Path("./data"),
        Path("/local_data/RIVA/annotations/annotations"),
        Path("/local_data/RIVA/annotations"),
    ])

    for root in search_roots:
        if root.is_dir():
            for name in ["val.csv", "train.csv", "annotations.csv"]:
                f = root / name
                if f.is_file() and str(f) not in found_files:
                    found_files.append(str(f))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in found_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def load_all_annotations(annotation_files: List[str]) -> Tuple[Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    """
    Loads bounding boxes and cell coordinates from discovered annotation files.
    Standardizes Bethesda labels to canonical classes.
    """
    all_dfs = []
    for fpath in annotation_files:
        try:
            df = pd.read_csv(fpath)
            # Ensure required columns exist
            if "image_filename" in df.columns and "x" in df.columns and "y" in df.columns:
                all_dfs.append(df)
                print(f"[INFO] Loaded annotations from {fpath} ({len(df)} cells)")
        except Exception as e:
            print(f"[WARNING] Could not read {fpath}: {e}")

    if not all_dfs:
        raise FileNotFoundError(
            "No ground-truth annotation CSV files could be found. "
            "Please specify --annotations_path pointing to train.csv or val.csv."
        )

    combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["image_filename", "x", "y"])

    # Resolve class names
    canonical_labels = []
    for _, row in combined_df.iterrows():
        raw_name = str(row.get("class_name", "")).strip().upper()
        if raw_name in BETHESDA_ALIASES:
            canonical_labels.append(BETHESDA_ALIASES[raw_name])
        elif "class" in row and pd.notna(row["class"]):
            cid = int(row["class"])
            canonical_labels.append(CLASS_ID_TO_CANONICAL.get(cid, f"CLASS_{cid}"))
        else:
            canonical_labels.append(raw_name if raw_name else "UNKNOWN")

    combined_df["canonical_class"] = canonical_labels

    annotations_by_img: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in combined_df.iterrows():
        fname = str(row["image_filename"]).strip()
        cell_dict = {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "width": float(row.get("width", 100.0)),
            "height": float(row.get("height", 100.0)),
            "class_name": str(row["canonical_class"]),
            "orig_class": row.get("class", -1),
        }
        if fname not in annotations_by_img:
            annotations_by_img[fname] = []
        annotations_by_img[fname].append(cell_dict)

    print(f"[INFO] Total indexed annotated images: {len(annotations_by_img)}, total cells: {len(combined_df)}")
    print("[INFO] Class distribution across loaded annotations:")
    print(combined_df["canonical_class"].value_counts().to_string())

    return annotations_by_img, combined_df


def find_image_directory(base_image_dir: str = "") -> str:
    """Auto-discovers the RIVA images root directory containing train/val/test subfolders."""
    if base_image_dir and os.path.isdir(base_image_dir):
        return base_image_dir

    search_dirs = [
        "./RIVA/images/images",
        "./RIVA/images",
        "./RIVA final/images/images",
        "./data/RIVA/images/images",
        "./data/images",
        "/local_data/RIVA/images/images",
        "/local_data/RIVA/images",
        "./RIVA",
    ]

    for d in search_dirs:
        p = Path(d)
        if p.is_dir() and ((p / "val").is_dir() or (p / "train").is_dir()):
            return str(p.resolve())

    # Fallback to local RIVA if exists
    if os.path.isdir("./RIVA"):
        return str(Path("./RIVA").resolve())

    raise FileNotFoundError(
        "Could not auto-discover RIVA image directory with train/val folders. "
        "Please provide --base_image_dir."
    )


class AnnotatedImageDataset(Dataset):
    """
    Dataset that loads original images, applies SAM3 resize+pad,
    and returns image tensor along with image path and original dimensions.
    """
    def __init__(self, image_paths: List[str], target_size: int = 1008):
        self.image_paths = image_paths
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, int]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size
        img_t = TVF.to_tensor(img)
        img_padded, _, _ = sam3_resize_longest_side_and_pad_square(img_t, target_size=self.target_size)
        return img_padded, img_path, h_orig, w_orig


# ==============================================================================
# Model Loading Helpers
# ==============================================================================

def load_sam3_rcnn(weight_checkpoint: str, device: torch.device) -> nn.Module:
    """Instantiates SAM3 Faster R-CNN and loads checkpoint weights."""
    print("[INFO] Building SAM3 Faster R-CNN backbone...")
    sam3_model = build_sam3_fasterrcnn(trainable_backbone=False)

    ckpt_to_load = weight_checkpoint
    if not (ckpt_to_load and os.path.exists(ckpt_to_load)):
        weight_dir = Path("./weights")
        weight_files = list(weight_dir.rglob("*.pt")) + list(weight_dir.rglob("*.pth"))
        ckpt_to_load = str(weight_files[0]) if weight_files else None

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"[INFO] Found SAM3 Faster R-CNN weights: {ckpt_to_load}. Loading...")
        state_dict = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        sam3_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARNING] Weight checkpoint {weight_checkpoint} not found. Using initialized SAM3 weights.")

    sam3_model.to(device)
    sam3_model.eval()
    return sam3_model


def build_and_load_sae(
    sae_type: str,
    checkpoint_dir: str,
    device: torch.device,
    input_dim: int = 256,
    expansion_factor: float = 4.0,
    k: int = 16,
    block_size: int = 4
) -> Tuple[nn.Module, bool, int]:
    """
    Builds the requested SAE or BSF model and loads its checkpoint if available.
    Returns (model, is_bsf, num_concepts).
    """
    hidden_dim = int(input_dim * expansion_factor)
    n_groups = max(1, hidden_dim // block_size)
    is_bsf = sae_type in ("VanillaBSF", "GrassmannianBSF")

    loaded_points = None
    if sae_type == "RATopKSAE":
        points_path = os.path.join(checkpoint_dir, "ratopk_points.pt")
        if os.path.exists(points_path):
            loaded_points = torch.load(points_path, map_location=device, weights_only=False)
        else:
            loaded_points = torch.randn(1024, input_dim, device=device)

    sae_constructors = {
        "TopKSAE": lambda: TopKSAE(input_dim, hidden_dim, top_k=k),
        "RATopKSAE": lambda: RATopKSAE(input_dim, hidden_dim, points=loaded_points.to(device), top_k=k),
        "MPSAE": lambda: MPSAE(input_dim, hidden_dim, k=k),
        "OMPSAE": lambda: OMPSAE(input_dim, hidden_dim, k=k),
        "VanillaBSF": lambda: VanillaBSF(input_dim, n_groups=n_groups, group_size=block_size, l0=k),
        "GrassmannianBSF": lambda: GrassmannianBSF(input_dim, n_groups=n_groups, group_size=block_size, l0=k),
    }

    if sae_type not in sae_constructors:
        raise ValueError(f"Unknown SAE type: {sae_type}. Supported: {list(sae_constructors.keys())}")

    model = sae_constructors[sae_type]()
    num_concepts = n_groups if is_bsf else hidden_dim

    # Check for checkpoint
    ckpt_candidates = [
        os.path.join(checkpoint_dir, f"{sae_type}_sae.pt"),
        os.path.join(checkpoint_dir, f"{sae_type}.pt"),
        f"./interpretability_analysis/checkpoints/{sae_type}_sae.pt",
        f"./interpretability_analysis/checkpoints/{sae_type}.pt",
    ]

    loaded = False
    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            print(f"[INFO] Found {sae_type} checkpoint: {ckpt}. Loading...")
            st = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(st, strict=False)
            loaded = True
            break

    if not loaded:
        print(f"[WARNING] Checkpoint for {sae_type} not found in {checkpoint_dir}. Using initialized model.")

    model.to(device)
    model.eval()
    return model, is_bsf, num_concepts


# ==============================================================================
# Cell Activation Extraction & Spatial Max-Pooling Pipeline
# ==============================================================================

def extract_cell_activations(
    sam3_model: nn.Module,
    sae_model: nn.Module,
    dataloader: DataLoader,
    annotations_by_img: Dict[str, List[Dict[str, Any]]],
    is_bsf: bool,
    num_concepts: int,
    device: torch.device,
    max_bg_tokens_per_img: int = 20
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts SAM3 FPN features, computes SAE/BSF activations, maps ground-truth
    cell coordinates to FPN grid, performs spatial max-pooling over bounding box
    footprint (min 3x3 window), and samples clean BG_EXUDATE tokens.
    """
    sam3_model.eval()
    sae_model.eval()

    activations = {}
    def hook_fn(module, input, output):
        activations["1"] = output["1"].detach()

    handle = sam3_model.backbone.register_forward_hook(hook_fn)

    all_cell_records: List[Dict[str, Any]] = []
    total_annotated_cells = 0
    total_bg_tokens = 0

    with torch.no_grad():
        for batch_imgs, batch_paths, batch_h, batch_w in tqdm(dataloader, desc="Extracting cell concept activations"):
            B = batch_imgs.shape[0]
            batch_dev = batch_imgs.to(device)

            # 1. Forward SAM3 backbone
            _ = sam3_model.backbone(batch_dev)
            feat = activations["1"]  # (B, C, H_feat, W_feat)
            B, C, H_feat, W_feat = feat.shape

            # Flatten to pass through SAE
            feat_flat = feat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

            # 2. Forward SAE/BSF
            if is_bsf:
                z_blk = sae_model.encode_blocks(feat_flat)  # (B*P, n_groups, group_size)
                z_norm = z_blk.norm(dim=-1)                 # (B*P, n_groups)
                z_spatial = z_norm.view(B, H_feat, W_feat, num_concepts)
                group_size = z_blk.shape[-1]
                z_blocks_spatial = z_blk.view(B, H_feat, W_feat, num_concepts, group_size)
            else:
                _, z, _ = sae_model(feat_flat)
                z_spatial = z.view(B, H_feat, W_feat, num_concepts)
                z_blocks_spatial = None
                group_size = None

            # Move activations to CPU for pooling
            z_spatial_cpu = z_spatial.cpu()
            if is_bsf:
                z_blocks_spatial_cpu = z_blocks_spatial.cpu()

            # 3. Process each image in batch
            for b in range(B):
                img_path = batch_paths[b]
                img_filename = os.path.basename(img_path)
                h_orig = int(batch_h[b])
                w_orig = int(batch_w[b])

                scale = 1008.0 / float(max(h_orig, w_orig))
                new_h = int(round(h_orig * scale))
                new_w = int(round(w_orig * scale))

                # Lookup annotations
                gt_cells = annotations_by_img.get(img_filename, [])
                if not gt_cells:
                    # Try matching filename without extension or case-insensitive
                    base_no_ext = os.path.splitext(img_filename)[0].lower()
                    for k, v in annotations_by_img.items():
                        if os.path.splitext(k)[0].lower() == base_no_ext:
                            gt_cells = v
                            break

                cell_grid_centers = []

                # A. Process Ground-Truth Cells
                for cell in gt_cells:
                    cx_orig, cy_orig = cell["x"], cell["y"]
                    w_cell_orig = cell.get("width", 100.0)
                    h_cell_orig = cell.get("height", 100.0)
                    cls_name = cell["class_name"]

                    # Map coordinates to 1008x1008 padded image
                    cx_pad = cx_orig * scale
                    cy_pad = cy_orig * scale
                    w_pad = w_cell_orig * scale
                    h_pad = h_cell_orig * scale

                    x1_pad = cx_pad - w_pad / 2.0
                    x2_pad = cx_pad + w_pad / 2.0
                    y1_pad = cy_pad - h_pad / 2.0
                    y2_pad = cy_pad + h_pad / 2.0

                    # Map to FPN spatial grid coordinates
                    r_grid = int(np.clip(cy_pad * (H_feat / 1008.0), 0, H_feat - 1))
                    c_grid = int(np.clip(cx_pad * (W_feat / 1008.0), 0, W_feat - 1))
                    cell_grid_centers.append((r_grid, c_grid))

                    # Footprint bounds
                    r_min = int(np.floor(y1_pad * (H_feat / 1008.0)))
                    r_max = int(np.ceil(y2_pad * (H_feat / 1008.0)))
                    c_min = int(np.floor(x1_pad * (W_feat / 1008.0)))
                    c_max = int(np.ceil(x2_pad * (W_feat / 1008.0)))

                    # Minimum 3x3 window centered at (r_grid, c_grid)
                    r_start = max(0, min(r_min, r_grid - 1))
                    r_end = min(H_feat, max(r_max + 1, r_grid + 2))
                    c_start = max(0, min(c_min, c_grid - 1))
                    c_end = min(W_feat, max(c_max + 1, c_grid + 2))

                    # Window slice
                    sub_z = z_spatial_cpu[b, r_start:r_end, c_start:c_end, :]  # (H_w, W_w, num_concepts)
                    # Spatial max-pooling across the cell footprint
                    pooled_z = sub_z.amax(dim=(0, 1)).numpy()  # (num_concepts,)

                    # For BSF: extract the 4D subspace coordinate vector at peak activation
                    subspace_coords = None
                    if is_bsf:
                        sub_z_flat = sub_z.view(-1, num_concepts)
                        max_indices = sub_z_flat.argmax(dim=0)  # (num_concepts,)
                        sub_blk_flat = z_blocks_spatial_cpu[b, r_start:r_end, c_start:c_end, :, :].view(-1, num_concepts, group_size)
                        cell_z_block = sub_blk_flat[max_indices, torch.arange(num_concepts), :].numpy()  # (num_concepts, group_size)
                        subspace_coords = cell_z_block

                    record = {
                        "image_path": img_path,
                        "image_filename": img_filename,
                        "class_name": cls_name,
                        "is_cell": True,
                        "cx_pad": float(cx_pad),
                        "cy_pad": float(cy_pad),
                        "x1_pad": float(x1_pad),
                        "y1_pad": float(y1_pad),
                        "x2_pad": float(x2_pad),
                        "y2_pad": float(y2_pad),
                        "r_grid": r_grid,
                        "c_grid": c_grid,
                        "pooled_activation": pooled_z,
                        "subspace_coords": subspace_coords,
                    }
                    all_cell_records.append(record)
                    total_annotated_cells += 1

                # B. Sample Non-Cell Background / Exudate Tokens (BG_EXUDATE)
                # Compute token-level mean RGB
                img_tensor_b = batch_dev[b]
                mean_rgb = img_tensor_b.mean(dim=0, keepdim=True).unsqueeze(0)
                mean_rgb_pooled = F.adaptive_avg_pool2d(mean_rgb, (H_feat, W_feat)).squeeze().cpu().numpy()

                stride = 1008.0 / H_feat
                valid_bg_mask = np.zeros((H_feat, W_feat), dtype=bool)

                for r in range(1, H_feat - 1):
                    for c in range(1, W_feat - 1):
                        # Within valid unpadded image boundaries
                        if (r + 0.5) * stride < new_h and (c + 0.5) * stride < new_w:
                            if mean_rgb_pooled[r, c] > 0.08:
                                # Distance constraint: at least 3 FPN grid cells from any cell center
                                is_far = True
                                for (cr, cc) in cell_grid_centers:
                                    if max(abs(r - cr), abs(c - cc)) < 3:
                                        is_far = False
                                        break
                                if is_far:
                                    valid_bg_mask[r, c] = True

                bg_r_idx, bg_c_idx = np.where(valid_bg_mask)
                if len(bg_r_idx) > 0:
                    n_sample = min(len(bg_r_idx), max_bg_tokens_per_img)
                    sel_indices = np.random.choice(len(bg_r_idx), size=n_sample, replace=False)
                    for s_i in sel_indices:
                        r_bg = bg_r_idx[s_i]
                        c_bg = bg_c_idx[s_i]
                        bg_act = z_spatial_cpu[b, r_bg, c_bg, :].numpy()

                        bg_subspace = None
                        if is_bsf:
                            bg_subspace = z_blocks_spatial_cpu[b, r_bg, c_bg, :, :].numpy()

                        cx_bg = (c_bg + 0.5) * stride
                        cy_bg = (r_bg + 0.5) * stride

                        record = {
                            "image_path": img_path,
                            "image_filename": img_filename,
                            "class_name": "BG_EXUDATE",
                            "is_cell": False,
                            "cx_pad": float(cx_bg),
                            "cy_pad": float(cy_bg),
                            "x1_pad": float(cx_bg - 16),
                            "y1_pad": float(cy_bg - 16),
                            "x2_pad": float(cx_bg + 16),
                            "y2_pad": float(cy_bg + 16),
                            "r_grid": int(r_bg),
                            "c_grid": int(c_bg),
                            "pooled_activation": bg_act,
                            "subspace_coords": bg_subspace,
                        }
                        all_cell_records.append(record)
                        total_bg_tokens += 1

    handle.remove()

    meta = {
        "num_concepts": num_concepts,
        "is_bsf": is_bsf,
        "total_annotated_cells": total_annotated_cells,
        "total_bg_tokens": total_bg_tokens,
        "H_feat": H_feat,
        "W_feat": W_feat,
    }
    print(f"[INFO] Extracted {total_annotated_cells} annotated cells and {total_bg_tokens} BG_EXUDATE tokens.")
    return all_cell_records, meta


# ==============================================================================
# Quantitative Bethesda Class Attribution Metrics
# ==============================================================================

def compute_bethesda_enrichment_metrics(
    cell_records: List[Dict[str, Any]],
    num_concepts: int,
    save_path: str
) -> pd.DataFrame:
    """
    Computes class-conditional mean activation, fire rate, selectivity ratio,
    cell-vs-background ratio, and one-vs-rest AUROC / point-biserial correlation
    for each concept across all Bethesda clinical classes and grouped categories.
    """
    N = len(cell_records)
    Z_all = np.vstack([rec["pooled_activation"] for rec in cell_records])  # (N, num_concepts)
    labels = np.array([rec["class_name"] for rec in cell_records])
    is_cell = np.array([rec["is_cell"] for rec in cell_records], dtype=bool)

    annotated_indices = np.where(is_cell)[0]
    bg_indices = np.where(~is_cell)[0]

    unique_classes = [c for c in PRIMARY_CLINICAL_CLASSES if np.sum(labels[is_cell] == c) > 0]
    if "BG_EXUDATE" not in unique_classes:
        unique_classes.append("BG_EXUDATE")

    rows = []

    for j in range(num_concepts):
        z_j = Z_all[:, j]
        row_dict = {"concept_id": j}

        # 1. Global Cell vs Background ratio
        mean_cells = float(np.mean(z_j[annotated_indices])) if len(annotated_indices) > 0 else 0.0
        mean_bg = float(np.mean(z_j[bg_indices])) if len(bg_indices) > 0 else 0.0
        cell_vs_bg_ratio = (mean_cells + 1e-3) / (mean_bg + 1e-3)
        row_dict["mean_act_all_cells"] = mean_cells
        row_dict["mean_act_bg_exudate"] = mean_bg
        row_dict["cell_vs_bg_ratio"] = cell_vs_bg_ratio

        # 2. Per-Bethesda Class metrics
        max_selectivity = 0.0
        top_class = "UNKNOWN"

        for c in unique_classes:
            if c == "BG_EXUDATE":
                mask_c = (labels == c)
                mask_not_c = is_cell
            else:
                mask_c = (labels == c) & is_cell
                mask_not_c = (labels != c) & is_cell

            n_c = int(np.sum(mask_c))
            n_not_c = int(np.sum(mask_not_c))

            mean_c = float(np.mean(z_j[mask_c])) if n_c > 0 else 0.0
            mean_not_c = float(np.mean(z_j[mask_not_c])) if n_not_c > 0 else 0.0
            fire_rate_c = float(np.mean(z_j[mask_c] > 1e-5)) if n_c > 0 else 0.0
            selectivity_c = (mean_c + 1e-3) / (mean_not_c + 1e-3)

            # AUROC & Point-Biserial correlation on annotated cells (or vs background for BG_EXUDATE)
            eval_mask = (mask_c | mask_not_c)
            y_eval = (labels[eval_mask] == c).astype(np.int32)
            z_eval = z_j[eval_mask]

            auroc_c = compute_auroc_safe(y_eval, z_eval)
            r_pb_c = safe_pearson_corr(z_eval, y_eval)

            row_dict[f"mean_act_{c}"] = mean_c
            row_dict[f"fire_rate_{c}"] = fire_rate_c
            row_dict[f"selectivity_{c}"] = selectivity_c
            row_dict[f"auroc_{c}"] = auroc_c
            row_dict[f"r_pb_{c}"] = r_pb_c

            if c != "BG_EXUDATE" and selectivity_c > max_selectivity:
                max_selectivity = selectivity_c
                top_class = c

        # 3. Grouped Dysplasia Categories
        # HIGH_GRADE = HSIL + SCC + ASC-H
        hg_mask = np.isin(labels, list(HIGH_GRADE_SET)) & is_cell
        not_hg_mask = (~np.isin(labels, list(HIGH_GRADE_SET))) & is_cell
        mean_hg = float(np.mean(z_j[hg_mask])) if hg_mask.any() else 0.0
        mean_not_hg = float(np.mean(z_j[not_hg_mask])) if not_hg_mask.any() else 0.0
        sel_hg = (mean_hg + 1e-3) / (mean_not_hg + 1e-3)
        y_hg = hg_mask[is_cell].astype(np.int32)
        auroc_hg = compute_auroc_safe(y_hg, z_j[is_cell])
        row_dict["mean_act_HIGH_GRADE"] = mean_hg
        row_dict["fire_rate_HIGH_GRADE"] = float(np.mean(z_j[hg_mask] > 1e-5)) if hg_mask.any() else 0.0
        row_dict["selectivity_HIGH_GRADE"] = sel_hg
        row_dict["auroc_HIGH_GRADE"] = auroc_hg

        # LOW_GRADE = LSIL + ASC-US
        lg_mask = np.isin(labels, list(LOW_GRADE_SET)) & is_cell
        not_lg_mask = (~np.isin(labels, list(LOW_GRADE_SET))) & is_cell
        mean_lg = float(np.mean(z_j[lg_mask])) if lg_mask.any() else 0.0
        mean_not_lg = float(np.mean(z_j[not_lg_mask])) if not_lg_mask.any() else 0.0
        sel_lg = (mean_lg + 1e-3) / (mean_not_lg + 1e-3)
        y_lg = lg_mask[is_cell].astype(np.int32)
        auroc_lg = compute_auroc_safe(y_lg, z_j[is_cell])
        row_dict["mean_act_LOW_GRADE"] = mean_lg
        row_dict["fire_rate_LOW_GRADE"] = float(np.mean(z_j[lg_mask] > 1e-5)) if lg_mask.any() else 0.0
        row_dict["selectivity_LOW_GRADE"] = sel_lg
        row_dict["auroc_LOW_GRADE"] = auroc_lg

        # NORMAL = NILM
        norm_mask = (labels == "NILM") & is_cell
        not_norm_mask = (labels != "NILM") & is_cell
        mean_norm = float(np.mean(z_j[norm_mask])) if norm_mask.any() else 0.0
        mean_not_norm = float(np.mean(z_j[not_norm_mask])) if not_norm_mask.any() else 0.0
        sel_norm = (mean_norm + 1e-3) / (mean_not_norm + 1e-3)
        y_norm = norm_mask[is_cell].astype(np.int32)
        auroc_norm = compute_auroc_safe(y_norm, z_j[is_cell])
        row_dict["mean_act_NORMAL"] = mean_norm
        row_dict["fire_rate_NORMAL"] = float(np.mean(z_j[norm_mask] > 1e-5)) if norm_mask.any() else 0.0
        row_dict["selectivity_NORMAL"] = sel_norm
        row_dict["auroc_NORMAL"] = auroc_norm

        row_dict["top_bethesda_class"] = top_class
        row_dict["max_selectivity"] = max_selectivity
        rows.append(row_dict)

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(save_path, index=False)
    print(f"[INFO] Saved Bethesda class attribution metrics to {save_path}")
    return df_metrics


# ==============================================================================
# Subspace Linear Probing for BSFs
# ==============================================================================

def run_bsf_subspace_probe(
    cell_records: List[Dict[str, Any]],
    df_metrics: pd.DataFrame,
    save_path: str,
    top_n_blocks: int = 15
) -> pd.DataFrame:
    """
    For the top 15 cell-selective BSF blocks, fits a multinomial logistic regression
    probe on the 4D subspace coordinates of active cells to measure separation
    of NILM (Normal) vs LSIL/ASC-US (Low Grade) vs HSIL/SCC/ASC-H (High Grade).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import balanced_accuracy_score

    # Rank candidate blocks by cell_vs_bg_ratio and variance across dysplasia classes
    df_valid = df_metrics.copy()
    if "selectivity_HIGH_GRADE" in df_valid.columns and "selectivity_NORMAL" in df_valid.columns:
        dysplasia_var = (
            np.abs(df_valid["mean_act_HIGH_GRADE"] - df_valid["mean_act_NORMAL"]) +
            np.abs(df_valid["mean_act_LOW_GRADE"] - df_valid["mean_act_NORMAL"])
        )
        df_valid["probe_priority"] = df_valid["cell_vs_bg_ratio"] * (1.0 + dysplasia_var)
    else:
        df_valid["probe_priority"] = df_valid["cell_vs_bg_ratio"]

    # Select top blocks, ensuring block 139 is included if available
    top_blocks = df_valid.sort_values(by="probe_priority", ascending=False)["concept_id"].tolist()
    selected_blocks = []
    if 139 in top_blocks:
        selected_blocks.append(139)
    for b in top_blocks:
        if b not in selected_blocks:
            selected_blocks.append(b)
        if len(selected_blocks) >= top_n_blocks:
            break

    print(f"[INFO] Running BSF Subspace Probes on Top {len(selected_blocks)} Blocks: {selected_blocks}")

    probe_results = []
    for g in selected_blocks:
        # Collect active cell coordinates and classes
        coords_list = []
        class_list = []

        for rec in cell_records:
            if not rec["is_cell"]:
                continue
            act = rec["pooled_activation"][g]
            if act > 1e-5 and rec["subspace_coords"] is not None:
                cls_name = rec["class_name"]
                if cls_name in NORMAL_SET:
                    group_label = 0  # Normal
                elif cls_name in LOW_GRADE_SET:
                    group_label = 1  # Low-grade
                elif cls_name in HIGH_GRADE_SET:
                    group_label = 2  # High-grade
                else:
                    continue
                coords_list.append(rec["subspace_coords"][g])
                class_list.append(group_label)

        if len(coords_list) < 20 or len(set(class_list)) < 2:
            print(f"[WARNING] Block {g} has insufficient active cells ({len(coords_list)}) for 3-class probe.")
            continue

        X = np.array(coords_list)  # (N_act, 4)
        y = np.array(class_list)

        # Standardize features
        mean_X = X.mean(axis=0, keepdims=True)
        std_X = X.std(axis=0, keepdims=True) + 1e-6
        X_norm = (X - mean_X) / std_X

        # Stratified 5-Fold Cross Validation
        n_splits = min(5, min(np.bincount(y)))
        if n_splits < 2:
            n_splits = 2

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        acc_scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="f1_macro")

        # Fit full model to compute class centroids & Fisher criterion
        clf.fit(X_norm, y)
        centroids = [X_norm[y == c].mean(axis=0) for c in np.unique(y)]
        between_var = np.var(centroids, axis=0).sum()
        within_var = np.mean([np.var(X_norm[y == c], axis=0).sum() for c in np.unique(y)])
        fisher_ratio = float(between_var / max(1e-6, within_var))

        # Check most separable principal axis
        U, S, Vt = np.linalg.svd(X - mean_X, full_matrices=False)
        pca_coords = (X - mean_X) @ Vt.T
        axis_f_ratios = []
        for ax in range(X.shape[1]):
            c_means = [pca_coords[y == c, ax].mean() for c in np.unique(y)]
            b_var = np.var(c_means)
            w_var = np.mean([pca_coords[y == c, ax].var() for c in np.unique(y)])
            axis_f_ratios.append(float(b_var / max(1e-6, w_var)))
        best_axis = int(np.argmax(axis_f_ratios)) + 1

        probe_results.append({
            "block_id": g,
            "active_cells_count": len(X),
            "normal_cells": int(np.sum(y == 0)),
            "low_grade_cells": int(np.sum(y == 1)),
            "high_grade_cells": int(np.sum(y == 2)),
            "cv_accuracy_mean": float(np.mean(acc_scores)),
            "cv_accuracy_std": float(np.std(acc_scores)),
            "cv_macro_f1_mean": float(np.mean(f1_scores)),
            "cv_macro_f1_std": float(np.std(f1_scores)),
            "fisher_separation_ratio": fisher_ratio,
            "most_separable_subspace_axis": best_axis,
            "cell_vs_bg_ratio": float(df_metrics.loc[df_metrics["concept_id"] == g, "cell_vs_bg_ratio"].values[0]),
        })

    df_probe = pd.DataFrame(probe_results)
    df_probe.to_csv(save_path, index=False)
    print(f"[INFO] Saved BSF Subspace Probe results to {save_path}")
    return df_probe


# ==============================================================================
# Visualization 1: Clustered Bethesda Heatmap
# ==============================================================================

def plot_bethesda_heatmap(
    df_metrics: pd.DataFrame,
    sae_type: str,
    save_path: str,
    top_n_concepts: int = 25
):
    """
    Generates a publication-quality clustered heatmap where rows are Bethesda classes
    and columns are the Top 25 most class-selective concepts, showing standardized
    mean activation per class with concept IDs.
    """
    eval_classes = [c for c in PRIMARY_CLINICAL_CLASSES if f"mean_act_{c}" in df_metrics.columns]
    eval_classes.append("BG_EXUDATE")

    # Select top concepts: take best 2-3 concepts per class, then fill with highest max_selectivity
    selected_concepts = []
    for c in eval_classes:
        sel_col = f"selectivity_{c}"
        if sel_col in df_metrics.columns:
            top_c = df_metrics.sort_values(by=sel_col, ascending=False)["concept_id"].tolist()
            count = 0
            for cid in top_c:
                if cid not in selected_concepts:
                    selected_concepts.append(cid)
                    count += 1
                if count >= 3:
                    break

    # Fill remaining from top overall selectivity
    overall_top = df_metrics.sort_values(by="max_selectivity", ascending=False)["concept_id"].tolist()
    for cid in overall_top:
        if cid not in selected_concepts:
            selected_concepts.append(cid)
        if len(selected_concepts) >= top_n_concepts:
            break
    selected_concepts = selected_concepts[:top_n_concepts]

    # Build activation matrix (num_classes, num_selected_concepts)
    act_matrix = np.zeros((len(eval_classes), len(selected_concepts)), dtype=np.float32)
    for j_idx, cid in enumerate(selected_concepts):
        row = df_metrics.loc[df_metrics["concept_id"] == cid].iloc[0]
        for i_idx, c in enumerate(eval_classes):
            if c == "BG_EXUDATE":
                act_matrix[i_idx, j_idx] = row["mean_act_bg_exudate"]
            else:
                act_matrix[i_idx, j_idx] = row[f"mean_act_{c}"]

    # Standardize per concept (z-score along classes) for clear relative contrast
    means = act_matrix.mean(axis=0, keepdims=True)
    stds = act_matrix.std(axis=0, keepdims=True) + 1e-6
    z_matrix = (act_matrix - means) / stds

    fig, ax = plt.subplots(figsize=(max(14, len(selected_concepts) * 0.55), 7.5), dpi=300)
    cmap = plt.cm.RdYlBu_r

    im = ax.imshow(z_matrix, cmap=cmap, aspect="auto", vmin=-2.0, vmax=2.5)

    # Label ticks
    ax.set_xticks(np.arange(len(selected_concepts)))
    col_labels = []
    for cid in selected_concepts:
        top_cls = df_metrics.loc[df_metrics["concept_id"] == cid, "top_bethesda_class"].values[0]
        col_labels.append(f"C{cid}\n({top_cls})")
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right", fontweight="medium")

    ax.set_yticks(np.arange(len(eval_classes)))
    ax.set_yticklabels(eval_classes, fontsize=10, fontweight="bold")

    # Add text annotations
    for i in range(len(eval_classes)):
        for j in range(len(selected_concepts)):
            val = z_matrix[i, j]
            text_color = "white" if abs(val) > 1.2 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Standardized Activation (Z-score)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    plt.title(
        f"Bethesda Class Concept Attribution Clustered Heatmap: {sae_type}\n"
        f"(Top {len(selected_concepts)} Class-Selective Concepts vs. Ground-Truth Clinical Labels)",
        fontsize=12, fontweight="bold", pad=14
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Bethesda heatmap visualization to {save_path}")


# ==============================================================================
# Visualization 2: Top Concept Image Crops Per Bethesda Class
# ==============================================================================

def plot_top_concepts_per_class(
    cell_records: List[Dict[str, Any]],
    df_metrics: pd.DataFrame,
    sae_type: str,
    save_path: str,
    top_k_cells: int = 6,
    crop_size: int = 96
):
    """
    For each Bethesda class, finds the #1 most selective concept and plots a row
    of the top 6 highest-activating annotated cells (96x96 RGB crops with bounding boxes).
    """
    eval_classes = [c for c in PRIMARY_CLINICAL_CLASSES if f"selectivity_{c}" in df_metrics.columns]
    eval_classes.append("BG_EXUDATE")

    # Find #1 concept for each class
    class_top_concepts = {}
    for c in eval_classes:
        sel_col = f"selectivity_{c}"
        if sel_col in df_metrics.columns:
            # Require minimum cell vs bg ratio > 0.8 for cellular classes
            valid_df = df_metrics.copy()
            if c != "BG_EXUDATE":
                valid_df = valid_df[valid_df["cell_vs_bg_ratio"] >= 0.8]
                if len(valid_df) == 0:
                    valid_df = df_metrics
            best_concept = int(valid_df.sort_values(by=sel_col, ascending=False)["concept_id"].iloc[0])
            sel_val = float(valid_df.sort_values(by=sel_col, ascending=False)[sel_col].iloc[0])
            class_top_concepts[c] = (best_concept, sel_val)

    n_rows = len(class_top_concepts)
    fig, axes = plt.subplots(n_rows, top_k_cells, figsize=(top_k_cells * 2.5, n_rows * 2.5), dpi=250)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    half_crop = crop_size // 2

    # Cache opened images to avoid disk thrashing
    img_cache: Dict[str, np.ndarray] = {}

    for row_idx, (c, (cid, sel_val)) in enumerate(class_top_concepts.items()):
        # Filter candidate cells of class c
        if c == "BG_EXUDATE":
            matching_cells = [r for r in cell_records if not r["is_cell"]]
        else:
            matching_cells = [r for r in cell_records if r["is_cell"] and r["class_name"] == c]

        if not matching_cells:
            for col_idx in range(top_k_cells):
                axes[row_idx, col_idx].axis("off")
            axes[row_idx, 0].set_ylabel(f"{c}\nNo cells", fontsize=9, fontweight="bold")
            continue

        # Sort by activation of concept cid
        matching_cells.sort(key=lambda r: float(r["pooled_activation"][cid]), reverse=True)
        top_cells = matching_cells[:top_k_cells]

        for col_idx in range(top_k_cells):
            ax = axes[row_idx, col_idx]

            if col_idx < len(top_cells):
                cell = top_cells[col_idx]
                act_val = float(cell["pooled_activation"][cid])
                ipath = cell["image_path"]

                if ipath not in img_cache:
                    pil_img = Image.open(ipath).convert("RGB")
                    img_t = TVF.to_tensor(pil_img)
                    img_pad, _, _ = sam3_resize_longest_side_and_pad_square(img_t, target_size=1008)
                    img_cache[ipath] = img_pad.permute(1, 2, 0).numpy()

                img_np = img_cache[ipath]
                cx, cy = int(cell["cx_pad"]), int(cell["cy_pad"])

                # Crop 96x96 centered at (cx, cy)
                y1 = max(0, cy - half_crop)
                y2 = min(1008, cy + half_crop)
                x1 = max(0, cx - half_crop)
                x2 = min(1008, cx + half_crop)

                crop = img_np[y1:y2, x1:x2]
                ax.imshow(crop)

                # Draw cell bounding box / center marker
                box_color = BETHESDA_COLORS.get(c, "#00ff00")
                if cell["is_cell"]:
                    w_box = cell["x2_pad"] - cell["x1_pad"]
                    h_box = cell["y2_pad"] - cell["y1_pad"]
                    bx = cell["x1_pad"] - x1
                    by = cell["y1_pad"] - y1
                    rect = patches.Rectangle(
                        (bx, by), w_box, h_box,
                        linewidth=1.8, edgecolor=box_color, facecolor="none"
                    )
                    ax.add_patch(rect)
                else:
                    # Draw center point for background token
                    ax.plot([cx - x1], [cy - y1], marker="x", color=box_color, markersize=8, markeredgewidth=2)

                ax.set_title(f"Act: {act_val:.2f}", fontsize=8, fontweight="bold")
            else:
                ax.imshow(np.zeros((crop_size, crop_size, 3)))
                ax.set_title("N/A", fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(
                    f"{c}\nC{cid}\n({sel_val:.1f}x sel)",
                    fontsize=9, fontweight="bold", rotation=0, labelpad=48, va="center"
                )

    plt.suptitle(
        f"Top-Activating Ground-Truth Bethesda Cells per Model Concept ({sae_type})\n"
        f"(96x96 Patches Centered on Ground-Truth Coordinates)",
        fontsize=12, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved top concepts per Bethesda class visualization to {save_path}")


# ==============================================================================
# Visualization 3: Dysplasia Manifold Projection (For GrassmannianBSF)
# ==============================================================================

def plot_bsf_dysplasia_manifold_projection(
    cell_records: List[Dict[str, Any]],
    df_metrics: pd.DataFrame,
    save_path: str,
    target_blocks: Optional[List[int]] = None
):
    """
    For the top 4 epithelial-selective blocks (highest cell_vs_bg_ratio and dysplasia variance,
    explicitly including concept_id=139), plots the 2D PCA scatter of the 4D subspace
    coordinates (PC1 vs PC2) for all active annotated cells, colored by ground-truth
    Bethesda class, with class centroids marked.
    """
    # Filter candidate blocks
    if target_blocks is None:
        df_valid = df_metrics.copy()
        if "selectivity_HIGH_GRADE" in df_valid.columns and "selectivity_NORMAL" in df_valid.columns:
            dysplasia_var = (
                np.abs(df_valid["mean_act_HIGH_GRADE"] - df_valid["mean_act_NORMAL"]) +
                np.abs(df_valid["mean_act_LOW_GRADE"] - df_valid["mean_act_NORMAL"])
            )
            df_valid["manifold_score"] = df_valid["cell_vs_bg_ratio"] * (1.0 + dysplasia_var)
        else:
            df_valid["manifold_score"] = df_valid["cell_vs_bg_ratio"]

        top_blocks = df_valid.sort_values(by="manifold_score", ascending=False)["concept_id"].tolist()
        target_blocks = []
        if 139 in top_blocks:
            target_blocks.append(139)
        for b in top_blocks:
            if b not in target_blocks:
                target_blocks.append(b)
            if len(target_blocks) >= 4:
                break

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=300)
    axes = axes.flatten()

    classes_to_plot = ["NILM", "ASC-US", "LSIL", "HSIL", "SCC", "ASC-H", "ENDOCERVICAL", "INFL"]

    for idx, block_id in enumerate(target_blocks[:4]):
        ax = axes[idx]

        # Extract 4D coordinates and class labels
        coords_list = []
        labels_list = []

        for rec in cell_records:
            if not rec["is_cell"]:
                continue
            act = rec["pooled_activation"][block_id]
            if act > 1e-5 and rec["subspace_coords"] is not None:
                cls_name = rec["class_name"]
                if cls_name in classes_to_plot:
                    coords_list.append(rec["subspace_coords"][block_id])
                    labels_list.append(cls_name)

        if len(coords_list) < 10:
            ax.text(0.5, 0.5, f"Block {block_id}: Insufficient active cells ({len(coords_list)})",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        X = np.array(coords_list)  # (N_act, 4)
        labels_arr = np.array(labels_list)

        # 2D PCA via SVD
        mean_X = X.mean(axis=0, keepdims=True)
        X_centered = X - mean_X
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        coords_2d = X_centered @ Vt[:2].T  # (N_act, 2)
        var_explained = (S ** 2) / np.maximum(np.sum(S ** 2), 1e-12) * 100.0

        # Scatter plot per Bethesda class
        for c in classes_to_plot:
            c_mask = (labels_arr == c)
            if not c_mask.any():
                continue
            color = BETHESDA_COLORS.get(c, "#333333")
            ax.scatter(
                coords_2d[c_mask, 0], coords_2d[c_mask, 1],
                c=color, label=f"{c} (n={int(c_mask.sum())})",
                alpha=0.65, s=28, edgecolors="none"
            )

            # Class centroid
            centroid = coords_2d[c_mask].mean(axis=0)
            ax.scatter(
                centroid[0], centroid[1],
                c=color, s=150, marker="*",
                edgecolors="black", linewidth=1.5, zorder=5
            )
            ax.text(
                centroid[0], centroid[1] + 0.08 * (coords_2d[:, 1].max() - coords_2d[:, 1].min()),
                c, fontsize=9, fontweight="bold", ha="center", va="bottom",
                color="black", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=6
            )

        cell_bg_ratio = float(df_metrics.loc[df_metrics["concept_id"] == block_id, "cell_vs_bg_ratio"].values[0])
        ax.set_title(
            f"Block C{block_id} (Cell-vs-BG Ratio: {cell_bg_ratio:.2f}x)\n"
            f"PC1 ({var_explained[0]:.1f}% var) vs PC2 ({var_explained[1]:.1f}% var)",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Principal Axis 1 (PC1)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Principal Axis 2 (PC2)", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    plt.suptitle(
        "GrassmannianBSF 4D Subspace Dysplasia Manifold Projections\n"
        "(Continuous Epithelial Separation: NILM vs. LSIL vs. HSIL/SCC Across Subspace Axes)",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Dysplasia Manifold Projection visualization to {save_path}")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Bethesda Concept Attribution Pipeline on {device} ===")

    # 1. Output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Auto-discover annotations
    annotation_files = find_annotations_files(args.annotations_path, args.base_image_dir)
    print(f"[INFO] Discovered annotation files: {annotation_files}")
    annotations_by_img, annotations_df = load_all_annotations(annotation_files)

    # 3. Auto-discover image directory
    image_dir = find_image_directory(args.base_image_dir)
    print(f"[INFO] Discovered image root directory: {image_dir}")

    # 4. Load dataset splits
    train_paths, val_paths, test_paths = get_data_splits(image_dir)
    print(f"[INFO] Splits: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

    # Select evaluation images
    if args.split == "val":
        candidate_paths = val_paths
    elif args.split == "train":
        candidate_paths = train_paths
    elif args.split == "all":
        candidate_paths = train_paths + val_paths + test_paths
    else:  # "val_test" (default)
        candidate_paths = val_paths + test_paths

    # Filter images that exist in annotations_by_img
    annotated_eval_paths = []
    for p in candidate_paths:
        fname = os.path.basename(p)
        base_no_ext = os.path.splitext(fname)[0].lower()
        if fname in annotations_by_img or any(os.path.splitext(k)[0].lower() == base_no_ext for k in annotations_by_img):
            annotated_eval_paths.append(p)

    # If no images matched split, fallback to all images that have annotations
    if not annotated_eval_paths:
        print("[WARNING] No images in chosen split matched annotations. Searching entire image directory for annotated files...")
        all_imgs = [str(f) for ext in ("*.png", "*.jpg") for f in Path(image_dir).rglob(ext)]
        for p in all_imgs:
            fname = os.path.basename(p)
            base_no_ext = os.path.splitext(fname)[0].lower()
            if fname in annotations_by_img or any(os.path.splitext(k)[0].lower() == base_no_ext for k in annotations_by_img):
                annotated_eval_paths.append(p)

    # Deduplicate
    seen_paths = set()
    eval_image_paths = []
    for p in annotated_eval_paths:
        if p not in seen_paths:
            seen_paths.add(p)
            eval_image_paths.append(p)

    if args.max_images and len(eval_image_paths) > args.max_images:
        print(f"[INFO] Limiting evaluation to {args.max_images} images (from {len(eval_image_paths)})")
        eval_image_paths = eval_image_paths[:args.max_images]

    print(f"[INFO] Total annotated images to evaluate: {len(eval_image_paths)}")
    if not eval_image_paths:
        raise RuntimeError("No annotated images found for evaluation.")

    eval_dataset = AnnotatedImageDataset(eval_image_paths, target_size=1008)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 5. Load SAM3 Faster R-CNN Model
    sam3_model = load_sam3_rcnn(args.weight_checkpoint, device)

    # 6. Iterate through SAE types
    for sae_type in args.sae_types:
        print(f"\n{'='*70}\n[PROCESSING] Evaluating Model: {sae_type}\n{'='*70}")

        # Build SAE and load checkpoint
        sae_model, is_bsf, num_concepts = build_and_load_sae(
            sae_type=sae_type,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            input_dim=256,
            expansion_factor=args.expansion_factor,
            k=args.k,
            block_size=args.block_size
        )

        # Extract Cell Activations
        cell_records, meta = extract_cell_activations(
            sam3_model=sam3_model,
            sae_model=sae_model,
            dataloader=eval_loader,
            annotations_by_img=annotations_by_img,
            is_bsf=is_bsf,
            num_concepts=num_concepts,
            device=device
        )

        # Compute Bethesda Class Attribution Metrics
        metrics_csv_path = str(save_dir / f"bethesda_class_enrichment_{sae_type}.csv")
        df_metrics = compute_bethesda_enrichment_metrics(cell_records, num_concepts, metrics_csv_path)

        # Save per-cell activations summary
        per_cell_meta_path = str(save_dir / f"bethesda_per_cell_metadata_{sae_type}.csv")
        cell_meta_df = pd.DataFrame([
            {
                "image_filename": r["image_filename"],
                "class_name": r["class_name"],
                "is_cell": r["is_cell"],
                "cx_pad": r["cx_pad"],
                "cy_pad": r["cy_pad"],
                "r_grid": r["r_grid"],
                "c_grid": r["c_grid"],
                "top_active_concept": int(np.argmax(r["pooled_activation"])),
                "max_activation": float(np.max(r["pooled_activation"])),
            }
            for r in cell_records
        ])
        cell_meta_df.to_csv(per_cell_meta_path, index=False)
        print(f"[INFO] Saved per-cell attribution metadata to {per_cell_meta_path}")

        # Subspace Linear Probe for BSFs
        if is_bsf:
            probe_csv_path = str(save_dir / f"bsf_subspace_probe_{sae_type}.csv")
            _ = run_bsf_subspace_probe(cell_records, df_metrics, probe_csv_path, top_n_blocks=15)

        # Visualizations
        # 1. Clustered Bethesda Heatmap
        heatmap_path = str(save_dir / f"{sae_type}_bethesda_heatmap.png")
        plot_bethesda_heatmap(df_metrics, sae_type, heatmap_path, top_n_concepts=25)

        # 2. Top Concepts Per Bethesda Class Image Crops
        top_crops_path = str(save_dir / f"{sae_type}_top_concepts_per_bethesda_class.png")
        plot_top_concepts_per_class(cell_records, df_metrics, sae_type, top_crops_path, top_k_cells=6, crop_size=96)

        # 3. Dysplasia Manifold Projection (For GrassmannianBSF)
        if sae_type == "GrassmannianBSF":
            manifold_path = str(save_dir / "GrassmannianBSF_dysplasia_manifold_projection.png")
            plot_bsf_dysplasia_manifold_projection(cell_records, df_metrics, manifold_path)

    print("\n" + "="*70)
    print(f"✨ Bethesda Concept Attribution Pipeline Successfully Completed! Results saved to {save_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantitative Bethesda Clinical Class Attribution Pipeline for SAEs and BSFs on RIVA PAP Smear Dataset."
    )
    parser.add_argument(
        "--sae_types", nargs="+",
        default=["GrassmannianBSF", "OMPSAE", "TopKSAE", "MPSAE", "VanillaBSF", "RATopKSAE"],
        help="List of SAE architectures to evaluate."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./interpretability_analysis/checkpoints",
        help="Directory where trained SAE/BSF checkpoints are stored."
    )
    parser.add_argument(
        "--weight_checkpoint", type=str, default="./weights/rcnn_best_anchors/best_checkpoint.pth",
        help="Path to SAM3 Faster R-CNN weights checkpoint."
    )
    parser.add_argument(
        "--base_image_dir", type=str, default="",
        help="Root directory for RIVA images (auto-discovered if empty)."
    )
    parser.add_argument(
        "--annotations_path", type=str, default="",
        help="Path to RIVA annotation CSV file or directory (auto-discovered if empty)."
    )
    parser.add_argument(
        "--k", type=int, default=16,
        help="Sparsity target / top_k used during SAE training."
    )
    parser.add_argument(
        "--expansion_factor", type=float, default=4.0,
        help="Expansion factor used during SAE training."
    )
    parser.add_argument(
        "--block_size", type=int, default=4,
        help="Subspace block size for BSF models."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for SAM3 feature extraction."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./interpretability_analysis/bethesda_attribution",
        help="Directory to save attribution CSV metrics and visualization figures."
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Optional upper limit on number of annotated images to process."
    )
    parser.add_argument(
        "--split", type=str, default="val_test",
        choices=["val_test", "val", "all", "train"],
        help="Dataset split to evaluate on."
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
