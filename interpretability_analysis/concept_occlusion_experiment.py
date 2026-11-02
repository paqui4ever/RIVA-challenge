"""
Causal Concept Occlusion Experiment
===================================
Links learned Sparse Autoencoder (SAE) and Block-Sparse Featurizer (BSF) concepts
directly to Faster R-CNN Bethesda class predictions on Ground-Truth annotated cells
via systematic causal occlusion.

Supports both:
- --target_layer fpn      (input_dim = 256, FPN Level '1' at 144x144, linear RoI token intervention)
- --target_layer box_head (input_dim = 1024, post-TwoMLPHead features, linear cls_score intervention)

Outputs:
1. Single-Concept Causal Attribution CSV (concept_occlusion_single_{sae_type}.csv)
2. Cumulative Top-M Occlusion Curves CSV (concept_occlusion_cumulative_{sae_type}.csv)
3. Cross-Model Summary CSV (concept_occlusion_summary_all_models.csv)
4. Three Publication-Quality Visualizations:
   - {sae_type}_causal_attribution_heatmap.png
   - {sae_type}_cumulative_occlusion_curves.png
   - {sae_type}_qualitative_occlusion_panels.png
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TVF
from torchvision.ops import roi_align
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root and interpretability_analysis are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Section A: Required Imports
from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square
from train_saes import get_data_splits, extract_box_head_spatial_grid

try:
    from overcomplete import TopKSAE
except ImportError:
    try:
        from overcomplete.sae.topk_sae import TopKSAE
    except ImportError:
        class TopKSAE(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, top_k: int = 16):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.top_k = top_k
                self.encoder = nn.Linear(input_dim, hidden_dim)
                self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

            def forward(self, x: torch.Tensor):
                pre = self.encoder(x)
                topk_vals, topk_idx = torch.topk(torch.relu(pre), min(self.top_k, pre.shape[-1]), dim=-1)
                z = torch.zeros_like(pre).scatter_(-1, topk_idx, topk_vals)
                x_hat = self.decoder(z)
                return None, z, x_hat

            def get_dictionary(self) -> torch.Tensor:
                return self.decoder.weight.t()

from custom_saes import (
    RATopKSAE,
    OMPSAE,
    MpSAE as MPSAE,
    BSF,
    VanillaBSF,
    GrassmannianBSF,
)
from bethesda_concept_attribution import (
    find_annotations_files,
    load_all_annotations,
    find_image_directory,
    AnnotatedImageDataset,
    load_sam3_rcnn,
    build_and_load_sae,
    PRIMARY_CLINICAL_CLASSES,
    BETHESDA_COLORS,
    CLASS_ID_TO_CANONICAL,
)

HIGH_GRADE_SET = {"HSIL", "SCC", "ASC-H"}
LOW_GRADE_SET = {"LSIL", "ASC-US"}
NORMAL_SET = {"NILM"}

OCCLUSION_BUDGETS = [0, 1, 2, 3, 5, 8, 12, 16, 24, 32]


# ==============================================================================
# Section B: Exact Faster R-CNN Class-to-Logit Index Mapping
# ==============================================================================

def build_class_logit_mappings(
    annotations_df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds class_to_logit_idx and logit_idx_to_class mappings.
    In data/dataset.py: class_id = int(row['class']) + 1 (0 = background, 1..8 = Bethesda classes).
    Fallback uses CLASS_ID_TO_CANONICAL so that every canonical class in
    PRIMARY_CLINICAL_CLASSES maps to its exact logit index in 1..8 of cls_score ([N, 9]).
    """
    class_to_logit_idx: Dict[str, int] = {v: k + 1 for k, v in CLASS_ID_TO_CANONICAL.items()}

    if annotations_df is not None:
        for _, row in annotations_df.iterrows():
            canonical_name = row.get("canonical_class")
            orig_class = row.get("orig_class", row.get("class", None))
            if pd.notna(canonical_name) and pd.notna(orig_class):
                try:
                    c_int = int(orig_class)
                    if 0 <= c_int <= 7:
                        class_to_logit_idx[str(canonical_name)] = c_int + 1
                except (ValueError, TypeError):
                    pass

    logit_idx_to_class: Dict[int, str] = {idx: cls_name for cls_name, idx in class_to_logit_idx.items()}
    logit_idx_to_class[0] = "BACKGROUND"

    print(f"[INFO] Built Faster R-CNN logit index mapping (1..8 foreground): {class_to_logit_idx}")
    return class_to_logit_idx, logit_idx_to_class


# ==============================================================================
# Section C: Ground-Truth Cell Feature Extraction Pipeline
# ==============================================================================

def extract_ground_truth_cell_features(
    sam3_model: nn.Module,
    dataloader: DataLoader,
    annotations_by_img: Dict[str, List[Dict[str, Any]]],
    class_to_logit_idx: Dict[str, int],
    target_layer: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Extracts normalized SAM3 representations on Ground-Truth annotated cells.
    - target_layer == 'box_head':
        Extracts pooled GT cell boxes -> box_head -> x_orig of shape (N_cells, 1024).
    - target_layer == 'fpn':
        Extracts FPN Level '1' (144x144) -> roi_align (11x11, spatial_scale=H_feat/1008.0) -> roi_orig of shape (N_cells, 256, 11, 11).
    Returns cached representations and metadata for all GT cells.
    """
    sam3_model.eval()

    mean_val = getattr(sam3_model.backbone, "image_mean", [0.5, 0.5, 0.5])
    std_val = getattr(sam3_model.backbone, "image_std", [0.5, 0.5, 0.5])
    mean_t = torch.tensor(mean_val, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std_val, device=device).view(1, 3, 1, 1)

    all_cell_metadata: List[Dict[str, Any]] = []
    collected_tensors: List[torch.Tensor] = []
    global_cell_idx = 0

    with torch.no_grad():
        for batch_imgs, batch_paths, batch_h, batch_w in tqdm(dataloader, desc=f"Extracting GT cell features ({target_layer})"):
            B = batch_imgs.shape[0]
            batch_dev = batch_imgs.to(device)

            # Step 1: Normalize & Run Backbone
            norm_imgs = (batch_dev - mean_t) / std_t
            fpn_feats = sam3_model.backbone(norm_imgs)

            # Step 2: Scale Ground-Truth Cell Boxes to 1008x1008
            batch_boxes_list: List[torch.Tensor] = []
            batch_cell_records: List[List[Dict[str, Any]]] = []

            for b in range(B):
                img_path = batch_paths[b]
                img_filename = os.path.basename(img_path)
                h_orig = int(batch_h[b])
                w_orig = int(batch_w[b])
                scale = 1008.0 / float(max(h_orig, w_orig))

                gt_cells = annotations_by_img.get(img_filename, [])
                if not gt_cells:
                    base_no_ext = os.path.splitext(img_filename)[0].lower()
                    for k, v in annotations_by_img.items():
                        if os.path.splitext(k)[0].lower() == base_no_ext:
                            gt_cells = v
                            break

                cur_img_boxes: List[List[float]] = []
                cur_img_records: List[Dict[str, Any]] = []

                for cell in gt_cells:
                    cx_orig, cy_orig = float(cell["x"]), float(cell["y"])
                    w_orig_cell = float(cell.get("width", 100.0))
                    h_orig_cell = float(cell.get("height", 100.0))
                    cls_name = str(cell["class_name"])

                    cx_pad = cx_orig * scale
                    cy_pad = cy_orig * scale
                    w_pad = w_orig_cell * scale
                    h_pad = h_orig_cell * scale

                    x1_pad = max(0.0, min(1008.0, cx_pad - w_pad / 2.0))
                    y1_pad = max(0.0, min(1008.0, cy_pad - h_pad / 2.0))
                    x2_pad = max(0.0, min(1008.0, cx_pad + w_pad / 2.0))
                    y2_pad = max(0.0, min(1008.0, cy_pad + h_pad / 2.0))

                    if x2_pad <= x1_pad + 1.0:
                        x2_pad = min(1008.0, x1_pad + 4.0)
                    if y2_pad <= y1_pad + 1.0:
                        y2_pad = min(1008.0, y1_pad + 4.0)

                    cur_img_boxes.append([x1_pad, y1_pad, x2_pad, y2_pad])
                    cur_img_records.append({
                        "cell_idx": global_cell_idx,
                        "image_path": img_path,
                        "image_filename": img_filename,
                        "class_name": cls_name,
                        "orig_class": cell.get("orig_class", -1),
                        "logit_idx": class_to_logit_idx.get(cls_name, -1),
                        "x1_pad": x1_pad,
                        "y1_pad": y1_pad,
                        "x2_pad": x2_pad,
                        "y2_pad": y2_pad,
                        "cx_pad": cx_pad,
                        "cy_pad": cy_pad,
                        "w_pad": w_pad,
                        "h_pad": h_pad,
                    })
                    global_cell_idx += 1

                if cur_img_boxes:
                    batch_boxes_list.append(torch.tensor(cur_img_boxes, dtype=torch.float32, device=device))
                else:
                    batch_boxes_list.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                batch_cell_records.append(cur_img_records)

            # Step 3: Extract Features per Target Layer
            total_b_cells = sum(len(r) for r in batch_cell_records)
            if total_b_cells == 0:
                continue

            if target_layer == "box_head":
                image_shapes = [(1008, 1008) for _ in range(B)]
                pooled = sam3_model.roi_heads.box_roi_pool(fpn_feats, batch_boxes_list, image_shapes)
                x_orig = sam3_model.roi_heads.box_head(pooled)  # (N_cells, 1024)
                collected_tensors.append(x_orig.detach().cpu())
            else:  # target_layer == "fpn"
                feat_1 = fpn_feats["1"]  # (B, 256, H_feat, W_feat)
                H_feat = feat_1.shape[2]
                spatial_scale = float(H_feat) / 1008.0
                roi_orig = roi_align(
                    feat_1,
                    batch_boxes_list,
                    output_size=(11, 11),
                    spatial_scale=spatial_scale,
                    sampling_ratio=3
                )  # (N_cells, 256, 11, 11)
                collected_tensors.append(roi_orig.detach().cpu())

            for cur_img_records in batch_cell_records:
                all_cell_metadata.extend(cur_img_records)

    if not collected_tensors:
        raise RuntimeError("No ground-truth cells were extracted. Please verify annotations and dataset paths.")

    stacked_feats = torch.cat(collected_tensors, dim=0)
    print(f"[INFO] Successfully extracted {stacked_feats.shape[0]} GT cell features (shape: {tuple(stacked_feats.shape)})")

    return {
        "features": stacked_feats,
        "metadata": all_cell_metadata,
        "num_cells": stacked_feats.shape[0],
    }


# ==============================================================================
# Occlusion Mechanics & Inference
# ==============================================================================

def run_causal_occlusion_for_model(
    sae_model: nn.Module,
    is_bsf: bool,
    num_concepts: int,
    cached_data: Dict[str, Any],
    sam3_model: nn.Module,
    target_layer: str,
    class_to_logit_idx: Dict[str, int],
    logit_idx_to_class: Dict[int, str],
    device: torch.device
) -> Dict[str, Any]:
    """
    Performs full SAE reconstruction, single-concept causal occlusion, and cumulative
    occlusion over ground-truth cell features.
    """
    sae_model.eval()
    sam3_model.eval()

    cls_score = sam3_model.roi_heads.box_predictor.cls_score
    box_head = sam3_model.roi_heads.box_head

    features_cpu = cached_data["features"]
    metadata = cached_data["metadata"]
    num_cells = cached_data["num_cells"]

    # Retrieve classification weights & bias
    W_cls = cls_score.weight.detach()  # [9, 1024]
    b_cls = cls_score.bias.detach()    # [9]

    # Pre-extract dictionary / atoms
    if is_bsf:
        atoms = sae_model.atoms().detach()  # [num_concepts, group_size, input_dim]
        group_size = atoms.shape[1]
    else:
        D = sae_model.get_dictionary().detach()  # [num_concepts, input_dim]
        atoms = None
        group_size = None

    if target_layer == "box_head":
        x_orig = features_cpu.to(device)  # [N_cells, 1024]

        with torch.no_grad():
            logits_orig = cls_score(x_orig)
            probs_orig = torch.softmax(logits_orig, dim=-1)

            if is_bsf:
                z_blk = sae_model.encode_blocks(x_orig)  # [N_cells, num_concepts, group_size]
                z_act = z_blk.norm(dim=-1)               # [N_cells, num_concepts]
                x_hat_full = sae_model.decode(z_blk)     # [N_cells, 1024]
                # Contribution vector per concept: contrib_all is [N_cells, num_concepts, 1024]
                contrib_all = torch.einsum("ncg,cgd->ncd", z_blk, atoms)
            else:
                _, z, x_hat_full = sae_model(x_orig)
                z_act = z                                # [N_cells, num_concepts]
                contrib_all = z.unsqueeze(-1) * D.unsqueeze(0)  # [N_cells, num_concepts, 1024]

            logits_full = cls_score(x_hat_full)  # [N_cells, 9]
            probs_full = torch.softmax(logits_full, dim=-1)

            # Vectorized single-concept occlusion:
            # delta_logits = contrib_all @ W_cls.T of shape [N_cells, num_concepts, 9]
            delta_logits = torch.einsum("ncd,kd->nck", contrib_all, W_cls)
            logits_minus_c = logits_full.unsqueeze(1) - delta_logits  # [N_cells, num_concepts, 9]
            probs_minus_c = torch.softmax(logits_minus_c, dim=-1)

            cell_max_act = z_act.detach().cpu()
            roi_spatial_activations = None

    else:  # target_layer == "fpn"
        roi_orig = features_cpu.to(device)  # [N_cells, 256, 11, 11]

        with torch.no_grad():
            logits_orig = cls_score(box_head(roi_orig))  # [N_cells, 9]
            probs_orig = torch.softmax(logits_orig, dim=-1)

            roi_tokens = roi_orig.permute(0, 2, 3, 1).contiguous().view(num_cells * 121, 256)

            if is_bsf:
                z_blk_flat = sae_model.encode_blocks(roi_tokens)  # [N*121, num_concepts, group_size]
                z_act_flat = z_blk_flat.norm(dim=-1)              # [N*121, num_concepts]
                x_hat_flat = sae_model.decode(z_blk_flat)         # [N*121, 256]

                z_roi = z_act_flat.view(num_cells, 121, num_concepts)
                z_blk_roi = z_blk_flat.view(num_cells, 121, num_concepts, group_size)
                roi_hat_full = x_hat_flat.view(num_cells, 121, 256)
            else:
                _, z_flat, x_hat_flat = sae_model(roi_tokens)
                z_act_flat = z_flat
                z_roi = z_act_flat.view(num_cells, 121, num_concepts)
                z_blk_roi = None
                roi_hat_full = x_hat_flat.view(num_cells, 121, 256)

            roi_hat_spatial = roi_hat_full.view(num_cells, 11, 11, 256).permute(0, 3, 1, 2).contiguous()
            logits_full = cls_score(box_head(roi_hat_spatial))  # [N_cells, 9]
            probs_full = torch.softmax(logits_full, dim=-1)

            # Max concept activation across the 121 RoI positions
            cell_max_act = z_roi.max(dim=1).values.detach().cpu()  # [N_cells, num_concepts]
            roi_spatial_activations = z_roi.detach().cpu()         # [N_cells, 121, num_concepts]

            # Fast Batched Single-Concept Occlusion on FPN RoIs
            logits_minus_c = logits_full.unsqueeze(1).repeat(1, num_concepts, 1)  # [N_cells, num_concepts, 9]

            # Gather all active cell-concept pairs (z_roi.max() > 1e-6)
            active_mask = (cell_max_act > 1e-6)  # [N_cells, num_concepts]
            active_cell_indices, active_concept_indices = torch.where(active_mask)
            total_active_interventions = len(active_cell_indices)
            print(f"[INFO] Computing {total_active_interventions} active concept occlusions across {num_cells} cells...")

            interv_batch_size = 256
            for start_idx in range(0, total_active_interventions, interv_batch_size):
                end_idx = min(start_idx + interv_batch_size, total_active_interventions)
                batch_cells = active_cell_indices[start_idx:end_idx]
                batch_concepts = active_concept_indices[start_idx:end_idx]

                interv_rois = []
                for i_c, c_id in zip(batch_cells, batch_concepts):
                    cell_i = int(i_c.item())
                    concept_c = int(c_id.item())

                    if is_bsf:
                        contrib_c = z_blk_roi[cell_i, :, concept_c, :] @ atoms[concept_c]  # [121, 256]
                    else:
                        contrib_c = z_roi[cell_i, :, concept_c:concept_c+1] * D[concept_c:concept_c+1, :]

                    roi_mod = (roi_hat_full[cell_i] - contrib_c).view(11, 11, 256).permute(2, 0, 1)  # [256, 11, 11]
                    interv_rois.append(roi_mod)

                batch_interv_tensor = torch.stack(interv_rois, dim=0)  # [B_interv, 256, 11, 11]
                batch_out_logits = cls_score(box_head(batch_interv_tensor))

                for k, (i_c, c_id) in enumerate(zip(batch_cells, batch_concepts)):
                    logits_minus_c[int(i_c.item()), int(c_id.item())] = batch_out_logits[k]

            probs_minus_c = torch.softmax(logits_minus_c, dim=-1)

    return {
        "logits_orig": logits_orig.detach().cpu(),
        "probs_orig": probs_orig.detach().cpu(),
        "logits_full": logits_full.detach().cpu(),
        "probs_full": probs_full.detach().cpu(),
        "logits_minus_c": logits_minus_c.detach().cpu(),
        "probs_minus_c": probs_minus_c.detach().cpu(),
        "cell_max_act": cell_max_act,
        "roi_spatial_activations": roi_spatial_activations,
        # Cached for cumulative occlusion
        "target_layer": target_layer,
        "is_bsf": is_bsf,
        "atoms": atoms,
        "D": D if not is_bsf else None,
        "x_orig": x_orig if target_layer == "box_head" else None,
        "contrib_all": contrib_all if target_layer == "box_head" else None,
        "roi_hat_full": roi_hat_full if target_layer == "fpn" else None,
        "z_roi": z_roi if target_layer == "fpn" else None,
        "z_blk_roi": z_blk_roi if (target_layer == "fpn" and is_bsf) else None,
    }


# ==============================================================================
# Section D: Experiment 1 - Single-Concept Causal Attribution
# ==============================================================================

def compute_single_concept_causal_attribution(
    occ_res: Dict[str, Any],
    metadata: List[Dict[str, Any]],
    num_concepts: int,
    class_to_logit_idx: Dict[str, int],
    logit_idx_to_class: Dict[int, str]
) -> pd.DataFrame:
    """
    Computes delta_logit and delta_prob per concept across all GT cells for
    each Bethesda class and grouped categories (NORMAL, LOW_GRADE, HIGH_GRADE).
    Saves sorted by max_mean_logit_drop descending.
    """
    logits_full = occ_res["logits_full"].detach().cpu()          # [N, 9]
    probs_full = occ_res["probs_full"].detach().cpu()            # [N, 9]
    logits_minus_c = occ_res["logits_minus_c"].detach().cpu()    # [N, C, 9]
    probs_minus_c = occ_res["probs_minus_c"].detach().cpu()      # [N, C, 9]
    cell_max_act = occ_res["cell_max_act"].detach().cpu()        # [N, C]

    num_cells = len(metadata)
    cell_classes = [r["class_name"] for r in metadata]

    pred_full = logits_full[:, 1:9].argmax(dim=-1) + 1  # [N] in 1..8

    rows: List[Dict[str, Any]] = []

    # Map class names to cell indices
    class_cell_indices: Dict[str, List[int]] = {cls_name: [] for cls_name in PRIMARY_CLINICAL_CLASSES}
    for i, c_name in enumerate(cell_classes):
        if c_name in class_cell_indices:
            class_cell_indices[c_name].append(i)

    # Grouped categories
    grouped_indices: Dict[str, List[int]] = {
        "NORMAL": [i for i, c in enumerate(cell_classes) if c in NORMAL_SET],
        "LOW_GRADE": [i for i, c in enumerate(cell_classes) if c in LOW_GRADE_SET],
        "HIGH_GRADE": [i for i, c in enumerate(cell_classes) if c in HIGH_GRADE_SET],
    }

    for c in range(num_concepts):
        row_dict: Dict[str, Any] = {"concept_id": c}

        # 1. Evaluate on 8 primary clinical classes
        for cls_name in PRIMARY_CLINICAL_CLASSES:
            y_idx = class_to_logit_idx[cls_name]
            indices = class_cell_indices[cls_name]

            if not indices:
                row_dict[f"mean_logit_drop_{cls_name}"] = 0.0
                row_dict[f"mean_prob_drop_{cls_name}"] = 0.0
                row_dict[f"active_prob_drop_{cls_name}"] = 0.0
                row_dict[f"flip_rate_{cls_name}"] = 0.0
                continue

            idx_t = torch.tensor(indices, dtype=torch.long)
            delta_logit = (logits_full[idx_t, y_idx] - logits_minus_c[idx_t, c, y_idx]).numpy()
            delta_prob = (probs_full[idx_t, y_idx] - probs_minus_c[idx_t, c, y_idx]).numpy()

            row_dict[f"mean_logit_drop_{cls_name}"] = float(np.mean(delta_logit))
            row_dict[f"mean_prob_drop_{cls_name}"] = float(np.mean(delta_prob))

            # Active probability drop
            active_mask = (cell_max_act[idx_t, c] > 1e-5).numpy()
            if np.any(active_mask):
                row_dict[f"active_prob_drop_{cls_name}"] = float(np.mean(delta_prob[active_mask]))
            else:
                row_dict[f"active_prob_drop_{cls_name}"] = 0.0

            # Flip rate for cells originally predicted as cls_name
            orig_pred_cls = (pred_full[idx_t] == y_idx)
            if orig_pred_cls.sum() > 0:
                pred_minus = logits_minus_c[idx_t, c, 1:9].argmax(dim=-1) + 1
                flips = (pred_minus[orig_pred_cls] != y_idx)
                row_dict[f"flip_rate_{cls_name}"] = float(flips.float().mean().item())
            else:
                row_dict[f"flip_rate_{cls_name}"] = 0.0

        # 2. Evaluate on Grouped Categories
        for group_name, g_set in [("NORMAL", NORMAL_SET), ("LOW_GRADE", LOW_GRADE_SET), ("HIGH_GRADE", HIGH_GRADE_SET)]:
            indices = grouped_indices[group_name]
            if not indices:
                row_dict[f"mean_logit_drop_{group_name}"] = 0.0
                row_dict[f"mean_prob_drop_{group_name}"] = 0.0
                row_dict[f"active_prob_drop_{group_name}"] = 0.0
                row_dict[f"flip_rate_{group_name}"] = 0.0
                continue

            g_delta_logits = []
            g_delta_probs = []
            g_active_delta_probs = []
            g_orig_in_group = 0
            g_flips = 0

            for idx in indices:
                c_name = cell_classes[idx]
                y_idx = class_to_logit_idx[c_name]
                dl = float(logits_full[idx, y_idx] - logits_minus_c[idx, c, y_idx])
                dp = float(probs_full[idx, y_idx] - probs_minus_c[idx, c, y_idx])
                g_delta_logits.append(dl)
                g_delta_probs.append(dp)

                if cell_max_act[idx, c] > 1e-5:
                    g_active_delta_probs.append(dp)

                pred_orig_cls = logit_idx_to_class.get(int(pred_full[idx].item()), "")
                if pred_orig_cls in g_set:
                    g_orig_in_group += 1
                    pred_minus_idx = int(logits_minus_c[idx, c, 1:9].argmax(dim=-1).item() + 1)
                    pred_minus_cls = logit_idx_to_class.get(pred_minus_idx, "")
                    if pred_minus_cls not in g_set:
                        g_flips += 1

            row_dict[f"mean_logit_drop_{group_name}"] = float(np.mean(g_delta_logits))
            row_dict[f"mean_prob_drop_{group_name}"] = float(np.mean(g_delta_probs))
            row_dict[f"active_prob_drop_{group_name}"] = float(np.mean(g_active_delta_probs)) if g_active_delta_probs else 0.0
            row_dict[f"flip_rate_{group_name}"] = float(g_flips / max(1, g_orig_in_group)) if g_orig_in_group > 0 else 0.0

        # 3. Overall Summary Metrics
        primary_drops = [row_dict[f"mean_logit_drop_{cn}"] for cn in PRIMARY_CLINICAL_CLASSES]
        primary_p_drops = [row_dict[f"mean_prob_drop_{cn}"] for cn in PRIMARY_CLINICAL_CLASSES]

        best_cls_idx = int(np.argmax(primary_drops))
        row_dict["top_causal_class"] = PRIMARY_CLINICAL_CLASSES[best_cls_idx]
        row_dict["max_mean_logit_drop"] = float(primary_drops[best_cls_idx])
        row_dict["max_mean_prob_drop"] = float(primary_p_drops[best_cls_idx])

        # Overall flip rate across all cells
        all_pred_minus = logits_minus_c[:, c, 1:9].argmax(dim=-1) + 1
        row_dict["overall_flip_rate"] = float((all_pred_minus != pred_full).float().mean().item())

        rows.append(row_dict)

    df_single = pd.DataFrame(rows)
    df_single = df_single.sort_values(by="max_mean_logit_drop", ascending=False).reset_index(drop=True)
    return df_single


# ==============================================================================
# Section E: Experiment 2 - Cumulative Top-M Concept Occlusion Curves
# ==============================================================================

def compute_cumulative_occlusion_curves(
    occ_res: Dict[str, Any],
    df_single: pd.DataFrame,
    metadata: List[Dict[str, Any]],
    sam3_model: nn.Module,
    class_to_logit_idx: Dict[str, int],
    device: torch.device,
    seeds: List[int] = [42, 1337, 2026]
) -> pd.DataFrame:
    """
    Computes cumulative Top-M concept occlusion vs. Random-M concept occlusion baseline
    across budgets m in [0, 1, 2, 3, 5, 8, 12, 16, 24, 32] for each Bethesda class.
    """
    target_layer = occ_res["target_layer"]
    is_bsf = occ_res["is_bsf"]
    logits_full = occ_res["logits_full"].to(device)
    cell_max_act = occ_res["cell_max_act"]
    num_concepts = df_single.shape[0]

    cls_score = sam3_model.roi_heads.box_predictor.cls_score
    box_head = sam3_model.roi_heads.box_head
    W_cls = cls_score.weight.detach()

    cell_classes = [r["class_name"] for r in metadata]
    pred_full = logits_full[:, 1:9].argmax(dim=-1) + 1  # [N]

    # Active concept pool across entire dataset
    global_active_concepts = torch.where(cell_max_act.sum(dim=0) > 1e-5)[0].tolist()
    if not global_active_concepts:
        global_active_concepts = list(range(num_concepts))

    curve_rows: List[Dict[str, Any]] = []

    for cls_name in PRIMARY_CLINICAL_CLASSES:
        y_idx = class_to_logit_idx[cls_name]
        cells_for_cls = [i for i, c in enumerate(cell_classes) if c == cls_name]

        if not cells_for_cls:
            continue

        cells_t = torch.tensor(cells_for_cls, dtype=torch.long, device=device)
        orig_predicted_cls = (pred_full[cells_t] == y_idx)

        # Rank all concepts descending by mean_logit_drop on this class
        col_name = f"mean_logit_drop_{cls_name}"
        ranked_concepts = df_single.sort_values(by=col_name, ascending=False)["concept_id"].tolist()

        for m in OCCLUSION_BUDGETS:
            m_eff = min(m, len(ranked_concepts))

            # -------------------------------------------------------------
            # A. Targeted Top-M Occlusion
            # -------------------------------------------------------------
            if m == 0:
                t_logits = logits_full[cells_t]
                t_probs = torch.softmax(t_logits, dim=-1)
            else:
                top_m_concepts = ranked_concepts[:m_eff]

                if target_layer == "box_head":
                    contrib_all = occ_res["contrib_all"].to(device)  # [N, C, 1024]
                    contrib_sum = contrib_all[cells_t][:, top_m_concepts, :].sum(dim=1)  # [N_cls, 1024]
                    t_logits = logits_full[cells_t] - (contrib_sum @ W_cls.T)
                    t_probs = torch.softmax(t_logits, dim=-1)
                else:  # target_layer == "fpn"
                    roi_hat_full = occ_res["roi_hat_full"].to(device)  # [N, 121, 256]
                    z_roi = occ_res["z_roi"].to(device)
                    z_blk_roi = occ_res["z_blk_roi"].to(device) if is_bsf else None
                    atoms = occ_res["atoms"].to(device) if is_bsf else None
                    D = occ_res["D"].to(device) if not is_bsf else None

                    t_rois = []
                    for idx_c in cells_for_cls:
                        if is_bsf:
                            contrib_m = sum(z_blk_roi[idx_c, :, c, :] @ atoms[c] for c in top_m_concepts)
                        else:
                            contrib_m = sum(z_roi[idx_c, :, c:c+1] * D[c:c+1, :] for c in top_m_concepts)
                        roi_mod = (roi_hat_full[idx_c] - contrib_m).view(11, 11, 256).permute(2, 0, 1)
                        t_rois.append(roi_mod)

                    batch_t_rois = torch.stack(t_rois, dim=0)
                    with torch.no_grad():
                        t_logits = cls_score(box_head(batch_t_rois))
                        t_probs = torch.softmax(t_logits, dim=-1)

            t_mean_prob = float(t_probs[:, y_idx].mean().item())
            t_mean_logit = float(t_logits[:, y_idx].mean().item())

            if orig_predicted_cls.sum() > 0:
                still_pred = (t_logits[:, 1:9].argmax(dim=-1) + 1 == y_idx)[orig_predicted_cls]
                t_retention = 100.0 * float(still_pred.float().mean().item())
            else:
                t_retention = 100.0 if m == 0 else 0.0

            t_flip_rate = 100.0 - t_retention

            # -------------------------------------------------------------
            # B. Random-M Baseline (Averaged over seeds)
            # -------------------------------------------------------------
            if m == 0:
                r_mean_prob = t_mean_prob
                r_mean_logit = t_mean_logit
                r_retention = t_retention
                r_flip_rate = t_flip_rate
            else:
                seed_probs = []
                seed_logits = []
                seed_retentions = []

                for s in seeds:
                    rng = random.Random(s + m * 100)

                    if target_layer == "box_head":
                        contrib_all = occ_res["contrib_all"].to(device)
                        # Sample m concepts per cell or globally
                        r_contrib_list = []
                        for idx_local, idx_c in enumerate(cells_for_cls):
                            active_c = torch.where(cell_max_act[idx_c] > 1e-5)[0].tolist()
                            cand_pool = active_c if len(active_c) >= m_eff else global_active_concepts
                            sampled_m = rng.sample(cand_pool, min(m_eff, len(cand_pool)))
                            contrib_sum_c = contrib_all[idx_c, sampled_m, :].sum(dim=0)
                            r_contrib_list.append(contrib_sum_c)

                        r_contrib_tensor = torch.stack(r_contrib_list, dim=0)
                        r_logits_s = logits_full[cells_t] - (r_contrib_tensor @ W_cls.T)
                        r_probs_s = torch.softmax(r_logits_s, dim=-1)
                    else:
                        roi_hat_full = occ_res["roi_hat_full"].to(device)
                        z_roi = occ_res["z_roi"].to(device)
                        z_blk_roi = occ_res["z_blk_roi"].to(device) if is_bsf else None
                        atoms = occ_res["atoms"].to(device) if is_bsf else None
                        D = occ_res["D"].to(device) if not is_bsf else None

                        r_rois = []
                        for idx_c in cells_for_cls:
                            active_c = torch.where(cell_max_act[idx_c] > 1e-5)[0].tolist()
                            cand_pool = active_c if len(active_c) >= m_eff else global_active_concepts
                            sampled_m = rng.sample(cand_pool, min(m_eff, len(cand_pool)))

                            if is_bsf:
                                contrib_m = sum(z_blk_roi[idx_c, :, c, :] @ atoms[c] for c in sampled_m)
                            else:
                                contrib_m = sum(z_roi[idx_c, :, c:c+1] * D[c:c+1, :] for c in sampled_m)
                            roi_mod = (roi_hat_full[idx_c] - contrib_m).view(11, 11, 256).permute(2, 0, 1)
                            r_rois.append(roi_mod)

                        batch_r_rois = torch.stack(r_rois, dim=0)
                        with torch.no_grad():
                            r_logits_s = cls_score(box_head(batch_r_rois))
                            r_probs_s = torch.softmax(r_logits_s, dim=-1)

                    seed_probs.append(float(r_probs_s[:, y_idx].mean().item()))
                    seed_logits.append(float(r_logits_s[:, y_idx].mean().item()))

                    if orig_predicted_cls.sum() > 0:
                        still_r = (r_logits_s[:, 1:9].argmax(dim=-1) + 1 == y_idx)[orig_predicted_cls]
                        seed_retentions.append(100.0 * float(still_r.float().mean().item()))
                    else:
                        seed_retentions.append(0.0)

                r_mean_prob = float(np.mean(seed_probs))
                r_mean_logit = float(np.mean(seed_logits))
                r_retention = float(np.mean(seed_retentions))
                r_flip_rate = 100.0 - r_retention

            curve_rows.append({
                "class_name": cls_name,
                "m": m,
                "targeted_mean_prob": t_mean_prob,
                "targeted_mean_logit": t_mean_logit,
                "targeted_class_retention_pct": t_retention,
                "targeted_flip_rate_pct": t_flip_rate,
                "random_mean_prob": r_mean_prob,
                "random_mean_logit": r_mean_logit,
                "random_class_retention_pct": r_retention,
                "random_flip_rate_pct": r_flip_rate,
            })

    df_cum = pd.DataFrame(curve_rows)
    return df_cum


def summarize_cross_model_faithfulness(
    all_model_curves: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Summarizes AUC of probability drop, top-1/5/16 probability drops, and flip rates
    across all evaluated SAE and BSF models.
    """
    summary_rows = []

    for sae_type, df_cum in all_model_curves.items():
        class_names = df_cum["class_name"].unique()
        if len(class_names) == 0:
            continue

        auc_list = []
        top1_drops = []
        top5_drops = []
        top16_drops = []
        top1_flips = []
        top5_flips = []
        top16_flips = []
        targeted_drops_5 = []
        random_drops_5 = []

        for cls_name in class_names:
            sub = df_cum[df_cum["class_name"] == cls_name].sort_values(by="m")
            if len(sub) == 0:
                continue

            m_vals = sub["m"].values
            t_probs = sub["targeted_mean_prob"].values
            r_probs = sub["random_mean_prob"].values
            t_flips = sub["targeted_flip_rate_pct"].values

            p0 = t_probs[0] if len(t_probs) > 0 else 1.0
            delta_p = np.maximum(0.0, p0 - t_probs)

            # Normalized trapezoidal AUC over budget [0, 32]
            if len(m_vals) > 1 and (m_vals[-1] - m_vals[0]) > 0:
                auc_norm = float(np.trapezoid(delta_p, m_vals) / (m_vals[-1] - m_vals[0]))
            else:
                auc_norm = float(np.mean(delta_p))
            auc_list.append(auc_norm)

            # Interpolate or lookup drops at m=1, m=5, m=16
            m_to_idx = {int(m): i for i, m in enumerate(m_vals)}

            if 1 in m_to_idx:
                i1 = m_to_idx[1]
                top1_drops.append(float(p0 - t_probs[i1]))
                top1_flips.append(float(t_flips[i1]))
            if 5 in m_to_idx:
                i5 = m_to_idx[5]
                top5_drops.append(float(p0 - t_probs[i5]))
                top5_flips.append(float(t_flips[i5]))
                targeted_drops_5.append(float(p0 - t_probs[i5]))
                random_drops_5.append(float(p0 - r_probs[i5]))
            if 16 in m_to_idx:
                i16 = m_to_idx[16]
                top16_drops.append(float(p0 - t_probs[i16]))
                top16_flips.append(float(t_flips[i16]))

        mean_t_drop_5 = float(np.mean(targeted_drops_5)) if targeted_drops_5 else 0.0
        mean_r_drop_5 = float(np.mean(random_drops_5)) if random_drops_5 else 0.0
        ratio_top5 = mean_t_drop_5 / max(mean_r_drop_5, 1e-6)

        summary_rows.append({
            "sae_type": sae_type,
            "auc_prob_drop": float(np.mean(auc_list)) if auc_list else 0.0,
            "top1_prob_drop": float(np.mean(top1_drops)) if top1_drops else 0.0,
            "top5_prob_drop": float(np.mean(top5_drops)) if top5_drops else 0.0,
            "top16_prob_drop": float(np.mean(top16_drops)) if top16_drops else 0.0,
            "top1_flip_rate_pct": float(np.mean(top1_flips)) if top1_flips else 0.0,
            "top5_flip_rate_pct": float(np.mean(top5_flips)) if top5_flips else 0.0,
            "top16_flip_rate_pct": float(np.mean(top16_flips)) if top16_flips else 0.0,
            "targeted_vs_random_drop_ratio_top5": ratio_top5,
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary


# ==============================================================================
# Section F: Three Publication-Quality Visualizations
# ==============================================================================

def plot_causal_attribution_heatmap(
    df_single: pd.DataFrame,
    sae_type: str,
    save_path: str
):
    """
    Visualization 1:
    Heatmap of top concepts per Bethesda class showing Mean Logit Drop when occluded.
    Rows: PRIMARY_CLINICAL_CLASSES, Cols: Selected Concept IDs.
    """
    selected_concepts: List[int] = []
    concept_labels: List[str] = []

    # Select up to top 4 concepts per Bethesda class with highest positive mean_logit_drop
    for cls_name in PRIMARY_CLINICAL_CLASSES:
        col = f"mean_logit_drop_{cls_name}"
        if col in df_single.columns:
            top_for_c = df_single.sort_values(by=col, ascending=False)
            top_c_ids = top_for_c["concept_id"].tolist()
            for cid in top_c_ids:
                if cid not in selected_concepts and len(selected_concepts) < 28:
                    row_c = df_single[df_single["concept_id"] == cid].iloc[0]
                    if row_c[col] > 0.0:
                        selected_concepts.append(cid)
                        top_cls = row_c.get("top_causal_class", cls_name)
                        concept_labels.append(f"C{cid}\n({top_cls})")
                if len([c for c in selected_concepts if df_single[df_single["concept_id"] == c]["top_causal_class"].iloc[0] == cls_name]) >= 4:
                    break

    # If fewer than 8 concepts collected, fill with overall highest
    if len(selected_concepts) < 8:
        for cid in df_single["concept_id"].tolist():
            if cid not in selected_concepts and len(selected_concepts) < 24:
                row_c = df_single[df_single["concept_id"] == cid].iloc[0]
                selected_concepts.append(cid)
                top_cls = row_c.get("top_causal_class", "UNK")
                concept_labels.append(f"C{cid}\n({top_cls})")

    # Build matrix: shape (num_classes, num_selected_concepts)
    matrix = np.zeros((len(PRIMARY_CLINICAL_CLASSES), len(selected_concepts)), dtype=np.float32)
    for r_idx, cls_name in enumerate(PRIMARY_CLINICAL_CLASSES):
        col = f"mean_logit_drop_{cls_name}"
        for c_idx, cid in enumerate(selected_concepts):
            val = df_single.loc[df_single["concept_id"] == cid, col].values[0]
            matrix[r_idx, c_idx] = float(val)

    fig_w = max(12, int(len(selected_concepts) * 0.75))
    fig, ax = plt.subplots(figsize=(fig_w, 7), dpi=300)

    # Plot heatmap
    vmax = max(1.0, float(np.percentile(matrix, 98)))
    vmin = min(0.0, float(np.percentile(matrix, 2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"Mean Logit Drop ($\Delta$Logit)", fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(len(selected_concepts)))
    ax.set_yticks(np.arange(len(PRIMARY_CLINICAL_CLASSES)))
    ax.set_xticklabels(concept_labels, fontsize=8, fontweight="bold", rotation=45, ha="right")
    ax.set_yticklabels(PRIMARY_CLINICAL_CLASSES, fontsize=10, fontweight="bold")

    # Annotate numeric values inside cells
    for r in range(len(PRIMARY_CLINICAL_CLASSES)):
        for c in range(len(selected_concepts)):
            val = matrix[r, c]
            text_color = "white" if val > (vmax * 0.6) else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5, fontweight="bold")

    ax.set_title(
        f"Causal Concept Attribution: Bethesda Class Sensitivity to Concept Occlusion ({sae_type})\n"
        r"(Higher $\Delta$Logit indicates strong causal alignment between learned concept and Bethesda class)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Learned SAE / BSF Concepts", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Bethesda Clinical Class", fontsize=11, fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Causal Attribution Heatmap to {save_path}")


def plot_cumulative_occlusion_curves(
    df_cum: pd.DataFrame,
    sae_type: str,
    save_path: str
):
    """
    Visualization 2:
    2x4 grid of subplots showing relative True-Class Probability Retention (%)
    under Targeted Top-M Occlusion vs. Random-M Baseline.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
    axes_flat = axes.flatten()

    for i, cls_name in enumerate(PRIMARY_CLINICAL_CLASSES):
        ax = axes_flat[i]
        sub = df_cum[df_cum["class_name"] == cls_name].sort_values(by="m")

        if len(sub) == 0:
            ax.text(0.5, 0.5, f"No cells for {cls_name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cls_name, fontsize=11, fontweight="bold")
            continue

        m_vals = sub["m"].values
        t_probs = sub["targeted_mean_prob"].values
        r_probs = sub["random_mean_prob"].values

        p0 = max(float(t_probs[0]), 1e-6)
        t_retention = 100.0 * (t_probs / p0)
        r_retention = 100.0 * (r_probs / p0)

        line_color = BETHESDA_COLORS.get(cls_name, "#2ca02c")

        # Targeted Top-M curve
        ax.plot(
            m_vals, t_retention,
            marker="o", markersize=6, linewidth=2.5,
            color=line_color, label="Targeted Top-M"
        )

        # Random-M Baseline curve
        ax.plot(
            m_vals, r_retention,
            marker="s", markersize=5, linewidth=1.8, linestyle="--",
            color="#7f7f7f", alpha=0.85, label="Random-M Baseline"
        )

        # Annotate drop at m=5 and m=16
        m_to_idx = {int(m): k for k, m in enumerate(m_vals)}
        if 5 in m_to_idx:
            k5 = m_to_idx[5]
            drop5 = 100.0 - t_retention[k5]
            ax.annotate(
                f"-{drop5:.1f}% @ m=5",
                xy=(5, t_retention[k5]),
                xytext=(6, min(95, t_retention[k5] + 12)),
                fontsize=7.5, fontweight="bold", color=line_color,
                arrowprops=dict(arrowstyle="->", color=line_color, lw=1.2)
            )

        if 16 in m_to_idx:
            k16 = m_to_idx[16]
            drop16 = 100.0 - t_retention[k16]
            ax.annotate(
                f"-{drop16:.1f}% @ m=16",
                xy=(16, t_retention[k16]),
                xytext=(17, min(80, t_retention[k16] + 12)),
                fontsize=7.5, fontweight="bold", color="#a00000",
                arrowprops=dict(arrowstyle="->", color="#a00000", lw=1.2)
            )

        ax.set_title(cls_name, fontsize=11, fontweight="bold", color=line_color)
        ax.set_xlabel("Number of Occluded Concepts ($m$)", fontsize=9, fontweight="bold")
        ax.set_ylabel("True-Class Prob Retention (%)", fontsize=9, fontweight="bold")
        ax.set_ylim(-2, 108)
        ax.set_xlim(-0.5, 32.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Causal Concept Occlusion Curves: Class Confidence Decay ({sae_type})\n"
        "(Targeted Top-M Concept Occlusion vs. Random-M Baseline across Ground-Truth Bethesda Cells)",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Cumulative Occlusion Curves to {save_path}")


def plot_qualitative_occlusion_panels(
    occ_res: Dict[str, Any],
    metadata: List[Dict[str, Any]],
    df_single: pd.DataFrame,
    class_to_logit_idx: Dict[str, int],
    logit_idx_to_class: Dict[int, str],
    sae_type: str,
    save_path: str,
    patch_size: int = 160
):
    """
    Visualization 3:
    6 representative classes x 3 columns:
    Col 1: Original 160x160 RGB patch centered on GT cell with dashed bbox.
    Col 2: Patch overlaid with spatial activation heatmap of top occluded concept C{c*}.
    Col 3: Grouped horizontal bar chart of 8 Bethesda class probabilities Before vs. After occlusion.
    """
    rep_classes = ["NILM", "ENDOCERVICAL", "LSIL", "HSIL", "SCC", "ASC-H"]
    cell_classes = [r["class_name"] for r in metadata]

    probs_full = occ_res["probs_full"].detach().cpu()          # [N, 9]
    probs_minus_c = occ_res["probs_minus_c"].detach().cpu()    # [N, C, 9]
    logits_full = occ_res["logits_full"].detach().cpu()        # [N, 9]
    logits_minus_c = occ_res["logits_minus_c"].detach().cpu()  # [N, C, 9]
    roi_spatial_activations = occ_res.get("roi_spatial_activations", None)
    if roi_spatial_activations is not None:
        roi_spatial_activations = roi_spatial_activations.detach().cpu()  # [N, 121, C]
    cell_max_act = occ_res["cell_max_act"].detach().cpu()

    pred_full = logits_full[:, 1:9].argmax(dim=-1) + 1  # [N]

    selected_rows = []
    for cls_name in rep_classes:
        if cls_name not in class_to_logit_idx:
            continue
        y_idx = class_to_logit_idx[cls_name]
        candidate_indices = [i for i, c in enumerate(cell_classes) if c == cls_name]
        if not candidate_indices:
            continue

        # Get top causal concept for this class
        col = f"mean_logit_drop_{cls_name}"
        top_concept_id = int(df_single.sort_values(by=col, ascending=False)["concept_id"].iloc[0])

        # Pick the cell with largest probability drop or class flip under occluding top_concept_id
        best_cell_idx = None
        best_score = -1e9
        for idx_c in candidate_indices:
            p_orig = float(probs_full[idx_c, y_idx])
            p_minus = float(probs_minus_c[idx_c, top_concept_id, y_idx])
            dp = p_orig - p_minus
            flip = int((logits_minus_c[idx_c, top_concept_id, 1:9].argmax(dim=-1) + 1) != y_idx)
            # Prioritize cells with high initial confidence that flip or have large drop
            score = (flip * 5.0) + (dp * 2.0) + p_orig
            if score > best_score:
                best_score = score
                best_cell_idx = idx_c

        if best_cell_idx is not None:
            selected_rows.append((cls_name, best_cell_idx, top_concept_id))

    if not selected_rows:
        print("[WARNING] Could not find suitable cells for qualitative occlusion panels.")
        return

    num_rows = len(selected_rows)
    fig, axes = plt.subplots(num_rows, 3, figsize=(16, 3.2 * num_rows), dpi=300)
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    half_patch = patch_size // 2
    img_cache: Dict[str, np.ndarray] = {}

    for row_idx, (cls_name, cell_idx, cid) in enumerate(selected_rows):
        cell_meta = metadata[cell_idx]
        y_idx = class_to_logit_idx[cls_name]
        img_path = cell_meta["image_path"]
        img_filename = cell_meta["image_filename"]

        # Cache & load 1008x1008 padded image
        if img_path not in img_cache:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_t = TVF.to_tensor(pil_img)
                img_pad, _, _ = sam3_resize_longest_side_and_pad_square(img_t, target_size=1008)
                img_cache[img_path] = img_pad.permute(1, 2, 0).numpy()
            except Exception:
                img_cache[img_path] = np.zeros((1008, 1008, 3), dtype=np.float32)

        img_np = img_cache[img_path]

        cx = int(round(cell_meta["cx_pad"]))
        cy = int(round(cell_meta["cy_pad"]))
        y1 = max(0, cy - half_patch)
        y2 = min(1008, cy + half_patch)
        x1 = max(0, cx - half_patch)
        x2 = min(1008, cx + half_patch)

        patch_rgb = img_np[y1:y2, x1:x2]

        # Coordinates inside patch
        bx1 = cell_meta["x1_pad"] - x1
        by1 = cell_meta["y1_pad"] - y1
        bw = cell_meta["x2_pad"] - cell_meta["x1_pad"]
        bh = cell_meta["y2_pad"] - cell_meta["y1_pad"]
        box_color = BETHESDA_COLORS.get(cls_name, "#00ff00")

        # -------------------------------------------------------------
        # Col 1: Original RGB Patch with Dashed Bounding Box
        # -------------------------------------------------------------
        ax1 = axes[row_idx, 0]
        ax1.imshow(patch_rgb)
        rect1 = patches.Rectangle(
            (bx1, by1), bw, bh,
            linewidth=2.0, edgecolor=box_color, facecolor="none", linestyle="--"
        )
        ax1.add_patch(rect1)
        ax1.set_title(f"GT: {cls_name} | Image: {img_filename}", fontsize=9, fontweight="bold")
        ax1.axis("off")

        # -------------------------------------------------------------
        # Col 2: Spatial Activation Heatmap Overlaid on Patch
        # -------------------------------------------------------------
        ax2 = axes[row_idx, 1]
        ax2.imshow(patch_rgb)

        # Generate spatial heatmap
        heatmap_canvas = np.zeros((patch_rgb.shape[0], patch_rgb.shape[1]), dtype=np.float32)

        if roi_spatial_activations is not None:
            # 11x11 RoI grid
            roi_11x11 = roi_spatial_activations[cell_idx, :, cid].view(11, 11).numpy()
            roi_min, roi_max = roi_11x11.min(), roi_11x11.max()
            if (roi_max - roi_min) > 1e-6:
                roi_norm = (roi_11x11 - roi_min) / (roi_max - roi_min)
            else:
                roi_norm = roi_11x11

            # Interpolate to bounding box size inside patch
            bw_int = max(4, int(round(bw)))
            bh_int = max(4, int(round(bh)))
            roi_t = torch.tensor(roi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            roi_upsampled = F.interpolate(roi_t, size=(bh_int, bw_int), mode="bilinear", align_corners=False)[0, 0].numpy()

            by1_int = max(0, min(patch_rgb.shape[0] - 1, int(round(by1))))
            bx1_int = max(0, min(patch_rgb.shape[1] - 1, int(round(bx1))))
            by2_int = min(patch_rgb.shape[0], by1_int + bh_int)
            bx2_int = min(patch_rgb.shape[1], bx1_int + bw_int)

            actual_h = by2_int - by1_int
            actual_w = bx2_int - bx1_int
            if actual_h > 0 and actual_w > 0:
                heatmap_canvas[by1_int:by2_int, bx1_int:bx2_int] = roi_upsampled[:actual_h, :actual_w]
        else:
            # box_head Mode: Smooth 2D Gaussian over cell box
            act_val = float(cell_max_act[cell_idx, cid])
            y_coords, x_coords = np.ogrid[:patch_rgb.shape[0], :patch_rgb.shape[1]]
            cy_box = by1 + bh / 2.0
            cx_box = bx1 + bw / 2.0
            sigma = max(bw, bh) / 3.0
            dist_sq = (x_coords - cx_box)**2 + (y_coords - cy_box)**2
            heatmap_canvas = act_val * np.exp(-dist_sq / (2.0 * sigma**2 + 1e-6))
            if heatmap_canvas.max() > 0:
                heatmap_canvas = heatmap_canvas / heatmap_canvas.max()

        ax2.imshow(heatmap_canvas, cmap="jet", alpha=0.45)
        rect2 = patches.Rectangle(
            (bx1, by1), bw, bh,
            linewidth=1.8, edgecolor=box_color, facecolor="none", linestyle="--"
        )
        ax2.add_patch(rect2)
        ax2.set_title(f"Concept C{cid} Activation Heatmap (RoI)", fontsize=9, fontweight="bold")
        ax2.axis("off")

        # -------------------------------------------------------------
        # Col 3: Grouped Horizontal Bar Chart (Before vs. After Occlusion)
        # -------------------------------------------------------------
        ax3 = axes[row_idx, 2]
        p_full_vec = probs_full[cell_idx, 1:9].numpy()
        p_minus_vec = probs_minus_c[cell_idx, cid, 1:9].numpy()

        p_before_true = float(probs_full[cell_idx, y_idx])
        p_after_true = float(probs_minus_c[cell_idx, cid, y_idx])
        delta_p = p_before_true - p_after_true

        pred_before_idx = int(logits_full[cell_idx, 1:9].argmax().item() + 1)
        pred_after_idx = int(logits_minus_c[cell_idx, cid, 1:9].argmax().item() + 1)
        pred_before_cls = logit_idx_to_class.get(pred_before_idx, "UNK")
        pred_after_cls = logit_idx_to_class.get(pred_after_idx, "UNK")

        y_pos = np.arange(len(PRIMARY_CLINICAL_CLASSES))
        bar_height = 0.38

        # Plot bars
        ax3.barh(y_pos + bar_height / 2, p_full_vec, height=bar_height, color="#1f77b4", label="Full SAE", alpha=0.85)
        ax3.barh(y_pos - bar_height / 2, p_minus_vec, height=bar_height, color="#d62728", label=f"Occluded C{cid}", alpha=0.85)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(PRIMARY_CLINICAL_CLASSES, fontsize=8, fontweight="bold")
        ax3.set_xlim(0.0, 1.05)
        ax3.set_xlabel("Predicted Probability", fontsize=8, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.3, axis="x")

        title_color = "#d62728" if pred_before_cls != pred_after_cls else "black"
        ax3.set_title(
            rf"$\Delta P$({cls_name}) = -{delta_p:.2f} | Before: {pred_before_cls} $\rightarrow$ After: {pred_after_cls}",
            fontsize=8.5, fontweight="bold", color=title_color
        )
        if row_idx == 0:
            ax3.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Qualitative Concept Occlusion Analysis ({sae_type}): Impact of Concept Occlusion on Ground-Truth Cells\n"
        "(Col 1: Original RGB Cell Patch | Col 2: Concept RoI Activation Heatmap | Col 3: Bethesda Probabilities Before vs. After)",
        fontsize=12, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Qualitative Occlusion Panels to {save_path}")


# ==============================================================================
# Section G: CLI Arguments & Main Pipeline
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal Concept Occlusion Experiment for Learned SAE and BSF Concepts."
    )
    parser.add_argument("--target_layer", type=str, default="fpn", choices=["fpn", "box_head"],
                        help="Target layer: 'fpn' (Level 1 at 144x144, 256-D) or 'box_head' (TwoMLPHead, 1024-D)")
    parser.add_argument("--sae_types", nargs="+",
                        default=["GrassmannianBSF", "VanillaBSF", "OMPSAE", "MPSAE", "TopKSAE", "RATopKSAE"],
                        help="List of SAE/BSF architectures to evaluate")
    parser.add_argument("--checkpoint_dir", type=str, default="./interpretability_analysis/checkpoints",
                        help="Directory containing trained SAE/BSF checkpoints")
    parser.add_argument("--save_dir", type=str, default="./interpretability_analysis/concept_occlusion",
                        help="Directory to save occlusion results and plots")
    parser.add_argument("--weight_checkpoint", type=str,
                        default="./weights/rcnn_best_anchors/best_checkpoint.pth",
                        help="Path to trained SAM3 Faster R-CNN checkpoint")
    parser.add_argument("--base_image_dir", type=str, default="",
                        help="Base directory containing RIVA dataset images (auto-discovered if empty)")
    parser.add_argument("--annotations_path", type=str, default="",
                        help="Path to annotations CSV/JSON file or directory (auto-discovered if empty)")
    parser.add_argument("--split", type=str, default="val_test", choices=["val_test", "val", "all", "train"],
                        help="Dataset split to evaluate")
    parser.add_argument("--k", type=int, default=16,
                        help="Sparsity parameter k / l0")
    parser.add_argument("--expansion_factor", type=float, default=4.0,
                        help="Expansion factor for hidden dimension")
    parser.add_argument("--block_size", type=int, default=4,
                        help="Block size for BSF models")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for feature extraction")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to evaluate (for debugging or fast runs)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Concept Occlusion Experiment on {device} ({args.target_layer}) ===")

    # Automatic checkpoint & save directory switching for box_head
    if args.target_layer == "box_head":
        input_dim = 1024
        if args.checkpoint_dir == "./interpretability_analysis/checkpoints":
            args.checkpoint_dir = "./interpretability_analysis/checkpoints_box_head"
        if args.save_dir == "./interpretability_analysis/concept_occlusion":
            args.save_dir = "./interpretability_analysis/concept_occlusion_box_head"
    else:
        input_dim = 256

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {save_dir.resolve()}")
    print(f"[INFO] Checkpoints directory: {Path(args.checkpoint_dir).resolve()}")

    # 1. Discover Annotations & Image Root
    annotation_files = find_annotations_files(args.annotations_path, args.base_image_dir)
    print(f"[INFO] Discovered annotation files: {annotation_files}")
    annotations_by_img, annotations_df = load_all_annotations(annotation_files)

    image_dir = find_image_directory(args.base_image_dir)
    print(f"[INFO] Discovered image root directory: {image_dir}")

    # 2. Build Exact Logit Mappings
    class_to_logit_idx, logit_idx_to_class = build_class_logit_mappings(annotations_df)

    # 3. Resolve Data Split Images
    train_paths, val_paths, test_paths = get_data_splits(image_dir)
    if args.split == "val":
        candidate_paths = val_paths
    elif args.split == "train":
        candidate_paths = train_paths
    elif args.split == "all":
        candidate_paths = train_paths + val_paths + test_paths
    else:  # "val_test"
        candidate_paths = val_paths + test_paths

    annotated_eval_paths = []
    for p in candidate_paths:
        fname = os.path.basename(p)
        base_no_ext = os.path.splitext(fname)[0].lower()
        if fname in annotations_by_img or any(os.path.splitext(k)[0].lower() == base_no_ext for k in annotations_by_img):
            annotated_eval_paths.append(p)

    if not annotated_eval_paths:
        print("[WARNING] No images in chosen split matched annotations. Searching entire image directory...")
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

    if args.max_images is not None and args.max_images > 0:
        eval_image_paths = eval_image_paths[:args.max_images]

    print(f"[INFO] Total evaluation images with GT annotations: {len(eval_image_paths)}")

    # 4. Build Dataset & DataLoader
    eval_dataset = AnnotatedImageDataset(eval_image_paths, target_size=1008)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 5. Load SAM3 Faster R-CNN Model
    sam3_model = load_sam3_rcnn(args.weight_checkpoint, device)

    # 6. Extract Ground-Truth Cell Features (Once for all SAE models)
    cached_cell_data = extract_ground_truth_cell_features(
        sam3_model=sam3_model,
        dataloader=dataloader,
        annotations_by_img=annotations_by_img,
        class_to_logit_idx=class_to_logit_idx,
        target_layer=args.target_layer,
        device=device
    )

    all_model_curves: Dict[str, pd.DataFrame] = {}

    # 7. Run Pipeline for Each Requested SAE / BSF Architecture
    for sae_type in args.sae_types:
        print("\n" + "=" * 70)
        print(f"Running Concept Occlusion Experiment for: {sae_type} ({args.target_layer})")
        print("=" * 70)

        # Build & Load SAE Model
        try:
            sae_model, is_bsf, num_concepts = build_and_load_sae(
                sae_type=sae_type,
                checkpoint_dir=args.checkpoint_dir,
                device=device,
                input_dim=input_dim,
                expansion_factor=args.expansion_factor,
                k=args.k,
                block_size=args.block_size
            )
        except Exception as e:
            print(f"[ERROR] Could not build/load {sae_type}: {e}. Skipping...")
            continue

        # Run Occlusion Mechanics
        occ_res = run_causal_occlusion_for_model(
            sae_model=sae_model,
            is_bsf=is_bsf,
            num_concepts=num_concepts,
            cached_data=cached_cell_data,
            sam3_model=sam3_model,
            target_layer=args.target_layer,
            class_to_logit_idx=class_to_logit_idx,
            logit_idx_to_class=logit_idx_to_class,
            device=device
        )

        # Experiment 1: Single-Concept Causal Attribution
        df_single = compute_single_concept_causal_attribution(
            occ_res=occ_res,
            metadata=cached_cell_data["metadata"],
            num_concepts=num_concepts,
            class_to_logit_idx=class_to_logit_idx,
            logit_idx_to_class=logit_idx_to_class
        )
        single_csv_path = save_dir / f"concept_occlusion_single_{sae_type}.csv"
        df_single.to_csv(single_csv_path, index=False)
        print(f"[INFO] Saved single-concept causal attribution to {single_csv_path}")

        # Experiment 2: Cumulative Top-M Occlusion Curves
        df_cum = compute_cumulative_occlusion_curves(
            occ_res=occ_res,
            df_single=df_single,
            metadata=cached_cell_data["metadata"],
            sam3_model=sam3_model,
            class_to_logit_idx=class_to_logit_idx,
            device=device
        )
        cum_csv_path = save_dir / f"concept_occlusion_cumulative_{sae_type}.csv"
        df_cum.to_csv(cum_csv_path, index=False)
        print(f"[INFO] Saved cumulative occlusion curves to {cum_csv_path}")
        all_model_curves[sae_type] = df_cum

        # Section F: Generate Visualizations
        heatmap_path = str(save_dir / f"{sae_type}_causal_attribution_heatmap.png")
        plot_causal_attribution_heatmap(df_single, sae_type, heatmap_path)

        curves_path = str(save_dir / f"{sae_type}_cumulative_occlusion_curves.png")
        plot_cumulative_occlusion_curves(df_cum, sae_type, curves_path)

        panels_path = str(save_dir / f"{sae_type}_qualitative_occlusion_panels.png")
        plot_qualitative_occlusion_panels(
            occ_res=occ_res,
            metadata=cached_cell_data["metadata"],
            df_single=df_single,
            class_to_logit_idx=class_to_logit_idx,
            logit_idx_to_class=logit_idx_to_class,
            sae_type=sae_type,
            save_path=panels_path
        )

    # 8. Cross-Model Summary
    if all_model_curves:
        df_summary = summarize_cross_model_faithfulness(all_model_curves)
        summary_csv_path = save_dir / "concept_occlusion_summary_all_models.csv"
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\n[INFO] Saved Cross-Model Faithfulness Summary to {summary_csv_path}")
        print("\n=== Cross-Model Causal Occlusion Summary ===")
        print(df_summary.to_string(index=False))

    print(f"\n[SUCCESS] Concept Occlusion Experiment completed successfully. Results saved in {save_dir.resolve()}.")


if __name__ == "__main__":
    main()
