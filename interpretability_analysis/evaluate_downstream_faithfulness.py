"""
Downstream Faster R-CNN Faithfulness Evaluation
==============================================
Evaluates how faithfully sparse representations from trained Sparse Autoencoders (SAEs)
and Block-Sparse Featurizers (BSFs) preserve downstream detection performance when
FPN feature map "1" is replaced in-place by the model's sparse reconstruction x_hat.

Models evaluated:
- GrassmannianBSF
- VanillaBSF
- OMPSAE
- MPSAE
- TopKSAE
- RATopKSAE
- Zero-Ablation (Baseline: FPN "1" set to all zeros)

Metrics recorded:
1. f1_box_retention: Retention F1 of baseline high-confidence detection bounding boxes (IoU > 0.5).
2. box_recall: % of original high-confidence boxes recovered under SAE reconstruction.
3. box_precision: Precision of intervention detections relative to baseline detections.
4. class_agreement_pct: % of matched boxes where predicted Bethesda class matches original prediction.
5. mean_score_correlation: Pearson correlation (r) between baseline confidence scores and reconstructed scores.
6. r2_fpn: Feature reconstruction R^2 on FPN level "1" across the evaluation set.

Saves:
- ./interpretability_analysis/results/sae_downstream_faithfulness.csv
- ./interpretability_analysis/results/sae_downstream_faithfulness.png
"""

import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TVF
from torchvision.ops import box_iou
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Pearson correlation safely with zero-variance protection."""
    if len(x) < 3:
        return 0.0
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return r if not np.isnan(r) else 0.0


def find_image_directory(base_image_dir: str = "") -> str:
    """Auto-discovers the RIVA images root directory."""
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
        if p.is_dir() and ((p / "test").is_dir() or (p / "val").is_dir() or (p / "train").is_dir()):
            return str(p.resolve())

    if os.path.isdir("./RIVA"):
        return str(Path("./RIVA").resolve())

    raise FileNotFoundError("Could not auto-discover RIVA image directory.")


class TestImagesDataset(Dataset):
    """Loads images, applies SAM3 square resize and pad to 1008x1008."""
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


def load_sam3_detector(weight_checkpoint: str, device: torch.device) -> nn.Module:
    """Loads full SAM3 Faster R-CNN model with trained weights in evaluation mode."""
    print("[INFO] Building SAM3 Faster R-CNN detector...")
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
        print(f"[WARNING] Checkpoint {weight_checkpoint} not found. Using initialized weights.")

    sam3_model.to(device)
    sam3_model.eval()
    return sam3_model


def build_and_load_sae_model(
    sae_type: str,
    checkpoint_dir: str,
    device: torch.device,
    input_dim: int = 256,
    expansion_factor: float = 4.0,
    k: int = 16,
    block_size: int = 4
) -> Tuple[Optional[nn.Module], bool]:
    """Instantiates SAE/BSF model and loads trained checkpoint if present."""
    if sae_type == "Zero-Ablation":
        return None, False

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
        raise ValueError(f"Unknown SAE type: {sae_type}")

    model = sae_constructors[sae_type]()

    # Checkpoint lookup
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
        print(f"[WARNING] Checkpoint for {sae_type} not found in {checkpoint_dir}. Using initialized SAE.")

    model.to(device)
    model.eval()
    return model, is_bsf


# ==============================================================================
# Detection Matching & Faithfulness Metrics
# ==============================================================================

def match_and_score_detections(
    baseline_boxes: torch.Tensor,
    baseline_labels: torch.Tensor,
    baseline_scores: torch.Tensor,
    interv_boxes: torch.Tensor,
    interv_labels: torch.Tensor,
    interv_scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> Tuple[int, int, List[float], List[float]]:
    """
    Matches baseline high-confidence detections to intervention detections with IoU > threshold.
    Returns:
    (n_matched, n_class_agreed, matched_base_scores, matched_interv_scores)
    """
    M = len(baseline_boxes)
    K = len(interv_boxes)

    if M == 0 or K == 0:
        return 0, 0, [], []

    ious = box_iou(baseline_boxes, interv_boxes).cpu().numpy()  # (M, K)

    matched_interv_indices = set()
    n_matched = 0
    n_class_agreed = 0
    matched_base_scores = []
    matched_interv_scores = []

    # Greedy matching in order of highest IoU
    # Flatten and sort candidate pairs
    flat_indices = np.argsort(-ious, axis=None)
    matched_base_indices = set()

    for idx in flat_indices:
        m = idx // K
        k = idx % K
        iou_val = ious[m, k]
        if iou_val < iou_threshold:
            break
        if m in matched_base_indices or k in matched_interv_indices:
            continue

        matched_base_indices.add(m)
        matched_interv_indices.add(k)
        n_matched += 1

        b_cls = int(baseline_labels[m].item())
        i_cls = int(interv_labels[k].item())
        if b_cls == i_cls:
            n_class_agreed += 1

        matched_base_scores.append(float(baseline_scores[m].item()))
        matched_interv_scores.append(float(interv_scores[k].item()))

    return n_matched, n_class_agreed, matched_base_scores, matched_interv_scores


# ==============================================================================
# Core Faithfulness Evaluation Pipeline
# ==============================================================================

def evaluate_downstream_faithfulness(
    sam3_model: nn.Module,
    dataloader: DataLoader,
    sae_models: Dict[str, Optional[nn.Module]],
    device: torch.device,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Executes the baseline pass and interventions for all SAE models + Zero-Ablation.
    Computes box retention F1, recall, precision, class agreement %, score correlation,
    and FPN level "1" feature reconstruction R^2.
    """
    sam3_model.eval()

    # 1. Baseline Detection Pass
    print("\n" + "="*70)
    print("STEP 1: Running Unperturbed Faster R-CNN Baseline Pass...")
    print("="*70)

    baseline_detections: List[Dict[str, torch.Tensor]] = []
    total_baseline_boxes = 0

    with torch.no_grad():
        for batch_imgs, _ in tqdm(dataloader, desc="Baseline Faster R-CNN Pass"):
            batch_dev = batch_imgs.to(device)
            # Faster R-CNN expects list of image tensors
            images_list = [img for img in batch_dev]
            outputs = sam3_model(images_list)

            for out in outputs:
                boxes = out["boxes"].detach().cpu()
                labels = out["labels"].detach().cpu()
                scores = out["scores"].detach().cpu()

                high_conf_mask = scores > score_threshold
                b_boxes = boxes[high_conf_mask]
                b_labels = labels[high_conf_mask]
                b_scores = scores[high_conf_mask]

                baseline_detections.append({
                    "boxes": b_boxes,
                    "labels": b_labels,
                    "scores": b_scores,
                })
                total_baseline_boxes += len(b_boxes)

    print(f"[INFO] Baseline pass recorded {total_baseline_boxes} high-confidence boxes (score > {score_threshold})")

    # 2. SAE Intervention Passes
    results = []

    for model_name, sae_model in sae_models.items():
        print("\n" + "-"*70)
        print(f"STEP 2: Evaluating Downstream Faithfulness for: {model_name}")
        print("-"*70)

        # Track FPN R^2 statistics
        total_sq_err = 0.0
        total_sq_var = 0.0

        # Define intervention hook on sam3_model.backbone
        def make_hook(name, sae_m):
            def hook_fn(module, input, output):
                # output is an OrderedDict[str, Tensor] with keys "0", "1", "2", "3"
                feat = output["1"]  # [B, C, H, W]
                B, C, H, W = feat.shape
                feat_flat = feat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

                if name == "Zero-Ablation":
                    x_hat_flat = torch.zeros_like(feat_flat)
                else:
                    with torch.no_grad():
                        _, _, x_hat_flat = sae_m(feat_flat)

                # Compute R2 components
                nonlocal total_sq_err, total_sq_var
                err = feat_flat - x_hat_flat
                total_sq_err += float(torch.sum(err ** 2).item())
                f_mean = feat_flat.mean(dim=0, keepdim=True)
                total_sq_var += float(torch.sum((feat_flat - f_mean) ** 2).item())

                # Reconstruct back to spatial tensor [B, C, H, W]
                x_hat = x_hat_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

                # Replace in-place and return modified OrderedDict
                new_out = OrderedDict(output)
                new_out["1"] = x_hat
                return new_out
            return hook_fn

        hook_handle = sam3_model.backbone.register_forward_hook(make_hook(model_name, sae_model))

        total_matched = 0
        total_class_agreed = 0
        total_interv_boxes = 0
        all_matched_base_scores = []
        all_matched_interv_scores = []

        img_idx = 0
        with torch.no_grad():
            for batch_imgs, _ in tqdm(dataloader, desc=f"Intervention Pass [{model_name}]"):
                batch_dev = batch_imgs.to(device)
                images_list = [img for img in batch_dev]
                outputs = sam3_model(images_list)

                for out in outputs:
                    b_det = baseline_detections[img_idx]
                    img_idx += 1

                    boxes = out["boxes"].detach().cpu()
                    labels = out["labels"].detach().cpu()
                    scores = out["scores"].detach().cpu()

                    high_conf_mask = scores > score_threshold
                    i_boxes = boxes[high_conf_mask]
                    i_labels = labels[high_conf_mask]
                    i_scores = scores[high_conf_mask]

                    total_interv_boxes += len(i_boxes)

                    n_m, n_ca, s_base, s_int = match_and_score_detections(
                        b_det["boxes"], b_det["labels"], b_det["scores"],
                        i_boxes, i_labels, i_scores,
                        iou_threshold=iou_threshold
                    )

                    total_matched += n_m
                    total_class_agreed += n_ca
                    all_matched_base_scores.extend(s_base)
                    all_matched_interv_scores.extend(s_int)

        hook_handle.remove()

        # Compute Summary Metrics
        recall = float(total_matched / max(1, total_baseline_boxes))
        precision = float(total_matched / max(1, total_interv_boxes))
        if (precision + recall) > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

        class_agreement = float(total_class_agreed / max(1, total_matched)) * 100.0

        if len(all_matched_base_scores) >= 3:
            score_corr = safe_pearson_corr(
                np.array(all_matched_base_scores),
                np.array(all_matched_interv_scores)
            )
        else:
            score_corr = 0.0

        r2_fpn = 1.0 - (total_sq_err / max(1e-8, total_sq_var))

        print(f"[{model_name}] Results:")
        print(f"  F1 Box Retention:     {f1 * 100.0:.2f}% (Recall: {recall * 100.0:.2f}%, Prec: {precision * 100.0:.2f}%)")
        print(f"  Class Agreement:      {class_agreement:.2f}% ({total_class_agreed}/{total_matched} matched boxes)")
        print(f"  Score Correlation:    r = {score_corr:.4f}")
        print(f"  FPN Level 1 R^2:      {r2_fpn:.4f}")

        results.append({
            "model": model_name,
            "f1_box_retention": f1 * 100.0,
            "box_recall": recall * 100.0,
            "box_precision": precision * 100.0,
            "class_agreement_pct": class_agreement,
            "mean_score_correlation": score_corr,
            "r2_fpn": r2_fpn,
            "baseline_boxes_count": total_baseline_boxes,
            "intervention_boxes_count": total_interv_boxes,
            "matched_boxes_count": total_matched,
        })

    df_results = pd.DataFrame(results)
    return df_results


# ==============================================================================
# Visualization Plot
# ==============================================================================

def plot_faithfulness_comparison(df_results: pd.DataFrame, save_path: str):
    """Generates a publication-quality 4-panel comparison bar plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    axes = axes.flatten()

    models = df_results["model"].tolist()
    colors = []
    for m in models:
        if m == "Zero-Ablation":
            colors.append("#7f7f7f")       # Gray
        elif "Grassmannian" in m:
            colors.append("#2ca02c")       # Green
        elif "Vanilla" in m:
            colors.append("#1f77b4")       # Blue
        elif "OMP" in m:
            colors.append("#ff7f0e")       # Orange
        elif "MP" in m:
            colors.append("#d62728")       # Red
        elif "RATopK" in m:
            colors.append("#9467bd")       # Purple
        else:
            colors.append("#17becf")       # Teal

    # Metric 1: F1 Box Retention
    ax = axes[0]
    bars = ax.bar(models, df_results["f1_box_retention"], color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("F1 Box Retention (%)", fontsize=11, fontweight="bold")
    ax.set_title("Downstream Detection Box Retention (IoU > 0.5)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9, fontweight="medium")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 2: Class Agreement
    ax = axes[1]
    bars = ax.bar(models, df_results["class_agreement_pct"], color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Bethesda Class Agreement (%)", fontsize=11, fontweight="bold")
    ax.set_title("Classification Faithfulness on Matched Cells", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9, fontweight="medium")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 3: Score Correlation
    ax = axes[2]
    bars = ax.bar(models, df_results["mean_score_correlation"], color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Confidence Score Correlation (r)", fontsize=11, fontweight="bold")
    ax.set_title("Detection Confidence Score Faithfulness", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9, fontweight="medium")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"r={h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, max(0, h)),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 4: FPN Reconstruction R^2
    ax = axes[3]
    bars = ax.bar(models, df_results["r2_fpn"], color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Feature Reconstruction R²", fontsize=11, fontweight="bold")
    ax.set_title("FPN Layer '1' Sparse Feature Reconstruction Quality", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9, fontweight="medium")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, max(0, h)),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle(
        "Downstream Faster R-CNN Intervention Faithfulness Evaluation\n"
        "(In-Place Sparse FPN Reconstruction vs. Unperturbed Detection Baseline)",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved downstream faithfulness comparison visualization to {save_path}")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Downstream Faster R-CNN Faithfulness on {device} ===")

    # Setup output paths
    save_csv_path = Path(args.save_csv)
    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_path = Path(args.save_plot)
    save_plot_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Discover image directory & split
    image_dir = find_image_directory(args.base_image_dir)
    print(f"[INFO] Using image directory: {image_dir}")

    train_paths, val_paths, test_paths = get_data_splits(image_dir)
    if args.split == "test":
        eval_paths = test_paths
    elif args.split == "val":
        eval_paths = val_paths
    elif args.split == "val_test":
        eval_paths = val_paths + test_paths
    else:
        eval_paths = train_paths + val_paths + test_paths

    if args.max_images and len(eval_paths) > args.max_images:
        print(f"[INFO] Limiting evaluation images to {args.max_images} (from {len(eval_paths)})")
        eval_paths = eval_paths[:args.max_images]

    print(f"[INFO] Evaluating on {len(eval_paths)} images from '{args.split}' split.")
    eval_dataset = TestImagesDataset(eval_paths, target_size=1008)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Load SAM3 Faster R-CNN
    sam3_model = load_sam3_detector(args.weight_checkpoint, device)

    # 3. Instantiate Models
    eval_models: Dict[str, Optional[nn.Module]] = {}

    for sae_type in args.sae_types:
        model, _ = build_and_load_sae_model(
            sae_type=sae_type,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            input_dim=256,
            expansion_factor=args.expansion_factor,
            k=args.k,
            block_size=args.block_size
        )
        eval_models[sae_type] = model

    if args.include_zero_ablation:
        eval_models["Zero-Ablation"] = None

    # 4. Run Evaluation
    df_results = evaluate_downstream_faithfulness(
        sam3_model=sam3_model,
        dataloader=eval_loader,
        sae_models=eval_models,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )

    # 5. Save Table & Plot
    df_results.to_csv(str(save_csv_path), index=False)
    print(f"\n[INFO] Saved downstream faithfulness results to {save_csv_path}")

    plot_faithfulness_comparison(df_results, str(save_plot_path))

    print("\n" + "="*80)
    print("FINAL SUMMARY: Downstream Faster R-CNN Faithfulness")
    print("="*80)
    summary_cols = ["model", "f1_box_retention", "class_agreement_pct", "mean_score_correlation", "r2_fpn"]
    print(df_results[summary_cols].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downstream Faster R-CNN Detection & Classification Faithfulness Evaluation."
    )
    parser.add_argument(
        "--sae_types", nargs="+",
        default=["GrassmannianBSF", "VanillaBSF", "OMPSAE", "MPSAE", "TopKSAE", "RATopKSAE"],
        help="List of SAE architectures to evaluate."
    )
    parser.add_argument(
        "--include_zero_ablation", action="store_true", default=True,
        help="Include zero-ablation baseline (setting FPN layer '1' to zeros)."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./interpretability_analysis/checkpoints",
        help="Directory containing trained SAE checkpoints."
    )
    parser.add_argument(
        "--weight_checkpoint", type=str, default="./weights/rcnn_best_anchors/best_checkpoint.pth",
        help="Path to SAM3 Faster R-CNN checkpoint."
    )
    parser.add_argument(
        "--base_image_dir", type=str, default="",
        help="Root directory for RIVA images."
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["test", "val", "val_test", "all"],
        help="Split to run faithfulness evaluation on."
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.3,
        help="Confidence score threshold for baseline detections."
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5,
        help="IoU threshold for matching baseline and intervention detections."
    )
    parser.add_argument(
        "--expansion_factor", type=float, default=4.0,
        help="Expansion factor for SAE models."
    )
    parser.add_argument(
        "--k", type=int, default=16,
        help="Target sparsity used during training."
    )
    parser.add_argument(
        "--block_size", type=int, default=4,
        help="Block size for BSF models."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for Faster R-CNN forward pass."
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Optional limit on number of images to evaluate."
    )
    parser.add_argument(
        "--save_csv", type=str, default="./interpretability_analysis/results/sae_downstream_faithfulness.csv",
        help="Path to save faithfulness CSV metrics."
    )
    parser.add_argument(
        "--save_plot", type=str, default="./interpretability_analysis/results/sae_downstream_faithfulness.png",
        help="Path to save faithfulness comparison bar plot."
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
