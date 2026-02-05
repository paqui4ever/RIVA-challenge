#!/usr/bin/env python3
import argparse
import math
import random

import numpy as np
import torch

from data.dataset import BethesdaDataset
from data.transforms import get_train_transforms_RCNN, get_valid_transforms
from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_strides(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def kmeans_1d(x: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    if x.size == 0:
        raise ValueError("No data points for clustering")

    percentiles = np.linspace(0, 100, k + 2)[1:-1]
    centroids = np.percentile(x, percentiles)

    for _ in range(max_iter):
        distances = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for i in range(k):
            cluster = x[labels == i]
            if cluster.size > 0:
                new_centroids[i] = cluster.mean()
        if np.allclose(new_centroids, centroids, atol=1e-3):
            break
        centroids = new_centroids

    return np.sort(centroids)


def assign_levels(box_sides: np.ndarray, strides, anchor_scale: float) -> list:
    nominal_sizes = [s * anchor_scale for s in strides]
    levels = [[] for _ in strides]
    for side in box_sides:
        diffs = [abs(math.log2(side / ns)) for ns in nominal_sizes]
        level_idx = int(np.argmin(diffs))
        levels[level_idx].append(float(side))
    return levels


def main():
    parser = argparse.ArgumentParser(description="Suggest anchor sizes for SAM3 RCNN v2")
    parser.add_argument("--csv", required=True, help="Path to train.csv")
    parser.add_argument("--images", required=True, help="Path to training images directory")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to use (0 = all)")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-train-transforms", action="store_true", help="Use training augmentations")
    parser.add_argument("--target-size", type=int, default=1008)
    parser.add_argument("--num-anchors-per-level", type=int, default=2, help="Anchor sizes per FPN level")
    parser.add_argument("--num-levels", type=int, default=4, help="Number of FPN levels")
    parser.add_argument("--anchor-scale", type=float, default=8.0, help="Nominal anchor size = stride * scale")
    parser.add_argument("--sam3-checkpoint", default="facebook/sam3", help="SAM3 checkpoint id or path")
    parser.add_argument("--skip-backbone", action="store_true", help="Skip backbone stride inference")
    parser.add_argument("--assume-strides", default="7,14,28,56", help="Fallback strides if backbone is skipped")
    args = parser.parse_args()

    set_seed(args.seed)

    transforms = get_train_transforms_RCNN() if args.use_train_transforms else get_valid_transforms()
    dataset = BethesdaDataset(csv_file=args.csv, root_dir=args.images, transforms=transforms)

    total = len(dataset)
    if args.num_samples <= 0 or args.num_samples > total:
        indices = list(range(total))
    elif args.random_sample:
        indices = random.sample(range(total), args.num_samples)
    else:
        indices = list(range(args.num_samples))

    widths = []
    heights = []
    areas = []
    ars = []

    for idx in indices:
        image, target = dataset[idx]
        image, target, _ = sam3_resize_longest_side_and_pad_square(
            image, target, target_size=args.target_size
        )
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue
        w = (boxes[:, 2] - boxes[:, 0]).cpu().numpy()
        h = (boxes[:, 3] - boxes[:, 1]).cpu().numpy()
        a = w * h
        widths.extend(w.tolist())
        heights.extend(h.tolist())
        areas.extend(a.tolist())
        ars.extend((w / np.maximum(h, 1e-6)).tolist())

    if len(areas) == 0:
        raise ValueError("No boxes found in selected samples")

    widths = np.asarray(widths)
    heights = np.asarray(heights)
    areas = np.asarray(areas)
    ars = np.asarray(ars)

    side = np.sqrt(areas)

    ar_p10 = float(np.percentile(ars, 10))
    ar_p50 = float(np.percentile(ars, 50))
    ar_p90 = float(np.percentile(ars, 90))
    ar_tuple = (round(ar_p10, 2), round(ar_p50, 2), round(ar_p90, 2))

    if args.skip_backbone:
        strides = parse_strides(args.assume_strides)
    else:
        model = build_sam3_fasterrcnn(
            model_name_or_path=args.sam3_checkpoint,
            num_classes_closed_set=8,
            trainable_backbone=False,
        )
        model.eval()
        dummy = torch.zeros(1, 3, args.target_size, args.target_size)
        with torch.no_grad():
            feats = model.backbone(dummy)
        strides = []
        for _, v in feats.items():
            stride = args.target_size / v.shape[-1]
            strides.append(int(round(stride)))
        strides = sorted(strides)

    if len(strides) != args.num_levels:
        raise ValueError(f"Expected {args.num_levels} strides, got {len(strides)}: {strides}")

    level_sizes = assign_levels(side, strides, args.anchor_scale)
    anchors_per_level = []
    level_counts = []
    for i, sizes in enumerate(level_sizes):
        level_counts.append(len(sizes))
        if len(sizes) >= args.num_anchors_per_level:
            anchors = kmeans_1d(np.asarray(sizes), args.num_anchors_per_level)
        else:
            nominal = strides[i] * args.anchor_scale
            anchors = np.array([nominal * 0.8, nominal * 1.2])[: args.num_anchors_per_level]
        anchors_per_level.append(tuple(int(round(v)) for v in anchors))

    print("=== Box Size Summary ===")
    print(f"samples: {len(indices)}")
    print(f"boxes: {len(areas)}")
    print(f"width  mean±std: {widths.mean():.1f} ± {widths.std():.1f}")
    print(f"height mean±std: {heights.mean():.1f} ± {heights.std():.1f}")
    print(f"side   mean±std: {side.mean():.1f} ± {side.std():.1f}")
    print(f"side   p10/p50/p90: {np.percentile(side, 10):.1f} / {np.percentile(side, 50):.1f} / {np.percentile(side, 90):.1f}")
    print(f"aspect ratio p10/p50/p90: {ar_tuple}")

    print("\n=== Suggested Anchor Sizes ===")
    print(f"strides: {strides}")
    print(f"boxes per level: {level_counts}")
    print(f"sizes: {tuple(anchors_per_level)}")
    print(f"aspect_ratios: {((ar_tuple),) * args.num_levels}")

    print("\n=== AnchorGenerator Snippet ===")
    print("AnchorGenerator(")
    print(f"    sizes={tuple(anchors_per_level)},")
    print(f"    aspect_ratios={((ar_tuple),) * args.num_levels},")
    print(")")


if __name__ == "__main__":
    main()
