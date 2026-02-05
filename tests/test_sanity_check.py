#!/usr/bin/env python3
import argparse
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from data.dataset import BethesdaDataset
from data.transforms import get_train_transforms_RCNN, get_valid_transforms
from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square


def iter_samples(dataset, max_samples):
    for i in range(min(len(dataset), max_samples)):
        yield dataset[i]


def main():
    parser = argparse.ArgumentParser(description="RCNN v2 pipeline sanity checks")
    parser.add_argument("--csv", required=True, help="Path to train.csv")
    parser.add_argument("--images", required=True, help="Path to image directory")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--use-train-transforms", action="store_true")
    parser.add_argument("--check-model-transform", action="store_true")
    args = parser.parse_args()
    transforms = get_train_transforms_RCNN() if args.use_train_transforms else get_valid_transforms()
    ds = BethesdaDataset(csv_file=args.csv, root_dir=args.images, transforms=transforms)
    # Stats
    label_min, label_max = 1e9, -1e9
    invalid_boxes = 0
    total_boxes = 0
    img_min, img_max = 1e9, -1e9
    preds = []
    targets = []
    for img, tgt in iter_samples(ds, args.num_samples):
        # Apply sam3 resize+pad (as in train.py)
        img, tgt, _ = sam3_resize_longest_side_and_pad_square(img, tgt, target_size=1008)

        # image range
        img_min = min(img_min, float(img.min()))
        img_max = max(img_max, float(img.max()))

        # labels range
        if tgt["labels"].numel() > 0:
            label_min = min(label_min, int(tgt["labels"].min()))
            label_max = max(label_max, int(tgt["labels"].max()))

        # box validity
        boxes = tgt["boxes"]
        if boxes.numel() > 0:
            invalid = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
            invalid_boxes += int(invalid.sum().item())
            total_boxes += boxes.shape[0]

        # perfect predictions for metric sanity
        preds.append({
            "boxes": boxes.clone(),
            "scores": torch.ones((boxes.shape[0],), dtype=torch.float32),
            "labels": tgt["labels"].clone(),
        })
        targets.append({
            "boxes": boxes.clone(),
            "labels": tgt["labels"].clone(),
        })
    print("=== Dataset + Transform Sanity ===")
    print(f"samples checked: {min(len(ds), args.num_samples)}")
    print(f"image range: [{img_min:.4f}, {img_max:.4f}] (expected ~0..1)")
    print(f"label range: [{label_min}, {label_max}] (expected 1..8)")
    print(f"boxes total: {total_boxes}, invalid: {invalid_boxes}")

    # Metric sanity (perfect predictions)
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    results = metric.compute()
    print("=== mAP sanity (perfect preds) ===")
    print(f"map: {results['map'].item():.4f}")
    print(f"map_50: {results['map_50'].item():.4f}")
    print(f"map_75: {results['map_75'].item():.4f}")

    # Optional: check model.transform behavior
    if args.check_model_transform:
        print("=== Model.transform sanity ===")
        model = build_sam3_fasterrcnn(
            model_name_or_path="facebook/sam3",
            num_classes_closed_set=8,
            trainable_backbone=False,
        )
        model.eval()
        # Take a small batch
        images = []
        targets_m = []
        for img, tgt in iter_samples(ds, min(2, args.num_samples)):
            img, tgt, _ = sam3_resize_longest_side_and_pad_square(img, tgt, target_size=1008)
            images.append(img)
            targets_m.append(tgt)
        image_list, targets_m = model.transform(images, targets_m)
        print(f"transformed tensor range: [{image_list.tensors.min():.4f}, {image_list.tensors.max():.4f}]")
        print(f"transform sizes: {image_list.tensors.shape}")

if __name__ == "__main__":
    main()