"""Compute class-agnostic mAP for SAM3 DETR v2 predictions."""

import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import Sam3Processor

from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from data.transforms import get_valid_transforms_DETR
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection


def parse_args():
    parser = argparse.ArgumentParser(description="Class-agnostic mAP for SAM3 DETR v2")
    parser.add_argument(
        "--csv",
        type=str,
        default="/local_data/RIVA/annotations/annotations/val.csv",
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="/local_data/RIVA/images/images/val",
        help="Path to images directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/sam3",
        help="SAM3 checkpoint name or path",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Optional Sam3ForClosedSetDetection checkpoint path",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="Number of foreground classes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=-1,
        help="Number of batches to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=1008,
        help="Target image size for SAM3 inputs",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="Score threshold for predictions",
    )
    parser.add_argument(
        "--max_detections",
        type=int,
        default=100,
        help="Max detections per image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Sam3Processor and dataset...")
    processor = Sam3Processor.from_pretrained(args.checkpoint)
    transforms = get_valid_transforms_DETR(processor, size=args.target_size)
    dataset = BethesdaDatasetForSam3DETR(
        csv_file=args.csv,
        root_dir=args.images,
        transforms=transforms,
    )
    collate_fn = make_detr_collate_fn(processor, target_size=args.target_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Loading Sam3ForClosedSetDetection...")
    model = Sam3ForClosedSetDetection(
        sam3_checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        freeze_sam3=False,
    ).to(device)

    if args.model_checkpoint:
        print(f"Loading model checkpoint: {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("criterion.")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint: {unexpected}")
        if missing:
            print(f"Warning: missing keys in checkpoint: {missing}")

    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    num_batches = args.num_batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if num_batches != -1 and batch_idx >= num_batches:
                break

            pixel_values, input_ids, attention_mask, targets, orig_sizes = batch
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = None if attention_mask is None else attention_mask.to(device)
            orig_sizes = orig_sizes.to(device)

            predictions = model.predict(
                pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                orig_sizes=orig_sizes,
                score_thresh=args.score_thresh,
                max_detections=args.max_detections,
            )

            preds_cpu = []
            targets_cpu = []

            for b in range(len(predictions)):
                pred_boxes = predictions[b]["boxes"].cpu()
                pred_scores = predictions[b]["scores"].cpu()
                pred_labels = predictions[b]["labels"].cpu()
                pred_labels = torch.zeros_like(pred_labels)

                preds_cpu.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                })

                h, w = orig_sizes[b].tolist()
                tgt_boxes = targets[b]["boxes"].clone()
                if tgt_boxes.numel() > 0:
                    tgt_boxes[:, [0, 2]] *= w
                    tgt_boxes[:, [1, 3]] *= h

                tgt_labels = targets[b]["labels"].clone()
                tgt_labels = torch.zeros_like(tgt_labels)

                targets_cpu.append({
                    "boxes": tgt_boxes.cpu(),
                    "labels": tgt_labels.cpu(),
                })

            metric.update(preds_cpu, targets_cpu)

    results = metric.compute()
    print("Class-agnostic Validation Results:")
    print(f"  mAP (0.50:0.95): {results['map'].item():.4f}")
    print(f"  mAP@50: {results['map_50'].item():.4f}")
    print(f"  mAP@75: {results['map_75'].item():.4f}")


if __name__ == "__main__":
    main()
