"""Debug SAM3 DETR v2 box format assumptions (cxcywh vs xyxy)."""

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import Sam3Processor

from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from data.transforms import get_valid_transforms_DETR
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection, box_cxcywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect SAM3 DETR v2 box format")
    parser.add_argument(
        "--csv",
        type=str,
        default="/local_data/RIVA/annotations/annotations/train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="/local_data/RIVA/images/images/train",
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
        help="Batch size for inspection",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches to inspect",
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
    return parser.parse_args()


def _as_float(losses, key):
    return float(losses[key].detach().cpu().item())


def _init_stats():
    return {
        "loss_total": [],
        "loss_ce": [],
        "loss_bbox": [],
        "loss_giou": [],
    }


def _record(stats, losses):
    for k in stats:
        stats[k].append(_as_float(losses, k))


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _invalid_fraction(boxes_xyxy: torch.Tensor) -> float:
    if boxes_xyxy.numel() == 0:
        return 0.0
    invalid = (boxes_xyxy[..., 2] < boxes_xyxy[..., 0]) | (boxes_xyxy[..., 3] < boxes_xyxy[..., 1])
    return float(invalid.float().mean().item())


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
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Loading Sam3ForClosedSetDetection...")
    model = (
        Sam3ForClosedSetDetection(
            sam3_checkpoint=args.checkpoint,
            num_classes=args.num_classes,
            freeze_sam3=False,
        )
        .build_criterion(
            class_cost=1.0,
            bbox_cost=5.0,
            giou_cost=2.0,
            eos_coef=0.1,
            loss_ce_w=1.0,
            loss_bbox_w=5.0,
            loss_giou_w=2.0,
        )
        .to(device)
    )

    if args.model_checkpoint:
        print(f"Loading model checkpoint: {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.eval()

    stats_cxcywh = _init_stats()
    stats_xyxy = _init_stats()
    invalid_cxcywh = []
    invalid_xyxy = []
    wins = {"cxcywh": 0, "xyxy": 0, "tie": 0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            pixel_values, input_ids, attention_mask, targets, _orig_sizes = batch
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = None if attention_mask is None else attention_mask.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model.sam3(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            dec_last = outputs.decoder_hidden_states[-1]
            pred_boxes = outputs.pred_boxes

            box_q = pred_boxes.shape[1]
            dec_last = dec_last[:, :box_q, :]
            logits = model.class_embed(dec_last)

            boxes_from_cxcywh = box_cxcywh_to_xyxy(pred_boxes)
            boxes_as_xyxy = pred_boxes

            losses_cx = model.criterion(logits, boxes_from_cxcywh, targets)
            losses_xy = model.criterion(logits, boxes_as_xyxy, targets)

            _record(stats_cxcywh, losses_cx)
            _record(stats_xyxy, losses_xy)
            invalid_cxcywh.append(_invalid_fraction(boxes_from_cxcywh))
            invalid_xyxy.append(_invalid_fraction(boxes_as_xyxy))

            l_cx = _as_float(losses_cx, "loss_total")
            l_xy = _as_float(losses_xy, "loss_total")

            if abs(l_cx - l_xy) < 1e-6:
                wins["tie"] += 1
            elif l_cx < l_xy:
                wins["cxcywh"] += 1
            else:
                wins["xyxy"] += 1

            print(
                f"Batch {batch_idx}: loss_total cxcywh={l_cx:.4f} | xyxy={l_xy:.4f}"
            )
            print(
                f"  invalid fraction: cxcywh={invalid_cxcywh[-1]:.4f} | "
                f"xyxy={invalid_xyxy[-1]:.4f}"
            )

    print("\nSummary:")
    if stats_cxcywh["loss_total"]:
        print(
            "  cxcywh mean losses: "
            f"total={_mean(stats_cxcywh['loss_total']):.4f}, "
            f"ce={_mean(stats_cxcywh['loss_ce']):.4f}, "
            f"bbox={_mean(stats_cxcywh['loss_bbox']):.4f}, "
            f"giou={_mean(stats_cxcywh['loss_giou']):.4f}"
        )
    if stats_xyxy["loss_total"]:
        print(
            "  xyxy mean losses: "
            f"total={_mean(stats_xyxy['loss_total']):.4f}, "
            f"ce={_mean(stats_xyxy['loss_ce']):.4f}, "
            f"bbox={_mean(stats_xyxy['loss_bbox']):.4f}, "
            f"giou={_mean(stats_xyxy['loss_giou']):.4f}"
        )
    if invalid_cxcywh:
        print(f"  invalid fraction mean (cxcywh): {sum(invalid_cxcywh) / len(invalid_cxcywh):.4f}")
    if invalid_xyxy:
        print(f"  invalid fraction mean (xyxy): {sum(invalid_xyxy) / len(invalid_xyxy):.4f}")
    print(f"  wins: {wins}")


if __name__ == "__main__":
    main()
