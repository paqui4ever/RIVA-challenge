"""Debug SAM3 DETR v2 query alignment by comparing loss slices."""

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import Sam3Processor

from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from data.transforms import get_valid_transforms_DETR
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection, box_cxcywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect SAM3 DETR v2 query alignment")
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
        default=3,
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

    stats_slice0 = _init_stats()
    stats_slice1 = _init_stats()
    wins = {"slice0": 0, "slice1": 0, "tie": 0}

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
            boxes = box_cxcywh_to_xyxy(pred_boxes)

            dec_q = dec_last.shape[1]
            box_q = boxes.shape[1]

            print(
                f"Batch {batch_idx}: decoder_queries={dec_q}, box_queries={box_q}, "
                f"delta={dec_q - box_q}"
            )

            losses0 = None
            losses1 = None

            if dec_q >= box_q:
                dec_slice0 = dec_last[:, :box_q, :]
                logits0 = model.class_embed(dec_slice0)
                losses0 = model.criterion(logits0, boxes, targets)
                _record(stats_slice0, losses0)

            if dec_q >= box_q + 1:
                dec_slice1 = dec_last[:, 1:box_q + 1, :]
                logits1 = model.class_embed(dec_slice1)
                losses1 = model.criterion(logits1, boxes, targets)
                _record(stats_slice1, losses1)

            if losses0 and losses1:
                l0 = _as_float(losses0, "loss_total")
                l1 = _as_float(losses1, "loss_total")
                if abs(l0 - l1) < 1e-6:
                    wins["tie"] += 1
                elif l0 < l1:
                    wins["slice0"] += 1
                else:
                    wins["slice1"] += 1

                print(
                    f"  slice0 loss_total={l0:.4f} | "
                    f"slice1 loss_total={l1:.4f}"
                )
            elif losses0:
                print(f"  slice0 loss_total={_as_float(losses0, 'loss_total'):.4f}")
            else:
                print("  No valid slice to compare")

            print(
                f"  pred_boxes range: min={pred_boxes.min().item():.4f}, "
                f"max={pred_boxes.max().item():.4f}"
            )

    print("\nSummary:")
    if stats_slice0["loss_total"]:
        print(
            "  slice0 mean losses: "
            f"total={_mean(stats_slice0['loss_total']):.4f}, "
            f"ce={_mean(stats_slice0['loss_ce']):.4f}, "
            f"bbox={_mean(stats_slice0['loss_bbox']):.4f}, "
            f"giou={_mean(stats_slice0['loss_giou']):.4f}"
        )
    if stats_slice1["loss_total"]:
        print(
            "  slice1 mean losses: "
            f"total={_mean(stats_slice1['loss_total']):.4f}, "
            f"ce={_mean(stats_slice1['loss_ce']):.4f}, "
            f"bbox={_mean(stats_slice1['loss_bbox']):.4f}, "
            f"giou={_mean(stats_slice1['loss_giou']):.4f}"
        )
    if stats_slice0["loss_total"] and stats_slice1["loss_total"]:
        print(f"  wins: {wins}")


if __name__ == "__main__":
    main()
