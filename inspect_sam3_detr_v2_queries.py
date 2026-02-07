"""
Inspect SAM3 DETR v2 query counts and alignment.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor

from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from data.transforms import get_valid_transforms_DETR


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect SAM3 DETR v2 query counts")
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


def _get_config_value(config, *keys):
    for key in keys:
        if config is None:
            return None
        config = getattr(config, key, None)
    return config


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

    print("Loading Sam3Model...")
    model = Sam3Model.from_pretrained(args.checkpoint).to(device)
    model.eval()

    decoder_config = _get_config_value(model.config, "detr_decoder_config")
    num_queries_cfg = _get_config_value(model.config, "detr_decoder_config", "num_queries")
    hidden_size_cfg = _get_config_value(model.config, "detr_decoder_config", "hidden_size")
    print("Model config summary:")
    print(f"  decoder_hidden_size: {hidden_size_cfg}")
    print(f"  decoder_num_queries: {num_queries_cfg}")
    if decoder_config is None:
        print("  warning: detr_decoder_config not found on model.config")

    total_batches = 0
    total_dec_queries = []
    total_box_queries = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            pixel_values, input_ids, attention_mask, _targets, _orig_sizes = batch
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = None if attention_mask is None else attention_mask.to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            dec_last = outputs.decoder_hidden_states[-1]
            pred_boxes = outputs.pred_boxes

            dec_q = dec_last.shape[1]
            box_q = pred_boxes.shape[1]
            total_dec_queries.append(dec_q)
            total_box_queries.append(box_q)

            print(
                f"Batch {batch_idx}: decoder_queries={dec_q}, box_queries={box_q}, "
                f"delta={dec_q - box_q}"
            )
            print(
                f"  decoder_hidden_state shape: {tuple(dec_last.shape)}"
            )
            print(
                f"  pred_boxes shape: {tuple(pred_boxes.shape)} | "
                f"min={pred_boxes.min().item():.4f}, max={pred_boxes.max().item():.4f}"
            )

            total_batches += 1

    if total_batches == 0:
        print("No batches processed. Check dataset paths.")
        return

    print("\nSummary:")
    print(f"  batches: {total_batches}")
    print(f"  decoder_queries: {total_dec_queries}")
    print(f"  box_queries: {total_box_queries}")


if __name__ == "__main__":
    main()
