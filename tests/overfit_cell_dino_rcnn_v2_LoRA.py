#!/usr/bin/env python3
import argparse
import random
import pandas as pd

import numpy as np
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from data.dataset import BethesdaDataset
from data.transforms import get_train_transforms_v2, get_valid_transforms
from models.cell_DINO_rcnn_v2 import cell_dino_resize_longest_side_and_pad_square
from models.cell_DINO_rcnn_v2_LoRA import build_cell_dino_fasterrcnn_lora


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def preprocess_batch(images, targets, target_size: int, device: torch.device):
    processed_images = []
    processed_targets = []
    for img, tgt in zip(images, targets):
        img, tgt, _ = cell_dino_resize_longest_side_and_pad_square(img, tgt, target_size=target_size)
        processed_images.append(img.to(device))
        processed_targets.append({k: v.to(device) for k, v in tgt.items()})
    return processed_images, processed_targets


def evaluate_map(model, loader, target_size: int, device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for images, targets in loader:
            images = list(images)
            targets = list(targets)
            images, targets = preprocess_batch(images, targets, target_size, device)

            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)

            outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(outputs_cpu, targets_cpu)

    results = metric.compute()
    return results


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_prediction(image, target, prediction, epoch, save_dir="."):
    """
    Visualizes the ground truth and predicted bounding boxes on the image.
    Args:
        image: Tensor (C, H, W).
        target: Dict with 'boxes' (N, 4).
        prediction: Dict with 'boxes' (M, 4), 'scores' (M,).
        epoch: Current epoch number.
        save_dir: Directory to save the visualization.
    """
    # Move to CPU and numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_np)
    
    # Draw Ground Truth in Green
    if target is not None:
        boxes = target['boxes'].cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none', label='GT')
            ax.add_patch(rect)
            
    # Draw Predictions in Red
    if prediction is not None:
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5: # Threshold
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Pred')
                ax.add_patch(rect)
                ax.text(x1, y1, f"{score:.2f}", color='white', fontsize=8, backgroundcolor='red')

    plt.title(f"Epoch {epoch}")
    plt.axis('off')
    
    # Save
    save_path = Path(save_dir) / f"vis_epoch_{epoch}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overfit CELL-DINO RCNN v2 LoRA on a few images")
    parser.add_argument("--csv", required=True, help="Path to train.csv")
    parser.add_argument("--images", required=True, help="Path to training images directory")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to overfit")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # parser.add_argument("--trainable-backbone", action="store_true", help="Unfreeze backbone (Ignored for LoRA which sets specific parts trainable)")
    parser.add_argument("--use-train-transforms", action="store_true", help="Use training augmentations")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16", help="AMP dtype")
    parser.add_argument("--weighted-sampling", action="store_true", default=False, help="Use weighted sampling")
    
    # LoRA specific args
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--viz-freq", type=int, default=10, help="Frequency of visualization (epochs)")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    df = pd.read_csv(args.csv)

    # INFL is the most common, so it gets weight 1.0. 
    # ASCUS is the rarest, so it gets weight 21.0.
    class_weights = {
        'INFL': 1.00,
        'NILM': 1.14,
        'LSIL': 2.76,
        'HSIL': 3.54,
        'SCC':  4.03,
        'ENDO': 6.53,
        'ASCH': 13.80,
        'ASCUS': 21.06
    }

    image_groups = df.groupby('image_filename')
    unique_images = df['image_filename'].unique()

    sample_weights = []

    for img_name in unique_images:
        # Get all classes present in this single image
        try:
            classes_in_img = image_groups.get_group(img_name)['class_name'].values
            max_weight = max([class_weights.get(c, 1.0) for c in classes_in_img])
        except KeyError:
             max_weight = 1.0
             
        sample_weights.append(max_weight)

    # Convert to tensor
    sample_weights = torch.DoubleTensor(sample_weights)

    train_transforms = get_train_transforms_v2() if args.use_train_transforms else get_valid_transforms()
    val_transforms = get_valid_transforms()

    full_train_ds = BethesdaDataset(csv_file=args.csv, root_dir=args.images, transforms=train_transforms)
    full_val_ds = BethesdaDataset(csv_file=args.csv, root_dir=args.images, transforms=val_transforms)

    total = len(full_train_ds)
    if args.num_images > total:
        print(f"Warning: num-images={args.num_images} exceeds dataset size {total}. Using all images.")
        args.num_images = total

    if args.random_sample:
        indices = random.sample(range(total), args.num_images)
    else:
        indices = list(range(args.num_images))

    train_ds = Subset(full_train_ds, indices)
    val_ds = Subset(full_val_ds, indices)

    batch_size = min(args.batch_size, args.num_images)

    if args.weighted_sampling and args.random_sample: 
        subset_weights = sample_weights[indices] 
        sampler = WeightedRandomSampler(
            weights=subset_weights,
            num_samples=len(subset_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Using device: {device}")
    print(f"Selected indices: {indices}")
    print(f"AMP enabled: {use_amp} ({args.amp_dtype})")
    print(f"LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}, Dropout: {args.lora_dropout}")

    # Build Cell-DINO LoRA model
    model = build_cell_dino_fasterrcnn_lora(
        num_classes_closed_set=8,
        # trainable_backbone is handled internally by Lora logic (usually freezes base + adds LoRA)
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(params)}")
    
    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=use_amp)

    target_size = model.backbone.target_size

    # Create a directory for visualizations
    vis_dir = Path("overfit_visualizations_cell_dino_lora")
    vis_dir.mkdir(exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(images)
            targets = list(targets)
            images, targets = preprocess_batch(images, targets, target_size, device)

            optimizer.zero_grad()

            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(v for v in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
        
        # Visualization
        if (epoch + 1) % args.viz_freq == 0:
            model.eval()
            with torch.no_grad():
                # Visualize the first image of the last batch in the loop
                viz_img = images[0]
                viz_tgt = targets[0]
                
                # Get predictions
                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                    viz_preds = model([viz_img])
                
                visualize_prediction(viz_img, viz_tgt, viz_preds[0], epoch + 1, save_dir=vis_dir)
            
            model.train()

        avg_loss = total_loss / max(1, len(train_loader))
        results = evaluate_map(model, val_loader, target_size, device, use_amp, amp_dtype)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss {avg_loss:.4f} | "
            f"mAP {results['map'].item():.4f} | "
            f"mAP@50 {results['map_50'].item():.4f}"
        )


if __name__ == "__main__":
    main()
