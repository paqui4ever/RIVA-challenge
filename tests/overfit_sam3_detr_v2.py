#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from data.transforms import get_train_transforms_DETR_v2, get_valid_transforms_DETR_v2
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection
from transformers import Sam3Processor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def visualize_prediction(image_tensor, target, prediction, epoch, save_dir=".", mean=None, std=None):
    """
    Visualizes the ground truth and predicted bounding boxes on the image.
    image_tensor: (C, H, W) normalized tensor
    """
    # Denormalize image for visualization if mean/std provided
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    if mean is not None and std is not None:
        img_np = (img_np * np.array(std) + np.array(mean)) * 255.0
        img_np = img_np.clip(0, 255).astype(np.uint8)
    else:
        # Assuming it might be 0-1 or 0-255 already if no stats provided, 
        # but SAM3 transforms usually normalize.
        # If it's float 0-1, just scale to 255 for display safety or keep 0-1.
        # Matplotlib handles 0-1 floats.
        pass

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_np.astype(np.uint8) if img_np.max() > 1.0 else img_np)

    # Draw Ground Truth in Green
    if target is not None and 'boxes' in target:
        boxes = target['boxes'].detach().cpu().numpy()
        # Boxes from target in this script might be normalized [0,1] or absolute?
        # In this script's loop, we pass 'target' which has absolute boxes for 'evaluate_map'?
        # Or normalized from 'preprocess'?
        # Let's handle both. If max value <= 1.0, assume normalized.
        
        # Actually, in this script, we'll ensure we pass absolute boxes for visualization to be easy
        h, w = img_np.shape[:2]
        
        for box in boxes:
            if box.max() <= 1.0:
                 x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
            else:
                 x1, y1, x2, y2 = box
            
            w_box, h_box = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='lime', facecolor='none', label='GT')
            ax.add_patch(rect)

    # Draw Predictions in Red
    if prediction is not None:
        boxes = prediction['boxes'].detach().cpu().numpy()
        scores = prediction['scores'].detach().cpu().numpy()
        labels = prediction['labels'].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:
                if box.max() <= 1.0:
                     x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
                else:
                     x1, y1, x2, y2 = box
                
                w_box, h_box = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='red', facecolor='none', label='Pred')
                ax.add_patch(rect)
                ax.text(x1, y1, f"{score:.2f}", color='white', fontsize=8, backgroundcolor='red')

    plt.title(f"Epoch {epoch}")
    plt.axis('off')

    save_path = Path(save_dir) / f"vis_epoch_{epoch}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def evaluate_map(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    with torch.no_grad():
        for batch in loader:
            pixel_values, input_ids, attention_mask, norm_targets, orig_sizes = batch
            pixel_values = pixel_values.to(device)
            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            orig_sizes = orig_sizes.to(device)

            # Get predictions (absolute coordinates)
            results = model.predict(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                orig_sizes=orig_sizes,
                score_thresh=0.01 # Low threshold for evaluation
            )
            
            # Prepare targets for metric (absolute coordinates)
            # norm_targets has normalized boxes. We need to recover absolute using orig_sizes.
            params_to_metric = []
            targets_to_metric = []
            
            for i, t in enumerate(norm_targets):
                h, w = orig_sizes[i].tolist()
                tgt_boxes = t["boxes"].to(device)
                
                # Un-normalize
                abs_boxes = tgt_boxes.clone()
                abs_boxes[:, [0, 2]] *= w
                abs_boxes[:, [1, 3]] *= h
                
                targets_to_metric.append({
                    "boxes": abs_boxes,
                    "labels": t["labels"].to(device)
                })
                
                # Move prediction to cpu for metric if needed, or keep on device
                # MAP metric handles both
                
            # Update metric
            # results is list of dicts: scores, labels, boxes
            metric.update(results, targets_to_metric)

    return metric.compute()


def main():
    parser = argparse.ArgumentParser(description="Overfit SAM3 DETR v2 on a few images")
    parser.add_argument("--csv", required=True, help="Path to train.csv")
    parser.add_argument("--images", required=True, help="Path to training images directory")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to overfit")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--trainable-backbone", action="store_true", help="Unfreeze SAM3 backbone")
    parser.add_argument("--sam3-checkpoint", default="facebook/sam3", help="SAM3 checkpoint id or path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Processors & Transforms
    processor = Sam3Processor.from_pretrained(args.sam3_checkpoint)
    train_transforms = get_train_transforms_DETR_v2(processor)
    val_transforms = get_valid_transforms_DETR_v2(processor)

    # Dataset
    # We use same dataset class for train/val, just different transforms
    full_train_ds = BethesdaDatasetForSam3DETR(csv_file=args.csv, root_dir=args.images, transforms=train_transforms)
    full_val_ds = BethesdaDatasetForSam3DETR(csv_file=args.csv, root_dir=args.images, transforms=val_transforms)

    total = len(full_train_ds)
    if args.num_images > total:
        print(f"Warning: num-images={args.num_images} > total={total}. Using all images.")
        args.num_images = total

    if args.random_sample:
        indices = random.sample(range(total), args.num_images)
    else:
        indices = list(range(args.num_images))
    
    print(f"Selected indices: {indices}")

    train_ds = Subset(full_train_ds, indices)
    val_ds = Subset(full_val_ds, indices)

    collate_fn = make_detr_collate_fn(processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    print("Building model...")
    model = Sam3ForClosedSetDetection(
        sam3_checkpoint=args.sam3_checkpoint,
        num_classes=8, 
        freeze_sam3=not args.trainable_backbone
    )
    model.build_criterion()
    model.to(device)

    # Optimizer
    # Different LR for backbone vs head is common, but for overfitting simple is fine
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Vis dir
    vis_dir = Path("overfit_visualizations_detr")
    vis_dir.mkdir(exist_ok=True)
    
    # Save standard stats for denormalization in vis
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            pixel_values, input_ids, attention_mask, norm_targets, orig_sizes = batch
            
            pixel_values = pixel_values.to(device)
            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # norm_targets is list of dicts. Move tensors to device.
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in norm_targets]

            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                targets=targets_device
            )
            
            # Loss is already computed in forward if targets provided
            losses = outputs["losses"]
            loss = losses["loss_total"]
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Visualization
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Visualize first item of last batch
                # To be consistent, let's pick info from the loop variables
                viz_img = pixel_values[0] # (3, H, W)
                viz_tgt = targets_device[0] # normalized boxes
                viz_orig_size = orig_sizes[0].to(device) # (H, W)

                # Predict
                viz_res = model.predict(
                     pixel_values=viz_img.unsqueeze(0),
                     input_ids=input_ids[0:1] if input_ids is not None else None,
                     attention_mask=attention_mask[0:1] if attention_mask is not None else None,
                     orig_sizes=viz_orig_size.unsqueeze(0)
                )[0]
                
                # For visualization, we need absolute GT too
                # viz_tgt has normalized boxes.
                viz_tgt_abs = viz_tgt.copy() # Shallow copy dict
                h_orig, w_orig = viz_orig_size.tolist()
                boxes_n = viz_tgt["boxes"].clone()
                boxes_abs = boxes_n.clone()
                boxes_abs[:, [0, 2]] *= w_orig
                boxes_abs[:, [1, 3]] *= h_orig
                viz_tgt_abs["boxes"] = boxes_abs
                
                visualize_prediction(
                    viz_img, 
                    viz_tgt_abs, 
                    viz_res, 
                    epoch + 1, 
                    save_dir=vis_dir,
                    mean=image_mean,
                    std=image_std
                )
        
        avg_loss = total_loss / max(1, len(train_loader))
        
        # Evaluate
        # Note: evaluating on valid_loader which is subset of same images
        results = evaluate_map(model, val_loader, device)
        
        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss {avg_loss:.4f} | "
            f"mAP {results['map'].item():.4f} | "
            f"mAP@50 {results['map_50'].item():.4f}"
        )

if __name__ == "__main__":
    main()
