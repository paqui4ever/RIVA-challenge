#!/usr/bin/env python3
import argparse
import random
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add root to sys.path
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from utils.anchors import LearnableAnchorGenerator, FPNLearnableAnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from data.dataset import BethesdaDataset
from data.transforms import get_train_transforms_v2, get_valid_transforms

# Model-specific imports will be handled dynamically or imported here if no conflict
from models.cell_DINO_rcnn_v2 import build_cell_dino_fasterrcnn, cell_dino_resize_longest_side_and_pad_square
from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square
from models.cell_DINO_rcnn_v2_LoRA import build_cell_dino_fasterrcnn_lora
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection
from data.transforms import get_train_transforms_DETR_v2, get_valid_transforms_DETR_v2
from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from transformers import Sam3Processor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def preprocess_batch(images, targets, target_size: int, device: torch.device, resize_fn):
    processed_images = []
    processed_targets = []
    for img, tgt in zip(images, targets):
        img, tgt, _ = resize_fn(img, tgt, target_size=target_size)
        processed_images.append(img.to(device))
        processed_targets.append({k: v.to(device) for k, v in tgt.items()})
    return processed_images, processed_targets


def evaluate_map(model, loader, target_size: int, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, resize_fn):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for images, targets in loader:
            images = list(images)
            targets = list(targets)
            images, targets = preprocess_batch(images, targets, target_size, device, resize_fn)

            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)

            outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(outputs_cpu, targets_cpu)

    results = metric.compute()
    return results


def visualize_prediction(image, target, prediction, epoch, save_dir="."):
    """
    Visualizes the ground truth and predicted bounding boxes on the image.
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


def evaluate_map_detr(model, loader, device):
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

            results = model.predict(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                orig_sizes=orig_sizes,
                score_thresh=0.01 
            )
            
            targets_to_metric = []
            for i, t in enumerate(norm_targets):
                h, w = orig_sizes[i].tolist()
                tgt_boxes = t["boxes"].to(device)
                abs_boxes = tgt_boxes.clone()
                abs_boxes[:, [0, 2]] *= w
                abs_boxes[:, [1, 3]] *= h
                targets_to_metric.append({
                    "boxes": abs_boxes,
                    "labels": t["labels"].to(device)
                })
            metric.update(results, targets_to_metric)

    return metric.compute()


def visualize_prediction_detr(image_tensor, target, prediction, epoch, save_dir=".", mean=None, std=None):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    if mean is not None and std is not None:
        img_np = (img_np * np.array(std) + np.array(mean)) * 255.0
        img_np = img_np.clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_np.astype(np.uint8) if img_np.max() > 1.0 else img_np)

    h, w = img_np.shape[:2]
    
    if target is not None and 'boxes' in target:
        boxes = target['boxes'].detach().cpu().numpy()
        for box in boxes:
            if box.max() <= 1.0:
                 x1, y1, x2, y2 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
            else:
                 x1, y1, x2, y2 = box
            w_box, h_box = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='lime', facecolor='none', label='GT')
            ax.add_patch(rect)

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

def main():
    parser = argparse.ArgumentParser(description="Overfit models on a few images")
    parser.add_argument("--model", required=True, choices=['cell_dino', 'cell_dino_lora', 'sam3', 'sam3_detr_v2'], help="Model to train")
    parser.add_argument("--csv", required=True, help="Path to train.csv")
    parser.add_argument("--images", required=True, help="Path to training images directory")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to overfit")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--trainable-backbone", action="store_true", help="Unfreeze backbone")
    parser.add_argument("--use-train-transforms", action="store_true", help="Use training augmentations")
    parser.add_argument("--sam3-checkpoint", default="facebook/sam3", help="SAM3 checkpoint id or path (only for sam3 model)")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16", help="AMP dtype")
    parser.add_argument("--weighted-sampling", action="store_true", default=False, help="Use weighted sampling")
    parser.add_argument("--viz-freq", type=int, default=10, help="Frequency of visualization (epochs)")
    parser.add_argument("--learn-anchors-single", action="store_true", help="Learn anchor sizes with single scale")
    parser.add_argument("--learn-anchors-multiple", action="store_true", help="Learn anchor sizes with multiple scale")
    
    # LoRA specific args
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    df = pd.read_csv(args.csv)

    # Class weights logic
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
        try:
            classes_in_img = image_groups.get_group(img_name)['class_name'].values
            max_weight = max([class_weights.get(c, 1.0) for c in classes_in_img])
        except KeyError:
            max_weight = 1.0
        sample_weights.append(max_weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    if args.model == "sam3_detr_v2":
        processor = Sam3Processor.from_pretrained(args.sam3_checkpoint)
        train_transforms = get_train_transforms_DETR_v2(processor) if args.use_train_transforms else get_valid_transforms_DETR_v2(processor)
        val_transforms = get_valid_transforms_DETR_v2(processor)
        DatasetClass = BethesdaDatasetForSam3DETR
        loader_collate_fn = make_detr_collate_fn(processor)
    else:
        processor = None
        train_transforms = get_train_transforms_v2() if args.use_train_transforms else get_valid_transforms()
        val_transforms = get_valid_transforms()
        DatasetClass = BethesdaDataset
        loader_collate_fn = collate_fn

    full_train_ds = DatasetClass(csv_file=args.csv, root_dir=args.images, transforms=train_transforms)
    full_val_ds = DatasetClass(csv_file=args.csv, root_dir=args.images, transforms=val_transforms)

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
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=loader_collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate_fn)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate_fn)

    print(f"Using device: {device}")
    print(f"Selected indices: {indices}")
    print(f"AMP enabled: {use_amp} ({args.amp_dtype})")
    print(f"Model selected: {args.model}")

    # Model specific logic
    if args.model == 'cell_dino':
        model = build_cell_dino_fasterrcnn(
            num_classes_closed_set=8,
            trainable_backbone=args.trainable_backbone,
        )
        resize_fn = cell_dino_resize_longest_side_and_pad_square
        vis_dir_name = "overfit_visualizations_cell_dino"
        
        # Default anchor configs for cell_dino
        SIZES = ((83, 84), (94, 96), (110, 112), (115, 117))
        ASPECT_RATIOS = ((0.825, 1.0, 1.05),) * 4
        
    elif args.model == 'sam3':
        model = build_sam3_fasterrcnn(
            model_name_or_path=args.sam3_checkpoint,
            num_classes_closed_set=8,
            trainable_backbone=args.trainable_backbone,
        )
        resize_fn = sam3_resize_longest_side_and_pad_square
        vis_dir_name = "overfit_visualizations_sam3"
        
        # Default anchor configs for sam3 (active ones in current script)
        SIZES = ((71, 78), (92, 104), (123, 135), (158, 168))
        ASPECT_RATIOS = ((0.82, 1.0, 1.12),) * 4

    elif args.model == 'cell_dino_lora':
        model = build_cell_dino_fasterrcnn_lora(
            num_classes_closed_set=8,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        resize_fn = cell_dino_resize_longest_side_and_pad_square
        vis_dir_name = "overfit_visualizations_cell_dino_lora"

    elif args.model == 'sam3_detr_v2':
        model = Sam3ForClosedSetDetection(
            sam3_checkpoint=args.sam3_checkpoint,
            num_classes=8, 
            freeze_sam3=not args.trainable_backbone
        )
        model.build_criterion()
        resize_fn = None  # Using processor for sam3_detr_v2
        vis_dir_name = "overfit_visualizations_detr"

    model.to(device)
    target_size = model.backbone.target_size

    # Anchor generator updates
    if args.learn_anchors_single:
        print("Replacing RPN anchor generator with learnable version (single)...")
        model.rpn.anchor_generator = LearnableAnchorGenerator(init_size=98.0).to(device)

    if args.learn_anchors_multiple:
        print("Replacing RPN anchor generator with learnable version (multiple)...")
        # Use SIZES and ASPECT_RATIOS defined in model block above
        model.rpn.anchor_generator = FPNLearnableAnchorGenerator(SIZES, ASPECT_RATIOS).to(device)

        num_anchors_per_location = len(SIZES[0]) * len(ASPECT_RATIOS[0]) 
        in_channels = model.backbone.out_channels 
        model.rpn.head = RPNHead(in_channels, num_anchors_per_location).to(device)

    # Optimizer
    if args.learn_anchors_single or args.learn_anchors_multiple:
        anchor_params = []
        base_params = []

        for name, param in model.named_parameters():
            if 'rpn.anchor_generator' in name:
                anchor_params.append(param)
            else:
                base_params.append(param)

        optimizer = AdamW(
            [
                {'params': base_params, 'lr': args.lr, 'weight_decay': 1e-5},
                {'params': anchor_params, 'lr': args.lr, 'weight_decay': 0.0}
            ]
        )
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=args.lr, weight_decay=1e-5)

    scaler = GradScaler(enabled=use_amp)

    # Visualization directory
    vis_dir = Path(vis_dir_name)
    vis_dir.mkdir(exist_ok=True)

    # DETR uses specific stats for denormalization
    if args.model == "sam3_detr_v2":
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
    else:
        image_mean = None
        image_std = None

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            
            if args.model == "sam3_detr_v2":
                pixel_values, input_ids, attention_mask, norm_targets, orig_sizes = batch_data
                pixel_values = pixel_values.to(device)
                if input_ids is not None: input_ids = input_ids.to(device)
                if attention_mask is not None: attention_mask = attention_mask.to(device)
                targets_device = [{k: v.to(device) for k, v in t.items()} for t in norm_targets]

                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        targets=targets_device
                    )
                    loss = outputs["losses"]["loss_total"]
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()

            else:
                images, targets = batch_data
                images = list(images)
                targets = list(targets)
                images, targets = preprocess_batch(images, targets, target_size, device, resize_fn)

                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                    loss_dict = model(images, targets)
                    loss = sum(v for v in loss_dict.values())

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item())
        
        # Visualization
        if (epoch + 1) % args.viz_freq == 0:
            model.eval()
            with torch.no_grad():
                if args.model == "sam3_detr_v2":
                    # Assume pixel_values, targets_device, orig_sizes exist from last loop iteration
                    viz_img = pixel_values[0]
                    viz_tgt = targets_device[0]
                    viz_orig_size = orig_sizes[0].to(device)

                    with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                        viz_res = model.predict(
                            pixel_values=viz_img.unsqueeze(0),
                            input_ids=input_ids[0:1] if input_ids is not None else None,
                            attention_mask=attention_mask[0:1] if attention_mask is not None else None,
                            orig_sizes=viz_orig_size.unsqueeze(0)
                        )[0]
                    
                    viz_tgt_abs = viz_tgt.copy()
                    h_orig, w_orig = viz_orig_size.tolist()
                    boxes_abs = viz_tgt["boxes"].clone()
                    boxes_abs[:, [0, 2]] *= w_orig
                    boxes_abs[:, [1, 3]] *= h_orig
                    viz_tgt_abs["boxes"] = boxes_abs
                    
                    visualize_prediction_detr(viz_img, viz_tgt_abs, viz_res, epoch + 1, save_dir=vis_dir, mean=image_mean, std=image_std)
                else:
                    viz_img = images[0]
                    viz_tgt = targets[0]
                    
                    with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                        viz_preds = model([viz_img])
                    
                    visualize_prediction(viz_img, viz_tgt, viz_preds[0], epoch + 1, save_dir=vis_dir)
            
            model.train()

        avg_loss = total_loss / max(1, len(train_loader))
        
        if args.model == "sam3_detr_v2":
            results = evaluate_map_detr(model, val_loader, device)
        else:
            results = evaluate_map(model, val_loader, target_size, device, use_amp, amp_dtype, resize_fn)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss {avg_loss:.4f} | "
            f"mAP {results['map'].item():.4f} | "
            f"mAP@50 {results['map_50'].item():.4f}"
        )


if __name__ == "__main__":
    main()
