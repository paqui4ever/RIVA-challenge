"""
Training script for Sam3ForClosedSetDetection (SAM3 + Closed-Set DETR Head).

This model (Cut A architecture):
  - Uses SAM3 perception encoder + DETR encoder/decoder
  - Replaces open-vocabulary classification with a learned linear classifier
  - Uses Hungarian matching + set-based loss (CE + L1 + GIoU)
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

from transformers import Sam3Processor

from data.transforms import get_train_transforms_DETR, get_valid_transforms_DETR
from data.detr_v2_utils import BethesdaDatasetForSam3DETR, make_detr_collate_fn
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection


# ----------------------------
# Argument Parser
# ----------------------------
parser = argparse.ArgumentParser(description="Train Sam3ForClosedSetDetection (SAM3 + DETR v2)")
parser.add_argument(
    "--freeze_sam3",
    action="store_true",
    help="If set, freeze the SAM3 backbone (only train the classification head)"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default='/local_data/RIVA/checkpoints/detr_v2',
    help="Directory to save checkpoints"
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size for training (default: 8, adjust based on GPU memory)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of training epochs"
)
parser.add_argument(
    "--lr_backbone",
    type=float,
    default=5e-6,
    help="Learning rate for SAM3 backbone"
)
parser.add_argument(
    "--lr_head",
    type=float,
    default=5e-4,
    help="Learning rate for classification head"
)
parser.add_argument(
    "--use_focal",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use focal loss for classification (default: True)"
)
parser.add_argument(
    "--focal_gamma",
    type=float,
    default=2.0,
    help="Focal loss gamma parameter"
)
parser.add_argument(
    "--focal_alpha",
    type=float,
    default=0.25,
    help="Focal loss alpha for foreground vs background"
)
parser.add_argument(
    "--score_thresh",
    type=float,
    default=0.0,
    help="Score threshold for inference during validation"
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)"
)
args = parser.parse_args()

# ----------------------------
# Configuration
# ----------------------------
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(CHECKPOINT_DIR, 'tensorboard'))

# Mixed precision settings
USE_AMP = True

# Data paths
CSV_PATH_TRAIN = '/local_data/RIVA/annotations/annotations/train.csv'
CSV_PATH_VAL = '/local_data/RIVA/annotations/annotations/val.csv'
TRAIN_PATH = '/local_data/RIVA/images/images/train'
VAL_PATH = '/local_data/RIVA/images/images/val'

# Model settings
SAM3_CHECKPOINT = "facebook/sam3"
NUM_CLASSES = 8  # 8 Bethesda classes (model adds +1 for no-object internally)

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Initialize Sam3Processor and Transforms
# ----------------------------
print("Loading Sam3Processor...")
processor = Sam3Processor.from_pretrained(SAM3_CHECKPOINT)

# Create transforms with normalization using SAM3's mean/std
TARGET_SIZE = 1008
train_transforms = get_train_transforms_DETR(processor, size=TARGET_SIZE)
val_transforms = get_valid_transforms_DETR(processor, size=TARGET_SIZE)

# ----------------------------
# Initialize Datasets
# ----------------------------
print("Initializing Datasets for Sam3ForClosedSetDetection...")
train_ds = BethesdaDatasetForSam3DETR(
    csv_file=CSV_PATH_TRAIN,
    root_dir=TRAIN_PATH,
    transforms=train_transforms
)
val_ds = BethesdaDatasetForSam3DETR(
    csv_file=CSV_PATH_VAL,
    root_dir=VAL_PATH,
    transforms=val_transforms
)

# ----------------------------
# Validate label ranges (CRITICAL for DETR)
# ----------------------------
print(f"\n{'='*60}")
print("VALIDATING LABEL RANGES (DETR expects 0-indexed labels)")
print(f"{'='*60}")

# Check a sample of the dataset to verify label indexing
import random
sample_indices = random.sample(range(len(train_ds)), min(50, len(train_ds)))
all_labels = []
for idx in sample_indices:
    _, target = train_ds[idx]
    all_labels.extend(target["labels"].tolist())

if all_labels:
    min_label, max_label = min(all_labels), max(all_labels)
    unique_labels = sorted(set(all_labels))
    print(f"  Sample labels found: {unique_labels}")
    print(f"  Label range: [{min_label}, {max_label}]")
    print(f"  NUM_CLASSES = {NUM_CLASSES}")
    print(f"  No-object index = {NUM_CLASSES} (should NOT appear in labels)")

    # Critical validation
    if max_label >= NUM_CLASSES:
        raise ValueError(
            f"FATAL: Found label {max_label} >= NUM_CLASSES ({NUM_CLASSES})!\n"
            f"This will collide with the no-object class index.\n"
            f"Labels must be in [0, {NUM_CLASSES-1}]. Found: {unique_labels}\n"
            f"Check if your CSV uses 1-indexed classes and ensure proper conversion."
        )
    if min_label < 0:
        raise ValueError(
            f"FATAL: Found negative label {min_label}!\n"
            f"Labels must be in [0, {NUM_CLASSES-1}]. Found: {unique_labels}"
        )
    print(f"  ✓ Labels are valid (0-indexed, in range [0, {NUM_CLASSES-1}])")
else:
    print("  WARNING: No labels found in sample - dataset may be empty")

print(f"{'='*60}\n")

# ----------------------------
# Initialize DataLoaders
# ----------------------------
collate_fn = make_detr_collate_fn(processor, target_size=TARGET_SIZE)

BATCH_SIZE = args.batch_size

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# ----------------------------
# Model Initialization
# ----------------------------
print(f"Loading Sam3ForClosedSetDetection (freeze_sam3={args.freeze_sam3})...")
model = (
    Sam3ForClosedSetDetection(
        sam3_checkpoint=SAM3_CHECKPOINT,
        num_classes=NUM_CLASSES,
        freeze_sam3=args.freeze_sam3
    )
    .build_criterion(
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        eos_coef=0.1,
        loss_ce_w=1.0,
        loss_bbox_w=5.0,
        loss_giou_w=2.0,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
    .to(device)
)

# ----------------------------
# Optimizer with Differential Learning Rates
# ----------------------------
if args.freeze_sam3:
    # Only train classification head
    optimizer = AdamW(model.class_embed.parameters(), lr=args.lr_head, weight_decay=1e-4)
else:
    # Different LR for backbone (lower) and head (higher)
    optimizer = AdamW([
        {"params": model.sam3.parameters(), "lr": args.lr_backbone},
        {"params": model.class_embed.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

# ----------------------------
# Learning Rate Scheduler
# ----------------------------
num_epochs = args.epochs
# Account for gradient accumulation: scheduler steps only when optimizer steps
num_batches_per_epoch = len(train_loader)
optimizer_steps_per_epoch = (num_batches_per_epoch + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
total_steps = num_epochs * optimizer_steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# ----------------------------
# Mixed Precision Scaler
# ----------------------------
scaler = GradScaler(enabled=USE_AMP)

# ----------------------------
# Resume from Checkpoint
# ----------------------------
start_epoch = 0
global_step = 0
best_map = 0.0

if args.resume and os.path.isfile(args.resume):
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint.get('global_step', 0)
    best_map = checkpoint.get('best_map', 0.0)
    print(f"Resumed from epoch {start_epoch}, global_step {global_step}, best_map {best_map:.4f}")

# ----------------------------
# Training Loop
# ----------------------------
ACCUM_STEPS = args.gradient_accumulation_steps
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUM_STEPS

print(f"\nStarting Training for {num_epochs} epochs...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation steps: {ACCUM_STEPS}")
print(f"  Effective batch size: {EFFECTIVE_BATCH_SIZE}")
print(f"  LR backbone: {args.lr_backbone}")
print(f"  LR head: {args.lr_head}")
print(f"  Freeze SAM3: {args.freeze_sam3}")

for epoch in range(start_epoch, num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")

    # --- TRAINING ---
    model.train()
    total_loss = 0.0
    epoch_losses = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        pixel_values, input_ids, attention_mask, targets, orig_sizes = batch

        # Mixed precision forward pass
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            outputs = model(pixel_values=pixel_values.to(device),
                            input_ids=input_ids.to(device),
                            attention_mask=None if attention_mask is None else attention_mask.to(device),
                            targets=[{k: v.to(device) for k, v in t.items()} for t in targets])
            losses = outputs["losses"]
            loss = losses["loss_total"]
            # Normalize loss for gradient accumulation
            loss = loss / ACCUM_STEPS

        # Backward pass (accumulate gradients)
        scaler.scale(loss).backward()

        # Only step optimizer every ACCUM_STEPS batches
        if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step scheduler after each optimizer step
            scheduler.step()

        # Accumulate losses (use original loss scale for logging)
        total_loss += loss.item() * ACCUM_STEPS
        for k in epoch_losses:
            if k in losses:
                epoch_losses[k] += losses[k].item()

        # TensorBoard logging
        writer.add_scalar("Losses/total_train", loss.item(), global_step)
        writer.add_scalar("Losses/train_ce", losses["loss_ce"].item(), global_step)
        writer.add_scalar("Losses/train_bbox", losses["loss_bbox"].item(), global_step)
        writer.add_scalar("Losses/train_giou", losses["loss_giou"].item(), global_step)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ce': f"{losses['loss_ce'].item():.4f}",
            'bbox': f"{losses['loss_bbox'].item():.4f}",
            'giou': f"{losses['loss_giou'].item():.4f}"
        })

        global_step += 1

    # Log learning rate
    writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

    avg_loss = total_loss / len(train_loader)
    print(f"\nAverage Training Loss: {avg_loss:.4f}")
    for k, v in epoch_losses.items():
        print(f"  {k}: {v / len(train_loader):.4f}")

    # --- VALIDATION ---
    print("\nValidating...")
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", class_metrics=True)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            pixel_values, input_ids, attention_mask, targets, orig_sizes = batch
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = None if attention_mask is None else attention_mask.to(device)
            orig_sizes = orig_sizes.to(device)

            # Get predictions using the model's predict method
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                predictions = model.predict(
                    pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    orig_sizes=orig_sizes,
                    score_thresh=args.score_thresh,
                    max_detections=100
                )

            # Convert targets to absolute coordinates for metric computation
            preds_cpu = []
            targets_cpu = []

            for b in range(len(predictions)):
                # Predictions are already in absolute coordinates from model.predict()
                pred_boxes = predictions[b]['boxes'].cpu()

                # Sanity check: predictions should be in valid absolute coordinate range
                h, w = orig_sizes[b].tolist()
                if pred_boxes.numel() > 0:
                    assert pred_boxes[:, [0, 2]].max() <= w * 1.01, \
                        f"Prediction x-coords exceed image width: max={pred_boxes[:, [0,2]].max():.1f}, w={w}"
                    assert pred_boxes[:, [1, 3]].max() <= h * 1.01, \
                        f"Prediction y-coords exceed image height: max={pred_boxes[:, [1,3]].max():.1f}, h={h}"

                preds_cpu.append({
                    'boxes': pred_boxes,
                    'scores': predictions[b]['scores'].cpu(),
                    'labels': predictions[b]['labels'].cpu()
                })

                # Convert normalized target boxes to absolute coordinates
                # GT boxes are in [0,1] relative to target_size (1008x1008 hard-resized)
                # Mapping: normalized_coord * orig_size = original_coord
                tgt_boxes = targets[b]['boxes'].clone()
                tgt_boxes[:, [0, 2]] *= w  # x coords × width
                tgt_boxes[:, [1, 3]] *= h  # y coords × height

                targets_cpu.append({
                    'boxes': tgt_boxes.cpu(),
                    'labels': targets[b]['labels'].cpu()
                })

            metric.update(preds_cpu, targets_cpu)

    results = metric.compute()
    current_map = results['map'].item()
    map_50 = results['map_50'].item()
    map_75 = results['map_75'].item()

    writer.add_scalar("Validation/mAP_50_95", current_map, epoch)
    writer.add_scalar("Validation/mAP_50", map_50, epoch)
    writer.add_scalar("Validation/mAP_75", map_75, epoch)

    print(f"\nValidation Results:")
    print(f"  mAP (0.50:0.95): {current_map:.4f}")
    print(f"  mAP@50: {map_50:.4f}")
    print(f"  mAP@75: {map_75:.4f}")

    map_per_class = results.get("map_per_class", None)
    if map_per_class is not None:
        print("  Per-class AP:")
        for class_idx, class_map in enumerate(map_per_class.tolist()):
            if class_map != class_map:
                class_str = "nan"
            else:
                class_str = f"{class_map:.4f}"
            print(f"    Class {class_idx}: {class_str}")

    # --- CHECKPOINTING ---
    checkpoint_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_map': best_map,
        'current_map': current_map,
        'args': vars(args),
    }

    # Save latest checkpoint
    latest_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    torch.save(checkpoint_dict, latest_path)
    print(f"Saved latest checkpoint to {latest_path}")

    # Save best checkpoint
    if current_map > best_map:
        best_map = current_map
        checkpoint_dict['best_map'] = best_map
        best_path = os.path.join(CHECKPOINT_DIR, 'best_checkpoint.pth')
        torch.save(checkpoint_dict, best_path)
        print(f"*** New best mAP: {best_map:.4f}. Saved to {best_path} ***")

    # Save periodic checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint_dict, periodic_path)
        print(f"Saved periodic checkpoint to {periodic_path}")

writer.close()
print("\nTraining complete!")
print(f"Best mAP: {best_map:.4f}")
