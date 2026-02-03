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

from data.dataset import BethesdaDataset
from data.transforms import get_train_transforms, get_valid_transforms
from models.sam3_DETR_v2 import Sam3ForClosedSetDetection, make_sam3_collate_fn, box_xyxy_clamp


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
    default=1e-5,
    help="Learning rate for SAM3 backbone"
)
parser.add_argument(
    "--lr_head",
    type=float,
    default=1e-4,
    help="Learning rate for classification head"
)
parser.add_argument(
    "--score_thresh",
    type=float,
    default=0.3,
    help="Score threshold for inference during validation"
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
# Dataset Wrapper for Sam3Processor
# ----------------------------
class BethesdaDatasetForSam3(BethesdaDataset):
    """
    Wrapper that returns PIL images instead of tensors.
    The Sam3Processor will handle the preprocessing.
    """
    def __init__(self, csv_file, root_dir):
        # Initialize without transforms - Sam3Processor handles preprocessing
        super().__init__(csv_file, root_dir, transforms=None)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        image_id = self.image_ids[idx]
        records = self.df[self.df['image_filename'] == image_id]

        image_path = os.path.join(self.root_dir, image_id)
        image = Image.open(image_path).convert('RGB')

        W, H = image.size  # PIL gives (W, H)

        boxes = []
        labels = []

        for _, row in records.iterrows():
            x_center, y_center = row['x'], row['y']
            width, height = row['width'], row['height']
            class_id = row['class']  # 0-indexed for this model (no +1, model handles no-object)

            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            # Clamp to image bounds
            x_min = max(0, min(x_min, W))
            y_min = max(0, min(y_min, H))
            x_max = max(0, min(x_max, W))
            y_max = max(0, min(y_max, H))

            if (x_max <= x_min) or (y_max <= y_min):
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


# ----------------------------
# Training Dataset with Augmentation
# ----------------------------
class BethesdaDatasetForSam3WithAug(BethesdaDataset):
    """
    Wrapper that applies augmentations and returns PIL images.
    Augmentations are applied via albumentations, then converted back.
    """
    def __init__(self, csv_file, root_dir, augment=True):
        super().__init__(csv_file, root_dir, transforms=None)
        self.augment = augment
        if augment:
            import albumentations as A
            self.aug_transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        image_id = self.image_ids[idx]
        records = self.df[self.df['image_filename'] == image_id]

        image_path = os.path.join(self.root_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        H, W, _ = image_np.shape

        boxes = []
        labels = []

        for _, row in records.iterrows():
            x_center, y_center = row['x'], row['y']
            width, height = row['width'], row['height']
            class_id = row['class']  # 0-indexed

            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            x_min = max(0, min(x_min, W))
            y_min = max(0, min(y_min, H))
            x_max = max(0, min(x_max, W))
            y_max = max(0, min(y_max, H))

            if (x_max <= x_min) or (y_max <= y_min):
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        # Apply augmentations
        if self.augment and len(boxes) > 0:
            transformed = self.aug_transforms(image=image_np, bboxes=boxes, labels=labels)
            image_np = transformed['image']
            boxes = list(transformed['bboxes'])
            labels = list(transformed['labels'])

        # Convert back to PIL for Sam3Processor
        image = Image.fromarray(image_np)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


# ----------------------------
# Initialize Datasets
# ----------------------------
print("Initializing Datasets for Sam3ForClosedSetDetection...")
train_ds = BethesdaDatasetForSam3WithAug(
    csv_file=CSV_PATH_TRAIN,
    root_dir=TRAIN_PATH,
    augment=True
)
val_ds = BethesdaDatasetForSam3(
    csv_file=CSV_PATH_VAL,
    root_dir=VAL_PATH
)

# ----------------------------
# Initialize Sam3Processor and DataLoaders
# ----------------------------
print("Loading Sam3Processor...")
processor = Sam3Processor.from_pretrained(SAM3_CHECKPOINT)
collate_fn = make_sam3_collate_fn(processor)

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
total_steps = num_epochs * len(train_loader)
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
print(f"\nStarting Training for {num_epochs} epochs...")
print(f"  Batch size: {BATCH_SIZE}")
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

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            outputs = model(pixel_values=pixel_values.to(device),
                            input_ids=input_ids.to(device),
                            attention_mask=None if attention_mask is None else attention_mask.to(device),
                            targets=[{k: v.to(device) for k, v in t.items()} for t in targets])
            losses = outputs["losses"]
            loss = losses["loss_total"]

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler
        scheduler.step()

        # Accumulate losses
        total_loss += loss.item()
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
    metric = MeanAveragePrecision(iou_type="bbox")

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
                preds_cpu.append({
                    'boxes': predictions[b]['boxes'].cpu(),
                    'scores': predictions[b]['scores'].cpu(),
                    'labels': predictions[b]['labels'].cpu()
                })

                # Convert normalized target boxes to absolute coordinates
                h, w = orig_sizes[b].tolist()
                tgt_boxes = targets[b]['boxes'].clone()
                tgt_boxes[:, [0, 2]] *= w
                tgt_boxes[:, [1, 3]] *= h

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
