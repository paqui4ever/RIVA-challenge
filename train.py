import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
import argparse

# Parser for choosing the model
parser = argparse.ArgumentParser(description="Train SAM3 Faster R-CNN")
parser.add_argument(
    "--model", 
    type=str, 
    choices=['sam3_rcnn', 'sam3_rcnn_v2', 'sam3_detr'],
    required=True,  # Make the flag mandatory
    help="The model architecture to use (sam3_rcnn, sam3_rcnn_v2, or sam3_detr)"
)
parser.add_argument(
    "--trainable_backbone",
    action="store_true",
    help="If set, unfreeze backbone for sam3_rcnn_v2 (fine-tuning)"
)
parser.add_argument(
    "--use_cosine_annealing",
    action="store_true",
    help="If set, enable CosineAnnealingLR scheduling"
)
parser.add_argument(
    "--use_reduce_on_plateau",
    action="store_true",
    help="If set, enable ReduceLROnPlateau scheduling with patience=5"
)
args = parser.parse_args()

# Checkpointing settings
CHECKPOINT_DIR = '/local_data/RIVA/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Tensorboard writer (logs saved in checkpoint directory)
writer = SummaryWriter(log_dir=os.path.join(CHECKPOINT_DIR, 'tensorboard'))
RESUME_CHECKPOINT = None  # Set to checkpoint path to resume training, e.g., './checkpoints/best_checkpoint.pth'

# Mixed precision settings
USE_AMP = True  # Set to False to disable mixed precision

# Paths
# CSV_PATH_TRAIN = 'RIVA/annotations/annotations/train.csv'
# CSV_PATH_VAL = 'RIVA/annotations/annotations/val.csv'
# TRAIN_PATH = 'RIVA/images/images/train'
# VAL_PATH = 'RIVA/images/images/val'
# TEST_PATH = 'RIVA/images/images/test'

CSV_PATH_TRAIN = '/local_data/RIVA/annotations/annotations/train.csv'
CSV_PATH_VAL = '/local_data/RIVA/annotations/annotations/val.csv'
TRAIN_PATH = '/local_data/RIVA/images/images/train'
VAL_PATH = '/local_data/RIVA/images/images/val'
TEST_PATH = '/local_data/RIVA/images/images/test'

# Imports from other libraries
try:
    from data.dataset import BethesdaDataset
    from models.sam3_rcnn import get_sam3_faster_rcnn
    from models.sam3_DETR import get_sam3_detr
    from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square
    from data.transforms import get_train_transforms_RCNN, get_valid_transforms
except ImportError as e:
    print(f"Import Error: {e}. Make sure 'models' and 'data' folders are in the path.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. DATASETS & DATALOADERS

print("Initializing Datasets with SAM3 transforms (1008x1008)...")
train_ds = BethesdaDataset(
    csv_file=CSV_PATH_TRAIN, 
    root_dir=TRAIN_PATH, 
    transforms=get_train_transforms_RCNN()
)
test_ds = BethesdaDataset(
    csv_file=CSV_PATH_VAL, 
    root_dir=VAL_PATH, 
    transforms=get_valid_transforms()
)

def collate_fn_sam(batch):
    return tuple(zip(*batch))

BATCH_SIZE = 4

train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn_sam
)
test_loader = DataLoader(
    test_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn_sam
)

# 4. MODEL INITIALIZATION

num_classes = 9 # 8 classes + background
if args.model == 'sam3_rcnn':
    print("Loading FasterRCNN with SAM3 backbone...")
    model = get_sam3_faster_rcnn(num_classes=num_classes)
elif args.model == 'sam3_rcnn_v2':
    print("Loading FasterRCNN with SAM3 Cut-B backbone (FPN multi-scale)...")
    model = build_sam3_fasterrcnn(
        model_name_or_path="facebook/sam3",
        num_classes_closed_set=num_classes - 1,  # v2 adds +1 internally for background
        trainable_backbone=args.trainable_backbone
    )
    print(f"  Backbone trainable: {args.trainable_backbone}")
elif args.model == 'sam3_detr':
    print("Loading DETR with SAM3 backbone...")
    model = get_sam3_detr(num_classes=num_classes)

model.to(device)

# 5. OPTIMIZER
# Filter parameters requiring gradients (SAM3 backbone is frozen by default)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-4, weight_decay=1e-5)

# 6. LEARNING RATE SCHEDULER
num_epochs = 200
total_steps = num_epochs * len(train_loader)
scheduler = None
scheduler_type = None
if args.use_cosine_annealing:
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scheduler_type = 'cosine'
elif args.use_reduce_on_plateau:
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=np.sqrt(0.1), verbose=True)
    scheduler_type = 'plateau'

# 7. MIXED PRECISION SCALER
scaler = GradScaler(enabled=USE_AMP)

# 8. CHECKPOINT RESUME
start_epoch = 0
global_step = 0
best_map = 0.0

if RESUME_CHECKPOINT and os.path.isfile(RESUME_CHECKPOINT):
    print(f"Loading checkpoint from {RESUME_CHECKPOINT}...")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint.get('global_step', 0)
    best_map = checkpoint.get('best_map', 0.0)
    print(f"Resumed from epoch {start_epoch}, global_step {global_step}, best_map {best_map:.4f}")

# 9. TRAINING LOOP

print("Starting Training...")

for epoch in range(start_epoch, num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # --- TRAINING ---
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for i, (images, targets) in enumerate(pbar):
        # Move to device and ensure float tensors
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Apply sam3_rcnn_v2-specific preprocessing (aspect-ratio preserving resize + pad)
        if args.model == 'sam3_rcnn_v2':
            target_size = model.backbone.target_size  # 1008 by default
            processed_images = []
            processed_targets = []
            for img, tgt in zip(images, targets):
                img, tgt, _ = sam3_resize_longest_side_and_pad_square(
                    img, tgt, target_size=target_size
                )
                processed_images.append(img)
                processed_targets.append(tgt)
            images = processed_images
            targets = processed_targets

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            if args.model == 'sam3_detr':
                # 1. Stack List[Tensor] -> Tensor (B, C, H, W)
                pixel_values = torch.stack(images) 
                
                # 2. Rename 'labels' to 'class_labels' for HF DETR
                formatted_targets = []
                for t in targets:
                    formatted_targets.append({
                        "class_labels": t["labels"], 
                        "boxes": t["boxes"]
                    })
                
                # 3. Forward pass with keyword args
                outputs = model(pixel_values=pixel_values, labels=formatted_targets)
                
                # 4. Extract losses (HF returns an output object, not a plain dict)
                losses = outputs.loss
                loss_dict = outputs.loss_dict
                
            else:
                # Forward pass returns dictionary of losses
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())  # Sum all losses

        # if args.model == 'sam3_rcnn':
        #     writer.add_scalar("Losses/total_train", losses, global_step)
        #     writer.add_scalar("Losses/train_rpn_box_reg", loss_dict["loss_rpn_box_reg"], global_step)
        #     writer.add_scalar("Losses/train_objectness", loss_dict["loss_objectness"], global_step)
        #     writer.add_scalar("Losses/train_box_reg", loss_dict["loss_box_reg"], global_step)
        #     writer.add_scalar("Losses/train_class", loss_dict["loss_classifier"], global_step)
        # else: 
        #     writer.add_scalar("Losses/total_train", losses, global_step)
        #     writer.add_scalar("Losses/train_bbox", loss_dict["loss_bbox"], global_step) # L1 loss for bounding boxes coordinates
        #     writer.add_scalar("Losses/train_giou", loss_dict["loss_giou"], global_step) # GIoU loss for bounding boxes
        #     writer.add_scalar("Losses/train_cardinality", loss_dict["loss_cardinality"], global_step) # cardinality loss for class prediction 
        #     writer.add_scalar("Losses/train_ce", loss_dict["loss_ce"], global_step) # negative log likelihood for class prediction 
        
        writer.add_scalar("Losses/total_train", losses, global_step)
        
        # Safe logging for keys that might not exist in both models
        for k, v in loss_dict.items():
            writer.add_scalar(f"Losses/train_{k}", v, global_step)

        # Mixed precision backward pass
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step the scheduler after each optimization step (only for cosine annealing)
        if scheduler is not None and scheduler_type == 'cosine':
            scheduler.step()

        total_loss += losses.item()
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})

        global_step += 1

    if scheduler is not None:
        lr_value = scheduler.get_last_lr()[0]
    else:
        lr_value = optimizer.param_groups[0]['lr']
    writer.add_scalar("LearningRate", lr_value, epoch)

    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    
    # --- VALIDATION (mAP) ---
    print("Validating...")
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Validation"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Apply sam3_rcnn_v2-specific preprocessing (aspect-ratio preserving resize + pad)
            if args.model == 'sam3_rcnn_v2':
                target_size = model.backbone.target_size  # 1008 by default
                processed_images = []
                processed_targets = []
                for img, tgt in zip(images, targets):
                    img, tgt, _ = sam3_resize_longest_side_and_pad_square(
                        img, tgt, target_size=target_size
                    )
                    processed_images.append(img)
                    processed_targets.append(tgt)
                images = processed_images
                targets = processed_targets

            # Mixed precision inference
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                if args.model == 'sam3_detr':
                    # Stack images for DETR
                    pixel_values = torch.stack(images)
                    detr_outputs = model(pixel_values=pixel_values)

                    # Convert DETR outputs to torchmetrics format
                    # DETR outputs: logits (B, num_queries, num_classes+1), pred_boxes (B, num_queries, 4) in cxcywh normalized
                    outputs = []
                    for b in range(pixel_values.shape[0]):
                        # Get predictions for this image
                        logits = detr_outputs.logits[b]  # (num_queries, num_classes+1)
                        pred_boxes = detr_outputs.pred_boxes[b]  # (num_queries, 4) in cxcywh normalized

                        # Get class predictions (exclude no-object class which is last)
                        probs = logits.softmax(-1)
                        scores, labels = probs[:, :-1].max(-1)  # Exclude last class (no-object)

                        # Filter out low confidence predictions (threshold can be adjusted)
                        keep = scores > 0.5
                        scores = scores[keep]
                        labels = labels[keep]
                        boxes_cxcywh = pred_boxes[keep]

                        # Convert from cxcywh normalized to xyxy absolute
                        img_h, img_w = pixel_values.shape[-2:]
                        cx, cy, w, h = boxes_cxcywh.unbind(-1)
                        boxes_xyxy = torch.stack([
                            (cx - 0.5 * w) * img_w,
                            (cy - 0.5 * h) * img_h,
                            (cx + 0.5 * w) * img_w,
                            (cy + 0.5 * h) * img_h
                        ], dim=-1)

                        outputs.append({
                            'boxes': boxes_xyxy,
                            'scores': scores,
                            'labels': labels
                        })
                else:
                    # Forward pass in eval mode returns detections (list of dicts)
                    outputs = model(images)

            # Send to metric. Both outputs and targets are lists of dicts.
            # Outputs on device, targets on device - metric handles it.
            # Some metrics need CPU conversion, but torchmetrics handles recent versions.
            # Ensuring CPU just in case for complex metrics if it fails.
            outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(outputs_cpu, targets_cpu)
            
    results = metric.compute()
    current_map = results['map'].item()
    writer.add_scalar("Validation/mAP_50_95", results['map'], epoch)
    writer.add_scalar("Validation/mAP_50", results['map_50'], epoch)
    
    print(f"Validation Results - mAP (0.50:0.95): {current_map:.4f}")

    # Step ReduceLROnPlateau scheduler based on validation mAP
    if scheduler is not None and scheduler_type == 'plateau':
        scheduler.step(current_map)

    # --- CHECKPOINTING ---
    checkpoint_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict(),
        'best_map': best_map,
        'current_map': current_map,
    }

    # Save latest checkpoint
    latest_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    torch.save(checkpoint_dict, latest_path)
    print(f"Saved latest checkpoint to {latest_path}")

    # Save best checkpoint if current mAP is better
    if current_map > best_map:
        best_map = current_map
        checkpoint_dict['best_map'] = best_map
        best_path = os.path.join(CHECKPOINT_DIR, 'best_checkpoint.pth')
        torch.save(checkpoint_dict, best_path)
        print(f"New best mAP: {best_map:.4f}. Saved best checkpoint to {best_path}")

writer.close()
