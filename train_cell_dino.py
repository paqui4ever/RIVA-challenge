import argparse
import os

import math
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# Imports
try:
    from data.dataset import BethesdaDataset
    from data.transforms import get_train_transforms_v2, get_valid_transforms
    from models.cell_DINO_rcnn_v2 import (
        build_cell_dino_fasterrcnn,
        cell_dino_resize_longest_side_and_pad_square
    )
except ImportError as e:
    raise ImportError(f"Import Error: {e}. Make sure 'models' and 'data' folders are in the path.")

# Parser
parser = argparse.ArgumentParser(description="Train Cell-DINO Faster R-CNN")
parser.add_argument(
    "--pretrained_checkpoint_path", 
    type=str, 
    default=None,
    help="Path to Cell-DINO high-res weights (.pth file). Optional if loading from URL."
)
parser.add_argument(
    "--trainable_backbone",
    action="store_true",
    help="If set, unfreeze backbone (fine-tuning). Default is Frozen."
)
parser.add_argument(
    "--use_cosine_annealing",
    action="store_true",
    help="If set, enable CosineAnnealingLR scheduling."
)
parser.add_argument(
    "--use_reduce_on_plateau",
    action="store_true",
    help="If set, enable ReduceLROnPlateau scheduling with patience=5."
)
parser.add_argument(
    "--use_weighted_sampler",
    action="store_true",
    help="If set, enable WeightedRandomSampler for class imbalance."
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=32,
    help="Number of steps to accumulate gradients before updating optimizer. Default 32 (effective batch 128 with batch size 4)."
)
args = parser.parse_args()

if args.use_cosine_annealing and args.use_reduce_on_plateau:
    raise ValueError("Choose only one LR scheduler: --use_cosine_annealing or --use_reduce_on_plateau")

# Checkpointing settings
CHECKPOINT_DIR = '/local_data/RIVA/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Tensorboard
writer = SummaryWriter(log_dir=os.path.join(CHECKPOINT_DIR, 'tensorboard_cell_dino'))
RESUME_CHECKPOINT = None 

# Mixed precision
USE_AMP = True

# Paths
CSV_PATH_TRAIN = '/local_data/RIVA/annotations/annotations/train.csv'
CSV_PATH_VAL = '/local_data/RIVA/annotations/annotations/val.csv'
TRAIN_PATH = '/local_data/RIVA/images/images/train'
VAL_PATH = '/local_data/RIVA/images/images/val'
TEST_PATH = '/local_data/RIVA/images/images/test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ADDRESSING CLASS IMBALANCE
sampler = None
train_shuffle = True
sample_weights = None

if args.use_weighted_sampler:
    print("Using WeightedRandomSampler for class imbalance handling...")
    df = pd.read_csv(CSV_PATH_TRAIN)

    class_weights = {
        'INFL': 1.00,
        'NILM': 1.14,
        'LSIL': 2.76,
        'HSIL': 3.54,
        'SCC': 4.03,
        'ENDO': 6.53,
        'ASCH': 13.80,
        'ASCUS': 21.06,
    }

    image_groups = df.groupby('image_filename')
    unique_images = df['image_filename'].unique()
    sample_weights = []

    for img_name in unique_images:
        classes_in_img = image_groups.get_group(img_name)['class_name'].values
        max_weight = max(class_weights[c] for c in classes_in_img)
        sample_weights.append(max_weight)

    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_shuffle = False
else:
    print("Using default shuffled sampling (no weighted sampler).")

# 3. DATASETS & DATALOADERS
print("Initializing Datasets with transforms...")
train_ds = BethesdaDataset(
    csv_file=CSV_PATH_TRAIN, 
    root_dir=TRAIN_PATH, 
    transforms=get_train_transforms_v2() # Using data augmentation mentioned in Cell-DINO paper
)
test_ds = BethesdaDataset(
    csv_file=CSV_PATH_VAL, 
    root_dir=VAL_PATH, 
    transforms=get_valid_transforms()
)

def collate_fn_basic(batch):
    return tuple(zip(*batch))

BATCH_SIZE = 4 # Cell-DINO ViT-L might take memory
train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    shuffle=train_shuffle,
    collate_fn=collate_fn_basic  # We do preprocessing in loop
)

if sampler is not None and sample_weights is not None and len(sample_weights) != len(train_ds):
    raise ValueError(
        f"Weighted sampler length ({len(sample_weights)}) does not match dataset length ({len(train_ds)})."
    )
test_loader = DataLoader(
    test_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn_basic
)

# 4. MODEL INITIALIZATION
num_classes = 9 # 8 classes + background
print("Loading FasterRCNN with Cell-DINO backbone...")
model = build_cell_dino_fasterrcnn(
    model_name="cell_dino_hpa_vitl14", # As confirmed by user
    pretrained_checkpoint_path=args.pretrained_checkpoint_path,
    num_classes_closed_set=num_classes - 1,
    trainable_backbone=args.trainable_backbone
)
model.to(device)

print(f"Backbone Trainable: {args.trainable_backbone}")

# 5. OPTIMIZER
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-4, weight_decay=1e-5)

# 6. SCHEDULER
num_epochs = 1000
num_epochs = 1000
steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
total_steps = num_epochs * steps_per_epoch
scheduler = None
scheduler_type = None

if args.use_reduce_on_plateau:
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=np.sqrt(0.1),
        verbose=True,
        min_lr=1e-6
    )
    scheduler_type = 'plateau'
else:
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scheduler_type = 'cosine'

# 7. SCALER
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
    print(f"Resumed from epoch {start_epoch}, best_map {best_map:.4f}")

# 9. TRAINING LOOP
print("Starting Training...")

for epoch in range(start_epoch, num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # --- TRAINING ---
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    for i, (images, targets) in enumerate(pbar):
        # We need to apply Cell-DINO Resize+Pad to 1008x1008
        target_size = model.backbone.target_size
        
        processed_images = []
        processed_targets = []
        
        for img, tgt in zip(images, targets):
            p_img, p_tgt, _ = cell_dino_resize_longest_side_and_pad_square(
                img, tgt, target_size=target_size
            )
            if p_tgt is None:
                raise ValueError("Cell-DINO preprocessing returned None target during training.")
            processed_images.append(p_img.to(device))
            processed_targets.append({k: v.to(device) for k, v in p_tgt.items()})
            
        with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            loss_dict = model(processed_images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            # Normalize loss for gradient accumulation
            losses = losses / args.gradient_accumulation_steps

        writer.add_scalar("Losses/total_train", losses * args.gradient_accumulation_steps, global_step)
        for k, v in loss_dict.items():
            writer.add_scalar(f"Losses/train_{k}", v, global_step)

        scaler.scale(losses).backward()

        if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and scheduler_type == 'cosine':
                scheduler.step()

        loss_value = float(losses * args.gradient_accumulation_steps)
        total_loss += loss_value
        pbar.set_postfix({'loss': f"{loss_value:.4f}"})
        global_step += 1

    if scheduler is not None and scheduler_type == 'cosine':
        lr_value = scheduler.get_last_lr()[0]
    else:
        lr_value = optimizer.param_groups[0]['lr']
    writer.add_scalar("LearningRate", lr_value, epoch)
    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    writer.add_scalar("Losses/avg_epoch_loss", avg_loss, epoch)
    
    # --- VALIDATION ---
    print("Validating...")
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Validation"):
            target_size = model.backbone.target_size
            processed_images = []
            processed_targets = [] # We need targets for metric
            
            for img, tgt in zip(images, targets):
                p_img, p_tgt, _ = cell_dino_resize_longest_side_and_pad_square(
                    img, tgt, target_size=target_size
                )
                if p_tgt is None:
                    raise ValueError("Cell-DINO preprocessing returned None target during validation.")
                processed_images.append(p_img.to(device))
                processed_targets.append(p_tgt) # Keep cpu or device? Metric handles both usually.

            # Inference
            with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                outputs = model(processed_images)
            
            # Move to CPU for metric
            outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in processed_targets]
            
            metric.update(outputs_cpu, targets_cpu)
            
    results = metric.compute()

    map_value = results['map']
    map_50_value = results['map_50']
    current_map = float(map_value)
    current_map_50 = float(map_50_value)

    writer.add_scalar("Validation/mAP_50_95", current_map, epoch)
    writer.add_scalar("Validation/mAP_50", current_map_50, epoch)
    print(f"Validation Results - mAP (0.50:0.95): {current_map:.4f}")

    if scheduler is not None and scheduler_type == 'plateau':
        scheduler.step(avg_loss)

    class_names = [
        "Background", "NILM", "ENDO", "INFL",
        "ASCUS", "LSIL", "HSIL", "ASCH", "SCC"
    ]

    map_per_class = results.get("map_per_class", None)
    classes = results.get("classes", None)

    if map_per_class is not None:
        map_values = None
        class_ids = None

        if isinstance(map_per_class, torch.Tensor):
            if map_per_class.ndim == 1:
                map_values = map_per_class.detach().cpu().tolist()
        elif isinstance(map_per_class, np.ndarray):
            if map_per_class.ndim == 1:
                map_values = map_per_class.tolist()
        elif isinstance(map_per_class, (list, tuple)):
            map_values = list(map_per_class)

        if isinstance(classes, torch.Tensor):
            class_ids = classes.detach().cpu().tolist()
        elif isinstance(classes, np.ndarray):
            class_ids = classes.tolist()
        elif isinstance(classes, (list, tuple)):
            class_ids = list(classes)

        if map_values is None:
            print("  Per-class AP unavailable (metric returned scalar map_per_class).")
        else:
            ap_dict = {}
            print("  Per-class AP:")
            for class_idx, class_map in enumerate(map_values):
                class_id = int(class_ids[class_idx]) if class_ids is not None and class_idx < len(class_ids) else class_idx

                if 0 <= class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = f"Class_{class_id}"

                class_value = float(class_map)
                if class_value != class_value:
                    class_str = "nan"
                else:
                    class_str = f"{class_value:.4f}"
                    ap_dict[label] = class_value

                print(f"    Class {label}: {class_str}")

            if ap_dict:
                writer.add_scalars("Validation/AP_per_class", ap_dict, epoch)
    
    # SAVE CHECKPOINT
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
    
    latest_path = os.path.join(CHECKPOINT_DIR, 'latest_cell_dino.pth')
    torch.save(checkpoint_dict, latest_path)
    print(f"Saved latest to {latest_path}")
    
    if current_map > best_map:
        best_map = current_map
        checkpoint_dict['best_map'] = best_map
        best_path = os.path.join(CHECKPOINT_DIR, 'best_cell_dino.pth')
        torch.save(checkpoint_dict, best_path)
        print(f"New best mAP: {best_map:.4f} -> Saved to {best_path}")

writer.close()
