import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# Imports
try:
    from data.dataset import BethesdaDataset
    from data.transforms import get_train_transforms_v2, get_valid_transforms
    from models.cell_DINO_rcnn_v2 import (
        build_cell_dino_fasterrcnn,
        cell_dino_resize_longest_side_and_pad_square
    )
except ImportError as e:
    print(f"Import Error: {e}. Make sure 'models' and 'data' folders are in the path.")

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
args = parser.parse_args()

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
    shuffle=True, 
    collate_fn=collate_fn_basic  # We do preprocessing in loop
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
total_steps = num_epochs * len(train_loader)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# 7. SCALER
scaler = GradScaler(enabled=USE_AMP)

# 8. CHECKPOINT RESUME
start_epoch = 0
global_step = 0
best_map = 0.0

if RESUME_CHECKPOINT and os.path.isfile(RESUME_CHECKPOINT):
    print(f"Loading checkpoint from {RESUME_CHECKPOINT}...")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        # Move raw images to device first or keep on cpu? 
        # Preprocessing needs them as tensors. BethesdaDataset returns tensors.
        
        # We need to apply Cell-DINO Resize+Pad to 1008x1008
        target_size = model.backbone.target_size
        
        processed_images = []
        processed_targets = []
        
        # Preprocess on CPU or Device? Usually safe on CPU or Device.
        # Images are likely on CPU from dataloader collate.
        for img, tgt in zip(images, targets):
            # Ensure tensor [C, H, W]
            if not isinstance(img, torch.Tensor):
                img = TVF.to_tensor(img)
            
            # Preprocess
            p_img, p_tgt, _ = cell_dino_resize_longest_side_and_pad_square(
                img, tgt, target_size=target_size
            )
            processed_images.append(p_img.to(device))
            processed_targets.append({k: v.to(device) for k, v in p_tgt.items()})
            
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            loss_dict = model(processed_images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

        # Logging
        # writer.add_scalar("Losses/total_train", losses, global_step)
        # for k, v in loss_dict.items():
        #     writer.add_scalar(f"Losses/train_{k}", v, global_step)

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += losses.item()
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
        global_step += 1

    writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)
    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    writer.add_scalar("Losses/avg_epoch_loss", avg_loss, epoch)
    
    # --- VALIDATION ---
    print("Validating...")
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Validation"):
            target_size = model.backbone.target_size
            processed_images = []
            processed_targets = [] # We need targets for metric
            
            for img, tgt in zip(images, targets):
                p_img, p_tgt, _ = cell_dino_resize_longest_side_and_pad_square(
                    img, tgt, target_size=target_size
                )
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
    current_map = results['map'].item()
    writer.add_scalar("Validation/mAP_50_95", current_map, epoch)
    writer.add_scalar("Validation/mAP_50", results['map_50'], epoch)
    print(f"Validation Results - mAP (0.50:0.95): {current_map:.4f}")
    
    # SAVE CHECKPOINT
    checkpoint_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
