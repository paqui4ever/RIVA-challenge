import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

# Tensorboard writer
writer = SummaryWriter()

# Paths 
CSV_PATH_TRAIN = './RIVA/annotations/annotations/train.csv'
CSV_PATH_VAL = './RIVA/annotations/annotations/val.csv'
TRAIN_PATH = './RIVA/images/images/train'
VAL_PATH = './RIVA/images/images/val'
TEST_PATH = './RIVA/images/images/test'

# Imports from other libraries
try:
    from data.dataset import BethesdaDataset
    from models.sam3_rcnn import get_sam3_faster_rcnn
    from data.transforms import get_train_transforms, get_valid_transforms
except ImportError as e:
    print(f"Import Error: {e}. Make sure 'models' and 'data' folders are in the path.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. DATASETS & DATALOADERS

print("Initializing Datasets with SAM3 transforms (1008x1008)...")
train_ds = BethesdaDataset(
    csv_file=CSV_PATH_TRAIN, 
    root_dir=TRAIN_PATH, 
    transforms=get_train_transforms()
)
test_ds = BethesdaDataset(
    csv_file=CSV_PATH_VAL, 
    root_dir=VAL_PATH, 
    transforms=get_valid_transforms()
)

def collate_fn_sam(batch):
    return tuple(zip(*batch))

BATCH_SIZE = 32

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
print("Loading FasterRCNN with SAM3 backbone...")
num_classes = 9 # 8 classes + background
# Using Hugging Face 'facebook/sam3' by default
model = get_sam3_faster_rcnn(num_classes=num_classes)
model.to(device)

# 5. OPTIMIZER
# Filter parameters requiring gradients (SAM3 backbone is frozen by default)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=1e-3, weight_decay=1e-4)

# 6. TRAINING LOOP
num_epochs = 100
global_step = 0 

print("Starting Training...")

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # --- TRAINING ---
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for i, (images, targets) in enumerate(pbar):
        # Move to device and ensure float tensors
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass returns dictionary of losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) # Sum all losses

        writer.add_scalar("Losses/total_train", losses, global_step)
        writer.add_scalar("Losses/train_rpn_box_reg", loss_dict["loss_rpn_box_reg"], global_step)
        writer.add_scalar("Losses/train_objectness", loss_dict["loss_objectness"], global_step)
        writer.add_scalar("Losses/train_box_reg", loss_dict["loss_box_reg"], global_step)
        writer.add_scalar("Losses/train_class", loss_dict["loss_classifier"], global_step)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})

        global_step += 1
        
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
    writer.add_scalar("Validation/mAP_50_95", results['map'], epoch)
    writer.add_scalar("Validation/mAP_50", results['map_50'], epoch)
    
    print(f"Validation Results - mAP (0.50:0.95): {results['map']:.4f}")
    
    # Save Model
    # torch.save(model.state_dict(), f"checkpoints/riva_sam3_epoch_{epoch+1}.pth")

writer.close()
