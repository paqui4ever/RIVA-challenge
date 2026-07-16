import argparse
import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms.functional as TVF

# Import custom SAM3 Faster RCNN components
from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square

from overcomplete import TopKSAE, MPSAE
from overcomplete.sae import RATopKSAE, OMPSAE
# Rename it so it doesn't clash with the function name on this script
from overcomplete.sae.train import train_sae as lib_train_sae

from overcomplete.metrics import (
    r2_score,
    avg_l2_loss,
    l1,
    sparsity_eps,
    dead_codes,
    relative_avg_l2_loss,
    hoyer,
    dictionary_collinearity,
    hungarian_loss,
    cosine_hungarian_loss,
)

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, target_size=1008):
        self.image_paths = image_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = TVF.to_tensor(img)
        img_padded, _, _ = sam3_resize_longest_side_and_pad_square(img_t, target_size=self.target_size)
        return img_padded

def get_data_splits(base_dir, seed=42):
    base_path = Path(base_dir)
    
    def get_images(folder):
        return [str(p) for ext in ('*.jpg', '*.png') for p in (base_path / folder).glob(ext)]
        
    train_files = get_images("train")
    test_files = get_images("test")
    val_files_all = get_images("val")
    
    random.seed(seed)
    random.shuffle(val_files_all)
    
    mid_point = len(val_files_all) // 2
    
    # Train: All train + 100% test
    # Val: 50% random val
    # Test: Remaining 50% val
    return (
        train_files + test_files,
        val_files_all[:mid_point],
        val_files_all[mid_point:]
    )

def extract_and_cache_features(model, image_paths, cache_path, device, batch_size=8):
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path)

    print(f"Extracting features to {cache_path}...")
    dataset = ImagePathDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    activations = []
    
    def hook_fn(module, input, output):
        # Flatten spatial dims: [B, C, H, W] -> [B, C, H*W] -> [B*H*W, C]
        feat = output["1"].detach().cpu()
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
        activations.append(feat)

    # Register hook on the FPN backbone
    handle = model.backbone.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            batch = batch.to(device)
            _ = model.backbone(batch) # Forward pass only backbone
            
    handle.remove()
    
    all_features = torch.cat(activations, dim=0)
    print(f"Extracted features shape: {all_features.shape}")
    
    # Save to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(all_features, cache_path)
    return all_features

def get_lr_scheduler(optimizer, warmup_steps, total_steps, initial_lr, base_lr):
    # Warmup for 5% of total steps starting from base_lr / 10
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=initial_lr / base_lr, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps
    )
    return SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )

def custom_criterion(x, x_hat, z_pre, z, dictionary):
    return torch.nn.functional.mse_loss(x, x_hat)

def train_sae(model, name, train_tensor, val_tensor, args, device, logger):
    print(f"--- Training {name} ---")
    model.to(device)
    
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(0.05 * total_steps))
    
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, 3e-5, args.lr)

    for epoch in range(args.epochs):
        # Use overcomplete's implemented method for one epoch at a time to maintain our custom logging/validation
        logs = lib_train_sae(
            model=model,
            dataloader=train_loader,
            criterion=custom_criterion,
            optimizer=optimizer,
            scheduler=scheduler, # Step scheduler per batch automatically inside train_sae
            nb_epochs=1,
            clip_grad=1.0,
            monitoring=1,
            device=device
        )
        
        train_loss = logs['avg_loss'][0] if 'avg_loss' in logs and len(logs['avg_loss']) > 0 else 0.0
        
        # Evaluation every epoch
        model.eval()
        with torch.no_grad():
            # Get full val batch for metric calculation if it fits in memory, else batched
            # For simplicity, calculate on a large subset
            val_batch = val_tensor[:10000].to(device)
            train_batch_subset = train_tensor[:10000].to(device)
            
            # forward returns (z_pre, z, x_hat)
            _, val_z, val_recon = model(val_batch)
            _, train_z, train_recon = model(train_batch_subset)

            # Compute metrics
            metrics = {
                f"{name}/train_loss": train_loss,
                f"{name}/lr": scheduler.get_last_lr()[0],
                f"{name}/r2_score_val": r2_score(val_batch, val_recon).item(),
                f"{name}/avg_l2_loss_train": avg_l2_loss(train_batch_subset, train_recon).item(),
                f"{name}/avg_l2_loss_val": avg_l2_loss(val_batch, val_recon).item(),
                f"{name}/l1_train": l1(train_z).item(),
                f"{name}/sparsity_eps_val": sparsity_eps(val_z, eps=1e-5).item(),
                f"{name}/dead_codes_val": dead_codes(val_z).item(),
                f"{name}/relative_avg_l2_loss_val": relative_avg_l2_loss(val_batch, val_recon).item(),
                f"{name}/hoyer_val": hoyer(val_z).item(),
            }
            
            if hasattr(model, 'get_dictionary') and callable(model.get_dictionary):
                metrics[f"{name}/dictionary_collinearity_val"] = dictionary_collinearity(model.get_dictionary())[0]
            elif hasattr(model, 'dictionary'):
                metrics[f"{name}/dictionary_collinearity_val"] = dictionary_collinearity(model.dictionary)[0]

            if logger:
                if args.logger == "wandb":
                    logger.log(metrics, step=epoch)
                elif args.logger == "tensorboard":
                    for k, v in metrics.items():
                        logger.add_scalar(k, v, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Validation R2: {metrics[f'{name}/r2_score_val']:.4f}")

    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.logger == "wandb":
        import wandb
        logger = wandb.init(project="sam3-sae-interpretability", config=vars(args))
    elif args.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(log_dir="./runs/sae_interpretability")
    else:
        logger = None

    # Load Model
    print("Loading SAM3 Faster R-CNN...")
    model = build_sam3_fasterrcnn(trainable_backbone=False)
    
    # Load weights
    weight_dir = Path("./weights")
    weight_files = list(weight_dir.glob("*.pt")) + list(weight_dir.glob("*.pth"))
    if weight_files:
        sam3_weights = [w for w in weight_files if "sam3" in w.name.lower()]
        best_weight = sam3_weights[0] if sam3_weights else weight_files[0]
        print(f"Found weights: {best_weight}. Loading...")
        state_dict = torch.load(best_weight, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        # Allow missing keys in case it's a partial weight
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No weights found in ./weights/. Using initialized weights.")

    model.to(device)
    model.eval()

    # Data pipeline
    base_image_dir = "./RIVA/images/images"
    train_paths, val_paths, test_paths = get_data_splits(base_image_dir)
    print(f"Data Splits -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    cache_dir = "./interpretability_analysis/features"
    train_features = extract_and_cache_features(model, train_paths, os.path.join(cache_dir, "train_features.pt"), device)
    val_features = extract_and_cache_features(model, val_paths, os.path.join(cache_dir, "val_features.pt"), device)
    # Test set extraction can be done later if needed for final eval
    # test_features = extract_and_cache_features(model, test_paths, os.path.join(cache_dir, "test_features.pt"), device)

    input_dim = train_features.shape[1]
    hidden_dim = int(input_dim * args.expansion_factor)
    print(f"Input Dim: {input_dim}, Hidden Dim: {hidden_dim}")

    # Initialize SAEs
    sae_models = {
        "TopKSAE": TopKSAE(input_dim, hidden_dim, k=args.k),
        "RATopKSAE": RATopKSAE(input_dim, hidden_dim, k=args.k),
        "MPSAE": MPSAE(input_dim, hidden_dim, k=args.k),
        "OMPSAE": OMPSAE(input_dim, hidden_dim, k=args.k),
    }

    checkpoint_dir = "./interpretability_analysis/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    trained_models = {}
    for name, sae in sae_models.items():
        trained = train_sae(sae, name, train_features, val_features, args, device, logger)
        trained_models[name] = trained
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}_sae.pt")
        torch.save(trained.state_dict(), checkpoint_path)
        print(f"Saved {name} to {checkpoint_path}")
        
        if args.logger == "wandb":
            artifact = wandb.Artifact(name=f"{name}_sae", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    print("\n--- Final Dictionary Comparison ---")
    names = list(trained_models.keys())
    
    def _get_dict(m):
        if hasattr(m, 'get_dictionary') and callable(m.get_dictionary):
            return m.get_dictionary()
        elif hasattr(m, 'dictionary'):
            return m.dictionary
        return None
        
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m1, m2 = trained_models[names[i]], trained_models[names[j]]
            d1 = _get_dict(m1)
            d2 = _get_dict(m2)
            
            if d1 is not None and d2 is not None:
                h_loss = hungarian_loss(d1, d2)
                ch_loss = cosine_hungarian_loss(d1, d2)
                print(f"Comparison {names[i]} vs {names[j]}:")
                print(f"  Hungarian Loss: {h_loss:.4f}")
                print(f"  Cosine Hungarian Loss: {ch_loss:.4f}")
                
                if logger:
                    comp_name = f"{names[i]}_vs_{names[j]}"
                    metrics = {
                        f"Comparison/{comp_name}_hungarian_loss": h_loss,
                        f"Comparison/{comp_name}_cosine_hungarian_loss": ch_loss
                    }
                    if args.logger == "wandb":
                        logger.log(metrics)
                    elif args.logger == "tensorboard":
                        logger.add_scalar(f"Comparison/{comp_name}_hungarian_loss", h_loss, 0)
                        logger.add_scalar(f"Comparison/{comp_name}_cosine_hungarian_loss", ch_loss, 0)

    if args.logger == "wandb":
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAEs on SAM3 Faster R-CNN features")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--k", type=int, default=16, help="Target sparsity (L0 / k)")
    parser.add_argument("--expansion_factor", type=float, default=4.0, help="Expansion factor for dictionary size")
    parser.add_argument("--logger", type=str, choices=["wandb", "tensorboard", "none"], default="wandb", help="Logger to use")
    
    args = parser.parse_args()
    main(args)
