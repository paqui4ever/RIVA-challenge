import argparse
import csv
from datetime import datetime
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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom SAM3 Faster RCNN components
from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square

from overcomplete import TopKSAE
from custom_saes import (
    RATopKSAE, 
    OMPSAE, 
    MpSAE as MPSAE,
    # Block-Sparse Featurizers (Goodfire AI: https://github.com/goodfire-ai/block-sparse-featurizer)
    VanillaBSF,
    GrassmannianBSF
)
# Rename it so it doesn't clash with the function name on this script
from overcomplete.sae.train import train_sae as lib_train_sae


from overcomplete.metrics import (
    r2_score,
    avg_l2_loss,
    l1,
    dead_codes,
    relative_avg_l2_loss,
    hoyer,
    dictionary_collinearity,
    hungarian_loss,
    cosine_hungarian_loss,
)

# overcomplete 0.3.0 renamed or removed sparsity_eps in favor of l0_eps
#try:
#    from overcomplete.metrics import sparsity_eps
#except ImportError:
#    try:
from overcomplete.metrics import l0_eps as sparsity_eps
#    except ImportError:
#        def sparsity_eps(x, dims=None, threshold=1e-6):
#            if dims is None:
#                return torch.mean((torch.abs(x) <= threshold).float())
#            return torch.mean((torch.abs(x) <= threshold).float(), dims)


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

def extract_and_cache_features(model, image_paths, cache_path, device, batch_size=1):
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    if not image_paths or len(image_paths) == 0:
        raise ValueError(
            f"Cannot extract features to '{cache_path}': no images found in the dataset folder. "
            "Please specify the dataset path using --base_image_dir or --dataset_path."
        )

    print(f"Extracting features to {cache_path} from {len(image_paths)} images...")
    dataset = ImagePathDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

def save_records_to_csv(filepath, records):
    """Saves or appends a list of dict records to a CSV file."""
    if not records:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    keys = list(records[0].keys())
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)
    print(f"Logged {len(records)} record(s) to CSV: {filepath}")

def train_sae(model, name, train_tensor, val_tensor, args, device, logger, results_dir="./interpretability_analysis/results"):
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

    history_records = []

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
        
        # Normalize decoder atoms if model supports it (e.g. VanillaBSF)
        if hasattr(model, 'normalize_decoder') and callable(model.normalize_decoder):
            model.normalize_decoder()
        
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

            def _get_val(m):
                if hasattr(m, 'numel') and m.numel() > 1:
                    return float(m.sum().item())
                return m.item() if hasattr(m, 'item') else float(m)

            collin_val = 0.0
            if hasattr(model, 'get_dictionary') and callable(model.get_dictionary):
                collin_val = float(dictionary_collinearity(model.get_dictionary())[0].item())
            elif hasattr(model, 'dictionary'):
                collin_val = float(dictionary_collinearity(model.dictionary)[0].item())

            # Compute metrics
            r2_val = _get_val(r2_score(val_batch, val_recon))
            l2_train = _get_val(avg_l2_loss(train_batch_subset, train_recon))
            l2_val = _get_val(avg_l2_loss(val_batch, val_recon))
            rel_l2_val = _get_val(relative_avg_l2_loss(val_batch, val_recon))
            l1_tr = _get_val(l1(train_z))
            sparsity_v = _get_val(sparsity_eps(val_z))
            dead_c_v = _get_val(dead_codes(val_z))
            hoyer_v = _get_val(hoyer(val_z))

            metrics = {
                f"{name}/train_loss": train_loss,
                f"{name}/lr": scheduler.get_last_lr()[0],
                f"{name}/r2_score_val": r2_val,
                f"{name}/avg_l2_loss_train": l2_train,
                f"{name}/avg_l2_loss_val": l2_val,
                f"{name}/l1_train": l1_tr,
                f"{name}/sparsity_eps_val": sparsity_v,
                f"{name}/dead_codes_val": dead_c_v,
                f"{name}/relative_avg_l2_loss_val": rel_l2_val,
                f"{name}/hoyer_val": hoyer_v,
                f"{name}/dictionary_collinearity_val": collin_val,
            }

            # Epoch record for CSV history
            epoch_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": name,
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "lr": scheduler.get_last_lr()[0],
                "r2_score_val": round(r2_val, 6),
                "avg_l2_loss_train": round(l2_train, 6),
                "avg_l2_loss_val": round(l2_val, 6),
                "relative_avg_l2_loss_val": round(rel_l2_val, 6),
                "l1_train": round(l1_tr, 6),
                "sparsity_eps_val": round(sparsity_v, 4),
                "dead_codes_val": round(dead_c_v, 4),
                "hoyer_val": round(hoyer_v, 6),
                "dictionary_collinearity_val": round(collin_val, 6),
            }
            history_records.append(epoch_record)

            if logger:
                if args.logger == "wandb":
                    logger.log(metrics, step=epoch)
                elif args.logger == "tensorboard":
                    for k, v in metrics.items():
                        logger.add_scalar(k, v, epoch)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs} - Validation R2: {metrics[f'{name}/r2_score_val']:.4f}")

    # Write training history to CSV
    if results_dir:
        history_csv = os.path.join(results_dir, "sae_training_history.csv")
        save_records_to_csv(history_csv, history_records)

        # Write summary row for this model to CSV
        if history_records:
            last_rec = history_records[-1]
            best_r2 = max(r["r2_score_val"] for r in history_records)
            best_l2 = min(r["avg_l2_loss_val"] for r in history_records)
            summary_record = [{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": name,
                "total_epochs": args.epochs,
                "k": args.k,
                "expansion_factor": args.expansion_factor,
                "block_size": getattr(args, "block_size", 4) if "BSF" in name else 1,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "final_train_loss": last_rec["train_loss"],
                "final_r2_val": last_rec["r2_score_val"],
                "best_r2_val": best_r2,
                "final_avg_l2_val": last_rec["avg_l2_loss_val"],
                "best_avg_l2_val": best_l2,
                "final_relative_l2_val": last_rec["relative_avg_l2_loss_val"],
                "final_sparsity_val": last_rec["sparsity_eps_val"],
                "final_dead_codes_val": last_rec["dead_codes_val"],
                "final_hoyer_val": last_rec["hoyer_val"],
                "final_collinearity_val": last_rec["dictionary_collinearity_val"],
            }]
            summary_csv = os.path.join(results_dir, "sae_summary.csv")
            save_records_to_csv(summary_csv, summary_record)

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

    cache_dir = "./interpretability_analysis/features"
    train_cache = os.path.join(cache_dir, "train_features.pt")
    val_cache = os.path.join(cache_dir, "val_features.pt")

    # If features are already cached, load them directly without loading SAM3
    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Found cached features at {cache_dir}. Loading directly...")
        train_features = torch.load(train_cache, weights_only=False)
        val_features = torch.load(val_cache, weights_only=False)
    else:
        # Load Model only if feature extraction is required
        print("Cached features not found. Initializing SAM3 Faster R-CNN for feature extraction...")
        model = build_sam3_fasterrcnn(trainable_backbone=False)
        
        # Load weights
        if args.weight_checkpoint and os.path.exists(args.weight_checkpoint):
            print(f"Found weights: {args.weight_checkpoint}. Loading...")
            state_dict = torch.load(args.weight_checkpoint, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: No weights found at {args.weight_checkpoint}. Using initialized weights.")

        model.to(device)
        model.eval()

        # Resolve dataset image path
        candidate_dirs = [
            args.base_image_dir,
            os.path.join(args.dataset_path, "images/images") if args.dataset_path else None,
            os.path.join(args.dataset_path, "images") if args.dataset_path else None,
            "/local_data/RIVA/images/images",
            "/local_data/RIVA/images",
            "./RIVA/images/images",
            "./data/RIVA/images/images",
            "./images/images"
        ]

        resolved_base_dir = None
        train_paths, val_paths, test_paths = [], [], []

        for candidate in candidate_dirs:
            if candidate and os.path.isdir(candidate):
                tr, va, te = get_data_splits(candidate)
                if len(tr) > 0 or len(va) > 0:
                    resolved_base_dir = candidate
                    train_paths, val_paths, test_paths = tr, va, te
                    break

        if not resolved_base_dir:
            raise FileNotFoundError(
                "Could not find dataset images in any of the default locations. "
                "Please specify the directory containing 'train', 'val', 'test' folders via: "
                "--base_image_dir <path> (e.g. --base_image_dir /local_data/RIVA/images/images)"
            )

        print(f"Using dataset from: {resolved_base_dir}")
        print(f"Data Splits -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

        train_features = extract_and_cache_features(model, train_paths, train_cache, device)
        val_features = extract_and_cache_features(model, val_paths, val_cache, device)
    # Test set extraction can be done later if needed for final eval
    # test_features = extract_and_cache_features(model, test_paths, os.path.join(cache_dir, "test_features.pt"), device)

    input_dim = train_features.shape[1]
    hidden_dim = int(input_dim * args.expansion_factor)
    n_groups = max(1, hidden_dim // args.block_size)
    print(f"Input Dim: {input_dim}, Hidden Dim: {hidden_dim}, BSF Groups: {n_groups} (Group Size: {args.block_size})")

    checkpoint_dir = "./interpretability_analysis/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    available_models = [
        "TopKSAE",
        "RATopKSAE",
        "MPSAE",
        "OMPSAE",
        "VanillaBSF",
        "GrassmannianBSF"
    ]

    def build_model(model_name):
        if model_name == "TopKSAE":
            return TopKSAE(input_dim, hidden_dim, top_k=args.k)
        elif model_name == "RATopKSAE":
            return RATopKSAE(input_dim, hidden_dim, points=train_features[:1024], top_k=args.k)
        elif model_name == "MPSAE":
            return MPSAE(input_dim, hidden_dim, k=args.k)
        elif model_name == "OMPSAE":
            return OMPSAE(input_dim, hidden_dim, k=args.k)
        elif model_name == "VanillaBSF":
            return VanillaBSF(d=input_dim, n_groups=n_groups, group_size=args.block_size, l0=args.k)
        elif model_name == "GrassmannianBSF":
            return GrassmannianBSF(d=input_dim, n_groups=n_groups, group_size=args.block_size, l0=args.k)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _get_dict(m):
        if hasattr(m, 'get_dictionary') and callable(m.get_dictionary):
            return m.get_dictionary()
        elif hasattr(m, 'dictionary'):
            return m.dictionary
        return None

    # Handle --quick_test mode
    if args.quick_test:
        print("\n*** [QUICK TEST MODE ENABLED] ***")
        print("Trimming dataset to 1000 samples and epochs to 2 for rapid pipeline verification.")
        args.epochs = 2
        train_features = train_features[:1000]
        val_features = val_features[:500]

    # Determine which models to train
    if "all" in args.models:
        target_models = available_models
    else:
        target_models = [m for m in args.models if m in available_models]

    trained_models = {}

    if not args.compare_only:
        print(f"\nTarget models to train: {target_models}")
        for name in target_models:
            print(f"\nInstantiating {name}...")
            sae = build_model(name)
            trained = train_sae(sae, name, train_features, val_features, args, device, logger, results_dir=args.results_dir)
            trained_models[name] = trained
            
            checkpoint_path = os.path.join(checkpoint_dir, f"{name}_sae.pt")
            torch.save(trained.state_dict(), checkpoint_path)
            print(f"Saved {name} to {checkpoint_path}")
            
            if args.logger == "wandb":
                artifact = wandb.Artifact(name=f"{name}_sae", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
    else:
        print("\n--compare_only enabled: Skipping training, proceeding directly to dictionary comparison.")

    # Dictionary comparison (gather in-memory models + existing saved checkpoints)
    print("\n--- Final Dictionary Comparison ---")
    all_dictionaries = {}

    # 1. From models just trained in this run
    for name, m in trained_models.items():
        d = _get_dict(m)
        if d is not None:
            all_dictionaries[name] = d.detach().cpu()

    # 2. From saved checkpoints on disk for models not trained in this run
    for name in available_models:
        if name not in all_dictionaries:
            ckpt_path = os.path.join(checkpoint_dir, f"{name}_sae.pt")
            if os.path.exists(ckpt_path):
                print(f"Loading existing checkpoint for comparison: {name} ({ckpt_path})")
                try:
                    m = build_model(name)
                    m.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
                    d = _get_dict(m)
                    if d is not None:
                        all_dictionaries[name] = d.detach().cpu()
                except Exception as e:
                    print(f"Warning: Could not load {name} for comparison: {e}")

    names = list(all_dictionaries.keys())
    print(f"Available dictionaries for comparison: {names}")
    
    comparison_records = []
    if len(names) < 2:
        print("Note: Less than 2 models available for pairwise comparison.")
    else:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                d1, d2 = all_dictionaries[n1], all_dictionaries[n2]
                
                # Check dimensions before Hungarian loss
                if d1.shape == d2.shape:
                    try:
                        h_loss = hungarian_loss(d1.to(device), d2.to(device))
                        ch_loss = cosine_hungarian_loss(d1.to(device), d2.to(device))
                        print(f"Comparison {n1} vs {n2}:")
                        print(f"  Hungarian Loss: {h_loss:.4f}")
                        print(f"  Cosine Hungarian Loss: {ch_loss:.4f}")
                        
                        comparison_records.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model_1": n1,
                            "model_2": n2,
                            "hungarian_loss": round(float(h_loss), 6),
                            "cosine_hungarian_loss": round(float(ch_loss), 6),
                        })

                        if logger:
                            comp_name = f"{n1}_vs_{n2}"
                            metrics = {
                                f"Comparison/{comp_name}_hungarian_loss": h_loss,
                                f"Comparison/{comp_name}_cosine_hungarian_loss": ch_loss
                            }
                            if args.logger == "wandb":
                                logger.log(metrics)
                            elif args.logger == "tensorboard":
                                logger.add_scalar(f"Comparison/{comp_name}_hungarian_loss", h_loss, 0)
                                logger.add_scalar(f"Comparison/{comp_name}_cosine_hungarian_loss", ch_loss, 0)
                    except Exception as e:
                        print(f"Comparison {n1} vs {n2} skipped due to error: {e}")
                else:
                    print(f"Comparison {n1} ({d1.shape}) vs {n2} ({d2.shape}): Shapes differ, skipping Hungarian loss.")

    if comparison_records and args.results_dir:
        comp_csv = os.path.join(args.results_dir, "sae_dictionary_comparisons.csv")
        save_records_to_csv(comp_csv, comparison_records)

    if args.logger == "wandb":
        wandb.finish()

if __name__ == "__main__":
    available_choices = ["all", "TopKSAE", "RATopKSAE", "MPSAE", "OMPSAE", "VanillaBSF", "GrassmannianBSF"]
    parser = argparse.ArgumentParser(description="Train SAEs and BSFs on SAM3 Faster R-CNN features")
    parser.add_argument("--models", nargs="+", default=["all"], choices=available_choices,
                        help="List of SAE/BSF models to train. (default: all)")
    parser.add_argument("--quick_test", action="store_true",
                        help="Fast test mode: trains for 2 epochs on 1000 samples to verify pipeline.")
    parser.add_argument("--compare_only", action="store_true",
                        help="Skip training and run dictionary comparison on existing checkpoints.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--k", type=int, default=16, help="Target sparsity (L0 / k)")
    parser.add_argument("--expansion_factor", type=float, default=4.0, help="Expansion factor for dictionary size")
    parser.add_argument("--block_size", type=int, default=4, help="Block size for Block-Sparse Featurizers (BSF)")
    parser.add_argument("--logger", type=str, choices=["wandb", "tensorboard", "none"], default="wandb", help="Logger to use")
    parser.add_argument("--results_dir", type=str, default="./interpretability_analysis/results",
                        help="Directory to save CSV logs and summaries")
    parser.add_argument("--base_image_dir", type=str, default="",
                        help="Path to folder containing 'train', 'val', 'test' image folders")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Root path to the RIVA dataset (e.g. /local_data/RIVA)")
    parser.add_argument("--weight_checkpoint", type=str, default="./weights/rcnn_best_anchors/best_checkpoint.pth", help="Path to weights checkpoint")
    
    args = parser.parse_args()
    main(args)
