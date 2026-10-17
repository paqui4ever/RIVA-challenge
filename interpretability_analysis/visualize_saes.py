import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add the parent directory to the path to import models from the root of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sam3_rcnn_v2 import build_sam3_fasterrcnn

# Import dataset and splitting utility from the training script
from interpretability_analysis.train_saes import get_data_splits, ImagePathDataset

# Import SAE models
from overcomplete import TopKSAE
from custom_saes import (
    RATopKSAE, 
    OMPSAE, 
    MpSAE as MPSAE,
    # Block-Sparse Featurizers (Goodfire AI: https://github.com/goodfire-ai/block-sparse-featurizer)
    BSF,
    VanillaBSF,
    GrassmannianBSF,
    plot_block_concepts_manifold
)


# Import overcomplete visualization functions
from overcomplete.visualization import (
    overlay_top_heatmaps,
    evidence_top_images,
    zoom_top_images,
    contour_top_image
)

def extract_images_and_heatmaps(sam3_model, sae_model, dataloader, device):
    all_images = []
    all_heatmaps = []
    all_z_blocks = []

    # Hook to capture FPN features
    activations = {}
    def hook_fn(module, input, output):
        activations["1"] = output["1"].detach()

    handle = sam3_model.backbone.register_forward_hook(hook_fn)

    sam3_model.eval()
    sae_model.eval()

    is_bsf = hasattr(sae_model, "encode_blocks") and callable(sae_model.encode_blocks)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            batch = batch.to(device)
            # Store original padded images (B, C, 1008, 1008)
            all_images.append(batch.cpu())

            # Forward pass through backbone
            _ = sam3_model.backbone(batch)
            
            feat = activations["1"]
            B, C, H_feat, W_feat = feat.shape
            
            # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
            feat_flat = feat.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
            
            # Pass through SAE. SAE forward returns (z_pre, z, x_hat)
            _, z, _ = sae_model(feat_flat)
            
            # Reshape activations z back to spatial dimensions: [B, H_feat, W_feat, num_concepts]
            num_concepts = z.shape[-1]
            z_spatial = z.view(B, H_feat, W_feat, num_concepts)
            all_heatmaps.append(z_spatial.cpu())

            if is_bsf:
                z_blk = sae_model.encode_blocks(feat_flat)
                all_z_blocks.append(z_blk.cpu())

    handle.remove()

    images_tensor = torch.cat(all_images, dim=0)
    heatmaps_tensor = torch.cat(all_heatmaps, dim=0)
    z_blocks_tensor = torch.cat(all_z_blocks, dim=0) if all_z_blocks else None

    return images_tensor, heatmaps_tensor, z_blocks_tensor, H_feat


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load SAM3 Backbone
    print("Loading SAM3 Faster R-CNN...")
    sam3_model = build_sam3_fasterrcnn(trainable_backbone=False)
    
    # Load weights
    ckpt_to_load = args.weight_checkpoint
    if not (ckpt_to_load and os.path.exists(ckpt_to_load)):
        weight_dir = Path("./weights")
        weight_files = list(weight_dir.rglob("*.pt")) + list(weight_dir.rglob("*.pth"))
        ckpt_to_load = str(weight_files[0]) if weight_files else None

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"Found SAM3 Faster R-CNN weights: {ckpt_to_load}. Loading...")
        state_dict = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        sam3_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: No weights found at {args.weight_checkpoint}. Using initialized weights.")

    sam3_model.to(device)
    sam3_model.eval()

    # 2. Get Test Dataset
    print(f"Loading data from {args.base_image_dir}...")
    _, _, test_paths = get_data_splits(args.base_image_dir)
    print(f"Test set size: {len(test_paths)} images")
    
    test_dataset = ImagePathDataset(test_paths)
    # Using a small batch size to fit in memory (images are 1008x1008)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Determine input_dim (assume same as train_saes.py logic for FPN)
    # The SAM3 FPN output typically has 256 channels
    input_dim = 256
    hidden_dim = int(input_dim * args.expansion_factor)
    print(f"SAE Input Dim: {input_dim}, Hidden Dim: {hidden_dim}")

    # Load RATopK anchor points if needed
    loaded_points = None
    if args.sae_type == "RATopKSAE":
        points_path = "./interpretability_analysis/checkpoints/ratopk_points.pt"
        if os.path.exists(points_path):
            print(f"Loading RATopK anchor points from {points_path}")
            loaded_points = torch.load(points_path, map_location=device, weights_only=False)
        else:
            feat_path = "./interpretability_analysis/features/train_features.pt"
            if os.path.exists(feat_path):
                print(f"Sampling RATopK anchor points from {feat_path}")
                feats = torch.load(feat_path, map_location=device, weights_only=False)
                step = max(1, len(feats) // 1024)
                loaded_points = feats[::step][:1024].clone()
            else:
                print("Warning: Anchor points and train_features not found, falling back to random points")
                loaded_points = torch.randn(1024, input_dim, device=device)

    # 3. Initialize and Load SAE
    sae_classes = {
        "TopKSAE": lambda d, h, k: TopKSAE(d, h, top_k=k),
        "RATopKSAE": lambda d, h, k: RATopKSAE(d, h, points=loaded_points.to(device), top_k=k),
        "MPSAE": lambda d, h, k: MPSAE(d, h, k=k),
        "OMPSAE": lambda d, h, k: OMPSAE(d, h, k=k),
        # Block-Sparse Featurizers (Goodfire AI: https://github.com/goodfire-ai/block-sparse-featurizer)
        "VanillaBSF": lambda d, h, k: VanillaBSF(d, n_groups=max(1, h // args.block_size), group_size=args.block_size, l0=k),
        "GrassmannianBSF": lambda d, h, k: GrassmannianBSF(d, n_groups=max(1, h // args.block_size), group_size=args.block_size, l0=k),
    }

    if args.sae_type not in sae_classes:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")

    sae_model = sae_classes[args.sae_type](input_dim, hidden_dim, k=args.k)
    sae_model.to(device)
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = f"./interpretability_analysis/checkpoints/{args.sae_type}_sae.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading SAE checkpoint from {checkpoint_path}")
        sae_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    else:
        print(f"Warning: SAE checkpoint {checkpoint_path} not found. Using untrained SAE.")
    
    sae_model.eval()

    # 4. Extract Images and Heatmaps
    print("Extracting images and computing heatmaps...")
    images, heatmaps, z_blocks, grid_size = extract_images_and_heatmaps(sam3_model, sae_model, test_loader, device)
    
    print(f"Images tensor shape: {images.shape}")
    print(f"Heatmaps tensor shape: {heatmaps.shape}")

    # 5. Generate Visualizations
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_id = args.concept_id
    print(f"Generating visualizations for Concept ID {concept_id} of {args.sae_type}...")

    def save_fig(fig, filename):
        save_path = out_dir / filename
        if fig is not None:
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        print(f"Saved visualization to {save_path}")

    # Visualization 1: Overlay Top Heatmaps
    if "overlay" in args.visualizations:
        try:
            fig_overlay = overlay_top_heatmaps(images, heatmaps, concept_id)
            save_fig(fig_overlay, f"{args.sae_type}_concept_{concept_id}_overlay.png")
        except Exception as e:
            print(f"Error generating overlay_top_heatmaps: {e}")

    # Visualization 2: Evidence Top Images
    if "evidence" in args.visualizations:
        try:
            fig_evidence = evidence_top_images(images, heatmaps, concept_id)
            save_fig(fig_evidence, f"{args.sae_type}_concept_{concept_id}_evidence.png")
        except Exception as e:
            print(f"Error generating evidence_top_images: {e}")

    # Visualization 3: Zoom Top Images
    if "zoom" in args.visualizations:
        try:
            fig_zoom = zoom_top_images(images, heatmaps, concept_id)
            save_fig(fig_zoom, f"{args.sae_type}_concept_{concept_id}_zoom.png")
        except Exception as e:
            print(f"Error generating zoom_top_images: {e}")

    # Visualization 4: Contour Top Image
    if "contour" in args.visualizations:
        try:
            fig_contour = contour_top_image(images, heatmaps, concept_id)
            save_fig(fig_contour, f"{args.sae_type}_concept_{concept_id}_contour.png")
        except Exception as e:
            print(f"Error generating contour_top_image: {e}")

    # Visualization 5: 3D Manifold Geometry & Patch Overlay (BSF Specific)
    if ("manifold" in args.visualizations or isinstance(sae_model, BSF)) and z_blocks is not None:
        try:
            # images: tensor [N, 3, H, W] -> numpy [N, H, W, 3] in [0, 1]
            imgs_np = images.permute(0, 2, 3, 1).cpu().numpy().clip(0, 1)
            save_manifold_path = str(out_dir / f"{args.sae_type}_concept_{concept_id}_manifold.png")
            plot_block_concepts_manifold(
                z_blocks=z_blocks.numpy(),
                atoms=sae_model.atoms().cpu().numpy(),
                images=imgs_np,
                concepts=[concept_id],
                grid_size=grid_size,
                n_img=5,
                save_path=save_manifold_path
            )
        except Exception as e:
            print(f"Error generating plot_block_concepts_manifold: {e}")

    print("Visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize learned concepts from trained SAEs.")
    parser.add_argument("--sae_type", type=str, default="TopKSAE", 
                        choices=["TopKSAE", "RATopKSAE", "MPSAE", "OMPSAE", "VanillaBSF", "GrassmannianBSF"], 
                        help="Type of SAE to visualize")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to SAE checkpoint. If empty, tries default path.")
    parser.add_argument("--concept_id", type=int, default=5, help="Concept ID to visualize")
    parser.add_argument("--k", type=int, default=16, help="Target sparsity (L0 / k) used during training")
    parser.add_argument("--expansion_factor", type=float, default=4.0, help="Expansion factor used during training")
    parser.add_argument("--block_size", type=int, default=4, help="Block size for Block-Sparse Featurizers (BSF)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference (reduce if OOM)")
    parser.add_argument("--base_image_dir", type=str, default="./RIVA/images/images", help="Base directory for dataset")
    parser.add_argument("--visualizations", nargs="+", default=["overlay", "evidence", "zoom", "contour", "manifold"], 
                        choices=["overlay", "evidence", "zoom", "contour", "manifold"], help="Visualization methods to run")
    parser.add_argument("--save_dir", type=str, default="./interpretability_analysis/visualizations", help="Directory to save the visualizations")
    parser.add_argument("--weight_checkpoint", type=str, default="./weights/rcnn_best_anchors/best_checkpoint.pth",
                        help="Path to SAM3 Faster R-CNN weights checkpoint")
    
    args = parser.parse_args()
    main(args)
