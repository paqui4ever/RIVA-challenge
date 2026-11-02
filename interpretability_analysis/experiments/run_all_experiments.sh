#!/usr/bin/env bash
set -e

# ==============================================================================
# End-to-End Interpretability Analysis Pipeline on Faster R-CNN box_head (1024-D)
# ==============================================================================

echo "======================================================================"
echo "Step 1: Train all 6 SAE/BSF models on box_head (input_dim = 1024)"
echo "======================================================================"
python interpretability_analysis/train_saes.py \
    --target_layer box_head \
    --models all \
    --epochs 10 \
    --batch_size 1024 \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --overwrite_results \
    --logger wandb

echo "======================================================================"
echo "Step 2: Evaluate Downstream Faster R-CNN Faithfulness on box_head"
echo "======================================================================"
python interpretability_analysis/evaluate_downstream_faithfulness.py \
    --target_layer box_head \
    --split val \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

echo "======================================================================"
echo "Step 3: Run Supervised Bethesda Concept Attribution on box_head"
echo "======================================================================"
python interpretability_analysis/bethesda_concept_attribution.py \
    --target_layer box_head \
    --sae_types GrassmannianBSF VanillaBSF OMPSAE MPSAE TopKSAE RATopKSAE \
    --split val \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

echo "======================================================================"
echo "Step 4: Run Unsupervised Manifold Analysis & Concept Ranking on box_head"
echo "======================================================================"
python interpretability_analysis/analyze_bsf_manifolds.py \
    --target_layer box_head \
    --sae_type GrassmannianBSF \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/analyze_bsf_manifolds.py \
    --target_layer box_head \
    --sae_type VanillaBSF \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/analyze_bsf_manifolds.py \
    --target_layer box_head \
    --sae_type OMPSAE \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/analyze_bsf_manifolds.py \
    --target_layer box_head \
    --sae_type TopKSAE \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

echo "======================================================================"
echo "Step 5: Generate Paper Figure 9 & Standard Visualizations on box_head"
echo "======================================================================"
python interpretability_analysis/visualize_bsf_paper_fig9.py \
    --target_layer box_head \
    --sae_type GrassmannianBSF \
    --directional_sae_type OMPSAE \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/visualize_bsf_paper_fig9.py \
    --target_layer box_head \
    --sae_type VanillaBSF \
    --directional_sae_type OMPSAE \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/visualize_saes.py \
    --target_layer box_head \
    --sae_type GrassmannianBSF \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/visualize_saes.py \
    --target_layer box_head \
    --sae_type OMPSAE \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

python interpretability_analysis/visualize_saes.py \
    --target_layer box_head \
    --sae_type TopKSAE \
    --k 32 \
    --block_size 4 \
    --expansion_factor 4.0 \
    --batch_size 2

echo "======================================================================"
echo "All box_head interpretability experiments completed successfully!"
echo "======================================================================"
