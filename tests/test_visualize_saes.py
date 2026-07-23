import pytest
import torch
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'interpretability_analysis'))

from interpretability_analysis.visualize_saes import extract_images_and_heatmaps, main

def test_extract_images_and_heatmaps_normal():
    # Mock SAM3 backbone
    sam3_model = MagicMock()
    mock_backbone = MagicMock()
    sam3_model.backbone = mock_backbone
    
    hook_fn_ref = []
    def register_hook(fn):
        hook_fn_ref.append(fn)
        return MagicMock()
    mock_backbone.register_forward_hook = register_hook
    
    # Mock SAE model
    sae_model = MagicMock()
    # SAE forward returns (z_pre, z, x_hat)
    def sae_forward(feat_flat):
        num_features = feat_flat.shape[0]
        # Return mock z with shape [B*H*W, num_concepts]
        num_concepts = 10
        return None, torch.randn(num_features, num_concepts), None
    sae_model.side_effect = sae_forward

    # Mock Dataloader
    batch_size = 2
    dummy_batch = torch.randn(batch_size, 3, 1008, 1008)
    dataloader = [dummy_batch]
    
    # Mock backbone forward
    def sam3_forward(x):
        # B=2, C=4, H=16, W=16
        out = {"1": torch.randn(2, 4, 16, 16)}
        if hook_fn_ref:
            hook_fn_ref[0](mock_backbone, x, out)
        return out
    mock_backbone.side_effect = sam3_forward
    
    device = torch.device('cpu')
    
    images, heatmaps = extract_images_and_heatmaps(sam3_model, sae_model, dataloader, device)
    
    assert images.shape == (2, 3, 1008, 1008)
    assert heatmaps.shape == (2, 16, 16, 10)

def test_extract_images_and_heatmaps_empty_dataloader():
    # Edge case: Empty dataloader
    sam3_model = MagicMock()
    sae_model = MagicMock()
    dataloader = []
    device = torch.device('cpu')
    
    with pytest.raises(ValueError):
        # torch.cat fails on empty sequence
        extract_images_and_heatmaps(sam3_model, sae_model, dataloader, device)

@patch('interpretability_analysis.visualize_saes.build_sam3_fasterrcnn')
@patch('interpretability_analysis.visualize_saes.get_data_splits')
@patch('interpretability_analysis.visualize_saes.ImagePathDataset')
@patch('interpretability_analysis.visualize_saes.DataLoader')
@patch('interpretability_analysis.visualize_saes.TopKSAE')
@patch('interpretability_analysis.visualize_saes.extract_images_and_heatmaps')
@patch('interpretability_analysis.visualize_saes.overlay_top_heatmaps')
@patch('torch.load')
@patch('os.path.exists')
def test_main_visualize_normal(mock_exists, mock_load, mock_overlay, mock_extract, mock_sae_class,
                               mock_dataloader, mock_dataset, mock_get_splits, mock_build_sam3):
    
    # Mocks setup
    mock_exists.return_value = True
    mock_load.return_value = {} # Empty state dict is fine for mocked model
    
    mock_sam3 = MagicMock()
    mock_build_sam3.return_value = mock_sam3
    
    mock_get_splits.return_value = ([], [], ["dummy1.jpg", "dummy2.jpg"])
    
    mock_extract.return_value = (torch.randn(2, 3, 1008, 1008), torch.randn(2, 16, 16, 10))
    mock_overlay.return_value = MagicMock() # Mock figure
    
    args = MagicMock()
    args.sae_type = "TopKSAE"
    args.checkpoint = ""
    args.concept_id = 5
    args.k = 16
    args.expansion_factor = 4.0
    args.batch_size = 4
    args.base_image_dir = "./fake_dir"
    args.visualizations = ["overlay"] # Test one visualization
    args.save_dir = "./fake_out_dir"
    
    # Run
    main(args)
    
    # Assertions
    mock_build_sam3.assert_called_once()
    mock_get_splits.assert_called_once()
    mock_extract.assert_called_once()
    mock_overlay.assert_called_once()

@patch('interpretability_analysis.visualize_saes.build_sam3_fasterrcnn')
@patch('interpretability_analysis.visualize_saes.get_data_splits')
def test_main_visualize_invalid_sae(mock_get_splits, mock_build_sam3):
    # Edge case: Invalid SAE type
    mock_sam3 = MagicMock()
    mock_build_sam3.return_value = mock_sam3
    mock_get_splits.return_value = ([], [], ["dummy1.jpg"])
    
    args = MagicMock()
    args.sae_type = "InvalidSAE"
    args.base_image_dir = "./fake_dir"
    args.batch_size = 1
    args.expansion_factor = 4.0
    
    with pytest.raises(ValueError, match="Unknown SAE type"):
        main(args)
