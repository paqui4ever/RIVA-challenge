import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from PIL import Image

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'interpretability_analysis'))

from interpretability_analysis.train_saes import (
    ImagePathDataset,
    get_data_splits,
    get_lr_scheduler,
    custom_criterion,
    extract_and_cache_features,
    train_sae
)

@pytest.fixture
def dummy_image_file(tmp_path):
    img_path = tmp_path / "dummy.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return str(img_path)

def test_image_path_dataset_normal(dummy_image_file):
    dataset = ImagePathDataset([dummy_image_file], target_size=1008)
    assert len(dataset) == 1
    
    img_tensor = dataset[0]
    assert img_tensor.shape == (3, 1008, 1008)

def test_image_path_dataset_empty():
    # Edge case: empty paths list
    dataset = ImagePathDataset([], target_size=1008)
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]

def test_get_data_splits_even(tmp_path):
    # Setup dummy directory structure
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "test").mkdir()
    
    for i in range(10):
        (tmp_path / "train" / f"{i}.jpg").touch()
        (tmp_path / "test" / f"{i}.jpg").touch()
        (tmp_path / "val" / f"{i}.jpg").touch()
        
    train_test, val_split_1, val_split_2 = get_data_splits(str(tmp_path), seed=42)
    
    # Train = 10 train + 10 test = 20
    assert len(train_test) == 20
    # Val total = 10. Split 50/50 -> 5 and 5
    assert len(val_split_1) == 5
    assert len(val_split_2) == 5

def test_get_data_splits_odd(tmp_path):
    # Edge case: Odd number of validation files
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "test").mkdir()
    
    (tmp_path / "train" / "1.jpg").touch()
    (tmp_path / "test" / "1.jpg").touch()
    # 3 val files
    for i in range(3):
        (tmp_path / "val" / f"{i}.jpg").touch()
        
    train_test, val_split_1, val_split_2 = get_data_splits(str(tmp_path), seed=42)
    
    assert len(train_test) == 2
    # Val total = 3. 3//2 = 1. val_split_1 = 1, val_split_2 = 2
    assert len(val_split_1) == 1
    assert len(val_split_2) == 2

def test_get_data_splits_empty(tmp_path):
    # Edge case: Empty directories
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "test").mkdir()
        
    train_test, val_split_1, val_split_2 = get_data_splits(str(tmp_path), seed=42)
    
    assert len(train_test) == 0
    assert len(val_split_1) == 0
    assert len(val_split_2) == 0

def test_get_lr_scheduler_normal():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=3e-4)
    warmup_steps = 10
    total_steps = 100
    initial_lr = 3e-5
    base_lr = 3e-4
    
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, initial_lr, base_lr)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr)
    
    for _ in range(warmup_steps):
        optimizer.step()
        scheduler.step()
        
    assert optimizer.param_groups[0]['lr'] == pytest.approx(base_lr)

def test_get_lr_scheduler_edge_case():
    # Edge case: warmup steps larger than total steps (though practically shouldn't happen, we should test robustness)
    # Actually, CosineAnnealingLR might fail if T_max is <= 0. In train_saes, it's T_max = total_steps - warmup_steps
    # So if total_steps == warmup_steps, T_max = 0, which raises an error in CosineAnnealingLR.
    # In `train_saes`, total_steps is epochs * steps_per_epoch, warmup is max(1, 0.05 * total).
    # What if total_steps = 1? warmup = max(1, 0) = 1. T_max = 1 - 1 = 0.
    
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=3e-4)
    warmup_steps = 1
    total_steps = 1
    
    # Ensure it doesn't crash on T_max=0 (handled gracefully in newer torch)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, 3e-5, 3e-4)
    assert scheduler is not None

def test_custom_criterion():
    x = torch.tensor([1.0, 2.0])
    x_hat = torch.tensor([1.0, 3.0])
    loss = custom_criterion(x, x_hat, None, None, None)
    assert loss.item() == 0.5

@patch('interpretability_analysis.train_saes.ImagePathDataset')
@patch('interpretability_analysis.train_saes.DataLoader')
def test_extract_and_cache_features_normal(mock_dataloader, mock_dataset, tmp_path):
    mock_model = MagicMock()
    mock_backbone = MagicMock()
    mock_model.backbone = mock_backbone
    
    hook_fn_ref = []
    def register_hook(fn):
        hook_fn_ref.append(fn)
        return MagicMock()
    
    mock_backbone.register_forward_hook = register_hook
    
    dummy_batch = torch.randn(2, 3, 1008, 1008)
    mock_dataloader.return_value = [dummy_batch]
    
    def dummy_forward(x):
        out = {"1": torch.randn(2, 4, 16, 16)}
        if hook_fn_ref:
            hook_fn_ref[0](mock_backbone, x, out)
        return out
        
    mock_backbone.side_effect = dummy_forward
    
    cache_path = tmp_path / "features.pt"
    device = torch.device('cpu')
    
    features = extract_and_cache_features(mock_model, ["dummy_path.jpg"], str(cache_path), device, batch_size=2)
    
    # Flattened shape check: B=2, H=16, W=16, C=4 -> B*H*W = 2*16*16 = 512
    assert features.shape == (512, 4)
    assert cache_path.exists()
    
    # Cache hit check
    features_cached = extract_and_cache_features(mock_model, ["dummy_path.jpg"], str(cache_path), device, batch_size=2)
    assert features_cached.shape == (512, 4)
    assert mock_backbone.call_count == 1

@patch('interpretability_analysis.train_saes.ImagePathDataset')
@patch('interpretability_analysis.train_saes.DataLoader')
def test_extract_and_cache_features_empty_dataloader(mock_dataloader, mock_dataset, tmp_path):
    # Edge case: Empty dataloader
    mock_model = MagicMock()
    mock_backbone = MagicMock()
    mock_model.backbone = mock_backbone
    
    hook_fn_ref = []
    def register_hook(fn):
        hook_fn_ref.append(fn)
        return MagicMock()
    
    mock_backbone.register_forward_hook = register_hook
    mock_dataloader.return_value = [] # Empty
    
    cache_path = tmp_path / "features_empty.pt"
    device = torch.device('cpu')
    
    # Should throw error when calling torch.cat on empty sequence
    with pytest.raises(ValueError):
        extract_and_cache_features(mock_model, [], str(cache_path), device, batch_size=2)

@patch('interpretability_analysis.train_saes.lib_train_sae')
def test_train_sae_normal(mock_lib_train_sae):
    mock_lib_train_sae.return_value = {'avg_loss': [0.1]}
    
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    # Mock forward return to match batch sizes
    mock_model.side_effect = lambda x: (torch.randn(x.size(0), 8), torch.randn(x.size(0), 8), torch.randn(x.size(0), 4))
    mock_model.get_dictionary.return_value = torch.randn(8, 4)
    
    train_tensor = torch.randn(20, 4)
    val_tensor = torch.randn(10, 4)
    
    args = MagicMock()
    args.batch_size = 2
    args.epochs = 2 # At least 2 epochs to avoid T_max=0 in scheduler for total_steps
    args.lr = 1e-3
    args.logger = "none"
    
    device = torch.device('cpu')
    logger = None
    
    trained_model = train_sae(mock_model, "TestSAE", train_tensor, val_tensor, args, device, logger)
    
    assert trained_model == mock_model
    assert mock_lib_train_sae.call_count == args.epochs

@patch('interpretability_analysis.train_saes.lib_train_sae')
def test_train_sae_small_dataset(mock_lib_train_sae):
    # Edge case: train/val tensors are smaller than 10000, testing slicing logic
    mock_lib_train_sae.return_value = {'avg_loss': [0.1]}
    
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    # Ensure forward return shape matches the small inputs (e.g., 5 rows)
    mock_model.side_effect = lambda x: (torch.randn(x.size(0), 8), torch.randn(x.size(0), 8), torch.randn(x.size(0), 4))
    mock_model.get_dictionary.return_value = torch.randn(8, 4)
    
    # Small tensors
    train_tensor = torch.randn(5, 4)
    val_tensor = torch.randn(3, 4)
    
    args = MagicMock()
    args.batch_size = 2
    args.epochs = 2
    args.lr = 1e-3
    args.logger = "none"
    
    device = torch.device('cpu')
    logger = None
    
    # Should run without index out of bounds error
    trained_model = train_sae(mock_model, "TestSAE", train_tensor, val_tensor, args, device, logger)
    assert trained_model == mock_model
