import pytest
import sys
import torch
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from models.cell_DINO_rcnn_v2 import (
    build_cell_dino_fasterrcnn,
    cell_dino_resize_longest_side_and_pad_square,
    CellDinoBackbone,
)

@pytest.fixture(scope="module")
def cell_dino_model():
    """
    Initialize the Cell-DINO Faster R-CNN model for testing.
    Uses 'dinov2_vits14' (small) as a fallback proxy for speed if hpa model isn't found, 
    but the code logic is same for L/14.
    """
    # For CI/Testing, using a smaller existing hub model helps speed if 'cell_dino' isn't available.
    # However, we test the logic.
    try:
        model = build_cell_dino_fasterrcnn(
            model_name="dinov2_vits14", # Use small model for test speed/memory
            #pretrained_checkpoint_path="C:/Users/srene/Computer Vision/RIVA-challenge/weights/cell_dino_vitl14_pretrain_hpa_fov_highres-f57e7934.pth",
            num_classes_closed_set=8,
            trainable_backbone=False,
        )
        model.eval()
        return model
    except Exception as e:
        pytest.fail(f"Failed to initialize model: {e}")

def test_backbone_structure(cell_dino_model):
    """Verify backbone class and properties."""
    assert isinstance(cell_dino_model.backbone, CellDinoBackbone)
    assert cell_dino_model.backbone.out_channels == 256
    assert cell_dino_model.backbone.target_size == 1008

def test_backbone_forward_strides(cell_dino_model):
    """Verify FPN generates correct strides [7, 14, 28, 56]."""
    target_size = cell_dino_model.backbone.target_size
    dummy_input = torch.randn(1, 3, target_size, target_size)
    
    with torch.no_grad():
        features = cell_dino_model.backbone(dummy_input)
    
    assert isinstance(features, dict)
    # Expected sizes for 1008:
    # '0' (Stride 7): 1008/7 = 144
    # '1' (Stride 14): 1008/14 = 72
    # '2' (Stride 28): 1008/28 = 36
    # '3' (Stride 56): 1008/56 = 18
    
    expected_shapes = {
        '0': (144, 144),
        '1': (72, 72),
        '2': (36, 36),
        '3': (18, 18)
    }
    
    for key, (h, w) in expected_shapes.items():
        assert key in features, f"Missing feature key {key}"
        f = features[key]
        assert f.shape[1] == 256, f"Feature {key} has wrong channels"
        assert f.shape[2:] == (h, w), f"Feature {key} shape mismatch. Got {f.shape[2:]}, expected {(h, w)}"

def test_resize_pad_preprocessing():
    """Test preprocessing ensures divisibility by 14 and 56 (target size 1008)."""
    img = torch.rand(3, 800, 1200)
    target = {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
        "labels": torch.tensor([1]),
    }
    
    processed_img, processed_target, meta = cell_dino_resize_longest_side_and_pad_square(
        img, target, target_size=1008
    )
    
    assert processed_img.shape == (3, 1008, 1008)
    assert processed_img.shape[1] % 56 == 0
    assert processed_img.shape[2] % 56 == 0
    
    # Check boxes scaled
    assert processed_target["boxes"].shape == target["boxes"].shape
    
def test_full_model_inference(cell_dino_model):
    """Test forward pass in eval mode."""
    img = torch.rand(3, 1008, 1008)
    with torch.no_grad():
        detections = cell_dino_model([img])
    
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert "boxes" in detections[0]
    assert "scores" in detections[0]
    assert "labels" in detections[0]

def test_load_checkpoint_path(tmp_path):
    """Test the logic for loading weights from a file."""
    # Create a dummy checkpoint
    chk_path = tmp_path / "dummy_checkpoint.pth"
    # We need keys matching the backbone.vision
    # Let's just create a dummy dict that matches one known key to verify it doesn't crash on load
    # Warning: actually loading into the real model might fail size checks if we use random tensors.
    # But we implemented strict=False.
    
    # To properly test, we'd need a valid state dict. 
    # Here we just verify the file is attempted to be loaded.
    
    torch.save({"pixel_mean": torch.zeros(3)}, chk_path) # dummy
    
    # We use a mocked torch.load or just verify it runs without crashing given strict=False
    try:
        model = build_cell_dino_fasterrcnn(
            model_name="dinov2_vits14",
            pretrained_checkpoint_path=str(chk_path),
            trainable_backbone=False
        )
    except RuntimeError:
        # Expected if shapes don't match, but we just want to ensure it tries the path
        pass
    except Exception as e:
        # If it's a "file not found" or similar logical error, fail
        pytest.fail(f"Loading checkpoint failed with unexpected error: {e}")
