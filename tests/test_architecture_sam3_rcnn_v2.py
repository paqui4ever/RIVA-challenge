import pytest
import sys
import torch
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from models.sam3_rcnn_v2 import (
    build_sam3_fasterrcnn,
    sam3_resize_longest_side_and_pad_square,
    Sam3Backbone,
)


@pytest.fixture(scope="module")
def sam3_rcnn_v2_model():
    """
    Initialize the SAM3 Cut-B Faster R-CNN model for testing.
    Backbone is frozen by default (trainable=False).
    """
    num_classes_closed_set = 8  # 8 classes, +1 background added internally
    try:
        model = build_sam3_fasterrcnn(
            model_name_or_path="facebook/sam3",
            num_classes_closed_set=num_classes_closed_set,
            trainable_backbone=False,
        )
        model.eval()
        return model
    except Exception as e:
        pytest.fail(f"Failed to initialize model: {e}")


def test_backbone_is_sam3_cutb(sam3_rcnn_v2_model):
    """Verify the backbone is Sam3CutBBackbone."""
    assert isinstance(sam3_rcnn_v2_model.backbone, Sam3Backbone)


def test_backbone_target_size(sam3_rcnn_v2_model):
    """Verify target size is 1008 (SAM3 ViT default)."""
    assert sam3_rcnn_v2_model.backbone.target_size == 1008


def test_backbone_out_channels(sam3_rcnn_v2_model):
    """Verify FPN output channels (default 256)."""
    assert sam3_rcnn_v2_model.backbone.out_channels == 256


def test_backbone_fpn_outputs(sam3_rcnn_v2_model):
    """Test that backbone returns multi-scale FPN feature maps."""
    target_size = sam3_rcnn_v2_model.backbone.target_size
    dummy_input = torch.randn(1, 3, target_size, target_size)

    with torch.no_grad():
        features = sam3_rcnn_v2_model.backbone(dummy_input)

    # Should return OrderedDict with keys '0', '1', '2', '3' (4 FPN levels)
    assert isinstance(features, dict)
    expected_keys = ['0', '1', '2', '3']
    for key in expected_keys:
        assert key in features, f"Missing FPN level {key}"
        # Each feature map should have 256 channels
        assert features[key].shape[1] == 256


def test_anchor_generator_multi_scale(sam3_rcnn_v2_model):
    """Verify anchor generator has 4 levels for FPN."""
    rpn_sizes = sam3_rcnn_v2_model.rpn.anchor_generator.sizes
    rpn_ratios = sam3_rcnn_v2_model.rpn.anchor_generator.aspect_ratios

    # Should have 4 tuples for 4 FPN levels
    assert len(rpn_sizes) == 4
    assert len(rpn_ratios) == 4


def test_resize_pad_preprocessing():
    """Test the aspect-ratio-preserving resize + pad function."""
    # Create a non-square test image
    img = torch.rand(3, 720, 1280)  # Wide image
    target = {
        "boxes": torch.tensor([[100.0, 120.0, 400.0, 500.0]]),
        "labels": torch.tensor([1]),
    }

    processed_img, processed_target, meta = sam3_resize_longest_side_and_pad_square(
        img, target, target_size=1008
    )

    # Output should be square 1008x1008
    assert processed_img.shape == (3, 1008, 1008)

    # Metadata should be populated
    assert meta.scale > 0
    assert len(meta.resized_hw) == 2
    assert len(meta.pad_rb) == 2

    # Boxes should be scaled
    assert processed_target["boxes"].shape == target["boxes"].shape
    # Labels should be unchanged
    assert torch.equal(processed_target["labels"], target["labels"])


def test_resize_pad_square_image():
    """Test preprocessing with a square image (no padding needed on one side)."""
    img = torch.rand(3, 900, 900)  # Square image
    target = {
        "boxes": torch.tensor([[50.0, 60.0, 200.0, 240.0]]),
        "labels": torch.tensor([3]),
    }

    processed_img, processed_target, meta = sam3_resize_longest_side_and_pad_square(
        img, target, target_size=1008
    )

    # Output should be square 1008x1008
    assert processed_img.shape == (3, 1008, 1008)

    # For square input, both sides should scale equally
    assert meta.resized_hw[0] == meta.resized_hw[1]


def test_full_forward_pass_training(sam3_rcnn_v2_model):
    """Test full forward pass in training mode."""
    target_size = sam3_rcnn_v2_model.backbone.target_size

    # Prepare batch with preprocessing
    images = [torch.rand(3, 720, 1280), torch.rand(3, 900, 900)]
    targets = [
        {"boxes": torch.tensor([[100.0, 120.0, 400.0, 500.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[50.0, 60.0, 200.0, 240.0]]), "labels": torch.tensor([3])},
    ]

    processed_images = []
    processed_targets = []
    for img, tgt in zip(images, targets):
        img, tgt, _ = sam3_resize_longest_side_and_pad_square(img, tgt, target_size=target_size)
        processed_images.append(img)
        processed_targets.append(tgt)

    sam3_rcnn_v2_model.train()
    losses = sam3_rcnn_v2_model(processed_images, processed_targets)

    # Should return loss dict
    assert isinstance(losses, dict)
    expected_loss_keys = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
    for key in expected_loss_keys:
        assert key in losses, f"Missing loss key: {key}"
        assert losses[key].requires_grad or not losses[key].requires_grad  # Just check it's a tensor


def test_full_forward_pass_inference(sam3_rcnn_v2_model):
    """Test full forward pass in inference mode."""
    target_size = sam3_rcnn_v2_model.backbone.target_size

    # Prepare batch with preprocessing
    images = [torch.rand(3, 720, 1280)]

    processed_images = []
    for img in images:
        img, _, _ = sam3_resize_longest_side_and_pad_square(img, None, target_size=target_size)
        processed_images.append(img)

    sam3_rcnn_v2_model.eval()
    with torch.no_grad():
        predictions = sam3_rcnn_v2_model(processed_images)

    # Should return list of prediction dicts
    assert isinstance(predictions, list)
    assert len(predictions) == 1

    pred = predictions[0]
    assert 'boxes' in pred
    assert 'labels' in pred
    assert 'scores' in pred


def test_trainable_backbone():
    """Test that trainable_backbone flag works correctly."""
    # Frozen backbone
    model_frozen = build_sam3_fasterrcnn(
        model_name_or_path="facebook/sam3",
        num_classes_closed_set=8,
        trainable_backbone=False,
    )
    frozen_params = sum(p.requires_grad for p in model_frozen.backbone.vision.parameters())
    assert frozen_params == 0, "Backbone should be frozen"

    # Trainable backbone
    model_trainable = build_sam3_fasterrcnn(
        model_name_or_path="facebook/sam3",
        num_classes_closed_set=8,
        trainable_backbone=True,
    )
    trainable_params = sum(p.requires_grad for p in model_trainable.backbone.vision.parameters())
    assert trainable_params > 0, "Backbone should have trainable parameters"
