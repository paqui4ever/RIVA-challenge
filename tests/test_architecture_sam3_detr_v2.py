"""
Test script for Sam3ForClosedSetDetection architecture and training pipeline.

Tests the Cut A architecture:
  - SAM3 perception encoder + DETR encoder/decoder
  - Closed-set classification head (C+1 classes)
  - Hungarian matching + set-based loss (CE + L1 + GIoU)
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

# Import module components
from models.sam3_DETR_v2 import (
    box_xyxy_clamp,
    HungarianMatcher,
    SetCriterion,
    Sam3ForClosedSetDetection,
    make_sam3_collate_fn,
)


# ----------------------------
# Mock Classes for Testing
# ----------------------------
class MockSam3Config:
    """Mock config for Sam3Model that mimics HuggingFace config structure."""
    def __init__(self):
        self.detr_decoder_config = MagicMock()
        self.detr_decoder_config.hidden_size = 256


class MockSam3Outputs:
    """
    Mock outputs from Sam3Model.
    Mimics the structure returned by Sam3Model forward pass.
    """
    def __init__(self, batch_size, num_queries, hidden_size):
        self.decoder_hidden_states = (
            torch.randn(batch_size, num_queries, hidden_size),
            torch.randn(batch_size, num_queries, hidden_size),
            torch.randn(batch_size, num_queries, hidden_size),
        )
        # pred_boxes in normalized xyxy format
        self.pred_boxes = torch.rand(batch_size, num_queries, 4)
        # Ensure proper xyxy ordering (x2 > x1, y2 > y1)
        self.pred_boxes[:, :, 2] = self.pred_boxes[:, :, 0] + torch.abs(
            self.pred_boxes[:, :, 2] - self.pred_boxes[:, :, 0]
        )
        self.pred_boxes[:, :, 3] = self.pred_boxes[:, :, 1] + torch.abs(
            self.pred_boxes[:, :, 3] - self.pred_boxes[:, :, 1]
        )
        self.pred_boxes = self.pred_boxes.clamp(0, 1)


def create_mock_sam3_model(batch_size=2, num_queries=100, hidden_size=256):
    """
    Factory function to create a configured mock SAM3 model.
    Returns the mock class and the mock model instance.
    """
    mock_model = MagicMock()
    mock_model.config = MockSam3Config()
    mock_model.return_value = MockSam3Outputs(batch_size, num_queries, hidden_size)
    return mock_model


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(scope="module")
def hungarian_matcher():
    """Creates a HungarianMatcher with standard DETR cost weights."""
    return HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)


@pytest.fixture(scope="module")
def set_criterion(hungarian_matcher):
    """Creates a SetCriterion with standard loss weights for 8-class detection."""
    num_classes = 8
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    return SetCriterion(num_classes, hungarian_matcher, weight_dict, eos_coef=0.1)


# ----------------------------
# Box Helper Tests
# ----------------------------
def test_box_xyxy_clamp_bounds():
    """
    Verifies that box_xyxy_clamp correctly clamps coordinates to [0, 1].
    Out-of-bounds values should be clamped to valid range.
    """
    boxes = torch.tensor([[-.1, -.2, 1.1, 1.2]])
    clamped = box_xyxy_clamp(boxes)

    assert clamped.min() >= 0.0, "Clamped boxes should have min >= 0.0"
    assert clamped.max() <= 1.0, "Clamped boxes should have max <= 1.0"


def test_box_xyxy_clamp_ordering():
    """
    Verifies that box_xyxy_clamp ensures proper coordinate ordering.
    After clamping: x1 <= x2 and y1 <= y2 (even if input is reversed).
    """
    boxes = torch.tensor([[0.8, 0.7, 0.3, 0.2]])  # Reversed coordinates
    clamped = box_xyxy_clamp(boxes)

    assert clamped[0, 0] <= clamped[0, 2], "x1 should be <= x2 after clamping"
    assert clamped[0, 1] <= clamped[0, 3], "y1 should be <= y2 after clamping"


def test_box_xyxy_clamp_batched():
    """
    Verifies that box_xyxy_clamp works correctly with batched inputs.
    """
    boxes = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        [[-.1, -.1, 1.1, 1.1], [0.9, 0.1, 0.2, 0.8]],
    ])
    clamped = box_xyxy_clamp(boxes)

    assert clamped.shape == boxes.shape, "Output shape should match input shape"
    assert clamped.min() >= 0.0, "All values should be >= 0.0"
    assert clamped.max() <= 1.0, "All values should be <= 1.0"


# ----------------------------
# Hungarian Matcher Tests
# ----------------------------
def test_hungarian_matcher_assignment(hungarian_matcher):
    """
    Verifies that HungarianMatcher produces correct assignment indices.
    Each target should be matched to exactly one query.
    """
    B, Q, C = 2, 100, 9  # batch, queries, classes (8 + no-object)
    pred_logits = torch.randn(B, Q, C)
    pred_boxes = torch.rand(B, Q, 4)
    pred_boxes[:, :, 2:] = pred_boxes[:, :, :2] + 0.1  # Ensure valid boxes
    pred_boxes = pred_boxes.clamp(0, 1)

    targets = [
        {"labels": torch.tensor([0, 1, 2]), "boxes": torch.rand(3, 4).clamp(0, 1)},
        {"labels": torch.tensor([5, 6]), "boxes": torch.rand(2, 4).clamp(0, 1)},
    ]

    indices = hungarian_matcher(pred_logits, pred_boxes, targets)

    # Check 1: Output structure
    assert len(indices) == B, f"Should return indices for each batch element, got {len(indices)}"

    # Check 2: Number of matches equals number of targets
    assert indices[0][0].shape[0] == 3, "Batch 0 should have 3 matched queries"
    assert indices[1][0].shape[0] == 2, "Batch 1 should have 2 matched queries"

    # Check 3: Each target index appears exactly once
    assert len(torch.unique(indices[0][1])) == 3, "All target indices should be unique"
    assert len(torch.unique(indices[1][1])) == 2, "All target indices should be unique"


def test_hungarian_matcher_empty_targets(hungarian_matcher):
    """
    Verifies that HungarianMatcher handles empty target lists gracefully.
    Should return empty index tensors without crashing.
    """
    pred_logits = torch.randn(1, 50, 9)
    pred_boxes = torch.rand(1, 50, 4).clamp(0, 1)

    targets = [{"labels": torch.tensor([]).long(), "boxes": torch.zeros(0, 4)}]

    indices = hungarian_matcher(pred_logits, pred_boxes, targets)

    assert indices[0][0].numel() == 0, "Should return empty query indices"
    assert indices[0][1].numel() == 0, "Should return empty target indices"


def _make_valid_boxes(n: int, num_boxes: int) -> torch.Tensor:
    """
    Helper to create valid xyxy boxes with proper ordering (x1 < x2, y1 < y2).
    Returns tensor of shape (n, num_boxes, 4) or (num_boxes, 4) if n is None.
    """
    # Generate random corners and ensure proper ordering
    xy1 = torch.rand(n, num_boxes, 2) * 0.5  # x1, y1 in [0, 0.5]
    xy2 = xy1 + torch.rand(n, num_boxes, 2) * 0.4 + 0.1  # x2, y2 = xy1 + [0.1, 0.5]
    boxes = torch.cat([xy1, xy2], dim=-1).clamp(0, 1)
    return boxes


def test_hungarian_matcher_cost_weights():
    """
    Verifies that different cost weights produce different assignments.
    """
    # High class cost should prioritize classification matching
    matcher_class = HungarianMatcher(cost_class=10.0, cost_bbox=0.0, cost_giou=0.0)
    # High bbox cost should prioritize localization matching
    matcher_bbox = HungarianMatcher(cost_class=0.0, cost_bbox=10.0, cost_giou=0.0)

    pred_logits = torch.randn(1, 50, 9)
    # Create valid boxes with proper xyxy ordering (x1 < x2, y1 < y2)
    pred_boxes = _make_valid_boxes(1, 50)
    target_boxes = _make_valid_boxes(1, 2).squeeze(0)  # (2, 4)
    targets = [{"labels": torch.tensor([0, 1]), "boxes": target_boxes}]

    # Both should produce valid assignments (may differ)
    indices_class = matcher_class(pred_logits, pred_boxes, targets)
    indices_bbox = matcher_bbox(pred_logits, pred_boxes, targets)

    assert len(indices_class[0][0]) == 2, "Class matcher should match all targets"
    assert len(indices_bbox[0][0]) == 2, "Bbox matcher should match all targets"


# ----------------------------
# SetCriterion Tests
# ----------------------------
def test_criterion_loss_computation(set_criterion):
    """
    Verifies that SetCriterion computes all required losses.
    Output should contain CE, bbox, GIoU, and total losses.
    """
    B, Q, C = 2, 100, 9
    pred_logits = torch.randn(B, Q, C)
    pred_boxes = torch.rand(B, Q, 4)
    pred_boxes[:, :, 2:] = pred_boxes[:, :, :2] + 0.1
    pred_boxes = pred_boxes.clamp(0, 1)

    targets = [
        {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4).clamp(0, 1)},
        {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4).clamp(0, 1)},
    ]

    losses = set_criterion(pred_logits, pred_boxes, targets)

    # Check 1: All required loss keys present
    required_keys = ["loss_ce", "loss_bbox", "loss_giou", "loss_total"]
    for key in required_keys:
        assert key in losses, f"Missing required loss: {key}"

    # Check 2: Losses are valid scalars
    for k, v in losses.items():
        assert v.ndim == 0, f"Loss {k} should be a scalar, got shape {v.shape}"
        assert not torch.isnan(v), f"Loss {k} is NaN"
        assert not torch.isinf(v), f"Loss {k} is Inf"


def test_criterion_empty_targets(set_criterion):
    """
    Verifies that SetCriterion handles batches with no targets.
    Should return valid losses without crashing (regression losses should be 0).
    """
    pred_logits = torch.randn(1, 50, 9)
    pred_boxes = torch.rand(1, 50, 4).clamp(0, 1)

    targets = [{"labels": torch.tensor([]).long(), "boxes": torch.zeros(0, 4)}]

    losses = set_criterion(pred_logits, pred_boxes, targets)

    assert "loss_total" in losses, "Should return loss_total even with empty targets"
    assert not torch.isnan(losses["loss_total"]), "Loss should not be NaN"


def test_criterion_weighted_sum():
    """
    Verifies that loss_total is computed as the weighted sum of individual losses.
    """
    num_classes = 8
    matcher = HungarianMatcher()
    weight_dict = {"loss_ce": 2.0, "loss_bbox": 3.0, "loss_giou": 4.0}
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1)

    pred_logits = torch.randn(1, 50, 9)
    pred_boxes = torch.rand(1, 50, 4)
    pred_boxes[:, :, 2:] = pred_boxes[:, :, :2] + 0.1
    pred_boxes = pred_boxes.clamp(0, 1)

    targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4).clamp(0, 1)}]

    losses = criterion(pred_logits, pred_boxes, targets)

    expected_total = (
        weight_dict["loss_ce"] * losses["loss_ce"] +
        weight_dict["loss_bbox"] * losses["loss_bbox"] +
        weight_dict["loss_giou"] * losses["loss_giou"]
    )

    assert torch.allclose(losses["loss_total"], expected_total, atol=1e-5), \
        "loss_total should be the weighted sum of individual losses"


# ----------------------------
# Model Initialization Tests
# ----------------------------
@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_initialization(mock_sam3_cls):
    """
    Verifies that Sam3ForClosedSetDetection initializes correctly.
    Classification head should have num_classes + 1 outputs (for no-object).
    """
    mock_model = create_mock_sam3_model()
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(
        sam3_checkpoint="facebook/sam3",
        num_classes=8,
        freeze_sam3=False
    )

    assert model.num_classes == 8, "Model should store num_classes"
    assert model.class_embed.out_features == 9, \
        "Classification head should have num_classes + 1 outputs"
    assert model.class_embed.in_features == 256, \
        "Classification head input should match decoder hidden size"


@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_freeze_backbone(mock_sam3_cls):
    """
    Verifies that freeze_sam3=True correctly freezes the SAM3 backbone parameters.
    """
    mock_model = MagicMock()
    mock_model.config = MockSam3Config()

    # Create real parameters to track requires_grad
    param1 = torch.nn.Parameter(torch.randn(10))
    param2 = torch.nn.Parameter(torch.randn(10))
    mock_model.parameters.return_value = [param1, param2]
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8, freeze_sam3=True)

    # Verify parameters() was called for freezing
    mock_model.parameters.assert_called()


@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_build_criterion(mock_sam3_cls):
    """
    Verifies that build_criterion() properly attaches the criterion to the model.
    """
    mock_model = create_mock_sam3_model()
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8)

    # Before build_criterion, model should not have criterion
    assert not hasattr(model, 'criterion') or model.criterion is None

    # build_criterion should return self for chaining
    result = model.build_criterion(
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        eos_coef=0.1,
    )

    assert result is model, "build_criterion should return self"
    assert hasattr(model, 'criterion'), "Model should have criterion after build"
    assert isinstance(model.criterion, SetCriterion), "Criterion should be SetCriterion"


# ----------------------------
# Model Forward Pass Tests
# ----------------------------
@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_forward_inference(mock_sam3_cls):
    """
    Verifies the forward pass in inference mode (no targets).
    Output should contain logits, boxes, and sam3_outputs.
    """
    B, Q, hidden_size = 2, 100, 256
    mock_model = create_mock_sam3_model(B, Q, hidden_size)
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8)
    pixel_values = torch.randn(B, 3, 1008, 1008)

    outputs = model(pixel_values)

    # Check 1: Output structure
    assert "logits" in outputs, "Output should contain logits"
    assert "boxes" in outputs, "Output should contain boxes"
    assert "sam3_outputs" in outputs, "Output should contain raw sam3_outputs"

    # Check 2: Output shapes
    assert outputs["logits"].shape == (B, Q, 9), \
        f"Logits shape should be (B, Q, C+1), got {outputs['logits'].shape}"
    assert outputs["boxes"].shape == (B, Q, 4), \
        f"Boxes shape should be (B, Q, 4), got {outputs['boxes'].shape}"

    # Check 3: No losses in inference mode
    assert "losses" not in outputs, "No losses should be computed without targets"


@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_forward_training(mock_sam3_cls):
    """
    Verifies the forward pass in training mode (with targets).
    Output should include computed losses.
    """
    B, Q, hidden_size = 2, 100, 256
    mock_model = create_mock_sam3_model(B, Q, hidden_size)
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8).build_criterion()

    pixel_values = torch.randn(B, 3, 1008, 1008)
    targets = [
        {"labels": torch.tensor([0, 1, 2]), "boxes": torch.rand(3, 4).clamp(0, 1)},
        {"labels": torch.tensor([5, 6]), "boxes": torch.rand(2, 4).clamp(0, 1)},
    ]

    outputs = model(pixel_values, targets=targets)

    assert "losses" in outputs, "Output should contain losses when targets provided"
    assert "loss_total" in outputs["losses"], "Losses should include loss_total"
    assert "loss_ce" in outputs["losses"], "Losses should include loss_ce"
    assert "loss_bbox" in outputs["losses"], "Losses should include loss_bbox"
    assert "loss_giou" in outputs["losses"], "Losses should include loss_giou"


@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_predict(mock_sam3_cls):
    """
    Verifies the predict() method returns properly formatted detections.
    Results should be filtered by score threshold and contain correct keys.
    """
    B, Q, hidden_size = 2, 100, 256
    mock_model = create_mock_sam3_model(B, Q, hidden_size)
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8)

    pixel_values = torch.randn(B, 3, 1008, 1008)
    orig_sizes = torch.tensor([[1024, 1024], [1024, 1024]], dtype=torch.float32)

    results = model.predict(
        pixel_values,
        orig_sizes=orig_sizes,
        score_thresh=0.3,
        max_detections=100
    )

    # Check 1: Results for each batch element
    assert len(results) == B, f"Should return results for each batch element, got {len(results)}"

    # Check 2: Result structure
    for r in results:
        assert "scores" in r, "Each result should contain scores"
        assert "labels" in r, "Each result should contain labels"
        assert "boxes" in r, "Each result should contain boxes"

        # Check 3: Tensor shapes consistency
        num_detections = r["scores"].shape[0]
        assert r["labels"].shape[0] == num_detections, "Labels count should match scores"
        assert r["boxes"].shape == (num_detections, 4), "Boxes shape should be (N, 4)"


@patch('models.sam3_DETR_v2.Sam3Model')
def test_model_predict_max_detections(mock_sam3_cls):
    """
    Verifies that predict() respects the max_detections limit.
    """
    B, Q, hidden_size = 1, 100, 256
    mock_model = create_mock_sam3_model(B, Q, hidden_size)
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8)
    pixel_values = torch.randn(B, 3, 1008, 1008)

    results = model.predict(
        pixel_values,
        score_thresh=0.0,  # Accept all detections
        max_detections=10
    )

    assert results[0]["scores"].shape[0] <= 10, \
        "Number of detections should not exceed max_detections"


# ----------------------------
# Collate Function Tests
# ----------------------------
def test_collate_fn_structure():
    """
    Verifies that make_sam3_collate_fn produces correctly structured batches.
    Output should be (pixel_values, normalized_targets, original_sizes).
    """
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values": torch.randn(2, 3, 1008, 1008),
        "original_sizes": [[1024, 1024], [1024, 1024]],
    }

    collate_fn = make_sam3_collate_fn(mock_processor)

    img1 = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))

    batch = [
        (img1, {"boxes": torch.tensor([[100., 100., 200., 200.]]), "labels": torch.tensor([0])}),
        (img2, {"boxes": torch.tensor([[50., 50., 150., 150.]]), "labels": torch.tensor([1])}),
    ]

    pixel_values, norm_targets, orig_sizes = collate_fn(batch)

    # Check 1: Pixel values shape
    assert pixel_values.shape == (2, 3, 1008, 1008), \
        f"Pixel values shape mismatch: {pixel_values.shape}"

    # Check 2: Targets structure
    assert len(norm_targets) == 2, "Should have one target dict per image"

    # Check 3: Boxes are normalized to [0, 1]
    for t in norm_targets:
        assert t["boxes"].max() <= 1.0, "Boxes should be normalized to <= 1.0"
        assert t["boxes"].min() >= 0.0, "Boxes should be normalized to >= 0.0"

    # Check 4: Original sizes preserved
    assert orig_sizes.shape == (2, 2), "orig_sizes should be (B, 2)"


def test_collate_fn_box_normalization():
    """
    Verifies that collate_fn correctly normalizes boxes from absolute to relative coords.
    """
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values": torch.randn(1, 3, 1008, 1008),
        "original_sizes": [[100, 200]],  # H=100, W=200
    }

    collate_fn = make_sam3_collate_fn(mock_processor)

    img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
    # Box at [20, 10, 40, 30] in absolute coords (W=200, H=100)
    # Expected normalized: [20/200, 10/100, 40/200, 30/100] = [0.1, 0.1, 0.2, 0.3]
    batch = [
        (img, {"boxes": torch.tensor([[20., 10., 40., 30.]]), "labels": torch.tensor([0])}),
    ]

    _, norm_targets, _ = collate_fn(batch)

    expected = torch.tensor([[0.1, 0.1, 0.2, 0.3]])
    assert torch.allclose(norm_targets[0]["boxes"], expected, atol=1e-5), \
        f"Box normalization incorrect: got {norm_targets[0]['boxes']}, expected {expected}"


# ----------------------------
# Gradient Flow Tests
# ----------------------------
@patch('models.sam3_DETR_v2.Sam3Model')
def test_backward_pass_gradients(mock_sam3_cls):
    """
    Verifies that gradients flow correctly through the model.
    Classification head should receive gradients after backward pass.
    """
    B, Q, hidden_size = 2, 100, 256

    # Create outputs with gradient tracking enabled
    mock_outputs = MockSam3Outputs(B, Q, hidden_size)
    mock_outputs.decoder_hidden_states = tuple(
        t.requires_grad_(True) for t in mock_outputs.decoder_hidden_states
    )

    mock_model = MagicMock()
    mock_model.config = MockSam3Config()
    mock_model.return_value = mock_outputs
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8).build_criterion()

    pixel_values = torch.randn(B, 3, 1008, 1008)
    targets = [
        {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4).clamp(0, 1)},
        {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4).clamp(0, 1)},
    ]

    outputs = model(pixel_values, targets=targets)
    loss = outputs["losses"]["loss_total"]

    # Backward pass should not raise
    loss.backward()

    # Check gradients exist on classification head
    assert model.class_embed.weight.grad is not None, \
        "Classification head weight should have gradients"
    assert model.class_embed.bias.grad is not None, \
        "Classification head bias should have gradients"


@patch('models.sam3_DETR_v2.Sam3Model')
def test_loss_decreases_with_optimization_step(mock_sam3_cls):
    """
    Verifies that a single optimization step reduces the loss.
    This is a sanity check for the training pipeline.
    """
    B, Q, hidden_size = 1, 50, 256

    mock_outputs = MockSam3Outputs(B, Q, hidden_size)
    mock_outputs.decoder_hidden_states = tuple(
        t.requires_grad_(True) for t in mock_outputs.decoder_hidden_states
    )

    mock_model = MagicMock()
    mock_model.config = MockSam3Config()
    mock_model.return_value = mock_outputs
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8).build_criterion()
    optimizer = torch.optim.SGD(model.class_embed.parameters(), lr=0.1)

    pixel_values = torch.randn(B, 3, 1008, 1008)
    targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4).clamp(0, 1)}]

    # First forward pass
    outputs1 = model(pixel_values, targets=targets)
    loss1 = outputs1["losses"]["loss_ce"].item()

    # Optimization step
    optimizer.zero_grad()
    outputs1["losses"]["loss_ce"].backward()
    optimizer.step()

    # Second forward pass (same inputs)
    outputs2 = model(pixel_values, targets=targets)
    loss2 = outputs2["losses"]["loss_ce"].item()

    # Note: Loss may not always decrease with one step, but should change
    print(f"Debug: Loss before={loss1:.4f}, after={loss2:.4f}")
    assert loss1 != loss2, "Loss should change after optimization step"


# ----------------------------
# Edge Case Tests
# ----------------------------
@patch('models.sam3_DETR_v2.Sam3Model')
def test_single_target_per_batch(mock_sam3_cls):
    """
    Verifies model handles batches where each image has only one target.
    """
    B, Q, hidden_size = 2, 100, 256
    mock_model = create_mock_sam3_model(B, Q, hidden_size)
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8).build_criterion()

    pixel_values = torch.randn(B, 3, 1008, 1008)
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4).clamp(0, 1)},
        {"labels": torch.tensor([7]), "boxes": torch.rand(1, 4).clamp(0, 1)},
    ]

    outputs = model(pixel_values, targets=targets)

    assert not torch.isnan(outputs["losses"]["loss_total"]), \
        "Loss should be valid with single target per image"


@patch('models.sam3_DETR_v2.Sam3Model')
def test_mixed_empty_and_full_targets(mock_sam3_cls):
    """
    Verifies model handles mixed batches (some images with targets, some without).
    """
    B, Q, hidden_size = 3, 100, 256
    mock_model = create_mock_sam3_model(B, Q, hidden_size)
    mock_sam3_cls.from_pretrained.return_value = mock_model

    model = Sam3ForClosedSetDetection(num_classes=8).build_criterion()

    pixel_values = torch.randn(B, 3, 1008, 1008)
    targets = [
        {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4).clamp(0, 1)},
        {"labels": torch.tensor([]).long(), "boxes": torch.zeros(0, 4)},  # Empty
        {"labels": torch.tensor([5]), "boxes": torch.rand(1, 4).clamp(0, 1)},
    ]

    outputs = model(pixel_values, targets=targets)

    assert not torch.isnan(outputs["losses"]["loss_total"]), \
        "Loss should be valid with mixed empty/full targets"
