import pytest
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

# Import your classes 
from models.sam3_DETR import get_sam3_detr, DetrBackboneAdapter
from models.sam3_rcnn import SAM3Backbone

@pytest.fixture(scope="module")
def sam3_detr_model():
    num_classes = 9 # 8 classes + background
    try:
        model = get_sam3_detr(num_classes=num_classes)
        model.eval()
        return model
    except Exception as e:
        pytest.fail(f"Failed to initialize model: {e}")

def test_backbone_contract(sam3_detr_model):
    """
    Verifies the backbone returns the specific tuple structure DETR expects:
    ([features], [masks])
    """
    dummy_input = torch.randn(1, 3, 1008, 1008)
    
    with torch.no_grad():
        # Access the wrapper directly
        outputs = sam3_detr_model.model.backbone(dummy_input)
    
    # 1. Structure Check: Must be a list of length 1 (Conv Encoder only returns features)
    assert isinstance(outputs, (tuple, list))
    assert len(outputs) == 2, "Backbone must return ([features], [masks])"
    
    features_list, masks_list = outputs
    
    # 2. Content Check
    assert len(features_list) == 1
    assert len(masks_list) == 1
    
    features = features_list[0]
    masks = masks_list[0]
    
    # 3. Shape Check
    f_shape = features.shape
    print(f"Debug: Feature shape: {f_shape}")
    
    # Check against our adapter's declared channels
    assert f_shape[1] == sam3_detr_model.model.backbone.out_channels
    assert f_shape[0] == dummy_input.shape[0] 
    
    # 4. Mask Check (Now this works because our Adapter handles it!)
    assert masks.shape[-2:] == features.shape[-2:]

def test_projection_integrity(sam3_detr_model):
    """
    Verifies that the 'input_proj' layer connects the backbone to the transformer correctly.
    """
    backbone_channels = sam3_detr_model.model.backbone.out_channels
    projector = sam3_detr_model.model.input_proj
    
    # Check 1: Input channels of conv1x1 must match backbone output
    assert projector.in_channels == backbone_channels, \
        f"Mismatch! Projector expects {projector.in_channels}, backbone gives {backbone_channels}"
    
    # Check 2: Output channels must match DETR hidden dimension (usually 256)
    expected_dim = sam3_detr_model.config.d_model
    assert projector.out_channels == expected_dim

def test_full_forward_pass_with_masks(sam3_detr_model):
    """
    Tests the full 'DetrForObjectDetection' pipeline.
    This verifies that the features + masks flow through the Transformer and Heads.
    """
    # Create a batch of 2 images
    B = 2
    dummy_input = torch.randn(B, 3, 1008, 1008)
    
    # Create a dummy mask (optional in forward, but good to test if explicit)
    # DETR Processor usually handles this, but we test the tensor flow here.
    # 0 = Keep, 1 = Masked (Padding)
    dummy_mask = torch.zeros((B, 1008, 1008), dtype=torch.float32) 
    
    with torch.no_grad():
        # HF DETR expects 'pixel_values' and optionally 'pixel_mask'
        outputs = sam3_detr_model(pixel_values=dummy_input, pixel_mask=dummy_mask)
    
    # Check Logic
    # DETR outputs 'logits' and 'pred_boxes' directly in the object
    assert hasattr(outputs, 'logits'), "Output missing logits"
    assert hasattr(outputs, 'pred_boxes'), "Output missing pred_boxes"
    
    # Check Shapes
    # (Batch, Num_Queries, Num_Classes + 1) -> +1 is "no object" class
    # Standard DETR has 100 queries
    num_queries = sam3_detr_model.config.num_queries 
    num_classes = sam3_detr_model.config.num_labels # HF config uses this name
    
    assert outputs.logits.shape == (B, num_queries, num_classes + 1)
    assert outputs.pred_boxes.shape == (B, num_queries, 4)

def test_variable_resolution_resilience(sam3_detr_model):
    """
    SAM works best on square inputs, but we should ensure the reshaping logic
    doesn't crash if we pass 1024x1024 in a batch that might have padding.
    """
    # Simulate a batch where image is 1024x1024
    dummy_input = torch.randn(1, 3, 1024, 1024)
    
    try:
        with torch.no_grad():
            _ = sam3_detr_model(dummy_input)
    except RuntimeError as e:
        pytest.fail(f"Model crashed on standard resolution forward pass: {e}")