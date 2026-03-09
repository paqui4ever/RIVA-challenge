import pytest 
import sys
import torch
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from models.sam3_rcnn import get_sam3_faster_rcnn, SAM3Backbone

@pytest.fixture(scope="module")
def sam3_rcnn_model():
    num_classes = 10 # 9 classes + background
    try:
        model = get_sam3_faster_rcnn(num_classes=num_classes)
        model.eval()
        return model
    except Exception as e:
        pytest.fail(f"Failed to initialize model: {e}")

def test_backbone_structure(sam3_rcnn_model):
    # Check Backbone
    # Use 1008 to match native resolution (avoiding transform which handles resizing)
    dummy_input = torch.randn(1, 3, 1008, 1008)
    with torch.no_grad():
        features = sam3_rcnn_model.backbone(dummy_input)
    
    assert isinstance(features, dict) and '0' in features
    feat_shape = features['0'].shape
    assert feat_shape[1] == 1024 and feat_shape[2] == 72

def test_anchor_generator_structure(sam3_rcnn_model):
    # Check Anchor Generator
    # We expect 100x100 sizes
    # Accessing internal rpn parameters
    rpn_sizes = sam3_rcnn_model.rpn.anchor_generator.sizes
    rpn_ratios = sam3_rcnn_model.rpn.anchor_generator.aspect_ratios

    assert rpn_sizes == ((100,),) and rpn_ratios == ((1.0,),)

def test_full_forward_encoder_pass(sam3_rcnn_model):
    # Check Full Forward Pass
    
    with torch.no_grad():
        # Input list of tensors
        images = [torch.randn(3, 1024, 1024), torch.randn(3, 1024, 1024)]
        # Original image sizes helper
        original_image_sizes = [(1024, 1024), (1024, 1024)]
        
        # FasterRCNN expects (images, targets) for train, or (images) for eval
        predictions = sam3_rcnn_model(images)
        
        for i, pred in enumerate(predictions):
            keys = pred.keys()
            assert 'boxes' in keys and 'labels' in keys and 'scores' in keys