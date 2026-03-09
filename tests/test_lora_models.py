
import torch
import sys
import os
import pytest

# Add local directory to path to import models
sys.path.append(os.getcwd())

def test_sam3_lora():
    """Test SAM3-DETR-v2-LoRA instantiation and trainable parameters."""
    print("Testing SAM3-DETR-v2-LoRA...")
    
    from models.sam3_DETR_v2_LoRA import Sam3DETRv2LoRA
    
    # Instantiate or mock if necessary. Assuming we can instantiate.
    model = Sam3DETRv2LoRA(sam3_checkpoint="facebook/sam3", lora_rank=4)
    
    assert model is not None, "Failed to instantiate Sam3DETRv2LoRA"
    
    # Check if there are trainable parameters (LoRA should add some)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "No trainable parameters found in LoRA model"
    
    # Optional: Check if specific LoRA layers are present in name
    has_lora_layers = any("lora" in n for n, _ in model.named_parameters())
    assert has_lora_layers, "No parameters with 'lora' in name found"
    
    print(f"SAM3-DETR-v2-LoRA instantiated. Trainable params: {trainable_params}")

def test_cell_dino_lora():
    """Test Cell-DINO-RCNN-v2-LoRA instantiation and parameter counts."""
    print("\nTesting Cell-DINO-RCNN-v2-LoRA...")
    
    from models.cell_DINO_rcnn_v2_LoRA import build_cell_dino_fasterrcnn_lora
    
    model = build_cell_dino_fasterrcnn_lora(lora_rank=4)
    
    assert model is not None, "Failed to instantiate Cell-DINO-RCNN-v2-LoRA"
    
    # Count trainable params in backbone.vision (LoRA) vs others
    vision_params = sum(p.numel() for p in model.backbone.vision.parameters() if p.requires_grad)
    other_params = sum(p.numel() for p in model.parameters() if p.requires_grad) - vision_params
    
    print(f"Vision LoRA Trainable Params: {vision_params}")
    print(f"Other Trainable Params (Heads/FPN): {other_params}")
    
    # Assertions
    assert vision_params > 0, "Backbone vision should have trainable LoRA parameters"
    assert other_params > 0, "Heads/FPN should have trainable parameters"
    
    # Verify backbone is in eval mode except for LoRA
    # This is a bit harder to test strictly without iterating everything, 
    # but we can check if at least one parameter in vision is frozen
    frozen_params = sum(p.numel() for p in model.backbone.vision.parameters() if not p.requires_grad)
    assert frozen_params > 0, "Backbone should have some frozen parameters (pre-trained weights)"

if __name__ == "__main__":
    # Allow running directly
    sys.exit(pytest.main([__file__]))
