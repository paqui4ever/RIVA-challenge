
import torch
import sys
import os
import pytest

# Add local directory to path to import models
sys.path.append(os.getcwd())

def test_sam3_lora():
    print("Testing SAM3-DETR-v2-LoRA...")
    try:
        from models.sam3_DETR_v2_LoRA import Sam3DETRv2LoRA
        model = Sam3DETRv2LoRA(sam3_checkpoint="facebook/sam3", lora_rank=4)
        print("SAM3-DETR-v2-LoRA instantiated successfully.")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"Failed to instantiate SAM3-DETR-v2-LoRA: {e}")

def test_cell_dino_lora():
    print("\nTesting Cell-DINO-RCNN-v2-LoRA...")
    try:
        from models.cell_DINO_rcnn_v2_LoRA import build_cell_dino_fasterrcnn_lora
        # Use a dummy model name or existing one. 'dinov2_vitl14' is safe if 'cell_dino_hpa_vitl14' fails download
        # Logic in backbone handles fallback.
        model = build_cell_dino_fasterrcnn_lora(lora_rank=4)
        print("Cell-DINO-RCNN-v2-LoRA instantiated successfully.")
        
        # Count trainable params in backbone.vision (LoRA) vs others
        vision_params = sum(p.numel() for p in model.backbone.vision.parameters() if p.requires_grad)
        other_params = sum(p.numel() for p in model.parameters() if p.requires_grad) - vision_params
        print(f"Vision LoRA Trainable Params: {vision_params}")
        print(f"Other Trainable Params (Heads/FPN): {other_params}")
        
    except Exception as e:
        print(f"Failed to instantiate Cell-DINO-RCNN-v2-LoRA: {e}")
