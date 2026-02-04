
import torch
import torch.nn as nn
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Warning: 'peft' library not found. LoRA features will not work.")
    LoraConfig = None
    get_peft_model = None

from torchvision.models.detection import FasterRCNN
from .cell_DINO_rcnn_v2 import CellDinoBackbone, build_cell_dino_fasterrcnn

def build_cell_dino_fasterrcnn_lora(
    model_name: str = "cell_dino_hpa_vitl14",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_classes_closed_set: int = 8,
    trainable_backbone: bool = False,
) -> FasterRCNN:
    """
    Builds a FasterRCNN model with a Cell-DINO backbone augmented with LoRA.
    """
    if LoraConfig is None:
        raise ImportError("Please install 'peft' to use LoRA: pip install peft")

    # 1. Build standard model (base backbone frozen by default if trainable_backbone=False)
    # We rely on PEFT to handle freezing/unfreezing logic for the backbone.
    model = build_cell_dino_fasterrcnn(
        model_name=model_name,
        num_classes_closed_set=num_classes_closed_set,
        trainable_backbone=False # We start with frozen base, then inject LoRA
    )
    
    # 2. Identify the part to apply LoRA: The DINOv2 Vision Transformer
    target_object = model.backbone.vision
    
    # 3. Configure LoRA
    # DINOv2 uses 'qkv' linear layers in its attention blocks.
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["qkv"], 
        bias="none",
    )
    
    # 4. Apply LoRA
    model.backbone.vision = get_peft_model(target_object, peft_config)
    
    # 5. Ensure FPN layers and Heads are ID-wise trainable
    # The 'build_cell_dino_fasterrcnn' sets model.backbone parameters to not require grad if trainable=False
    # But FPN layers (fpn_conv, smooth_*) are initialized in the wrapper, so they are trainable by default
    # unless explicitly frozen. Let's verify and ensure they are trainable.
    
    model.backbone.train() # Set to train mode
    
    # Ensure backbone adapter layers (non-DINO) are trainable
    for name, param in model.backbone.named_parameters():
        if "vision" not in name:
            param.requires_grad = True
            
    # Ensure RPN and ROI heads are trainable
    for param in model.rpn.parameters():
        param.requires_grad = True
    for param in model.roi_heads.parameters():
        param.requires_grad = True
        
    return model
