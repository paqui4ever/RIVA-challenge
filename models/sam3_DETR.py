import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers import DetrForObjectDetection

from models.sam3_backbone import SAM3Backbone

class DetrBackboneAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # 1. Mirror key attributes required by DETR
        self.out_channels = backbone.out_channels
        self.num_channels = backbone.out_channels # Some implementations check this name
        
    def forward(self, pixel_values, pixel_mask=None):
        # 1. Get raw features from the pure backbone
        features_dict = self.backbone(pixel_values)
        
        # Handle dict vs tensor output
        if isinstance(features_dict, dict):
            # usually {'0': features}
            features = list(features_dict.values())[-1] 
        else:
            features = features_dict

        # 2. Mask Logic (The DETR requirement)
        if pixel_mask is None:
            pixel_mask = torch.zeros(
                (features.shape[0], features.shape[2], features.shape[3]), 
                dtype=torch.bool, 
                device=features.device
            )
        else:
            pixel_mask = torch.nn.functional.interpolate(
                pixel_mask[None].float(), 
                size=features.shape[-2:], 
                mode="nearest"
            ).to(torch.bool)[0]

        # 3. Return the tuple list DETR expects
        return [features], [pixel_mask]


def get_sam3_detr(num_classes, sam_checkpoint="facebook/sam3"):

    pure_backbone = SAM3Backbone(checkpoint=sam_checkpoint, out_channels=1024)

    detr_compatible_backbone = DetrBackboneAdapter()
    
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Replace the feature extractor
    model.model.backbone.conv_encoder = detr_compatible_backbone

    # We must match SAM's output channels (1024) to DETR's Transformer dimension (256)
    # If we don't do this, the shapes will mismatch in the very next line of the forward pass.
    model.model.input_proj = nn.Conv2d(
        in_channels=detr_compatible_backbone.out_channels, # 1024 from SAM3
        out_channels=config.d_model,           # 256 from standard DETR
        kernel_size=1
    )
    
    return model    

