import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers import DetrConfig, DetrForObjectDetection
import torch.nn.functional as F

from models.sam3_rcnn import SAM3Backbone

class DetrBackboneAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # 1. Mirror key attributes required by DETR
        self.out_channels = backbone.out_channels
        self.num_channels = backbone.out_channels # Some implementations check this name

        # # --- ROBUST CHANNEL DETECTION ---
        # # Instead of trusting backbone.out_channels (which seems to be reporting 2048 erroneously),
        # # we run a cheap dummy pass to see what the tensor actually looks like.
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, 3, 1008, 1008)
        #     # We assume the backbone might live on CPU for now; if on GPU, move dummy.
        #     if next(backbone.parameters()).is_cuda:
        #         dummy_input = dummy_input.cuda()
                
        #     dummy_out = backbone(dummy_input)
        #     if isinstance(dummy_out, dict):
        #         dummy_feat = list(dummy_out.values())[-1]
        #     else:
        #         dummy_feat = dummy_out
                
        #     detected_channels = dummy_feat.shape[1]
            
        # self.out_channels = detected_channels
        # self.num_channels = detected_channels 
        # print(f"DetrBackboneAdapter: Detected actual output channels: {self.out_channels}")

        
    def forward(self, pixel_values, pixel_mask=None):
        # 1. Get raw features from the pure backbone
        # We resize to 1008x1008 to ensure stability, as 1024 causes rotary embedding mismatches.
        if pixel_values.shape[-2:] != (1008, 1008):
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, 
                size=(1008, 1008), 
                mode='bilinear', 
                align_corners=False
            )
            
        features_dict = self.backbone(pixel_values)
        
        if isinstance(features_dict, dict):
            # usually {'0': features}
            features = list(features_dict.values())[-1] 
        else:
            features = features_dict
        
        # DETR requires a mask for every feature map.
        if pixel_mask is None:
            # Create default "all valid" mask (0 = keep)
            pixel_mask = torch.zeros(
                (features.shape[0], features.shape[2], features.shape[3]), 
                dtype=torch.bool, 
                device=features.device
            )
        else:
            # Resize provided mask to match feature map resolution
            pixel_mask = F.interpolate(
                pixel_mask[None].float(), 
                size=features.shape[-2:], 
                mode="nearest"
            ).to(torch.bool)[0]

        # 4. Return Tuple (Features, Masks)
        # DETR expects lists because it supports multi-scale, but we use single scale here.
        return [features], [pixel_mask]


def get_sam3_detr(num_classes, sam_checkpoint="facebook/sam3"):
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes

    pure_backbone = SAM3Backbone(checkpoint=sam_checkpoint, out_channels=256)

    detr_compatible_backbone = DetrBackboneAdapter(pure_backbone)
    
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
        )

    # Replace the feature extractor
    model.model.backbone = detr_compatible_backbone

    # 4. PROJECTION LAYER REPLACEMENT 
    print(f"Replacing Input Projection: {detr_compatible_backbone.out_channels} -> {model.config.d_model}")
    model.model.input_proj = nn.Conv2d(
        in_channels=detr_compatible_backbone.out_channels, 
        out_channels=model.config.d_model,                 
        kernel_size=1
    )
    
    return model    

