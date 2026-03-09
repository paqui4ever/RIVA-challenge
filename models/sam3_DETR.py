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

import math

class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal position embedding, adapted from DETR.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        """
        Args:
            mask: (B, H, W) boolean mask where True = masked (padding), False = valid
        Returns:
            pos: (B, hidden_dim, H, W) positional embeddings
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DetrBackboneAdapter(nn.Module):
    def __init__(self, backbone, hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        # Mirror key attributes required by DETR
        self.out_channels = backbone.out_channels
        self.num_channels = backbone.out_channels # Some implementations check this name

        # Position embedding for DETR (sinusoidal)
        self.hidden_dim = hidden_dim
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
    def forward(self, pixel_values, pixel_mask=None):
            
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

        # Generate position embeddings
        # The position embedding expects a boolean mask where True = masked (padding)
        pos_embed = self.position_embedding(pixel_mask)

        # 5. Return in the format HF DETR expects:
        # - features: list of (feature_map, mask) tuples - one per scale level
        # - position_embeddings: list of position embeddings - one per scale level
        # HF DETR unpacks as: feature_map, mask = features[-1]
        # and: object_queries = object_queries_list[-1]
        return [(features, pixel_mask)], [pos_embed]


def get_sam3_detr(num_classes, sam_checkpoint="facebook/sam3"):
    """ Returns DETR model with SAM3 backbone """

    # Loading the DETR config 
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes

    # Setting up the backbone
    pure_backbone = SAM3Backbone(checkpoint=sam_checkpoint, out_channels=256)

    # Adapting the backbone
    detr_compatible_backbone = DetrBackboneAdapter(pure_backbone)
    
    # Assembling
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
        )

    # Replace the feature extractor
    model.model.backbone = detr_compatible_backbone

    # Replacing projection layer
    print(f"Replacing Input Projection: {detr_compatible_backbone.out_channels} -> {model.config.d_model}")
    model.model.input_projection = nn.Conv2d(
        in_channels=detr_compatible_backbone.out_channels,
        out_channels=model.config.d_model,                 
        kernel_size=1
    )
    
    return model    

