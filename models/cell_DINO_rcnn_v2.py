from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# -------------------------
# Preprocess: Square resize + pad (aligned with DINOv2 patch size)
# -------------------------

# Read the .env for the weights url
load_dotenv()

# Get the weights url from the .env
CELL_DINO_WEIGHTS_URL = os.environ.get('CELL_DINO_WEIGHTS_URL')

@dataclass
class CellDinoResizePadMeta:
    scale: float
    resized_hw: Tuple[int, int]     # (h, w) after resize, before pad
    pad_rb: Tuple[int, int]         # (pad_right, pad_bottom)

def cell_dino_resize_longest_side_and_pad_square(
    img: torch.Tensor,
    target: Optional[Dict[str, torch.Tensor]] = None,
    target_size: int = 1008,  # Must be divisible by 14 (and 56 for FPN)
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], CellDinoResizePadMeta]:
    """
    img: Float tensor [C,H,W] in [0,1]
    target: dict with 'boxes' (FloatTensor [N,4] in xyxy, absolute pixels) and 'labels'
    Returns: resized+padded img [C,target_size,target_size], updated target, and meta.
    """
    assert img.ndim == 3 and img.shape[0] in (1, 3), f"Expected [C,H,W], got {img.shape}"
    c, h, w = img.shape

    # Resize so that longest side == target_size (keep aspect ratio)
    scale = target_size / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    img = TVF.resize(img, [new_h, new_w], antialias=True)

    # Pad (right, bottom) to reach square target_size
    pad_right = target_size - new_w
    pad_bottom = target_size - new_h
    if pad_right < 0 or pad_bottom < 0:
        # Should guard against rounding weirdness
        img = img[:, : min(new_h, target_size), : min(new_w, target_size)]
        new_h, new_w = img.shape[-2:]
        pad_right = target_size - new_w
        pad_bottom = target_size - new_h

    img = TVF.pad(img, [0, 0, pad_right, pad_bottom], fill=0)

    if target is not None and "boxes" in target:
        tgt = dict(target)
        boxes = tgt["boxes"].clone()
        boxes = boxes * scale

        # Clip boxes to resized (pre-pad) region
        boxes[:, 0::2].clamp_(0, new_w - 1)
        boxes[:, 1::2].clamp_(0, new_h - 1)

        tgt["boxes"] = boxes
        target = tgt

    meta = CellDinoResizePadMeta(scale=scale, resized_hw=(new_h, new_w), pad_rb=(pad_right, pad_bottom))
    return img, target, meta


# -------------------------
# Backbone: Cell-DINO (DINOv2) -> FPN Strides [7, 14, 28, 56]
# -------------------------
class CellDinoBackbone(nn.Module):
    """
    Wraps DINOv2 (ViT-L/14) to behave like a detection backbone.
    Expects input: images tensor [B,3,H,W] padded to multiples of 14 (e.g. 1008).
    Returns: OrderedDict of multi-scale feature maps.
    
    Strides:
    - '0' (Stride 7): 2x Upsample from base
    - '1' (Stride 14): Identity (Base ViT output)
    - '2' (Stride 28): 2x Pool from base
    - '3' (Stride 56): 4x Pool from base
    """
    def __init__(
        self, 
        model_name: str = "cell_dino_hpa_vitl14", 
        pretrained_checkpoint_path: Optional[str] = None,
        trainable: bool = False
    ):
        super().__init__()
        
        # Load backbone from torch hub
        # Fallback to 'dinov2_vitl14' if cell_dino specific name fails or isn't in default hub.
        
        # Note: We rely on 'facebookresearch/dinov2' hub. 
        # 'cell_dino_hpa_vitl14' is the specific model that we are looking for.
        # If not, we fall back to 'dinov2_vitl14' structure-wise.
        
        try:
            self.vision = torch.hub.load("facebookresearch/dinov2", model_name, source='local', pretrained_url=CELL_DINO_WEIGHTS_URL)
        except Exception as e:
            print(f"Warning: Could not load {model_name} from hub, falling back to 'dinov2_vitl14'. Error: {e}")
            self.vision = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)

        # if pretrained_checkpoint_path:
        #     print(f"Loading Cell-DINO weights from {pretrained_checkpoint_path}...")
        #     state_dict = torch.load(pretrained_checkpoint_path, map_location="cpu")
        #     # Handle potential checkpoint key differences (e.g. 'teacher' prefix)
        #     if "teacher" in state_dict:
        #         state_dict = state_dict["teacher"]
        #     # Remove potential prefixes if they don't match
        #     clean_state_dict = {}
        #     for k, v in state_dict.items():
        #         k = k.replace("backbone.", "") # generic cleanup
        #         clean_state_dict[k] = v
            
        #     missing, unexpected = self.vision.load_state_dict(clean_state_dict, strict=False)
        #     print(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        if not trainable:
            for p in self.vision.parameters():
                p.requires_grad = False

        # ViT-L dim
        self.embed_dim = self.vision.embed_dim if hasattr(self.vision, "embed_dim") else 1024
        self.patch_size = self.vision.patch_size if hasattr(self.vision, "patch_size") else 14
        self.out_channels = 256
        
        # Adapter layers to create FPN pyramid from single scale
        self.fpn_conv = nn.Conv2d(self.embed_dim, self.out_channels, 1)
        
        # Optional: Smooth layers for upsampled/downsampled features
        self.smooth_7 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.smooth_14 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.smooth_28 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.smooth_56 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)

    @property
    def image_mean(self) -> List[float]:
        return [0.485, 0.456, 0.406] # Standard ImageNet

    @property
    def image_std(self) -> List[float]:
        return [0.229, 0.224, 0.225] # Standard ImageNet

    @property
    def target_size(self) -> int:
        return 1008  # Multiple of 14 and 56

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        B, C, H, W = x.shape
        
        # DINOv2 forward_features returns dict with keys like 'x_norm_clstoken', 'x_norm_patchtokens'
        # We need to manually call the components if forward_features isn't exposed or differs
        # Standard DINOv2 hub model has forward_features
        
        outputs = self.vision.forward_features(x)
        patch_tokens = outputs["x_norm_patchtokens"] # [B, N_patches, Embed_Dim]
        
        # Reshape to grid
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        
        # [B, H*W, D] -> [B, D, H, W]
        assert patch_tokens.shape[1] == h_grid * w_grid, f"Shape mismatch: {patch_tokens.shape} vs {h_grid}x{w_grid}"
        feat = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h_grid, w_grid)
        
        # Base feature (Stride 14)
        m14 = self.fpn_conv(feat)
        
        # Generate pyramid
        # Stride 7: Upsample
        m7 = F.interpolate(m14, scale_factor=2.0, mode="nearest")
        m7 = self.smooth_7(m7)
        
        # Stride 14: Identity + Smooth
        m14_out = self.smooth_14(m14)
        
        # Stride 28: Pool
        m28 = F.max_pool2d(m14, kernel_size=2, stride=2)
        m28 = self.smooth_28(m28)
        
        # Stride 56: Pool
        m56 = F.max_pool2d(m14, kernel_size=4, stride=4)
        m56 = self.smooth_56(m56)
        
        results = OrderedDict()
        results["0"] = m7
        results["1"] = m14_out
        results["2"] = m28
        results["3"] = m56
        
        return results

# -------------------------
# Build Faster R-CNN
# -------------------------
def build_cell_dino_fasterrcnn(
    model_name: str = "cell_dino_hpa_vitl14",
    pretrained_checkpoint_path: Optional[str] = None,
    num_classes_closed_set: int = 8,
    trainable_backbone: bool = False,
) -> FasterRCNN:
    
    backbone = CellDinoBackbone(
        model_name=model_name, 
        # pretrained_checkpoint_path=pretrained_checkpoint_path,
        trainable=trainable_backbone
    )
    
    # Strides: [7, 14, 28, 56]
    # We need anchors that cover suitable pixel ranges for these strides.
    # Standard FPN strides [4, 8, 16, 32] usually have sizes (32, 64, 128, 256, 512)
    # Here we are shifted/scaled roughly.
    
    # Stride 7 (Map "0"): Small objects. ~14-28px
    # Stride 14 (Map "1"): Medium. ~56px
    # Stride 28 (Map "2"): Large. ~112px
    # Stride 56 (Map "3"): Extra Large. ~224px+
    
    anchor_generator = AnchorGenerator(
        sizes=(
            (16, 24, 32),       # Stride 7
            (48, 64, 80),       # Stride 14
            (96, 128, 160),     # Stride 28
            (192, 256, 320),    # Stride 56
        ),
        aspect_ratios=((0.85, 1.0, 1.15),) * 4,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes_closed_set + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=backbone.target_size,
        max_size=backbone.target_size,
        image_mean=backbone.image_mean,
        image_std=backbone.image_std,
    )
    
    # Disable internal resize since we pre-pad
    # And handle size_divisible for our coarser stride 56 (max stride)
    model.transform.size_divisible = 56 
    
    return model
