from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from utils.loss import custom_faster_rcnn_loss

import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from torchvision.models.detection import FasterRCNN, roi_heads
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from transformers import Sam3Processor, Sam3Model


_ORIGINAL_FASTRCNN_LOSS = roi_heads.fastrcnn_loss


# -------------------------
# Preprocess: SAM3-like square resize + pad (so FasterRCNN's internal resize becomes a no-op)
# -------------------------
@dataclass
class Sam3ResizePadMeta:
    scale: float
    resized_hw: Tuple[int, int]     # (h, w) after resize, before pad
    pad_rb: Tuple[int, int]         # (pad_right, pad_bottom)

def sam3_resize_longest_side_and_pad_square(
    img: torch.Tensor,
    target: Optional[Dict[str, torch.Tensor]] = None,
    target_size: int = 1008,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Sam3ResizePadMeta]:
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
        # Shouldn't happen, but guard against rounding weirdness:
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

    meta = Sam3ResizePadMeta(scale=scale, resized_hw=(new_h, new_w), pad_rb=(pad_right, pad_bottom))
    return img, target, meta


# -------------------------
# Cut B backbone: SAM3 Vision (ViT + FPN) -> OrderedDict[str, Tensor] for FasterRCNN
# -------------------------
class Sam3Backbone(nn.Module):
    """
    Wraps Sam3Model so it behaves like a torchvision detection backbone.
    Expects input: images tensor [B,3,H,W] already normalized & padded to 1008x1008.
    Returns: OrderedDict of multi-scale FPN feature maps with consistent channels.
    """
    def __init__(self, model_name_or_path: str = "facebook/sam3", trainable: bool = True):
        super().__init__()
        self.processor = Sam3Processor.from_pretrained(model_name_or_path)
        model = Sam3Model.from_pretrained(model_name_or_path)
        self.vision = model.vision_encoder if hasattr(model, "vision_encoder") else model

        # FPN channels are fixed by config (default 256)
        self.out_channels = int(getattr(self.vision.config, "fpn_hidden_size", 256))

        if not trainable:
            for p in self.vision.parameters():
                p.requires_grad = False

    @property
    def image_mean(self) -> List[float]:
        # Use the processor's normalization to keep SAM3-compatible inputs.
        # (Works even if Meta changes the defaults in future checkpoints.)
        mean = getattr(self.processor.image_processor, "image_mean", None)
        if mean is None:
            # Fallback: ImageNet mean
            mean = [0.485, 0.456, 0.406]
        return list(mean)

    @property
    def image_std(self) -> List[float]:
        std = getattr(self.processor.image_processor, "image_std", None)
        if std is None:
            # Fallback: ImageNet std
            std = [0.229, 0.224, 0.225]
        return list(std)

    @property
    def target_size(self) -> int:
        # SAM3 ViT default image_size is 1008 in config; keep it explicit for detection.  :contentReference[oaicite:3]{index=3}
        return int(getattr(getattr(self.vision.config, "backbone_config", None), "image_size", 1008))

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        # torchvision FasterRCNN calls backbone(images.tensors) where tensors is [B,3,H,W]
        outputs = self.vision(pixel_values=x, return_dict=True)

        # Prefer the FPN outputs if available (Cut B).
        fpn_feats = getattr(outputs, "fpn_hidden_states", None)
        if fpn_feats is None:
            raise RuntimeError(
                "Sam3VisionModel did not return `fpn_hidden_states`. "
                "Make sure you're using a Transformers version/checkpoint that supports SAM3 Vision FPN outputs."
            )

        # fpn_feats is expected to be a tuple/list of [B,C,H,W] tensors (4 levels by default). :contentReference[oaicite:4]{index=4}
        feats = OrderedDict((str(i), fpn_feats[i]) for i in range(len(fpn_feats)))

        # Sanity: ensure channel dim matches out_channels
        for k, v in feats.items():
            if v.shape[1] != self.out_channels:
                raise RuntimeError(f"Feature {k} has {v.shape[1]} channels, expected {self.out_channels}.")
        return feats


# -------------------------
# Build Faster R-CNN with SAM3 Cut B backbone (8 classes => num_classes=9 incl background)
# -------------------------
def build_sam3_fasterrcnn(
    model_name_or_path: str = "facebook/sam3",
    num_classes_closed_set: int = 8,
    trainable_backbone: bool = True,
    loss_type: str = "custom",
) -> FasterRCNN:
    if loss_type not in {"custom", "original"}:
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Expected one of: 'custom', 'original'.")

    backbone = Sam3Backbone(model_name_or_path, trainable=trainable_backbone)
    target_size = backbone.target_size

    # SAM3 FPN has 4 levels by default (scale_factors length = 4). :contentReference[oaicite:5]{index=5}
    featmap_names = ["0", "1", "2", "3"]
    
    anchor_generator = AnchorGenerator(
         # 4 FPN levels (featmap_names ["0","1","2","3"])
         # Cover ~74–173px after resize (1008/1024)
         # BASELINE
         #sizes=((64, 80), (96, 112), (128, 160), (176, 192)),
         # FOURTH BEST SO FAR
         #sizes=((70, 80), (93, 110), (125, 150), (160, 180)),
         # SECOND BEST SO FAR (MARGINAL DIFFERENCE WITH BEST)
         # sizes=((70, 77), (93, 105), (125, 140), (160, 170)),
         # THIRD BEST SO FAR 
         # sizes=((72, 77), (96, 103), (130, 138), (163, 170)),
         # BEST 
         sizes=((71, 78), (92, 104), (123, 135), (158, 168)),
         # Mostly square objects -> keep ratios tight around 1
         # BASELINE
         #aspect_ratios=((0.85, 1.0, 1.15),) * 4,
         # BEST 
         aspect_ratios=((0.82, 1.0, 1.12),) * 4,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        #output_size=14,
        output_size=11, # Tried with 9, 10, 11, 12. 11 is the best so far.
        sampling_ratio=3, # Tried with 2, 3 and 4. They are all very similar, keeping it at 3 for now.
    )

    selected_loss_fn = custom_faster_rcnn_loss if loss_type == "custom" else _ORIGINAL_FASTRCNN_LOSS
    roi_heads.fastrcnn_loss = selected_loss_fn

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes_closed_set + 1,  # +1 background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        #rpn_nms_thresh=0.75, # Increase a little bit to prevent merging two cells into one
        # # Defines what the RPN considers a "positive" anchor to train on.
        #box_fg_iou_thresh=0.40,  
        # # Usually kept the same as box_fg_iou_thresh to define the boundary 
        # # between "background" and "foreground".
        # Make FasterRCNN's internal resize a no-op (we pre-pad to target_size x target_size)
        min_size=target_size,
        max_size=target_size,
        image_mean=backbone.image_mean,
        image_std=backbone.image_std,
    )

    assert roi_heads.fastrcnn_loss == selected_loss_fn, \
    "ERROR: The patch failed! The selected Faster R-CNN loss function was not applied."

    # CRITICAL: FasterRCNN defaults to size_divisible=32.
    # This causes padding to 1024, resulting in 73x73 patches, mismatching 72x72 embeddings.
    # SAM3 uses patch_size=14, so we need size_divisible=14 (or 1 to disable padding entirely).
    model.transform.size_divisible = 14

    return model


# -------------------------
# Example usage (training step)
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_fasterrcnn("facebook/sam3", num_classes_closed_set=8, trainable_backbone=True).to(device)

    # Dummy batch: list[Tensor] images + list[Dict] targets (torchvision detection convention)
    # Assume you load an image as float tensor [3,H,W] in [0,1] and boxes in xyxy pixel coords.
    images: List[torch.Tensor] = [torch.rand(3, 720, 1280), torch.rand(3, 900, 900)]
    targets: List[Dict[str, torch.Tensor]] = [
        {"boxes": torch.tensor([[100.0, 120.0, 400.0, 500.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[50.0, 60.0, 200.0, 240.0]]), "labels": torch.tensor([3])},
    ]

    # Preprocess each image/target to SAM3 expected square size
    processed_images: List[torch.Tensor] = []
    processed_targets: List[Dict[str, torch.Tensor]] = []
    target_size = model.backbone.target_size  # 1008 by default

    for img, tgt in zip(images, targets):
        img, tgt, _ = sam3_resize_longest_side_and_pad_square(img, tgt, target_size=target_size)
        processed_images.append(img.to(device))
        processed_targets.append({k: v.to(device) for k, v in tgt.items()})

    model.train()
    losses = model(processed_images, processed_targets)
    loss = sum(losses.values())
    loss.backward()

    # Inference
    model.eval()
    with torch.no_grad():
        detections = model(processed_images)
        # detections: list of dicts with boxes, labels, scores
        print(detections[0].keys())
