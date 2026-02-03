import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.ops import generalized_box_iou

from scipy.optimize import linear_sum_assignment

from transformers import Sam3Model, Sam3Processor


# ----------------------------
# Box helpers (xyxy throughout)
# ----------------------------
def box_xyxy_clamp(boxes: Tensor) -> Tensor:
    # boxes: (..., 4)
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(0.0, 1.0)
    y1 = y1.clamp(0.0, 1.0)
    x2 = x2.clamp(0.0, 1.0)
    y2 = y2.clamp(0.0, 1.0)
    # Ensure proper ordering
    x1_, x2_ = torch.min(x1, x2), torch.max(x1, x2)
    y1_, y2_ = torch.min(y1, y2), torch.max(y1, y2)
    return torch.stack([x1_, y1_, x2_, y2_], dim=-1)


# ----------------------------
# Hungarian matcher (DETR-style)
# ----------------------------
class HungarianMatcher(nn.Module):
    """
    Computes an assignment between targets and predictions for each batch element.
    """
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All costs can't be 0")

    @torch.no_grad()
    def forward(self, pred_logits: Tensor, pred_boxes: Tensor, targets: List[Dict[str, Tensor]]):
        """
        pred_logits: (B, Q, C+1)  (includes no-object at index C)
        pred_boxes:  (B, Q, 4)    (xyxy, normalized)
        targets: list of dicts with:
            - labels: (Ni,)
            - boxes:  (Ni, 4) (xyxy, normalized)
        """
        B, Q, num_classes_plus_bg = pred_logits.shape
        device = pred_logits.device

        # Softmax over classes for classification cost
        out_prob = pred_logits.softmax(-1)  # (B, Q, C+1)

        indices: List[Tuple[Tensor, Tensor]] = []

        for b in range(B):
            tgt_labels = targets[b]["labels"]          # (Nb,)
            tgt_boxes  = targets[b]["boxes"]           # (Nb, 4)
            if tgt_labels.numel() == 0:
                # No targets: match nothing
                indices.append((torch.empty(0, dtype=torch.long, device=device),
                                torch.empty(0, dtype=torch.long, device=device)))
                continue

            # Classification cost: -P(class)
            # out_prob[b]: (Q, C+1) -> take columns of tgt_labels -> (Q, Nb)
            cost_class = -out_prob[b][:, tgt_labels]

            # L1 bbox cost: (Q, Nb)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)

            # GIoU cost: -(giou) (Q, Nb)
            cost_giou = -generalized_box_iou(pred_boxes[b], tgt_boxes)

            C = (
                self.cost_class * cost_class
                + self.cost_bbox  * cost_bbox
                + self.cost_giou  * cost_giou
            )

            # Hungarian assignment on CPU (SciPy)
            C_cpu = C.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_cpu)

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=device),
                torch.as_tensor(col_ind, dtype=torch.long, device=device),
            ))

        return indices


# ----------------------------
# SetCriterion (losses)
# ----------------------------
class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
    ):
        """
        num_classes: number of foreground classes (8)
        eos_coef: weight for the "no-object" class in CE loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        # Class weights: down-weight the no-object (background) class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[num_classes] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, pred_logits: Tensor, indices, targets):
        """
        Classification loss (cross entropy) over Q queries.
        """
        B, Q, _ = pred_logits.shape
        device = pred_logits.device

        # Fill all queries with "no-object"
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=device)

        # Set matched queries to their target labels
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, C+1, Q)
            target_classes,
            weight=self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, pred_boxes: Tensor, indices, targets):
        """
        Box regression losses: L1 + (1 - GIoU)
        """
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            # No matched pairs in entire batch
            return {
                "loss_bbox": pred_boxes.sum() * 0.0,
                "loss_giou": pred_boxes.sum() * 0.0,
            }

        src_boxes = pred_boxes[idx]  # (num_matches, 4)

        target_boxes = torch.cat([
            t["boxes"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)
        ], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / max(1, target_boxes.shape[0])

        giou = generalized_box_iou(src_boxes, target_boxes)
        loss_giou = (1.0 - giou.diag()).sum() / max(1, target_boxes.shape[0])

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, pred_logits: Tensor, pred_boxes: Tensor, targets: List[Dict[str, Tensor]]):
        # Match
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # Losses
        losses = {}
        losses.update(self.loss_labels(pred_logits, indices, targets))
        losses.update(self.loss_boxes(pred_boxes, indices, targets))

        # Weighted sum
        total = 0.0
        for k, v in losses.items():
            w = self.weight_dict.get(k, 1.0)
            total = total + w * v
        losses["loss_total"] = total
        return losses


# ----------------------------
# Cut A Model: SAM3 + closed-set head
# ----------------------------
class Sam3ForClosedSetDetection(nn.Module):
    """
    Cut A:
      - Keep SAM3 perception encoder + DETR encoder/decoder
      - Use decoder_hidden_states[-1] as query features
      - Replace open-vocab pred_logits with learned linear classifier (C+1)
      - Reuse SAM3 pred_boxes (xyxy) for regression target
    """
    def __init__(
        self,
        sam3_checkpoint: str = "facebook/sam3",
        num_classes: int = 8,
        freeze_sam3: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.sam3 = Sam3Model.from_pretrained(sam3_checkpoint)

        hidden_size = self.sam3.config.detr_decoder_config.hidden_size  # typically 256
        self.class_embed = nn.Linear(hidden_size, num_classes + 1)      # +1 = no-object

        if freeze_sam3:
            for p in self.sam3.parameters():
                p.requires_grad = False

    def forward(
        self,
        pixel_values: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **sam3_kwargs,
    ):
        """
        pixel_values: (B, 3, 1008, 1008) from Sam3Processor
        targets: list of dicts with normalized xyxy boxes in [0,1] and labels in [0..C-1]
        """
        outputs = self.sam3(
            pixel_values=pixel_values,
            output_hidden_states=True,  # needed for decoder_hidden_states
            return_dict=True,
            **sam3_kwargs,
        )

        # Query features from last decoder layer
        # outputs.decoder_hidden_states: tuple[L] of (B, Q, hidden_size)
        dec_last = outputs.decoder_hidden_states[-1]

        logits = self.class_embed(dec_last)  # (B, Q, C+1)

        # SAM3 provides pred_boxes already in xyxy (per HF docs).
        # We assume normalized [0,1] (common for transformer detectors); clamp for safety.
        boxes = box_xyxy_clamp(outputs.pred_boxes)  # (B, Q, 4)

        out = {"logits": logits, "boxes": boxes, "sam3_outputs": outputs}

        if targets is not None:
            out["losses"] = self.criterion(logits, boxes, targets)

        return out

    def build_criterion(
        self,
        class_cost: float = 1.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        eos_coef: float = 0.1,
        loss_ce_w: float = 1.0,
        loss_bbox_w: float = 5.0,
        loss_giou_w: float = 2.0,
    ):
        matcher = HungarianMatcher(cost_class=class_cost, cost_bbox=bbox_cost, cost_giou=giou_cost)
        weight_dict = {"loss_ce": loss_ce_w, "loss_bbox": loss_bbox_w, "loss_giou": loss_giou_w}
        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
        )
        return self

    @torch.no_grad()
    def predict(
        self,
        pixel_values: Tensor,
        orig_sizes: Optional[Tensor] = None,
        score_thresh: float = 0.3,
        max_detections: int = 100,
    ):
        """
        Returns per-image detections in absolute xyxy if orig_sizes provided, else normalized xyxy.
        orig_sizes: (B, 2) as (H, W) from Sam3Processor's `original_sizes`.
        """
        self.eval()
        out = self(pixel_values)
        logits, boxes = out["logits"], out["boxes"]

        prob = logits.softmax(-1)                  # (B, Q, C+1)
        scores, labels = prob[..., :-1].max(-1)    # ignore no-object

        results = []
        B, Q, _ = logits.shape
        for b in range(B):
            keep = scores[b] > score_thresh
            s = scores[b][keep]
            l = labels[b][keep]
            bx = boxes[b][keep]

            if s.numel() > max_detections:
                topk = torch.topk(s, k=max_detections)
                s = topk.values
                idx = topk.indices
                l = l[idx]
                bx = bx[idx]

            if orig_sizes is not None:
                h, w = orig_sizes[b].tolist()
                bx_abs = bx.clone()
                bx_abs[:, [0, 2]] *= float(w)
                bx_abs[:, [1, 3]] *= float(h)
                bx = bx_abs

            results.append({"scores": s, "labels": l, "boxes": bx})
        return results


# ----------------------------
# Processor + collate_fn example
# ----------------------------
def make_sam3_collate_fn(processor: Sam3Processor):
    """
    Expects dataset to return: (PIL_or_np_image, target_dict)
      target_dict:
        - boxes: FloatTensor (N,4) absolute xyxy in original image pixels (1024x1024 in your case)
        - labels: LongTensor  (N,) in [0..7]
    """
    def collate(batch):
        images, targets = zip(*batch)
        enc = processor(images=list(images), return_tensors="pt")
        pixel_values = enc["pixel_values"]  # (B,3,1008,1008) :contentReference[oaicite:4]{index=4}
        orig_sizes = torch.tensor(enc["original_sizes"], dtype=torch.float32)  # (B,2) (H,W) :contentReference[oaicite:5]{index=5}

        # Normalize boxes to [0,1] in xyxy using original image size.
        # This stays consistent under uniform resize (1024 -> 1008).
        norm_targets = []
        for t, (h, w) in zip(targets, orig_sizes):
            boxes = t["boxes"].clone().float()
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
            boxes = box_xyxy_clamp(boxes)
            norm_targets.append({
                "labels": t["labels"].long(),
                "boxes": boxes,
            })

        return pixel_values, norm_targets, orig_sizes

    return collate


# ----------------------------
# Minimal training step sketch
# ----------------------------
def train_one_step(model, batch, device="cuda"):
    pixel_values, targets, _orig_sizes = batch
    pixel_values = pixel_values.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    out = model(pixel_values=pixel_values, targets=targets)
    losses = out["losses"]
    loss = losses["loss_total"]

    return loss, losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Sam3Processor.from_pretrained("facebook/sam3")

    model = (
        Sam3ForClosedSetDetection("facebook/sam3", num_classes=8, freeze_sam3=False)
        .build_criterion()
        .to(device)
    )

    # Example optimizer (you'll likely want smaller LR for SAM3, bigger for class head)
    optimizer = torch.optim.AdamW([
        {"params": model.sam3.parameters(), "lr": 1e-5},
        {"params": model.class_embed.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    # In your DataLoader:
    # collate_fn = make_sam3_collate_fn(processor)
    # loader = DataLoader(dataset, batch_size=..., collate_fn=collate_fn, ...)

    # Then:
    # for batch in loader:
    #     loss, losses = train_one_step(model, batch, device=device)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
