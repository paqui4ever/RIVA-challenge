import os

import torch

from data.dataset import BethesdaDataset, filter_boxes_and_labels_pascal_voc


class BethesdaDatasetForSam3DETR(BethesdaDataset):
    """
    Dataset that uses albumentations transforms for augmentation and normalization.
    Returns pre-processed tensors (already normalized with SAM3 mean/std).
    """

    def __init__(self, csv_file, root_dir, transforms):
        super().__init__(csv_file, root_dir, transforms=None)
        self.albu_transforms = transforms

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        image_id = self.image_ids[idx]
        records = self.df[self.df['image_filename'] == image_id]

        image_path = os.path.join(self.root_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        h, w, _ = image_np.shape

        boxes = []
        labels = []

        for _, row in records.iterrows():
            x_center, y_center = row['x'], row['y']
            width, height = row['width'], row['height']
            raw_class = row['class']
            class_id = raw_class

            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            x_min = max(0, min(x_min, w))
            y_min = max(0, min(y_min, h))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))

            if (x_max <= x_min) or (y_max <= y_min):
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        transformed = self.albu_transforms(image=image_np, bboxes=boxes, labels=labels)

        bboxes_f, labels_f = filter_boxes_and_labels_pascal_voc(
            transformed["bboxes"], transformed["labels"], min_side=32.0, max_ar=3.0
        )

        image_tensor = transformed['image']
        boxes = bboxes_f
        labels = labels_f

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([h, w], dtype=torch.float32),
        }

        return image_tensor, target


def make_detr_collate_fn(processor, target_size: int = 1008):
    """
    Collate function for datasets that return pre-processed tensors.
    Images are already normalized by albumentations, so we skip processor normalization.

    COORDINATE SYSTEM NOTE:
    - We use HARD RESIZE (A.Resize) which warps images to target_size x target_size
    - This changes aspect ratio but simplifies coordinate handling
    - Boxes are normalized to [0,1] by dividing by target_size
    - To recover original coordinates: multiply by original (W, H)
    - This works because: orig_coord -> (orig_coord * target_size / orig_size) -> (orig_coord / orig_size) -> orig_coord

    ALTERNATIVE: Letterbox resizing (resize longest side + pad) preserves aspect ratio
    but requires tracking padding offsets. If SAM3 was pre-trained with letterbox,
    hard resize may slightly reduce feature quality, but coordinates remain consistent.
    """

    def collate(batch):
        images, targets = zip(*batch)

        pixel_values = torch.stack(images, dim=0)

        orig_sizes = []
        norm_targets = []

        for t in targets:
            orig_h, orig_w = t["orig_size"].tolist()
            orig_sizes.append([orig_h, orig_w])

            boxes = t["boxes"].clone().float()
            if boxes.numel() > 0:
                boxes[:, [0, 2]] /= target_size
                boxes[:, [1, 3]] /= target_size

            norm_targets.append({
                "labels": t["labels"].long(),
                "boxes": boxes,
            })

        orig_sizes = torch.tensor(orig_sizes, dtype=torch.float32)

        texts = ["cells"] * len(images)
        text_enc = processor.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = text_enc["input_ids"]
        # Values can't be None, look for way to not use text prompts
        #input_ids = None
        #attention_mask = None
        attention_mask = text_enc.get("attention_mask", None)

        return pixel_values, input_ids, attention_mask, norm_targets, orig_sizes

    return collate

def postprocess_and_unletterbox(boxes_norm, orig_size, target_size=1008):
    """
    Converts normalized prediction boxes (cx,cy,w,h) OR (x1,y1,x2,y2) 
    back to original image coordinates (x1,y1,x2,y2), accounting for Letterbox padding.
    """
    # 1. Convert from Center-Format (cx, cy, w, h) to Corner-Format (x1, y1, x2, y2)
    #    CRITICAL: Check if your model outputs cxcywh. Most DETR-based models do.
    #    If your model already outputs xyxy, comment this block out.
    cx, cy, w, h = boxes_norm.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # 2. Un-Letterbox Logic
    #    Calculate the scale and padding that was applied during preprocessing
    h_orig, w_orig = orig_size[0], orig_size[1]
    
    # Scale factor used (fitting longest side)
    scale = target_size / max(h_orig, w_orig)
    
    # Dimensions of the image inside the padded square
    new_h = h_orig * scale
    new_w = w_orig * scale
    
    # Calculate padding (assuming Albumentations pads to center)
    pad_h = (target_size - new_h) / 2
    pad_w = (target_size - new_w) / 2
    
    # 3. Denormalize and Remove Padding
    #    Predictions are in [0, 1] relative to the PADDED (1008x1008) image
    boxes_real = boxes_xyxy.clone()
    
    # Scale normalized coords to target_size (pixels)
    boxes_real[:, [0, 2]] *= target_size
    boxes_real[:, [1, 3]] *= target_size
    
    # Subtract padding
    boxes_real[:, [0, 2]] -= pad_w
    boxes_real[:, [1, 3]] -= pad_h
    
    # Divide by scale to get back to original resolution
    boxes_real /= scale
    
    # Clamp to original image boundaries
    boxes_real[:, 0] = boxes_real[:, 0].clamp(min=0, max=w_orig)
    boxes_real[:, 1] = boxes_real[:, 1].clamp(min=0, max=h_orig)
    boxes_real[:, 2] = boxes_real[:, 2].clamp(min=0, max=w_orig)
    boxes_real[:, 3] = boxes_real[:, 3].clamp(min=0, max=h_orig)
    
    return boxes_real