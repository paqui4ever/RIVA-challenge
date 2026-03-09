from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image


def filter_boxes_and_labels_pascal_voc(
    bboxes, labels,
    min_side: float = 20.0,
    max_ar: float = 3.0,
):
    """
    bboxes: list of [x_min, y_min, x_max, y_max] in pascal_voc coords (absolute pixels)
    labels: list aligned with bboxes
    Returns filtered (bboxes, labels).
    """
    if len(bboxes) == 0:
        return bboxes, labels

    b = np.asarray(bboxes, dtype=np.float32)
    w = np.maximum(0.0, b[:, 2] - b[:, 0])
    h = np.maximum(0.0, b[:, 3] - b[:, 1])

    min_wh = np.minimum(w, h)
    ar = np.where(h > 0, w / h, 1e9)

    keep = (min_wh >= min_side) & (ar <= max_ar) & (ar >= 1.0 / max_ar)

    b_f = b[keep].tolist()
    l_f = np.asarray(labels)[keep].tolist()

    return b_f, l_f


class BethesdaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # Loading the CSV
        self.df = pd.read_csv(csv_file)

        # Collecting the unique image_ids
        self.image_ids = self.df['image_filename'].unique()

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx] # id is reserved by python

        records = self.df[self.df['image_filename'] == image_id]

        image_path = os.path.join(self.root_dir, image_id) # It returns e.g. /images/image_01
        image = np.array(Image.open(image_path).convert('RGB')) # Converted to np array so Albumentations library supports it

        H, W, _ = image.shape

        boxes = []
        labels = []

        for _, row in records.iterrows():
            x_center, y_center = row['x'] , row['y']
            width, height = row['width'], row['height']
            class_id = row['class'] + 1

            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            x_min = max(0, min(x_min, W))
            y_min = max(0, min(y_min, H))
            x_max = max(0, min(x_max, W))
            y_max = max(0, min(y_max, H))

            if (x_max <= x_min) or (y_max <= y_min):
                continue

            # We have to adapt the center based description to the one used in FasterRCNN
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id) # 0 is reserved for the background

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)

            bboxes_f, labels_f = filter_boxes_and_labels_pascal_voc(
                transformed["bboxes"], transformed["labels"], min_side=32.0, max_ar=3.0
            )

            image = transformed['image']
            boxes = bboxes_f
            labels = labels_f

        if len(boxes) > 0:
            # If theres boxes, we need to convert them to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
            # Safety check: ensure correct shape (N, 4)
            if boxes.ndim == 1 and len(boxes) == 4:
                boxes = boxes.unsqueeze(0) 
                 
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # Calculate area and iscrowd
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
        else:
            # Case zero boxes
            # We must create an empty tensor with the correct shape (0, 4)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        tensor_image_id = torch.as_tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = tensor_image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
