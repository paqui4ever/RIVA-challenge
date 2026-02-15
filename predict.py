
import os
import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import models
try:
    from models.sam3_rcnn import get_sam3_faster_rcnn
    from models.sam3_DETR import get_sam3_detr
    from models.sam3_rcnn_v2 import build_sam3_fasterrcnn, sam3_resize_longest_side_and_pad_square
    from models.cell_DINO_rcnn_v2 import (
        build_cell_dino_fasterrcnn,
        cell_dino_resize_longest_side_and_pad_square,
    )
    from data.transforms import get_valid_transforms
except ImportError as e:
    raise ImportError(f"Import Error: {e}. Make sure 'models' and 'data' folders are in the path.")

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = get_valid_transforms()
        # Collect all images
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = np.array(Image.open(image_path).convert('RGB'))
        orig_h, orig_w, _ = image.shape
        
        # Validation transforms require bboxes/labels argument due to BboxParams
        # We pass dummy boxes
        dummy_boxes = []
        dummy_labels = []
        
        transformed = self.transforms(image=image, bboxes=dummy_boxes, labels=dummy_labels)
        image_tensor = transformed['image']

        return image_tensor, filename, orig_w, orig_h

def main():
    parser = argparse.ArgumentParser(description="Predict RIVA Challenge")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=['sam3_rcnn', 'sam3_rcnn_v2', 'sam3_detr', 'cell_dino_rcnn_v2', 'cell_dino'],
        required=True,
        help="The model architecture to use"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default='RIVA/images/images/test',
        help="Path to the test images directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default='./results/submission.csv',
        help="Output CSV filename"
    )
    parser.add_argument(
        "--conf_thresh", 
        type=float, 
        default=0.0,
        help="Confidence threshold for predictions"
    )
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model
    num_classes = 9 # 8 classes + background (0)
    
    if args.model == 'sam3_rcnn':
        print("Loading FasterRCNN with SAM3 backbone...")
        model = get_sam3_faster_rcnn(num_classes=num_classes)
    elif args.model == 'sam3_rcnn_v2':
        print("Loading FasterRCNN with SAM3 backbone (FPN multi-scale)...")
        model = build_sam3_fasterrcnn(
            model_name_or_path="facebook/sam3",
            num_classes_closed_set=num_classes - 1
        )
    elif args.model == 'sam3_detr':
        print("Loading DETR with SAM3 backbone...")
        model = get_sam3_detr(num_classes=num_classes)
    elif args.model in ('cell_dino_rcnn_v2', 'cell_dino'):
        print("Loading FasterRCNN with Cell-DINO backbone (FPN multi-scale)...")
        model = build_cell_dino_fasterrcnn(
            model_name="cell_dino_hpa_vitl14",
            num_classes_closed_set=num_classes - 1
        )
    else:
        raise ValueError(f"Unsupported model '{args.model}'.")

    # Load Checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
             model.load_state_dict(checkpoint['model_state_dict'])
        else:
             model.load_state_dict(checkpoint) # In case it's just the state dict
    else:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    model.to(device)
    model.eval()

    # Dataset & Loader
    print(f"Loading test data from {args.data_path}...")
    test_ds = TestDataset(root_dir=args.data_path)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    results = []

    print("Starting Inference...")
    with torch.no_grad():
        for image_tensors, filenames, orig_ws, orig_hs in tqdm(test_loader):
            # Batch size is 1, but we get batched tensors
            image_tensor = image_tensors.to(device)
            # filenames is a tuple of size 1
            filename = filenames[0]
            orig_w = orig_ws.item()
            orig_h = orig_hs.item()

            if args.model == 'sam3_rcnn':
                # Returns list of dicts [{'boxes':, 'labels':, 'scores':}]
                # Input expects list of tensors
                inputs = [image_tensor[0]] # Unbatch to list of 3D tensors
                outputs = model(inputs)
                output = outputs[0]
                
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                # Rescale boxes: 1008x1008 -> orig_w x orig_h
                scale_x = orig_w / 1008.0
                scale_y = orig_h / 1008.0

                for box, label, score in zip(boxes, labels, scores):
                    if score < args.conf_thresh:
                        continue
                    
                    # RCNN Box: x1, y1, x2, y2
                    x1, y1, x2, y2 = box
                    
                    # Convert to Original Scale
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    
                    # Convert to Center Format: x, y, width, height
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2
                    
                    # Convert Label
                    # Dataset logic: class_id = row['class'] + 1.
                    # So row['class'] = class_id - 1.
                    pred_class = label - 1
                    
                    # Filter invalid classes if any (e.g. background 0 -> -1)
                    if pred_class < 0:
                        continue

                    results.append({
                        'image_filename': filename,
                        'x': x_center,
                        'y': y_center,
                        'width': width,
                        'height': height,
                        'conf': score,
                        'class': pred_class
                    })

            elif args.model == 'sam3_rcnn_v2':
                # FasterRCNN expects list of tensors; apply SAM3 resize+pad for Cut-B backbone
                target_size = model.backbone.target_size
                processed_image, _, meta = sam3_resize_longest_side_and_pad_square(
                    image_tensor[0],
                    target=None,
                    target_size=target_size
                )
                outputs = model([processed_image])
                output = outputs[0]

                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                image_h, image_w = image_tensor.shape[-2:]
                scale_x = orig_w / float(image_w)
                scale_y = orig_h / float(image_h)

                scale = meta.scale
                resized_h, resized_w = meta.resized_hw
                resized_w_limit = max(resized_w - 1, 0)
                resized_h_limit = max(resized_h - 1, 0)

                for box, label, score in zip(boxes, labels, scores):
                    if score < args.conf_thresh:
                        continue

                    # RCNN Box: x1, y1, x2, y2
                    x1, y1, x2, y2 = box

                    # Clip to resized (pre-pad) region, then map back to preprocessed image
                    x1 = min(max(x1, 0.0), resized_w_limit) / scale
                    x2 = min(max(x2, 0.0), resized_w_limit) / scale
                    y1 = min(max(y1, 0.0), resized_h_limit) / scale
                    y2 = min(max(y2, 0.0), resized_h_limit) / scale

                    # Convert to original image scale
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y

                    # Convert to Center Format: x, y, width, height
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    # Convert Label
                    pred_class = label - 1

                    # Filter invalid classes if any (e.g. background 0 -> -1)
                    if pred_class < 0:
                        continue

                    results.append({
                        'image_filename': filename,
                        'x': x_center,
                        'y': y_center,
                        'width': width,
                        'height': height,
                        'conf': score,
                        'class': pred_class
                    })

            elif args.model == 'sam3_detr':
                # Wrapper takes pixel_values
                # Standard DETR output: logits (B, Q, num_classes+1), pred_boxes (B, Q, 4)
                # Note: sam3_detr calls model(pixel_values=stack(images))
                # For batch 1:
                pixel_values = image_tensor # (1, 3, 1008, 1008)
                outputs = model(pixel_values=pixel_values)
                
                # logits
                logits = outputs.logits[0] # (Q, num_classes+1)
                pred_boxes = outputs.pred_boxes[0] # (Q, 4) Normalized (cx, cy, w, h)
                
                probs = logits.softmax(-1) # Softmax over classes
                # Exclude the last class (no-object)
                # The classes are 0..num_labels-1. The last index 'num_labels' is no-obj.
                # In our case num_labels=9. So indices 0..8 are classes. 9 is no-obj.
                # We want the max probability among 0..8
                
                scores, labels = probs[:, :-1].max(-1)
                
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                pred_boxes = pred_boxes.cpu().numpy()
                
                for box, label, score in zip(pred_boxes, labels, scores):
                     if score < args.conf_thresh:
                        continue
                     
                     # If model predicts index 0, it's garbage/background.
                     if label == 0:
                         continue
                        
                     pred_class = label - 1
                     
                     # Box is (cx, cy, w, h) normalized [0, 1]
                     # Convert to original scale
                     cx_norm, cy_norm, w_norm, h_norm = box
                     
                     x_center = cx_norm * orig_w
                     y_center = cy_norm * orig_h
                     width = w_norm * orig_w
                     height = h_norm * orig_h
                     
                     results.append({
                        'image_filename': filename,
                        'x': x_center,
                        'y': y_center,
                        'width': width,
                        'height': height,
                        'conf': score,
                        'class': pred_class
                    })

            elif args.model in ('cell_dino_rcnn_v2', 'cell_dino'):
                # FasterRCNN expects list of tensors; apply Cell-DINO resize+pad
                target_size = model.backbone.target_size
                processed_image, _, meta = cell_dino_resize_longest_side_and_pad_square(
                    image_tensor[0],
                    target=None,
                    target_size=target_size
                )
                outputs = model([processed_image])
                output = outputs[0]

                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                image_h, image_w = image_tensor.shape[-2:]
                scale_x = orig_w / float(image_w)
                scale_y = orig_h / float(image_h)

                scale = meta.scale
                resized_h, resized_w = meta.resized_hw
                resized_w_limit = max(resized_w - 1, 0)
                resized_h_limit = max(resized_h - 1, 0)

                for box, label, score in zip(boxes, labels, scores):
                    if score < args.conf_thresh:
                        continue

                    # RCNN Box: x1, y1, x2, y2
                    x1, y1, x2, y2 = box

                    # Clip to resized (pre-pad) region, then map back to preprocessed image
                    x1 = min(max(x1, 0.0), resized_w_limit) / scale
                    x2 = min(max(x2, 0.0), resized_w_limit) / scale
                    y1 = min(max(y1, 0.0), resized_h_limit) / scale
                    y2 = min(max(y2, 0.0), resized_h_limit) / scale

                    # Convert to original image scale
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y

                    # Convert to center format: x, y, width, height
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    # Convert label to dataset class id
                    pred_class = label - 1

                    # Filter invalid classes if any (e.g. background 0 -> -1)
                    if pred_class < 0:
                        continue

                    results.append({
                        'image_filename': filename,
                        'x': x_center,
                        'y': y_center,
                        'width': width,
                        'height': height,
                        'conf': score,
                        'class': pred_class
                    })

    # Save Results
    df = pd.DataFrame(results)
    # Reorder columns to match sample
    columns = ['image_filename', 'x', 'y', 'width', 'height', 'conf', 'class']
    if not df.empty:
        df = df[columns]
    else:
        # Create empty with columns if no predictions
        df = pd.DataFrame(columns=columns)

    # Insert the 'id' column at the first position
    # Each row gets a unique index starting from 0
    df.insert(0, 'id', range(len(df)))
        
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
