
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Returns transformations for the training set.
    Includes augmentations suitable for cell/cytology images (rotation invariant)
    and resizing to 1008x1008 for optimal SAM3 backbone alignment.
    """
    return A.Compose([
        # Resize to 1008x1008 to match SAM3 patch size (14 * 72 = 1008)
        # This avoids padding issues in the model.
        A.Resize(height=1008, width=1008),
        
        # Cytology images are usually rotation invariant
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Color jitter to be robust to staining variations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        
        # Normalize to 0-1 and convert to Tensor
        # Note: Mean/Std normalization is typically handled by the FasterRCNN model internal transform
        # checking against ImageNet stats. We output 0-1 float tensors here.
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transforms():
    """
    Returns transformations for validation/testing.
    Only resizing and tensor conversion.
    """
    return A.Compose([
        A.Resize(height=1008, width=1008),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
