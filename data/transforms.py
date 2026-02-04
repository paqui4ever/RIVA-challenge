
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

def get_train_transforms_v2():
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
        
        # RB and RC: Adjusts lighting and sharpness of your PAP smear images
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        
        # RCD: Randomly "turns off" channels (e.g., sets a channel to 0)
        # This forces the model to learn from available dye/stain info
        A.ChannelDropout(
            channel_drop_range=(1, 1), # Drops exactly 1 channel
            fill_value=0, 
            p=0.5
        ),
        
        # Normalize to 0-1 and convert to Tensor
        # Note: Mean/Std normalization is typically handled by the FasterRCNN model internal transform
        # checking against ImageNet stats. We output 0-1 float tensors here.
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_train_transforms_v3():
    """
    Returns transformations for the training set (v3).
    Includes robust augmentations optimized for cytology as requested:
    - Geometric: Flips, Rotation
    - Noise: Light GaussNoise
    - Blur: Blur
    - Color: CLAHE, Emboss, RandomBrightnessContrast
    - Dropout: ChannelDropout
    - Cutout: CoarseDropout (proxy for CutMix/Mosaic)
    """
    return A.Compose([
        A.Resize(height=1008, width=1008),
        
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Noise (Very Light)
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        
        # Blur
        A.Blur(blur_limit=3, p=0.2),
        
        # Color/Contrast
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        
        # Dropout
        A.ChannelDropout(p=0.1),
        
        # CoarseDropout (Cutout)
        A.CoarseDropout(
            max_holes=8, 
            max_height=64, 
            max_width=64, 
            fill_value=0, 
            p=0.2
        ),
        
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
