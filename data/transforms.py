
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Sam3Processor


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

        A.Affine(
          scale=(0.95, 1.05),      # Scale between 95% and 105%
          translate_percent=0.02,  # Shift up to 5%
          rotate=(-5, 5),        # Rotation between -15 and 15 degrees
          shear=(-1.2, 1.2),           # Very slight stretching (mimics slide tilt)
          p=0.2
        ),

        A.GaussNoise(std_range=(0.075, 0.12), p=0.4),

        A.CoarseDropout(
          num_holes_range=(1, 8),
          hole_height_range=(10, 28),
          hole_width_range=(10, 28),
          fill=0,
          p=0.1
        ),

        # RB and RC: Adjusts lighting and sharpness of your PAP smear images
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.2
        ),

        # Normalize to 0-1 and convert to Tensor
        # Note: Mean/Std normalization is typically handled by the FasterRCNN model internal transform
        # checking against ImageNet stats. We output 0-1 float tensors here.
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['labels'],
        min_visibility=0.3 # Drops boxes that are heavily clipped or cut off
    ))


def get_train_transforms_RCNN(size: int = 1008):
    # Version-robust noise/occlusion definitions
    noise = A.OneOf(
        [
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),  # image assumed 0..255 before ToFloat
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ],
        p=0.25
    )

    dropout = A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(10, 28),
        hole_width_range=(10, 28),
        fill=0,
        p=0.1
    )

    return A.Compose(
        [
            A.Resize(size, size),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.03, 0.03),
                rotate=(-8, 8),
                shear=(-5, 5),
                fit_output=False,
                p=0.6,
            ),

            A.OneOf(
                [
                    A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.08),
                    A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20),
                    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=12),
                    A.RandomGamma(gamma_limit=(85, 115)),
                ],
                p=0.6,
            ),

            noise,
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Sharpen(alpha=(0.10, 0.30), lightness=(0.7, 1.0), p=1.0),
                ],
                p=0.20,
            ),

            dropout,

            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
            min_visibility=0.15,
            min_area=64,
        ),
    )


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
    noise = A.OneOf(
        [
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ],
        p=0.25,
    )

    dropout = A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(10, 28),
        hole_width_range=(10, 28),
        fill=0,
        p=0.1,
    )

    return A.Compose(
        [
            A.Resize(height=1008, width=1008),

            # Mild affine to avoid clipped slivers
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.03, 0.03),
                rotate=(-8, 8),
                shear=(-5, 5),
                fit_output=False,
                p=0.6,
            ),

            A.OneOf(
                [
                    A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.08),
                    A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20),
                    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=12),
                    A.RandomGamma(gamma_limit=(85, 115)),
                ],
                p=0.6,
            ),

            noise,
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Sharpen(alpha=(0.10, 0.30), lightness=(0.7, 1.0), p=1.0),
                ],
                p=0.20,
            ),

            dropout,

            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
            min_visibility=0.15,  # slightly stricter than 0.10 to reduce slivers
            min_area=64,
        ),
    )


def get_train_transforms_DETR(processor: Sam3Processor, size: int = 1008):
    # Pull the exact mean/std SAM3 expects
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    return A.Compose(
        [
            A.Resize(size, size),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=0.02,
                rotate=(-5, 5),
                shear=(-1.2, 1.2),
                p=0.2,
            ),

            A.GaussNoise(std_range=(0.075, 0.12), p=0.4),

            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(10, 28),
                hole_width_range=(10, 28),
                fill=0,
                p=0.1,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.2,
            ),

            A.ToFloat(max_value=255.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
            min_visibility=0.15,
            min_area=64,
        ),
    )

def get_train_transforms_DETR_v2(processor: Sam3Processor, size: int = 1008):
    """
    Applies Letterbox Resize (LongestMaxSize + Pad) to preserve aspect ratio.
    """
    return A.Compose(
        [
            # 1. Resize the longest side to 'size', keeping aspect ratio
            A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC),
            
            # 2. Pad the shorter side with zeros (black) to make it square
            A.PadIfNeeded(
                min_height=size, 
                min_width=size, 
                border_mode=cv2.BORDER_CONSTANT,
            ),
            
            # 3. Augmentations (Safe for cytology)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # 4. Normalize (using SAM3 stats)
            A.Normalize(
                mean=processor.image_processor.image_mean,
                std=processor.image_processor.image_std,
                max_pixel_value=255.0,
            ),
            
            # 5. Convert to Tensor
            A.pytorch.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", 
            label_fields=["labels"], 
            min_area=1.0,  # Keep small boxes!
            min_visibility=0.1
        ),
    )

def get_valid_transforms_DETR_v2(processor, size=1008):
    """
    Validation transforms must match training geometry (Letterbox).
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(
                min_height=size, 
                min_width=size, 
                border_mode=cv2.BORDER_CONSTANT, 
            ),
            A.Normalize(
                mean=processor.image_processor.image_mean,
                std=processor.image_processor.image_std,
                max_pixel_value=255.0,
            ),
            A.pytorch.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

def get_valid_transforms_DETR(processor: Sam3Processor, size: int = 1008):
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.ToFloat(max_value=255.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
        ),
    )


def get_valid_transforms():
    """
    Returns transformations for validation/testing.
    Only resizing and tensor conversion.
    """
    return A.Compose([
        A.Resize(height=1008, width=1008),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True))
