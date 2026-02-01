
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers import Sam3Model, Sam3Config

class SAM3Backbone(nn.Module):
    """
    Wraps the Hugging Face SAM3 model (image encoder) to be used as a backbone.
    """
    def __init__(self, checkpoint="facebook/sam3", out_channels=256):
        super().__init__()
        
        print(f"Loading SAM3 model from {checkpoint}...")
        try:
            self.model = Sam3Model.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading {checkpoint}: {e}")
            # Fallback or re-raise depending on strictness. 
            # Re-raising is better so user knows it failed.
            raise e
            
        # We only need the image encoding part essentially.
        # Sam3Model's structure might differ slightly but usually has a vision_encoder.
        # Check if vision_encoder exists, otherwise inspect structure.
        if hasattr(self.model, "vision_encoder"):
             self.vision_encoder = self.model.vision_encoder
        #elif hasattr(self.model, "image_encoder"):
        #     self.vision_encoder = self.model.image_encoder
        else:
             print("Warning: Could not identify vision_encoder/image_encoder. Using full model for forward.")
             self.vision_encoder = self.model
        
        # Freezing
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # Try to detect hidden size from config
        
        self.hidden_size = 1024 # default fallback
             
        self.out_channels = self.hidden_size

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            features (OrderedDict): {'0': feature_map}
        """
        # DEBUG print
        # print(f"DEBUG: Backbone input shape: {x.shape}")

        # Transformers SAM usually returns (B, SeqLen, Dim)
        outputs = self.vision_encoder(pixel_values=x)
        
        if hasattr(outputs, "last_hidden_state"):
             embeddings = outputs.last_hidden_state
        else:
             embeddings = outputs
        
        # Current shape: (B, 5184, 1024) for 1008x1008 input
        # We need (B, C, H, W) -> (B, 1024, 72, 72)
        
        if len(embeddings.shape) == 3:
            B, Seq, Dim = embeddings.shape
            Side = int(Seq ** 0.5)
            # Reshape: (B, Side, Side, Dim) -> Permute: (B, Dim, Side, Side)
            embeddings = embeddings.view(B, Side, Side, Dim).permute(0, 3, 1, 2)
            
        return OrderedDict([('0', embeddings)])


def get_sam3_faster_rcnn(num_classes, sam_checkpoint="facebook/sam3"):
    """
    Creates a FasterRCNN model with a SAM backbone from Hugging Face.
    """
    
    # 1. Backbone
    backbone = SAM3Backbone(checkpoint=sam_checkpoint, out_channels=256)
    
    # 2. Anchor Generator
    # user request: 100x100 fixed
    anchor_sizes = ((100,),) 
    aspect_ratios = ((1.0,),)
    
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # 3. ROI Pooler (Scale 1/16 typically for SAM)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=(7,7),
        sampling_ratio=2
    )
    
    # 4. Assemble
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler,
        box_nms_thresh=0.5,
        
        # SAM3 from config uses 1008 image size (patch size 14 * 72 = 1008)
        # Previous 1024 caused shape mismatch in rotary embeddings.
        # Size must be divisible by 7, 16 and 32
        min_size=1008, 
        max_size=1008,
    )
    
    # CRITICAL: FasterRCNN defaults to size_divisible=32.
    # This causes padding to 1024, resulting in 73x73 patches, mismatching 72x72 embeddings.
    model.transform.size_divisible = 14
    
    return model
