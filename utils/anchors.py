import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList
import math

class LearnableAnchorGenerator(nn.Module):
    def __init__(self, init_size=98.0):
        super().__init__()
        # 1. Define anchor dimensions as learnable parameters.
        # They are initialized to your dataset's fixed 98x98 size.
        self.anchor_w = nn.Parameter(torch.tensor(init_size, dtype=torch.float32))
        self.anchor_h = nn.Parameter(torch.tensor(init_size, dtype=torch.float32))
        
    def generate_base_anchors(self):
        # 2. Construct the base bounding box [x_min, y_min, x_max, y_max] centered at (0,0)
        # Because we use self.anchor_w and self.anchor_h, this tensor requires grad.
        half_w = self.anchor_w / 2.0
        half_h = self.anchor_h / 2.0
        return torch.stack([-half_w, -half_h, half_w, half_h]).view(1, 4)
        
    def forward(self, image_list: ImageList, feature_maps: list[torch.Tensor]):
        image_size = image_list.tensors.shape[-2:] # (H, W) of the padded batch
        device = feature_maps[0].device
        
        # Base anchors with active computation graph
        base_anchors = self.generate_base_anchors().to(device)
        
        anchors = []
        for fmap in feature_maps:
            # Calculate dynamic strides for the current feature map
            stride_h = image_size[0] / fmap.shape[-2]
            stride_w = image_size[1] / fmap.shape[-1]
            
            # Create the spatial grid of shifts
            shifts_x = torch.arange(0, fmap.shape[-1], device=device) * stride_w
            shifts_y = torch.arange(0, fmap.shape[-2], device=device) * stride_h
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            shifts = torch.stack([
                shift_x.flatten(), shift_y.flatten(), 
                shift_x.flatten(), shift_y.flatten()
            ], dim=1)
            
            # Broadcast base_anchors over the grid. 
            # This crucial step maintains the autograd graph back to our parameters.
            grid_anchors = (shifts + base_anchors).reshape(-1, 4)
            anchors.append(grid_anchors)
            
        # Standard torchvision Faster R-CNN expects a list of anchor tensors 
        # (one identical set of anchors per image in the batch)
        batch_size = len(image_list.tensors)
        return [torch.cat(anchors)] * batch_size

class FPNLearnableAnchorGenerator(nn.Module):
    def __init__(self, sizes: tuple, aspect_ratios: tuple):
        super().__init__()
        self.num_levels = len(sizes)
        
        # 1. Use ParameterList to hold separate parameters for each FPN level
        self.anchor_w = nn.ParameterList()
        self.anchor_h = nn.ParameterList()
        
        # 2. Translate sizes and aspect ratios into learnable widths and heights
        for level_sizes, level_ars in zip(sizes, aspect_ratios):
            level_w = []
            level_h = []
            for size in level_sizes:
                for ar in level_ars:
                    # Calculate initial dimensions
                    h = size * math.sqrt(ar)
                    w = size / math.sqrt(ar)
                    level_w.append(w)
                    level_h.append(h)
            
            # Register them as parameters for this specific level
            self.anchor_w.append(nn.Parameter(torch.tensor(level_w, dtype=torch.float32)))
            self.anchor_h.append(nn.Parameter(torch.tensor(level_h, dtype=torch.float32)))
            
    def generate_base_anchors(self, level_idx: int):
        # Retrieve the learnable parameters for this FPN level
        half_w = self.anchor_w[level_idx] / 2.0
        half_h = self.anchor_h[level_idx] / 2.0
        
        # Output shape: (num_anchors_per_level, 4)
        return torch.stack([-half_w, -half_h, half_w, half_h], dim=1)
        
    def forward(self, image_list: ImageList, feature_maps: list[torch.Tensor]):
        image_size = image_list.tensors.shape[-2:] 
        device = feature_maps[0].device
        
        anchors = []
        # Iterate over each FPN level
        for level_idx, fmap in enumerate(feature_maps):
            # Generate the differentiable base anchors for this level
            base_anchors = self.generate_base_anchors(level_idx)
            
            stride_h = image_size[0] / fmap.shape[-2]
            stride_w = image_size[1] / fmap.shape[-1]
            
            shifts_x = torch.arange(0, fmap.shape[-1], device=device) * stride_w
            shifts_y = torch.arange(0, fmap.shape[-2], device=device) * stride_h
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            shifts = torch.stack([
                shift_x.flatten(), shift_y.flatten(), 
                shift_x.flatten(), shift_y.flatten()
            ], dim=1)
            
            # Broadcast and shift
            grid_anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            anchors.append(grid_anchors)
            
        batch_size = len(image_list.tensors)
        return [torch.cat(anchors)] * batch_size