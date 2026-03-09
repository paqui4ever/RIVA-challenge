import torch
import torch.nn.functional as F

def softmax_focal_loss(logits, labels, gamma=2.0, alpha=None, reduction='mean'):
    """
    Multi-class Focal Loss formulation:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    # Calculate probabilities (Softmax)
    probs = F.softmax(logits, dim=1)
    
    # Gather the probabilities of the correct classes (p_t)
    # labels.view(-1, 1) reshapes to (N, 1) to gather along dim 1
    p_t = probs.gather(1, labels.view(-1, 1))
    
    # Calculate the focusing factor (1 - p_t)^gamma
    gamma_factor = (1 - p_t) ** gamma
    
    # Calculate Cross Entropy: -log(p_t)
    # We use log_softmax for numerical stability instead of log(probs)
    log_p_t = F.log_softmax(logits, dim=1).gather(1, labels.view(-1, 1))
    ce_loss = -log_p_t
    
    # Apply Alpha (Class Weights) if provided
    if alpha is not None:
        # alpha is a tensor of shape [num_classes]
        # We select the alpha corresponding to the correct label
        alpha_t = alpha.gather(0, labels.view(-1))
        # Add a dimension to match shapes
        alpha_t = alpha_t.view(-1, 1)
        
        # Combine everything
        loss = alpha_t * gamma_factor * ce_loss
    else:
        loss = gamma_factor * ce_loss
        
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def custom_faster_rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (List[int64])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    # We replace F.cross_entropy with our softmax_focal_loss

    # concatenating the labels and regression_targets
    # just like the original fastrcnn_loss does
    if isinstance(labels, (list, tuple)):
        labels = torch.cat(labels, dim=0)
    if isinstance(regression_targets, (list, tuple)):
        regression_targets = torch.cat(regression_targets, dim=0)

    # Define your class weights (Move to GPU inside the function or global scope)
    alpha = torch.tensor([
         1.00,  # Background (0)
         1.14,  # NILM
         6.53,  # ENDO
         1.00,  # INFL
         21.06,  # ASCUS (1) - HIGH PRIORITY
         2.76,  # LSIL
         3.54,  # HSIL
         13.80,  # ASCH
         4.03   # SCC
    ], device=class_logits.device) # Ensure it moves to same device as logits

    classification_loss = softmax_focal_loss(
        class_logits, 
        labels, 
        gamma=2.0,       # standard for Focal Loss
        alpha=alpha,     # should be remove if pure Focal Loss without weights is wanted
        reduction='mean'
    )

    # Box regression loss (Standard implementation)
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1.0 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss