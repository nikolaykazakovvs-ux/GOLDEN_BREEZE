"""
Golden Breeze v4 - Focal Loss Implementation

FocalLoss for imbalanced classification:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

Where:
- p_t = probability of correct class
- alpha = class balancing weight
- gamma = focusing parameter (gamma > 0 reduces easy examples)

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Reduces the relative loss for well-classified examples (p_t > 0.5),
    putting more focus on hard, misclassified examples.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for class imbalance. Can be:
               - float: Same weight for all classes
               - Tensor of shape (num_classes,): Per-class weights
               - None: No class weighting
        gamma: Focusing parameter. gamma > 0 reduces easy examples.
               gamma = 0 is equivalent to CrossEntropyLoss.
               Recommended: gamma = 2.0
        reduction: 'mean', 'sum', or 'none'
        
    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 3)  # batch_size=32, num_classes=3
        >>> targets = torch.randint(0, 3, (32,))
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Logits of shape (N, C) where C = number of classes
            targets: Ground truth class indices of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Get probabilities from logits
        p = F.softmax(inputs, dim=1)
        
        # Get number of classes
        num_classes = inputs.shape[1]
        
        # Create one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Get p_t (probability of correct class)
        p_t = (p * targets_one_hot).sum(dim=1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy: -log(p_t)
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.weight,
            reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha per class
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DualLoss(nn.Module):
    """
    Combined loss for the v4 Fusion Transformer.
    
    Total Loss = score_weight * HuberLoss + class_weight * FocalLoss
    
    - Score Head: HuberLoss (robust regression)
    - Class Head: FocalLoss (imbalanced classification)
    
    Args:
        score_weight: Weight for regression loss (default: 1.0)
        class_weight: Weight for classification loss (default: 0.5)
        huber_delta: Delta for Huber loss (default: 1.0)
        focal_alpha: Alpha for Focal loss (default: 0.25)
        focal_gamma: Gamma for Focal loss (default: 2.0)
        
    Example:
        >>> criterion = DualLoss(score_weight=1.0, class_weight=0.5)
        >>> logits = torch.randn(32, 3)
        >>> scores = torch.randn(32, 1)
        >>> targets = torch.randint(0, 3, (32,))
        >>> target_scores = torch.randn(32)
        >>> loss = criterion(logits, scores, targets, target_scores)
    """
    
    def __init__(
        self,
        score_weight: float = 1.0,
        class_weight: float = 0.5,
        huber_delta: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.score_weight = score_weight
        self.class_weight = class_weight
        
        # Huber loss for regression
        self.huber = nn.HuberLoss(reduction='mean', delta=huber_delta)
        
        # Focal loss for classification
        self.focal = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            weight=class_weights,
        )
    
    def forward(
        self,
        class_logits: torch.Tensor,
        score_pred: torch.Tensor,
        class_targets: torch.Tensor,
        score_targets: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            class_logits: Classification logits (N, num_classes)
            score_pred: Score predictions (N, 1) or (N,)
            class_targets: Ground truth classes (N,)
            score_targets: Ground truth scores (N,)
            
        Returns:
            Tuple of (total_loss, class_loss, score_loss)
        """
        # Ensure score_pred is the right shape
        if score_pred.dim() == 2 and score_pred.shape[1] == 1:
            score_pred = score_pred.squeeze(1)
        
        # Classification loss
        class_loss = self.focal(class_logits, class_targets)
        
        # Regression loss
        score_loss = self.huber(score_pred, score_targets)
        
        # Combined loss
        total_loss = self.score_weight * score_loss + self.class_weight * class_loss
        
        return total_loss, class_loss, score_loss


class ClassificationOnlyLoss(nn.Module):
    """
    Classification-only loss wrapper using FocalLoss with label smoothing.
    
    Use this when you only have classification targets (no regression scores).
    
    Features:
    - Focal Loss for handling class imbalance
    - Label smoothing to prevent overconfidence
    - Per-class weights for additional balancing
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        # Register class weights as buffer (moves to device with model)
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
        
        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            weight=weight,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with optional label smoothing.
        
        Args:
            logits: (N, num_classes) raw predictions
            targets: (N,) ground truth class indices
            
        Returns:
            Scalar loss value
        """
        num_classes = logits.shape[1]
        
        if self.label_smoothing > 0:
            # Apply label smoothing
            # Convert to soft targets: (1 - eps) * one_hot + eps / num_classes
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / num_classes)
                smooth_targets.scatter_(
                    1, 
                    targets.unsqueeze(1), 
                    1 - self.label_smoothing + self.label_smoothing / num_classes
                )
            
            # Compute focal loss on smoothed targets
            p = F.softmax(logits, dim=1)
            p_t = (p * smooth_targets).sum(dim=1)
            focal_weight = (1 - p_t) ** self.gamma
            
            # Cross entropy with soft targets
            log_p = F.log_softmax(logits, dim=1)
            ce_loss = -(smooth_targets * log_p).sum(dim=1)
            
            # Apply class weights if provided
            if self.weight is not None:
                w = self.weight[targets]
                ce_loss = ce_loss * w
            
            # Apply alpha and focal weight
            focal_loss = self.alpha * focal_weight * ce_loss
            
            return focal_loss.mean()
        else:
            # Standard focal loss without smoothing
            return self.focal(logits, targets)


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Focal Loss - Quick Test")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Test FocalLoss
    print("\n1. Testing FocalLoss...")
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    logits = torch.randn(32, 3)  # batch=32, classes=3
    targets = torch.randint(0, 3, (32,))
    
    loss = criterion(logits, targets)
    print(f"   FocalLoss: {loss.item():.4f}")
    
    # Compare with CrossEntropy (gamma=0)
    ce_loss = F.cross_entropy(logits, targets)
    print(f"   CrossEntropyLoss: {ce_loss.item():.4f}")
    
    # Test DualLoss
    print("\n2. Testing DualLoss...")
    dual_criterion = DualLoss(score_weight=1.0, class_weight=0.5)
    
    class_logits = torch.randn(32, 3)
    score_pred = torch.randn(32, 1)
    class_targets = torch.randint(0, 3, (32,))
    score_targets = torch.randn(32)
    
    total, class_l, score_l = dual_criterion(
        class_logits, score_pred, class_targets, score_targets
    )
    print(f"   Total Loss: {total.item():.4f}")
    print(f"   Class Loss (Focal): {class_l.item():.4f}")
    print(f"   Score Loss (Huber): {score_l.item():.4f}")
    print(f"   Combined: {1.0 * score_l.item() + 0.5 * class_l.item():.4f}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    logits.requires_grad = True
    loss = criterion(logits, targets)
    loss.backward()
    print(f"   Gradient shape: {logits.grad.shape}")
    print(f"   Gradient norm: {logits.grad.norm().item():.4f}")
    
    print("\n✅ FocalLoss tests passed!")
