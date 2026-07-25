"""
Metrics for medical image segmentation evaluation
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F


class SegmentationMetrics:
    """Calculate segmentation metrics"""

    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        """
        Calculate Dice Coefficient
        
        Args:
            pred: Predicted segmentation (binary or one-hot)
            target: Ground truth segmentation (binary or one-hot)
            smooth: Smoothing constant to avoid division by zero
            
        Returns:
            Dice coefficient value
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item() if hasattr(dice, 'item') else float(dice)

    @staticmethod
    def iou(pred, target, smooth=1e-6):
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            smooth: Smoothing constant
            
        Returns:
            IoU value
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item() if hasattr(iou, 'item') else float(iou)

    @staticmethod
    def sensitivity(pred, target):
        """
        Calculate Sensitivity (True Positive Rate)
        
        Args:
            pred: Predicted segmentation (binary)
            target: Ground truth segmentation (binary)
            
        Returns:
            Sensitivity value
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))
        
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    @staticmethod
    def specificity(pred, target):
        """
        Calculate Specificity (True Negative Rate)
        
        Args:
            pred: Predicted segmentation (binary)
            target: Ground truth segmentation (binary)
            
        Returns:
            Specificity value
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        tn = np.sum((pred_flat == 0) & (target_flat == 0))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        
        if tn + fp == 0:
            return 0.0
        return tn / (tn + fp)

    @staticmethod
    def hausdorff_distance(pred, target):
        """
        Calculate Hausdorff Distance
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Hausdorff distance value
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        pred_points = np.argwhere(pred > 0.5)
        target_points = np.argwhere(target > 0.5)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')

        # Calculate distances from pred to target
        distances_pred_to_target = np.min(
            np.sqrt(np.sum((pred_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]) ** 2, axis=2)),
            axis=1
        )
        
        # Calculate distances from target to pred
        distances_target_to_pred = np.min(
            np.sqrt(np.sum((target_points[:, np.newaxis, :] - pred_points[np.newaxis, :, :]) ** 2, axis=2)),
            axis=1
        )
        
        hausdorff_dist = max(np.max(distances_pred_to_target), np.max(distances_target_to_pred))
        return hausdorff_dist

    @staticmethod
    def calculate_all_metrics(pred, target):
        """
        Calculate all metrics at once
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'dice': SegmentationMetrics.dice_coefficient(pred, target),
            'iou': SegmentationMetrics.iou(pred, target),
            'sensitivity': SegmentationMetrics.sensitivity(pred, target),
            'specificity': SegmentationMetrics.specificity(pred, target),
            'hausdorff': SegmentationMetrics.hausdorff_distance(pred, target),
        }
        return metrics


class DiceLoss(torch.nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted segmentation (batch_size, num_classes, height, width)
            target: Ground truth segmentation (batch_size, height, width)
        """
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(torch.nn.Module):
    """Combined Dice and Cross-Entropy Loss"""
    
    def __init__(self, smooth=1e-6, weight=0.5):
        super(DiceCrossEntropyLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.weight = weight
    
    def forward(self, pred, target):
        """Combined loss"""
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.weight * dice + (1 - self.weight) * ce