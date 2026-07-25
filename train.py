"""
Training script for medical image segmentation models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from config import TRAINING_CONFIG, MODEL_CONFIGS, DATASET_CONFIG
from models import UNet, ViTSegmentation, UNetAttention
from utils.metrics import SegmentationMetrics, DiceCrossEntropyLoss
from utils.data_loader import create_data_loaders, get_preprocessing


class Trainer:
    """Model Trainer"""
    
    def __init__(self, model, model_name='unet', device='cuda'):
        self.model = model
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Loss function
        self.criterion = DiceCrossEntropyLoss(weight=0.5)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TRAINING_CONFIG['num_epochs']
        )
        
        # Metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_dice = 0
        self.patience_counter = 0
        
        # Output directories
        self.checkpoint_dir = Path('checkpoints') / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path('logs') / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        dice_scores = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Calculate Dice for monitoring
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                for i in range(images.shape[0]):
                    dice = SegmentationMetrics.dice_coefficient(
                        preds[i].cpu().numpy(),
                        masks[i].cpu().numpy()
                    )
                    dice_scores.append(dice)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Dice': f'{np.mean(dice_scores[-len(images):]):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = np.mean(dice_scores) if dice_scores else 0
        
        self.train_metrics = {
            'loss': avg_loss,
            'dice': avg_dice
        }
        
        return avg_loss, avg_dice
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        dice_scores = []
        iou_scores = []
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Metrics
                preds = torch.argmax(outputs, dim=1)
                for i in range(images.shape[0]):
                    dice = SegmentationMetrics.dice_coefficient(
                        preds[i].cpu().numpy(),
                        masks[i].cpu().numpy()
                    )
                    iou = SegmentationMetrics.iou(
                        preds[i].cpu().numpy(),
                        masks[i].cpu().numpy()
                    )
                    dice_scores.append(dice)
                    iou_scores.append(iou)
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Dice': f'{np.mean(dice_scores[-len(images):]):.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = np.mean(dice_scores) if dice_scores else 0
        avg_iou = np.mean(iou_scores) if iou_scores else 0
        
        self.val_metrics = {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou
        }
        
        return avg_loss, avg_dice, avg_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        
        print(f"\n{'='*80}")
        print(f"Training {self.model_name} for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_dice = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Dice/train', train_dice, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/val', val_dice, epoch)
            self.writer.add_scalar('IoU/val', val_iou, epoch)
            
            # Save checkpoint
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print()
        
        self.writer.close()
        print(f"\nTraining completed! Best Val Dice: {self.best_val_dice:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def train_model(model_name='unet', data_dir=None, num_epochs=100):
    """
    Train a model
    
    Args:
        model_name: 'unet', 'vit_segmentation', or 'unet_attention'
        data_dir: Directory containing training data
        num_epochs: Number of training epochs
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    if model_name == 'unet':
        model = UNet(
            in_channels=DATASET_CONFIG['num_classes'],
            out_channels=DATASET_CONFIG['num_classes']
        )
    elif model_name == 'vit_segmentation':
        model = ViTSegmentation(
            in_channels=4,
            out_channels=DATASET_CONFIG['num_classes'],
            image_size=DATASET_CONFIG['image_size']
        )
    elif model_name == 'unet_attention':
        model = UNetAttention(
            in_channels=4,
            out_channels=DATASET_CONFIG['num_classes']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create trainer
    trainer = Trainer(model, model_name, device)
    
    # Create data loaders (placeholder - use your actual data)
    if data_dir is None:
        print("Error: Please provide data_dir with training images and masks")
        print("Expected structure:")
        print("  data_dir/")
        print("    ├── images/")
        print("    └── masks/")
        return
    
    # You would implement this with your actual data
    # train_loader, val_loader, test_loader = create_data_loaders(...)
    
    print(f"Model created: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train medical image segmentation model')
    parser.add_argument('--model', type=str, default='unet', 
                       choices=['unet', 'vit_segmentation', 'unet_attention'],
                       help='Model to train')
    parser.add_argument('--data-dir', type=str, required=False,
                       help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config
    TRAINING_CONFIG['batch_size'] = args.batch_size
    TRAINING_CONFIG['learning_rate'] = args.lr
    
    # Train model
    train_model(model_name=args.model, data_dir=args.data_dir, num_epochs=args.epochs)
