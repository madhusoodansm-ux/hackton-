"""
Evaluation script for medical image segmentation models
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from tqdm import tqdm

from utils.metrics import SegmentationMetrics
from models import UNet, ViTSegmentation, UNetAttention


class ModelEvaluator:
    """Evaluate segmentation models"""
    
    def __init__(self, model, checkpoint_path=None, device='cuda'):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.results = {}
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        all_metrics = {
            'dice': [],
            'iou': [],
            'sensitivity': [],
            'specificity': [],
            'hausdorff': []
        }
        
        print("\nEvaluating on test set...")
        pbar = tqdm(test_loader, desc='Evaluation')
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate metrics
                for i in range(images.shape[0]):
                    pred = preds[i].cpu().numpy()
                    mask = masks[i].cpu().numpy()
                    
                    all_metrics['dice'].append(
                        SegmentationMetrics.dice_coefficient(pred, mask)
                    )
                    all_metrics['iou'].append(
                        SegmentationMetrics.iou(pred, mask)
                    )
                    all_metrics['sensitivity'].append(
                        SegmentationMetrics.sensitivity(pred, mask)
                    )
                    all_metrics['specificity'].append(
                        SegmentationMetrics.specificity(pred, mask)
                    )
                    all_metrics['hausdorff'].append(
                        SegmentationMetrics.hausdorff_distance(pred, mask)
                    )
        
        # Calculate statistics
        self.results = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
            for metric, values in all_metrics.items()
        }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("No results to print. Run evaluate() first.")
            return
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")
        
        for metric, stats in self.results.items():
            print(f"{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print()
    
    def save_results(self, output_dir='results'):
        """Save evaluation results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results_path = Path(output_dir) / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Convert numpy values to Python floats for JSON serialization
        serializable_results = {}
        for metric, stats in self.results.items():
            serializable_results[metric] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def visualize_predictions(self, test_loader, num_samples=4, output_dir='results'):
        """
        Visualize model predictions
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to visualize
            output_dir: Output directory for visualizations
        """
        self.model.eval()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        count = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                for i in range(images.shape[0]):
                    if count >= num_samples:
                        return
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Image (first channel)
                    image = images[i, 0].cpu().numpy()
                    axes[0].imshow(image, cmap='gray')
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    # Ground truth
                    mask = masks[i].cpu().numpy()
                    axes[1].imshow(mask, cmap='jet')
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    # Prediction
                    pred = preds[i].cpu().numpy()
                    axes[2].imshow(pred, cmap='jet')
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    
                    # Save
                    save_path = Path(output_dir) / f'prediction_{count}.png'
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    count += 1
        
        print(f"Saved {count} visualizations to {output_dir}")


def evaluate_model(model_name, checkpoint_path, test_loader, output_dir='results'):
    """
    Evaluate a trained model
    
    Args:
        model_name: 'unet', 'vit_segmentation', or 'unet_attention'
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        output_dir: Output directory for results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    if model_name == 'unet':
        model = UNet(in_channels=4, out_channels=4)
    elif model_name == 'vit_segmentation':
        model = ViTSegmentation(in_channels=4, out_channels=4)
    elif model_name == 'unet_attention':
        model = UNetAttention(in_channels=4, out_channels=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, checkpoint_path, device)
    
    # Evaluate
    evaluator.evaluate(test_loader)
    evaluator.print_results()
    evaluator.save_results(output_dir)
    evaluator.visualize_predictions(test_loader, output_dir=output_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate medical image segmentation model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'vit_segmentation', 'unet_attention'],
                       help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # This is a template - implement with your actual data loading
    print(f"Evaluating {args.model}...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
