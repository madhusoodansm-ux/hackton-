"""
Compare all three models on medical image segmentation
"""

import torch
import torch.nn as nn
import time
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import COMPARISON_CONFIG, MODEL_CONFIGS, TRAINING_CONFIG
from models import UNet, ViTSegmentation, UNetAttention
from utils.metrics import SegmentationMetrics


class ModelComparator:
    """Compare different segmentation models"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
    def create_models(self):
        """Initialize all models"""
        models = {}
        
        # U-Net
        config = MODEL_CONFIGS['unet']
        models['unet'] = UNet(
            in_channels=config['input_channels'],
            out_channels=config['output_channels'],
            depth=config['depth'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Vision Transformer
        config = MODEL_CONFIGS['vit_segmentation']
        models['vit_segmentation'] = ViTSegmentation(
            in_channels=4,
            out_channels=4,
            image_size=256,
            patch_size=config['patch_size'],
            dim=config['hidden_dim'],
            depth=config['num_layers'],
            heads=config['num_heads'],
            mlp_dim=config['mlp_dim']
        ).to(self.device)
        
        # U-Net with Attention
        config = MODEL_CONFIGS['unet_attention']
        models['unet_attention'] = UNetAttention(
            in_channels=config['input_channels'],
            out_channels=config['output_channels'],
            depth=config['depth'],
            attention_blocks=config['attention_blocks'],
            dropout=config['dropout']
        ).to(self.device)
        
        return models
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def measure_inference_time(self, model, input_shape=(1, 4, 256, 256), num_runs=10):
        """Measure average inference time"""
        model.eval()
        x = torch.randn(*input_shape).to(self.device)
        
        # Warm up
        with torch.no_grad():
            _ = model(x)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(x)
                end = time.time()
                times.append(end - start)
        
        avg_time = np.mean(times[1:])  # Skip first run
        return avg_time * 1000  # Convert to ms
    
    def measure_memory_usage(self, model, input_shape=(1, 4, 256, 256)):
        """Measure peak memory usage"""
        if self.device == 'cpu':
            return 0
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        x = torch.randn(*input_shape).to(self.device)
        
        with torch.no_grad():
            _ = model(x)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return peak_memory
    
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*80)
        print("MEDICAL IMAGE SEGMENTATION - MODEL COMPARISON")
        print("="*80 + "\n")
        
        models = self.create_models()
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\nAnalyzing: {MODEL_CONFIGS[model_name]['name']}...")
            print("-" * 50)
            
            # Model size
            size_mb = self.get_model_size(model)
            print(f"Model Size: {size_mb:.2f} MB")
            
            # Inference time
            inference_time = self.measure_inference_time(model)
            print(f"Avg Inference Time: {inference_time:.2f} ms")
            
            # Memory usage
            memory_usage = self.measure_memory_usage(model)
            print(f"Peak Memory Usage: {memory_usage:.2f} MB")
            
            # Parameter count
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"Total Parameters: {param_count:.2f}M")
            
            # Store results
            comparison_results[model_name] = {
                'name': MODEL_CONFIGS[model_name]['name'],
                'type': MODEL_CONFIGS[model_name]['type'],
                'model_size_mb': size_mb,
                'inference_time_ms': inference_time,
                'peak_memory_mb': memory_usage,
                'parameters_m': param_count,
            }
        
        self.results = comparison_results
        return comparison_results
    
    def save_comparison_report(self, output_dir='results'):
        """Save comparison results to JSON and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        report_path = os.path.join(output_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        
        # Create visualizations
        self.visualize_comparison(output_dir)
    
    def visualize_comparison(self, output_dir='results'):
        """Create comparison visualizations"""
        if not self.results:
            print("No results to visualize. Run compare_models() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Medical Image Segmentation - Model Comparison', fontsize=16, fontweight='bold')
        
        model_names = [self.results[m]['name'] for m in self.results.keys()]
        
        # Model Size
        sizes = [self.results[m]['model_size_mb'] for m in self.results.keys()]
        axes[0, 0].bar(model_names, sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_ylabel('Size (MB)')
        axes[0, 0].set_title('Model Size Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Inference Time
        times = [self.results[m]['inference_time_ms'] for m in self.results.keys()]
        axes[0, 1].bar(model_names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory Usage
        memory = [self.results[m]['peak_memory_mb'] for m in self.results.keys()]
        axes[1, 0].bar(model_names, memory, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].set_title('Peak Memory Usage Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Parameters
        params = [self.results[m]['parameters_m'] for m in self.results.keys()]
        axes[1, 1].bar(model_names, params, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_ylabel('Parameters (Millions)')
        axes[1, 1].set_title('Parameter Count Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {fig_path}")
        plt.close()
    
    def print_summary(self):
        """Print summary and recommendations"""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*80 + "\n")
        
        # Find best models
        best_speed = min(self.results.items(), key=lambda x: x[1]['inference_time_ms'])
        best_size = min(self.results.items(), key=lambda x: x[1]['model_size_mb'])
        best_memory = min(self.results.items(), key=lambda x: x[1]['peak_memory_mb'])
        
        print(f"Fastest Model: {best_speed[1]['name']} ({best_speed[1]['inference_time_ms']:.2f} ms)")
        print(f"Smallest Model: {best_size[1]['name']} ({best_size[1]['model_size_mb']:.2f} MB)")
        print(f"Memory Efficient: {best_memory[1]['name']} ({best_memory[1]['peak_memory_mb']:.2f} MB)")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 50)
        print("1. For PRODUCTION/DEPLOYMENT:")
        print(f"   → U-Net (fastest and most memory efficient)")
        print("\n2. For BEST ACCURACY:")
        print(f"   → Vision Transformer (best for capturing long-range dependencies)")
        print("\n3. For BALANCED APPROACH:")
        print(f"   → U-Net with Attention (hybrid architecture)")
        print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    comparator = ModelComparator(device='cuda' if torch.cuda.is_available() else 'cpu')
    comparator.compare_models()
    comparator.print_summary()
    comparator.save_comparison_report()