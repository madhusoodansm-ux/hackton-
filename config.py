"""
Configuration file for Medical Image Enhancement and Segmentation Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "name": "BRATS",
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "num_classes": 4,  # Background, NCR/NET, Edema, Enhancing Tumor
    "image_size": 256,
    "modalities": ["T1", "T1ce", "T2", "FLAIR"],
}

# Model configurations
MODEL_CONFIGS = {
    "unet": {
        "name": "U-Net CNN",
        "type": "cnn",
        "input_channels": 4,
        "output_channels": 4,
        "depth": 5,
        "batch_norm": True,
        "dropout": 0.2,
    },
    "vit_segmentation": {
        "name": "Vision Transformer",
        "type": "transformer",
        "patch_size": 16,
        "hidden_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "mlp_dim": 3072,
    },
    "unet_attention": {
        "name": "U-Net with Attention",
        "type": "hybrid",
        "input_channels": 4,
        "output_channels": 4,
        "depth": 5,
        "attention_blocks": [2, 3, 4],
        "dropout": 0.2,
    },
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "loss_function": "dice_cross_entropy",
    "scheduler": "cosine",
    "early_stopping_patience": 15,
    "device": "cuda",  # "cuda" or "cpu"
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.2,
    "brightness_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": True,
    "fill_mode": "constant",
    "cval": 0,
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "normalize": True,
    "normalization_method": "minmax",  # "minmax" or "zscore"
    "clip_values": True,
    "clip_range": (0, 1),
    "resize": True,
    "target_size": (256, 256),
}

# Evaluation metrics
EVALUATION_METRICS = [
    "dice_coefficient",
    "iou",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1_score",
    "hausdorff_distance",
]

# Comparison parameters
COMPARISON_CONFIG = {
    "models_to_compare": ["unet", "vit_segmentation", "unet_attention"],
    "metrics": [
        "dice_coefficient",
        "iou",
        "sensitivity",
        "inference_time",
        "model_size",
        "memory_usage",
    ],
    "num_test_samples": 50,
}