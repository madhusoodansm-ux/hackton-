"""
Data loading and preprocessing utilities for medical images
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


class MedicalImageDataset(Dataset):
    """Medical Image Dataset"""
    
    def __init__(self, image_paths, mask_paths=None, augmentation=None, preprocessing=None, mode='train'):
        """
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks (can be None for inference)
            augmentation: Albumentations augmentation pipeline
            preprocessing: Preprocessing function
            mode: 'train', 'val', or 'test'
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Load mask if available
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = self.load_mask(mask_path)
        else:
            mask = None
        
        # Augmentation
        if self.augmentation and self.mode == 'train':
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            if mask is not None:
                mask = augmented['mask']
        
        # Preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
        
        if mask is not None:
            return torch.from_numpy(image).float(), torch.from_numpy(mask).long()
        else:
            return torch.from_numpy(image).float()
    
    def load_image(self, path):
        """Load image from file"""
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            # NIfTI format (medical imaging)
            img = nib.load(path).get_fdata()
        elif path.endswith(('.png', '.jpg', '.jpeg')):
            # Standard image formats
            from PIL import Image
            img = np.array(Image.open(path))
        else:
            # NumPy array
            img = np.load(path)
        
        # Ensure 3D
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        return img
    
    def load_mask(self, path):
        """Load segmentation mask"""
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            mask = nib.load(path).get_fdata()
        elif path.endswith(('.png', '.jpg', '.jpeg')):
            from PIL import Image
            mask = np.array(Image.open(path))
        else:
            mask = np.load(path)
        
        # Ensure 2D
        if len(mask.shape) == 3:
            mask = np.argmax(mask, axis=-1)
        
        return mask


class DataPreprocessor:
    """Preprocessing utilities for medical images"""
    
    @staticmethod
    def normalize_minmax(image, axis=None):
        """Min-Max normalization"""
        if axis is not None:
            min_val = np.min(image, axis=axis, keepdims=True)
            max_val = np.max(image, axis=axis, keepdims=True)
        else:
            min_val = np.min(image)
            max_val = np.max(image)
        
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        return np.clip(normalized, 0, 1)
    
    @staticmethod
    def normalize_zscore(image, axis=None):
        """Z-score normalization"""
        if axis is not None:
            mean = np.mean(image, axis=axis, keepdims=True)
            std = np.std(image, axis=axis, keepdims=True)
        else:
            mean = np.mean(image)
            std = np.std(image)
        
        normalized = (image - mean) / (std + 1e-8)
        return normalized
    
    @staticmethod
    def clip_values(image, min_val=0, max_val=1):
        """Clip image values"""
        return np.clip(image, min_val, max_val)
    
    @staticmethod
    def resize_image(image, target_size=(256, 256)):
        """Resize image"""
        from skimage.transform import resize
        if len(image.shape) == 3:
            resized = resize(image, (target_size[0], target_size[1], image.shape[2]), order=1)
        else:
            resized = resize(image, target_size, order=1)
        return resized


def get_augmentation_pipeline(image_size=256, mode='train'):
    """Get data augmentation pipeline"""
    if mode == 'train':
        return A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(image_size, image_size),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2(),
        ])


def get_preprocessing(normalization='minmax'):
    """Get preprocessing function"""
    def preprocess(image):
        # Handle multi-channel images
        if len(image.shape) == 3:
            processed = np.zeros_like(image)
            for i in range(image.shape[2]):
                if normalization == 'minmax':
                    processed[:, :, i] = DataPreprocessor.normalize_minmax(image[:, :, i])
                elif normalization == 'zscore':
                    processed[:, :, i] = DataPreprocessor.normalize_zscore(image[:, :, i])
        else:
            if normalization == 'minmax':
                processed = DataPreprocessor.normalize_minmax(image)
            elif normalization == 'zscore':
                processed = DataPreprocessor.normalize_zscore(image)
        
        return processed.astype(np.float32)
    
    return preprocess


def create_data_loaders(image_paths, mask_paths, batch_size=16, num_workers=4, 
                        test_split=0.2, val_split=0.15, random_state=42):
    """
    Create train, validation, and test data loaders
    
    Args:
        image_paths: List of image paths
        mask_paths: List of mask paths
        batch_size: Batch size
        num_workers: Number of workers for data loading
        test_split: Test set ratio
        val_split: Validation set ratio
        random_state: Random seed
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split data
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_split, random_state=random_state
    )
    
    val_size = val_split / (1 - test_split)
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=val_size, random_state=random_state
    )
    
    # Preprocessing
    preprocessing = get_preprocessing(normalization='minmax')
    
    # Training dataset
    train_dataset = MedicalImageDataset(
        train_images, train_masks,
        augmentation=get_augmentation_pipeline(mode='train'),
        preprocessing=preprocessing,
        mode='train'
    )
    
    # Validation dataset
    val_dataset = MedicalImageDataset(
        val_images, val_masks,
        augmentation=get_augmentation_pipeline(mode='val'),
        preprocessing=preprocessing,
        mode='val'
    )
    
    # Test dataset
    test_dataset = MedicalImageDataset(
        test_images, test_masks,
        augmentation=get_augmentation_pipeline(mode='val'),
        preprocessing=preprocessing,
        mode='test'
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def load_brats_dataset(data_dir, modalities=['T1', 'T1ce', 'T2', 'FLAIR']):
    """
    Load BRATS dataset
    
    Args:
        data_dir: Directory containing BRATS data
        modalities: MRI modalities to load
    
    Returns:
        image_paths, mask_paths
    """
    data_dir = Path(data_dir)
    image_paths = []
    mask_paths = []
    
    for patient_dir in sorted(data_dir.glob('*/')):
        if patient_dir.is_dir():
            # Find segmentation mask
            seg_files = list(patient_dir.glob('*seg.nii.gz'))
            if seg_files:
                mask_path = str(seg_files[0])
                # Find modality files
                modality_files = []
                for mod in modalities:
                    mod_files = list(patient_dir.glob(f'*{mod}.nii.gz'))
                    if mod_files:
                        modality_files.append(str(mod_files[0]))
                
                if len(modality_files) == len(modalities):
                    image_paths.append(modality_files)
                    mask_paths.append(mask_path)
    
    return image_paths, mask_paths
