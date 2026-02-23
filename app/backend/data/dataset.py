"""Dataset classes for face super-resolution."""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class FaceSRDataset(Dataset):
    """Face Super-Resolution dataset.
    
    Loads high-resolution face images and creates LR counterparts.
    """
    
    def __init__(self, image_dir, image_degrader, hr_size=128, 
                 augment=True, max_images=None):
        """Initialize dataset.
        
        Args:
            image_dir: Directory containing HR face images
            image_degrader: ImageDegrader instance
            hr_size: Size of HR images (will be resized)
            augment: Whether to apply data augmentation
            max_images: Maximum number of images to load (None for all)
        """
        self.image_dir = Path(image_dir)
        self.degrader = image_degrader
        self.hr_size = hr_size
        
        # Get image paths
        self.image_paths = self._get_image_paths(max_images)
        
        # Augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ])
        else:
            self.transform = None
    
    def _get_image_paths(self, max_images=None):
        """Get list of image paths."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = []
        
        for ext in extensions:
            paths.extend(self.image_dir.glob(f'*{ext}'))
            paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        paths = sorted(paths)
        
        if max_images:
            paths = paths[:max_images]
        
        return paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get item.
        
        Returns:
            Dictionary with 'lr' and 'hr' images as tensors [3, H, W]
        """
        # Load HR image
        img_path = self.image_paths[idx]
        hr_image = cv2.imread(str(img_path))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        # Resize to target HR size
        hr_image = cv2.resize(hr_image, (self.hr_size, self.hr_size),
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=hr_image)
            hr_image = augmented['image']
        
        # Convert to tensor [0, 1]
        hr_tensor = torch.from_numpy(hr_image).float() / 255.0
        hr_tensor = hr_tensor.permute(2, 0, 1)  # [3, H, W]
        
        # Create LR image
        lr_tensor = self.degrader.degrade(hr_tensor)
        
        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'path': str(img_path)
        }


class UnpairedDarkDataset(Dataset):
    """Unpaired dark images for Zero-DCE pretraining.
    
    Only loads dark images, no paired HR needed.
    """
    
    def __init__(self, image_dir, image_size=128, max_images=None):
        """Initialize dataset.
        
        Args:
            image_dir: Directory containing face images
            image_size: Size to resize images
            max_images: Maximum number of images
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = self._get_image_paths(max_images)
    
    def _get_image_paths(self, max_images=None):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        paths = []
        for ext in extensions:
            paths.extend(self.image_dir.glob(f'*{ext}'))
        paths = sorted(paths)
        if max_images:
            paths = paths[:max_images]
        return paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        
        # Apply random darkening
        gamma = np.random.uniform(2.5, 5.0)
        tensor = torch.pow(tensor, gamma)
        
        return {
            'dark': tensor,
            'path': str(img_path)
        }


def create_dataloaders(train_dir, val_dir, image_degrader,
                      batch_size=16, num_workers=4, hr_size=128,
                      max_train=None, max_val=None):
    """Create train and validation dataloaders.
    
    Args:
        train_dir: Training images directory
        val_dir: Validation images directory
        image_degrader: ImageDegrader instance
        batch_size: Batch size
        num_workers: Number of dataloader workers
        hr_size: HR image size
        max_train: Max training images
        max_val: Max validation images
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = FaceSRDataset(
        train_dir, image_degrader, hr_size=hr_size,
        augment=True, max_images=max_train
    )
    
    val_dataset = FaceSRDataset(
        val_dir, image_degrader, hr_size=hr_size,
        augment=False, max_images=max_val
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


