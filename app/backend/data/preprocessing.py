
"""Image preprocessing and degradation pipeline.

Creates low-quality inputs from high-resolution faces by:
1. Downsampling (4x)
2. Adding blur
3. Adding noise
4. Darkening (gamma correction)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import random


class ImageDegrader:
    """Image degradation pipeline for creating training pairs."""
    
    def __init__(self, scale=4, blur_kernel_size=7, noise_std=0.01,
                 gamma_range=(2.5, 5.0)):
        """Initialize degrader.
        
        Args:
            scale: Downsampling scale factor
            blur_kernel_size: Gaussian blur kernel size
            noise_std: Gaussian noise standard deviation
            gamma_range: Range for gamma correction (darkening)
        """
        self.scale = scale
        self.blur_kernel_size = blur_kernel_size
        self.noise_std = noise_std
        self.gamma_range = gamma_range
    
    def degrade(self, hr_image):
        """Create low-quality image from high-resolution input.
        
        Args:
            hr_image: High-res image tensor [3, H, W] in range [0, 1]
                     or numpy array [H, W, 3] in range [0, 255]
            
        Returns:
            Degraded LR image in same format as input
        """
        is_tensor = isinstance(hr_image, torch.Tensor)
        
        if is_tensor:
            # Convert to numpy for OpenCV operations
            image = hr_image.permute(1, 2, 0).cpu().numpy() * 255
            image = image.astype(np.uint8)
        else:
            image = hr_image.copy()
        
        # 1. Downsample
        h, w = image.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        lr_image = cv2.resize(image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Add blur
        if self.blur_kernel_size > 0:
            lr_image = cv2.GaussianBlur(
                lr_image,
                (self.blur_kernel_size, self.blur_kernel_size),
                0
            )
        
        # 3. Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std * 255, lr_image.shape)
            lr_image = np.clip(lr_image + noise, 0, 255).astype(np.uint8)
        
        # 4. Darken (gamma correction)
        gamma = random.uniform(*self.gamma_range)
        lr_image = self._adjust_gamma(lr_image, gamma)
        
        # Convert back to original format
        if is_tensor:
            lr_tensor = torch.from_numpy(lr_image).float() / 255.0
            lr_tensor = lr_tensor.permute(2, 0, 1)
            return lr_tensor
        else:
            return lr_image
    
    @staticmethod
    def _adjust_gamma(image, gamma):
        """Apply gamma correction to darken image.
        
        Args:
            image: Image array [H, W, 3] in [0, 255]
            gamma: Gamma value (>1 darkens)
            
        Returns:
            Gamma-corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        return cv2.LUT(image, table)
    
    def create_paired_data(self, hr_image):
        """Create LR-HR paired data.
        
        Args:
            hr_image: High-resolution image
            
        Returns:
            Tuple of (lr_image, hr_image)
        """
        lr_image = self.degrade(hr_image)
        return lr_image, hr_image


class FaceAligner:
    """Face alignment and cropping utilities."""
    
    @staticmethod
    def crop_face(image, bbox, expand_ratio=0.2):
        """Crop face region with expansion.
        
        Args:
            image: Input image [H, W, 3]
            bbox: Bounding box [x1, y1, x2, y2]
            expand_ratio: Expansion ratio around bbox
            
        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Expand bbox
        x1 = max(0, int(x1 - w * expand_ratio))
        y1 = max(0, int(y1 - h * expand_ratio))
        x2 = min(image.shape[1], int(x2 + w * expand_ratio))
        y2 = min(image.shape[0], int(y2 + h * expand_ratio))
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def resize_and_pad(image, target_size=(128, 128)):
        """Resize image to target size with padding.
        
        Args:
            image: Input image [H, W, 3]
            target_size: Target (height, width)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Compute scale to fit
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded


def normalize_image(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Normalize image tensor.
    
    Args:
        image: Image tensor [3, H, W] in range [0, 1]
        mean: Mean values per channel
        std: Std values per channel
        
    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (image - mean) / std


def denormalize_image(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Denormalize image tensor.
    
    Args:
        image: Normalized image tensor [3, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized tensor in range [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    return torch.clamp(image * std + mean, 0, 1)


