import torch
import numpy as np
import cv2

def normalize_image_pipeline(img, target_format='numpy', target_dtype='float32', target_size=None, color_conv=None):
    """
    Central utility for image format standardization.
    Supports standardizing images between numpy (HWC) and torch (CHW).
    
    Args:
        img: Input image (numpy array or torch tensor)
        target_format: 'numpy' or 'torch'
        target_dtype: 'float32' or 'uint8'
        target_size: (H, W) tuple or None
        color_conv: cv2.COLOR_... code or None
    """
    # 1. To Numpy Base
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] < img.shape[2]: # CHW -> HWC
            img = img.transpose(1, 2, 0)
    
    # 2. Color Conversion
    if color_conv is not None:
        img = cv2.cvtColor(img, color_conv)
        
    # 3. Resize
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
    # 4. Range and Dtype
    if target_dtype == 'float32':
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0, 1).astype(np.float32)
    elif target_dtype == 'uint8':
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
            
    # 5. Final Format
    if target_format == 'torch':
        if img.ndim == 3:
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        else:
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            
    return img
