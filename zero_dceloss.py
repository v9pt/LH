"""Zero-DCE specific losses for low-light enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """Spatial consistency loss for Zero-DCE.
    
    Encourages local spatial coherence in the enhanced image.
    """
    
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
    
    def forward(self, enhanced):
        """Compute spatial consistency loss.
        
        Args:
            enhanced: Enhanced images [B, 3, H, W]
            
        Returns:
            Loss scalar
        """
        # Compute differences in 4 directions
        diff_h = torch.pow(enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :], 2)
        diff_w = torch.pow(enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1], 2)
        
        loss = torch.mean(diff_h) + torch.mean(diff_w)
        return loss


class ExposureControlLoss(nn.Module):
    """Exposure control loss for Zero-DCE.
    
    Constrains the enhanced image to have proper exposure (gray level around 0.6).
    """
    
    def __init__(self, patch_size=16, target_gray=0.6):
        super(ExposureControlLoss, self).__init__()
        self.patch_size = patch_size
        self.target = target_gray
        self.pool = nn.AvgPool2d(patch_size)
    
    def forward(self, enhanced):
        """Compute exposure control loss.
        
        Args:
            enhanced: Enhanced images [B, 3, H, W]
            
        Returns:
            Loss scalar
        """
        # Convert to grayscale
        gray = 0.299 * enhanced[:, 0, :, :] + \
               0.587 * enhanced[:, 1, :, :] + \
               0.114 * enhanced[:, 2, :, :]
        gray = gray.unsqueeze(1)
        
        # Average pooling to get local patch exposure
        mean_intensity = self.pool(gray)
        
        # Loss is deviation from target
        loss = torch.mean(torch.pow(mean_intensity - self.target, 2))
        return loss


class ColorConstancyLoss(nn.Module):
    """Color constancy loss for Zero-DCE.
    
    Maintains color balance in the enhanced image.
    """
    
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
    
    def forward(self, enhanced):
        """Compute color constancy loss.
        
        Args:
            enhanced: Enhanced images [B, 3, H, W]
            
        Returns:
            Loss scalar
        """
        # Average intensity per channel
        mean_r = torch.mean(enhanced[:, 0, :, :])
        mean_g = torch.mean(enhanced[:, 1, :, :])
        mean_b = torch.mean(enhanced[:, 2, :, :])
        
        # Deviation between channels
        loss = torch.pow(mean_r - mean_g, 2) + \
               torch.pow(mean_g - mean_b, 2) + \
               torch.pow(mean_b - mean_r, 2)
        
        return loss


class IlluminationSmoothnessLoss(nn.Module):
    """Illumination smoothness loss for Zero-DCE.
    
    Encourages smooth illumination enhancement curves.
    """
    
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()
    
    def forward(self, curve_params):
        """Compute illumination smoothness loss.
        
        Args:
            curve_params: Light curve parameters [B, C*n_iter, H, W]
            
        Returns:
            Loss scalar
        """
        # Compute TV (Total Variation)
        diff_h = torch.pow(curve_params[:, :, 1:, :] - curve_params[:, :, :-1, :], 2)
        diff_w = torch.pow(curve_params[:, :, :, 1:] - curve_params[:, :, :, :-1], 2)
        
        loss = torch.mean(diff_h) + torch.mean(diff_w)
        return loss


class ZeroDCELoss:
    """Combined loss for Zero-DCE training."""
    
    def __init__(self, device='cpu',
                 w_spa=1.0, w_exp=10.0, w_col=5.0, w_tvA=200.0):
        """Initialize Zero-DCE loss.
        
        Args:
            device: Device
            w_spa: Weight for spatial consistency
            w_exp: Weight for exposure control
            w_col: Weight for color constancy
            w_tvA: Weight for illumination smoothness
        """
        self.device = device
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tvA = w_tvA
        
        self.spa_loss = SpatialConsistencyLoss()
        self.exp_loss = ExposureControlLoss()
        self.col_loss = ColorConstancyLoss()
        self.tvA_loss = IlluminationSmoothnessLoss()
    
    def compute(self, enhanced, original):
        """Compute total Zero-DCE loss.
        
        Args:
            enhanced: Enhanced images [B, 3, H, W]
            original: Original images [B, 3, H, W]
            
        Returns:
            Dictionary of losses
        """
        spa = self.spa_loss(enhanced)
        exp = self.exp_loss(enhanced)
        col = self.col_loss(enhanced)
        
        total = self.w_spa * spa + self.w_exp * exp + self.w_col * col
        
        return {
            'total': total,
            'spatial': spa,
            'exposure': exp,
            'color': col
        }
