"""Loss functions for face super-resolution training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class PixelLoss(nn.Module):
    """L1 Pixel-wise reconstruction loss."""
    
    def __init__(self):
        super(PixelLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, sr, hr):
        """Compute L1 loss between SR and HR images.
        
        Args:
            sr: Super-resolved images [B, 3, H, W]
            hr: High-resolution ground truth [B, 3, H, W]
            
        Returns:
            L1 loss scalar
        """
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    """VGG-19 perceptual loss for texture quality.
    
    Computes feature-level loss using pretrained VGG-19 network.
    """
    
    def __init__(self, device='cpu', layers=None):
        """Initialize perceptual loss.
        
        Args:
            device: 'cpu' or 'cuda'
            layers: List of VGG layers to use (e.g., ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'])
        """
        super(PerceptualLoss, self).__init__()
        self.device = device
        
        # Load pretrained VGG-19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract specific layers
        if layers is None:
            # Use conv layers before pooling: conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
            self.layer_indices = [3, 8, 17, 26, 35]
        else:
            self.layer_indices = layers
        
        self.vgg = vgg
        self.loss = nn.L1Loss()
    
    def extract_features(self, x):
        """Extract VGG features from multiple layers.
        
        Args:
            x: Input images [B, 3, H, W] in range [0, 1]
            
        Returns:
            List of feature tensors
        """
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        
        return features
    
    def forward(self, sr, hr):
        """Compute perceptual loss.
        
        Args:
            sr: Super-resolved images [B, 3, H, W]
            hr: High-resolution ground truth [B, 3, H, W]
            
        Returns:
            Perceptual loss scalar
        """
        sr_features = self.extract_features(sr)
        hr_features = self.extract_features(hr)
        
        loss = 0
        for sr_feat, hr_feat in zip(sr_features, hr_features):
            loss += self.loss(sr_feat, hr_feat)
        
        return loss / len(sr_features)


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training.
    
    Supports both standard GAN and WGAN-GP losses.
    """
    
    def __init__(self, loss_type='wgan-gp'):
        """Initialize adversarial loss.
        
        Args:
            loss_type: 'vanilla', 'lsgan', or 'wgan-gp'
        """
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
    
    def forward(self, pred, is_real):
        """Compute adversarial loss.
        
        Args:
            pred: Discriminator predictions [B, 1, H, W]
            is_real: True for real images, False for fake
            
        Returns:
            Loss scalar
        """
        if self.loss_type == 'wgan-gp':
            # WGAN loss
            if is_real:
                return -torch.mean(pred)
            else:
                return torch.mean(pred)
        else:
            # Standard GAN loss
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)
    
    @staticmethod
    def compute_gradient_penalty(discriminator, real_images, fake_images, device):
        """Compute gradient penalty for WGAN-GP.
        
        Args:
            discriminator: Discriminator network
            real_images: Real images [B, 3, H, W]
            fake_images: Fake images [B, 3, H, W]
            device: Device
            
        Returns:
            Gradient penalty scalar
        """
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        
        # Interpolate between real and fake
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        
        d_interpolates = discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
