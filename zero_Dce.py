"""Zero-DCE (Zero-Reference Deep Curve Estimation) Network
Non-reference low-light enhancement using light curve estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCE_Net(nn.Module):
    """Deep Curve Estimation Network for low-light enhancement.
    
    Uses 7 convolutional layers to estimate higher-order curves for pixel-wise
    enhancement without paired training data.
    """
    
    def __init__(self, n_iterations=8):
        super(DCE_Net, self).__init__()
        self.n_iterations = n_iterations
        
        # 7 convolutional layers (Conv-ReLU structure)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # Skip connection
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # Skip connection
        
        # Final layer outputs curve parameters (8 iterations × 3 channels)
        self.conv7 = nn.Conv2d(32, 3 * n_iterations, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Enhanced image tensor [B, 3, H, W]
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # Skip connections
        x5 = self.relu(self.conv5(torch.cat([x4, x3], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x5, x2], dim=1)))
        
        # Curve parameter estimation
        curve_params = self.tanh(self.conv7(x6))  # [-1, 1]
        
        # Apply light enhancement curve
        enhanced = x
        for i in range(self.n_iterations):
            enhanced = enhanced + curve_params[:, i*3:(i+1)*3, :, :] * (torch.pow(enhanced, 2) - enhanced)
        
        return torch.clamp(enhanced, 0, 1)


class ZeroDCE:
    """Zero-DCE Model Wrapper with training and inference utilities."""
    
    def __init__(self, device='cpu', n_iterations=8):
        self.device = device
        self.model = DCE_Net(n_iterations=n_iterations).to(device)
        
    def enhance(self, image):
        """Enhance a single image.
        
        Args:
            image: Input tensor [1, 3, H, W] or [3, H, W] in range [0, 1]
            
        Returns:
            Enhanced image tensor
        """
        self.model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            enhanced = self.model(image.to(self.device))
        return enhanced
    
    def load_checkpoint(self, checkpoint_path):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded Zero-DCE checkpoint from {checkpoint_path}")
    
    def save_checkpoint(self, save_path, optimizer=None, epoch=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'n_iterations': self.model.n_iterations
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
