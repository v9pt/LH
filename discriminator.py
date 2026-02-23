"""PatchGAN Discriminator for adversarial training."""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for realistic texture generation.
    
    Outputs a matrix of real/fake predictions for overlapping patches.
    """
    
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        """Initialize discriminator.
        
        Args:
            in_channels: Input channels (3 for RGB)
            ndf: Base number of discriminator filters
            n_layers: Number of layers
        """
        super(PatchGANDiscriminator, self).__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                   kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                               kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Patch predictions [B, 1, H', W']
        """
        return self.model(x)
