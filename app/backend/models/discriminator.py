import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, normalize=True):
        super(DiscriminatorBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies NxN image patches as real or fake, rather than the whole image.
    Crucial for pushing the NIQE score below 5 (high naturalness) and preventing blur.
    """
    def __init__(self, in_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()

        # Input: [B, 3, H, W]
        self.model = nn.Sequential(
            DiscriminatorBlock(in_channels, features, normalize=False),      # [B, 64, H/2, W/2]
            DiscriminatorBlock(features, features * 2),                      # [B, 128, H/4, W/4]
            DiscriminatorBlock(features * 2, features * 4),                  # [B, 256, H/8, W/8]
            DiscriminatorBlock(features * 4, features * 8, stride=1),        # [B, 512, H/8, W/8]
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)   # [B, 1, H/8-1, W/8-1]
        )

    def forward(self, x):
        return self.model(x)
