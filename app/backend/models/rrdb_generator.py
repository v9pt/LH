"""RRDB (Residual-in-Residual Dense Block) Generator for Super-Resolution.
Based on ESRGAN/Real-ESRGAN architecture with facial priors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Dense block with 5 convolutional layers."""

    def __init__(self, nf=64, gc=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.DB1 = DenseBlock(nf, gc)
        self.DB2 = DenseBlock(nf, gc)
        self.DB3 = DenseBlock(nf, gc)

    def forward(self, x):
        out = self.DB1(x)
        out = self.DB2(out)
        out = self.DB3(out)
        return out * 0.2 + x


class RRDBGenerator(nn.Module):
    """RRDB-based Generator for Face Super-Resolution.

    Upscales faces by 4× using RRDB blocks and pixel shuffle.
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        """Initialize RRDB Generator.

        Args:
            in_nc: Input channels (3 for RGB)
            out_nc: Output channels (3 for RGB)
            nf: Number of features
            nb: Number of RRDB blocks
            gc: Growth channels in dense blocks
            scale: Upscaling factor (4 for 4×)
        """
        super(RRDBGenerator, self).__init__()
        self.scale = scale

        # First convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        # RRDB blocks
        self.RRDB_blocks = nn.ModuleList([RRDB(nf, gc) for _ in range(nb)])

        # Middle convolution
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling layers
        if scale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif scale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        # Final layers
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input LR image [B, 3, H, W]

        Returns:
            SR image [B, 3, H*scale, W*scale]
        """
        fea = self.conv_first(x)
        trunk = fea

        # RRDB blocks
        for block in self.RRDB_blocks:
            trunk = block(trunk)

        trunk = self.conv_body(trunk)
        fea = fea + trunk

        # Upsampling
        if self.scale == 4:
            fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
            fea = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        elif self.scale == 2:
            fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))

        # HR conv
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return torch.clamp(out, 0, 1)


class RRDBNet:
    """RRDB Network Wrapper."""

    def __init__(self, device='cpu', scale=4):
        self.device = device
        self.model = RRDBGenerator(scale=scale).to(device)

    def super_resolve(self, image):
        """Super-resolve an image.

        Args:
            image: Input tensor [1, 3, H, W] or [3, H, W]

        Returns:
            SR image tensor
        """
        self.model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            sr = self.model(image.to(self.device))
        return sr

    def load_checkpoint(self, checkpoint_path):
        """Load pretrained weights.

        Handles Real-ESRGAN official checkpoint format which stores
        weights under 'params_ema' key.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        # Try common key names used by different ESRGAN releases in order
        for key in ('params_ema', 'params', 'model_state_dict', 'state_dict'):
            if isinstance(checkpoint, dict) and key in checkpoint:
                checkpoint = checkpoint[key]
                print(f"  Using checkpoint key: '{key}'")
                break
        self.model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded RRDB checkpoint from {checkpoint_path}")

    def save_checkpoint(self, save_path, optimizer=None, epoch=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict()
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
