import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFeatureTransform(nn.Module):
    """
    SFT Layer (Spatial Feature Transform).
    Acts as the 'Fuzzy Router' in the architecture.
    Takes the fuzzy condition vector from ANFIS and modulates the SwinIR features.
    """
    def __init__(self, feature_dim, condition_dim=10): # Updated to 10-D ANFIS
        super(SpatialFeatureTransform, self).__init__()
        self.mlp_scale = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim)
        )
        self.mlp_shift = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim)
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        # x: [B, C, H, W]
        # condition: [B, condition_dim]
        
        # Explicit float hardening
        condition = condition.float()
        
        # Conservative bounded FiLM. The previous +/-50% scaling was strong
        # enough to move identity embeddings before reconstruction.
        scale = torch.tanh(self.mlp_scale(condition)).unsqueeze(-1).unsqueeze(-1) * 0.15
        shift = torch.tanh(self.mlp_shift(condition)).unsqueeze(-1).unsqueeze(-1) * 0.05
        gate = self.residual_gate(condition).unsqueeze(-1).unsqueeze(-1) * 0.35

        modulated = x * (1.0 + scale) + shift
        out = x + gate * (modulated - x)
        
        # Return the feature map and the scale map (which serves as the Fuzzy Attention Heatmap)
        return out, scale

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module for refining facial features."""
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class WaveletFusionBlock(nn.Module):
    """
    Fuses high-frequency details using a simplified Haar-like feature decomposition.
    """
    def __init__(self, channels):
        super(WaveletFusionBlock, self).__init__()
        # High-frequency extractor (edge-like)
        self.hf_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        # Low-frequency extractor (smooth-like)
        self.lf_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        
        # Initialize with Haar-like approximations (Laplacian/Gaussian)
        # We let them be learnable but initialize to encourage frequency separation
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        lf = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        hf = x - lf
        
        # Process frequency bands independently
        lf_feat = self.lf_conv(lf)
        hf_feat = self.hf_conv(hf)
        
        # Fuse
        fused = self.fusion(torch.cat([lf_feat, hf_feat], dim=1))
        return fused + x # Residual connection


class SwinFuzzyLCR(nn.Module):
    """
    The Ultimate SOTA Hybrid Architecture:
    ZeroDCE (Features) -> SFT Modulated SwinIR -> CodeFormer (VQ-Codebook LCR)
    """
    def __init__(self, in_channels=3, out_channels=3, feature_dim=64):
        super(SwinFuzzyLCR, self).__init__()
        
        # 1. Feature Extraction (Simulating ZeroDCE initial convolution)
        self.feat_extract = nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        
        # 2. Spatial Feature Transform (Fuzzy Modulation)
        # Synchronized with 10-D ANFIS feature vector
        self.sft = SpatialFeatureTransform(feature_dim, condition_dim=10)
        
        # 3. SwinIR Backbone (Simplified for demonstration, would typically import from models)
        # In a full implementation, this uses shifted-window attention blocks.
        self.swin_blocks = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )
        
        # NEW: CBAM Attention to refine facial structures
        self.cbam = CBAMBlock(feature_dim)
        
        # NEW: Wavelet-Frequency Fusion for High-Frequency (texture/pores) recovery
        self.wavelet_fusion = WaveletFusionBlock(feature_dim)
        
        # 4. LCR VQ-Codebook Projection Layer (Connecting to CodeFormer latent space)
        # Modernized locality constraints
        self.lcr_projection = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=1)
        )
        self.identity_projection = nn.Conv2d(feature_dim, 256, kernel_size=1)
        
        # 5. Output Reconstruction & Upsampling (4x)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(256, feature_dim * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(4), # 4x upscaling: feature_dim*16 -> feature_dim, H*4, W*4
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, anfis_condition):
        """
        x: [B, 3, H, W] Low-light degraded image
        anfis_condition: [B, 10] Fuzzy degradation scores
        """
        # Explicit float hardening
        x = x.to(torch.float32)
        anfis_condition = anfis_condition.to(torch.float32)
        
        # Extract base features
        feat = self.feat_extract(x)
        
        # Modulate features adaptively based on Fuzzy Logic (ANFIS)
        feat_sft, fuzzy_heatmap = self.sft(feat, anfis_condition)
        
        # Hierarchical Feature Processing (SwinIR)
        deep_feat = self.swin_blocks(feat_sft)
        
        # Refine with CBAM Attention
        deep_feat = self.cbam(deep_feat)
        deep_feat = deep_feat + feat_sft # Residual connection
        
        # Fuse High-Frequency Wavelet details
        deep_feat = self.wavelet_fusion(deep_feat)
        
        # Project to Locality Constrained Representation (VQ-Codebook Latent Space)
        projected_latent = self.lcr_projection(deep_feat)
        identity_latent = self.identity_projection(deep_feat)
        
        # Adaptive Residual Gating: 
        # Clean images (low darkness/blur) rely more on input structure
        # High degradation images rely more on deep processing
        darkness = anfis_condition[:, 0:1]
        blur = anfis_condition[:, 9:10] if anfis_condition.shape[1] > 9 else anfis_condition.mean(dim=1, keepdim=True)
        degradation = torch.clamp(0.65 * darkness + 0.35 * blur, 0.0, 1.0)
        latent_weight = torch.clamp(0.12 + 0.38 * degradation, 0.12, 0.50).unsqueeze(-1).unsqueeze(-1)
        lcr_latent = latent_weight * projected_latent + (1.0 - latent_weight) * identity_latent

        gate = torch.clamp(0.18 + 0.42 * degradation, 0.18, 0.60).unsqueeze(-1).unsqueeze(-1)
        
        # Reconstruct high-resolution image
        out_deep = self.reconstruction(lcr_latent)
        
        # Identity Preservation: Blend deep reconstruction with bicubic/input
        # This ensures that even if LCR fails, the identity structure remains.
        out_base = F.interpolate(x, size=out_deep.shape[2:], mode='bilinear', align_corners=True)
        out = gate * out_deep + (1.0 - gate) * out_base
        out = torch.clamp(out, 0.0, 1.0)
        
        return out, lcr_latent, fuzzy_heatmap
