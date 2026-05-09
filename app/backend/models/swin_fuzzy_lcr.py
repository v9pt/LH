import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFeatureTransform(nn.Module):
    """
    SFT Layer (Spatial Feature Transform).
    Acts as the 'Fuzzy Router' in the architecture.
    Takes the fuzzy condition vector from ANFIS and modulates the SwinIR features.
    """
    def __init__(self, feature_dim, condition_dim=2):
        super(SpatialFeatureTransform, self).__init__()
        # We map the low-dim condition (e.g. [darkness, blur]) to the feature channels
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

    def forward(self, x, condition):
        # x: [B, C, H, W]
        # condition: [B, condition_dim]
        scale = self.mlp_scale(condition).unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        shift = self.mlp_shift(condition).unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        
        # FiLM: Feature-wise Linear Modulation
        out = x * (1 + scale) + shift
        
        # Return the feature map and the scale map (which serves as the Fuzzy Attention Heatmap)
        return out, scale


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
        self.sft = SpatialFeatureTransform(feature_dim, condition_dim=2)
        
        # 3. SwinIR Backbone (Simplified for demonstration, would typically import from models)
        # In a full implementation, this uses shifted-window attention blocks.
        self.swin_blocks = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )
        
        # 4. LCR VQ-Codebook Projection Layer (Connecting to CodeFormer)
        # Maps deep features to the latent space expected by the VQ-GAN
        self.lcr_projection = nn.Conv2d(feature_dim, 256, kernel_size=1)
        
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
        anfis_condition: [B, 2] Fuzzy degradation scores (e.g., [darkness_factor, blur_severity])
        """
        # Extract base features
        feat = self.feat_extract(x)
        
        # Modulate features adaptively based on Fuzzy Logic (ANFIS)
        feat_sft, fuzzy_heatmap = self.sft(feat, anfis_condition)
        
        # Hierarchical Feature Processing (SwinIR)
        deep_feat = self.swin_blocks(feat_sft)
        deep_feat = deep_feat + feat_sft # Residual connection
        
        # Project to Locality Constrained Representation (VQ-Codebook Latent Space)
        lcr_latent = self.lcr_projection(deep_feat)
        
        # Reconstruct high-resolution image
        out = self.reconstruction(lcr_latent)
        
        return out, lcr_latent, fuzzy_heatmap

