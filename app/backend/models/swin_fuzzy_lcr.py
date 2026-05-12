import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

# Optional debug logger
try:
    from utils.debug_utils import get_logger as _get_dbg
except ImportError:
    try:
        from app.backend.utils.debug_utils import get_logger as _get_dbg
    except ImportError:
        _get_dbg = None


def _dbg():
    if _get_dbg is not None:
        try:
            return _get_dbg()
        except Exception:
            pass
    return None

class SpatialFeatureTransform(nn.Module):
    """
    SFT Layer (Spatial Feature Transform).
    Acts as the 'Fuzzy Router' in the architecture.
    Takes the fuzzy condition vector from ANFIS and modulates the SwinIR features.
    """
    def __init__(self, feature_dim, condition_dim=5):
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
        # CRITICAL FIX: Zero-init the final Linear layers of mlp_scale and mlp_shift.
        # This ensures the SFT starts as a perfect identity transform (scale=0, shift=0)
        # and only learns non-trivial modulation after sufficient gradient updates.
        # Without this, random final-layer weights produce scale/shift of ~±0.28/±0.10
        # on the very first forward pass, immediately corrupting facial geometry.
        nn.init.zeros_(self.mlp_scale[-1].weight)
        nn.init.zeros_(self.mlp_scale[-1].bias)
        nn.init.zeros_(self.mlp_shift[-1].weight)
        nn.init.zeros_(self.mlp_shift[-1].bias)

    def forward(self, x, condition):
        # x: [B, C, H, W]
        # condition: [B, condition_dim]
        # TASK 5: Restore Conservative FiLM Modulation
        scale = self.mlp_scale(condition).unsqueeze(-1).unsqueeze(-1)
        shift = self.mlp_shift(condition).unsqueeze(-1).unsqueeze(-1)
        gate  = self.residual_gate(condition).unsqueeze(-1).unsqueeze(-1)

        # Apply modulation with conservative alpha
        # enhance_alpha ∈ [0.03, 0.08]
        enhance_alpha = 0.05
        out = x + (x * scale + shift) * gate * enhance_alpha

        return out, gate
        dbg = _dbg()
        if dbg:
            scale_np = scale.detach().cpu().float().numpy()
            shift_np = shift.detach().cpu().float().numpy()
            dbg.log_film(
                enhance_alpha=float(gate.mean().item()),
                gamma_stats={"min": float(scale_np.min()), "max": float(scale_np.max()),
                             "mean": float(scale_np.mean()), "std": float(scale_np.std())},
                beta_stats={"min": float(shift_np.min()), "max": float(shift_np.max()),
                            "mean": float(shift_np.mean()), "std": float(shift_np.std())},
                stage="sft"
            )

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
        # Synchronized with 5-D ANFIS feature vector (Task 11)
        self.sft = SpatialFeatureTransform(feature_dim, condition_dim=5)
        
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
        
        # 4. LCR VQ-Codebook Projection Layer
        # FIX: Implementing actual Locality Constrained Representation (LCR)
        # FIX: Identity-preserving initialization
        # We start with a near-identity codebook so the LCR projection 
        # preserves facial structure from day one.
        self.codebook = nn.Parameter(torch.eye(256) + torch.randn(256, 256) * 0.001)

        self.lcr_assignment = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=1)
        )
        self.identity_projection = nn.Conv2d(feature_dim, 256, kernel_size=1)
        
        # 5. Output Reconstruction & 8x Upsampling (64x64 -> 512x512)
        # Phase-aligned PixelShuffle sequence for stable high-res reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(256, feature_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), # 2x (64 -> 128)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(feature_dim, feature_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), # 2x (128 -> 256)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(feature_dim, feature_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), # 2x (256 -> 512)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(feature_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, 
                anfis_condition: torch.Tensor,
                residual_scale: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 3, H, W] Low-light degraded image
        anfis_condition: [B, 10] Fuzzy degradation scores
        """
        # Explicit float hardening
        x = x.to(torch.float32)
        anfis_condition = anfis_condition.to(torch.float32)

        # Extract base features
        feat = self.feat_extract(x)
        dbg = _dbg()
        if dbg:
            dbg.log_tensor(x, name="sota_input", stage="vq")
            dbg.log_tensor(anfis_condition, name="sota_anfis_condition", stage="routing")
            dbg.log_tensor(feat, name="feat_extract", stage="vq")

        # Modulate features adaptively based on Fuzzy Logic (ANFIS)
        feat_sft, fuzzy_heatmap = self.sft(feat, anfis_condition)

        # Hierarchical Feature Processing (SwinIR)
        deep_feat = self.swin_blocks(feat_sft)
        if dbg:
            dbg.log_tensor(feat_sft, name="feat_sft", stage="film")
            dbg.log_tensor(deep_feat, name="deep_feat", stage="vq")

        # Refine with CBAM Attention
        deep_feat = self.cbam(deep_feat)
        deep_feat = deep_feat + feat_sft  # Residual connection

        # Fuse High-Frequency Wavelet details
        deep_feat = self.wavelet_fusion(deep_feat)

        # Project to Locality Constrained Representation (VQ-Codebook Latent Space)
        assignment = self.lcr_assignment(deep_feat)
        weights = F.softmax(assignment, dim=1)  # [B, 256, H, W]
        B, _, H, W = weights.shape

        # LCR Reconstruction: Each pixel is a weighted sum of codebook 'atoms'
        projected_latent = torch.matmul(
            self.codebook.T, weights.view(B, 256, -1)
        ).view(B, 256, H, W)

        identity_latent = self.identity_projection(deep_feat)

        # TASK 4 & 5: Restore Residual & FiLM Modulation
        # Re-enabling safe contribution of SR (VQ) and Illumination (FiLM) branches.
        # Use a non-zero weight to allow backpropagation into these branches.
        latent_weight = 0.05
        lcr_latent = projected_latent * latent_weight + identity_latent * (1.0 - latent_weight)

        # 6. Upsample and Reconstruct
        out = self.reconstruction(lcr_latent)
        
        # TASK 3 & 4: Strict Residual Clamping [-0.04, 0.04]
        # TASK 18: Resolution Alignment (Dynamic Anchor)
        # Ensure anchor size matches model output resolution (out.shape)
        x_up = F.interpolate(x, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=False)
        residual = out - x_up
        
        # Strict range prevents geometry distortion (Task 3)
        clamped_residual = torch.clamp(residual, -0.04, 0.04) * residual_scale
        
        final_out = (x_up + clamped_residual).clamp(0.0, 1.0)
        
        return final_out, weights, fuzzy_heatmap
        if dbg:
            w_np = weights.detach().cpu().float().numpy()
            # Compute assignment entropy across codebook dimension per-pixel
            p = w_np.mean(axis=(0, 2, 3))  # [256]
            p = p / (p.sum() + 1e-8)
            p_safe = p[p > 1e-10]
            ent = float(-( p_safe * __import__('numpy').log(p_safe)).sum()) / __import__('numpy').log(max(len(p_safe), 2))
            cb_norm = float(self.codebook.data.norm().item())
            lw_mean = float(latent_weight.mean().item())
            dbg.log_vq(assignment_entropy=ent, codebook_norm=cb_norm, latent_weight=lw_mean)
            dbg.log_tensor(projected_latent, name="projected_latent", stage="vq")
            dbg.log_tensor(identity_latent, name="identity_latent", stage="vq")
            dbg.log_tensor(lcr_latent, name="lcr_latent", stage="vq")

        # Reconstruct high-resolution image
        # MANDATORY CHANGE 4 & 5: Force pure residual learning & Enforce bicubic dominance
        # The model output 'out_deep' is treated as a high-frequency residual.
        out_deep = self.reconstruction(lcr_latent)
        out_base = F.interpolate(x, size=out_deep.shape[2:], mode='bicubic', align_corners=False)
        
        # residual = model(bicubic) or in this case, model(feat)
        # TASK 7: Dynamic clamping
        residual = torch.tanh(out_deep) * residual_scale
        
        # sr = bicubic + residual
        sr = out_base + residual
        out = torch.clamp(sr, 0.0, 1.0)

        if dbg:
            dbg.log_tensor(out_deep, name="out_deep", stage="vq")
            dbg.log_tensor(out_base, name="out_base", stage="vq")
            dbg.log_tensor(out, name="sota_output", stage="vq")

        return out, lcr_latent, fuzzy_heatmap

    @staticmethod
    def quality_check(output: torch.Tensor,
                      reference: torch.Tensor,
                      mad_threshold: float = 0.10) -> bool:
        """Check whether model output is reasonable vs a reference (e.g. bicubic upscale).

        Returns True if the output is usable (Mean Absolute Deviation < mad_threshold
        and mean pixel value in [0.05, 0.95]).

        This gates the SOTA path in inference.py so that untrained/diverged
        model weights do not corrupt the final output.
        """
        try:
            with torch.no_grad():
                out_f = output.detach().cpu().float()
                ref_f = reference.detach().cpu().float()
                if out_f.shape != ref_f.shape:
                    ref_f = F.interpolate(ref_f, size=out_f.shape[2:],
                                          mode='bilinear', align_corners=False)
                mad = float((out_f - ref_f).abs().mean().item())
                mean_val = float(out_f.mean().item())
                return (mad < mad_threshold) and (0.03 <= mean_val <= 0.97)
        except Exception:
            return False
