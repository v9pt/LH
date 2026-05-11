"""
Robust Blur Analyzer — Multi-Factor Fusion
=========================================

Integrates multiple blur estimation techniques to produce a stable 
severity score [0, 1].

Factors:
1. Laplacian Variance (Spatial)
2. Tenengrad Gradient (Edge)
3. FFT Power Spectrum Slope (Frequency)
4. Edge Density (Structural)
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class BlurAnalyzer:
    def __init__(self, target_size=(128, 128), debug: bool = False):
        self.target_size = target_size
        self.debug = debug

    def compute_laplacian_variance(self, gray):
        """Standard Laplacian variance measure."""
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def compute_tenengrad(self, gray):
        """Sobel-based gradient magnitude (Tenengrad)."""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        return np.mean(mag**2)

    def compute_fft_score(self, gray):
        """Return high-frequency energy ratio in [0, 1]."""
        h, w = gray.shape
        gray_f = gray.astype(np.float32) / 255.0
        gray_f = gray_f - float(gray_f.mean())
        f = np.fft.fft2(gray_f)
        fshift = np.fft.fftshift(f)
        power = np.abs(fshift) ** 2

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius = np.sqrt(x * x + y * y)
        hf_mask = radius > min(h, w) * 0.22
        lf_mask = radius <= min(h, w) * 0.22
        hf_energy = float(power[hf_mask].mean())
        lf_energy = float(power[lf_mask].mean())
        return hf_energy / (hf_energy + lf_energy + 1e-8)

    def compute_edge_density(self, gray):
        """Canny edge density with adaptive thresholds."""
        med = float(np.median(gray))
        lower = int(max(0, 0.66 * med))
        upper = int(min(255, 1.33 * med + 1))
        edges = cv2.Canny(gray.astype(np.uint8), lower, upper)
        return float(np.mean(edges > 0))

    def analyze(self, image: np.ndarray) -> dict:
        """
        Analyze image for blur using multiple methods.
        Returns a dict of scores and a normalized severity [0, 1].
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        if gray.dtype != np.uint8:
            max_val = 1.0 if float(np.nanmax(gray)) <= 1.0 else 255.0
            gray = np.clip(gray.astype(np.float32) / max_val * 255.0, 0, 255).astype(np.uint8)

        gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        lap_var = self.compute_laplacian_variance(gray)
        tenengrad = self.compute_tenengrad(gray)
        fft_ratio = self.compute_fft_score(gray)
        edge_density = self.compute_edge_density(gray)
        
        # Log compression keeps the score useful across dark faces, JPEG input,
        # and synthetic high-contrast test images.
        norm_lap = np.clip(np.log1p(lap_var) / np.log1p(900.0), 0, 1)
        norm_ten = np.clip(np.log1p(tenengrad) / np.log1p(12000.0), 0, 1)
        norm_fft = np.clip(fft_ratio / 0.38, 0, 1)
        norm_edge = np.clip(edge_density / 0.18, 0, 1)
        
        sharpness = (
            norm_lap * 0.38 +
            norm_ten * 0.24 +
            norm_fft * 0.26 +
            norm_edge * 0.12
        )
        
        severity = float(np.clip(1.0 - sharpness, 0.0, 1.0))
        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "blur severity=%.4f lap=%.3f ten=%.3f fft=%.4f edge=%.4f "
                "norms=(%.3f, %.3f, %.3f, %.3f)",
                severity, lap_var, tenengrad, fft_ratio, edge_density,
                norm_lap, norm_ten, norm_fft, norm_edge,
            )
        
        return {
            'blur_severity': severity,
            'laplacian_var': float(lap_var),
            'tenengrad': float(tenengrad),
            'fft_hf_ratio': float(fft_ratio),
            'edge_density': float(edge_density),
            'sharpness_score': float(sharpness),
            'norm_laplacian': float(norm_lap),
            'norm_tenengrad': float(norm_ten),
            'norm_fft': float(norm_fft),
            'norm_edge_density': float(norm_edge),
        }

def detect_blur_severity(image: np.ndarray) -> float:
    """Wrapper for backward compatibility."""
    analyzer = BlurAnalyzer()
    return analyzer.analyze(image)['blur_severity']
