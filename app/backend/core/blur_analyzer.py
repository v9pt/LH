import cv2
import numpy as np
from typing import Dict

class BlurAnalyzer:
    """Multi-factor blur analyzer using frequency and structural cues.
    
    Implements parts of Paper 5: Hybrid Motion Blur Detection.
    """
    
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def compute_laplacian_variance(self, image: np.ndarray) -> float:
        """Measure focus via the variance of the Laplacian."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return float(laplacian.var())

    def compute_tenengrad(self, image: np.ndarray) -> float:
        """Measure sharpness via Tenengrad (gradient magnitude)."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        return float(np.mean(grad_mag**2))

    def compute_fft_score(self, image: np.ndarray) -> float:
        """Measure blur via High-Frequency / Low-Frequency energy ratio."""
        gray = image
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            
        gray_f = gray.astype(np.float32) / 255.0
        dft = np.fft.fft2(gray_f)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude = np.abs(dft_shift)
        power = magnitude ** 2
        
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius = np.sqrt(x * x + y * y)
        
        # High-frequency: outside central 22%
        hf_mask = radius > min(h, w) * 0.22
        lf_mask = radius <= min(h, w) * 0.22
        
        hf_energy = float(power[hf_mask].mean())
        lf_energy = float(power[lf_mask].mean())
        
        return hf_energy / (hf_energy + lf_energy + 1e-8)

    def compute_edge_density(self, image: np.ndarray) -> float:
        """Measure sharpness via Canny edge density."""
        edges = cv2.Canny(image, 100, 200)
        return float(np.mean(edges > 0))

    def analyze(self, image_rgb: np.ndarray) -> Dict:
        """Compute multi-factor blur severity (0=sharp, 1=blurry)."""
        if len(image_rgb.shape) == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb

        gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # FIX: Denoise before analysis to stop sensor noise from 'faking' sharpness
        gray_denoise = cv2.GaussianBlur(gray, (3, 3), 0)
        
        lap_var = self.compute_laplacian_variance(gray_denoise)
        tenengrad = self.compute_tenengrad(gray)
        fft_ratio = self.compute_fft_score(gray)
        edge_density = self.compute_edge_density(gray)
        
        # Normalise based on empirical ranges for 128x128 faces
        norm_lap = np.clip(np.log1p(lap_var) / np.log1p(900.0), 0, 1)
        norm_ten = np.clip(np.log1p(tenengrad) / np.log1p(12000.0), 0, 1)
        norm_fft = np.clip(fft_ratio / 0.38, 0, 1)
        norm_edge = np.clip(edge_density / 0.18, 0, 1)
        
        sharpness = (
            0.35 * norm_lap +
            0.25 * norm_ten +
            0.20 * norm_fft +
            0.20 * norm_edge
        )
        
        severity = 1.0 - sharpness
        
        return {
            'blur_severity': float(np.clip(severity, 0, 1)),
            'metrics': {
                'laplacian': round(lap_var, 2),
                'tenengrad': round(tenengrad, 2),
                'fft_ratio': round(fft_ratio, 4),
                'edge_density': round(edge_density, 4)
            }
        }
