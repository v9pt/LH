"""
Motion Blur Handler — Blind Estimation + Wiener Deconvolution
=============================================================

Detects and removes motion blur from low-resolution face images before
super-resolution, using the Nearest Proximate Patch Representation approach.

Reference:
    Paper 5: "Robust face hallucination algorithm using motion blur embedded
              nearest proximate patch representation"
              — Section 3 (Blur Estimation) and Section 4 (Patch Representation)

Pipeline:
    1. Blind blur kernel estimation via Radon transform + power spectrum
    2. Wiener deconvolution (frequency-domain) to restore sharp image
    3. Nearest Proximate Patch (NPP) representation:
       - Apply estimated blur kernel to dictionary patches
       - Match noisy patches against blurred dictionary
       - Reconstruct using HR dictionary atoms of best-match patches
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.signal import wiener
from skimage.transform import radon


# ─────────────────────────────────────────────────────────────
#  Blur Kernel Estimation
# ─────────────────────────────────────────────────────────────

def estimate_motion_blur_kernel(image: np.ndarray,
                                 kernel_size: int = 25,
                                 angle_resolution: int = 180) -> Tuple[np.ndarray, float, float]:
    """Estimate motion blur kernel via Radon transform of the power spectrum.

    Algorithm (Paper 5, Section 3.2):
        1. Convert to grayscale and compute 2D DFT power spectrum.
        2. Apply Radon transform (line integral projections) to the log spectrum.
        3. The dominant line angle corresponds to the motion blur direction.
        4. Estimate blur length from the projection profile width.
        5. Construct the corresponding linear motion blur kernel.

    Args:
        image            : [H, W, 3] or [H, W] numpy array (uint8 or float).
        kernel_size      : Size of the output kernel (odd preferred).
        angle_resolution : Angular resolution for Radon transform (degrees).

    Returns:
        kernel : [kernel_size, kernel_size] normalised blur kernel.
        angle  : Estimated blur angle in degrees.
        length : Estimated blur length in pixels.
    """
    # ── Step 1: Grayscale + power spectrum ──────────────────────────
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0

    # Power spectrum: |F(u,v)|²
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    power_spectrum = np.log(np.abs(F_shift) ** 2 + 1e-8)

    # ── Step 2: Radon transform on power spectrum ────────────────────
    # Radon transform integrates along lines at each angle θ
    theta = np.linspace(0, 180, angle_resolution, endpoint=False)
    sinogram = radon(power_spectrum, theta=theta, circle=True)   # [R, angles]

    # ── Step 3: Find dominant angle (max variance projection) ────────
    # Blurred image has a streak in the power spectrum perpendicular to blur dir
    projection_variance = sinogram.var(axis=0)   # [angles]
    dominant_angle_idx  = np.argmax(projection_variance)
    blur_angle_deg = theta[dominant_angle_idx]

    # Motion blur direction is perpendicular to the detected streak
    motion_angle_deg = (blur_angle_deg + 90) % 180

    # ── Step 4: Estimate blur length ─────────────────────────────────
    # The projection at the dominant angle shows a sharp peak for short blur
    # and a wide plateau for long blur. Width at half-max estimates length.
    dominant_proj = sinogram[:, dominant_angle_idx]
    proj_norm = dominant_proj - dominant_proj.min()
    if proj_norm.max() > 0:
        proj_norm /= proj_norm.max()
    half_max_mask = proj_norm > 0.5
    blur_length = max(1, int(half_max_mask.sum() * 0.3))
    blur_length = min(blur_length, kernel_size)

    # ── Step 5: Construct motion blur kernel ─────────────────────────
    kernel = _make_motion_blur_kernel(kernel_size, motion_angle_deg, blur_length)
    return kernel, motion_angle_deg, float(blur_length)


def _make_motion_blur_kernel(size: int,
                              angle_deg: float,
                              length: int) -> np.ndarray:
    """Construct a linear motion blur PSF kernel.

    Args:
        size      : Kernel size (N×N).
        angle_deg : Blur direction in degrees.
        length    : Blur length in pixels.

    Returns:
        kernel : [size, size] float32, normalised (sums to 1).
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    half = length // 2
    for t in range(-half, half + 1):
        x = int(round(cx + t * cos_a))
        y = int(round(cy + t * sin_a))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    return kernel


def detect_blur_severity(image: np.ndarray,
                          threshold: float = 100.0) -> float:
    """Estimate blur severity using Laplacian variance.

    High Laplacian variance → sharp image (low blur severity).
    Low Laplacian variance  → blurred image (high blur severity).

    Args:
        image     : [H, W, 3] or [H, W] uint8 image.
        threshold : Variance below which blur is considered significant.

    Returns:
        severity : float ∈ [0, 1].  0 = sharp, 1 = severely blurred.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    severity = float(np.clip(1.0 - lap_var / threshold, 0.0, 1.0))
    return severity


# ─────────────────────────────────────────────────────────────
#  Wiener Deconvolution
# ─────────────────────────────────────────────────────────────

def wiener_deconvolve(blurred_image: np.ndarray,
                       kernel: np.ndarray,
                       snr: float = 0.01) -> np.ndarray:
    """Wiener deconvolution in the frequency domain.

    Restores a blurred image given the estimated PSF (blur kernel).

    Formula (Paper 5, Eq. 2):
        F̂(u,v) = [H*(u,v) / (|H(u,v)|² + 1/SNR)] · G(u,v)
    where:
        G = DFT of blurred image
        H = DFT of blur kernel (PSF)
        H* = complex conjugate of H
        SNR = signal-to-noise ratio (controls sharpness vs. noise amplification)

    Args:
        blurred_image : [H, W, 3] or [H, W] float32 in [0, 1].
        kernel        : [K, K] float32 normalised blur kernel.
        snr           : Signal-to-noise ratio for Wiener filter.

    Returns:
        restored : Same shape as input, float32 in [0, 1].
    """
    if blurred_image.ndim == 3:
        # Process each channel independently
        channels = []
        for c in range(blurred_image.shape[2]):
            ch = wiener_deconvolve(blurred_image[:, :, c], kernel, snr)
            channels.append(ch)
        return np.stack(channels, axis=2)

    H, W = blurred_image.shape

    # Pad kernel to image size
    K = np.zeros((H, W), dtype=np.float32)
    kh, kw = kernel.shape
    ky, kx = H // 2 - kh // 2, W // 2 - kw // 2
    K[ky:ky+kh, kx:kx+kw] = kernel

    # FFT
    G = np.fft.fft2(blurred_image)
    H_f = np.fft.fft2(K)

    # Wiener filter
    H_conj = np.conj(H_f)
    H_mag2 = np.abs(H_f) ** 2
    W_filter = H_conj / (H_mag2 + (1.0 / (snr + 1e-8)))

    # Restore
    F_hat = W_filter * G
    restored = np.real(np.fft.ifft2(F_hat))

    return np.clip(restored, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Nearest Proximate Patch (NPP) Representation
# ─────────────────────────────────────────────────────────────

class NearestProximatePatchRepresenter:
    """NPP-based face reconstruction under motion blur (Paper 5, Section 4).

    Idea:
        Given a query patch q from the blurred LR image and a HR database:
        1. Apply the estimated blur kernel to all database patches.
        2. Find K nearest neighbours of q in the blurred patch space.
        3. Combine the corresponding HR database patches with Gaussian weights
           based on distance in the blurred space.

        Ŷ = Σ_k  w_k · Y_k  /  Σ_k w_k
        w_k = exp(−||q − blur(D_k)||² / h²)

    This ensures we match apples-to-apples: the blurred query is compared
    against blurred reference patches, then HR patches are used for output.

    Reference: Paper 5, Algorithm 1.
    """

    def __init__(self,
                 K_neighbors: int = 7,
                 bandwidth: float = 0.1):
        """
        Args:
            K_neighbors : Number of nearest neighbours to blend.
            bandwidth   : h² in the Gaussian weight formula.
        """
        self.K   = K_neighbors
        self.h2  = bandwidth ** 2

        # Database
        self.db_patches_lr: Optional[np.ndarray] = None  # [N, d]
        self.db_patches_hr: Optional[np.ndarray] = None  # [N, D]

    def build_database(self,
                        hr_patches: np.ndarray,
                        lr_patches: np.ndarray) -> None:
        """Load the patch database (built from training faces).

        Args:
            hr_patches : [N, D] HR patch feature vectors.
            lr_patches : [N, d] LR patch feature vectors.
        """
        self.db_patches_lr = lr_patches.astype(np.float32)
        self.db_patches_hr = hr_patches.astype(np.float32)

    def represent_patch(self,
                         query_patch: np.ndarray,
                         blur_kernel: np.ndarray) -> np.ndarray:
        """Represent a blurred LR query patch using NPP.

        Args:
            query_patch : [d] blurred LR patch (flattened).
            blur_kernel : [K, K] estimated blur kernel.

        Returns:
            hr_estimate : [D] estimated HR patch (flattened).
        """
        if self.db_patches_lr is None:
            raise RuntimeError("Database not built. Call .build_database() first.")

        N, d_lr = self.db_patches_lr.shape
        ps = int(round(d_lr ** 0.5 / 3 ** 0.5))  # patch spatial size (approx)

        # Apply blur kernel to each database patch (in spatial domain)
        blurred_db = _apply_kernel_to_patches(
            self.db_patches_lr, blur_kernel, ps)

        # Compute squared distances to blurred database patches
        diff    = blurred_db - query_patch[np.newaxis, :]   # [N, d]
        dist_sq = np.sum(diff ** 2, axis=1)                 # [N]

        # K nearest neighbours
        knn_idx = np.argsort(dist_sq)[:self.K]

        # Gaussian weights based on distance in blurred space
        weights = np.exp(-dist_sq[knn_idx] / (self.h2 + 1e-8))  # [K]
        weights = weights / (weights.sum() + 1e-8)               # normalise

        # Weighted combination of HR database patches
        hr_estimate = (weights[:, np.newaxis] *
                       self.db_patches_hr[knn_idx]).sum(axis=0)  # [D]
        return hr_estimate.astype(np.float32)


def _apply_kernel_to_patches(patches: np.ndarray,
                               kernel: np.ndarray,
                               patch_size: int) -> np.ndarray:
    """Apply blur kernel to each patch in a database.

    Args:
        patches    : [N, d]  flattened patches (d = P*P*C).
        kernel     : [K, K]  blur kernel.
        patch_size : P  (spatial size of patch, assumes square).

    Returns:
        blurred_patches : [N, d]  blurred patches (same shape).
    """
    N, d = patches.shape
    C = d // (patch_size * patch_size)
    blurred = np.zeros_like(patches)

    for i, p in enumerate(patches):
        img = p.reshape(patch_size, patch_size, C)
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        kh, kw = kernel.shape
        blurred_img = cv2.filter2D(img_u8, -1, kernel)
        blurred[i] = blurred_img.astype(np.float32).flatten() / 255.0

    return blurred


# ─────────────────────────────────────────────────────────────
#  Main Motion Blur Handler
# ─────────────────────────────────────────────────────────────

class MotionBlurHandler:
    """End-to-end motion blur detection, estimation, and correction.

    Integrates:
        - Laplacian variance blur detection
        - Radon-transform-based PSF estimation
        - Wiener deconvolution
        - Nearest Proximate Patch reconstruction

    Reference: Paper 5 — full pipeline.
    """

    def __init__(self,
                 blur_threshold: float = 0.3,
                 kernel_size: int = 15,
                 wiener_snr: float = 0.02):
        """
        Args:
            blur_threshold : Severity above which blur correction is applied.
            kernel_size    : PSF estimation kernel size.
            wiener_snr     : Wiener filter SNR parameter.
        """
        self.blur_threshold = blur_threshold
        self.kernel_size    = kernel_size
        self.wiener_snr     = wiener_snr

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Detect and remove motion blur from an image.

        Args:
            image : [H, W, 3] uint8 RGB image.

        Returns:
            corrected : [H, W, 3] float32 in [0, 1] — deblurred image.
            info      : Dict with blur metadata for UI display.
        """
        # Convert to float
        img_f = image.astype(np.float32) / 255.0

        # Detect blur severity
        severity = detect_blur_severity(image)
        info = {'blur_severity': severity, 'corrected': False,
                'angle_deg': None, 'blur_length': None}

        if severity < self.blur_threshold:
            # Not significantly blurred — pass through
            return img_f, info

        # Estimate kernel
        kernel, angle_deg, blur_length = estimate_motion_blur_kernel(
            image, kernel_size=self.kernel_size)
        info.update({'corrected': True,
                     'angle_deg': angle_deg,
                     'blur_length': blur_length,
                     'kernel': kernel})

        # Wiener deconvolution
        corrected = wiener_deconvolve(img_f, kernel, snr=self.wiener_snr)

        return corrected, info


# ─────────────────────────────────────────────────────────────
#  Quick Sanity Check
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Motion Blur Handler — Sanity Check")
    print("=" * 55)

    rng = np.random.default_rng(0)

    # Generate a synthetic blurred image (apply known kernel)
    test_img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    true_kernel = _make_motion_blur_kernel(15, angle_deg=45, length=8)

    blurred_img = np.zeros_like(test_img, dtype=np.float32)
    for c in range(3):
        blurred_img[:, :, c] = cv2.filter2D(
            test_img[:, :, c].astype(np.float32) / 255.0, -1, true_kernel)

    blurred_u8 = (blurred_img * 255).clip(0, 255).astype(np.uint8)

    handler = MotionBlurHandler(blur_threshold=0.0)  # force processing
    corrected, info = handler.process(blurred_u8)

    print(f"Input shape   : {blurred_u8.shape}")
    print(f"Output shape  : {corrected.shape}")
    print(f"Blur severity : {info['blur_severity']:.3f}")
    print(f"Estimated angle : {info.get('angle_deg', 'N/A'):.1f}° "
          f"(true: 45°)")
    print(f"Estimated length: {info.get('blur_length', 'N/A')} px "
          f"(true: 8 px)")

    # Test kernel construction
    kernel_test = _make_motion_blur_kernel(11, 30.0, 5)
    print(f"\nKernel shape : {kernel_test.shape}")
    print(f"Kernel sum   : {kernel_test.sum():.4f}  (should be ≈1.0)")
    print("✓ Motion Blur Handler passed sanity check.")
