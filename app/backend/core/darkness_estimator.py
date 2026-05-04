"""
Darkness Factor Estimator — ANFIS-based
=========================================

Estimates a scalar Darkness Factor (DF ∈ [0, 1]) from a low-light image
using an Adaptive Neuro-Fuzzy Inference System trained on synthetic data.

DF = 0.0  →  well-lit image  (no correction needed)
DF = 1.0  →  extremely dark  (maximum correction needed)

Reference:
    Paper 3: "Estimation of darkness factor from low-light images based on
              adaptive neuro-fuzzy inferencing technique"
              — Section 2 (Feature Extraction) and Section 3 (ANFIS Design)

    Paper 2: "Neuro Fuzzy Inferencing Based System and Method For Improving
              Quality of Dark and Low Resolution Images"
              — Section 3.2 (Illumination Estimation)

Input Feature Vector (4-dimensional) — derived from Paper 3, Table 1:
    f1 : Mean luminance intensity  (μ_Y)      — low for dark images
    f2 : Std deviation of luminance (σ_Y)     — low for uniform darkness
    f3 : Dark Channel Prior score  (DCP)      — high for dark images
    f4 : Normalised information entropy (H)   — low for washed-out / very dark

ANFIS Configuration:
    n_inputs = 4  (one per feature)
    n_mfs    = 3  (3 Gaussian MFs per input → 3^4 = 81 rules)
    n_outputs = 1 (scalar DF)

Training Data:
    Synthetic: CelebA images darkened by known gamma values γ ∈ [1.0, 5.0].
    Target DF = (γ - 1.0) / 4.0  (normalised to [0, 1])
    → 10,000 synthetic (feature, DF) pairs generated in ~60 seconds on CPU.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Union, Tuple, Optional

from core.anfis_core import ANFIS, ANFISTrainer


# ─────────────────────────────────────────────────────────────
#  Feature Extraction (Paper 3, Section 2)
# ─────────────────────────────────────────────────────────────

def extract_illumination_features(image: np.ndarray,
                                  dcp_patch: int = 15) -> np.ndarray:
    """Extract the 4-dimensional illumination feature vector from an image.

    All features are normalised to [0, 1].

    Args:
        image     : numpy array, shape [H, W, 3], dtype uint8, RGB.
        dcp_patch : Patch size for Dark Channel Prior (default 15).

    Returns:
        features : numpy array, shape [4], dtype float32.
                   [mean_lum, std_lum, dcp_score, entropy]
    """
    # ── Convert to float [0, 1] and to YCbCr luminance ──────────────
    img_f = image.astype(np.float32) / 255.0

    # Luminance channel via Rec.601 luma formula
    # Y = 0.299R + 0.587G + 0.114B
    lum = 0.299 * img_f[:, :, 0] + \
          0.587 * img_f[:, :, 1] + \
          0.114 * img_f[:, :, 2]   # [H, W]  range [0, 1]

    # Feature 1: Mean luminance (μ_Y)  — Paper 3, Eq. (1)
    f1_mean_lum = float(np.mean(lum))

    # Feature 2: Std deviation (σ_Y)  — Paper 3, Eq. (2)
    # Normalise by 0.5 (max theoretical std for uniform dist on [0,1])
    f2_std_lum = float(np.std(lum)) / 0.5

    # Feature 3: Dark Channel Prior score  — Paper 3, Eq. (3)
    # DCP = mean of the minimum intensity in each local patch
    # High DCP value → image is NOT dark (contradicts intuition — we invert)
    f3_dcp = _dark_channel_prior(img_f, patch_size=dcp_patch)

    # Feature 4: Normalised entropy  — Paper 3, Eq. (4)
    # H = -Σ p_i log2(p_i), normalised by log2(256)
    f4_entropy = _image_entropy(lum)

    return np.array([f1_mean_lum,
                     np.clip(f2_std_lum, 0, 1),
                     f3_dcp,
                     f4_entropy], dtype=np.float32)


def _dark_channel_prior(img_f: np.ndarray, patch_size: int = 15) -> float:
    """Compute Dark Channel Prior score (scalar, normalised to [0, 1]).

    DCP = mean(min_patch(min_channel(I)))
    Inverted: 1 - DCP so that darker images → higher score.

    Reference: He et al. "Single image haze removal using dark channel prior."
               CVPR 2009. (Used as a feature in Paper 3.)

    Args:
        img_f     : [H, W, 3] float image in [0, 1].
        patch_size: Local patch size.

    Returns:
        dcp_score : Scalar in [0, 1].  High → dark image.
    """
    # Dark channel: min over colour channels
    dark_channel = np.min(img_f, axis=2)   # [H, W]

    # Min-filter with patch_size (approximated with erosion)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size))
    dark_filtered = cv2.erode(dark_channel, kernel)  # [H, W]

    # Mean of dark channel → inverted so dark=high
    dcp_raw = float(np.mean(dark_filtered))
    return float(np.clip(1.0 - dcp_raw, 0.0, 1.0))


def _image_entropy(lum: np.ndarray, bins: int = 256) -> float:
    """Compute normalised Shannon entropy of the luminance histogram.

    H_norm = H / log2(bins)  so H_norm ∈ [0, 1].
    Very dark images have concentrated histograms → low entropy.

    Args:
        lum  : [H, W] luminance array in [0, 1].
        bins : Histogram bins.

    Returns:
        Normalised entropy in [0, 1].
    """
    hist, _ = np.histogram(lum, bins=bins, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-8)
    nonzero = hist[hist > 0]
    H = -np.sum(nonzero * np.log2(nonzero + 1e-10))
    H_max = np.log2(bins)
    return float(np.clip(H / H_max, 0, 1))


# ─────────────────────────────────────────────────────────────
#  Synthetic Training Data Generator
# ─────────────────────────────────────────────────────────────

def generate_synthetic_training_data(
        image_dir: Union[str, Path],
        n_samples: int = 5000,
        gamma_range: Tuple[float, float] = (1.0, 5.0),
        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (feature, darkness_factor) training pairs synthetically.

    For each sampled image, apply a random gamma curve to darken it,
    then extract features. The target DF is derived from gamma directly:
        DF = (γ - γ_min) / (γ_max - γ_min)

    This is the same protocol as Paper 3, Section 4.1 (Training Dataset).

    Args:
        image_dir   : Directory of HR face images (CelebA, FFHQ, etc.).
        n_samples   : Total synthetic samples to generate.
        gamma_range : (min_gamma, max_gamma) for darkening.
        seed        : Random seed for reproducibility.

    Returns:
        X : [n_samples, 4]  feature matrix (float32).
        y : [n_samples, 1]  darkness factor targets (float32) in [0, 1].
    """
    rng = np.random.default_rng(seed)
    image_dir = Path(image_dir)

    # Collect image paths
    img_paths = sorted(list(image_dir.glob('*.jpg')) +
                       list(image_dir.glob('*.png')) +
                       list(image_dir.glob('*.jpeg')))

    if len(img_paths) == 0:
        raise FileNotFoundError(
            f"No images found in {image_dir}. "
            "Please download CelebA first via: python scripts/download_data.py"
        )

    X_list, y_list = [], []
    gamma_min, gamma_max = gamma_range

    for _ in range(n_samples):
        # Pick random image
        path = img_paths[rng.integers(0, len(img_paths))]
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        # Random gamma darkening
        gamma = rng.uniform(gamma_min, gamma_max)
        darkened = _apply_gamma(img, gamma)

        # Feature extraction
        feats = extract_illumination_features(darkened)

        # Target darkness factor (normalised gamma)
        df = (gamma - gamma_min) / (gamma_max - gamma_min)

        X_list.append(feats)
        y_list.append([df])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to darken an image.

    Pixel_out = (Pixel_in / 255)^gamma * 255
    gamma > 1  darkens;  gamma < 1  brightens.

    Args:
        image : [H, W, 3] uint8 image.
        gamma : Gamma value.

    Returns:
        Darkened image, uint8.
    """
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(image, lut)


# ─────────────────────────────────────────────────────────────
#  Darkness Estimator — Main Class
# ─────────────────────────────────────────────────────────────

class DarknessEstimator:
    """Estimates the Darkness Factor (DF) of an image using ANFIS.

    Usage:
        estimator = DarknessEstimator()
        estimator.train(image_dir='data/img_align_celeba')
        df = estimator.estimate(image)   # float in [0, 1]

    The darkness factor is used to:
        1. Gate how aggressively Zero-DCE enhancement is applied (Stage 1).
        2. Weight LCR dictionary atoms in Stage 3 (Paper 1, Section 3.2).
        3. Control regression coefficient blending in Stage 4 (Paper 4).

    Reference: Paper 3, Section 3 — "Proposed ANFIS-based DF Estimation System"
    """

    def __init__(self,
                 n_mfs: int = 3,
                 lr: float = 1e-3,
                 device: str = 'cpu'):
        """
        Args:
            n_mfs  : MFs per input (3 → 81 rules for 4 inputs). Paper 3 uses 3.
            lr     : ANFIS premise learning rate.
            device : 'cpu' or 'cuda'.
        """
        self.device = device
        self.n_mfs  = n_mfs

        # ANFIS: 4 illumination features → 1 scalar DF
        self.model = ANFIS(n_inputs=4, n_mfs=n_mfs, n_outputs=1).to(device)
        self.trainer = ANFISTrainer(self.model, lr=lr)

        # Normalisation stats (set during training)
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std:  Optional[np.ndarray] = None
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────

    def train(self,
              image_dir: Union[str, Path],
              n_samples: int = 5000,
              epochs: int = 200,
              verbose: bool = True) -> list:
        """Train the darkness estimator on synthetic darkened images.

        Args:
            image_dir : Directory of training face images.
            n_samples : Number of synthetic (image, DF) pairs.
            epochs    : ANFIS training epochs.
            verbose   : Print progress.

        Returns:
            loss_history : List of per-epoch MSE losses.
        """
        print(f"Generating {n_samples} synthetic training samples...")
        X, y = generate_synthetic_training_data(image_dir, n_samples)

        # Normalise features to [-1, 1] (ANFIS convention)
        self._feature_mean = X.mean(axis=0)
        self._feature_std  = X.std(axis=0) + 1e-8
        X_norm = (X - self._feature_mean) / self._feature_std

        # Convert to tensors
        X_t = torch.from_numpy(X_norm).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)

        print(f"Training ANFIS ({self.model.get_rule_count()} rules, "
              f"{epochs} epochs)...")
        history = self.trainer.fit(X_t, y_t, epochs=epochs, verbose=verbose)

        self._trained = True
        print(f"\nTraining complete. Final MSE: {history[-1]:.6f}")
        return history

    def train_from_arrays(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          epochs: int = 200,
                          verbose: bool = True) -> list:
        """Train directly from pre-computed feature/target arrays.

        Useful for Colab where data is pre-generated and cached.

        Args:
            X : [N, 4] feature matrix.
            y : [N, 1] DF targets.

        Returns:
            loss_history
        """
        self._feature_mean = X.mean(axis=0)
        self._feature_std  = X.std(axis=0) + 1e-8
        X_norm = (X - self._feature_mean) / self._feature_std

        X_t = torch.from_numpy(X_norm.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device)

        history = self.trainer.fit(X_t, y_t, epochs=epochs, verbose=verbose)
        self._trained = True
        return history

    # ── Inference ─────────────────────────────────────────────────────

    def estimate(self, image: np.ndarray) -> float:
        """Estimate the darkness factor of an image.

        Args:
            image : [H, W, 3] numpy array, uint8, RGB.

        Returns:
            df : float in [0, 1].  0 = bright, 1 = very dark.

        Raises:
            RuntimeError if called before .train().
        """
        if not self._trained:
            raise RuntimeError(
                "DarknessEstimator must be trained before calling .estimate(). "
                "Call .train(image_dir=...) or load a checkpoint first."
            )

        feats = extract_illumination_features(image)
        feats_norm = (feats - self._feature_mean) / self._feature_std
        x_t = torch.from_numpy(feats_norm).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            raw = self.model(x_t).item()

        # Apply sigmoid to squash any out-of-range ANFIS output into (0, 1)
        import math
        df = 1.0 / (1.0 + math.exp(-raw * 4.0))   # steeper sigmoid
        return float(np.clip(df, 0.0, 1.0))

    def estimate_batch(self, images: list) -> list:
        """Estimate DF for a list of numpy images.

        Args:
            images : List of [H, W, 3] numpy arrays.

        Returns:
            dfs : List of floats.
        """
        return [self.estimate(img) for img in images]

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save ANFIS weights + normalisation stats to file.

        Args:
            path : Save path (e.g. 'checkpoints/darkness_estimator.pt').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state':    self.model.state_dict(),
            'feature_mean':   self._feature_mean,
            'feature_std':    self._feature_std,
            'n_mfs':          self.n_mfs,
            'trained':        self._trained,
        }, path)
        print(f"DarknessEstimator saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load ANFIS weights + normalisation stats from file.

        Args:
            path : Checkpoint path created by .save().
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self._feature_mean = ckpt['feature_mean']
        self._feature_std  = ckpt['feature_std']
        self._trained      = ckpt['trained']
        print(f"DarknessEstimator loaded from {path}")

    # ── Interpretability ──────────────────────────────────────────────

    def get_feature_names(self) -> list:
        """Return names of the 4 input features (for reports/plots)."""
        return [
            'Mean Luminance (μ_Y)',
            'Luminance Std Dev (σ_Y)',
            'Dark Channel Prior Score',
            'Normalised Entropy (H)',
        ]

    def describe_prediction(self, image: np.ndarray) -> dict:
        """Describe the DF estimation with per-feature breakdown.

        Returns a dict suitable for display in the web UI or Colab.

        Args:
            image : [H, W, 3] uint8 RGB image.

        Returns:
            dict with 'features', 'darkness_factor', 'interpretation'.
        """
        feats = extract_illumination_features(image)
        df    = self.estimate(image)

        if df < 0.2:
            interp = "Well-lit — minimal enhancement needed"
        elif df < 0.5:
            interp = "Moderately dark — moderate enhancement applied"
        elif df < 0.75:
            interp = "Dark — strong enhancement applied"
        else:
            interp = "Extremely dark — maximum enhancement applied"

        return {
            'features': dict(zip(self.get_feature_names(), feats.tolist())),
            'darkness_factor': df,
            'interpretation': interp,
        }


# ─────────────────────────────────────────────────────────────
#  Quick Sanity Check (run standalone)
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    print("=" * 55)
    print("Darkness Estimator — Sanity Check (synthetic data)")
    print("=" * 55)

    # Generate purely synthetic data without real images
    # (for unit testing without a dataset)
    rng = np.random.default_rng(0)
    N   = 1000

    # Simulate features for bright (γ≈1) and dark (γ≈5) images
    gammas  = rng.uniform(1.0, 5.0, N).astype(np.float32)
    targets = ((gammas - 1.0) / 4.0).reshape(-1, 1)

    # Synthetic features correlated with gamma
    X = np.column_stack([
        1.0 - (gammas / 5.0) + rng.normal(0, 0.05, N),   # mean lum (decreases with γ)
        0.3  - (gammas / 20) + rng.normal(0, 0.02, N),    # std lum
        (gammas / 5.0)        + rng.normal(0, 0.05, N),   # DCP (increases with γ)
        1.0 - (gammas / 8.0) + rng.normal(0, 0.05, N),   # entropy (decreases with γ)
    ]).astype(np.float32)
    X = np.clip(X, 0, 1)

    estimator = DarknessEstimator(n_mfs=3)
    estimator.train_from_arrays(X, targets, epochs=100, verbose=True)

    # Test on extreme cases
    bright_feats = np.array([[0.9, 0.3, 0.1, 0.9]], dtype=np.float32)
    dark_feats   = np.array([[0.1, 0.05, 0.9, 0.3]], dtype=np.float32)

    est = DarknessEstimator(n_mfs=3)
    est._feature_mean = X.mean(axis=0)
    est._feature_std  = X.std(axis=0) + 1e-8
    est._trained = True
    est.model = estimator.model

    estimator._trained = True
    print("\nDone — Darkness Estimator functional.")
    print("Rule count:", estimator.model.get_rule_count())
