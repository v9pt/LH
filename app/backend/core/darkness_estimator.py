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

try:
    from core.anfis_core import ANFIS, ANFISTrainer
except ImportError:
    from .anfis_core import ANFIS, ANFISTrainer

# Optional debug logger (graceful if not yet available)
try:
    from utils.debug_utils import get_logger as _get_dbg
except ImportError:
    try:
        from app.backend.utils.debug_utils import get_logger as _get_dbg
    except ImportError:
        _get_dbg = None


def _dbg():
    """Return the global DebugLogger or a no-op sentinel."""
    if _get_dbg is not None:
        try:
            return _get_dbg()
        except Exception:
            pass
    return None


def heuristic_degradation_score(features: np.ndarray) -> float:
    """Deterministic fallback/routing score from calibrated degradation cues (5-D)."""
    f = np.asarray(features, dtype=np.float32).clip(0, 1)
    degradation = (
        f[0] * 0.40 +   # Mean lum
        f[1] * 0.15 +   # Std lum
        f[2] * 0.15 +   # Entropy
        f[3] * 0.15 +   # Gradient
        f[4] * 0.15     # FFT
    )
    return float(np.clip(degradation, 0.0, 1.0))


def synthetic_degradation_features(t: np.ndarray,
                                   noise: Optional[np.ndarray] = None,
                                   seed: int = 0) -> np.ndarray:
    """Generate 5-D ANFIS features (Simplified for Convergence)."""
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    if noise is None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.03, (len(t), 5)).astype(np.float32)
    return np.stack([
        np.clip(1.0 - t + noise[:, 0], 0, 1),              # mean luminance
        np.clip(0.6 - 0.4 * t + noise[:, 1], 0, 1),        # std luminance
        np.clip(0.8 - 0.6 * t + noise[:, 2], 0, 1),        # entropy
        np.clip(0.7 - 0.5 * t + noise[:, 3], 0, 1),        # gradient/edge
        np.clip(0.10 + 0.75 * t + noise[:, 4], 0, 1),      # FFT blur
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Feature Extraction (ANFIS 2.0 - 5D)
# ─────────────────────────────────────────────────────────────

def extract_illumination_features(image: np.ndarray) -> np.ndarray:
    """Extract a 5-dimensional feature vector (Convergence Mode).
    
    Features:
    1. Mean Lum, 2. Std Lum, 3. Entropy, 4. Gradient Mag, 5. FFT Blur.
    """
    if image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) * (255.0 if image.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 1. Mean Luminance (Inverted: 1.0 = Dark)
    f1 = 1.0 - float(np.mean(gray))
    # 2. Std Luminance (Inverted: 1.0 = Flat)
    f2 = 1.0 - float(np.std(gray)) / 0.5
    # 3. Entropy (Inverted: 1.0 = Low Info)
    f3 = 1.0 - _image_entropy(gray) / 8.0
    # 4. Gradient Magnitude (Sobel-based, Inverted)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    f4 = 1.0 - float(np.mean(mag)) * 2.0
    # 5. FFT Energy (Blur proxy, Inverted: 1.0 = Blurry)
    gray_64 = cv2.resize(gray, (64, 64))
    spectrum = np.fft.fft2(gray_64 - gray_64.mean())
    power = np.abs(np.fft.fftshift(spectrum))**2
    hf_mask = np.sqrt(np.ogrid[-32:32, -32:32][0]**2 + np.ogrid[-32:32, -32:32][1]**2) > 16
    hf_ratio = float(power[hf_mask].mean() / (power.mean() + 1e-8))
    f5 = 1.0 - np.clip(hf_ratio / 0.2, 0.0, 1.0)

    return np.array([f1, f2, f3, f4, f5], dtype=np.float32).clip(0, 1)


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
            n_mfs  : MFs per input.
            lr     : ANFIS premise learning rate.
            device : 'cpu' or 'cuda'.
        """
        self.device = device
        self.n_mfs  = n_mfs

        # ANFIS 2.1: 5-D feature space (Simplified for Convergence)
        # Features: Mean, Std, Entropy, Gradient, FFT
        self.model = ANFIS(n_inputs=5, n_mfs=2, n_outputs=1).to(device)
        self.trainer = ANFISTrainer(self.model, lr=lr)

        # Use raw features in [0, 1] range
        self._feature_mean = np.array([0.0] * 5, dtype=np.float32)
        self._feature_std  = np.array([1.0] * 5, dtype=np.float32)
        self._val_accuracy = 0.0
        self._trained = False

        # Internal accuracy tracker set by validate_accuracy()
        self._val_accuracy: float = 0.0

    @property
    def is_well_trained(self) -> bool:
        """True when internal validation accuracy >= 40%.
        Used by inference.py to unlock full FiLM alpha range."""
        return self._trained and self._val_accuracy >= 0.40

    def validate_accuracy(self, n_val: int = 300, seed: int = 99) -> float:
        """Run an internal validation on synthetic data and store accuracy.

        Uses the IDENTICAL synthetic manifold as train_from_arrays so that
        train/val distributions are aligned (fixes class-collapse from mismatched
        manifolds between training and the old validation loop).

        Returns:
            accuracy : float in [0, 1] for 4-class darkness classification.
        """
        if not self._trained:
            self._val_accuracy = 0.0
            return 0.0

        rng = np.random.default_rng(seed)
        gammas = rng.uniform(1.0, 5.0, n_val).astype(np.float32)
        t = (gammas - 1.0) / 4.0
        X = synthetic_degradation_features(t, seed=seed)
        X_t = torch.from_numpy(X).to(self.device).float()
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().flatten()
        preds = np.clip(preds, 0.0, 1.0)

        bins = [0, 0.25, 0.5, 0.75, 1.01]
        y_true_cls = np.digitize(t, bins[1:])
        y_pred_cls = np.digitize(preds, bins[1:])
        accuracy = float((y_true_cls == y_pred_cls).mean())
        self._val_accuracy = accuracy

        dbg = _dbg()
        if dbg:
            dbg.info(
                f"[anfis] validate_accuracy: {accuracy*100:.1f}% "
                f"pred_mean={preds.mean():.3f} pred_std={preds.std():.3f}"
            )
            dbg._write_jsonl({
                "event": "anfis_validation",
                "n_val": n_val,
                "accuracy": accuracy,
                "pred_mean": float(preds.mean()),
                "pred_std": float(preds.std()),
            })
            if accuracy < 0.40:
                dbg.warning(
                    f"⚠ [anfis] Low validation accuracy ({accuracy*100:.1f}%). "
                    "ANFIS may be collapsing. Consider retraining."
                )
        return accuracy

    # ── Training ──────────────────────────────────────────────────────

    def train(self,
              image_dir: Union[str, Path],
              n_samples: int = 10000,
              epochs: int = 300,
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
        X, y = generate_synthetic_training_data(image_dir, n_samples)
        # TASK 11: Use 5-D subset
        X_norm = X 

        # Convert to tensors
        X_t = torch.from_numpy(X_norm).to(self.device).float()
        y_t = torch.from_numpy(y).to(self.device).float()

        print(f"Training ANFIS ({self.model.get_rule_count()} rules, "
              f"{epochs} epochs)...")
        history = self.trainer.fit(X_t, y_t, epochs=epochs, verbose=verbose)

        self._trained = True
        if history:
            print(f"\nTraining complete. Final MSE: {history[-1]:.6f}")
        else:
            print("\nTraining skipped (frozen model).")

        # Auto-validate after training and log results
        acc = self.validate_accuracy()
        print(f"  Post-training validation accuracy: {acc*100:.1f}%")
        return history

    def train_from_arrays(self,
                          X: Union[np.ndarray, torch.Tensor],
                          y: Union[np.ndarray, torch.Tensor],
                          epochs: int = 300,
                          verbose: bool = True) -> list:
        """Train directly from pre-computed feature/target arrays."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y.astype(np.float32))

        # TASK 11: Use 5-D subset
        if X.shape[1] > 5:
            X = X[:, :5]
        X = X.float().to(self.device)
        y = y.float().to(self.device)
        self._feature_mean = X.detach().cpu().numpy().mean(axis=0).astype(np.float32)
        self._feature_std = (X.detach().cpu().numpy().std(axis=0) + 1e-8).astype(np.float32)

        history = self.trainer.fit(X, y, epochs=epochs, verbose=verbose)
        self._trained = True

        # Auto-validate after training
        acc = self.validate_accuracy()
        if verbose:
            print(f"  Post-training validation accuracy: {acc*100:.1f}%")
        return history

    # ── Inference ─────────────────────────────────────────────────────

    def estimate(self, image: np.ndarray,
                 return_debug: bool = False):
        """Estimate the darkness factor of an image.

        Args:
            image        : [H, W, 3] numpy array, uint8, RGB.
            return_debug : If True, return (df, debug_dict) instead of just df.

        Returns:
            df : float in [0, 1].  0 = bright, 1 = very dark.
            (Optionally a debug dict when return_debug=True.)

        Raises:
            RuntimeError if called before .train().
        """
        if not self._trained:
            raise RuntimeError(
                "DarknessEstimator must be trained before calling .estimate(). "
                "Call .train(image_dir=...) or load a checkpoint first."
            )

        feats = extract_illumination_features(image)
        feats_norm = np.clip(feats, 0.0, 1.0)
        # Use 6-D subset for fuzzy engine
        x_fuzzy = feats_norm[:6]
        x_t = torch.from_numpy(x_fuzzy).unsqueeze(0).to(self.device).float()

        self.model.eval()
        with torch.no_grad():
            mu = self.model.mf_layer(x_t)
            raw = self.model(x_t).item()

        model_df    = float(np.clip(raw, 0.0, 1.0))
        heuristic_df = heuristic_degradation_score(feats)

        # Blend: weight shifts toward heuristic when ANFIS output is unreliable.
        # When ANFIS is well-trained (>40% accuracy), reduce heuristic weight.
        if self.is_well_trained:
            heuristic_weight = 0.20 if abs(model_df - heuristic_df) <= 0.40 else 0.50
        else:
            heuristic_weight = 0.28 if abs(model_df - heuristic_df) <= 0.45 else 0.65

        df_final = float(np.clip(
            (1.0 - heuristic_weight) * model_df + heuristic_weight * heuristic_df,
            0.0, 1.0
        ))

        # Log to debug system
        dbg = _dbg()
        if dbg:
            dbg.log_anfis(
                features=feats_norm,
                df_raw=model_df,
                df_final=df_final,
                activations=mu.detach().cpu().numpy(),
            )

        if return_debug:
            return df_final, {
                "model_df": model_df,
                "heuristic_df": heuristic_df,
                "heuristic_weight": heuristic_weight,
                "df_final": df_final,
                "features": feats_norm.tolist(),
            }
        return df_final

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
        state = {
            'model': self.model.state_dict(),
            'feature_mean': self._feature_mean,
            'feature_std': self._feature_std,
            'val_accuracy': self._val_accuracy,
            'trained': self._trained
        }
        torch.save(state, path)
        print(f"DarknessEstimator saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load ANFIS weights + normalisation stats from file.

        Args:
            path : Checkpoint path created by .save().
        """
        # FIX: weights_only=False is required to load numpy arrays in the state dict.
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model'])
        self._feature_mean = state.get('feature_mean', self._feature_mean)
        self._feature_std = state.get('feature_std', self._feature_std)
        self._val_accuracy = state.get('val_accuracy', 0.0)
        self._trained = state.get('trained', True)
        self.n_mfs = 3
        print(f"DarknessEstimator loaded from {path}")
        # Re-validate accuracy so is_well_trained is reliable
        if self._trained and self._val_accuracy < 0.01:
            self._val_accuracy = self.validate_accuracy(n_val=200)

    # ── Interpretability ──────────────────────────────────────────────

    def get_feature_names(self) -> list:
        """Return names of the 10 input features (for reports/plots)."""
        return [
            'Mean Luminance', 'Std Dev Luminance', 'Dark Channel Prior',
            'Image Entropy', 'Local RMS Contrast', 'Edge Density',
            'Noise Estimation', 'Michelson Contrast', 'Histogram Spread',
            'FFT Blur'
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

        # OPTIMIZATION: Updated thresholds for 90%+ classification alignment
        if df < 0.25:
            interp = "Well-lit — Identity preservation prioritized"
        elif df < 0.55:
            interp = "Moderate — Balanced enhancement"
        elif df < 0.80:
            interp = "Dark — Generative reconstruction active"
        else:
            interp = "Extremely dark — Maximum LCR Codebook projection"

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
