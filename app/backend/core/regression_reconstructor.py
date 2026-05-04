"""
Regression-Based Face Reconstructor
=====================================

Implements position-patch kernel ridge regression for face reconstruction
from noisy, low-resolution inputs.

Reference:
    Paper 4: "A new face reconstruction technique for noisy low-resolution
              images using regression learning"
              — Section 3 (Position-Patch Representation) and
                Section 4 (Regression Learning)

Core Idea (Paper 4, Section 3.1):
    The face is divided into a grid of fixed spatial positions. For each
    position, a local regressor maps the LR patch feature vector to the
    corresponding HR patch.

    This works because facial structure is highly regular: the left eye always
    appears near position (y=0.3, x=0.2), the nose near (0.5, 0.5), etc.
    Position-specific regressors exploit this spatial prior.

Regressor (Paper 4, Eq. 8 — Kernel Ridge Regression):
    Given training pairs (X_pos, Y_pos) for each position:
        X_pos : [N, d_lr]   LR patch features
        Y_pos : [N, d_hr]   HR patch targets

    Kernel Ridge Regression:
        W* = (K + λ I)^{-1} Y_pos
    where K[i,j] = κ(x_i, x_j)  is the RBF kernel matrix.

    Prediction:
        ŷ = K_test · W*
    where K_test[j] = κ(x_query, x_j)  for training points x_j.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Union, List, Tuple
from sklearn.kernel_ridge import KernelRidge
import joblib
import warnings
warnings.filterwarnings('ignore')

from core.anfis_lcr import extract_patches, reconstruct_from_patches


# ─────────────────────────────────────────────────────────────
#  Position Grid
# ─────────────────────────────────────────────────────────────

def build_position_grid(image_size: Tuple[int, int],
                        patch_size: int,
                        stride: int) -> List[Tuple[int, int]]:
    """Enumerate all patch positions (y, x) for a fixed image size.

    Args:
        image_size : (H, W) of the LR image.
        patch_size : Patch size in pixels.
        stride     : Stride.

    Returns:
        positions : List of (y, x) top-left corners.
    """
    H, W = image_size
    positions = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            positions.append((y, x))
    return positions


def get_patch_at(image: np.ndarray,
                 y: int, x: int,
                 patch_size: int) -> np.ndarray:
    """Extract a single patch from an image.

    Args:
        image      : [H, W, C] float32 in [0,1] or uint8.
        y, x       : Top-left corner.
        patch_size : Patch size.

    Returns:
        patch : flattened float32 vector.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    p = image[y:y+patch_size, x:x+patch_size]
    return p.flatten().astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Position-Patch Regressor
# ─────────────────────────────────────────────────────────────

class PositionPatchRegressor:
    """Position-specific patch regression for face hallucination.

    Trains one Kernel Ridge Regressor per spatial position in the face.
    During inference, each position's regressor independently predicts
    the HR patch from the LR patch at that position.

    Reference: Paper 4, Algorithm 1 — Position-Patch Regression Training.

    Args:
        image_size_lr : (H_lr, W_lr) of the LR face (e.g. 32x32).
        patch_size    : LR patch size (e.g. 8).
        stride        : Patch extraction stride (e.g. 4).
        scale         : SR scale factor (e.g. 4 → HR is 128x128).
        alpha         : Regularisation parameter λ for Ridge regression.
        kernel        : Kernel type: 'rbf' (default) or 'linear'.
        gamma         : RBF kernel width γ (None → 1/d_lr).
    """

    def __init__(self,
                 image_size_lr: Tuple[int, int] = (32, 32),
                 patch_size: int = 8,
                 stride: int = 4,
                 scale: int = 4,
                 alpha: float = 1e-3,
                 kernel: str = 'rbf',
                 gamma: Optional[float] = None):
        self.image_size_lr = image_size_lr
        self.patch_size    = patch_size
        self.stride        = stride
        self.scale         = scale
        self.alpha_reg     = alpha
        self.kernel        = kernel
        self.gamma         = gamma

        # Compute positions on the LR grid
        self.positions = build_position_grid(image_size_lr, patch_size, stride)
        self.n_positions = len(self.positions)

        # One regressor per position — initialised in .train()
        self.regressors: List[Optional[KernelRidge]] = [None] * self.n_positions
        self._trained = False

        # HR position list (scaled)
        self.hr_patch = patch_size * scale
        self.positions_hr = [
            (y * scale, x * scale) for (y, x) in self.positions
        ]

    # ── Training ──────────────────────────────────────────────────────

    def train(self,
              hr_images: List[np.ndarray],
              verbose: bool = True) -> None:
        """Train position-specific regressors from HR training images.

        For each training image, create LR version, extract patches at
        all positions, and accumulate training data per position.
        Then fit one KernelRidge regressor per position.

        Args:
            hr_images : List of [H, W, 3] uint8 HR face images (aligned,
                        same size). Resized to match image_size_lr * scale.
            verbose   : Print progress.
        """
        H_lr, W_lr = self.image_size_lr
        H_hr, W_hr = H_lr * self.scale, W_lr * self.scale

        # Accumulate (LR patch, HR patch) per position
        X_by_pos: List[List[np.ndarray]] = [[] for _ in range(self.n_positions)]
        Y_by_pos: List[List[np.ndarray]] = [[] for _ in range(self.n_positions)]

        for i, hr_img in enumerate(hr_images):
            # Resize to fixed HR size
            hr_resized = cv2.resize(hr_img, (W_hr, H_hr),
                                    interpolation=cv2.INTER_LANCZOS4)
            # Downsample to LR
            lr_resized = cv2.resize(hr_resized, (W_lr, H_lr),
                                    interpolation=cv2.INTER_CUBIC)

            hr_f = hr_resized.astype(np.float32) / 255.0
            lr_f = lr_resized.astype(np.float32) / 255.0

            # Extract patches at each position
            for pos_idx, ((y_lr, x_lr), (y_hr, x_hr)) in enumerate(
                    zip(self.positions, self.positions_hr)):
                lr_patch = get_patch_at(lr_f, y_lr, x_lr, self.patch_size)
                hr_patch = get_patch_at(hr_f, y_hr, x_hr, self.hr_patch)
                X_by_pos[pos_idx].append(lr_patch)
                Y_by_pos[pos_idx].append(hr_patch)

        if verbose:
            print(f"Fitting {self.n_positions} position regressors "
                  f"on {len(hr_images)} images...")

        # Fit one regressor per position
        for pos_idx in range(self.n_positions):
            X = np.array(X_by_pos[pos_idx], dtype=np.float32)   # [N, d_lr]
            Y = np.array(Y_by_pos[pos_idx], dtype=np.float32)   # [N, d_hr]

            reg = KernelRidge(
                alpha=self.alpha_reg,
                kernel=self.kernel,
                gamma=self.gamma,
            )
            reg.fit(X, Y)
            self.regressors[pos_idx] = reg

            if verbose and (pos_idx + 1) % max(1, self.n_positions // 5) == 0:
                print(f"  Position {pos_idx+1}/{self.n_positions} done.")

        self._trained = True
        if verbose:
            print("Position-patch regressors training complete.")

    # ── Inference ─────────────────────────────────────────────────────

    def reconstruct(self, lr_image: np.ndarray) -> np.ndarray:
        """Reconstruct HR face from LR input using learned regressors.

        Args:
            lr_image : [H_lr, W_lr, 3] LR face image (uint8 or float32).

        Returns:
            hr_image : [H_hr, W_hr, 3] float32 in [0, 1].

        Raises:
            RuntimeError if called before .train() or .load().
        """
        if not self._trained:
            raise RuntimeError(
                "PositionPatchRegressor must be trained before .reconstruct()."
            )

        H_lr, W_lr = self.image_size_lr
        H_hr = H_lr * self.scale
        W_hr = W_lr * self.scale

        # Resize LR input to expected size
        if lr_image.dtype == np.uint8:
            lr_f = lr_image.astype(np.float32) / 255.0
        else:
            lr_f = lr_image.copy()
        lr_f = cv2.resize(lr_f, (W_lr, H_lr), interpolation=cv2.INTER_CUBIC)

        # Predict HR patch at each position
        hr_patches_est = []
        for pos_idx, (y_lr, x_lr) in enumerate(self.positions):
            lr_patch = get_patch_at(lr_f, y_lr, x_lr, self.patch_size)
            # Predict: ŷ = κ(x_query, X_train) · W*
            hr_patch = self.regressors[pos_idx].predict(
                lr_patch.reshape(1, -1))[0]
            hr_patch = np.clip(hr_patch, 0, 1)
            hr_patches_est.append(hr_patch)

        # Reconstruct from HR patches
        C = lr_f.shape[2] if lr_f.ndim == 3 else 1
        hr_image = reconstruct_from_patches(
            np.array(hr_patches_est),
            self.positions_hr,
            (H_hr, W_hr, C),
            patch_size=self.hr_patch
        )
        return hr_image   # float32 in [0, 1]

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save all regressors to disk using joblib.

        Args:
            path : Directory path to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, reg in enumerate(self.regressors):
            if reg is not None:
                joblib.dump(reg, path / f'regressor_{i:04d}.pkl')
        # Save config
        np.savez(path / 'config.npz',
                 image_size_lr=np.array(self.image_size_lr),
                 patch_size=np.array(self.patch_size),
                 stride=np.array(self.stride),
                 scale=np.array(self.scale))
        print(f"PositionPatchRegressor saved to {path}/  ({len(self.regressors)} regressors)")

    def load(self, path: Union[str, Path]) -> None:
        """Load all regressors from disk.

        Args:
            path : Directory created by .save().
        """
        path = Path(path)
        cfg  = np.load(path / 'config.npz')
        self.image_size_lr = tuple(cfg['image_size_lr'].tolist())
        self.patch_size    = int(cfg['patch_size'])
        self.stride        = int(cfg['stride'])
        self.scale         = int(cfg['scale'])
        self.positions     = build_position_grid(
            self.image_size_lr, self.patch_size, self.stride)
        self.n_positions   = len(self.positions)
        self.hr_patch      = self.patch_size * self.scale
        self.positions_hr  = [
            (y * self.scale, x * self.scale) for (y, x) in self.positions
        ]

        self.regressors = []
        for i in range(self.n_positions):
            pkl = path / f'regressor_{i:04d}.pkl'
            if pkl.exists():
                self.regressors.append(joblib.load(pkl))
            else:
                self.regressors.append(None)

        self._trained = True
        print(f"PositionPatchRegressor loaded from {path}/")


# ─────────────────────────────────────────────────────────────
#  Regression-Guided Blending with ANFIS DF
# ─────────────────────────────────────────────────────────────

def blend_lcr_and_regression(lcr_output: np.ndarray,
                              reg_output: np.ndarray,
                              darkness_factor: float) -> np.ndarray:
    """Blend LCR and regression outputs using ANFIS darkness factor.

    Paper 4, Section 5 — "Adaptive Blending":
    When DF is high (very dark), the regression model is more reliable
    because it was trained on darkened examples. When DF is low (bright),
    LCR works better. We use a linear blend:

        output = (1 - DF) * LCR + DF * Regression

    Args:
        lcr_output : [H, W, C] LCR hallucination result.
        reg_output : [H, W, C] Regression reconstruction result.
        darkness_factor : float ∈ [0, 1].

    Returns:
        blended : [H, W, C] float32.
    """
    df = float(np.clip(darkness_factor, 0, 1))
    blended = (1.0 - df) * lcr_output + df * reg_output
    return np.clip(blended, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Quick Sanity Check
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Regression Reconstructor — Sanity Check")
    print("=" * 55)

    rng = np.random.default_rng(42)

    # Tiny test: 8×8 LR → 32×32 HR (scale=4), 2×2 patch, stride=2
    LR_SIZE = (8, 8)
    HR_SIZE = (32, 32)

    regressor = PositionPatchRegressor(
        image_size_lr=LR_SIZE, patch_size=2, stride=2, scale=4, kernel='linear')

    print(f"Positions: {regressor.n_positions}")

    # Synthetic training data: random HR images
    train_imgs = [(rng.random((32, 32, 3)) * 255).astype(np.uint8)
                  for _ in range(20)]
    regressor.train(train_imgs, verbose=True)

    # Test reconstruction
    test_lr = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    hr_out  = regressor.reconstruct(test_lr)
    print(f"\nLR input:  {test_lr.shape}")
    print(f"HR output: {hr_out.shape}  (should be 32×32×3)")
    print(f"Value range: [{hr_out.min():.3f}, {hr_out.max():.3f}]")

    # Test blending
    fake_lcr = rng.random((32, 32, 3)).astype(np.float32)
    blended  = blend_lcr_and_regression(fake_lcr, hr_out, darkness_factor=0.6)
    print(f"\nBlended output shape: {blended.shape}")
    print("✓ Regression Reconstructor passed sanity check.")
