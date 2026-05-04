"""
ANFIS Training Script — Trains all classical components
========================================================

Run this in Colab after mounting Drive and cloning the repo.
Or run locally: python -m training.train_anfis

Training order:
    1. Generate synthetic (feature, DF) data from CelebA
    2. Train ANFIS darkness estimator (200 epochs, ~30 min on CPU)
    3. Build LCR dictionary (K-means on 50k patches, ~20 min)
    4. Train position-patch regressors (~1 hour)
    5. Save all checkpoints to checkpoints/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
from pathlib import Path
import time

from core.darkness_estimator import (DarknessEstimator,
                                      generate_synthetic_training_data)
from core.anfis_lcr import FaceDictionary, ANFISLocalityRepresentation
from core.regression_reconstructor import PositionPatchRegressor

# ─── Config ────────────────────────────────────────────────────────────
DATA_DIR       = Path('data/img_align_celeba')
CKPT_DIR       = Path('checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

TRAIN_N        = 5_000   # synthetic darkness samples
ANFIS_EPOCHS   = 200
DICT_ATOMS     = 512
DICT_IMAGES    = 2_000   # HR images for dictionary (out of 8k available)
REG_IMAGES     = 1_000   # HR images for regression training
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print(f"ANFIS Training Script  |  Device: {DEVICE}")
print("=" * 60)


# ─── Step 1: ANFIS Darkness Estimator ──────────────────────────────────
print("\n[1/3] Training ANFIS Darkness Estimator...")
t0 = time.time()

de = DarknessEstimator(n_mfs=3, lr=1e-3, device=DEVICE)

if DATA_DIR.exists() and len(list(DATA_DIR.glob('*.jpg'))) > 100:
    history = de.train(DATA_DIR, n_samples=TRAIN_N,
                       epochs=ANFIS_EPOCHS, verbose=True)
else:
    # Fallback: pure synthetic data (no real images needed)
    print("  ⚠  CelebA not found — using fully synthetic training data.")
    rng = np.random.default_rng(42)
    N   = TRAIN_N
    gammas  = rng.uniform(1.0, 5.0, N).astype(np.float32)
    targets = ((gammas - 1.0) / 4.0).reshape(-1, 1)
    X = np.column_stack([
        np.clip(1.0 - gammas/5 + rng.normal(0, 0.05, N), 0, 1),
        np.clip(0.3 - gammas/20 + rng.normal(0, 0.02, N), 0, 1),
        np.clip(gammas/5       + rng.normal(0, 0.05, N), 0, 1),
        np.clip(1.0 - gammas/8 + rng.normal(0, 0.05, N), 0, 1),
    ]).astype(np.float32)
    history = de.train_from_arrays(X, targets, epochs=ANFIS_EPOCHS, verbose=True)

de.save(CKPT_DIR / 'darkness_estimator.pt')
print(f"  ✓ Done in {time.time()-t0:.1f}s  |  Final MSE: {history[-1]:.6f}")


# ─── Step 2: LCR Dictionary ────────────────────────────────────────────
print("\n[2/3] Building LCR Face Dictionary...")
t0 = time.time()

face_dict = FaceDictionary(n_atoms=DICT_ATOMS, patch_size=8, stride=4, scale=4)

if DATA_DIR.exists():
    img_paths = sorted(DATA_DIR.glob('*.jpg'))[:DICT_IMAGES]
    hr_images, dfs = [], []

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        hr_images.append(img)
        # Estimate DF for each training image
        df = de.estimate(img)
        dfs.append(df)

    face_dict.build(hr_images, darkness_factors=dfs,
                    max_patches=50_000, seed=42)
    face_dict.save(CKPT_DIR / 'face_dictionary.npz')
    print(f"  ✓ Done in {time.time()-t0:.1f}s")
else:
    print("  ⚠  Skipping dictionary (no data). LCR will use bicubic fallback.")


# ─── Step 3: Position-Patch Regressors ────────────────────────────────
print("\n[3/3] Training Position-Patch Regressors...")
t0 = time.time()

regressor = PositionPatchRegressor(
    image_size_lr=(32, 32), patch_size=8, stride=4, scale=4, kernel='linear')

if DATA_DIR.exists():
    img_paths = sorted(DATA_DIR.glob('*.jpg'))[DICT_IMAGES:DICT_IMAGES+REG_IMAGES]
    reg_images = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reg_images.append(img)   # raw, regressor will resize internally

    regressor.train(reg_images, verbose=True)
    regressor.save(CKPT_DIR / 'regressors')
    print(f"  ✓ Done in {time.time()-t0:.1f}s")
else:
    print("  ⚠  Skipping regressors (no data).")


print("\n" + "=" * 60)
print(f"✓ Training complete. Checkpoints saved to: {CKPT_DIR}/")
print("Next: python app/backend/inference.py <image.jpg>")
print("=" * 60)
