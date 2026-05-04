"""
CelebA Dataset Downloader
=========================
Downloads CelebA aligned face images for training.
Run: python scripts/download_data.py
"""

import os
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# CelebA aligned face images — official Google Drive link
# 202,599 face images at 178×218px, ~1.3GB
CELEBA_URL   = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
CELEBA_ZIP   = DATA_DIR / 'img_align_celeba.zip'
CELEBA_DIR   = DATA_DIR / 'img_align_celeba'

def download_celeba():
    if CELEBA_DIR.exists() and len(list(CELEBA_DIR.glob('*.jpg'))) > 1000:
        print(f"✓ CelebA already downloaded: {CELEBA_DIR}  "
              f"({len(list(CELEBA_DIR.glob('*.jpg')))} images)")
        return

    print("Downloading CelebA aligned faces (~1.3GB)...")
    gdown.download(CELEBA_URL, str(CELEBA_ZIP), quiet=False)

    print("Extracting...")
    with zipfile.ZipFile(CELEBA_ZIP, 'r') as zf:
        zf.extractall(DATA_DIR)

    CELEBA_ZIP.unlink(missing_ok=True)   # remove zip to save space
    n = len(list(CELEBA_DIR.glob('*.jpg')))
    print(f"✓ CelebA extracted: {n} images in {CELEBA_DIR}")


def create_split_files(train_n: int = 8000, val_n: int = 1000):
    """Write train.txt and val.txt listing image filenames."""
    all_imgs = sorted(CELEBA_DIR.glob('*.jpg'))
    if not all_imgs:
        print("ERROR: No images found in", CELEBA_DIR)
        return

    train_imgs = all_imgs[:train_n]
    val_imgs   = all_imgs[train_n:train_n + val_n]

    with open(DATA_DIR / 'train.txt', 'w') as f:
        f.write('\n'.join([str(p) for p in train_imgs]))

    with open(DATA_DIR / 'val.txt', 'w') as f:
        f.write('\n'.join([str(p) for p in val_imgs]))

    print(f"✓ Split files created: {len(train_imgs)} train, {len(val_imgs)} val")


if __name__ == '__main__':
    download_celeba()
    create_split_files()
    print("\nDone! Run training next:")
    print("  python -m training.train_anfis")
