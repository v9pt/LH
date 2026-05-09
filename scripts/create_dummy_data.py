import os
import numpy as np
import cv2
from pathlib import Path

DATA_DIR = Path('data/img_align_celeba')
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Generating dummy synthetic CelebA dataset for local testing...")

for i in range(100):
    # Generate a dummy "face" image (178x218)
    img = np.zeros((218, 178, 3), dtype=np.uint8)
    # Background color
    img[:] = np.random.randint(50, 150, (3,))
    # Draw a "face" circle
    cv2.circle(img, (89, 109), 60, (200, 150, 120), -1)
    # Draw "eyes"
    cv2.circle(img, (60, 90), 10, (50, 50, 50), -1)
    # Draw "mouth"
    cv2.ellipse(img, (89, 140), (20, 10), 0, 0, 180, (50, 50, 50), -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(DATA_DIR / f"{i:06d}.jpg"), img)

print(f"Generated 100 images in {DATA_DIR}")

# Create splits
with open('data/train.txt', 'w') as f:
    for i in range(80):
        f.write(str(DATA_DIR / f"{i:06d}.jpg") + '\n')
        
with open('data/val.txt', 'w') as f:
    for i in range(80, 100):
        f.write(str(DATA_DIR / f"{i:06d}.jpg") + '\n')

print("Done generating synthetic data.")
