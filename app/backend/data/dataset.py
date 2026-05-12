from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from app.backend.core.darkness_estimator import extract_illumination_features
    from app.backend.utils.debug_utils import get_logger
except ImportError:
    from core.darkness_estimator import extract_illumination_features
    try:
        from utils.debug_utils import get_logger
    except ImportError:
        get_logger = None


class FaceSRDataset(Dataset):
    """Image folder dataset returning LR/HR tensors and 10-D ANFIS condition."""

    def __init__(self, data_dirs, degrader, hr_size=128, augment=True, max_images=None):
        self.degrader = degrader
        self.hr_size = hr_size
        self.augment = augment
        paths = []
        for data_dir in data_dirs:
            root = Path(data_dir)
            if root.exists():
                # Optimized fast-scan for large datasets
                img_gen = root.iterdir()
                for p in img_gen:
                    if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        paths.append(p)
                    if max_images and len(paths) >= max_images:
                        break
        if max_images is not None:
            paths = paths[:max_images]
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            if get_logger is not None:
                try:
                    get_logger().log_failure(path.name, "dataset_read", f"Could not read image: {path}")
                except Exception:
                    pass
            raise RuntimeError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.hr_size, self.hr_size), interpolation=cv2.INTER_AREA)

        rng = np.random.default_rng(index + 17)
        if self.augment and rng.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])

        lr = self.degrader(rgb, rng=rng)
        condition = extract_illumination_features((lr * 255.0).astype(np.uint8))

        hr_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        lr_t = torch.from_numpy(lr).permute(2, 0, 1)
        cond_t = torch.from_numpy(condition.astype(np.float32))

        return {"lr": lr_t.float(), "hr": hr_t.float(), "condition": cond_t.float()}


FaceDataset = FaceSRDataset
