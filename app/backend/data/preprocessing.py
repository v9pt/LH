import cv2
import numpy as np


class ImageDegrader:
    """Face SR degradation: blur, noise, JPEG, and downsampling simulation."""

    def __init__(self, scale=4, blur_kernel_size=5, noise_std=0.01, jpeg_quality=(35, 92)):
        self.scale = scale
        self.blur_kernel_size = blur_kernel_size
        self.noise_std = noise_std
        self.jpeg_quality = jpeg_quality

    def __call__(self, hr_rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        img = hr_rgb.astype(np.float32) / 255.0

        gamma = float(rng.uniform(1.0, 4.2))
        img = np.clip(img ** gamma, 0.0, 1.0)

        if rng.random() < 0.85:
            k = int(rng.choice([3, 5, 7, self.blur_kernel_size]))
            if k % 2 == 0:
                k += 1
            sigma = float(rng.uniform(0.2, 1.8))
            img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

        if rng.random() < 0.75:
            img = np.clip(img + rng.normal(0.0, self.noise_std, img.shape).astype(np.float32), 0.0, 1.0)

        h, w = img.shape[:2]
        lr = cv2.resize(img, (w // self.scale, h // self.scale), interpolation=cv2.INTER_AREA)

        if rng.random() < 0.70:
            q = int(rng.integers(self.jpeg_quality[0], self.jpeg_quality[1] + 1))
            lr_u8 = np.clip(lr * 255.0, 0, 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(lr_u8, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                lr = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return np.clip(lr, 0.0, 1.0).astype(np.float32)


DegradationPipeline = ImageDegrader
