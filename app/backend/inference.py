"""
Updated End-to-End Inference Pipeline
=====================================

Integrates all 5 paper implementations into a single coherent pipeline:

    Stage 1: ANFIS Darkness Factor Estimation  (Paper 3)
    Stage 2: Motion Blur Detection & Removal   (Paper 5)
    Stage 3a: Zero-DCE Enhancement             (Paper 2 — neural component)
    Stage 3b: ANFIS-LCR Face Hallucination     (Paper 1)
    Stage 4:  Regression-Guided Blending       (Paper 4)
    Stage 5:  RRDB Neural Refinement           (ESRGAN-based, final polish)

The darkness factor computed in Stage 1 gates all downstream stages —
it is the central signal that makes this an ANFIS-driven pipeline.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional, Dict

# Existing models
from models.zero_dce import ZeroDCE
from models.rrdb_generator import RRDBNet
from utils.model_manager import ModelManager

# NEW: Paper implementations
from core.darkness_estimator import DarknessEstimator
from core.motion_blur_handler import MotionBlurHandler
from core.anfis_lcr import ANFISLocalityRepresentation, FaceDictionary
from core.regression_reconstructor import PositionPatchRegressor, blend_lcr_and_regression


class ANFISFaceSRPipeline:
    """ANFIS-guided Low-Light Face Super-Resolution Pipeline.

    Full 5-stage pipeline combining all research paper contributions.

    Usage:
        pipeline = ANFISFaceSRPipeline()
        pipeline.load_pretrained()
        results = pipeline.enhance('path/to/dark_face.jpg')
    """

    def __init__(self,
                 device: str = 'cpu',
                 use_blur_correction: bool = True,
                 use_lcr: bool = True,
                 use_regression: bool = True,
                 use_rrdb: bool = True):
        self.device = device
        self.use_blur_correction = use_blur_correction
        self.use_lcr             = use_lcr
        self.use_regression      = use_regression
        self.use_rrdb            = use_rrdb

        print("Initialising ANFIS Face SR Pipeline...")

        # ── Stage 1: Darkness Estimator (Paper 3) ──────────────────
        self.darkness_estimator = DarknessEstimator(n_mfs=3, device=device)

        # ── Stage 2: Motion Blur Handler (Paper 5) ─────────────────
        if use_blur_correction:
            self.blur_handler = MotionBlurHandler(
                blur_threshold=0.3, kernel_size=15, wiener_snr=0.02)

        # ── Stage 3a: Zero-DCE Enhancement (Paper 2 neural part) ──
        self.zero_dce = ZeroDCE(device=device)

        # ── Stage 3b: ANFIS-LCR Hallucinator (Paper 1) ────────────
        if use_lcr:
            self.face_dict    = FaceDictionary(n_atoms=512, patch_size=8,
                                               stride=4, scale=4)
            self.lcr_hallucinator = None   # set after dict.load()

        # ── Stage 4: Regression Reconstructor (Paper 4) ────────────
        if use_regression:
            self.regressor = PositionPatchRegressor(
                image_size_lr=(32, 32), patch_size=8, stride=4, scale=4)

        # ── Stage 5: RRDB Neural Refinement ────────────────────────
        if use_rrdb:
            self.rrdb = RRDBNet(device=device, scale=4)

        self.model_manager = ModelManager()
        print("Pipeline initialised.")

    # ── Loading ─────────────────────────────────────────────────────

    def load_pretrained(self,
                        checkpoint_dir: Union[str, Path] = 'checkpoints'):
        """Load all pretrained component weights.

        Args:
            checkpoint_dir : Directory containing saved checkpoints.
        """
        ckpt = Path(checkpoint_dir)

        # Darkness estimator
        de_path = ckpt / 'darkness_estimator.pt'
        if de_path.exists():
            self.darkness_estimator.load(de_path)
        else:
            print(f"  ⚠ No darkness estimator checkpoint at {de_path}. "
                  "Train first with: python -m training.train_anfis")

        # LCR dictionary
        if self.use_lcr:
            dict_path = ckpt / 'face_dictionary.npz'
            if dict_path.exists():
                self.face_dict.load(dict_path)
                self.lcr_hallucinator = ANFISLocalityRepresentation(
                    self.face_dict, lam=1e-4)
            else:
                print(f"  ⚠ No dictionary at {dict_path}. "
                      "Build first with training script.")

        # Regression regressors
        if self.use_regression:
            reg_path = ckpt / 'regressors'
            if reg_path.exists():
                self.regressor.load(reg_path)
            else:
                print(f"  ⚠ No regressors at {reg_path}.")

        # RRDB — try pretrained Real-ESRGAN weights
        if self.use_rrdb:
            rrdb_path = ckpt / 'rrdb.pth'
            if rrdb_path.exists():
                self.rrdb.load_checkpoint(str(rrdb_path))
            else:
                try:
                    pretrained = self.model_manager.download_model('rrdb_esrgan')
                    if pretrained:
                        self.rrdb.load_checkpoint(str(pretrained))
                except Exception as e:
                    print(f"  ⚠ RRDB pretrained not available: {e}")

    # ── Preprocessing ───────────────────────────────────────────────

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or numpy array (uint8 RGB)."""
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return source.copy()

    def _to_tensor(self, img_f: np.ndarray) -> torch.Tensor:
        """float32 [H,W,3] → [1,3,H,W] tensor."""
        t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        """[1,3,H,W] tensor → float32 [H,W,3]."""
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # ── Main Inference ──────────────────────────────────────────────

    def enhance(self,
                image_source: Union[str, Path, np.ndarray],
                target_size: int = 128) -> Dict:
        """Run the full 5-stage ANFIS pipeline on a low-light face image.

        Args:
            image_source : Path to image or [H,W,3] uint8 numpy array.
            target_size  : LR face size before upsampling (default 32px,
                           producing 128px output via 4× upscaling).

        Returns:
            results : Dict with keys:
                'input'           — original LR input (uint8)
                'darkness_factor' — ANFIS DF estimate (float)
                'blur_info'       — blur estimation metadata
                'deblurred'       — after Stage 2 (float32)
                'enhanced'        — after Zero-DCE Stage 3a (float32)
                'lcr_output'      — after LCR Stage 3b (float32)
                'regression_output' — after regression Stage 4 (float32)
                'final_output'    — after RRDB Stage 5 (float32)
                'final_uint8'     — final output as uint8 (for display)
        """
        results = {}

        # ── Load input ────────────────────────────────────────────
        img_rgb = self._load_image(image_source)
        img_rgb = cv2.resize(img_rgb, (target_size // 4, target_size // 4))
        results['input'] = img_rgb

        # ── Stage 1: ANFIS Darkness Factor Estimation (Paper 3) ───
        if self.darkness_estimator._trained:
            df = self.darkness_estimator.estimate(img_rgb)
        else:
            # Fallback: heuristic brightness
            df = 1.0 - float(img_rgb.mean() / 255.0)
        results['darkness_factor'] = df
        print(f"  [Stage 1] Darkness Factor: {df:.3f}")

        current = img_rgb.astype(np.float32) / 255.0

        # ── Stage 2: Motion Blur Correction (Paper 5) ─────────────
        if self.use_blur_correction:
            deblurred, blur_info = self.blur_handler.process(img_rgb)
            results['blur_info']  = blur_info
            results['deblurred']  = deblurred
            print(f"  [Stage 2] Blur severity: {blur_info['blur_severity']:.3f}  "
                  f"{'(corrected)' if blur_info['corrected'] else '(no correction)'}")
            current = deblurred

        # ── Stage 3a: Zero-DCE Enhancement (Paper 2 neural) ────────
        current_t = self._to_tensor(current)
        with torch.no_grad():
            enhanced_t = self.zero_dce.enhance(current_t)
        enhanced_f = self._to_numpy(enhanced_t)

        # Scale enhancement by darkness factor:
        # Very dark → use full Zero-DCE output; bright → minimal blending
        current = df * enhanced_f + (1.0 - df) * current
        results['enhanced'] = current.copy()
        print(f"  [Stage 3a] Zero-DCE applied (DF-scaled blending: {df:.2f})")

        # ── Stage 3b: ANFIS-LCR Hallucination (Paper 1) ────────────
        if self.use_lcr and self.lcr_hallucinator is not None:
            lcr_out = self.lcr_hallucinator.hallucinate(current, df)
            results['lcr_output'] = lcr_out
            print(f"  [Stage 3b] LCR hallucination → {lcr_out.shape}")
            current_upscaled = lcr_out
        else:
            # Fallback: bicubic upscale
            H, W, C = current.shape
            current_upscaled = cv2.resize(
                current, (W * 4, H * 4),
                interpolation=cv2.INTER_LANCZOS4)
            results['lcr_output'] = current_upscaled
            print("  [Stage 3b] LCR skipped (no dictionary). Bicubic fallback.")

        # ── Stage 4: Regression Blending (Paper 4) ─────────────────
        if self.use_regression and self.regressor._trained:
            reg_out = self.regressor.reconstruct(img_rgb)
            reg_out = cv2.resize(reg_out, (current_upscaled.shape[1],
                                            current_upscaled.shape[0]))
            blended = blend_lcr_and_regression(current_upscaled, reg_out, df)
            results['regression_output'] = blended
            print(f"  [Stage 4] Regression blend (df={df:.2f})")
            current_upscaled = blended
        else:
            results['regression_output'] = current_upscaled
            print("  [Stage 4] Regression skipped (not trained).")

        # ── Stage 5: RRDB Neural Refinement ────────────────────────
        if self.use_rrdb:
            inp_t = self._to_tensor(current_upscaled)
            with torch.no_grad():
                final_t = self.rrdb.super_resolve(inp_t)
            final_f = self._to_numpy(final_t)
            results['final_output'] = final_f
            print(f"  [Stage 5] RRDB output → {final_f.shape}")
        else:
            results['final_output'] = current_upscaled

        # Convert final to uint8
        results['final_uint8'] = (results['final_output'] * 255).clip(
            0, 255).astype(np.uint8)

        return results

    def get_pipeline_info(self) -> dict:
        """Return pipeline configuration info for display."""
        return {
            'stages': [
                {'id': 1, 'name': 'ANFIS Darkness Estimation',
                 'paper': 'Paper 3', 'active': True},
                {'id': 2, 'name': 'Motion Blur Correction',
                 'paper': 'Paper 5', 'active': self.use_blur_correction},
                {'id': '3a', 'name': 'Zero-DCE Enhancement',
                 'paper': 'Paper 2', 'active': True},
                {'id': '3b', 'name': 'ANFIS-LCR Hallucination',
                 'paper': 'Paper 1', 'active': self.use_lcr},
                {'id': 4, 'name': 'Regression Blending',
                 'paper': 'Paper 4', 'active': self.use_regression},
                {'id': 5, 'name': 'RRDB Neural Refinement',
                 'paper': 'ESRGAN', 'active': self.use_rrdb},
            ],
            'device': self.device,
            'n_anfis_rules': self.darkness_estimator.model.get_rule_count(),
        }


# ── CLI Demo ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("ANFIS Face SR Pipeline — Demo")
    print("=" * 60)

    pipeline = ANFISFaceSRPipeline(
        device='cpu',
        use_blur_correction=True,
        use_lcr=True,
        use_regression=True,
        use_rrdb=True)

    pipeline.load_pretrained('checkpoints/')

    info = pipeline.get_pipeline_info()
    print("\n Pipeline stages:")
    for s in info['stages']:
        status = "✓" if s['active'] else "○"
        print(f"  {status} Stage {s['id']}: {s['name']}  [{s['paper']}]")

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"\nProcessing: {img_path}")
        results = pipeline.enhance(img_path)
        print(f"\nDarkness Factor: {results['darkness_factor']:.3f}")
        print(f"Final output:    {results['final_output'].shape}")
    else:
        print("\nRun:  python inference.py <path_to_face_image.jpg>")
