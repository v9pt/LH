#!/usr/bin/env python3
"""
ANFIS-LLFSR: Full Training + Evaluation Script
================================================

Trains all pipeline components on local CelebA data, then evaluates
the full pipeline on a held-out test set.

Outputs:
    checkpoints/                 — all trained model weights
    results/evaluation_report.txt  — human-readable metrics table
    results/evaluation_metrics.csv — per-image metrics for analysis
    results/eval_vis_*.jpg       — visual LR | LCR | Full | HR comparisons

Usage (from project root):
    cd /Users/zaif/Desktop/BTPPF/LH
    python scripts/run_training_and_eval.py

Options:
    --data_dir      Path to CelebA images  [default: data/img_align_celeba]
    --train_n       ANFIS synthetic samples [default: 5000]
    --anfis_epochs  ANFIS training epochs   [default: 200]
    --dict_images   Images for LCR dict     [default: 2000]
    --reg_images    Images for regressors   [default: 800]
    --eval_images   Images for evaluation   [default: 100]
    --ckpt_dir      Checkpoint directory    [default: app/backend/checkpoints]
"""

import sys, os
# Allow imports from both the project root and app/backend
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, 'app', 'backend')
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

import argparse
import time
import json
import numpy as np
import cv2
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import (mean_squared_error, r2_score,
                              mean_absolute_error, confusion_matrix,
                              classification_report)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import zipfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Project imports
from app.backend.core.darkness_estimator import (DarknessEstimator,
                                      generate_synthetic_training_data)
from app.backend.inference import ANFISFaceSRPipeline
from app.backend.models.arcface_model import ArcFaceModel
from app.backend.evaluation.metrics import FaceRecognitionEvaluator

# ─── Colours for terminal output ───────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def hdr(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def ok(text):  print(f"  {GREEN}✓{RESET} {text}")
def warn(text): print(f"  {YELLOW}⚠{RESET} {text}")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def ensure_data_unzipped(data_dir: Path, zip_path: Path):
    """Ensure data is extracted if zip is available."""
    if data_dir.exists() and any(data_dir.iterdir()):
        return # Data already exists
    if not zip_path.exists():
        warn(f"Data directory {data_dir} is empty, and zip {zip_path} not found.")
        return
    
    print(f"\n{BOLD}{CYAN}Extracting {zip_path} to {data_dir}...{RESET}")
    # Extract only to data_dir directly or appropriately
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # CelebA zip usually contains a folder 'img_align_celeba/'
        # Let's extract everything to the parent of data_dir
        zip_ref.extractall(data_dir.parent)
    ok(f"Extraction complete.")

def load_celeba_images(data_dir: Path, n: int, offset: int = 0):
    """Load up to n CelebA images starting from offset, resized to 128×128."""
    paths = sorted(data_dir.glob('*.jpg'))[offset:offset + n]
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)
    return images


def degrade_for_eval(hr_img: np.ndarray,
                     lr_size: int = 64,
                     gamma: float = None,
                     rng=None) -> np.ndarray:
    """Simulate 8× low-light degradation on an HR image."""
    if rng is None:
        rng = np.random.default_rng(42)
    g = gamma if gamma is not None else rng.uniform(1.5, 3.5)
    lr = cv2.resize(hr_img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
    # Apply gamma darkening
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(lr, lut)


def psnr(gt, pred):
    gt_f   = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    return peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0)


def ssim(gt, pred):
    gt_f   = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    return structural_similarity(gt_f, pred_f, channel_axis=2, data_range=1.0)


def try_lpips(sr_f, hr_f):
    """Return LPIPS score, or None if lpips is not installed."""
    try:
        import lpips as lpips_lib
        if not hasattr(try_lpips, '_model'):
            try_lpips._model = lpips_lib.LPIPS(net='alex')
            try_lpips._model.eval()
        sr_t = torch.from_numpy(sr_f).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        hr_t = torch.from_numpy(hr_f).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            return try_lpips._model(sr_t, hr_t).item()
    except Exception:
        return None


# ─── Phase 1: ANFIS Darkness Estimator ────────────────────────────────────────

def train_anfis(data_dir: Path, ckpt_dir: Path,
                n_samples: int, epochs: int) -> DarknessEstimator:
    hdr("Phase 1 / 3 — ANFIS Darkness Estimator Training")

    de = DarknessEstimator(n_mfs=3, lr=1e-3, device='cpu')

    if data_dir.exists() and len(list(data_dir.glob('*.jpg'))) > 100:
        ok(f"CelebA found at {data_dir}. Generating {n_samples} synthetic pairs...")
        history = de.train(str(data_dir), n_samples=n_samples,
                           epochs=epochs, verbose=True)
        ok(f"ANFIS trained. Final MSE = {history[-1]:.6f}")
    else:
        warn("CelebA not found. Using pure-synthetic fallback training data.")
        rng = np.random.default_rng(42)
        gammas = rng.uniform(1.0, 5.0, n_samples).astype(np.float32)
        targets = ((gammas - 1.0) / 4.0).reshape(-1, 1)
        X = np.column_stack([
            1.0 - (gammas / 5.0) + rng.normal(0, 0.05, n_samples),
            0.3  - (gammas / 20)  + rng.normal(0, 0.02, n_samples),
            (gammas / 5.0)         + rng.normal(0, 0.05, n_samples),
            1.0 - (gammas / 8.0)  + rng.normal(0, 0.05, n_samples),
        ]).astype(np.float32)
        X = np.clip(X, 0, 1)
        history = de.train_from_arrays(X, targets, epochs=epochs, verbose=True)
        ok(f"Synthetic ANFIS trained. Final MSE = {history[-1]:.6f}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    de.save(ckpt_dir / 'darkness_estimator.pt')
    ok(f"Saved → {ckpt_dir / 'darkness_estimator.pt'}")

    # ── ANFIS Evaluation (classification into 4 darkness levels) ──
    print(f"\n  {BOLD}ANFIS Darkness Estimator — Validation Metrics{RESET}")
    rng2 = np.random.default_rng(123)
    n_val = 500
    gammas_val = rng2.uniform(1.0, 5.0, n_val).astype(np.float32)
    df_true    = (gammas_val - 1.0) / 4.0

    # Normalise features same way as training
    from app.backend.core.darkness_estimator import extract_illumination_features
    df_pred = []
    for g in gammas_val:
        # Synthetic feature vector
        f = np.array([
            max(0, 1.0 - g / 5.0 + rng2.normal(0, 0.05)),
            max(0, 0.3  - g / 20  + rng2.normal(0, 0.02)),
            min(1, g / 5.0         + rng2.normal(0, 0.05)),
            max(0, 1.0 - g / 8.0  + rng2.normal(0, 0.05)),
        ], dtype=np.float32)
        f_norm = (f - de._feature_mean) / de._feature_std
        import torch as _t
        x_t = _t.from_numpy(f_norm).unsqueeze(0)
        de.model.eval()
        with _t.no_grad():
            raw = de.model(x_t).item()
        import math
        df_val = float(np.clip(1.0 / (1.0 + math.exp(-raw * 4.0)), 0, 1))
        df_pred.append(df_val)

    df_pred = np.array(df_pred)

    mse  = mean_squared_error(df_true, df_pred)
    mae  = mean_absolute_error(df_true, df_pred)
    r2   = r2_score(df_true, df_pred)

    # Bin into 4 classes: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
    bins    = [0, 0.25, 0.5, 0.75, 1.01]
    labels  = ['Well-lit', 'Moderate', 'Dark', 'Very Dark']
    y_true_cls = np.digitize(df_true, bins[1:])
    y_pred_cls = np.digitize(df_pred, bins[1:])

    acc = (y_true_cls == y_pred_cls).mean()
    cr  = classification_report(y_true_cls, y_pred_cls,
                                 target_names=labels, digits=3,
                                 zero_division=0)

    print(f"\n  Regression metrics (500 validation pairs):")
    print(f"    MSE  = {mse:.6f}")
    print(f"    MAE  = {mae:.6f}")
    print(f"    R²   = {r2:.4f}")
    print(f"\n  Classification metrics (4 darkness levels):")
    print(f"    Accuracy = {acc*100:.2f}%")
    print(f"\n{cr}")

    return de, {
        'anfis_mse': round(mse, 6),
        'anfis_mae': round(mae, 6),
        'anfis_r2':  round(r2, 4),
        'anfis_accuracy_pct': round(acc * 100, 2),
        'classification_report': cr,
    }


# ─── Phase 4: Full Pipeline Evaluation ────────────────────────────────────────

def evaluate_pipeline(data_dir: Path, ckpt_dir: Path,
                      n_eval: int, out_dir: Path) -> dict:
    hdr("Phase 4 — Full Pipeline Evaluation")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the pipeline (uses just-trained checkpoints)
    print("  Loading pipeline from checkpoints...")
    pipeline = ANFISFaceSRPipeline(
        device='cpu',
        use_blur_correction=True,
        use_gfpgan=True
    )
    pipeline.load_pretrained(str(ckpt_dir))
    ok("Pipeline loaded.")

    # Use the last n_eval images as test set (not seen during training)
    if data_dir.exists():
        all_paths = sorted(data_dir.glob('*.jpg'))
        test_paths = all_paths[-n_eval:]
    else:
        warn("No CelebA data. Cannot run full pipeline evaluation.")
        return {}

    print(f"\n  Evaluating on {len(test_paths)} test images...")
    print(f"  (Metrics: PSNR, SSIM, LPIPS, Face Recognition Accuracy)")

    rng   = np.random.default_rng(42)
    
    # Initialize Face Evaluator
    print("  Initializing Face Recognition Model (ArcFace)...")
    try:
        arcface = ArcFaceModel(device='cpu')
        face_evaluator = FaceRecognitionEvaluator(arcface)
        ok("ArcFace loaded.")
    except Exception as e:
        warn(f"Could not load ArcFace: {e}. Face metrics will be skipped.")
        face_evaluator = None
    rows  = []
    psnr_bicubic_list, ssim_bicubic_list = [], []
    psnr_full_list,    ssim_full_list    = [], []
    lpips_full_list = []
    
    # Face acc tracking
    face_lr_correct, face_bic_correct, face_sr_correct, face_total = 0, 0, 0, 0

    TARGET_OUTPUT = 512   # 8× pipeline output size

    for i, p in enumerate(test_paths):
        hr_img = cv2.imread(str(p))
        if hr_img is None:
            continue
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.resize(hr_img, (TARGET_OUTPUT, TARGET_OUTPUT))

        # Create LR (64×64 dark) input
        lr_img = degrade_for_eval(hr_img, lr_size=64, rng=rng)

        # Bicubic baseline (64→512)
        bicubic = cv2.resize(lr_img, (TARGET_OUTPUT, TARGET_OUTPUT),
                             interpolation=cv2.INTER_CUBIC)

        # Full pipeline
        try:
            result  = pipeline.enhance(lr_img, target_size=TARGET_OUTPUT)
            sr_u8   = result['final_uint8']
            df_val  = result['darkness_factor']
        except Exception as e:
            warn(f"Pipeline error on {p.name}: {e}")
            continue

        # Resize to same target for fair comparison
        sr_512  = cv2.resize(sr_u8,  (TARGET_OUTPUT, TARGET_OUTPUT))
        bic_512 = cv2.resize(bicubic, (TARGET_OUTPUT, TARGET_OUTPUT))

        # Metrics
        p_bic = psnr(hr_img, bic_512)
        s_bic = ssim(hr_img, bic_512)
        p_ful = psnr(hr_img, sr_512)
        s_ful = ssim(hr_img, sr_512)

        hr_f  = hr_img.astype(np.float32)  / 255.0
        sr_f  = sr_512.astype(np.float32)  / 255.0
        lp    = try_lpips(sr_f, hr_f)

        psnr_bicubic_list.append(p_bic)
        ssim_bicubic_list.append(s_bic)
        psnr_full_list.append(p_ful)
        ssim_full_list.append(s_ful)
        if lp is not None:
            lpips_full_list.append(lp)
            
        # Face Verification
        if face_evaluator is not None:
            # Expand dims to batch size 1, input to extract_embeddings_batch is [B, 3, H, W]
            # Wait, extract_embeddings accepts numpy [H, W, 3] directly
            emb_hr = face_evaluator.arcface.extract_embedding(hr_img)
            if emb_hr is not None:
                face_total += 1
                
                emb_lr = face_evaluator.arcface.extract_embedding(cv2.resize(lr_img, (TARGET_OUTPUT, TARGET_OUTPUT)))
                if emb_lr is not None and face_evaluator.arcface.cosine_similarity(emb_lr, emb_hr) > 0.5:
                    face_lr_correct += 1
                    
                emb_bic = face_evaluator.arcface.extract_embedding(bic_512)
                if emb_bic is not None and face_evaluator.arcface.cosine_similarity(emb_bic, emb_hr) > 0.5:
                    face_bic_correct += 1
                    
                emb_sr = face_evaluator.arcface.extract_embedding(sr_512)
                if emb_sr is not None and face_evaluator.arcface.cosine_similarity(emb_sr, emb_hr) > 0.5:
                    face_sr_correct += 1

        rows.append({
            'image':          p.name,
            'darkness_factor': round(df_val, 4),
            'psnr_bicubic':   round(p_bic, 3),
            'ssim_bicubic':   round(s_bic, 4),
            'psnr_full':      round(p_ful, 3),
            'ssim_full':      round(s_ful, 4),
            'lpips_full':     round(lp, 4) if lp is not None else None,
            'psnr_gain':      round(p_ful - p_bic, 3),
            'ssim_gain':      round(s_ful - s_bic, 4),
        })

        # Save visual for first 8 images
        if i < 8:
            lr_disp = cv2.resize(lr_img, (TARGET_OUTPUT, TARGET_OUTPUT),
                                 interpolation=cv2.INTER_NEAREST)
            vis = np.hstack([lr_disp, bic_512, sr_512, hr_img])
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f'eval_vis_{i:02d}.jpg'), vis_bgr)

        sys.stdout.write(f"\r  Progress: {i+1}/{len(test_paths)}  "
                         f"PSNR={p_ful:.2f}dB SSIM={s_ful:.4f}")
        sys.stdout.flush()

    print()  # newline after progress

    # Aggregate
    results = {
        'n_evaluated':       len(rows),
        'psnr_bicubic':      round(np.mean(psnr_bicubic_list), 3),
        'ssim_bicubic':      round(np.mean(ssim_bicubic_list), 4),
        'psnr_full':         round(np.mean(psnr_full_list), 3),
        'ssim_full':         round(np.mean(ssim_full_list), 4),
        'psnr_gain':         round(np.mean(psnr_full_list) - np.mean(psnr_bicubic_list), 3),
        'ssim_gain':         round(np.mean(ssim_full_list) - np.mean(ssim_bicubic_list), 4),
    }
    if lpips_full_list:
        results['lpips_full'] = round(np.mean(lpips_full_list), 4)
        
    if face_total > 0:
        results['face_lr_acc'] = round(face_lr_correct / face_total * 100, 2)
        results['face_bic_acc'] = round(face_bic_correct / face_total * 100, 2)
        results['face_sr_acc'] = round(face_sr_correct / face_total * 100, 2)
        results['face_acc_gain'] = round(results['face_sr_acc'] - results['face_bic_acc'], 2)

    # Save CSV
    df_csv = pd.DataFrame(rows)
    csv_path = out_dir / 'evaluation_metrics.csv'
    df_csv.to_csv(csv_path, index=False)
    ok(f"Per-image metrics saved → {csv_path}")

    return results


# ─── Final Report ──────────────────────────────────────────────────────────────

def print_report(anfis_metrics: dict, sr_metrics: dict, out_dir: Path):
    report_lines = []

    report_lines += [
        "=" * 60,
        "  ANFIS-LLFSR — Training & Evaluation Report",
        "=" * 60,
        "",
        "─── ANFIS Darkness Estimator (Paper 3) ───────────────────────",
        f"  Regression MSE      : {anfis_metrics.get('anfis_mse', 'N/A')}",
        f"  Regression MAE      : {anfis_metrics.get('anfis_mae', 'N/A')}",
        f"  Regression R²       : {anfis_metrics.get('anfis_r2', 'N/A')}",
        f"  Classification Acc  : {anfis_metrics.get('anfis_accuracy_pct', 'N/A')} %",
        "",
        "  Per-Class F1 Scores:",
        anfis_metrics.get('classification_report', '  (not available)'),
        "",
        "─── Super-Resolution Evaluation (8× scale: 64→512) ──────────",
        f"  Images evaluated    : {sr_metrics.get('n_evaluated', 'N/A')}",
        "",
        f"  {'Metric':<25} {'Bicubic':>12} {'ANFIS-LLFSR':>12} {'Gain':>10}",
        f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}",
        f"  {'PSNR (dB)':<25} {sr_metrics.get('psnr_bicubic', '-'):>12.3f} {sr_metrics.get('psnr_full', '-'):>12.3f} {sr_metrics.get('psnr_gain', '-'):>+10.3f}",
        f"  {'SSIM':<25} {sr_metrics.get('ssim_bicubic', '-'):>12.4f} {sr_metrics.get('ssim_full', '-'):>12.4f} {sr_metrics.get('ssim_gain', '-'):>+10.4f}",
    ]

    if 'lpips_full' in sr_metrics:
        report_lines.append(
            f"  {'LPIPS (↓ better)':<25} {'—':>12} {sr_metrics['lpips_full']:>12.4f} {'—':>10}"
        )
        
    if 'face_sr_acc' in sr_metrics:
        report_lines.append(
            f"  {'Face Recog Acc (%)':<25} {sr_metrics.get('face_bic_acc', '-'):>12.2f} {sr_metrics.get('face_sr_acc', '-'):>12.2f} {sr_metrics.get('face_acc_gain', '-'):>+10.2f}"
        )

    report_lines += [
        "",
        "─── Visual Comparisons ───────────────────────────────────────",
        "  See results/eval_vis_*.jpg  (columns: LR | Bicubic | Ours | GT)",
        "",
        "─── Saved Checkpoints ────────────────────────────────────────",
        "  app/backend/checkpoints/darkness_estimator.pt",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    print(f"\n{BOLD}" + report_text + RESET)

    report_path = out_dir / 'evaluation_report.txt'
    report_path.write_text(report_text)
    ok(f"Full report saved → {report_path}")

    # Also save metrics as JSON for programmatic access
    all_metrics = {**anfis_metrics, **sr_metrics}
    all_metrics.pop('classification_report', None)  # not JSON-serialisable cleanly
    
    for k, v in all_metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            all_metrics[k] = float(v) if isinstance(v, np.floating) else int(v)
            
    json_path = out_dir / 'metrics_summary.json'
    json_path.write_text(json.dumps(all_metrics, indent=2))
    ok(f"Metrics JSON saved → {json_path}")


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ANFIS-LLFSR 2.0: Train ANFIS and evaluate the SOTA pipeline")
    parser.add_argument('--data_dir',     default='data/img_align_celeba')
    parser.add_argument('--ckpt_dir',     default='app/backend/checkpoints')
    parser.add_argument('--output_dir',   default='results')
    parser.add_argument('--train_n',      type=int, default=5000)
    parser.add_argument('--anfis_epochs', type=int, default=200)
    parser.add_argument('--eval_images',  type=int, default=100)
    parser.add_argument('--zip_path',     default='/Users/zaif/Desktop/BTPPF/LH/img_align_celeba.zip')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.output_dir)
    zip_path = Path(args.zip_path)

    total_start = time.time()

    print(f"\n{BOLD}{CYAN}ANFIS-LLFSR Training & Evaluation Pipeline{RESET}")
    print(f"  Data   : {data_dir}")
    print(f"  Checkpoints : {ckpt_dir}")
    print(f"  Results     : {out_dir}")

    ensure_data_unzipped(data_dir, zip_path)

    # Phase 1 — ANFIS
    de, anfis_metrics = train_anfis(data_dir, ckpt_dir,
                                    args.train_n, args.anfis_epochs)

    # Phase 2 — Full Pipeline Evaluation
    sr_metrics = evaluate_pipeline(data_dir, ckpt_dir,
                                   args.eval_images, out_dir)

    # Final Report
    print_report(anfis_metrics, sr_metrics, out_dir)

    total_min = (time.time() - total_start) / 60
    print(f"\n{GREEN}{BOLD}✓ Complete! Total time: {total_min:.1f} minutes{RESET}")


if __name__ == '__main__':
    main()
