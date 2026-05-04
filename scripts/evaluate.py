"""
Batch Evaluation Script for ANFIS-LLFSR
=======================================

Evaluates the full ANFIS-LLFSR pipeline on a test set to generate
the PSNR, SSIM, and LPIPS metrics for the final report.

Usage:
    python scripts/evaluate.py --data_dir data/img_align_celeba --n_images 50
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

from app.backend.inference import ANFISFaceSRPipeline
from app.backend.evaluation.metrics import MetricsCalculator

def degrade_image(img):
    """Simulate low-light, blur, and noise on an HR image."""
    # This is a simple degradation for evaluation purposes
    h, w = img.shape[:2]
    # Resize to LR
    lr = cv2.resize(img, (w // 4, h // 4))
    
    # Darken
    gamma = np.random.uniform(2.0, 4.0)
    lr_dark = np.clip((lr / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    
    return lr_dark

def evaluate_pipeline(data_dir, n_images=50, output_dir='results'):
    """Run evaluation and save metrics to CSV."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(" ANFIS-LLFSR Evaluation Setup")
    print("=" * 60)
    
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} not found.")
        print("Please run scripts/download_data.py first.")
        return
        
    img_paths = list(data_path.glob('*.jpg'))
    if not img_paths:
        print(f"Error: No images found in {data_dir}.")
        return
        
    # Use images from the end of the dataset for testing
    test_paths = sorted(img_paths)[-n_images:]
    print(f"Found {len(test_paths)} test images.")
    
    # Initialize Pipeline
    pipeline = ANFISFaceSRPipeline(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_blur_correction=True,
        use_lcr=True,
        use_regression=True,
        use_rrdb=True
    )
    # pipeline.load_pretrained('app/backend/checkpoints/')
    
    # Initialize Metrics
    metrics_calc = MetricsCalculator(device=pipeline.device)
    
    results = []
    
    print("\nStarting Evaluation...")
    import torch
    
    # We will compute metrics for different stages to do an ablation study
    for i, p in enumerate(test_paths):
        img_hr = cv2.imread(str(p))
        if img_hr is None: continue
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_hr = cv2.resize(img_hr, (128, 128))
        
        # Create degraded LR image
        img_lr = degrade_image(img_hr)
        
        # Run pipeline
        try:
            out = pipeline.enhance(img_lr, target_size=128)
            sr_full = out['final_output']
            sr_lcr = out['lcr_output']
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue
            
        # Add batch dim for metrics calc
        hr_batch = np.expand_dims(img_hr.astype(np.float32) / 255.0, axis=0)
        
        # Ensure sr_full and sr_lcr are correct shapes
        if sr_full.shape != (128, 128, 3):
            sr_full = cv2.resize(sr_full, (128, 128))
        if sr_lcr.shape != (128, 128, 3):
            sr_lcr = cv2.resize(sr_lcr, (128, 128))
            
        sr_full_batch = np.expand_dims(sr_full, axis=0)
        sr_lcr_batch = np.expand_dims(sr_lcr, axis=0)
        
        # Compute metrics
        m_full = metrics_calc.compute_all_metrics(sr_full_batch, hr_batch)
        m_lcr = metrics_calc.compute_all_metrics(sr_lcr_batch, hr_batch)
        
        results.append({
            'Image': p.name,
            'DarknessFactor': out.get('darkness_factor', 0),
            'PSNR_LCR': m_lcr['psnr'],
            'SSIM_LCR': m_lcr['ssim'],
            'LPIPS_LCR': m_lcr['lpips'],
            'PSNR_Full': m_full['psnr'],
            'SSIM_Full': m_full['ssim'],
            'LPIPS_Full': m_full['lpips'],
        })
        
        # Save visualization for first 5 images
        if i < 5:
            vis = np.hstack([
                cv2.resize(img_lr, (128, 128)),
                (sr_lcr * 255).astype(np.uint8),
                (sr_full * 255).astype(np.uint8),
                img_hr
            ])
            cv2.imwrite(str(out_path / f"eval_vis_{i}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
        sys.stdout.write(f"\rProcessed {i+1}/{len(test_paths)}")
        sys.stdout.flush()
        
    print("\n\nEvaluation Complete!")
    df = pd.DataFrame(results)
    csv_path = out_path / 'evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print(" Average Results (Full Pipeline)")
    print("=" * 60)
    print(f" PSNR:  {df['PSNR_Full'].mean():.2f} dB")
    print(f" SSIM:  {df['SSIM_Full'].mean():.4f}")
    print(f" LPIPS: {df['LPIPS_Full'].mean():.4f}")
    print(f"\nDetailed metrics saved to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ANFIS Pipeline")
    parser.add_argument('--data_dir', type=str, default='data/img_align_celeba', help='Path to test images')
    parser.add_argument('--n_images', type=int, default=50, help='Number of images to evaluate')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    import torch # imported again in global scope just to be safe
    evaluate_pipeline(args.data_dir, args.n_images, args.output_dir)
