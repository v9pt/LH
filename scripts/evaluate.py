"""
Batch Evaluation Script for ANFIS-LLFSR
=======================================

Evaluates the full ANFIS-LLFSR pipeline on a test set to generate
the PSNR, SSIM, and LPIPS metrics for the final report.

Usage:
    python scripts/evaluate.py --data_dir data/img_align_celeba --n_images 50
"""

import sys, os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'app', 'backend'))

import argparse
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import torch

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

def evaluate_pipeline(data_dirs, n_images=50, output_dir='results'):
    """Run evaluation and save metrics to CSV."""
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
        
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    target_size = 128
    
    print("=" * 60)
    print(" ANFIS-LLFSR Evaluation Setup")
    print("=" * 60)
    
    img_paths = []
    for d in data_dirs:
        d_path = Path(d)
        if d_path.exists():
            img_paths.extend(list(d_path.glob('*.jpg')) + list(d_path.glob('*.png')))
            
    if not img_paths:
        print(f"Error: No images found in {data_dirs}.")
        return
        
    # Use images from the end of the dataset for testing
    test_paths = sorted(img_paths)[-n_images:]
    print(f"Found {len(test_paths)} test images.")
    
    # Initialize Pipeline
    pipeline = ANFISFaceSRPipeline(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_blur_correction=True,
        use_gfpgan=True
    )
    # pipeline.load_pretrained('app/backend/checkpoints/')
    
    # Initialize Metrics
    metrics_calc = MetricsCalculator(device=pipeline.device)
    
    results = []
    
    print("\nStarting Evaluation...")
    
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
            sr_lcr = out['enhanced']
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
            'PSNR_Full': m_full['psnr'],
            'SSIM_Full': m_full['ssim'],
            'LPIPS_Full': m_full['lpips'],
            'NIQE_Full': m_full['niqe'],
            'MUSIQ_Full': m_full['musiq'],
            'ID_Sim_Full': m_full['id_sim'],
            'PSNR_ZeroDCE': m_lcr['psnr'],
            'SSIM_ZeroDCE': m_lcr['ssim'],
            'LPIPS_ZeroDCE': m_lcr['lpips'],
        })
        
        # Save SR images for FID calculation later
        sr_save_path = out_path / 'sr_temp'
        sr_save_path.mkdir(exist_ok=True)
        cv2.imwrite(str(sr_save_path / p.name), cv2.cvtColor((sr_full * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Save HR images for FID calculation later (ensure they are aligned/resized)
        hr_save_path = out_path / 'hr_temp'
        hr_save_path.mkdir(exist_ok=True)
        cv2.imwrite(str(hr_save_path / p.name), cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR))
        
        # Save visualization for first 5 images
        if i < 5:
            # Prepare segments
            in_img = cv2.cvtColor(out['input'], cv2.COLOR_RGB2BGR)
            in_img = cv2.resize(in_img, (target_size, target_size))
            
            # Fuzzy Attention Map (Color coded)
            attn_map = out.get('fuzzy_attention', np.zeros_like(sr_full[:,:,0]))
            attn_u8 = (attn_map * 255).astype(np.uint8)
            attn_vis = cv2.applyColorMap(attn_u8, cv2.COLORMAP_VIRIDIS)
            attn_vis = cv2.resize(attn_vis, (target_size, target_size))
            
            # Phase 2 Enhanced
            enh_img = cv2.cvtColor((out['enhanced'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            enh_img = cv2.resize(enh_img, (target_size, target_size))
            
            # Phase 3 Final SR
            sr_img = cv2.cvtColor((sr_full * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Ground Truth
            gt_img = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
            gt_img = cv2.resize(gt_img, (target_size, target_size))
            
            # Frequency Spectrum of SR
            fft_sr = metrics_calc.visualize_frequency_spectrum(sr_full)
            fft_sr_bgr = cv2.cvtColor(fft_sr, cv2.COLOR_RGB2BGR)
            
            # Combine into a stunning research strip
            vis = np.hstack([in_img, attn_vis, enh_img, sr_img, gt_img, fft_sr_bgr])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis, "Input", (10, 30), font, 0.8, (255,255,255), 2)
            cv2.putText(vis, "Fuzzy-Attn", (target_size + 10, 30), font, 0.8, (255,255,255), 2)
            cv2.putText(vis, "Enhanced", (target_size*2 + 10, 30), font, 0.8, (255,255,255), 2)
            cv2.putText(vis, "Final SR", (target_size*3 + 10, 30), font, 0.8, (255,255,255), 2)
            cv2.putText(vis, "GT", (target_size*4 + 10, 30), font, 0.8, (255,255,255), 2)
            cv2.putText(vis, "FFT-Spec", (target_size*5 + 10, 30), font, 0.8, (255,255,255), 2)

            cv2.imwrite(str(out_path / f'eval_vis_{p.stem}.jpg'), vis)
            
        sys.stdout.write(f"\rProcessed {i+1}/{len(test_paths)}")
        sys.stdout.flush()
        
    print("\n\nEvaluation Complete!")
    df = pd.DataFrame(results)
    
    # Compute FID at the end
    print("Computing FID (distribution distance)...")
    try:
        fid_score = metrics_calc.compute_fid(str(out_path / 'hr_temp'), str(out_path / 'sr_temp'))
    except Exception as e:
        print(f"FID calculation failed (requires more images or clean-fid setup): {e}")
        fid_score = float('nan')
        
    csv_path = out_path / 'evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print(" Average Results (Full Pipeline) — Industry Grade")
    print("=" * 60)
    print(f" PSNR:   {df['PSNR_Full'].mean():.2f} dB")
    print(f" SSIM:   {df['SSIM_Full'].mean():.4f}")
    print(f" LPIPS:  {df['LPIPS_Full'].mean():.4f} (Perceptual)")
    print(f" NIQE:   {df['NIQE_Full'].mean():.4f} (Naturalness - Lower is better)")
    print(f" MUSIQ:  {df['MUSIQ_Full'].mean():.2f} (Blind Quality - Higher is better)")
    print(f" ID_Sim: {df['ID_Sim_Full'].mean():.4f} (Identity Similarity - Higher is better)")
    print(f" FID:    {fid_score:.2f} (Lower is better)")
    print(f"\nDetailed metrics saved to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ANFIS Pipeline")
    parser.add_argument('--data_dirs', nargs='+', default=['data/img_align_celeba', 'data/ffhq'], help='Paths to test images')
    parser.add_argument('--n_images', type=int, default=50, help='Number of images to evaluate')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    evaluate_pipeline(args.data_dirs, args.n_images, args.output_dir)
