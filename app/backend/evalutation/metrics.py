"""Evaluation metrics for face super-resolution.

Implements:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Face Recognition Accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from pytorch_msssim import ssim as pytorch_ssim


class MetricsCalculator:
    """Calculator for image quality metrics."""
    
    def __init__(self, device='cpu'):
        """Initialize metrics calculator.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
    
    def compute_psnr(self, sr_images, hr_images):
        """Compute PSNR between SR and HR images.
        
        Args:
            sr_images: SR images [B, 3, H, W] or numpy [B, H, W, 3]
            hr_images: HR images [B, 3, H, W] or numpy [B, H, W, 3]
            
        Returns:
            Average PSNR in dB
        """
        if isinstance(sr_images, torch.Tensor):
            sr_images = sr_images.permute(0, 2, 3, 1).cpu().numpy()
            hr_images = hr_images.permute(0, 2, 3, 1).cpu().numpy()
            sr_images = sr_images.astype(np.float32)
            hr_images = hr_images.astype(np.float32)
        
        psnr_values = []
        for i in range(sr_images.shape[0]):
            psnr = peak_signal_noise_ratio(
                hr_images[i],
                sr_images[i],
                data_range=1.0
            )
            psnr_values.append(psnr)
        
        return np.mean(psnr_values)
    
    def compute_ssim(self, sr_images, hr_images):
        """Compute SSIM between SR and HR images.
        
        Args:
            sr_images: SR images [B, 3, H, W] or numpy [B, H, W, 3]
            hr_images: HR images [B, 3, H, W] or numpy [B, H, W, 3]
            
        Returns:
            Average SSIM
        """
        if isinstance(sr_images, torch.Tensor):
            # Use PyTorch SSIM for batched computation
            ssim_val = pytorch_ssim(sr_images, hr_images, data_range=1.0)
            return ssim_val.item()
        else:
            # Use scikit-image SSIM
            ssim_values = []
            for i in range(sr_images.shape[0]):
                ssim_val = structural_similarity(
                    hr_images[i],
                    sr_images[i],
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
    
    def compute_lpips(self, sr_images, hr_images):
        """Compute LPIPS (perceptual distance).
        
        Args:
            sr_images: SR images [B, 3, H, W] in range [0, 1]
            hr_images: HR images [B, 3, H, W] in range [0, 1]
            
        Returns:
            Average LPIPS (lower is better)
        """
        if not isinstance(sr_images, torch.Tensor):
            sr_images = torch.from_numpy(sr_images).permute(0, 3, 1, 2).float()
            hr_images = torch.from_numpy(hr_images).permute(0, 3, 1, 2).float()
        
        sr_images = sr_images.to(self.device).float()
        hr_images = hr_images.to(self.device).float()
        
        # LPIPS expects input in range [-1, 1]
        sr_normalized = sr_images * 2 - 1
        hr_normalized = hr_images * 2 - 1
        
        with torch.no_grad():
            lpips_val = self.lpips_model(sr_normalized, hr_normalized)
        
        return lpips_val.mean().item()
    
    def compute_all_metrics(self, sr_images, hr_images):
        """Compute all metrics.
        
        Args:
            sr_images: SR images [B, 3, H, W]
            hr_images: HR images [B, 3, H, W]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'psnr': self.compute_psnr(sr_images, hr_images),
            'ssim': self.compute_ssim(sr_images, hr_images),
            'lpips': self.compute_lpips(sr_images, hr_images)
        }
        
        return metrics


class FaceRecognitionEvaluator:
    """Evaluator for face recognition accuracy."""
    
    def __init__(self, arcface_model):
        """Initialize evaluator.
        
        Args:
            arcface_model: ArcFaceModel instance
        """
        self.arcface = arcface_model
    
    def compute_face_accuracy(self, lr_images, sr_images, hr_images, 
                             threshold=0.5):
        """Compute face recognition accuracy.
        
        Measures how well SR improves face recognition compared to LR.
        
        Args:
            lr_images: LR images [B, 3, H, W]
            sr_images: SR images [B, 3, H, W]
            hr_images: HR images [B, 3, H, W]
            threshold: Similarity threshold for matching
            
        Returns:
            Dictionary with accuracies
        """
        batch_size = hr_images.shape[0]
        
        lr_correct = 0
        sr_correct = 0
        total = 0
        
        for i in range(batch_size):
            # Extract embeddings
            emb_hr = self.arcface.extract_embedding(hr_images[i])
            emb_lr = self.arcface.extract_embedding(lr_images[i])
            emb_sr = self.arcface.extract_embedding(sr_images[i])
            
            if emb_hr is None:
                continue
            
            total += 1
            
            # Compute similarities
            if emb_lr is not None:
                sim_lr = self.arcface.cosine_similarity(emb_lr, emb_hr)
                if sim_lr > threshold:
                    lr_correct += 1
            
            if emb_sr is not None:
                sim_sr = self.arcface.cosine_similarity(emb_sr, emb_hr)
                if sim_sr > threshold:
                    sr_correct += 1
        
        if total == 0:
            return {'lr_acc': 0.0, 'sr_acc': 0.0, 'improvement': 0.0}
        
        lr_acc = lr_correct / total
        sr_acc = sr_correct / total
        improvement = sr_acc - lr_acc
        
        return {
            'lr_acc': lr_acc * 100,  # Percentage
            'sr_acc': sr_acc * 100,
            'improvement': improvement * 100
        }


def evaluate_model(generator, enhancer, test_loader, metrics_calc, 
                  face_evaluator, device='cpu', max_batches=None):
    """Evaluate model on test set.
    
    Args:
        generator: SR generator model
        enhancer: Zero-DCE enhancer
        test_loader: Test data loader
        metrics_calc: MetricsCalculator instance
        face_evaluator: FaceRecognitionEvaluator instance
        device: Device
        max_batches: Maximum batches to evaluate (None for all)
        
    Returns:
        Dictionary of averaged metrics
    """
    generator.eval()
    if enhancer:
        enhancer.model.eval()
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    lr_accs = []
    sr_accs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            lr_images = batch['lr'].to(device)
            hr_images = batch['hr'].to(device)
            
            # Enhance + Super-resolve
            if enhancer:
                enhanced = enhancer.model(lr_images)
                sr_images = generator(enhanced)
            else:
                sr_images = generator(lr_images)
            
            # Compute metrics
            batch_metrics = metrics_calc.compute_all_metrics(sr_images, hr_images)
            all_psnr.append(batch_metrics['psnr'])
            all_ssim.append(batch_metrics['ssim'])
            all_lpips.append(batch_metrics['lpips'])
            
            # Face recognition accuracy (sample subset for speed)
            if batch_idx % 5 == 0 and face_evaluator:
                face_metrics = face_evaluator.compute_face_accuracy(
                    lr_images, sr_images, hr_images
                )
                lr_accs.append(face_metrics['lr_acc'])
                sr_accs.append(face_metrics['sr_acc'])
    
    results = {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'lpips': np.mean(all_lpips),
    }
    
    if lr_accs and sr_accs:
        results['lr_face_acc'] = np.mean(lr_accs)
        results['sr_face_acc'] = np.mean(sr_accs)
        results['face_acc_improvement'] = results['sr_face_acc'] - results['lr_face_acc']
    
    return results
