"""Visualization utilities for training and results."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import cv2


class Visualizer:
    """Utilities for visualizing training progress and results."""
    
    def __init__(self, output_dir='/app/backend/outputs/visualizations'):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, history, title='Training History', 
                           filename='training_curves.png'):
        """Plot training loss/metric curves.
        
        Args:
            history: Dictionary with lists of values (e.g., {'loss': [...]})  
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(1, len(history), figsize=(6*len(history), 5))
        
        if len(history) == 1:
            axes = [axes]
        
        for idx, (key, values) in enumerate(history.items()):
            axes[idx].plot(values)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].set_xlabel('Iteration / Epoch')
            axes[idx].set_ylabel(key)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves to {save_path}")
    
    def visualize_comparison(self, lr_image, sr_image, hr_image,
                           filename='comparison.png'):
        """Visualize LR, SR, and HR comparison.
        
        Args:
            lr_image: LR image tensor [3, H, W] or numpy [H, W, 3]
            sr_image: SR image tensor
            hr_image: HR image tensor
            filename: Output filename
        """
        # Convert tensors to numpy
        if isinstance(lr_image, torch.Tensor):
            lr_image = lr_image.permute(1, 2, 0).cpu().numpy()
            sr_image = sr_image.permute(1, 2, 0).cpu().numpy()
            hr_image = hr_image.permute(1, 2, 0).cpu().numpy()
        
        lr_image = np.clip(lr_image, 0, 1)
        sr_image = np.clip(sr_image, 0, 1)
        hr_image = np.clip(hr_image, 0, 1)
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_image)
        axes[0].set_title('Low-Resolution Input', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(sr_image)
        axes[1].set_title('Super-Resolved (Ours)', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(hr_image)
        axes[2].set_title('Ground Truth HR', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_ablation_table(self, results, filename='ablation_table.txt'):
        """Create ablation study table.
        
        Args:
            results: List of dicts with configuration and metrics
            filename: Output filename
        """
        save_path = self.output_dir / filename
        
        with open(save_path, 'w') as f:
            f.write("Ablation Study Results\n")
            f.write("=" * 80 + "\n\n")
            
            # Header
            f.write(f"{'Configuration':<40} {'PSNR':>10} {'SSIM':>10} {'Face Acc':>12}\n")
            f.write("-" * 80 + "\n")
            
            # Results
            for result in results:
                config = result['config']
                psnr = result.get('psnr', 0)
                ssim = result.get('ssim', 0)
                face_acc = result.get('face_acc', 0)
                
                f.write(f"{config:<40} {psnr:>10.2f} {ssim:>10.3f} {face_acc:>11.1f}%\n")
        
        print(f"Saved ablation table to {save_path}")
    
    def save_image_grid(self, images, filename='grid.png', nrow=4):
        """Save grid of images.
        
        Args:
            images: List of image tensors [3, H, W]
            filename: Output filename
            nrow: Number of images per row
        """
        from torchvision.utils import make_grid
        
        if isinstance(images, list):
            images = torch.stack(images)
        
        grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(15, 15))
        plt.imshow(grid_np)
        plt.axis('off')
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return save_path


def tensor_to_image(tensor, denormalize=False):
    """Convert tensor to displayable image.
    
    Args:
        tensor: Image tensor [3, H, W] or [B, 3, H, W]
        denormalize: Whether to denormalize from [-1, 1]
        
    Returns:
        Numpy image [H, W, 3] in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    if denormalize:
        tensor = (tensor + 1) / 2
    
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image
