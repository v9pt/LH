import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.backend.models.swin_fuzzy_lcr import SwinFuzzyLCR
from app.backend.models.losses import HybridLossCombiner
from app.backend.models.discriminator import PatchGANDiscriminator
from app.backend.data.dataset import FaceSRDataset
from app.backend.data.preprocessing import ImageDegrader
from app.backend.models.arcface_model import ArcFaceModel

def train_sota(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # 1. Initialize Dataset & Dataloader
    degrader = ImageDegrader(scale=4, blur_kernel_size=5, noise_std=0.01)
    
    # args.data_dirs is a list of directories
    print(f"Loading datasets from: {args.data_dirs}")
    limit = 2000 if args.fast_mode else args.limit
    dataset = FaceSRDataset(args.data_dirs, degrader, hr_size=128, augment=True, max_images=limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. Initialize Models
    generator = SwinFuzzyLCR().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # Skip ArcFace in fast mode to avoid CPU bottleneck
    arcface = None
    if not args.fast_mode:
        print("Initializing ArcFace (Warning: Slows down training)...")
        arcface = ArcFaceModel(device=device)
    
    # 3. Loss Combiner & Optimizers
    loss_combiner = HybridLossCombiner(device=device)
    
    optimizer_G = optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs)

    # 4. Training Loop
    print("\nStarting SOTA Training (Swin-Fuzzy-LCR)...")
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Inputs
            lr_img = batch['lr'].to(device)
            hr_img = batch['hr'].to(device)
            
            # Dummy ANFIS Condition for training formulation [darkness, blur]
            # In a fully integrated pipeline, this comes from the ANFIS sub-network
            anfis_condition = torch.rand(lr_img.size(0), 2).to(device)
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            sr_img, _, _ = generator(lr_img, anfis_condition)
            
            # Adversarial Loss (Generator wants to fool discriminator)
            d_out_fake = discriminator(sr_img)
            valid = torch.ones_like(d_out_fake, requires_grad=False).to(device)
            g_adv_loss = torch.nn.functional.mse_loss(d_out_fake, valid)
            
            # Hybrid Loss
            total_g_loss, loss_dict = loss_combiner(sr_img, hr_img, arcface)
            total_g_loss = total_g_loss + (g_adv_loss * 0.1)
            
            total_g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # To match sizes, we evaluate discriminator again or use detached outputs
            d_out_real = discriminator(hr_img)
            # Since HR and LR have different shapes (128 vs 32), we need to ensure the target matches
            valid_hr = torch.ones_like(d_out_real, requires_grad=False).to(device)
            real_loss = torch.nn.functional.mse_loss(d_out_real, valid_hr)
            
            d_out_fake_detached = discriminator(sr_img.detach())
            fake = torch.zeros_like(d_out_fake_detached, requires_grad=False).to(device)
            fake_loss = torch.nn.functional.mse_loss(d_out_fake_detached, fake)
            
            d_loss = 0.5 * (real_loss + fake_loss)
            
            d_loss.backward()
            optimizer_D.step()
            
            # Logging
            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_Loss': f"{total_g_loss.item():.4f}", 
                'D_Loss': f"{d_loss.item():.4f}",
                'ID_Loss': f"{loss_dict.get('id', 0):.4f}"
            })
            
        scheduler_G.step()
        scheduler_D.step()
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = Path(args.ckpt_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # Epoch specific
            torch.save(generator.state_dict(), ckpt_path / f"swin_fuzzy_lcr_epoch_{epoch+1}.pth")
            # Latest for inference
            torch.save(generator.state_dict(), ckpt_path / "swin_fuzzy_lcr.pth")
            print(f"Checkpoint saved: {ckpt_path / 'swin_fuzzy_lcr.pth'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', default=['data/ffhq', 'data/img_align_celeba'], help='List of dataset directories')
    parser.add_argument('--ckpt_dir', type=str, default='app/backend/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fast_mode', action='store_true', help='Train faster by limiting data and skipping ID loss')
    parser.add_argument('--limit', type=int, default=None, help='Max images to use')
    args = parser.parse_args()
    
    train_sota(args)
