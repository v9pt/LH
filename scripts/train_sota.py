import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import copy

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.backend.models.swin_fuzzy_lcr import SwinFuzzyLCR
from app.backend.models.losses import HybridLossCombiner
from app.backend.models.discriminator import PatchGANDiscriminator
from app.backend.data.dataset import FaceSRDataset
from app.backend.data.preprocessing import ImageDegrader
from app.backend.models.arcface_model import DifferentiableArcFace


class EMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)


def build_warmup_cosine(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
    if len(dataset) == 0:
        raise RuntimeError(f"No training images found in: {args.data_dirs}")
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 4 if device.type == 'cuda' else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # 2. Initialize Models
    generator = SwinFuzzyLCR().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    ema = EMA(generator, decay=args.ema_decay)
    
    # Skip ArcFace in fast mode to avoid CPU bottleneck
    arcface = None
    if not args.fast_mode:
        print("Initializing differentiable ArcFace identity loss...")
        arcface = DifferentiableArcFace(device=device)
        for p in arcface.parameters():
            p.requires_grad_(False)
    
    # 3. Loss Combiner & Optimizers
    loss_combiner = HybridLossCombiner(device=device)
    
    optimizer_G = optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=1e-4)
    
    total_steps = max(1, len(dataloader) * args.epochs)
    scheduler_G = build_warmup_cosine(optimizer_G, args.warmup_steps, total_steps)
    scheduler_D = build_warmup_cosine(optimizer_D, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

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
            
            anfis_condition = batch['condition'].to(device)
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                sr_img, _, _ = generator(lr_img, anfis_condition)
                d_out_fake = discriminator(sr_img)
                valid = torch.ones_like(d_out_fake, requires_grad=False).to(device)
                g_adv_loss = F.mse_loss(d_out_fake, valid)
                total_g_loss, loss_dict = loss_combiner(sr_img, hr_img, arcface)
                total_g_loss = total_g_loss + (g_adv_loss * args.adv_weight)

            scaler.scale(total_g_loss).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            scaler.step(optimizer_G)
            scaler.update()
            ema.update(generator)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                d_out_real = discriminator(hr_img)
                valid_hr = torch.ones_like(d_out_real, requires_grad=False).to(device)
                real_loss = F.mse_loss(d_out_real, valid_hr)
                d_out_fake_detached = discriminator(sr_img.detach())
                fake = torch.zeros_like(d_out_fake_detached, requires_grad=False).to(device)
                fake_loss = F.mse_loss(d_out_fake_detached, fake)
                d_loss = 0.5 * (real_loss + fake_loss)

            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
            scaler.step(optimizer_D)
            scaler.update()
            scheduler_G.step()
            scheduler_D.step()
            
            # Logging
            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_Loss': f"{total_g_loss.item():.4f}", 
                'D_Loss': f"{d_loss.item():.4f}",
                'ID_Loss': f"{loss_dict.get('id', 0):.4f}"
            })
            
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = Path(args.ckpt_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # Epoch specific
            torch.save(ema.ema.state_dict(), ckpt_path / f"swin_fuzzy_lcr_epoch_{epoch+1}.pth")
            # Latest for inference
            torch.save(ema.ema.state_dict(), ckpt_path / "swin_fuzzy_lcr.pth")
            print(f"Checkpoint saved: {ckpt_path / 'swin_fuzzy_lcr.pth'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', default=['data/ffhq', 'data/img_align_celeba'], help='List of dataset directories')
    parser.add_argument('--ckpt_dir', type=str, default='app/backend/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--adv_weight', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--fast_mode', action='store_true', help='Train faster by limiting data and skipping ID loss')
    parser.add_argument('--limit', type=int, default=None, help='Max images to use')
    args = parser.parse_args()
    
    train_sota(args)
