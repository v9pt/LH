#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import numpy as np
import cv2
import sys, os
import argparse
from pytorch_msssim import ms_ssim

# Project imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'app', 'backend'))

from app.backend.models.swin_fuzzy_lcr import SwinFuzzyLCR
from app.backend.models.arcface_model import ArcFaceModel
from app.backend.core.darkness_estimator import extract_illumination_features

# ─── Loss Functions ─────────────────────────────────────────────────────────

class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, x, y):
        kx, ky = self.kernel_x.to(x.device), self.kernel_y.to(x.device)
        x_g, y_g = x.mean(dim=1, keepdim=True), y.mean(dim=1, keepdim=True)
        gx_x = F.conv2d(x_g, kx, padding=1)
        gy_x = F.conv2d(x_g, ky, padding=1)
        gx_y = F.conv2d(y_g, kx, padding=1)
        gy_y = F.conv2d(y_g, ky, padding=1)
        return F.l1_loss(torch.abs(gx_x) + torch.abs(gy_x), torch.abs(gx_y) + torch.abs(gy_y))

class FFTLoss(nn.Module):
    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x, norm='ortho')
        y_fft = torch.fft.rfft2(y, norm='ortho')
        return F.l1_loss(torch.abs(x_fft), torch.abs(y_fft))

class VGGPercLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features[:17].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y, layer_idx=16):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.vgg(x), self.vgg(y))

def extract_identity_patches(img_t, landmarks):
    """Task 6: Crop eyes, nose, mouth patches from 512x512 image."""
    # Assuming standard landmarks (InsightFace 5 points)
    patches = []
    B = img_t.shape[0]
    for b in range(B):
        if landmarks[b] is None: continue
        for pt_idx in range(min(5, len(landmarks[b]))):
            cx, cy = landmarks[b][pt_idx].astype(int)
            # 64x64 patch
            x1 = max(0, cx - 32); x2 = min(512, cx + 32)
            y1 = max(0, cy - 32); y2 = min(512, cy + 32)
            patches.append(img_t[b:b+1, :, y1:y2, x1:x2])
    return torch.cat(patches, dim=0) if patches else img_t

# ─── Dataset & Training Logic ───────────────────────────────────────────────

class CelebADataset(Dataset):
    def __init__(self, data_dir, scale=8, subset_size=None):
        self.paths = sorted(list(Path(data_dir).glob('*.jpg')))
        if subset_size:
            self.paths = self.paths[:subset_size]
        self.scale = scale
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        hr = cv2.imread(str(self.paths[idx]))
        if hr is None: return self.__getitem__((idx + 1) % len(self))
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = cv2.resize(hr, (512, 512))
        
        lr_size = 512 // self.scale
        lr = cv2.resize(hr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        
        feats = extract_illumination_features(lr)
        
        hr_t = torch.from_numpy(hr.transpose(2,0,1)).float() / 255.0
        lr_t = torch.from_numpy(lr.transpose(2,0,1)).float() / 255.0
        feats_t = torch.from_numpy(feats).float()
        
        return lr_t, hr_t, feats_t

def train_convergence(epochs=200, batch_size=8, subset=5000, lr=1e-4, save_every=5, 
                      identity_weight=4.0, ssim_weight=1.0, fft_weight=0.5, 
                      sobel_weight=0.5, residual_scale=0.05, blend_alpha=0.85, 
                      disable_fallback=True, save_visuals=False):
    # Mac Optimization: Use MPS if available
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    ckpt_dir = Path('app/backend/checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Models
    model = SwinFuzzyLCR(feature_dim=64).to(device)
    arcface = ArcFaceModel(device='cpu') 
    
    # 2. Loss & Optimizer
    criterion_l1 = nn.L1Loss()
    sobel = SobelLoss().to(device)
    fft = FFTLoss().to(device)
    vgg = VGGPercLoss().to(device)
    
    # Task 5: Freeze Early Structure (Stem)
    # We freeze the initial feature extraction and SFT modulation
    # to prevent the model from relearning/destroying basic face geometry.
    for name, param in model.named_parameters():
        if 'feat_extract' in name or 'sft' in name:
            param.requires_grad = False
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Data
    data_path = 'data/img_align_celeba'
    dataset = CelebADataset(data_path, subset_size=subset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Starting IDENTITY-SAFE TRAINING | Device={device} | Subset={len(dataset)}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        for lr_img, hr_img, feats in loader:
            lr_img, hr_img, feats = lr_img.to(device), hr_img.to(device), feats.to(device)
            
            # Get landmarks for patch loss (Task 6)
            with torch.no_grad():
                hr_np = (hr_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                face = arcface.get_face_info(hr_np)
                landmarks = [face.kps] * hr_img.shape[0] if face else None
            
            optimizer.zero_grad()
            sr, _, _ = model(lr_img, feats, residual_scale=residual_scale)
            
            # Task 4: Residual-Only Learning Objective
            bicubic = F.interpolate(lr_img, size=(512, 512), mode='bilinear')
            
            # Task 9: Identity-Dominant Loss Stack
            loss_l1 = criterion_l1(sr, hr_img)
            loss_msssim = 1.0 - ms_ssim(sr, hr_img, data_range=1.0, size_average=True)
            loss_id = vgg(sr, hr_img)
            loss_fft = fft(sr, hr_img)
            loss_edge = sobel(sr, hr_img)
            loss_drift = criterion_l1(sr, bicubic)
            
            # Task 6: Identity Patch Loss
            if landmarks:
                sr_patches = extract_identity_patches(sr, landmarks)
                hr_patches = extract_identity_patches(hr_img, landmarks)
                loss_patch = criterion_l1(sr_patches, hr_patches)
            else:
                loss_patch = 0
                
            # Task 10: Progressive Curriculum (Stage-based Identity focus)
            # Epoch 1-10: Low ID weight to stabilize geometry
            # Epoch 11+: High ID weight to refine features
            current_id_weight = 1.0 if epoch <= 10 else identity_weight
            
            total_loss = (
                1.0 * loss_l1 + 
                ssim_weight * loss_msssim + 
                current_id_weight * loss_id + 
                1.0 * loss_patch +
                0.5 * loss_drift +
                fft_weight * loss_fft + 
                sobel_weight * loss_edge
            )
            
            # Task 7: Zero Residual Audit during training
            with torch.no_grad():
                res_mag = torch.abs(sr - bicubic).mean().item()
                if res_mag < 1e-7 and epoch > 1:
                    print("  ⚠ WARNING: Potential Residual Collapse detected (ResMag < 1e-7)")
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
        
        if epoch % save_every == 0:
            ckpt_path = ckpt_dir / f'swin_fuzzy_lcr_epoch_{epoch}.pth'
            # Also save as master for inference
            master_path = ckpt_dir / 'swin_fuzzy_lcr_convergence.pth'
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), master_path)
            
            # Task 9: Verify Checkpoint immediately
            try:
                test_model = SwinFuzzyLCR(feature_dim=64).to(device)
                test_model.load_state_dict(torch.load(ckpt_path, map_location=device))
                test_model.eval()
                with torch.no_grad():
                    # Dummy forward pass
                    _ = test_model(lr_img[:1], feats[:1])
                print(f"  ✓ Checkpoint Verified: {ckpt_path.name}")
            except Exception as e:
                print(f"  ⚠ CRITICAL: Checkpoint verification failed: {e}")

    # Task 10: Final Training Summary
    print("\n" + "="*60)
    print("CONVERGENCE TRAINING COMPLETE")
    print("="*60)
    print(f"Total Epochs: {epochs}")
    print(f"Final Loss:   {epoch_loss/len(loader):.4f}")
    print(f"Checkpoint:   {ckpt_dir / 'swin_fuzzy_lcr_convergence.pth'}")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--subset', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--identity_weight', type=float, default=4.0)
    parser.add_argument('--ssim_weight', type=float, default=1.0)
    parser.add_argument('--fft_weight', type=float, default=0.5)
    parser.add_argument('--sobel_weight', type=float, default=0.5)
    parser.add_argument('--residual_scale', type=float, default=0.05)
    parser.add_argument('--blend_alpha', type=float, default=0.85)
    parser.add_argument('--disable_fallback', action='store_true')
    parser.add_argument('--save_visuals', action='store_true')
    args = parser.parse_args()
    
    train_convergence(
        epochs=args.epochs, 
        subset=args.subset,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        identity_weight=args.identity_weight,
        ssim_weight=args.ssim_weight,
        fft_weight=args.fft_weight,
        sobel_weight=args.sobel_weight,
        residual_scale=args.residual_scale,
        blend_alpha=args.blend_alpha,
        disable_fallback=args.disable_fallback,
        save_visuals=args.save_visuals
    )
