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

class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, x, y):
        k = self.kernel.to(x.device)
        lx = F.conv2d(x.mean(dim=1, keepdim=True), k, padding=1)
        ly = F.conv2d(y.mean(dim=1, keepdim=True), k, padding=1)
        return F.l1_loss(lx, ly)

class VGGPercLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y, layer_idx=9):
        feat_x = self.vgg[:layer_idx](x)
        feat_y = self.vgg[:layer_idx](y)
        return F.mse_loss(feat_x, feat_y)

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

def train_convergence(epochs=200, batch_size=8, subset=5000):
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
    # ArcFace is used for metrics/logging (CPU usually best for InsightFace)
    arcface = ArcFaceModel(device='cpu') 
    
    # 2. Loss & Optimizer
    criterion_l1 = nn.L1Loss()
    sobel = SobelLoss().to(device)
    fft = FFTLoss().to(device)
    laplacian = LaplacianLoss().to(device)
    vgg = VGGPercLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Data
    data_path = 'data/img_align_celeba'
    if not os.path.exists(data_path):
        print(f"⚠ {data_path} not found.")
        return
        
    dataset = CelebADataset(data_path, subset_size=subset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting CONVERGENCE TRAINING | Device={device} | Subset={len(dataset)}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        for lr, hr, feats in loader:
            lr, hr, feats = lr.to(device), hr.to(device), feats.to(device)
            
            optimizer.zero_grad()
            sr, _, _ = model(lr, feats, residual_scale=0.15)
            
            loss_l1 = criterion_l1(sr, hr)
            loss_edge = sobel(sr, hr)
            loss_fft = fft(sr, hr)
            loss_lap = laplacian(sr, hr)
            loss_id = vgg(sr, hr, layer_idx=16) 
            
            total_loss = (1.0 * loss_l1 + 4.0 * loss_id + 0.5 * loss_fft + 0.3 * loss_edge + 0.2 * loss_lap)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / 'swin_fuzzy_lcr_convergence.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--subset', type=int, default=5000)
    args = parser.parse_args()
    
    train_convergence(epochs=args.epochs, subset=args.subset)
