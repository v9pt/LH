import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class CharbonnierLoss(nn.Module):
    """
    L1 Charbonnier loss (a differentiable variant of L1 loss).
    Less prone to oversmoothing than MSE (L2) loss.
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss / (x.size(0) * x.size(1) * x.size(2) * x.size(3))

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL).
    Computes the distance between images in the Fourier domain.
    Massively improves high-frequency detail recovery (hair, pores).
    """
    def __init__(self, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Convert to grayscale for FFT
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        # 2D FFT
        pred_fft = torch.fft.fft2(pred_gray, norm='ortho')
        target_fft = torch.fft.fft2(target_gray, norm='ortho')
        
        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # L1 distance in frequency domain
        loss = F.l1_loss(pred_mag, target_mag)
        return loss * self.alpha

class EdgeLoss(nn.Module):
    """
    Penalizes differences in structural edges using Sobel filters.
    Crucial for hitting high SSIM scores.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)

    def forward(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        pred_gx = F.conv2d(pred_gray, self.k_x, padding=1)
        pred_gy = F.conv2d(pred_gray, self.k_y, padding=1)
        pred_edge = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        
        target_gx = F.conv2d(target_gray, self.k_x, padding=1)
        target_gy = F.conv2d(target_gray, self.k_y, padding=1)
        target_edge = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)

class HybridLossCombiner(nn.Module):
    """
    The Ultimate Multi-Task Loss formulation for the Swin-Fuzzy-LCR architecture.
    """
    def __init__(self, device='cpu'):
        super(HybridLossCombiner, self).__init__()
        self.device = device
        
        # Initialize individual losses
        self.l1_loss = CharbonnierLoss()
        
        # Perceptual Loss (LPIPS)
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device)
        self.perceptual_loss.eval()
        
        # Frequency Loss
        self.freq_loss = FocalFrequencyLoss()
        
        # NEW: Edge Loss for structural integrity (SSIM boost)
        self.edge_loss = EdgeLoss()
        
        # Weights (as defined in the implementation plan)
        self.w_l1 = 1.0
        self.w_percep = 1.0
        self.w_id = 0.8
        self.w_freq = 0.1
        self.w_edge = 0.5
        self.w_adv = 0.1 # Adversarial loss handles separately by a discriminator

    def forward(self, pred, target, arcface_model=None):
        """
        pred: SR image [B, 3, H, W], range [0, 1]
        target: HR image [B, 3, H, W], range [0, 1]
        arcface_model: instance of ArcFaceModel for Identity Loss
        """
        loss_dict = {}
        
        # 1. Pixel Loss (L1)
        l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = l1 * self.w_l1
        
        # 2. Perceptual Loss (VGG)
        # lpips expects inputs in [-1, 1]
        pred_norm = pred * 2 - 1
        target_norm = target * 2 - 1
        percep = self.perceptual_loss(pred_norm, target_norm).mean()
        loss_dict['percep'] = percep * self.w_percep
        
        # 3. Frequency Loss
        freq = self.freq_loss(pred, target)
        loss_dict['freq'] = freq * self.w_freq
        
        # NEW: 4. Edge Loss
        edge = self.edge_loss(pred, target)
        loss_dict['edge'] = edge * self.w_edge
        
        # 5. Identity Loss
        if arcface_model is not None:
            # pred and target are [B, 3, H, W]
            pred_embs = arcface_model.extract_embeddings_batch(pred)
            target_embs = arcface_model.extract_embeddings_batch(target)
            
            valid_sims = []
            for p_emb, t_emb in zip(pred_embs, target_embs):
                if p_emb is not None and t_emb is not None:
                    valid_sims.append(arcface_model.cosine_similarity(p_emb, t_emb))
            
            if valid_sims:
                avg_sim = sum(valid_sims) / len(valid_sims)
                # Convert to tensor and invert for loss
                id_loss = 1.0 - torch.tensor(avg_sim, device=self.device)
                loss_dict['id'] = id_loss * self.w_id
            else:
                loss_dict['id'] = torch.tensor(0.0, device=self.device)
        else:
            loss_dict['id'] = torch.tensor(0.0, device=self.device)
            
        # Total Generator Loss
        total_loss = sum(loss_dict.values())
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
