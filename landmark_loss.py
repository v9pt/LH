"""Landmark consistency loss for geometric alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkLoss(nn.Module):
    """Landmark consistency loss using facial landmarks.
    
    Ensures geometric alignment between SR and HR faces.
    """
    
    def __init__(self, landmark_detector):
        """Initialize landmark loss.
        
        Args:
            landmark_detector: Pretrained landmark detector
        """
        super(LandmarkLoss, self).__init__()
        self.detector = landmark_detector
        
        # Freeze detector parameters
        if hasattr(landmark_detector, 'parameters'):
            for param in self.detector.parameters():
                param.requires_grad = False
    
    def forward(self, sr, hr):
        """Compute landmark loss.
        
        Args:
            sr: Super-resolved images [B, 3, H, W]
            hr: High-resolution ground truth [B, 3, H, W]
            
        Returns:
            Landmark L2 loss scalar
        """
        # Detect landmarks
        if hasattr(self.detector, 'forward'):
            # Neural network detector
            lmks_sr = self.detector(sr)
            lmks_hr = self.detector(hr)
            
            # L2 loss
            loss = torch.mean((lmks_sr - lmks_hr) ** 2)
        else:
            # External detector (with batch detection)
            loss = self._compute_with_detection(sr, hr)
        
        return loss
    
    def _compute_with_detection(self, sr_images, hr_images):
        """Compute loss with external face detection.
        
        Args:
            sr_images: SR images [B, 3, H, W]
            hr_images: HR images [B, 3, H, W]
            
        Returns:
            Landmark loss or 0 if landmarks not detected
        """
        batch_size = sr_images.shape[0]
        losses = []
        
        for i in range(batch_size):
            lmks_sr = self.detector.detect_landmarks(sr_images[i])
            lmks_hr = self.detector.detect_landmarks(hr_images[i])
            
            if lmks_sr is not None and lmks_hr is not None:
                # Normalize landmarks by image size
                h, w = sr_images.shape[2:]
                lmks_sr_norm = lmks_sr / torch.tensor([w, h]).to(lmks_sr.device)
                lmks_hr_norm = lmks_hr / torch.tensor([w, h]).to(lmks_hr.device)
                
                # L2 distance
                diff = lmks_sr_norm - lmks_hr_norm
                loss = torch.mean(diff ** 2)
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True).to(sr_images.device)
        
        return torch.stack(losses).mean()


class LandmarkLossWithDetection:
    """Landmark loss with external face detection."""
    
    def __init__(self, landmark_detector):
        """Initialize landmark loss.
        
        Args:
            landmark_detector: LandmarkDetector instance
        """
        self.detector = landmark_detector
    
    def compute_loss(self, sr_images, hr_images):
        """Compute landmark consistency loss.
        
        Args:
            sr_images: SR images [B, 3, H, W] torch tensors
            hr_images: HR images [B, 3, H, W] torch tensors
            
        Returns:
            Landmark loss scalar
        """
        batch_size = sr_images.shape[0]
        losses = []
        
        for i in range(batch_size):
            lmks_sr = self.detector.detect_landmarks(sr_images[i])
            lmks_hr = self.detector.detect_landmarks(hr_images[i])
            
            if lmks_sr is not None and lmks_hr is not None:
                # Normalize by image size
                h, w = sr_images.shape[2:]
                lmks_sr_norm = lmks_sr / torch.tensor([w, h]).float().to(lmks_sr.device)
                lmks_hr_norm = lmks_hr / torch.tensor([w, h]).float().to(lmks_hr.device)
                
                # L2 loss
                diff = lmks_sr_norm - lmks_hr_norm
                loss = torch.mean(diff ** 2)
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0).to(sr_images.device)
        
        return torch.stack(losses).mean()
