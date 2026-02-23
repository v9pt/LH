"""Identity loss using ArcFace embeddings for face recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """Identity preservation loss using ArcFace embeddings.
    
    Ensures that the super-resolved face maintains the same identity
    as the ground truth using cosine similarity in embedding space.
    """
    
    def __init__(self, arcface_model):
        """Initialize identity loss.
        
        Args:
            arcface_model: Pretrained ArcFace model for embedding extraction
        """
        super(IdentityLoss, self).__init__()
        self.arcface = arcface_model
        
        # Freeze ArcFace parameters
        for param in self.arcface.parameters():
            param.requires_grad = False
    
    def forward(self, sr, hr):
        """Compute identity loss.
        
        Args:
            sr: Super-resolved images [B, 3, H, W]
            hr: High-resolution ground truth [B, 3, H, W]
            
        Returns:
            Identity loss scalar (1 - cosine_similarity)
        """
        # Extract embeddings
        emb_sr = self.arcface(sr)
        emb_hr = self.arcface(hr)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb_sr, emb_hr, dim=1)
        
        # Loss is 1 - similarity (minimize distance)
        loss = 1 - similarity.mean()
        
        return loss


class IdentityLossWithDetection:
    """Identity loss with face detection.
    
    Uses InsightFace for face detection and embedding extraction.
    Handles cases where faces may not be detected.
    """
    
    def __init__(self, arcface_detector):
        """Initialize identity loss with detection.
        
        Args:
            arcface_detector: ArcFaceModel instance from models/arcface_model.py
        """
        self.detector = arcface_detector
    
    def compute_loss(self, sr_images, hr_images):
        """Compute identity loss with face detection.
        
        Args:
            sr_images: SR images [B, 3, H, W] torch tensors
            hr_images: HR images [B, 3, H, W] torch tensors
            
        Returns:
            Identity loss scalar or 0 if faces not detected
        """
        batch_size = sr_images.shape[0]
        losses = []
        
        for i in range(batch_size):
            # Extract embeddings
            emb_sr = self.detector.extract_embedding(sr_images[i])
            emb_hr = self.detector.extract_embedding(hr_images[i])
            
            if emb_sr is not None and emb_hr is not None:
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    emb_sr.unsqueeze(0),
                    emb_hr.unsqueeze(0),
                    dim=1
                )
                loss = 1 - similarity
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True).to(sr_images.device)
        
        return torch.stack(losses).mean()
