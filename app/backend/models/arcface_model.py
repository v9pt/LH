"""ArcFace model for identity embedding extraction.
Uses pretrained InsightFace models for face recognition.
"""

import torch
import torch.nn as nn
import numpy as np
from insightface.app import FaceAnalysis
import cv2


class ArcFaceModel:
    """ArcFace identity embedding extractor.
    
    Uses InsightFace's pretrained ArcFace model for 512-dim embeddings.
    """
    
    def __init__(self, device='cpu', model_name='buffalo_l'):
        """Initialize ArcFace model.
        
        Args:
            device: 'cpu' or 'cuda'
            model_name: InsightFace model name
        """
        self.device = device
        self.app = FaceAnalysis(name=model_name, providers=[
            'CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'
        ])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(128, 128))
    
    def extract_embedding(self, image):
        """Extract 512-dim ArcFace embedding.
        
        Args:
            image: Numpy array [H, W, 3] in RGB, range [0, 255]
                   or torch tensor [3, H, W] in range [0, 1]
            
        Returns:
            Embedding tensor [512] or None if no face detected
        """
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # Remove batch dimension
            image = image.permute(1, 2, 0).cpu().numpy() * 255
            image = image.astype(np.uint8)
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect face and extract embedding
        faces = self.app.get(image_bgr)
        
        if len(faces) == 0:
            return None
        
        # Return embedding of first detected face
        embedding = torch.from_numpy(faces[0].embedding).float()
        return embedding
    
    def extract_embeddings_batch(self, images):
        """Extract embeddings for batch of images.
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            Embeddings [B, 512] or None for images without faces
        """
        embeddings = []
        for i in range(images.shape[0]):
            emb = self.extract_embedding(images[i])
            embeddings.append(emb)
        
        return embeddings
    
    @staticmethod
    def cosine_similarity(emb1, emb2):
        """Compute cosine similarity between two embeddings.
        
        Args:
            emb1: Embedding tensor [512]
            emb2: Embedding tensor [512]
            
        Returns:
            Similarity score in [-1, 1]
        """
        return torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()


class SimplifiedArcFace(nn.Module):
    """Simplified ArcFace model for training pipeline.
    
    This is a lightweight version for computing identity loss during training.
    """
    
    def __init__(self, embedding_size=512):
        super(SimplifiedArcFace, self).__init__()
        self.embedding_size = embedding_size
        
        # Simple CNN for feature extraction (placeholder)
        # In practice, load pretrained ArcFace backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, embedding_size)
    
    def forward(self, x):
        """Extract embeddings.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Embeddings [B, 512]
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        emb = self.fc(feat)
        # L2 normalize
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


