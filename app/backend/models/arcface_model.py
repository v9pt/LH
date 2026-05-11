"""ArcFace model for identity embedding extraction.
Uses pretrained InsightFace models for face recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            image = image.detach().permute(1, 2, 0).cpu().numpy() * 255
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


class IResNetBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(IResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class DifferentiableArcFace(nn.Module):
    """
    Differentiable ArcFace (IResNet50) for Identity-Preserving Training.
    Allows the restoration model to receive gradients from the identity space.
    """
    def __init__(self, device='cpu'):
        super(DifferentiableArcFace, self).__init__()
        self.device = device
        # Simplified IResNet-50 structure for training efficiency
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.to(device)
        self.eval()

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(IResNetBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(IResNetBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input expected in range [0, 1] and size [112, 112]
        if x.shape[2] != 112 or x.shape[3] != 112:
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=True)
        
        x = (x - 0.5) / 0.5 # Normalise to [-1, 1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn3(x)
        return F.normalize(x, p=2, dim=1)
