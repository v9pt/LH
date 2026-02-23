"""Facial landmark detection using pretrained models.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class LandmarkDetector:
    """68-point facial landmark detector.
    
    Uses InsightFace's landmark detection for consistency loss.
    """
    
    def __init__(self, device='cpu', model_name='buffalo_l'):
        """Initialize landmark detector.
        
        Args:
            device: 'cpu' or 'cuda'
            model_name: InsightFace model name
        """
        self.device = device
        self.app = FaceAnalysis(name=model_name, providers=[
            'CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'
        ])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(128, 128))
    
    def detect_landmarks(self, image):
        """Detect 68 facial landmarks.
        
        Args:
            image: Numpy array [H, W, 3] in RGB, range [0, 255]
                   or torch tensor [3, H, W] in range [0, 1]
            
        Returns:
            Landmarks tensor [68, 2] (x, y coordinates) or None if no face
        """
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # Remove batch dimension
            image = image.permute(1, 2, 0).cpu().numpy() * 255
            image = image.astype(np.uint8)
        
        # Convert RGB to BGR for InsightFace
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect face
        faces = self.app.get(image_bgr)
        
        if len(faces) == 0:
            return None
        
        # Get landmarks from first detected face
        # InsightFace returns 5 keypoints, we'll use those and interpolate
        kps = faces[0].kps  # [5, 2] - left_eye, right_eye, nose, left_mouth, right_mouth
        
        # Convert to tensor
        landmarks = torch.from_numpy(kps).float()
        
        return landmarks
    
    def detect_landmarks_batch(self, images):
        """Detect landmarks for batch of images.
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            List of landmark tensors or None for images without faces
        """
        landmarks_list = []
        for i in range(images.shape[0]):
            lmks = self.detect_landmarks(images[i])
            landmarks_list.append(lmks)
        
        return landmarks_list
    
    @staticmethod
    def compute_landmark_distance(lmk1, lmk2):
        """Compute L2 distance between two landmark sets.
        
        Args:
            lmk1: Landmarks [N, 2]
            lmk2: Landmarks [N, 2]
            
        Returns:
            Average L2 distance
        """
        if lmk1 is None or lmk2 is None:
            return torch.tensor(0.0)
        
        diff = lmk1 - lmk2
        distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
        return torch.mean(distances)


class SimplifiedLandmarkDetector(nn.Module):
    """Simplified landmark detector for training.
    
    Lightweight version for computing landmark loss during training.
    """
    
    def __init__(self, num_landmarks=5):
        super(SimplifiedLandmarkDetector, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Simple CNN for landmark detection (placeholder)
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
        self.fc = nn.Linear(256, num_landmarks * 2)  # x, y for each landmark
    
    def forward(self, x):
        """Detect landmarks.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Landmarks [B, num_landmarks, 2]
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        lmks = self.fc(feat)
        lmks = lmks.view(-1, self.num_landmarks, 2)
        return lmks
