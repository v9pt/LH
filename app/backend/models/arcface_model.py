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
        # FIX: Production-standard detection size for high-res (512x512) images
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        self.last_face_count = 0
    
    def extract_embedding_with_conf(self, image):
        """Extract embedding and detection confidence.
        
        Args:
            image: Numpy array [H, W, 3] in RGB, range [0, 255]
            
        Returns:
            (Embedding tensor [512], Confidence float) or (None, 0.0)
        """
        # Convert RGB to BGR for InsightFace
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # TASK 9: Multi-pass face detection sequence
        passes = []
        
        # Pass 1: CLAHE Enhanced
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        passes.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        
        # Pass 2: Raw BGR (sometimes enhancement confuses the model)
        passes.append(img_bgr)
        
        # Pass 3: Brightness Boost (alpha=1.5, beta=30)
        passes.append(cv2.convertScaleAbs(img_bgr, alpha=1.5, beta=30))
        
        # Pass 4: Gamma Boost (Power law)
        gamma = 0.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        passes.append(cv2.LUT(img_bgr, table))

        faces = []
        for i, det_img in enumerate(passes):
            # Add padding to handle faces at edges
            det_img_pad = cv2.copyMakeBorder(det_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0,0,0])
            faces = self.app.get(det_img_pad)
            if len(faces) > 0:
                if i > 0: print(f"  [DETECTION] Pass {i} SUCCESS")
                break
        
        self.last_face_count = len(faces)
        if len(faces) == 0:
            return None, 0.0

        # TASK 10: Largest-face selection (prevent identity confusion)
        if len(faces) > 1:
            # Sort by area (bbox: x1, y1, x2, y2)
            faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
            print(f"  [DETECTION] Multiple faces found ({len(faces)}). Selecting largest.")
        
        face = faces[0]
        embedding = torch.from_numpy(face.embedding).float()
        confidence = float(face.det_score)
        
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            return None, 0.0
            
        return embedding, confidence

    def extract_embedding(self, image):
        """Standard embedding extraction (backward compatibility)."""
        emb, _ = self.extract_embedding_with_conf(image)
        return emb

    def get_face_info(self, image):
        """Detect faces and return the largest face object (with landmarks)."""
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Multi-pass detection sequence (Task 9)
        passes = []
        passes.append(img_bgr) # Pass 1: Raw
        # Pass 2: CLAHE
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        passes.append(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))
        
        # Pass 3: Gamma
        gamma = 0.5
        table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        passes.append(cv2.LUT(img_bgr, table))

        for i, det_img in enumerate(passes):
            # Add padding for edge detection
            pad = 30
            det_img_pad = cv2.copyMakeBorder(det_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
            faces = self.app.get(det_img_pad)
            if len(faces) > 0:
                # Correct coordinates for padding
                for face in faces:
                    face.bbox -= [pad, pad, pad, pad]
                    face.kps -= [pad, pad]
                
                # Select largest face
                if len(faces) > 1:
                    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
                return faces[0]
        return None

    def extract_with_landmarks(self, image, landmarks):
        """Extract embedding using fixed landmarks for alignment. Task 3."""
        from insightface.utils import face_align
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Align using fixed landmarks
        aligned = face_align.norm_crop(img_bgr, landmarks)
        
        # 2. Extract embedding using the recognition model
        # We must provide a dummy face object to satisfy the wrapper
        from insightface.app.common import Face
        dummy_face = Face(kps=landmarks)
        
        net = self.app.models['recognition']
        # The wrapper .get() usually handles the forward pass
        # If it's the ArcFaceONNX wrapper, it takes (img, face)
        embedding = net.get(img_bgr, dummy_face)
        
        return torch.from_numpy(embedding).float()

    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None: return 0.0
        return float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())

    def align_face(self, image):
        """Return aligned face crop [112, 112] using largest detected face landmarks."""
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Add padding
        img_pad = cv2.copyMakeBorder(img_bgr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0,0,0])
        faces = self.app.get(img_pad)
        if len(faces) == 0: return None
        
        # Largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
        
        face = faces[0]
        # InsightFace handles alignment internally if we use face.norm_face
        # but here we just return the face object for external landmark transfer
        return face

    
    def extract_batch(self, images_t):
        """Extract embeddings for a batch of images (Tensors).
        
        Args:
            images_t: [B, 3, H, W] Tensor in [0, 1] RGB
            
        Returns:
            [B, 512] Tensor of embeddings
        """
        embs = []
        for i in range(images_t.shape[0]):
            # Convert to [0, 255] RGB numpy
            img_np = (images_t[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            emb = self.extract_embedding(img_np)
            if emb is None:
                embs.append(torch.zeros(512, device=images_t.device))
            else:
                embs.append(emb.to(images_t.device))
        
        return torch.stack(embs)

    def extract_embeddings_batch(self, images):
        """Legacy alias."""
        return self.extract_batch(images)
    
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
        # Input expected in range [0, 1]
        
        # TASK 7: EXACT PREPROCESSING MATCHING TRAINING
        # 1. Batch dimension correct
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # 2. Resize 112x112
        if x.shape[2] != 112 or x.shape[3] != 112:
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=True)
            
        # 3. BGR->RGB
        # Assuming input is BGR from OpenCV, convert to RGB by channel flipping
        x = x[:, [2, 1, 0], :, :]
        
        # 4. Normalize to [-1, 1]
        x = (x - 0.5) / 0.5
        
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
