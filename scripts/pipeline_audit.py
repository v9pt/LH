import torch
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from app.backend.models.arcface_model import DifferentiableArcFace, ArcFaceModel
from app.backend.core.darkness_estimator import extract_illumination_features

def audit_identity():
    print("--- 1. Auditing ArcFace Identity Anchor ---")
    device = 'cpu'
    # Use the Differentiable version used during training
    model = DifferentiableArcFace(device=device)
    
    # Create a dummy face-like tensor [0, 1]
    # We use a non-zero pattern to ensure gradients and features are active
    dummy = torch.ones(1, 3, 112, 112) * 0.5
    dummy[0, 0, 40:60, 40:60] = 0.8 # Add a "feature"
    
    # GT to GT similarity check
    with torch.no_grad():
        emb1 = model(dummy)
        emb2 = model(dummy)
    
    sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    print(f"  GT-GT Cosine Similarity (Self): {sim:.4f}")
    
    # Check normalization
    print(f"  Embedding Norm: {torch.norm(emb1).item():.4f}")
    
    if sim < 0.99:
        print("  [ERROR] ArcFace is non-deterministic!")
    else:
        print("  [OK] Identity Anchor is mathematically stable.")

def audit_features():
    print("\n--- 2. Auditing ANFIS Feature Physics ---")
    # We check if features move as expected with Gamma
    print(f"{'Gamma':>6} | {'MeanLum':>8} | {'DCP':>8} | {'Entropy':>8} | {'Contrast':>8}")
    print("-" * 50)
    
    for gamma in [1.0, 2.0, 3.0, 4.0, 5.0]:
        # Create gray image
        img = np.ones((128, 128, 3), dtype=np.uint8) * 180
        # Apply gamma
        img_f = (img / 255.0) ** gamma
        img_u8 = (img_f * 255).astype(np.uint8)
        
        feats = extract_illumination_features(img_u8)
        print(f"{gamma:>6.1f} | {feats[0]:>8.3f} | {feats[2]:>8.3f} | {feats[3]:>8.3f} | {feats[7]:>8.3f}")

if __name__ == "__main__":
    try:
        audit_identity()
        audit_features()
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Audit crashed: {e}")
        import traceback
        traceback.print_exc()
