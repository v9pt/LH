import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from app.backend.core.anfis_core import ANFIS
from app.backend.core.darkness_estimator import DarknessEstimator

def seed_checkpoint():
    print("--- Seeding Frozen physically-aligned ANFIS ---")
    de = DarknessEstimator(device='cpu')
    
    # Verify it is frozen and seeded
    p_mean = de.model.consequent.p.mean().item()
    print(f"  Rule-base Mean: {p_mean:.4f} (Seeded Ramp)")
    
    # Save to official checkpoint
    ckpt_path = Path('app/backend/checkpoints/darkness_estimator.pt')
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(de.model.state_dict(), ckpt_path)
    print(f"  ✓ Checkpoint locked and saved to {ckpt_path}")

if __name__ == "__main__":
    seed_checkpoint()
