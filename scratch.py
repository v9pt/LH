import torch
from app.backend.models.swin_fuzzy_lcr import SwinFuzzyLCR
from app.backend.models.losses import HybridLossCombiner

model = SwinFuzzyLCR()
x = torch.randn(2, 3, 32, 32)
cond = torch.randn(2, 2)
out, latent, heatmap = model(x, cond)

loss_fn = HybridLossCombiner()
target = torch.randn(2, 3, 128, 128)
loss, loss_dict = loss_fn(out, target)

print("Forward pass successful!")
print(f"Output shape: {out.shape}")
print(f"Total loss: {loss.item()}")
