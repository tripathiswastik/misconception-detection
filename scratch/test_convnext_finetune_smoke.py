import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import torchvision.models as models

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# 1. Load cached 224 images
images_pt = PROJECT_ROOT / "artifacts" / "train_images_224.pt"
print(f"Loading {images_pt.name}...", flush=True)
t0 = time.time()
X = torch.load(images_pt, map_location="cpu")
print(f"Loaded X shape: {X.shape} in {time.time() - t0:.2f}s", flush=True)

# 2. Build ConvNeXt-Tiny with stage 3 unfrozen
print("Building ConvNeXt-Tiny...", flush=True)
model = models.convnext_tiny(weights=None)
# Load cached weights directly from local checkpoint
weights_path = Path("C:/Users/hp/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth")
model.load_state_dict(torch.load(weights_path, map_location="cpu"))

# Freeze stem and stages 0, 1, 2 (features[0..5])
for i in range(6):
    for param in model.features[i].parameters():
        param.requires_grad = False

# Stage 3 (features[6..7]) remains unfrozen
for i in range(6, len(model.features)):
    for param in model.features[i].parameters():
        param.requires_grad = True

# Custom 76-class classifier
model.classifier[2] = nn.Linear(768, 76)
model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable params: {trainable_params:,} | Frozen params: {frozen_params:,}")

# Test 1 forward and backward pass with batch size 16
b_x = X[:16].to(device)
b_y = torch.randint(0, 76, (16,), device=device)

optimizer = torch.optim.AdamW([
    {"params": [p for p in model.features[6:].parameters() if p.requires_grad], "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 1e-3}
])
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=True)

with torch.cuda.amp.autocast(enabled=True):
    logits = model(b_x)
    loss = criterion(logits, b_y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

vram_used = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
print(f"Forward + backward successful! Loss: {loss.item():.4f}")
print(f"Peak VRAM used: {vram_used:.1f} MB (out of {torch.cuda.get_device_properties(0).total_memory / (1024*1024):.0f} MB)")
