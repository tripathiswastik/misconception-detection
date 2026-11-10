"""Handwriting-Domain Contrastive Self-Supervised Pretraining Pilot.

Pretrains visual encoder on 2,000 unlabeled student handwriting images from competition_data.zip.
Uses SimCLR / InfoNCE contrastive learning with handwriting-specific data augmentations:
- RandomResizedCrop (scale 0.75-1.0)
- RandomRotation (+/- 7 degrees, camera tilt)
- ColorJitter (brightness/contrast, pencil hardness)
- Grayscale conversion
Extracts 384-d frozen domain-adapted visual embeddings on all 1,500 train images:
- Output: artifacts/train_handwriting_adapted_dino_384.npy (1500, 384)
"""
import io
from pathlib import Path
import random
import sys
import time
from typing import Any, List, Tuple, cast
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
ZIP_PATH = Path("d:/ag-ai/competition_data.zip")

class ContrastiveHandwritingDataset(Dataset):
    """Yields two independently augmented views of each unlabeled handwriting image."""
    def __init__(self, image_bytes_list: List[bytes], transform: transforms.Compose):
        self.image_bytes = image_bytes_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_bytes)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_bytes = self.image_bytes[index]
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2

class DINOContrastiveModel(nn.Module):
    def __init__(self, encoder: nn.Module, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = cast(Any, self.encoder)(pixel_values=x)
        cls_rep = out.last_hidden_state[:, 0, :]  # 384-d
        z = self.proj(cls_rep)                    # 128-d projection
        z_norm = F.normalize(z, dim=-1)
        return cls_rep, z_norm

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """Standard NT-Xent / InfoNCE contrastive loss across two augmented views."""
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # 2N x D
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / temperature

    # Create mask for positive pairs: (i, i + batch_size) and (i + batch_size, i)
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size, device=z1.device),
                        torch.arange(0, batch_size, device=z1.device)], dim=0)

    # Mask out self-contrast
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def main():
    print("=" * 80)
    print("PILOT: HANDWRITING-DOMAIN CONTRASTIVE PRETRAINING (SimCLR / InfoNCE)")
    print("Pretraining on 2,000 Unlabeled Handwriting Images from Competition Pool")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    start_time = time.time()

    # 1. Sample 2,000 Unlabeled Handwriting Images from ZIP
    N_UNLABELED = 2000
    print(f"\nReading {N_UNLABELED} random unlabeled images from {ZIP_PATH.name}...")
    unlabeled_bytes: List[bytes] = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        unlabeled_names = [n for n in z.namelist() if n.startswith("images/unlabelled/") and n.endswith(".jpeg")]
        random.seed(42)
        sample_names = random.sample(unlabeled_names, N_UNLABELED)

        for name in sample_names:
            unlabeled_bytes.append(z.read(name))

    print(f"Loaded {len(unlabeled_bytes)} unlabeled handwriting images into memory.")

    # 2. Handwriting-Specific Data Augmentation Pipeline
    hw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomRotation(degrees=7, fill=255),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ContrastiveHandwritingDataset(unlabeled_bytes, hw_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # 3. Model Setup: DINO ViT-S/16 with Top Blocks Unfrozen for Domain Adaptation
    model_name = "facebook/dino-vits16"
    print(f"\nInitializing {model_name} backbone...")
    encoder = cast(Any, AutoModel).from_pretrained(model_name, local_files_only=True)

    # Freeze earlier layers (0-9), unfreeze blocks 10 and 11
    for param in encoder.parameters():
        param.requires_grad = False

    # Unfreeze top transformer blocks
    for name, param in encoder.encoder.layer[-2:].named_parameters():
        param.requires_grad = True

    model = DINOContrastiveModel(encoder=encoder, proj_dim=128).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Blocks 10-11 + Projection Head): {trainable_params:,}")

    optimizer = torch.optim.AdamW([
        {"params": encoder.encoder.layer[-2:].parameters(), "lr": 1e-5, "weight_decay": 0.01},
        {"params": model.proj.parameters(), "lr": 1e-3, "weight_decay": 0.01}
    ])

    # 4. Contrastive Training Loop (5 Epochs)
    print("\nStarting InfoNCE contrastive pretraining on handwriting images...")
    model.train()
    EPOCHS = 5

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        for v1, v2 in loader:
            optimizer.zero_grad()
            _, z1 = model(v1.to(device))
            _, z2 = model(v2.to(device))
            loss = info_nce_loss(z1, z2, temperature=0.10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/{EPOCHS} | InfoNCE Loss: {avg_loss:.4f}")

    # 5. Extract Domain-Adapted Frozen Embeddings on all 1,500 Train Images
    print("\nExtracting domain-adapted embeddings for 1,500 train images...")
    model.eval()

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    n_train = len(train_df)
    train_image_names = train_df["image_file"].tolist()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_embs = []
    batch_size = 32

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_names = train_image_names[start_idx:end_idx]

            batch_tensors: List[torch.Tensor] = []
            for img_name in batch_names:
                zip_member = f"images/train/{img_name}"
                try:
                    raw_b = z.read(zip_member)
                    img = Image.open(io.BytesIO(raw_b)).convert("RGB")
                except KeyError:
                    img = Image.new("RGB", (224, 224), (255, 255, 255))
                batch_tensors.append(val_transform(img))

            batch_x = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                cls_embs, _ = model(batch_x)
                train_embs.append(cls_embs.cpu().numpy())

    train_adapted_features = np.vstack(train_embs).astype(np.float32)
    assert train_adapted_features.shape == (n_train, 384)

    out_path = ART_DIR / "train_handwriting_adapted_dino_384.npy"
    np.save(out_path, train_adapted_features)

    elapsed = time.time() - start_time
    print(f"\n[Saved] Domain-adapted features to {out_path.name}")
    print(f"Shape: {train_adapted_features.shape} | Dtype: {train_adapted_features.dtype} | Total Time: {elapsed:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()

