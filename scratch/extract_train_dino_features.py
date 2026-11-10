"""Extract 384-d Frozen DINO (ViT-S/16) Visual Embeddings for Train Images.

Uses facebook/dino-vits16 (self-supervised vision transformer) on CUDA:0.
Images read directly from competition_data.zip matching train.csv order.
Outputs: artifacts/train_dino_features_384.npy (1500, 384)
"""
import io
from pathlib import Path
import sys
import time
from typing import Any, List, cast
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
ZIP_PATH = Path("d:/ag-ai/competition_data.zip")

def main():
    print("=" * 80)
    print("EXTRACTING FROZEN DINO (ViT-S/16) VISUAL EMBEDDINGS")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    t0 = time.time()
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    n_samples = len(train_df)
    image_names = train_df["image_file"].tolist()
    print(f"Total train images to process: {n_samples}")

    # Load DINO Model & Processor
    model_name = "facebook/dino-vits16"
    processor = cast(Any, AutoImageProcessor).from_pretrained(model_name, local_files_only=True)
    model = cast(Any, AutoModel).from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()

    all_embs = []
    batch_size = 32

    print(f"Reading images from {ZIP_PATH.name} and extracting embeddings in batches of {batch_size}...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_names = image_names[start_idx:end_idx]

            batch_imgs: List[Image.Image] = []
            for img_name in batch_names:
                zip_member = f"images/train/{img_name}"
                try:
                    img_data = z.read(zip_member)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                except KeyError:
                    # Fallback to white canvas if image missing
                    img = Image.new("RGB", (224, 224), (255, 255, 255))
                batch_imgs.append(img)

            inputs = processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                # DINO ViT-S [CLS] token embedding (384-d)
                cls_embs = outputs.last_hidden_state[:, 0, :]
                all_embs.append(cls_embs.cpu().numpy())

            if (start_idx // batch_size) % 10 == 0:
                print(f"  Processed {end_idx} / {n_samples} images ({end_idx / n_samples:.1%})...")

    dino_features = np.vstack(all_embs).astype(np.float32)
    assert dino_features.shape == (n_samples, 384)

    out_path = ART_DIR / "train_dino_features_384.npy"
    np.save(out_path, dino_features)

    elapsed = time.time() - t0
    print(f"\n[Saved] Extracted DINO features to {out_path.name}")
    print(f"Shape: {dino_features.shape} | Dtype: {dino_features.dtype} | Time: {elapsed:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()

