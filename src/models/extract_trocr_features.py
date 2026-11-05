"""Extract TrOCR Handwritten Math Visual Features on Local CUDA GPU.

Pillar 3 of 0.60+ Roadmap:
Extracts 384-dimensional handwriting-specialized visual representation vectors
from microsoft/trocr-small-handwritten using FP16 precision on cuda:0 (GTX 1050).
Processes:
- artifacts/train_images_224.pt (1500, 3, 224, 224) -> artifacts/train_trocr_features.npy (1500, 384)
- artifacts/test_images_224.pt (3498, 3, 224, 224) -> artifacts/test_trocr_features.npy (3498, 384)
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_features(
    image_tensor: torch.Tensor,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    target_size: int = 384
) -> np.ndarray:
    """Extract CLS token embeddings in batches from TrOCR vision encoder."""
    n_samples = len(image_tensor)
    features_list = []
    start_time = time.time()

    for i in range(0, n_samples, batch_size):
        batch = image_tensor[i : i + batch_size].to(device).half()
        with torch.no_grad():
            batch_384 = F.interpolate(batch, size=(target_size, target_size), mode="bilinear", align_corners=False)
            out = encoder(batch_384)
            # CLS token at index 0
            cls_token = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            features_list.append(cls_token)

        if (i // batch_size + 1) % 20 == 0 or (i + batch_size >= n_samples):
            elapsed = time.time() - start_time
            print(f"  Processed {min(i + batch_size, n_samples)}/{n_samples} samples ({elapsed:.1f}s elapsed)...")

    return np.vstack(features_list)


def run_trocr_feature_extraction():
    print("=" * 70)
    print("[TrOCR] Extracting Handwritten Math Features on GPU (GTX 1050 FP16)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    art_dir = PROJECT_ROOT / "artifacts"

    # 1. Load Model
    model_name = "microsoft/trocr-small-handwritten"
    print(f"\nLoading {model_name}...")
    model = cast(Any, VisionEncoderDecoderModel).from_pretrained(model_name)
    encoder = model.encoder.half().to(device)
    encoder.eval()
    print("TrOCR Vision Encoder loaded successfully in FP16.")

    # 2. Extract Train Images
    train_pt_path = art_dir / "train_images_224.pt"
    print(f"\nLoading train images: {train_pt_path.name}...")
    train_images = torch.load(train_pt_path, map_location="cpu")
    print(f"Train tensor shape: {train_images.shape}")

    print("Extracting train handwriting features...")
    train_trocr = extract_features(train_images, encoder, device, batch_size=16)
    print(f"Train features shape: {train_trocr.shape}")

    train_out = art_dir / "train_trocr_features.npy"
    np.save(train_out, train_trocr)
    print(f"Saved: {train_out.name} ({train_out.stat().st_size / (1024*1024):.2f} MB)")

    # Clean cache
    del train_images
    torch.cuda.empty_cache()

    # 3. Extract Test Images
    test_pt_path = art_dir / "test_images_224.pt"
    print(f"\nLoading test images: {test_pt_path.name}...")
    test_images = torch.load(test_pt_path, map_location="cpu")
    print(f"Test tensor shape: {test_images.shape}")

    print("Extracting test handwriting features...")
    test_trocr = extract_features(test_images, encoder, device, batch_size=16)
    print(f"Test features shape: {test_trocr.shape}")

    test_out = art_dir / "test_trocr_features.npy"
    np.save(test_out, test_trocr)
    print(f"Saved: {test_out.name} ({test_out.stat().st_size / (1024*1024):.2f} MB)")

    print("\n[TrOCR] Feature extraction complete!")


if __name__ == "__main__":
    run_trocr_feature_extraction()

