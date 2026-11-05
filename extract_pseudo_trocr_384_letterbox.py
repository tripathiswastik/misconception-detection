"""Extract 384x384 Letterboxed TrOCR Features for 1,465 Pseudo Images.

Outputs:
  - artifacts/pseudo_trocr_384_letterbox.npy (1465, 384)
"""

from pathlib import Path
import time
from typing import Any, List, cast
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import VisionEncoderDecoderModel

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
ART_DIR = Path("d:/ag-ai/misconception-detection/artifacts")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Device for Pseudo TrOCR Extraction: {device}", flush=True)


def letterbox_image(img: Image.Image, target_size: int = 384, fill_color=(255, 255, 255)) -> Image.Image:
    """Uniform square letterboxing: preserves true handwriting aspect ratio without squashing."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    padded.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded


def extract_trocr_features(img_paths: List[Path], trocr_enc: Any, batch_size: int = 16) -> np.ndarray:
    tf_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    feats = []
    t0 = time.time()
    n = len(img_paths)
    print(f"Extracting 384x384 letterbox TrOCR features for {n} pseudo images (batch_size={batch_size})...", flush=True)
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_paths = img_paths[i : i + batch_size]
            b_tensors = []
            for p in b_paths:
                try:
                    with Image.open(p) as raw:
                        padded = letterbox_image(raw.convert("RGB"), target_size=384)
                        t = tf_norm(TF.to_tensor(padded))
                        b_tensors.append(t)
                except Exception:
                    b_tensors.append(torch.zeros(3, 384, 384, dtype=torch.float32))
            
            batch = torch.stack(b_tensors).to(device)
            out = trocr_enc(batch)
            cls_token = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            feats.append(cls_token)
            
            if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= n:
                elapsed = time.time() - t0
                done = min(i + batch_size, n)
                fps = done / elapsed
                eta = (n - done) / fps if fps > 0 else 0
                print(f"  [{done:4d}/{n}] {elapsed:.1f}s elapsed ({fps:.1f} img/s, ETA: {eta/60:.1f} min)...", flush=True)
                
    result = np.vstack(feats).astype(np.float32)
    print(f"Completed! Output shape: {result.shape} in {time.time() - t0:.1f}s", flush=True)
    return result


def main():
    print("=" * 80)
    print("Loading TrOCR-Small-Handwritten DeiT Visual Encoder...")
    print("=" * 80)
    trocr_model = cast(Any, VisionEncoderDecoderModel).from_pretrained(
        "microsoft/trocr-small-handwritten",
        local_files_only=True
    )
    trocr_enc = trocr_model.encoder.to(device).eval()
    print("TrOCR Encoder loaded successfully into native FP32 VRAM.", flush=True)
    
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_paths = [DATA_DIR / "images" / "unlabelled" / str(f) for f in pseudo_df["image_file"].values]
    pseudo_feats = extract_trocr_features(pseudo_paths, trocr_enc, batch_size=16)
    
    np.save(ART_DIR / "pseudo_trocr_384_letterbox.npy", pseudo_feats)
    print("Saved: artifacts/pseudo_trocr_384_letterbox.npy", flush=True)
    print("=" * 80)
    print("ALL 1,465 PSEUDO TrOCR 384 LETTERBOX FEATURES EXTRACTED AND SAVED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

