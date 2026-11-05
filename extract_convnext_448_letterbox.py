from pathlib import Path
import time
from typing import cast
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
ART_DIR = Path("d:/ag-ai/misconception-detection/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def letterbox_image(img: Image.Image, target_size: int = 448, fill_color=(255, 255, 255)) -> Image.Image:
    """Preserves handwriting aspect ratio with uniform square padding (white background)."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    padded.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded

def extract_features_for_paths(img_paths, conv_model, batch_size=16):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    feats = []
    t0 = time.time()
    n = len(img_paths)
    print(f"Extracting 448x448 letterbox features for {n} images (batch_size={batch_size})...")
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_paths = img_paths[i:i + batch_size]
            b_tensors = []
            for p in b_paths:
                try:
                    with Image.open(p) as raw:
                        padded = letterbox_image(raw.convert("RGB"), target_size=448)
                        arr = np.array(padded, dtype=np.float32) / 255.0
                        arr = (arr - mean) / std
                        t = torch.from_numpy(arr).permute(2, 0, 1).float()
                        b_tensors.append(t)
                except Exception as e:
                    b_tensors.append(torch.zeros(3, 448, 448, dtype=torch.float32))
            
            batch = torch.stack(b_tensors).to(device)
            out = conv_model(batch)
            feats.append(out.cpu().numpy())
            
            if (i // batch_size + 1) % 25 == 0 or (i + batch_size) >= n:
                elapsed = time.time() - t0
                fps = (min(i + batch_size, n)) / elapsed
                print(f"  Processed {min(i + batch_size, n)}/{n} images in {elapsed:.1f}s ({fps:.1f} img/s)...")
                
    result = np.vstack(feats).astype(np.float32)
    print(f"Finished extraction! Output matrix shape: {result.shape} in {time.time() - t0:.1f}s")
    return result

def main():
    # 1. Load ConvNeXt-Tiny
    print("Loading ConvNeXt-Tiny backbone...")
    conv_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    conv_model.classifier[2] = nn.Identity()
    conv_model = conv_model.to(device).eval()

    # 2. Extract Train (1,500)
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    train_paths = [DATA_DIR / "images" / "train" / str(f) for f in train_df["image_file"].values]
    train_feats = extract_features_for_paths(train_paths, conv_model, batch_size=16)
    np.save(ART_DIR / "train_convnext_448_letterbox.npy", train_feats)
    print("Saved train_convnext_448_letterbox.npy!")

    # 3. Extract Test (3,498)
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    test_paths = [DATA_DIR / "images" / "test" / str(f) for f in test_df["image_file"].values]
    test_feats = extract_features_for_paths(test_paths, conv_model, batch_size=16)
    np.save(ART_DIR / "test_convnext_448_letterbox.npy", test_feats)
    print("Saved test_convnext_448_letterbox.npy!")

    # 4. Extract Pseudo (1,465)
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_paths = [DATA_DIR / "images" / "unlabelled" / str(f) for f in pseudo_df["image_file"].values]
    pseudo_feats = extract_features_for_paths(pseudo_paths, conv_model, batch_size=16)
    np.save(ART_DIR / "pseudo_convnext_448_letterbox.npy", pseudo_feats)
    print("Saved pseudo_convnext_448_letterbox.npy!")

    print("\nAll 448x448 letterbox features extracted successfully!")

if __name__ == "__main__":
    main()

