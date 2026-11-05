"""Extract 768-d ConvNeXt-Tiny Visual Embeddings on cuda:0 in FP16.

Features are extracted using frozen ImageNet-1K pretrained weights.
Outputs:
- artifacts/train_convnext_features.npy: (1500, 768)
- artifacts/test_convnext_features.npy: (3498, 768)
"""
import sys
import time
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        p = self.paths[index]
        try:
            if p.exists():
                with Image.open(p) as img:
                    return self.transform(img.convert("RGB"))
        except Exception:
            pass
        return torch.zeros(3, 224, 224)


def extract_features():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[ConvNeXt] Initializing feature extractor on {device}...")

    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = torch.nn.Identity()  # 768-d representation
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()

    train_paths = [PATHS.data_dir / "images" / "train" / str(f) for f in train_df["image_file"].values]
    test_paths = [PATHS.data_dir / "images" / "test" / str(f) for f in test_df["image_file"].values]

    def extract_loader(paths: List[Path], desc: str) -> np.ndarray:
        t0 = time.time()
        print(f"[ConvNeXt] Extracting {len(paths)} {desc} images in batches...")
        dataset = ImagePathDataset(paths, preprocess)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        feats_list = []

        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feats = model(batch_x)
                feats_list.append(feats.cpu().float().numpy())

        res = np.concatenate(feats_list, axis=0)
        print(f"[ConvNeXt] {desc} extraction complete in {time.time() - t0:.2f}s. Shape: {res.shape}")
        return res

    train_feats = extract_loader(train_paths, "train")
    test_feats = extract_loader(test_paths, "test")

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "train_convnext_features.npy", train_feats)
    np.save(artifacts_dir / "test_convnext_features.npy", test_feats)
    print(f"[ConvNeXt] Successfully saved feature matrices to {artifacts_dir}")


if __name__ == "__main__":
    extract_features()
