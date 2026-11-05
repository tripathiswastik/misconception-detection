"""Extract 768-d Swin Transformer (swin_t) Visual Embeddings on cuda:0 in FP16.

Features are extracted using ImageNet-1K pretrained weights with shifted-window self-attention.
Reads cached preprocessed image tensors:
- artifacts/train_images_224.pt: (1500, 3, 224, 224)
- artifacts/test_images_224.pt: (3498, 3, 224, 224)

Outputs:
- artifacts/train_swin_features.npy: (1500, 768)
- artifacts/test_swin_features.npy: (3498, 768)
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import clear_gpu_memory, get_device


def extract_swin_features(batch_size: int = 32):
    device = get_device(prefer_gpu=True)
    print(f"\n[Swin-T] Initializing Swin Transformer feature extractor on {device}...")

    # Load Swin-T with ImageNet-1K weights
    weights = models.Swin_T_Weights.DEFAULT
    model = models.swin_t(weights=weights)
    setattr(model, "head", torch.nn.Identity())  # 768-d representation
    model = model.to(device).eval()

    artifacts_dir = PROJECT_ROOT / "artifacts"
    train_tensors_path = artifacts_dir / "train_images_224.pt"
    test_tensors_path = artifacts_dir / "test_images_224.pt"

    if not train_tensors_path.exists() or not test_tensors_path.exists():
        raise FileNotFoundError(f"Missing cached image tensors at {train_tensors_path} or {test_tensors_path}")

    def run_extraction(tensor_path: Path, desc: str) -> np.ndarray:
        t0 = time.time()
        print(f"[Swin-T] Loading {desc} cached image tensors from {tensor_path.name}...")
        img_tensors = torch.load(tensor_path, map_location="cpu")
        print(f"[Swin-T] Loaded {desc} tensor shape: {img_tensors.shape}")

        dataset = TensorDataset(img_tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        feats_list = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                if device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        feats = model(batch_x)
                else:
                    feats = model(batch_x)
                feats_list.append(feats.cpu().float().numpy())

        res = np.concatenate(feats_list, axis=0)
        elapsed = time.time() - t0
        print(f"[Swin-T] {desc} extraction complete in {elapsed:.2f}s. Extracted shape: {res.shape}")
        del img_tensors
        clear_gpu_memory()
        return res

    train_swin_feats = run_extraction(train_tensors_path, "train")
    test_swin_feats = run_extraction(test_tensors_path, "test")

    np.save(artifacts_dir / "train_swin_features.npy", train_swin_feats)
    np.save(artifacts_dir / "test_swin_features.npy", test_swin_feats)
    print(f"\n[Swin-T] Successfully persisted feature matrices to:")
    print(f"  - {artifacts_dir / 'train_swin_features.npy'}")
    print(f"  - {artifacts_dir / 'test_swin_features.npy'}")


if __name__ == "__main__":
    extract_swin_features()
