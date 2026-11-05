"""Extract Perceptual Stroke and Layout Features from Student Workings.

Extracts 24 scale-invariant stroke and geometric layout features:
1. Global ink density and bounding-box spatial coverage.
2. Aspect ratio and line step distribution.
3. 4x4 spatial grid distribution of student working.
4. Bottom 25% final-answer box ink concentration.
5. Horizontal stroke continuity (fraction lines, minus/equals signs).

Saves:
- artifacts/train_perceptual_strokes.npy (1500, 24)
- artifacts/test_perceptual_strokes.npy (3498, 24)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS


def extract_stroke_features_single(img_path: Path) -> np.ndarray:
    try:
        img = Image.open(img_path).convert("L").resize((512, 512), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)

        # Ink threshold (dark pencil/pen strokes on light paper)
        ink = (arr < 150).astype(np.float32)
        ink_sum = float(ink.sum())
        if ink_sum < 10:
            return np.zeros(24, dtype=np.float32)

        ink_density = ink_sum / (512.0 * 512.0)

        # Bounding box of handwritten ink
        rows = np.where(ink.any(axis=1))[0]
        cols = np.where(ink.any(axis=0))[0]
        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        box_h = max(1, y1 - y0)
        box_w = max(1, x1 - x0)
        box_coverage = (box_h * box_w) / (512.0 * 512.0)
        box_ink_density = ink_sum / (box_h * box_w)

        # 4x4 spatial quadrant distribution
        box_ink = ink[y0:y1+1, x0:x1+1]
        grid_feats = []
        for r_i in range(4):
            r_start = int(r_i * box_h / 4)
            r_end = int((r_i + 1) * box_h / 4)
            for c_i in range(4):
                c_start = int(c_i * box_w / 4)
                c_end = int((c_i + 1) * box_w / 4)
                cell = box_ink[r_start:r_end, c_start:c_end]
                grid_feats.append(float(cell.sum()) / max(1, cell.size))

        # Bottom 25% final answer region
        bottom_start = int(0.75 * box_h)
        bottom_ink = float(box_ink[bottom_start:, :].sum())
        bottom_ratio = bottom_ink / max(1.0, ink_sum)

        # Step / line projection profile
        h_profile = box_ink.sum(axis=1) / float(box_w)
        line_thresh = np.mean(h_profile) * 0.5
        active_lines = float(np.sum(h_profile > line_thresh)) / float(box_h)
        profile_std = float(np.std(h_profile))

        # Horizontal stroke continuity (fraction bars & equals)
        diff_h = np.abs(np.diff(box_ink, axis=1))
        h_edges = float(diff_h.sum()) / max(1.0, ink_sum)

        features = [
            ink_density,
            float(box_w) / float(box_h),
            box_coverage,
            box_ink_density,
            bottom_ratio,
            active_lines,
            profile_std,
            h_edges,
            *grid_feats  # 16 spatial grid bins
        ]
        return np.array(features, dtype=np.float32)
    except Exception:
        return np.zeros(24, dtype=np.float32)


def run_stroke_extraction():
    print("\n" + "=" * 75)
    print("EXTRACTING PERCEPTUAL STROKE & WORKING LAYOUT FEATURES")
    print("=" * 75 + "\n")

    train_df = pd.read_csv(PATHS.train_csv)
    test_df = pd.read_csv(PATHS.test_csv)
    artifacts_dir = PROJECT_ROOT / "artifacts"

    # Train Extraction
    train_cache = artifacts_dir / "train_perceptual_strokes.npy"
    if train_cache.exists():
        print(f"Loading existing train perceptual strokes from {train_cache}")
        train_strokes = np.load(train_cache)
    else:
        print(f"Extracting stroke features for {len(train_df)} train images...")
        train_strokes = np.zeros((len(train_df), 24), dtype=np.float32)
        for i, row in train_df.iterrows():
            img_p = PATHS.data_dir / "images" / "train" / str(row["image_file"])
            train_strokes[i] = extract_stroke_features_single(img_p)
            if (i + 1) % 300 == 0 or (i + 1) == len(train_df):
                print(f"  Processed {i+1}/{len(train_df)} train images")
        np.save(train_cache, train_strokes)
        print(f"Saved train strokes to {train_cache} (shape: {train_strokes.shape})")

    # Test Extraction
    test_cache = artifacts_dir / "test_perceptual_strokes.npy"
    if test_cache.exists():
        print(f"Loading existing test perceptual strokes from {test_cache}")
        test_strokes = np.load(test_cache)
    else:
        print(f"\nExtracting stroke features for {len(test_df)} test images...")
        test_strokes = np.zeros((len(test_df), 24), dtype=np.float32)
        for i, row in test_df.iterrows():
            img_p = PATHS.data_dir / "images" / "test" / str(row["image_file"])
            test_strokes[i] = extract_stroke_features_single(img_p)
            if (i + 1) % 500 == 0 or (i + 1) == len(test_df):
                print(f"  Processed {i+1}/{len(test_df)} test images")
        np.save(test_cache, test_strokes)
        print(f"Saved test strokes to {test_cache} (shape: {test_strokes.shape})")

    print("\n[Done] Perceptual stroke feature extraction complete.")
    return train_strokes, test_strokes


if __name__ == "__main__":
    run_stroke_extraction()
