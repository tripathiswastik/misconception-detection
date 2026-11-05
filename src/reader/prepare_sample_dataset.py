"""Prepare 50-Sample Benchmark Dataset for Kaggle Upload.

Selects 50 stratified images (25 correct, 25 incorrect) from train.csv
and copies them into artifacts/dataset_50_sample/ with clean metadata.
"""
import json
import shutil
from pathlib import Path
import pandas as pd

from src.data.dataset import MathMisconceptionData

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "train"
OUT_DIR = PROJECT_ROOT / "artifacts" / "dataset_50_sample"

OUT_DIR.mkdir(parents=True, exist_ok=True)
images_out_dir = OUT_DIR / "images"
images_out_dir.mkdir(parents=True, exist_ok=True)

data_mgr = MathMisconceptionData(DATA_DIR)
train_df = data_mgr.load_train()

mask_corr = train_df["final_answer_correct"].astype(bool) == True
sample_corr = train_df[mask_corr].sample(25, random_state=42)
sample_incorr = train_df[~mask_corr].sample(25, random_state=42)
sample_df = pd.concat([sample_corr, sample_incorr]).sample(frac=1.0, random_state=42).reset_index(drop=True)

csv_out_path = OUT_DIR / "sample_50.csv"
sample_df.to_csv(csv_out_path, index=False)
print(f"Saved {len(sample_df)} sample records to {csv_out_path}")

copied = 0
for _, row in sample_df.iterrows():
    img_name = str(row["image_file"])
    src_img = IMAGES_DIR / img_name
    dst_img = images_out_dir / img_name
    if src_img.exists():
        shutil.copy2(src_img, dst_img)
        copied += 1
    else:
        print(f"Warning: Missing image {src_img}")

print(f"Copied {copied}/{len(sample_df)} sample images to {images_out_dir}")

# Write Kaggle dataset-metadata.json
metadata = {
    "title": "Misconception 50 Benchmark Sample",
    "id": "tripathiswastik/misconception-50-benchmark",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(OUT_DIR / "dataset-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Dataset prepared successfully in:", OUT_DIR)
