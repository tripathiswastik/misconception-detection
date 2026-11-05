"""Prepare Fresh Unseen 150-Sample Holdout Dataset for Final-Answer Verification.

Preregisters a fresh, holdout evaluation dataset:
- Strictly excludes the 150 traces used in the development benchmark.
- Samples 75 correct and 75 incorrect from the remaining 1,350 training traces.
- Uses strict parse_bool to prevent boolean string casting issues.
- Serializes to artifacts/dataset_150_fresh/sample_150_fresh.csv with SHA-256 tracking.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Set, cast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reader.verification_gate import parse_bool
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
DEV_150_CSV = PROJECT_ROOT / "artifacts" / "dataset_150_sample" / "sample_150.csv"
OUT_DIR = PROJECT_ROOT / "artifacts" / "dataset_150_fresh"
OUT_CSV = OUT_DIR / "sample_150_fresh.csv"
IMAGES_OUT_DIR = OUT_DIR / "images"

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_fresh_holdout_150() -> pd.DataFrame:
    """Generate fresh 150-sample holdout dataset without data leakage from dev set."""
    train_df = cast(pd.DataFrame, pd.read_csv(TRAIN_CSV))
    ans_col = cast(pd.Series, train_df["final_answer_correct"]).map(parse_bool)
    train_df["final_answer_correct_parsed"] = ans_col

    used_ids: Set[str] = set()
    if DEV_150_CSV.exists():
        dev_df = cast(pd.DataFrame, pd.read_csv(DEV_150_CSV))
        dev_trace_col = cast(pd.Series, dev_df["trace_id"])
        used_ids = set(dev_trace_col.astype(str))
        print(f"[Fresh150] Found {len(used_ids)} traces in dev benchmark set to exclude.")

    train_trace_col = cast(pd.Series, train_df["trace_id"])
    fresh_pool = cast(pd.DataFrame, train_df[~train_trace_col.astype(str).isin(used_ids)]).copy()
    assert isinstance(fresh_pool, pd.DataFrame)
    print(f"[Fresh150] Fresh candidate pool size: {len(fresh_pool)} traces.")

    parsed_col = cast(pd.Series, fresh_pool["final_answer_correct_parsed"])
    mask_corr = parsed_col == True
    n_corr_avail = int(mask_corr.sum())
    n_incorr_avail = int((~mask_corr).sum())

    if n_corr_avail < 75 or n_incorr_avail < 75:
        raise ValueError(f"Insufficient samples in fresh pool: {n_corr_avail} correct, {n_incorr_avail} incorrect")

    c_df = cast(pd.DataFrame, fresh_pool[mask_corr])
    inc_df = cast(pd.DataFrame, fresh_pool[~mask_corr])
    sample_corr = c_df.sample(75, random_state=101)
    sample_incorr = inc_df.sample(75, random_state=101)

    fresh_concat = pd.concat([sample_corr, sample_incorr])
    assert isinstance(fresh_concat, pd.DataFrame)
    fresh_150 = fresh_concat.sample(frac=1.0, random_state=101).reset_index(drop=True)
    assert isinstance(fresh_150, pd.DataFrame)

    fresh_trace_col = cast(pd.Series, fresh_150["trace_id"])
    fresh_parsed_col = cast(pd.Series, fresh_150["final_answer_correct_parsed"])

    # Sanity checks
    assert len(fresh_150) == 150, f"Expected 150 samples, got {len(fresh_150)}"
    assert int((fresh_parsed_col == True).sum()) == 75
    assert int((fresh_parsed_col == False).sum()) == 75

    overlap = set(fresh_trace_col.astype(str)).intersection(used_ids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} IDs overlap with dev set!"

    # Save CSV
    fresh_150.to_csv(OUT_CSV, index=False)
    csv_bytes = OUT_CSV.read_bytes()
    sha256_hash = hashlib.sha256(csv_bytes).hexdigest()

    manifest = {
        "dataset_name": "sample_150_fresh",
        "n_samples": 150,
        "n_correct": 75,
        "n_incorrect": 75,
        "random_seed": 101,
        "excluded_dev_ids_count": len(used_ids),
        "sha256": sha256_hash,
        "csv_path": str(OUT_CSV.relative_to(PROJECT_ROOT))
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Fresh150] Saved fresh 150 holdout set to {OUT_CSV}")
    print(f"[Fresh150] SHA-256: {sha256_hash}")

    # Copy images if local image directories exist
    img_dirs = [DATA_DIR / "images" / "train", DATA_DIR / "train_images"]
    found_dir = next((d for d in img_dirs if d.exists()), None)
    if found_dir:
        copied = 0
        for _, row in fresh_150.iterrows():
            img_name = str(row["image_file"])
            src_img = found_dir / img_name
            dst_img = IMAGES_OUT_DIR / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied += 1
        print(f"[Fresh150] Copied {copied}/150 sample images to {IMAGES_OUT_DIR}")

    return fresh_150


if __name__ == "__main__":
    prepare_fresh_holdout_150()
