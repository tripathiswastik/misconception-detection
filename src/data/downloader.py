"""Kaggle competition data downloader and directory resolver.

Handles downloading or locating files for:
'misconception-detection-in-k-12-student-math-workings'
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

COMPETITION_SLUG = "misconception-detection-in-k-12-student-math-workings"
EXPECTED_FILES = [
    "train.csv",
    "test.csv",
    "taxonomy_v1.csv",
    "sample_submission.csv"
]


def resolve_existing_dataset_dir(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Search for existing competition dataset directories."""
    candidate_paths = []
    
    if base_dir:
        candidate_paths.append(Path(base_dir))
    
    # Check local data folder
    candidate_paths.append(Path(__file__).resolve().parent.parent.parent / "data")
    
    # Check Kagglehub standard cache
    home = Path.home()
    candidate_paths.append(home / ".cache" / "kagglehub" / "competitions" / COMPETITION_SLUG)
    candidate_paths.append(home / ".kaggle" / "competitions" / COMPETITION_SLUG)
    
    for path in candidate_paths:
        if path.exists() and path.is_dir():
            has_files = all((path / f).exists() for f in EXPECTED_FILES)
            if has_files:
                return path
    
    return None


def verify_dataset_structure(data_dir: Path) -> Dict[str, Any]:
    """Verify and summarize the files in the dataset directory."""
    results = {
        "data_dir": str(data_dir),
        "files_present": {},
        "missing_files": [],
        "image_counts": {}
    }
    
    for f in EXPECTED_FILES + ["unlabelled.csv"]:
        target = data_dir / f
        if target.exists():
            results["files_present"][f] = target.stat().st_size
        else:
            if f != "unlabelled.csv":
                results["missing_files"].append(f)
    
    # Check images directories
    for img_subdir in ["train", "test", "questions", "unlabelled"]:
        img_path = data_dir / "images" / img_subdir
        if img_path.exists() and img_path.is_dir():
            try:
                count = len(list(img_path.glob("*.jpeg")) + list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")))
                results["image_counts"][img_subdir] = count
            except Exception:
                results["image_counts"][img_subdir] = -1
        else:
            results["image_counts"][img_subdir] = 0
            
    return results


def get_competition_data_dir(destination: Optional[str] = None) -> Path:
    """Resolve or download competition files, returning the directory Path."""
    existing = resolve_existing_dataset_dir(Path(destination) if destination else None)
    if existing:
        print(f"[DatasetResolver] Found existing dataset at: {existing}")
        return existing
        
    target_dir = Path(destination) if destination else Path(__file__).resolve().parent.parent.parent / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DatasetResolver] Attempting to download '{COMPETITION_SLUG}' into {target_dir}...")
    
    # Method 1: Try kagglehub
    try:
        import kagglehub
        if hasattr(kagglehub, "competition_download"):
            path = kagglehub.competition_download(COMPETITION_SLUG)
            print(f"[DatasetResolver] kagglehub download successful: {path}")
            return Path(path)
    except Exception as e:
        print(f"[DatasetResolver] kagglehub attempt note: {e}")

    # Method 2: Try kaggle CLI/API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[DatasetResolver] Authenticated with Kaggle API. Downloading files...")
        api.competition_download_files(COMPETITION_SLUG, path=str(target_dir), quiet=False)
        
        # Check if zip needs extraction
        zip_file = target_dir / f"{COMPETITION_SLUG}.zip"
        if zip_file.exists():
            import zipfile
            print(f"[DatasetResolver] Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            zip_file.unlink()
        return target_dir
    except Exception as e:
        print(f"[DatasetResolver] kaggle API attempt note: {e}")

    print("\n" + "=" * 70)
    print("ACTION REQUIRED: KAGGLE AUTHENTICATION NEEDED")
    print("To download the competition dataset, please either:")
    print("1. Set environment variables KAGGLE_USERNAME and KAGGLE_KEY, or")
    print("2. Place your 'kaggle.json' token into: C:\\Users\\hp\\.kaggle\\kaggle.json, or")
    print(f"3. Download the data manually and place extracted files into:\n   {target_dir}")
    print("=" * 70 + "\n")
    return target_dir


if __name__ == "__main__":
    resolved_path = get_competition_data_dir()
    print(f"Resolved Path: {resolved_path}")
    status = verify_dataset_structure(resolved_path)
    print(f"Dataset Verification:\n{status}")
