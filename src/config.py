"""Centralized Configuration for Misconception Detection System.

Defines global parameters, directory paths, random seeds, hardware policies,
and model hyperparameters to prevent hardcoding across modules.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathConfig:
    """Canonical filesystem paths."""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    train_csv: Path = PROJECT_ROOT / "data" / "train.csv"
    test_csv: Path = PROJECT_ROOT / "data" / "test.csv"
    unlabelled_csv: Path = PROJECT_ROOT / "data" / "unlabelled.csv"
    sample_sub_csv: Path = PROJECT_ROOT / "data" / "sample_submission.csv"
    taxonomy_csv: Path = PROJECT_ROOT / "data" / "taxonomy_v1.csv"
    transcriptions_cache: Path = PROJECT_ROOT / "data" / "transcriptions_cache.json"
    splits_cache: Path = PROJECT_ROOT / "data" / "splits.json"
    submissions_dir: Path = PROJECT_ROOT / "submissions"
    docs_dir: Path = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and model architecture settings."""
    random_state: int = 42
    n_splits: int = 5
    default_alpha: float = 1e-4
    tuned_alpha: float = 5e-5
    max_features_word: int = 8000
    max_features_char: int = 12000
    domain_mask_damping: float = 0.85
    rare_class_threshold: int = 5
    top_k: int = 3
    prefer_gpu: bool = True
    fp16_amp: bool = True
    max_image_base64_bytes: int = 10 * 1024 * 1024  # 10 MB maximum image payload


PATHS = PathConfig()
CONFIG = ModelConfig()
