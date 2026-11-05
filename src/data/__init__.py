"""Data ingestion, dataset loading, and taxonomy parsing."""
from .downloader import get_competition_data_dir, verify_dataset_structure

__all__ = ["get_competition_data_dir", "verify_dataset_structure"]
