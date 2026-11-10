from collections import Counter
import json
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
classes = tax.all_codes(include_none=True)
none_idx = classes.index("NONE")
n_classes = len(classes)

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")

print(f"Loaded components successfully. Sample count: {len(train_df)}")

