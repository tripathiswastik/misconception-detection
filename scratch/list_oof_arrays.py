from pathlib import Path
import numpy as np

ART_DIR = Path("d:/ag-ai/misconception-detection/artifacts")
oof_files = sorted(ART_DIR.glob("oof_*.npy"))
print(f"Total OOF files found: {len(oof_files)}")
for f in oof_files:
    arr = np.load(f)
    print(f"  {f.name:40s} | Shape: {str(arr.shape):16s} | dtype: {arr.dtype}")
