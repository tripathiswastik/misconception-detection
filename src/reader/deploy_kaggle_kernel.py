"""Deploy and execute 50-Image Final Answer Verification on Kaggle GPU.

Pushes src/reader/kaggle_gemini_verification.py to user's Kaggle account:
- Connected to: 'misconception-detection-in-k-12-student-math-workings'
- GPU: True (NVIDIA T4 / P100 16GB)
- Internet: True
- Outputs: /kaggle/working/vlm_50_sample_benchmark.json
"""
import json
import os
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KERNEL_FILE = PROJECT_ROOT / "src" / "reader" / "kaggle_gemini_verification.py"
TOKEN_PATH = Path.home() / ".kaggle" / "bearer_token"

if not TOKEN_PATH.exists():
    raise FileNotFoundError("Missing ~/.kaggle/bearer_token")

TOKEN = TOKEN_PATH.read_text(encoding="utf-8").strip()


def push_kernel_to_kaggle():
    print("[Kaggle Deploy] Reading kernel script from disk...")
    code_text = KERNEL_FILE.read_text(encoding="utf-8")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "slug": "tripathiswastik/misconception-vlm-verification",
        "newTitle": "Misconception VLM Verification",
        "text": code_text,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "competitionSources": ["misconception-detection-in-k-12-student-math-workings"],
        "competition_sources": ["misconception-detection-in-k-12-student-math-workings"],
        "datasetSources": [],
        "dataset_sources": []
    }

    url = "https://www.kaggle.com/api/v1/kernels/push"
    print(f"[Kaggle Deploy] Pushing script kernel ({len(code_text)} bytes) to: {payload['slug']}...")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"[Kaggle Deploy] Response Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"[Kaggle Deploy] Successfully pushed kernel! Version: {data.get('versionNumber')}")
        print(f"[Kaggle Deploy] URL: https://www.kaggle.com/code/{payload['slug']}")
        return data
    else:
        raise RuntimeError(f"Failed to push kernel: {resp.status_code} - {resp.text}")


if __name__ == "__main__":
    push_kernel_to_kaggle()
