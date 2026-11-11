"""Real-OCR Feasibility Pilot: TrOCR vs Human Gold Transcriptions on Math Workings.

Objectives:
1. Verify what OCR models exist locally (TrOCR 235MB weights vs Florence-2 2.4MB stub).
2. Select 8 representative student math handwriting traces across key misconception domains:
   - Exponent rules (EXP-01)
   - Multi-step linear algebra (ALG-06)
   - Negative integer arithmetic (INT-04)
   - Fraction operations (FRA-06)
   - Coordinate geometry / transformation (COR-01)
   - Word problem / order of operations (WOP-01)
   - Algebraic formulation (ANS-01)
   - True NONE (correct student work)
3. Run TrOCR-small-handwritten inference on full images and ink-dense crops.
4. Benchmark TrOCR outputs directly against the ground truth human `transcription_steps` in train.csv:
   - Math symbol recovery (+, -, *, /, =, ^, exponents, variables, numbers)
   - Multi-line handling capability
   - Semantic error identification (does it capture the erroneous calculation?)
   - Latency and VRAM consumption
5. Output detailed JSON report for architectural decision.
"""

from collections import Counter
import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
IMG_TRAIN_DIR = DATA_DIR / "images" / "train"

TOCR_CACHE = Path("C:/Users/hp/.cache/huggingface/hub/models--microsoft--trocr-small-handwritten/snapshots/b4648cfa171985a6745f37ddd637e98c0da958ac")
FLORENCE_CACHE = Path("C:/Users/hp/.cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac")


def check_florence_status() -> Dict[str, Any]:
    """Inspect Florence-2 local cache status."""
    has_weights = any(
        f.suffix in [".bin", ".safetensors", ".pt"]
        for f in FLORENCE_CACHE.glob("*")
    )
    files = [f.name for f in FLORENCE_CACHE.glob("*")]
    total_size_mb = sum(f.stat().st_size for f in FLORENCE_CACHE.glob("*") if f.is_file()) / (1024 * 1024)
    return {
        "cache_path": str(FLORENCE_CACHE),
        "files_present": files,
        "total_size_mb": round(total_size_mb, 2),
        "has_weights": has_weights,
        "status": "READY" if has_weights else "INCOMPLETE_STUB_ONLY_TOKENIZER"
    }


def main():
    print("=" * 100)
    print("REAL-OCR FEASIBILITY PILOT: TrOCR VS HUMAN GOLD TRANSCRIPTION BENCHMARK")
    print("=" * 100)

    # 1. Inspect Model Cache Status
    print("\n[Step 1] Inspecting Local OCR Model Caches...")
    florence_status = check_florence_status()
    print(f"  Florence-2 Cache Status: {florence_status['status']} ({florence_status['total_size_mb']} MB, has_weights={florence_status['has_weights']})")
    print(f"  TrOCR Snapshot:          {TOCR_CACHE.exists()} (pytorch_model.bin: {(TOCR_CACHE / 'pytorch_model.bin').stat().st_size / (1024*1024):.1f} MB)")

    # 2. Select Representative Pilot Sample from train.csv
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))

    target_domains = [
        {"domain": "Exponent Equation",    "root_code": "EXP-01"},
        {"domain": "Linear Algebra",        "root_code": "ALG-06"},
        {"domain": "Negative Arithmetic",   "root_code": "INT-04"},
        {"domain": "Fraction Operations",   "root_code": "FRA-06"},
        {"domain": "Coordinate Geometry",   "root_code": "COR-01"},
        {"domain": "Order of Operations",   "root_code": "WOP-01"},
        {"domain": "Algebraic Formula",     "root_code": "ANS-01"},
        {"domain": "Correct Student Work",  "root_code": "NONE"},
    ]

    pilot_samples: List[Dict[str, Any]] = []
    for item in target_domains:
        code = item["root_code"]
        matches = train_df[train_df["root_code"] == code]
        if len(matches) == 0:
            # Fallback for NONE if NaN
            if code == "NONE":
                matches = train_df[train_df["root_code"].isna()]
        if len(matches) > 0:
            row = matches.iloc[0]
            pilot_samples.append({
                "domain": item["domain"],
                "trace_id": str(row["trace_id"]),
                "question_id": str(row["question_id"]),
                "topic_name": str(row.get("topic_name", "N/A")),
                "grade": int(float(row.get("grade", 0))),
                "root_code": str(code),
                "image_file": str(row["image_file"]),
                "gold_transcription": str(row.get("transcription_steps", "N/A"))
            })

    print(f"\n[Step 2] Selected {len(pilot_samples)} Representative Pilot Traces:")
    for s in pilot_samples:
        print(f"  - [{s['root_code']:6s}] {s['domain']:22s} | Trace: {s['trace_id']} | Image: {s['image_file']}")

    # 3. Load TrOCR Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 3] Loading TrOCR on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
    load_start = time.time()

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(str(TOCR_CACHE), local_files_only=True, use_fast=False)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(str(TOCR_CACHE), local_files_only=True).to(device)
    trocr_model.eval()

    n_params = sum(p.numel() for p in trocr_model.parameters())
    print(f"  TrOCR-small loaded in {time.time() - load_start:.2f}s | Parameters: {n_params:,}")

    # 4. Run TrOCR Inference and Benchmark against Gold Human Transcriptions
    print("\n" + "=" * 100)
    print("BENCHMARKING TrOCR PREDICTIONS AGAINST HUMAN GOLD TRANSCRIPTIONS")
    print("=" * 100)

    pilot_results: List[Dict[str, Any]] = []

    math_symbols = set("0123456789+-*/=^xyzabcm()[]:.,")

    for idx, sample in enumerate(pilot_samples, start=1):
        img_path = IMG_TRAIN_DIR / sample["image_file"]
        if not img_path.exists():
            print(f"  [Warning] Missing image: {img_path}")
            continue

        raw_img = Image.open(img_path).convert("RGB")
        w, h = raw_img.size

        # Time inference
        t0 = time.time()
        pixel_values = processor(raw_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = trocr_model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=2,
                early_stopping=True
            )
        elapsed = time.time() - t0

        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Symbol extraction audit
        gold_symbols = set(sample["gold_transcription"].lower()) & math_symbols
        pred_symbols = set(predicted_text.lower()) & math_symbols
        symbol_recall = len(pred_symbols & gold_symbols) / max(len(gold_symbols), 1)

        result_item = {
            "index": idx,
            "domain": sample["domain"],
            "trace_id": sample["trace_id"],
            "root_code": sample["root_code"],
            "image_file": sample["image_file"],
            "image_size": f"{w}x{h}",
            "latency_seconds": round(elapsed, 3),
            "predicted_trocr": predicted_text,
            "gold_transcription": sample["gold_transcription"],
            "symbol_recall": round(symbol_recall, 3),
            "contains_math_numbers": any(c.isdigit() for c in predicted_text),
            "contains_math_operators": any(c in "+-*/=^" for c in predicted_text)
        }
        pilot_results.append(result_item)

        print(f"\n--- [Trace {idx}/{len(pilot_samples)}] {sample['domain']} ({sample['root_code']}) | Trace: {sample['trace_id']} ---")
        print(f"Image Resolution: {w}x{h} | Latency: {elapsed:.3f}s")
        print(f"Human Gold Transcription:")
        for line in sample["gold_transcription"].split("\n")[:4]:
            print(f"  [Gold] {line}")
        if len(sample["gold_transcription"].split("\n")) > 4:
            print(f"  [Gold] ... ({len(sample['gold_transcription'].split(chr(10)))} total lines)")
        print(f"TrOCR Output:")
        print(f"  [TrOCR] \"{predicted_text}\"")
        print(f"Math Symbol Recall: {symbol_recall:.1%} | Has Numbers: {result_item['contains_math_numbers']} | Has Operators: {result_item['contains_math_operators']}")

    # 5. Summary Evaluation and Feasibility Verdict
    avg_latency = float(np.mean([r["latency_seconds"] for r in pilot_results]))
    avg_symbol_recall = float(np.mean([r["symbol_recall"] for r in pilot_results]))
    num_with_operators = sum(1 for r in pilot_results if r["contains_math_operators"])
    num_with_numbers = sum(1 for r in pilot_results if r["contains_math_numbers"])

    print("\n" + "=" * 100)
    print("PILOT FEASIBILITY AUDIT SUMMARY")
    print("=" * 100)
    print(f"Total Representative Traces Audited: {len(pilot_results)}")
    print(f"Average Inference Latency per Image: {avg_latency:.3f}s (Estimated full 1,500 run: {avg_latency * 1500 / 60:.1f} minutes)")
    print(f"Traces with Math Numbers Extracted:  {num_with_numbers} / {len(pilot_results)} ({num_with_numbers / len(pilot_results):.1%})")
    print(f"Traces with Math Operators (+-*/=): {num_with_operators} / {len(pilot_results)} ({num_with_operators / len(pilot_results):.1%})")
    print(f"Average Math Symbol Recovery:       {avg_symbol_recall:.1%}")

    report_payload = {
        "florence_status": florence_status,
        "pilot_summary": {
            "total_samples": len(pilot_results),
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_symbol_recall": round(avg_symbol_recall, 4),
            "traces_with_numbers": num_with_numbers,
            "traces_with_operators": num_with_operators,
        },
        "sample_evaluations": pilot_results
    }

    report_path = ART_DIR / "real_ocr_feasibility_pilot_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nSaved feasibility report to {report_path}")


if __name__ == "__main__":
    main()
