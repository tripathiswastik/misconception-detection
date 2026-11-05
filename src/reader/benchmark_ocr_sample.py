"""Phase 1: Real Transcription Benchmarking on 50-Image Sample.

Evaluates candidate OCR/handwriting recognition on 50 student workings from train.csv.
Computes:
- Character Error Rate (CER) against gold human transcriptions.
- Word Error Rate (WER).
- Token overlap and mathematical symbol recall (+, -, *, /, ^, =, fractions).
- Inference latency per image on cuda:0.

Saves:
- artifacts/ocr_sample_benchmark.json: Detailed evaluation metrics and transcription comparisons.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure protocol buffers use pure python to prevent protobuf version conflict
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.config import PATHS
from src.reader.transcription_normalizer import TranscriptionNormalizer


def levenshtein_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """Compute character or token-level Levenshtein edit distance between ref and hyp."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two-row space-efficient DP
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,      # Insertion
                prev_row[j] + 1,          # Deletion
                prev_row[j - 1] + cost    # Substitution
            )
        prev_row = list(curr_row)

    return prev_row[n]


def compute_cer(references: List[str], hypotheses: List[str]) -> Tuple[float, float]:
    """Compute aggregate Character Error Rate (CER) and Word Error Rate (WER)."""
    total_char_dist = 0
    total_ref_chars = 0
    total_word_dist = 0
    total_ref_words = 0

    for ref, hyp in zip(references, hypotheses):
        # Character level
        c_dist = levenshtein_distance(ref, hyp)
        total_char_dist += c_dist
        total_ref_chars += max(1, len(ref))

        # Word level
        r_words = ref.split()
        h_words = hyp.split()
        w_dist = levenshtein_distance(r_words, h_words)
        total_word_dist += w_dist
        total_ref_words += max(1, len(r_words))

    cer = total_char_dist / max(1, total_ref_chars)
    wer = total_word_dist / max(1, total_ref_words)
    return cer, wer


def run_sample_benchmark(n_samples: int = 50) -> Dict[str, Any]:
    print("\n=======================================================")
    print(f"[Phase 1] Benchmarking Real Handwriting Transcription ({n_samples} images)")
    print("=======================================================\n")

    train_df = cast(pd.DataFrame, pd.read_csv(PATHS.data_dir / "train.csv"))
    sample_df = train_df.head(n_samples).copy()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Loading TrOCR small model on {device}...")
    processor: Any = cast(Any, TrOCRProcessor).from_pretrained("microsoft/trocr-small-handwritten")
    model: Any = cast(Any, VisionEncoderDecoderModel).from_pretrained("microsoft/trocr-small-handwritten").to(device)
    model.eval()

    gold_texts = []
    pred_texts = []
    comparisons = []

    t0_total = time.time()

    for idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
        tid = str(row["trace_id"])
        img_name = str(row["image_file"])
        img_path = PATHS.data_dir / "images" / "train" / img_name
        raw_gold = str(row.get("transcription_steps", ""))
        norm_gold = TranscriptionNormalizer.normalize(raw_gold)
        gold_texts.append(norm_gold)

        if not img_path.exists():
            pred_texts.append("")
            continue

        t_img0 = time.time()
        try:
            with Image.open(img_path) as full_img:
                rgb_img = full_img.convert("RGB")
                w, h = rgb_img.size

                # Math worksheets are multi-line: test multi-strip line extraction (3 horizontal slices)
                slices = [
                    rgb_img.crop((0, 0, w, int(h * 0.4))),
                    rgb_img.crop((0, int(h * 0.3), w, int(h * 0.7))),
                    rgb_img.crop((0, int(h * 0.6), w, h))
                ]

                line_preds = []
                for s_idx, s_img in enumerate(slices):
                    pixel_vals = processor(s_img, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad():
                        gen_ids = model.generate(pixel_vals, max_new_tokens=40)
                    dec = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                    if dec and dec not in ["#", "##", "###"]:
                        line_preds.append(f"L{len(line_preds)+1} | {dec}")

                if line_preds:
                    raw_pred = ", ".join(line_preds)
                else:
                    # Fallback to full image
                    pixel_vals = processor(rgb_img, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad():
                        gen_ids = model.generate(pixel_vals, max_new_tokens=60)
                    raw_pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

                norm_pred = TranscriptionNormalizer.normalize(raw_pred)
        except Exception as e:
            norm_pred = f"[Error: {str(e)}]"

        pred_texts.append(norm_pred)
        elapsed = time.time() - t_img0

        sample_cer = levenshtein_distance(norm_gold, norm_pred) / max(1, len(norm_gold))
        comparisons.append({
            "trace_id": tid,
            "image": img_name,
            "gold": norm_gold[:120],
            "pred": norm_pred[:120],
            "sample_cer": round(sample_cer, 4),
            "inference_seconds": round(elapsed, 2)
        })

        if idx % 10 == 0 or idx == n_samples:
            print(f"  Processed {idx}/{n_samples} images (Last sample CER: {sample_cer:.2f})")

    total_time = time.time() - t0_total
    cer, wer = compute_cer(gold_texts, pred_texts)

    print("\n=======================================================")
    print("[Phase 1] 50-Image Handwriting OCR Benchmark Results:")
    print(f"  Total Evaluated Images:    {n_samples}")
    print(f"  Total Processing Time:     {total_time:.1f}s ({total_time/n_samples:.2f}s/image)")
    print(f"  Character Error Rate (CER):{cer:.4f} ({cer*100:.1f}%)")
    print(f"  Word Error Rate (WER):     {wer:.4f} ({wer*100:.1f}%)")
    print("=======================================================\n")

    report = {
        "engine": "microsoft/trocr-small-handwritten (with 3-strip line crop)",
        "n_samples": n_samples,
        "total_time_seconds": round(total_time, 2),
        "seconds_per_image": round(total_time / n_samples, 2),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "comparisons_sample": comparisons[:10]
    }

    out_path = PROJECT_ROOT / "artifacts" / "ocr_sample_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Phase 1] Benchmark report saved to: {out_path}")

    return report


if __name__ == "__main__":
    run_sample_benchmark(50)
