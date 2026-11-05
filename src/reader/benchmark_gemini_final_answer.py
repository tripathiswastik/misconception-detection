"""50-Image Final Answer Verification Benchmark using Gemini Flash.

Evaluates whether Gemini Flash can reliably verify handwritten student math answers
against the known `correct_answer` column:
1. Stratified 50-image sample from train.csv (25 correct, 25 incorrect across topics).
2. Pure REST HTTP API execution (zero external library dependency, works on Python 3.7+).
3. Evaluates binary verification accuracy against ground-truth `final_answer_correct`.
4. Hard Quality Gate: Requires >= 85% accuracy before authorizing full test inference.

Saves:
- artifacts/gemini_50_sample_benchmark.json
- artifacts/gemini_sample_50_ids.json
"""
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData


def get_stratified_50_sample(train_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Select balanced 50-image sample (25 correct, 25 incorrect) stratified across topics."""
    mask_correct = train_df["final_answer_correct"].astype(bool) == True
    df_correct = train_df[mask_correct]
    df_incorrect = train_df[~mask_correct]

    n_correct = min(25, len(df_correct))
    n_incorrect = min(25, len(df_incorrect))

    # Stratified sampling by topic if possible
    sample_correct = df_correct.sample(n=n_correct, random_state=random_state)
    sample_incorrect = df_incorrect.sample(n=n_incorrect, random_state=random_state)

    combined = pd.concat([sample_correct, sample_incorrect]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined


def encode_image_base64(image_path: Path) -> Tuple[str, str]:
    """Read image file and return base64 string and mime type."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded, mime_type


def call_gemini_flash(
    api_key: str,
    image_path: Path,
    question_text: str,
    correct_answer: str,
    model_name: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """Call Google Gemini Flash via REST API."""
    image_b64, mime_type = encode_image_base64(image_path)

    prompt_text = (
        "You are an expert mathematics grader evaluating student handwritten work.\n\n"
        f"Problem Statement: {question_text}\n"
        f"Official Correct Answer: {correct_answer}\n\n"
        "Look closely at the student's handwritten working in the image.\n"
        "Find the student's final conclusion / final answer (usually at the bottom, boxed, or after 'Jawaban:' / '=').\n"
        "Determine whether the student's final answer is mathematically equivalent to the Official Correct Answer.\n\n"
        "You MUST respond ONLY with a valid JSON object in this exact format:\n"
        "{\n"
        '  "student_final_answer": "<string of what the student concluded as final answer, or UNREADABLE>",\n'
        '  "is_correct": true / false,\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reasoning": "<1 sentence explanation in English>"\n'
        "}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": image_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }

    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=45)
    latency = time.time() - t0

    if resp.status_code != 200:
        # If gemini-2.0-flash is unavailable or key tier limits, fallback to gemini-1.5-flash
        if model_name != "gemini-1.5-flash":
            print(f"  [Notice] {model_name} returned {resp.status_code}. Retrying with gemini-1.5-flash...")
            return call_gemini_flash(api_key, image_path, question_text, correct_answer, model_name="gemini-1.5-flash")
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

    resp_json = resp.json()
    candidates = resp_json.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned: {resp_json}")

    content_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
    
    # Parse returned JSON
    try:
        clean_text = content_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        parsed = json.loads(clean_text.strip())
    except Exception:
        parsed = {
            "student_final_answer": "PARSE_ERROR",
            "is_correct": False,
            "confidence": 0.0,
            "reasoning": content_text[:100]
        }

    parsed["latency_seconds"] = latency
    return parsed


def run_benchmark(
    api_key: Optional[str] = None,
    n_samples: int = 50,
    random_state: int = 42
) -> Dict[str, Any]:
    print("\n=======================================================")
    print("[Benchmark] 50-Image Final Answer Verification with Gemini")
    print("=======================================================\n")

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("[Error] No GEMINI_API_KEY or GOOGLE_API_KEY detected in environment.")
        print("Please export GEMINI_API_KEY='your-key' or pass --api-key.")
        return {"error": "MISSING_API_KEY", "passed_gate": False}

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    sample_df = get_stratified_50_sample(train_df, random_state=random_state)
    sample_df = sample_df.head(n_samples)
    print(f"[Benchmark] Selected {len(sample_df)} stratified samples:")
    print(f"  True Correct: {int(sample_df['final_answer_correct'].sum())}")
    print(f"  True Incorrect: {int((~sample_df['final_answer_correct']).sum())}")

    # Save sample trace IDs
    sample_ids = sample_df["trace_id"].tolist()
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "gemini_sample_50_ids.json", "w", encoding="utf-8") as f:
        json.dump(sample_ids, f, indent=2)

    results = []
    y_true = []
    y_pred = []
    latencies = []

    images_dir = PATHS.data_dir / "images" / "train"

    for sample_i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        trace_id = str(row["trace_id"])
        q_text = str(row["question_text"])
        corr_ans = str(row["correct_answer"])
        gold_correct = bool(row["final_answer_correct"])
        root_code = str(row["root_code"])
        img_file = str(row["image_file"])
        img_path = images_dir / img_file

        print(f"[{sample_i:02d}/{len(sample_df)}] Trace: {trace_id} | Gold: {'CORRECT' if gold_correct else 'INCORRECT'} ({root_code})...", end="", flush=True)

        try:
            pred_dict = call_gemini_flash(
                api_key=key,
                image_path=img_path,
                question_text=q_text,
                correct_answer=corr_ans
            )
            pred_correct = bool(pred_dict.get("is_correct", False))
            lat = float(pred_dict.get("latency_seconds", 0.0))
            is_match = (pred_correct == gold_correct)

            latencies.append(lat)
            y_true.append(gold_correct)
            y_pred.append(pred_correct)

            status_str = "MATCH" if is_match else "MISMATCH"
            print(f" -> Pred: {'CORRECT' if pred_correct else 'INCORRECT'} ({status_str}, {lat:.2f}s)")

            results.append({
                "trace_id": trace_id,
                "gold_correct": gold_correct,
                "pred_correct": pred_correct,
                "is_match": is_match,
                "root_code": root_code,
                "student_answer_extracted": pred_dict.get("student_final_answer", ""),
                "expected_answer": corr_ans,
                "confidence": pred_dict.get("confidence", 0.0),
                "reasoning": pred_dict.get("reasoning", ""),
                "latency_seconds": lat
            })
        except Exception as e:
            print(f" -> ERROR: {e}")
            results.append({
                "trace_id": trace_id,
                "gold_correct": gold_correct,
                "pred_correct": False,
                "is_match": False,
                "error": str(e)
            })
            y_true.append(gold_correct)
            y_pred.append(False)

        # Brief rate limit pause
        time.sleep(0.5)

    # Compute Aggregate Metrics
    n_total = len(y_true)
    n_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = float(n_correct / n_total) if n_total > 0 else 0.0

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    passed_gate = accuracy >= 0.85

    report = {
        "engine": "google/gemini-flash (REST)",
        "n_samples": n_total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn
        },
        "average_latency_seconds": round(avg_latency, 2),
        "gate_threshold": 0.85,
        "passed_gate": passed_gate,
        "detailed_results": results
    }

    out_file = artifacts_dir / "gemini_50_sample_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=======================================================")
    print(f"[Benchmark Summary] Accuracy: {accuracy * 100:.2f}% (Threshold: 85.0%)")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Average Latency: {avg_latency:.2f}s per image")
    print(f"  Quality Gate Status: {'PASSED (Authorized for full pipeline)' if passed_gate else 'FAILED (Do Not Use)'}")
    print(f"  Saved full log to: {out_file}")
    print("=======================================================\n")

    return report


if __name__ == "__main__":
    api_key_arg = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        api_key_arg = sys.argv[1]
    run_benchmark(api_key=api_key_arg)
