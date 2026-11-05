"""Kaggle Autonomous VLM Verification Benchmark (50 Samples).

Executes on Kaggle GPU:
1. Auto-discovers competition images and train.csv.
2. Selects 50 stratified samples (25 correct, 25 incorrect, seed=42).
3. If GEMINI_API_KEY is found in Kaggle Secrets / env, queries Gemini Flash via REST.
4. Otherwise, loads Qwen2-VL-2B-Instruct onto GPU in FP16 to verify student final answers.
5. Saves results and confusion matrix to /kaggle/working/vlm_50_sample_benchmark.json.
"""
import base64
import glob
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

print("=======================================================")
print("[Kaggle VLM Benchmark] Initializing Verification Engine")
print("=======================================================")

print("Searching /kaggle filesystem:")
for root, dirs, files in os.walk("/kaggle"):
    if len(files) > 0 or len(dirs) > 0:
        print(f"  {root}: {dirs[:3]} | {files[:5]}")

csv_candidates = glob.glob("/kaggle/input/**/train.csv", recursive=True)
if not csv_candidates:
    csv_candidates = glob.glob("/kaggle/**/train.csv", recursive=True)
if not csv_candidates:
    print("Could not find train.csv. Listing all csv files in /kaggle:")
    print(glob.glob("/kaggle/**/*.csv", recursive=True))
    raise FileNotFoundError("Could not find train.csv in /kaggle")
train_csv_path = Path(csv_candidates[0])
comp_dir = train_csv_path.parent
train_df = pd.read_csv(train_csv_path)

# Stratified 50 sample
mask_corr = train_df["final_answer_correct"].astype(bool) == True
sample_corr = train_df[mask_corr].sample(25, random_state=42)
sample_incorr = train_df[~mask_corr].sample(25, random_state=42)
sample_df = pd.concat([sample_corr, sample_incorr]).sample(frac=1.0, random_state=42).reset_index(drop=True)

train_img_candidates = glob.glob("/kaggle/input/**/images/train", recursive=True)
if train_img_candidates:
    images_dir = Path(train_img_candidates[0])
elif (comp_dir / "competition_data" / "images" / "train").exists():
    images_dir = comp_dir / "competition_data" / "images" / "train"
elif (comp_dir / "images" / "train").exists():
    images_dir = comp_dir / "images" / "train"
else:
    sample_img_name = str(sample_df.iloc[0]["image_file"])
    found_sample = glob.glob(f"/kaggle/input/**/{sample_img_name}", recursive=True)
    if found_sample:
        images_dir = Path(found_sample[0]).parent
    else:
        raise FileNotFoundError(f"Could not locate train images directory containing {sample_img_name}")

print(f"Data root: {comp_dir}")
print(f"Images root: {images_dir}")
print(f"Selected {len(sample_df)} stratified samples (25 correct, 25 incorrect).")

# Check for Gemini key in Kaggle Secrets
gemini_key = ""
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    gemini_key = user_secrets.get_secret("GEMINI_API_KEY")
    print("Found GEMINI_API_KEY in Kaggle Secrets.")
except Exception:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

results = []
y_true = []
y_pred = []
latencies = []

if gemini_key:
    print("Using Gemini Flash REST API...")
    for sample_i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        trace_id = str(row["trace_id"])
        q_text = str(row["question_text"])
        corr_ans = str(row["correct_answer"])
        gold_correct = bool(row["final_answer_correct"])
        img_path = images_dir / str(row["image_file"])

        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            f"Question: {q_text}\nExpected Answer: {corr_ans}\n\n"
            "Look at the student's handwritten working in the image. "
            "Determine whether the student's final conclusion matches the Expected Answer.\n"
            'Return JSON: {"is_correct": true/false, "student_final_answer": "...", "confidence": 0.0-1.0}'
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64_data}}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
        }
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=45)
            lat = time.time() - t0
            latencies.append(lat)
            if r.status_code == 200:
                t_out = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if t_out.startswith("```json"): t_out = t_out[7:]
                if t_out.startswith("```"): t_out = t_out[3:]
                if t_out.endswith("```"): t_out = t_out[:-3]
                parsed = json.loads(t_out.strip())
                pred_c = bool(parsed.get("is_correct", False))
                extracted = parsed.get("student_final_answer", "")
            else:
                pred_c = False
                extracted = f"HTTP_{r.status_code}"
        except Exception as e:
            lat = time.time() - t0
            latencies.append(lat)
            pred_c = False
            extracted = str(e)

        is_match = (pred_c == gold_correct)
        y_true.append(gold_correct)
        y_pred.append(pred_c)
        status_msg = "MATCH" if is_match else "MISMATCH"
        print(f"[{sample_i:02d}/50] Trace: {trace_id} | Gold: {gold_correct} | Pred: {pred_c} ({status_msg}, {lat:.2f}s)")
        results.append({
            "trace_id": trace_id, "gold": gold_correct, "pred": pred_c,
            "is_match": is_match, "extracted": extracted
        })
        time.sleep(0.5)

else:
    print("No GEMINI_API_KEY found. Running open-weight Qwen2-VL-2B on Kaggle GPU...")
    os.system("pip install -q transformers>=4.45.0 accelerate qwen-vl-utils")
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("Loading Qwen2-VL-2B-Instruct onto GPU...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    for sample_i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        trace_id = str(row["trace_id"])
        q_text = str(row["question_text"])
        corr_ans = str(row["correct_answer"])
        gold_correct = bool(row["final_answer_correct"])
        img_path = images_dir / str(row["image_file"])

        prompt_text = (
            f"Problem: {q_text}\n"
            f"Official Correct Answer: {corr_ans}\n"
            "Does the student's handwritten final answer in this image equal the Official Correct Answer? "
            "Answer with exactly YES or NO."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=32)
        lat = time.time() - t0
        latencies.append(lat)

        gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        ans = gen_text.split("assistant")[-1].strip().upper()
        pred_c = ("YES" in ans)

        is_match = (pred_c == gold_correct)
        y_true.append(gold_correct)
        y_pred.append(pred_c)
        status_msg = "MATCH" if is_match else "MISMATCH"
        print(f"[{sample_i:02d}/50] Trace: {trace_id} | Gold: {gold_correct} | Pred: {pred_c} ({status_msg}, {lat:.2f}s)")
        results.append({
            "trace_id": trace_id, "gold": gold_correct, "pred": pred_c,
            "is_match": is_match, "output": ans
        })

acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
passed = acc >= 0.85

print("\n=======================================================")
print(f"VLM Verification Accuracy: {acc*100:.2f}% (Threshold: 85.0%)")
print(f"Average Latency: {np.mean(latencies):.2f}s per image")
print(f"Quality Gate Status: {'PASSED' if passed else 'FAILED'}")
print("=======================================================")

out_path = Path("/kaggle/working/vlm_50_sample_benchmark.json")
with open(out_path, "w") as f:
    json.dump({
        "accuracy": acc, "passed_gate": passed,
        "n_samples": len(y_true), "average_latency": float(np.mean(latencies)),
        "results": results
    }, f, indent=2)

print(f"Saved results to {out_path}")
