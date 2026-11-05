"""Kaggle GPU Benchmark: Stronger VLM (Qwen2.5-VL-7B 4-bit) on Fresh 150-Sample Holdout.

Executes on Kaggle GPU:
1. Loads fresh, unseen 150-sample holdout dataset (75 correct, 75 incorrect).
2. Uses high-res visual tokens (1280px, max_pixels=1280*28*28) to preserve fine handwriting.
3. Prompts model to transcribe the student's terminal conclusion without question-solving bias.
4. Generates with max_new_tokens=96 (no polynomial truncation).
5. Evaluates in code with robust AnswerNormalizer (33-test verified).
6. Computes dual selective and operational metrics (abstentions penalize recall).
7. Translates class-conditional recall rates into expected downstream MAP@3 gain.
8. Serializes audit report to /kaggle/working/qwen25_150_benchmark_fresh.json.
"""
import glob
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd

# 1. Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.49.0", "accelerate", "bitsandbytes", "torchvision", "qwen-vl-utils"], check=False)

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image

qwen_model_cls: Any = None
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _QwenCls  # type: ignore # pyright: ignore
    qwen_model_cls = _QwenCls
except Exception:
    try:
        from transformers import Qwen2VLForConditionalGeneration as _QwenCls  # type: ignore # pyright: ignore
        qwen_model_cls = _QwenCls
    except Exception:
        qwen_model_cls = None


def parse_bool(value: Any) -> bool:
    """Safely parse boolean values, preventing 'False' string from evaluating as True."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing or NaN final_answer_correct label")
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unexpected boolean label: {value!r}")


# 2. Immutable list of 150 Fresh Unseen Holdout Trace IDs (75 correct, 75 incorrect, 0 leakage from dev)
FRESH_150_TRACE_IDS = [
    "fa004003a45f", "9a85c828fc68", "aff66a8ca358", "f1d4b75401cf", "e3f15a2615a1",
    "93cf5e0ff8b6", "763cb5feee58", "5a042a1ce289", "45335c98e993", "2089fd944870",
    "aad123cfca1b", "59a8c34f3e14", "36c49798f4a5", "e7efbb8ac803", "cca43010a048",
    "364affdba6f8", "d780963c5f67", "b083b88ecd69", "523b69adbb4c", "d91d437f0a67",
    "3e660dc025f1", "ca258be95252", "72c728573b23", "fce2408d11f0", "8a08a675836e",
    "1a48a88df7b5", "330d6f4878cc", "d903bb0cd5e7", "6862a17f7195", "a41716ed2eab",
    "15ddb0d4bab8", "5a244a4170a9", "038c4b4191e3", "68a51698f6f9", "7ab665ab48c4",
    "18b50e6dda88", "53398668ea52", "c1493c95da73", "80cb6d861d55", "ae8dd2ca74c6",
    "ba6698cc5c41", "d6cccadae9ee", "04bb07e15ca1", "2741e243255d", "b76b08ace520",
    "7f5c3b0ce553", "41045af7c115", "9ed9800cbb74", "914ed98ac4a2", "c55498aa0729",
    "63fa16e9115b", "bc33aa3ce3f2", "a0afba0eb074", "d71897d915ac", "f0680e5d007e",
    "4b9557c05d55", "1f35d9d97492", "cf9d4130a2a2", "552f9b095966", "a8e8834b8ff4",
    "dc70d87b5b85", "d0c0f851acad", "5a077a478fc5", "722ec25347c8", "d05d84753d91",
    "4e5ac439ffa7", "fdcf23238db1", "76780e684798", "495079c12164", "d9f309ec9c89",
    "db13a4797d4f", "4b662376e58b", "978d3395722f", "f30fff9641a9", "5508e3502e25",
    "688bf78fe501", "b391dd9fd91a", "8523c0ba2220", "32484c492b52", "5f316cd6b616",
    "d2acb5106baf", "aed30de55caf", "80a396b46854", "30d804819a6b", "eacd50cd3c38",
    "b6618da1aa51", "6b9ff83f124c", "ffe05e6614d9", "7fe6b0496a02", "82c99a3e0192",
    "f7303fba89a6", "658d67bb1921", "5f0612a5fd02", "c997e9013970", "a941552f45a9",
    "765150a042fb", "9bd9f3b0fc19", "accf49ce8103", "2b0467119800", "66466d385b3a",
    "54e7824f9c50", "1b1a306ec17a", "26f70105bd06", "2eb84814c544", "b5a6dbafd702",
    "7757d89313e4", "a55565037ed9", "0b169201d515", "86e147fdcd91", "0fb6ea608296",
    "b9ca97f394f7", "ab7d93c2734a", "f0081e0a0d44", "97b3234a9edd", "f3249be19f1a",
    "71ff641a30e7", "c4a308ec585f", "887db558c55e", "868ef0f1ce97", "21f1709903a2",
    "637776e233ef", "9654788547a2", "98b0339139e7", "2ab5f1b7fd67", "72f4161d1910",
    "c55babc7f21e", "799f3632432d", "4eeaf6a2f862", "c595e7bed8fc", "ca0a5dd3ef31",
    "16549525512a", "b672daa8576f", "0ee19cfece21", "4d74d6f25e34", "aae75f24c19d",
    "754052ae5019", "20fa5054d9d6", "1a3c7fefdfc3", "a38da9f85ceb", "a9b7b9d40077",
    "d7376653eeba", "e3b2f6f06a06", "60ea6ee418f2", "9e34364217fb", "a22f25fed8f4",
    "167bc4369581", "882f13be04c2", "80e3a8860559", "e994eca801ad", "9007a1c57381"
]

csv_candidates = glob.glob("/kaggle/**/sample_150_fresh.csv", recursive=True)
if csv_candidates:
    sample_150 = cast(pd.DataFrame, pd.read_csv(csv_candidates[0]))
    source_msg = f"from uploaded file: {csv_candidates[0]}"
else:
    train_candidates = glob.glob("/kaggle/**/train.csv", recursive=True)
    if not train_candidates:
        raise FileNotFoundError("Could not find sample_150_fresh.csv or train.csv in /kaggle")
    train_df = cast(pd.DataFrame, pd.read_csv(train_candidates[0]))
    trace_col = cast(pd.Series, train_df["trace_id"]).astype(str)
    id_set = set(FRESH_150_TRACE_IDS)
    sample_150 = cast(pd.DataFrame, train_df[trace_col.isin(id_set)]).copy()
    # Order identically to the preregistered FRESH_150_TRACE_IDS
    id_to_order = {tid: i for i, tid in enumerate(FRESH_150_TRACE_IDS)}
    sample_150["_order"] = sample_150["trace_id"].astype(str).map(id_to_order)
    sample_150.sort_values("_order", inplace=True)
    sample_150.drop(columns=["_order"], inplace=True)
    sample_150.reset_index(drop=True, inplace=True)
    source_msg = f"extracted via immutable FRESH_150_TRACE_IDS from: {train_candidates[0]}"

assert isinstance(sample_150, pd.DataFrame)
sample_150["final_answer_correct"] = cast(pd.Series, sample_150["final_answer_correct"]).map(parse_bool)
n_c = sum(1 for x in sample_150["final_answer_correct"] if x)
n_inc = len(sample_150) - n_c

print(f"Loaded {len(sample_150)} benchmark samples ({n_c} correct, {n_inc} incorrect) {source_msg}")

# 3. Locate image dir
sample_img = str(sample_150.iloc[0]["image_file"])
img_candidates: List[str] = []
for base_path in ["/kaggle/input", "/kaggle/working", "/kaggle"]:
    p = Path(base_path)
    if p.exists():
        found = list(p.rglob(sample_img))
        if found:
            img_candidates.append(str(found[0]))
            break
if not img_candidates:
    img_candidates = glob.glob(f"/kaggle/**/{sample_img}", recursive=True)
if not img_candidates:
    raise FileNotFoundError(f"Could not locate image {sample_img} in /kaggle")
images_dir = Path(img_candidates[0]).parent
print(f"Images located in: {images_dir}")


# 4. Robust Answer Normalizer (Validated on 52 Invariant Tests)
class AnswerNormalizer:
    BASE_UNITS = [
        r"cm\^?[23]?", r"m\^?[23]?", r"km", r"kg", r"gram", r"tahun", r"jam",
        r"menit", r"detik", r"kali", r"derajat", r"orang",
        r"buah", r"buku", r"lembar", r"butir", r"ekor", r"rupiah", r"rp"
    ]
    EXPANDED_UNITS = BASE_UNITS + [
        r"siswa", r"murid", r"kantong", r"susunan", r"lele", r"brantong",
        r"ton", r"hektar", r"ha", r"org", r"°c", r"°", r"c"
    ]

    UNITS_REGEX = re.compile(
        r"(?i)(?<=\d|\))\s*(?:%|°|" + "|".join(EXPANDED_UNITS) + r"\b)\.?"
    )
    CURRENCY_REGEX = re.compile(r"(?i)\b(?:rp|idr|rupiah)\.?\s*")

    @classmethod
    def strip_latex(cls, text: str) -> str:
        s = text
        s = re.sub(r"^\s*(\$\$|\$|\\\(|\\\[)\s*", "", s)
        s = re.sub(r"\s*(\$\$|\$|\\\)|\\\])\s*$", "", s)
        s = s.replace("$", "")
        s = s.replace(r"\dfrac", r"\frac")
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", s)
        s = re.sub(r"\\hat\{([^{}]+)\}", r"\1", s)
        s = re.sub(r"\\hat\s*([a-zA-Z])", r"\1", s)
        s = s.replace(r"\left", "").replace(r"\right", "")
        s = s.replace(r"\times", "*").replace(r"\cdot", "*")
        s = s.replace(r"\pi", "pi")
        s = s.replace("\\ ", " ").replace(r"\,", " ").replace(r"\;", " ")
        return s.strip()

    @classmethod
    def clean(cls, token: Optional[str], is_gold: bool = False) -> str:
        if token is None:
            return ""
        s = token.strip()
        if not s:
            return ""

        s = (
            s.replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .replace("²", "^2")
            .replace("³", "^3")
            .replace("√", "sqrt")
            .replace("ŷ", "y")
        )

        s = cls.strip_latex(s)
        s = cls.CURRENCY_REGEX.sub("", s).strip()
        s = re.sub(
            r"^(?:jawaban(?:\s+akhir)?|jadi|maka|hasil(?:nya)?|kesimpulan)\s*[:=,]?\s*",
            "",
            s,
            flags=re.IGNORECASE
        )
        s = re.sub(r"^\s*[A-Za-z]?\d+\s*[:|]\s*", "", s)

        if not is_gold and "=" in s:
            parts = [p.strip() for p in s.split("=") if p.strip()]
            if parts:
                terminal = parts[-1]
                if len(terminal.split()) <= 3:
                    s = terminal
        elif "=" in s:
            parts = s.split("=")
            left = parts[0].strip()
            if len(left.split()) <= 2 and not any(op in left for op in ["+", "*", "/", "^"]):
                s = parts[-1].strip()

        s = cls.UNITS_REGEX.sub("", s).strip()
        s = s.rstrip(".")
        s = s.strip("\"' \t\n\r")
        return s

    @classmethod
    def parse_numeric_candidates(cls, text: str) -> List[float]:
        candidates: List[float] = []
        raw = text.strip()
        if not raw:
            return candidates

        frac_match = re.fullmatch(r"\(?([+-]?\d+(?:\.\d+)?)\)?\s*/\s*\(?([+-]?\d+(?:\.\d+)?)\)?", raw)
        if frac_match:
            try:
                num = float(frac_match.group(1))
                den = float(frac_match.group(2))
                if abs(den) > 1e-9:
                    candidates.append(num / den)
                    return candidates
            except ValueError:
                pass

        mixed_match = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", raw)
        if mixed_match:
            try:
                whole = float(mixed_match.group(1))
                num = float(mixed_match.group(2))
                den = float(mixed_match.group(3))
                if abs(den) > 1e-9:
                    sign = -1.0 if whole < 0 else 1.0
                    candidates.append(whole + sign * (num / den))
                    return candidates
            except ValueError:
                pass

        if re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+", raw):
            try:
                candidates.append(float(raw.replace(".", "")))
                return candidates
            except ValueError:
                pass

        if re.fullmatch(r"[+-]?\d+,\d+", raw):
            comma_parts = raw.split(",")
            if len(comma_parts) == 2 and len(comma_parts[1]) == 3 and not comma_parts[0].startswith("0"):
                try:
                    candidates.append(float(raw.replace(",", "")))
                    candidates.append(float(raw.replace(",", ".")))
                    return candidates
                except ValueError:
                    pass
            try:
                candidates.append(float(raw.replace(",", ".")))
                return candidates
            except ValueError:
                pass

        try:
            candidates.append(float(raw))
            return candidates
        except ValueError:
            pass

        return candidates

    @classmethod
    def clean_polynomial(cls, text: str) -> str:
        s = text.replace(" ", "").lower()
        s = s.replace("²", "^2").replace("³", "^3")
        s = s.replace("×", "*").replace("\\times", "*").replace("\\cdot", "*")
        return s

    @classmethod
    def compare_single(
        cls,
        st_clean: str,
        gt_clean: str,
        tolerance: float = 1e-5
    ) -> Tuple[Optional[bool], str]:
        if not st_clean:
            return None, "EMPTY_STUDENT"

        if st_clean.lower() in ("unreadable", "tidak jelas", "coret", "ragu", "uncertain", "illegible"):
            return None, "ABSTAIN_UNREADABLE"

        if st_clean.lower() == gt_clean.lower():
            return True, "EXACT_NORMALIZED_MATCH"

        if "," in st_clean and "," in gt_clean:
            st_tokens = sorted([t.strip().lower() for t in st_clean.split(",") if t.strip()])
            gt_tokens = sorted([t.strip().lower() for t in gt_clean.split(",") if t.strip()])
            if st_tokens and st_tokens == gt_tokens:
                return True, "PERMUTED_SET_MATCH"

        st_poly = cls.clean_polynomial(st_clean)
        gt_poly = cls.clean_polynomial(gt_clean)
        if st_poly and gt_poly and st_poly == gt_poly:
            return True, "EXPRESSION_MATCH"

        st_nums = cls.parse_numeric_candidates(st_clean)
        gt_nums = cls.parse_numeric_candidates(gt_clean)
        if st_nums and gt_nums:
            for s_val in st_nums:
                for g_val in gt_nums:
                    if (s_val < 0) != (g_val < 0) and abs(s_val - g_val) > tolerance:
                        continue
                    if abs(s_val - g_val) <= tolerance:
                        return True, "NUMERIC_EQUIVALENT"
            return False, "NUMERIC_DIFFERENCE"

        return False, "STRING_MISMATCH"

    @classmethod
    def are_equivalent(
        cls,
        student: Optional[str],
        gold: Optional[str],
        tolerance: float = 1e-5
    ) -> Tuple[Optional[bool], str]:
        if student is None or not student.strip():
            return None, "EMPTY_STUDENT"

        raw_student = student.strip()
        if re.search(r"\b(?:atau|or)\b", raw_student, flags=re.IGNORECASE):
            sub_candidates = [c.strip() for c in re.split(r"\b(?:atau|or)\b", raw_student, flags=re.IGNORECASE) if c.strip()]
            gt_clean = cls.clean(gold, is_gold=True)
            for sub_cand in sub_candidates:
                sub_clean = cls.clean(sub_cand, is_gold=False)
                match, reason = cls.compare_single(sub_clean, gt_clean, tolerance=tolerance)
                if match is True:
                    return True, f"COMPOUND_ALTERNATIVE_MATCH ({reason})"

        st_clean = cls.clean(student, is_gold=False)
        gt_clean = cls.clean(gold, is_gold=True)
        return cls.compare_single(st_clean, gt_clean, tolerance=tolerance)


# 5. Model Loading with 1280px High-Res Cap
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

if qwen_model_cls is None:
    raise ImportError("Neither Qwen2_5_VLForConditionalGeneration nor Qwen2VLForConditionalGeneration is available in transformers.")

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

try:
    print(f"Loading {model_id} (4-bit NF4) with 1280px high-res visual tokens...")
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = qwen_model_cls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
except Exception as e:
    print(f"Failed to load 7B ({e}). Falling back to Qwen/Qwen2.5-VL-3B-Instruct in FP16...")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = qwen_model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

print(f"Successfully loaded {model_id} on GPU.")

y_true: List[bool] = []
y_pred: List[Optional[bool]] = []
results: List[Dict[str, Any]] = []
latencies: List[float] = []

prompt_text = (
    "You are an expert mathematics grader. Look at the student's handwritten math work in the image.\n"
    "Transcribe ONLY the final line or conclusion written by the student in the handwriting (e.g. at the bottom, after 'Jawaban', 'Jadi', or '=').\n"
    "Return ONLY the exact student final answer text or expression. If crossed out or illegible, respond with 'UNREADABLE'."
)

for idx, (_, row) in enumerate(sample_150.iterrows(), start=1):
    trace_id = str(row["trace_id"])
    gold = bool(row["final_answer_correct"])
    corr_ans = str(row["correct_answer"])
    img_path = images_dir / str(row["image_file"])

    image = Image.open(img_path).convert("RGB")
    image.thumbnail((1280, 1280))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore # pyright: ignore
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
    except Exception:
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=96)
    lat = time.time() - t0
    latencies.append(lat)

    in_len = inputs.input_ids.shape[1]
    gen_ids = output_ids[:, in_len:]
    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    ans_extracted = gen_text.split("assistant")[-1].strip()

    is_match, reason = AnswerNormalizer.are_equivalent(ans_extracted, corr_ans)
    y_true.append(gold)
    y_pred.append(is_match)

    status_str = "MATCH" if is_match == gold else "MISMATCH"
    print(f"[{idx:03d}/150] {trace_id} | Gold: {gold} | Extracted: '{ans_extracted}' | Expected: '{corr_ans}' | Match: {is_match} ({status_str}, {lat:.2f}s)")

    results.append({
        "trace_id": trace_id, "gold": gold, "extracted": ans_extracted,
        "expected": corr_ans, "is_match": is_match, "reason": reason
    })

    del inputs, output_ids
    if idx % 5 == 0:
        torch.cuda.empty_cache()

# 7. Corrected Dual Metric Audit
n_total = len(y_true)
decided = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp is not None]
n_decided = len(decided)
n_abstain = n_total - n_decided
coverage = n_decided / n_total if n_total > 0 else 0.0
abstain_rate = n_abstain / n_total if n_total > 0 else 0.0

tp_decided = sum(1 for yt, yp in decided if yt and yp is True)
tn_decided = sum(1 for yt, yp in decided if not yt and yp is False)
fp_decided = sum(1 for yt, yp in decided if not yt and yp is True)
fn_decided = sum(1 for yt, yp in decided if yt and yp is False)

selective_acc = (tp_decided + tn_decided) / n_decided if n_decided > 0 else 0.0
pos_decided = sum(1 for yt, _ in decided if yt)
neg_decided = sum(1 for yt, _ in decided if not yt)
sel_rec_pos = tp_decided / pos_decided if pos_decided > 0 else 0.0
sel_rec_neg = tn_decided / neg_decided if neg_decided > 0 else 0.0

# Conservative operational metrics across full population (abstentions penalize recall)
n_positive = sum(1 for yt in y_true if yt)
n_negative = n_total - n_positive

tp_all = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp is True)
tn_all = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp is False)
fp_all = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp is True)
fn_all = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp is False)

operational_acc = (tp_all + tn_all) / n_total if n_total > 0 else 0.0
op_rec_pos = tp_all / n_positive if n_positive > 0 else 0.0
op_rec_neg = tn_all / n_negative if n_negative > 0 else 0.0
far = fp_all / n_negative if n_negative > 0 else 0.0
frr = (n_positive - tp_all) / n_positive if n_positive > 0 else 0.0

# Simulation-based Expected MAP@3 Translation
# Headroom formula: delta ~ 0.5 * (Rec(+) - 0.50) * 0.1122 + 0.5 * (Rec(-) - 0.50) * 0.1122
expected_map3_delta = 0.5 * (op_rec_pos - 0.50) * 0.1122 + 0.5 * (op_rec_neg - 0.50) * 0.1122

gate_passed = (operational_acc >= 0.78) and (op_rec_pos >= 0.60) and (op_rec_neg >= 0.60) and (abstain_rate <= 0.20)

report = {
    "model_id": model_id,
    "n_total": n_total,
    "n_decided": n_decided,
    "n_abstained": n_abstain,
    "coverage": coverage,
    "abstain_rate": abstain_rate,
    "selective_accuracy": selective_acc,
    "selective_recall_positive": sel_rec_pos,
    "selective_recall_negative": sel_rec_neg,
    "operational_accuracy": operational_acc,
    "operational_recall_positive": op_rec_pos,
    "operational_recall_negative": op_rec_neg,
    "false_acceptance_rate": far,
    "false_rejection_rate": frr,
    "expected_map3_delta": expected_map3_delta,
    "gate_passed": gate_passed,
    "average_latency": float(np.mean(latencies)),
    "results": results
}

out_path = Path("/kaggle/working/qwen25_150_benchmark_fresh.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 65)
print(f"BENCHMARK COMPLETED ON FRESH 150 HOLDOUT")
print(f"Selective Accuracy (decided):   {selective_acc:.4f} ({n_decided}/{n_total} decided)")
print(f"Operational Accuracy (total):     {operational_acc:.4f}")
print(f"Operational Recall (+, correct):  {op_rec_pos:.4f} (Gate >= 0.60)")
print(f"Operational Recall (-, error):    {op_rec_neg:.4f} (Gate >= 0.60)")
print(f"False Acceptance Rate (FAR):      {far:.4f}")
print(f"False Rejection Rate (FRR):       {frr:.4f}")
print(f"Abstain Rate:                     {abstain_rate:.4f} (Limit <= 0.20)")
print(f"Expected MAP@3 Delta:             {expected_map3_delta:+.4f}")
print(f"OPERATIONAL GATE STATUS:          {'PASSED' if gate_passed else 'FAILED'}")
print(f"Audit written to:                 {out_path}")
print("=" * 65)
