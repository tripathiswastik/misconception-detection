"""Decoupled High-Throughput Feature Extraction & Distillation for 3,500 Traces.

Architecture:
  - Step 1: ConvNeXt-448 Uniform Letterbox (768-d, batch 16, ~65s)
  - Step 2: Swin-T @ 224x224 (768-d, batch 32, ~30s)
  - Step 3: MiniLM Text Embeddings (384-d, 282 questions, ~2s)
  - Step 4: Fast Vectorized Strokes (24-d, 128x128, ~12s)
  - Step 5: Soft Probability Distillation from 0.54616 Champion Teacher (~2s)
  
Outputs:
  - artifacts/pseudo_3500_features_1944.npy (3500, 1944)
  - artifacts/pseudo_3500_soft_targets_76.npy (3500, 76)
"""

import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Device for 3,500 Pipeline: {device}", flush=True)


def letterbox_image(img: Image.Image, target_size: int = 448, fill_color=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    padded.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded


def extract_stroke_fast(img_rgb: Image.Image) -> np.ndarray:
    try:
        img_small = img_rgb.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img_small, dtype=np.uint8)
        ink = (arr < 150).astype(np.float32)
        ink_sum = float(ink.sum())
        if ink_sum < 5:
            return np.zeros(24, dtype=np.float32)

        ink_density = ink_sum / (128.0 * 128.0)
        rows = np.where(ink.any(axis=1))[0]
        cols = np.where(ink.any(axis=0))[0]
        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        box_h = max(1, y1 - y0)
        box_w = max(1, x1 - x0)
        box_coverage = (box_h * box_w) / (128.0 * 128.0)
        box_ink_density = ink_sum / (box_h * box_w)

        box_ink = ink[y0 : y1 + 1, x0 : x1 + 1]
        grid_feats = []
        for r_i in range(4):
            r_start = int(r_i * box_h / 4)
            r_end = int((r_i + 1) * box_h / 4)
            for c_i in range(4):
                c_start = int(c_i * box_w / 4)
                c_end = int((c_i + 1) * box_w / 4)
                cell = box_ink[r_start:r_end, c_start:c_end]
                grid_feats.append(float(cell.sum()) / (ink_sum + 1e-6))

        h_runs = float((box_ink[:, :-1] * box_ink[:, 1:]).sum() / (box_ink.sum() + 1e-6))
        v_runs = float((box_ink[:-1, :] * box_ink[1:, :]).sum() / (box_ink.sum() + 1e-6))
        ans_box_start = int(0.75 * box_h)
        ans_box_ink = float(box_ink[ans_box_start:, :].sum()) / (ink_sum + 1e-6)
        ar = float(box_w) / float(box_h)
        feats = [ink_density, box_coverage, box_ink_density, ar, h_runs, v_runs, ans_box_ink] + grid_feats
        while len(feats) < 24:
            feats.append(0.0)
        return np.array(feats[:24], dtype=np.float32)
    except Exception:
        return np.zeros(24, dtype=np.float32)


def main():
    print("=" * 80)
    print("3,500 Unlabelled Trace Feature Extraction & Distillation Pipeline")
    print("=" * 80)

    meta_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_3500_metadata.csv"))
    img_paths = [DATA_DIR / "images" / "unlabelled" / str(f) for f in meta_df["image_file"].values]
    n_samples = len(img_paths)
    print(f"Dataset Size: {n_samples} images across {meta_df['question_id'].nunique()} test questions\n", flush=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Step 1: ConvNeXt-448 Letterbox Extraction (768-d, batch 16)
    # -----------------------------------------------------------------------
    print("[1/5] Extracting ConvNeXt-448 Letterbox Features (batch_size=16)...", flush=True)
    conv_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    conv_model.classifier[2] = nn.Identity()
    conv_model = conv_model.to(device).eval()

    conv_feats_list = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_samples, 16):
            b_paths = img_paths[i : i + 16]
            b_tensors = []
            for p in b_paths:
                try:
                    with Image.open(p) as raw:
                        padded = letterbox_image(raw.convert("RGB"), target_size=448)
                        arr = (np.array(padded, dtype=np.float32) / 255.0 - mean) / std
                        b_tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
                except Exception:
                    b_tensors.append(torch.zeros(3, 448, 448, dtype=torch.float32))

            batch = torch.stack(b_tensors).to(device)
            out = conv_model(batch)
            conv_feats_list.append(out.cpu().numpy())

            if (i // 16 + 1) % 25 == 0 or (i + 16) >= n_samples:
                done = min(i + 16, n_samples)
                elapsed = time.time() - t0
                fps = done / elapsed
                print(f"  Processed {done}/{n_samples} in {elapsed:.1f}s ({fps:.1f} img/s)...", flush=True)

    conv_feats = np.vstack(conv_feats_list).astype(np.float32)
    del conv_model, conv_feats_list
    torch.cuda.empty_cache()
    gc.collect()
    print(f"ConvNeXt-448 extraction complete: {conv_feats.shape} in {time.time() - t0:.1f}s\n", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: Swin-T @ 224x224 Extraction (768-d, batch 32)
    # -----------------------------------------------------------------------
    print("[2/5] Extracting Swin-T 224x224 Features (batch_size=32)...", flush=True)
    swin_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    setattr(swin_model, "head", nn.Identity())
    swin_model = swin_model.to(device).eval()

    swin_feats_list = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_samples, 32):
            b_paths = img_paths[i : i + 32]
            b_tensors = []
            for p in b_paths:
                try:
                    with Image.open(p) as raw:
                        resized = raw.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                        arr = (np.array(resized, dtype=np.float32) / 255.0 - mean) / std
                        b_tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
                except Exception:
                    b_tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))

            batch = torch.stack(b_tensors).to(device)
            out = swin_model(batch)
            swin_feats_list.append(out.cpu().numpy())

            if (i // 32 + 1) % 25 == 0 or (i + 32) >= n_samples:
                done = min(i + 32, n_samples)
                elapsed = time.time() - t0
                fps = done / elapsed
                print(f"  Processed {done}/{n_samples} in {elapsed:.1f}s ({fps:.1f} img/s)...", flush=True)

    swin_feats = np.vstack(swin_feats_list).astype(np.float32)
    del swin_model, swin_feats_list
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Swin-T extraction complete: {swin_feats.shape} in {time.time() - t0:.1f}s\n", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Fast Vectorized Strokes (24-d)
    # -----------------------------------------------------------------------
    print("[3/5] Extracting Fast Perceptual Strokes (24-d)...", flush=True)
    t0 = time.time()
    strokes_list = []
    for p in img_paths:
        try:
            with Image.open(p) as raw:
                strokes_list.append(extract_stroke_fast(raw))
        except Exception:
            strokes_list.append(np.zeros(24, dtype=np.float32))

    strk_feats = np.vstack(strokes_list).astype(np.float32)
    print(f"Strokes extraction complete: {strk_feats.shape} in {time.time() - t0:.1f}s\n", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: MiniLM Semantic Text Embeddings (384-d, 282 Unique Questions)
    # -----------------------------------------------------------------------
    print("[4/5] Extracting MiniLM Semantic Text Embeddings...", flush=True)
    t0 = time.time()
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_name)
    lm_model = cast(Any, AutoModel).from_pretrained(model_name).to(device).eval()

    unique_q = cast(pd.DataFrame, meta_df.drop_duplicates(subset=["question_id"])).reset_index(drop=True)
    q_texts = [
        f"Topik: {r['topic_name']}. Kelas: {r['grade']}. Pertanyaan: {r['question_text']}. Jawaban Benar: {r['correct_answer']}"
        for _, r in unique_q.iterrows()
    ]
    q_emb_list = []
    with torch.no_grad():
        for i in range(0, len(q_texts), 32):
            b_texts = q_texts[i : i + 32]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = lm_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            sum_emb = torch.sum(out[0] * mask, dim=1)
            mean_emb = sum_emb / torch.clamp(mask.sum(dim=1), min=1e-9)
            norm_emb = F.normalize(mean_emb, p=2, dim=1)
            q_emb_list.append(norm_emb.cpu().numpy())

    del lm_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    q_embs = np.vstack(q_emb_list).astype(np.float32)
    q_to_emb = {qid: q_embs[idx] for idx, qid in enumerate(unique_q["question_id"].values)}
    text_feats = np.vstack([q_to_emb[qid] for qid in meta_df["question_id"].values]).astype(np.float32)
    print(f"Text embeddings mapped: {text_feats.shape} in {time.time() - t0:.1f}s\n", flush=True)

    # -----------------------------------------------------------------------
    # Step 5: Assemble 1,944-d Representation & Distill Soft Targets
    # -----------------------------------------------------------------------
    print("[5/5] Assembling 1,944-d Matrix & Distilling Soft Targets...", flush=True)
    # Canonical order: Swin (768), ConvNeXt-448 (768), Text (384), Strokes (24) = 1,944
    features_1944 = np.hstack([swin_feats, conv_feats, text_feats, strk_feats]).astype(np.float32)
    print(f"Final 1,944-d Matrix Shape: {features_1944.shape}, NaNs: {np.isnan(features_1944).sum()}", flush=True)
    np.save(ART_DIR / "pseudo_3500_features_1944.npy", features_1944)
    print("Saved: artifacts/pseudo_3500_features_1944.npy", flush=True)

    # Distill soft targets using 0.54616 Champion Teacher (Adaptive Prior + Text Diagnoser)
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))

    # Fit Text Diagnoser
    text_model = TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=5e-5)
    text_model.fit(train_df)
    raw_text_probs = text_model.predict_proba(unique_q)

    aligned_text_probs = np.zeros((len(unique_q), len(classes)), dtype=np.float32)
    for m_idx, c in enumerate(text_model.classes_):
        if c in classes:
            aligned_text_probs[:, classes.index(c)] = raw_text_probs[:, m_idx]

    # Fit Adaptive Question Prior
    prior_engine = AdaptiveQuestionPriorEngine(
        classes=classes, m_q_base=10.0, m_tg=10.0, adaptive_nq=True, adaptive_power=0.70,
        use_semantic_backoff=False
    )
    prior_engine.fit(train_df)
    q_prior_probs = prior_engine.predict_proba(unique_q)

    # Weighted Teacher Distribution (0.6728 Prior + 0.3272 Text)
    q_dist = 0.6728 * q_prior_probs + 0.3272 * aligned_text_probs
    q_to_dist = {qid: q_dist[idx] for idx, qid in enumerate(unique_q["question_id"].values)}

    soft_targets = np.vstack([q_to_dist[qid] for qid in meta_df["question_id"].values]).astype(np.float32)
    soft_targets = np.clip(soft_targets, 1e-7, 1.0)
    soft_targets = soft_targets / soft_targets.sum(axis=1, keepdims=True)
    print(f"Soft Targets Shape: {soft_targets.shape}, Row sum min={soft_targets.sum(axis=1).min():.5f}, max={soft_targets.sum(axis=1).max():.5f}", flush=True)

    np.save(ART_DIR / "pseudo_3500_soft_targets_76.npy", soft_targets)
    print("Saved: artifacts/pseudo_3500_soft_targets_76.npy", flush=True)

    print("\n" + "=" * 80)
    print("ALL 3,500-TRACE FEATURES & SOFT TARGETS GENERATED SUCCESSFULLY!")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

