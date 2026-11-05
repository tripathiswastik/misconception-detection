"""Extract 1,944-d Multimodal Features & Distill 3,500 Unlabelled Traces.

SDLC Protocol:
1. Loads 3,500 curated traces from artifacts/pseudo_3500_metadata.csv.
2. Extracts 1,944-d representation:
   - ConvNeXt-448 Uniform Square Letterbox (768-d)
   - Swin-T @ 224x224 (768-d)
   - Multilingual MiniLM Question Text Embeddings (384-d)
   - Perceptual Strokes (24-d)
3. Generates teacher soft targets using 0.54616 Champion teacher (Adaptive Prior + Text Diagnoser).
4. Outputs:
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
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Device for 3,500 Extraction: {device}", flush=True)


# ---------------------------------------------------------------------------
# 1. Image Letterboxing & Batch Dataset
# ---------------------------------------------------------------------------
def letterbox_image(img: Image.Image, target_size: int = 448, fill_color=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    padded.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded


def extract_stroke(img_gray: Image.Image) -> np.ndarray:
    try:
        arr = np.array(img_gray, dtype=np.uint8)
        ink = (arr < 150).astype(np.float32)
        ink_sum = float(ink.sum())
        if ink_sum < 10:
            return np.zeros(24, dtype=np.float32)

        ink_density = ink_sum / (512.0 * 512.0)
        rows = np.where(ink.any(axis=1))[0]
        cols = np.where(ink.any(axis=0))[0]
        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        box_h = max(1, y1 - y0)
        box_w = max(1, x1 - x0)
        box_coverage = (box_h * box_w) / (512.0 * 512.0)
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


class StreamingUnlabelledDataset(Dataset):
    def __init__(self, img_paths: List[Path]):
        self.img_paths = img_paths
        self.tf_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        p = self.img_paths[index]
        try:
            with Image.open(p) as raw:
                rgb = raw.convert("RGB")
                gray = raw.convert("L").resize((512, 512), Image.BILINEAR)

                # 448x448 letterbox
                lb_448 = letterbox_image(rgb, target_size=448)
                t_448 = self.tf_norm(TF.to_tensor(lb_448))

                # 224x224 for Swin
                r_224 = rgb.resize((224, 224), Image.BILINEAR)
                t_224 = self.tf_norm(TF.to_tensor(r_224))

                strk = extract_stroke(gray)
                return t_448, t_224, torch.from_numpy(strk)
        except Exception:
            return torch.zeros(3, 448, 448), torch.zeros(3, 224, 224), torch.zeros(24)


def main():
    print("=" * 80)
    print("Extracting 1,944-d Multimodal Features for 3,500 Curated Traces...")
    print("=" * 80)

    meta_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_3500_metadata.csv"))
    img_paths = [DATA_DIR / "images" / "unlabelled" / str(f) for f in meta_df["image_file"].values]
    n_samples = len(img_paths)
    print(f"Dataset Size: {n_samples} images across {meta_df['question_id'].nunique()} test questions")

    # 1. Streaming Extraction for ConvNeXt-448, Swin-T, and Strokes
    ds = StreamingUnlabelledDataset(img_paths)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    print("\n[1/4] Loading ConvNeXt-Tiny & Swin-T Backbones into VRAM...")
    conv_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    conv_model.classifier[2] = nn.Identity()
    conv_model = conv_model.to(device).eval()

    swin_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    setattr(swin_model, "head", nn.Identity())
    swin_model = swin_model.to(device).eval()

    conv_feats_list = []
    swin_feats_list = []
    strk_feats_list = []

    t0 = time.time()
    print("  Extracting visual & stroke features in batched streaming mode...")
    with torch.no_grad():
        for i, (b_448, b_224, b_strk) in enumerate(loader):
            # ConvNeXt-448
            c_out = conv_model(b_448.to(device)).cpu().numpy()
            conv_feats_list.append(c_out)

            # Swin-T
            s_out = swin_model(b_224.to(device)).cpu().numpy()
            swin_feats_list.append(s_out)

            # Strokes
            strk_feats_list.append(b_strk.numpy())

            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                done = min((i + 1) * 32, n_samples)
                elapsed = time.time() - t0
                fps = done / elapsed
                print(f"    [{done:4d}/{n_samples}] {elapsed:.1f}s elapsed ({fps:.1f} img/s)...", flush=True)

    del conv_model, swin_model
    torch.cuda.empty_cache()

    conv_feats = np.vstack(conv_feats_list).astype(np.float32)
    swin_feats = np.vstack(swin_feats_list).astype(np.float32)
    strk_feats = np.vstack(strk_feats_list).astype(np.float32)
    print(f"Visual extraction complete in {time.time() - t0:.1f}s: ConvNeXt {conv_feats.shape}, Swin {swin_feats.shape}, Strokes {strk_feats.shape}")

    # 2. Text Embeddings via MiniLM (computed across unique questions)
    print("\n[2/4] Extracting MiniLM Semantic Text Embeddings...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_name)
    lm_model = cast(Any, AutoModel).from_pretrained(model_name).to(device).eval()

    # Unique questions mapping
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

    q_embs = np.vstack(q_emb_list).astype(np.float32)
    q_to_emb = {qid: q_embs[idx] for idx, qid in enumerate(unique_q["question_id"].values)}
    text_feats = np.vstack([q_to_emb[qid] for qid in meta_df["question_id"].values]).astype(np.float32)
    print(f"Text embeddings mapped: {text_feats.shape}")

    # 3. Concatenate to 1,944-d
    print("\n[3/4] Assembling Unified 1,944-d Representation...")
    # Order: Swin (768), ConvNeXt-448 (768), Text (384), Strokes (24) = 1944
    features_1944 = np.hstack([swin_feats, conv_feats, text_feats, strk_feats]).astype(np.float32)
    print(f"Features 1,944 Shape: {features_1944.shape}, dtype={features_1944.dtype}, NaNs={np.isnan(features_1944).sum()}")
    np.save(ART_DIR / "pseudo_3500_features_1944.npy", features_1944)
    print("Saved: artifacts/pseudo_3500_features_1944.npy")

    # 4. Generate Soft Targets using 0.54616 Champion Teacher
    print("\n[4/4] Distilling Soft Probability Targets from Champion Teacher...")
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

    # Weighted Teacher Distribution (0.67 Prior + 0.33 Text)
    q_dist = 0.6728 * q_prior_probs + 0.3272 * aligned_text_probs
    q_to_dist = {qid: q_dist[idx] for idx, qid in enumerate(unique_q["question_id"].values)}

    soft_targets = np.vstack([q_to_dist[qid] for qid in meta_df["question_id"].values]).astype(np.float32)
    # Ensure exact probability simplex: sum to 1.0, non-negative
    soft_targets = np.clip(soft_targets, 1e-7, 1.0)
    soft_targets = soft_targets / soft_targets.sum(axis=1, keepdims=True)
    print(f"Soft Targets Shape: {soft_targets.shape}, Row sum min={soft_targets.sum(axis=1).min():.5f}, max={soft_targets.sum(axis=1).max():.5f}")

    np.save(ART_DIR / "pseudo_3500_soft_targets_76.npy", soft_targets)
    print("Saved: artifacts/pseudo_3500_soft_targets_76.npy")

    print("\n" + "=" * 80)
    print("3,500-TRACE EXTRACTION & SOFT DISTILLATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

