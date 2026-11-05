"""Fast In-Memory Feature Extraction & Training for 1,465 Harvested Traces.

Pillar 4 SOTA Pipeline:
1. High-Throughput Preloading: Decodes each unlabelled image once into RAM.
2. GPU Batched Inference: ConvNeXt (768) + Swin-T (768) + TrOCR DeiT (384) + MiniLM (384) + Strokes (24).
3. Expanded Training: 1,500 original + 1,465 pseudo = 2,965 total training traces.
4. Untainted 5-Fold Validation: Evaluates strictly on original splits.
5. Submission: submissions/blend_m15_expanded_1465_submission.csv.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, VisionEncoderDecoderModel

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Stroke Extractor from PIL Image (In-Memory)
# ---------------------------------------------------------------------------
def extract_stroke_from_pil(img_gray_512: Image.Image) -> np.ndarray:
    try:
        arr = np.array(img_gray_512, dtype=np.uint8)
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
        ans_box_ink = float(box_ink[ans_box_start:, :].sum() / (ink_sum + 1e-6))
        ar = float(box_w) / float(box_h)
        feats = [ink_density, box_coverage, box_ink_density, ar, h_runs, v_runs, ans_box_ink] + grid_feats
        # Pad to exactly 24 dimensions
        while len(feats) < 24:
            feats.append(0.0)
        return np.array(feats[:24], dtype=np.float32)
    except Exception:
        return np.zeros(24, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Deep ArcFace Model
# ---------------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = float(np.cos(m))
        self.sin_m = float(np.sin(m))
        self.th = float(np.cos(np.pi - m))
        self.mm = float(np.sin(np.pi - m) * m)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class MultimodalDeepArcFace(nn.Module):
    def __init__(self, in_features: int = 2328, hidden1: int = 768, hidden2: int = 384, n_classes: int = 76):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = ArcMarginProduct(hidden2, n_classes, s=30.0, m=0.20)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat, label)


def compute_map3(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    top3 = np.argsort(-y_probs.astype(np.float64), axis=1)[:, :3]
    scores = [1.0 if t == p[0] else 0.5 if t == p[1] else 1.0 / 3.0 if t == p[2] else 0.0 for t, p in zip(y_true, top3)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 3. Fast In-Memory Feature Extraction for 1,465 Traces
# ---------------------------------------------------------------------------
def extract_pseudo_features_fast(pseudo_df: pd.DataFrame, device: torch.device) -> np.ndarray:
    cache_path = ART_DIR / "pseudo_1465_features_2328.npy"
    if cache_path.exists():
        feats = np.load(cache_path)
        if feats.shape == (len(pseudo_df), 2328):
            print(f"[FeatureExtractor] Loading cached features: {cache_path} ({feats.shape})")
            return feats

    print(f"\n[FeatureExtractor] Starting fast in-memory extraction for {len(pseudo_df)} traces on {device}...")
    t_start = time.time()

    img_dir = DATA_DIR / "images" / "unlabelled"
    img_files = pseudo_df["image_file"].tolist()

    # Preload transforms
    tf_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("  Preloading images and stroke features into RAM (one-pass decode)...")
    t_load = time.time()
    tensors_224 = []
    tensors_384 = []
    strokes_list = []

    for idx, fname in enumerate(img_files):
        p = img_dir / fname
        try:
            with Image.open(p) as img_raw:
                img_rgb = img_raw.convert("RGB")
                img_gray = img_raw.convert("L").resize((512, 512), Image.BILINEAR)

                # 224x224 tensor for ConvNeXt & Swin
                img_224 = img_rgb.resize((224, 224), Image.BILINEAR)
                t_224 = tf_norm(transforms.functional.to_tensor(img_224))
                tensors_224.append(t_224)

                # 384x384 tensor for TrOCR
                img_384 = img_rgb.resize((384, 384), Image.BILINEAR)
                t_384 = tf_norm(transforms.functional.to_tensor(img_384))
                tensors_384.append(t_384)

                # Strokes
                strokes_list.append(extract_stroke_from_pil(img_gray))
        except Exception:
            tensors_224.append(torch.zeros(3, 224, 224))
            tensors_384.append(torch.zeros(3, 384, 384))
            strokes_list.append(np.zeros(24, dtype=np.float32))

        if (idx + 1) % 300 == 0 or (idx + 1) == len(img_files):
            print(f"    Decoded {idx + 1}/{len(img_files)} images in {time.time() - t_load:.1f}s...")

    batch_224 = torch.stack(tensors_224)
    batch_384 = torch.stack(tensors_384)
    strk_feats = np.vstack(strokes_list).astype(np.float32)
    print(f"  In-memory decoding completed in {time.time() - t_load:.1f}s!")

    # 1. ConvNeXt-Tiny (768)
    print("  1/4 Extracting ConvNeXt-Tiny (768-d)...")
    conv_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    conv_model.classifier[2] = nn.Identity()
    conv_model = conv_model.to(device).eval()

    conv_loader = DataLoader(TensorDataset(batch_224), batch_size=64, shuffle=False)
    conv_list = []
    with torch.no_grad():
        for (bx,) in conv_loader:
            conv_list.append(conv_model(bx.to(device)).cpu().numpy())
    conv_feats = np.vstack(conv_list).astype(np.float32)
    del conv_model
    torch.cuda.empty_cache()

    # 2. Swin-T (768)
    print("  2/4 Extracting Swin-T (768-d)...")
    swin_model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    setattr(swin_model, "head", nn.Identity())
    swin_model = swin_model.to(device).eval()

    swin_loader = DataLoader(TensorDataset(batch_224), batch_size=64, shuffle=False)
    swin_list = []
    with torch.no_grad():
        for (bx,) in swin_loader:
            swin_list.append(swin_model(bx.to(device)).cpu().numpy())
    swin_feats = np.vstack(swin_list).astype(np.float32)
    del swin_model
    torch.cuda.empty_cache()

    # 3. TrOCR DeiT Visual Encoder (384)
    print("  3/4 Extracting TrOCR DeiT Visual Features (384-d)...")
    trocr_model = cast(Any, VisionEncoderDecoderModel).from_pretrained("microsoft/trocr-small-handwritten")
    trocr_enc = trocr_model.encoder.to(device).eval()

    trocr_loader = DataLoader(TensorDataset(batch_384), batch_size=32, shuffle=False)
    trocr_list = []
    with torch.no_grad():
        for (bx,) in trocr_loader:
            out = trocr_enc(bx.to(device).half())
            cls_token = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            trocr_list.append(cls_token)
    trocr_feats = np.vstack(trocr_list).astype(np.float32)
    del trocr_model, trocr_enc
    torch.cuda.empty_cache()

    # 4. MiniLM Semantic Text Embeddings (384)
    print("  4/4 Extracting MiniLM Semantic Text Embeddings (384-d)...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(model_name)
    lm_model = cast(Any, AutoModel).from_pretrained(model_name).to(device).eval()

    texts = [
        f"Topik: {r['topic_name']}. Kelas: {r['grade']}. Pertanyaan: {r['question_text']}. Jawaban Benar: {r['correct_answer']}"
        for _, r in pseudo_df.iterrows()
    ]
    lm_list = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b_texts = texts[i : i + 64]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = lm_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
            sum_emb = torch.sum(out[0] * mask, dim=1)
            mean_emb = sum_emb / torch.clamp(mask.sum(dim=1), min=1e-9)
            norm_emb = F.normalize(mean_emb, p=2, dim=1)
            lm_list.append(norm_emb.cpu().numpy())
    lm_feats = np.vstack(lm_list).astype(np.float32)
    del lm_model, tokenizer
    torch.cuda.empty_cache()

    # Concatenate all 5 modalities
    pseudo_2328 = np.hstack([swin_feats, conv_feats, lm_feats, trocr_feats, strk_feats]).astype(np.float32)
    print(f"Extracted feature shape: {pseudo_2328.shape} in {time.time() - t_start:.1f}s!")

    np.save(cache_path, pseudo_2328)
    print(f"Saved: {cache_path}")
    return pseudo_2328


# ---------------------------------------------------------------------------
# 4. End-to-End Training & Ensembling Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 75)
    print("[Pipeline] Expanded Deep ArcFace Training (1,465 Harvested Traces -> 2,965 Dataset)")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    with open(ART_DIR / "taxonomy_classes.json") as f:
        tax_classes: List[str] = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(tax_classes)}
    n_classes = len(tax_classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    y_train_orig = np.array([class_to_idx[r] for r in train_df["root_code"]], dtype=np.int64)

    with open(DATA_DIR / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    # Load Original Features (2,328-d)
    swin_tr = np.load(ART_DIR / "train_swin_features.npy")
    conv_tr = np.load(ART_DIR / "train_convnext_features.npy")
    text_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    trocr_tr = np.load(ART_DIR / "train_trocr_features.npy")
    strk_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")

    swin_te = np.load(ART_DIR / "test_swin_features.npy")
    conv_te = np.load(ART_DIR / "test_convnext_features.npy")
    text_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    trocr_te = np.load(ART_DIR / "test_trocr_features.npy")
    strk_te = np.load(ART_DIR / "test_perceptual_strokes.npy")

    X_train_orig = np.hstack([swin_tr, conv_tr, text_tr, trocr_tr, strk_tr]).astype(np.float32)
    X_test = np.hstack([swin_te, conv_te, text_te, trocr_te, strk_te]).astype(np.float32)

    # Load 1,465 Harvested Traces
    pseudo_csv = ART_DIR / "pseudo_labeled_rare_traces.csv"
    pseudo_df = cast(pd.DataFrame, pd.read_csv(pseudo_csv))
    print(f"Loaded {len(pseudo_df)} pseudo-labeled traces from {pseudo_csv.name}")

    X_pseudo = extract_pseudo_features_fast(pseudo_df, device)
    y_pseudo = np.array([class_to_idx[r] for r in pseudo_df["root_code"]], dtype=np.int64)

    print(f"\nExpanded Dataset: {len(X_train_orig)} original + {len(X_pseudo)} pseudo = {len(X_train_orig) + len(X_pseudo)} total traces!")

    # Baseline M15 Probabilities
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    base_m15_map3 = compute_map3(y_train_orig, oof_m15)
    print(f"Active M15 Baseline OOF MAP@3: {base_m15_map3:.5f}")

    # 5-Fold Stratified Training with Untainted Validation
    oof_expanded = np.zeros((len(y_train_orig), n_classes), dtype=np.float32)
    test_fold_preds = np.zeros((len(folds), len(X_test), n_classes), dtype=np.float32)

    epochs = 25
    batch_size = 64
    lr = 1e-3

    print(f"\nTraining Expanded ArcFace across {len(folds)} folds ({epochs} epochs/fold, lr={lr})...")
    t_train_start = time.time()

    for fold_num, fold in enumerate(folds, 1):
        tr_orig_idx = np.array(fold["train_indices"])
        val_orig_idx = np.array(fold["val_indices"])

        X_tr = np.vstack([X_train_orig[tr_orig_idx], X_pseudo])
        y_tr = np.concatenate([y_train_orig[tr_orig_idx], y_pseudo])

        X_val = X_train_orig[val_orig_idx]
        y_val = y_train_orig[val_orig_idx]

        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_tr)
        X_val_norm = scaler.transform(X_val)
        X_te_norm = scaler.transform(X_test)

        tr_ds = TensorDataset(torch.from_numpy(X_tr_norm).float(), torch.from_numpy(y_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val_norm).float())
        te_ds = TensorDataset(torch.from_numpy(X_te_norm).float())

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        model = MultimodalDeepArcFace(in_features=2328, hidden1=768, hidden2=384, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        for epoch in range(epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx, by)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                logits = model(bx, label=None)
                val_preds.append(F.softmax(logits, dim=-1).cpu().numpy())
        fold_val_probs = np.vstack(val_preds)
        oof_expanded[val_orig_idx] = fold_val_probs

        te_preds = []
        with torch.no_grad():
            for (bx,) in te_loader:
                bx = bx.to(device)
                logits = model(bx, label=None)
                te_preds.append(F.softmax(logits, dim=-1).cpu().numpy())
        test_fold_preds[fold_num - 1] = np.vstack(te_preds)

        fold_score = compute_map3(y_val, fold_val_probs)
        print(f"  Fold {fold_num}/{len(folds)} Val MAP@3: {fold_score:.5f}")

    total_arc_map3 = compute_map3(y_train_orig, oof_expanded)
    test_arc_probs = np.mean(test_fold_preds, axis=0)
    print(f"\nExpanded ArcFace Total Untainted OOF MAP@3: {total_arc_map3:.5f} ({time.time() - t_train_start:.1f}s)")

    # Save artifacts
    np.save(ART_DIR / "oof_arcface_expanded_1465.npy", oof_expanded)
    np.save(ART_DIR / "test_arcface_expanded_1465.npy", test_arc_probs)
    print(f"Saved: {ART_DIR / 'oof_arcface_expanded_1465.npy'}")
    print(f"Saved: {ART_DIR / 'test_arcface_expanded_1465.npy'}")

    # Ensembling with M15 (90/10 Champion Formula)
    final_test_blend = 0.90 * test_m15 + 0.10 * test_arc_probs

    top1_classes = [tax_classes[idx] for idx in np.argmax(final_test_blend, axis=1)]
    none_count = sum(1 for c in top1_classes if c == "NONE")
    none_rate = none_count / len(top1_classes)
    print(f"\nTest Top-1 NONE: {none_count} / {len(top1_classes)} ({none_rate:.2%})")
    assert 0.68 <= none_rate <= 0.73, f"NONE invariant violated: {none_rate:.2%}"
    print("NONE Preservation Invariant SATISFIED!")

    # Compare differences vs Champion (0.54483)
    champ_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_expanded_pseudo_submission.csv"))
    top3_indices = np.argsort(-final_test_blend.astype(np.float64), axis=1)[:, :3]
    pred_strings = [" ".join(tax_classes[idx] for idx in row) for row in top3_indices]

    diff_vs_champ = sum(1 for a, b in zip(pred_strings, champ_sub["root_code"]) if a != b)
    print(f"Differences vs Active Champion (0.54483): {diff_vs_champ} / {len(test_df)} ({diff_vs_champ / len(test_df):.2%})")

    # Generate Submission
    out_file = SUB_DIR / "blend_m15_expanded_1465_submission.csv"
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": pred_strings})
    sub_df.to_csv(out_file, index=False)
    print(f"\nGenerated: {out_file} ({len(sub_df)} rows)")

    import shutil
    dl_path = Path("C:/Users/hp/Downloads/blend_m15_expanded_1465_submission.csv")
    shutil.copy(out_file, dl_path)
    print(f"Copied to: {dl_path}")


if __name__ == "__main__":
    run_pipeline()

