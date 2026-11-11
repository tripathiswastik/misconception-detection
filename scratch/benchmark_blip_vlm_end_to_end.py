"""End-to-End Benchmark of Open-Weight VLM (Salesforce/blip-vqa-base) on 1,500 Labeled Images.

Protocol:
1. Feature Extraction:
   - Vision-Language cross-modal representation: h_i in R^768 from BLIP cross-attention text encoder
     conditioned on the raw student math working image and question context.
   - Caches: artifacts/train_blip_vlm_features_768.npy (1500, 768).
2. Canonical 5-Fold Question-Grouped Evaluation (data/splits.json):
   - Strict fold isolation: Train classification head strictly on 1,200 training traces per fold.
   - Predict continuous 76-class probabilities on 300 held-out validation traces.
   - Fit and apply empirical domain constraint masking (DomainConstraintEngine) in-fold.
3. Metric Benchmarks:
   - Standalone VLM: Composite MAP@3 (0.957 Seen + 0.043 Unseen), Seen, Unseen, Rare Recall@10, False NONE.
   - Integration with S550 Baseline (0.54214 Composite MAP@3, 104 / 506 False NONE).
4. Gate Decision:
   - Must improve held-out Composite MAP@3 (> 0.54214) AND preserve NONE calibration (<= 104 false NONE).
   - If passes: Propose candidate for evaluation.
   - If fails: Formally report findings and maintain S550 as the immutable champion.
"""

from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForQuestionAnswering, BlipProcessor

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRATCH_DIR = PROJECT_ROOT / "scratch"
if str(SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_DIR))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
IMG_DIR = DATA_DIR / "images"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.metrics.map3 import calculate_map3
from audit_rare_class_s550_oof_error_breakdown import S550AuditEngine


class StudentWorkingVLMDataset(Dataset):
    """PyTorch Dataset for raw image + question text loading."""
    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_name = str(row["image_file"])
        img_path = self.img_dir / img_name

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        q_text = str(row.get("question_text", ""))
        key = str(row.get("correct_answer", ""))
        topic = str(row.get("topic_name", ""))
        grade = str(row.get("grade", ""))

        prompt = f"Question: {q_text}. Correct answer: {key}. Topic: {topic}."
        return {
            "image": image,
            "prompt": prompt,
            "trace_id": str(row["trace_id"])
        }


def extract_blip_multimodal_features(
    df: pd.DataFrame,
    img_dir: Path,
    device: torch.device,
    cache_path: Path
) -> np.ndarray:
    """Extract fused 768-dim multimodal cross-modal embeddings using BLIP VQA."""
    if cache_path.exists():
        print(f"Loading cached BLIP VLM multimodal features from {cache_path}...")
        features = np.load(cache_path)
        if features.shape == (len(df), 768):
            return features
        print("Cache shape mismatch, re-extracting...")

    n_samples = len(df)
    ckpt_path = cache_path.with_name(cache_path.stem + "_checkpoint.npy")
    if ckpt_path.exists():
        print(f"Loading partial checkpoint from {ckpt_path}...")
        features_raw = np.load(ckpt_path)
        if features_raw.shape != (n_samples, 768):
            features_raw = np.zeros((n_samples, 768), dtype=np.float32)
    else:
        features_raw = np.zeros((n_samples, 768), dtype=np.float32)

    computed_mask = np.linalg.norm(features_raw, axis=1) > 1e-6
    n_already_done = int(np.sum(computed_mask))
    if n_already_done == n_samples:
        print("All samples already computed in checkpoint! Normalizing and saving...")
        norms = np.linalg.norm(features_raw, axis=1, keepdims=True)
        normalized = (features_raw / np.maximum(norms, 1e-12)).astype(np.float32)
        np.save(cache_path, normalized)
        return normalized

    print(f"\n[VLM Feature Extraction] Loading Salesforce/blip-vqa-base (Already computed: {n_already_done}/{n_samples})...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    model.eval()

    dataset = StudentWorkingVLMDataset(df, img_dir)
    batch_size = 16

    print(f"Extracting VLM cross-modal representations for remaining {n_samples - n_already_done} student workings (batch_size={batch_size})...")
    start_t = time.time()

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            if np.all(computed_mask[start_idx:end_idx]):
                continue

            batch_items = [dataset[i] for i in range(start_idx, end_idx)]
            images = [item["image"] for item in batch_items]
            prompts = [item["prompt"] for item in batch_items]

            inputs = processor(
                images=images,
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            # Vision encoder
            vision_outputs = model.vision_model(
                pixel_values=inputs.pixel_values
            )
            image_embeds = vision_outputs[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

            # Cross-attention text encoder (multimodal fusion)
            text_outputs = model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True
            )

            # Extract [CLS] multimodal token representation (768-dim)
            cls_repr = text_outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            features_raw[start_idx:end_idx] = cls_repr
            computed_mask[start_idx:end_idx] = True

            # Save checkpoint every 5 batches or at end
            if (start_idx // batch_size) % 5 == 0 or end_idx == n_samples:
                np.save(ckpt_path, features_raw)
                elapsed = time.time() - start_t
                n_done_now = int(np.sum(computed_mask))
                n_processed_this_run = n_done_now - n_already_done
                rate = n_processed_this_run / max(1.0, elapsed)
                rem_samples = n_samples - n_done_now
                eta_min = (rem_samples / max(0.1, rate)) / 60.0
                print(f"  Processed {n_done_now}/{n_samples} ({rate:.2f} s/s, ETA: {eta_min:.1f} min)...", flush=True)

    # L2 normalize feature vectors
    norms = np.linalg.norm(features_raw, axis=1, keepdims=True)
    features_matrix = (features_raw / np.maximum(norms, 1e-12)).astype(np.float32)

    np.save(cache_path, features_matrix)
    print(f"Saved BLIP multimodal features to {cache_path} (Shape: {features_matrix.shape})")
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except Exception:
            pass
    return features_matrix


class VLMClassificationHead(nn.Module):
    """Cosine classification head for VLM feature vectors."""
    def __init__(self, in_dim: int = 768, num_classes: int = 76):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)
        return self.scale * cosine


def train_vlm_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    num_classes: int = 76,
    epochs: int = 40,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Train classification head on VLM embeddings in-fold."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    head = VLMClassificationHead(in_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    n = len(X_train)
    batch_size = 128

    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bx = X_train_d[idx]
            by = y_train_d[idx]

            logits = head(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    with torch.no_grad():
        val_logits = head(X_val.to(device))
        val_probs = F.softmax(val_logits, dim=1).cpu().numpy().astype(np.float64)

    return val_probs


def main():
    print("=" * 115)
    print("OPEN-WEIGHT VLM END-TO-END BENCHMARK: SALESFORCE/BLIP-VQA-BASE")
    print("=" * 115)
    start_time = time.time()

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_true_int = np.array([class_to_idx.get(r, none_idx) for r in y_raw], dtype=np.int64)
    n_samples = len(train_df)

    class_counts = Counter(y_raw)
    rare_classes: Set[str] = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    rare_indices = [i for i in range(n_samples) if y_raw[i] in rare_classes]
    none_indices = [i for i in range(n_samples) if y_raw[i] == "NONE"]

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)["folds"]

    seen_mask = np.zeros(n_samples, dtype=bool)
    for fold in splits:
        tr_qids = set(train_df.iloc[fold["train_indices"]]["question_id"].astype(str))
        for i in fold["val_indices"]:
            if str(train_df.iloc[i]["question_id"]) in tr_qids:
                seen_mask[i] = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset: {n_samples} total traces | True NONE: {len(none_indices)} | Rare Classes: {len(rare_indices)}")

    # Step 1: Extract VLM Multimodal Features
    cache_path = ART_DIR / "train_blip_vlm_features_768.npy"
    vlm_features = extract_blip_multimodal_features(train_df, IMG_DIR, device, cache_path)

    # Step 2: 5-Fold Question-Grouped Cross-Validation
    print("\n[Step 2] Executing 5-Fold Question-Grouped Cross-Validation...")
    oof_p_vlm = np.zeros((n_samples, n_classes), dtype=np.float64)

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        X_tr = torch.from_numpy(vlm_features[tr_idx]).float()
        y_tr = torch.from_numpy(y_true_int[tr_idx]).long()
        X_val = torch.from_numpy(vlm_features[val_idx]).float()

        val_probs = train_vlm_head(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            num_classes=n_classes,
            epochs=40,
            lr=1e-2,
            weight_decay=1e-4,
            device=device
        )

        # Apply in-fold empirical domain constraint masking
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        v_topics = [str(t) for t in train_df.iloc[val_idx]["topic_name"].values]
        v_grades = [int(float(g)) for g in train_df.iloc[val_idx]["grade"].values]
        val_probs_cal = masker.apply_mask_batch(val_probs, classes, v_topics, v_grades, damping=0.05)

        oof_p_vlm[val_idx] = val_probs_cal

        gold_v = [y_raw[i] for i in val_idx]
        preds_v = [[classes[c] for c in row[:3]] for row in np.argsort(-val_probs_cal, axis=1)]
        fold_map3 = calculate_map3(gold_v, preds_v)
        print(f"  Fold {fold_idx}/5 Held-Out MAP@3: {fold_map3:.4f}")

    # Save VLM OOF probabilities
    np.save(ART_DIR / "oof_p_blip_vlm.npy", oof_p_vlm)

    # Step 3: Standalone VLM Evaluation
    print("\n" + "=" * 115)
    print("STANDALONE VLM BENCHMARK EVALUATION")
    print("=" * 115)

    preds_vlm_all = [[classes[c] for c in row[:3]] for row in np.argsort(-oof_p_vlm, axis=1)]
    vlm_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_vlm_all[i] for i in range(n_samples) if seen_mask[i]])
    vlm_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_vlm_all[i] for i in range(n_samples) if not seen_mask[i]])
    vlm_comp = 0.957 * vlm_seen + 0.043 * vlm_unseen

    vlm_r10 = sum(1 for i in rare_indices if y_true_int[i] in np.argsort(-oof_p_vlm[i])[:10])
    vlm_fn = sum(1 for i in none_indices if preds_vlm_all[i][0] != "NONE")

    print(f"  VLM Standalone Composite MAP@3: {vlm_comp:.5f} (Seen: {vlm_seen:.5f}, Unseen: {vlm_unseen:.5f})")
    print(f"  VLM Rare Recall@10:              {vlm_r10}/{len(rare_indices)} ({vlm_r10 / len(rare_indices):.1%})")
    print(f"  VLM False NONE Demotions:        {vlm_fn}/{len(none_indices)} ({vlm_fn / len(none_indices):.1%})")

    # Step 4: S550 Reference Benchmark & Integration Sweep
    print("\n" + "=" * 115)
    print("S550 CHAMPION REFERENCE & INTEGRATION GATE CHECK")
    print("=" * 115)
    engine = S550AuditEngine()
    oof_p_s550 = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}

    for fold_idx, fold in enumerate(splits, start=1):
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        p_val, q_dist = engine.generate_s550_probs(tr_idx, val_idx)
        oof_p_s550[val_idx] = p_val
        q_dists_per_fold[fold_idx] = q_dist

    preds_s550_all = [[] for _ in range(n_samples)]
    for fold_idx, fold in enumerate(splits, start=1):
        val_idx = np.array(fold["val_indices"], dtype=np.int64)
        preds_val = engine.rank_and_gate(oof_p_s550[val_idx], val_idx, q_dists_per_fold[fold_idx])
        for i, idx in enumerate(val_idx):
            preds_s550_all[idx] = preds_val[i]

    s550_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_s550_all[i] for i in range(n_samples) if seen_mask[i]])
    s550_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_s550_all[i] for i in range(n_samples) if not seen_mask[i]])
    s550_comp = 0.957 * s550_seen + 0.043 * s550_unseen
    s550_r10 = sum(1 for i in rare_indices if y_true_int[i] in np.argsort(-oof_p_s550[i])[:10])
    s550_fn = sum(1 for i in none_indices if preds_s550_all[i][0] != "NONE")

    print(f"  S550 Champion Baseline:")
    print(f"    Composite MAP@3: {s550_comp:.5f} | Rare R@10: {s550_r10}/{len(rare_indices)} | False NONE: {s550_fn}/{len(none_indices)}")

    # Integration Blends with S550
    integration_results = []
    print("\n--- Testing VLM Blends with S550 under Precision Gating ---")
    for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        p_blend = (1.0 - w) * oof_p_s550 + w * oof_p_vlm
        p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)

        preds_blend = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_blend[idx] = preds_val[i]

        b_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_blend[i] for i in range(n_samples) if seen_mask[i]])
        b_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_blend[i] for i in range(n_samples) if not seen_mask[i]])
        b_comp = 0.957 * b_seen + 0.043 * b_unseen
        b_r10 = sum(1 for i in rare_indices if y_true_int[i] in np.argsort(-p_blend[i])[:10])
        b_fn = sum(1 for i in none_indices if preds_blend[i][0] != "NONE")

        gate_pass = bool(b_comp > s550_comp and b_fn <= s550_fn)
        item = {
            "weight": w,
            "composite_map3": round(b_comp, 5),
            "delta_map3": round(b_comp - s550_comp, 5),
            "rare_r10": b_r10,
            "false_none": b_fn,
            "delta_fn": b_fn - s550_fn,
            "gate_pass": gate_pass
        }
        integration_results.append(item)
        print(f"  w={w:0.2f} | Comp MAP@3: {b_comp:.5f} (Delta: {b_comp - s550_comp:+.5f}) | "
              f"Rare R@10: {b_r10} | False NONE: {b_fn} (Delta: {b_fn - s550_fn:+d}) | Gate: {'PASS' if gate_pass else 'FAIL'}")

    # Strict NONE-Protected Blend
    print("\n--- Testing Strict NONE-Protected Blend (Preserving P(NONE) exactly) ---")
    p_s550_none = oof_p_s550[:, none_idx : none_idx + 1]
    p_s550_misc = np.delete(oof_p_s550, none_idx, axis=1)
    sum_s550_misc = p_s550_misc.sum(axis=1, keepdims=True)
    p_s550_misc_norm = np.where(sum_s550_misc > 1e-12, p_s550_misc / sum_s550_misc, 1.0 / (n_classes - 1))

    p_vlm_misc = np.delete(oof_p_vlm, none_idx, axis=1)
    sum_vlm_misc = p_vlm_misc.sum(axis=1, keepdims=True)
    p_vlm_misc_norm = np.where(sum_vlm_misc > 1e-12, p_vlm_misc / sum_vlm_misc, 1.0 / (n_classes - 1))

    protected_results = []
    for w in [0.02, 0.05, 0.10, 0.15, 0.20]:
        p_cond = (1.0 - w) * p_s550_misc_norm + w * p_vlm_misc_norm
        p_prot_blend = (1.0 - p_s550_none) * p_cond
        p_prot_blend = np.insert(p_prot_blend, none_idx, p_s550_none[:, 0], axis=1)

        preds_blend = [[] for _ in range(n_samples)]
        for fold_idx, fold in enumerate(splits, start=1):
            val_idx = np.array(fold["val_indices"], dtype=np.int64)
            preds_val = engine.rank_and_gate(p_prot_blend[val_idx], val_idx, q_dists_per_fold[fold_idx])
            for i, idx in enumerate(val_idx):
                preds_blend[idx] = preds_val[i]

        b_seen = calculate_map3([y_raw[i] for i in range(n_samples) if seen_mask[i]], [preds_blend[i] for i in range(n_samples) if seen_mask[i]])
        b_unseen = calculate_map3([y_raw[i] for i in range(n_samples) if not seen_mask[i]], [preds_blend[i] for i in range(n_samples) if not seen_mask[i]])
        b_comp = 0.957 * b_seen + 0.043 * b_unseen
        b_r10 = sum(1 for i in rare_indices if y_true_int[i] in np.argsort(-p_prot_blend[i])[:10])
        b_fn = sum(1 for i in none_indices if preds_blend[i][0] != "NONE")

        gate_pass = bool(b_comp > s550_comp and b_fn <= s550_fn)
        item = {
            "weight": w,
            "composite_map3": round(b_comp, 5),
            "delta_map3": round(b_comp - s550_comp, 5),
            "rare_r10": b_r10,
            "false_none": b_fn,
            "delta_fn": b_fn - s550_fn,
            "gate_pass": gate_pass
        }
        protected_results.append(item)
        print(f"  w={w:0.2f} (Protected) | Comp MAP@3: {b_comp:.5f} (Delta: {b_comp - s550_comp:+.5f}) | "
              f"Rare R@10: {b_r10} | False NONE: {b_fn} (Delta: {b_fn - s550_fn:+d}) | Gate: {'PASS' if gate_pass else 'FAIL'}")

    # Step 5: Final Decision & Report Generation
    all_experiments = integration_results + protected_results
    passing = [e for e in all_experiments if e["gate_pass"]]

    print("\n" + "=" * 115)
    print("FINAL BENCHMARK VERDICT")
    print("=" * 115)
    if len(passing) > 0:
        best_cfg = max(passing, key=lambda x: x["composite_map3"])
        decision = "PASS"
        print(f"GATE PASS: Configuration w={best_cfg['weight']} cleared the gate.")
        print(f"  Composite MAP@3: {best_cfg['composite_map3']} (Delta: {best_cfg['delta_map3']:+.5f})")
        print(f"  False NONE Demotions: {best_cfg['false_none']} (Delta: {best_cfg['delta_fn']:+d})")
    else:
        decision = "FAIL_MAINTAIN_S550"
        print("GATE FAIL: 0 VLM configurations cleared the held-out gate (> 0.54214 Composite MAP@3 and <= 104 False NONE).")
        print("Action: S550 remains locked as the selected final competition entry.")

    report = {
        "execution_time_seconds": round(time.time() - start_time, 2),
        "vlm_model": "Salesforce/blip-vqa-base",
        "gate_decision": decision,
        "vlm_standalone": {
            "composite_map3": round(vlm_comp, 5),
            "seen_map3": round(vlm_seen, 5),
            "unseen_map3": round(vlm_unseen, 5),
            "rare_r10": vlm_r10,
            "false_none": vlm_fn
        },
        "s550_reference": {
            "composite_map3": round(s550_comp, 5),
            "seen_map3": round(s550_seen, 5),
            "unseen_map3": round(s550_unseen, 5),
            "rare_r10": s550_r10,
            "false_none": s550_fn
        },
        "linear_blends": integration_results,
        "protected_blends": protected_results
    }

    report_path = ART_DIR / "blip_vlm_end_to_end_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved benchmark report to {report_path}")


if __name__ == "__main__":
    main()

