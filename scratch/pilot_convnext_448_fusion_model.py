"""Controlled Backbone Substitution: ConvNeXt-Tiny 448px Letterbox + Question-Conditioned Fusion Head.

Experiment Specification per User Directive:
1. Controlled Backbone Substitution:
   - Visual Backbone: Existing ConvNeXt-Tiny 448px letterbox image pipeline (artifacts/train_convnext_448_letterbox.npy, 768-d).
   - Question Context: Sentence-Transformers dense context embeddings (artifacts/train_text_embeddings_384.npy, 384-d).
   - Question-Conditioned Fusion Head:
     * FiLM (Feature-wise Linear Modulation) conditioning visual representations on question semantics.
     * Adaptive gating combining modulated handwriting visual evidence with question context.
     * Hardswish MLP classification head mapping to 76 misconception logits.
2. Canonical 5-Fold Protocol:
   - Strictly uses data/splits.json (canonical 5 folds, 1200 train, 300 val).
   - In-fold leak-free regeneration of champion S550 baseline.
   - Reports per-fold MAP@3 for both S550 and ConvNeXt-Tiny Fusion model.
   - Evaluates transductive Seen (1,335 traces) and inductive Unseen (165 traces) holdouts.
   - Weighted Composite MAP@3: 0.957 * Seen + 0.043 * Unseen.
3. Decision Gate:
   - Clear held-out promotion gate: best ensemble or standalone must achieve positive delta over S550.
   - If gate is not cleared, reject and do not submit.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

class ConvNeXtQuestionFusionModel(nn.Module):
    """ConvNeXt-Tiny 448px Letterbox (768-d) + Dense Question Text (384-d) Fusion Head."""
    def __init__(
        self,
        num_classes: int = 76,
        text_dim: int = 384,
        vision_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.30
    ):
        super().__init__()
        # Visual projection
        self.vis_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # FiLM Generator: produces gamma and beta for hidden_dim
        self.film_gamma = nn.Linear(text_dim, hidden_dim)
        self.film_beta = nn.Linear(text_dim, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Multimodal fusion layer norm & classifier
        self.ln_fused = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, vis_feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        v_h = self.vis_proj(vis_feat)   # (B, hidden_dim)
        t_h = self.text_proj(text_emb)  # (B, hidden_dim)

        # FiLM modulation of visual features by question context
        gamma = self.film_gamma(text_emb)
        beta = self.film_beta(text_emb)
        v_film = (1.0 + gamma) * v_h + beta

        # Adaptive gating between modulated visual features and question semantics
        g = self.gate(torch.cat([v_film, t_h], dim=-1))
        fused = self.ln_fused(g * v_film + (1.0 - g) * t_h)

        logits = self.classifier(fused)
        return logits

class CanonicalDataBundle:
    def __init__(self, data_dir: Path, art_dir: Path):
        self.tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
        self.classes = self.tax.all_codes(include_none=True)
        self.none_idx = self.classes.index("NONE")
        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
        self.y_raw = self.train_df["root_code"].fillna("NONE").astype(str).tolist()
        self.n_samples = len(self.train_df)
        self.targets = np.array([self.class_to_idx.get(y, self.none_idx) for y in self.y_raw], dtype=np.int64)

        # Baseline components for S550 regeneration
        self.oof_m15 = np.load(art_dir / "oof_m15_balanced_fusion.npy")
        self.oof_dual = np.load(art_dir / "oof_arcface_dualhead_448_soft.npy")
        self.oof_comp = np.load(art_dir / "oof_arcface_composite_448.npy")
        self.oof_v1 = np.load(art_dir / "oof_arcface_pseudo_3500.npy")
        self.honest_text = np.load(art_dir / "honest_oof_text.npy")
        self.oof_sem_text = np.load(art_dir / "oof_semantic_text.npy")
        self.oof_convnext = np.load(art_dir / "oof_convnext_vision.npy")
        self.oof_swin = np.load(art_dir / "oof_swin_vision.npy")
        self.oof_finetuned = np.load(art_dir / "oof_finetuned_vision.npy")
        self.oof_trocr = np.load(art_dir / "oof_trocr_vision.npy")
        self.oof_clean_arcface = np.load(art_dir / "oof_clean_arcface_2328.npy")
        self.oof_bin_v1 = np.load(art_dir / "oof_compact_binary_prob.npy")

        # ConvNeXt-Tiny 448px letterbox features
        print("Loading train_convnext_448_letterbox.npy...", flush=True)
        self.convnext_feat = torch.from_numpy(np.load(art_dir / "train_convnext_448_letterbox.npy")).float()
        print(f"Loaded ConvNeXt-Tiny features: {self.convnext_feat.shape}, {self.convnext_feat.dtype}")

        # Text embeddings
        self.text_emb_tensor = torch.from_numpy(np.load(art_dir / "train_text_embeddings_384.npy")).float()
        print(f"Loaded text embeddings: {self.text_emb_tensor.shape}, {self.text_emb_tensor.dtype}")

        # Splits
        splits_json = json.loads((data_dir / "splits.json").read_text())
        self.folds = splits_json["folds"]

    def build_fold_clean_multimodal(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        prior_eng = AdaptiveQuestionPriorEngine(classes=self.classes).fit(trn_df_fold)
        val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

        p_clean_val = (
            0.6448 * val_prior +
            0.0839 * sharp(self.honest_text[val_idx], 0.6) +
            0.0401 * sharp(self.oof_sem_text[val_idx], 0.6) +
            0.0458 * self.oof_convnext[val_idx] +
            0.0454 * self.oof_swin[val_idx] +
            0.0483 * self.oof_finetuned[val_idx] +
            0.0455 * self.oof_trocr[val_idx] +
            0.0462 * sharp(self.oof_clean_arcface[val_idx], 0.8)
        )
        return p_clean_val

    def generate_s550_probs(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, Dict[str, Counter]]:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        n_val = len(val_idx)

        p_base_v1_trn = 0.95 * self.oof_m15[trn_idx] + 0.02 * self.oof_dual[trn_idx] + 0.03 * self.oof_comp[trn_idx]
        top3_trn = np.argsort(-p_base_v1_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(self.classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
        mult = np.ones(self.n_classes, dtype=np.float64)
        for i, c in enumerate(self.classes):
            if c != "NONE":
                mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
        mult[self.none_idx] = 0.9840

        p_base_v1_val = (0.95 * self.oof_m15[val_idx] + 0.02 * self.oof_dual[val_idx] + 0.03 * self.oof_comp[val_idx]) * mult
        p_base_v1_val[:, self.none_idx] *= 0.9995

        q_label_dist: Dict[str, Counter] = {
            q: Counter(grp["root_code"].fillna("NONE").values)
            for q, grp in trn_df_fold.groupby("question_id")
        }

        p_boosted_v1 = np.copy(p_base_v1_val)
        for i in range(n_val):
            qid = val_df_fold.iloc[i]["question_id"]
            if qid in q_label_dist:
                allowed_misconceptions = {c for c, cnt in q_label_dist[qid].items() if c != "NONE"}
                for ci, c in enumerate(self.classes):
                    if c in allowed_misconceptions:
                        p_boosted_v1[i, ci] *= 1.20

        p_arc_v1 = (0.95 * self.oof_m15[val_idx] + 0.05 * self.oof_v1[val_idx]) * mult
        p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
        p_clean = self.build_fold_clean_multimodal(trn_idx, val_idx)
        p_s550 = 0.96 * p_champ_v1 + 0.04 * p_clean

        return p_s550, q_label_dist

    def rank_and_gate(self, p_matrix: np.ndarray, val_idx: np.ndarray, q_label_dist: Dict[str, Counter], beta: float = 0.82) -> List[List[str]]:
        cand = p_matrix.copy()
        n = len(cand)
        row_idx = np.arange(n)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != self.none_idx, self.none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[self.classes[r1[i]], self.classes[top2[i, 0]], self.classes[top2[i, 1]]] for i in range(n)]

        p_bin = self.oof_bin_v1[val_idx]
        val_df = self.train_df.iloc[val_idx].reset_index(drop=True)

        for i in range(n):
            if preds[i][0] == "NONE" and p_bin[i] >= 0.78:
                row_p = p_matrix[i].copy()
                p_none = row_p[self.none_idx]
                row_p[self.none_idx] = -1.0
                top_m_idx = int(np.argmax(row_p))
                top_m_code = self.classes[top_m_idx]
                gap = p_none - row_p[top_m_idx]

                if 0 < gap < 0.04:
                    qid = val_df.iloc[i]["question_id"]
                    allowed = {c for c, cnt in q_label_dist.get(qid, Counter()).items() if c != "NONE"}
                    if top_m_code in allowed:
                        row_p[top_m_idx] = -1.0
                        sec_m = self.classes[int(np.argmax(row_p))]
                        preds[i] = [top_m_code, "NONE", sec_m]
        return preds

def train_and_eval_fold(
    data: CanonicalDataBundle,
    fold_idx: int,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Counter], float, float]:
    """Trains ConvNeXt-Tiny Question Fusion model on fold training indices, evaluates on holdout."""
    fold_info = data.folds[fold_idx - 1]
    trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
    val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

    # 1. Regenerate leak-free S550 baseline inside fold
    p_s550, q_dist = data.generate_s550_probs(trn_idx, val_idx)

    # 2. Build datasets
    trn_idx_t = torch.as_tensor(trn_idx, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long)

    X_vis_trn = data.convnext_feat[trn_idx_t]
    X_txt_trn = data.text_emb_tensor[trn_idx_t]
    y_trn = torch.as_tensor(data.targets[trn_idx], dtype=torch.int64)

    X_vis_val = data.convnext_feat[val_idx_t]
    X_txt_val = data.text_emb_tensor[val_idx_t]

    train_loader = DataLoader(
        TensorDataset(X_vis_trn, X_txt_trn, y_trn),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_vis_val, X_txt_val),
        batch_size=64,
        shuffle=False
    )

    # 3. Model & Optimizer
    model = ConvNeXtQuestionFusionModel(num_classes=data.n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    for ep in range(epochs):
        for b_vis, b_txt, b_y in train_loader:
            b_vis = b_vis.to(device)
            b_txt = b_txt.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            logits = model(b_vis, b_txt)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # 4. Out-of-fold inference
    model.eval()
    val_logits_list = []
    with torch.no_grad():
        for b_vis, b_txt in val_loader:
            b_vis = b_vis.to(device)
            b_txt = b_txt.to(device)
            logits = model(b_vis, b_txt)
            val_logits_list.append(logits.cpu())

    val_logits = torch.cat(val_logits_list, dim=0)
    p_convnext_val = F.softmax(val_logits, dim=-1).numpy()

    # Calculate Fold MAP@3
    y_true_fold = [data.y_raw[i] for i in val_idx]
    preds_s550 = data.rank_and_gate(p_s550, val_idx, q_dist)
    sc_s550 = calculate_map3(y_true_fold, preds_s550)

    preds_convnext = [[data.classes[c] for c in np.argsort(-p_convnext_val[r])[:3]] for r in range(len(val_idx))]
    sc_convnext = calculate_map3(y_true_fold, preds_convnext)

    return p_s550, p_convnext_val, q_dist, sc_s550, sc_convnext

def main():
    print("=" * 95)
    print("CONTROLLED BACKBONE SUBSTITUTION: CONVNEXT-TINY 448PX LETTERBOX + QUESTION FUSION")
    print("=" * 95)
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data = CanonicalDataBundle(DATA_DIR, ART_DIR)
    n_samples = data.n_samples

    seen_mask = np.zeros(n_samples, dtype=bool)
    oof_p_s550 = np.zeros((n_samples, data.n_classes), dtype=np.float64)
    oof_p_conv = np.zeros((n_samples, data.n_classes), dtype=np.float32)

    q_dists_per_fold: Dict[int, Dict[str, Counter]] = {}
    val_indices_all: List[int] = []
    fold_scores_s550: List[float] = []
    fold_scores_conv: List[float] = []

    cached_conv_oof_path = ART_DIR / "oof_convnext_448_question_fusion.npy"
    has_cached_conv = cached_conv_oof_path.exists()

    if has_cached_conv:
        print(f"\nFound existing {cached_conv_oof_path.name}. Loading predictions and regenerating in-fold S550 baseline...")
        oof_p_conv = np.load(cached_conv_oof_path)

    print("\nProcessing 5 Canonical Folds...")
    for fold_num in range(1, 6):
        t0 = time.time()
        fold_info = data.folds[fold_num - 1]
        trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
        val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

        trn_qids = set(data.train_df.iloc[trn_idx]["question_id"])
        for i in val_idx:
            if data.train_df.iloc[i]["question_id"] in trn_qids:
                seen_mask[i] = True

        p_550_val, q_dist = data.generate_s550_probs(trn_idx, val_idx)
        oof_p_s550[val_idx] = p_550_val
        q_dists_per_fold[fold_num] = q_dist

        y_true_fold = [data.y_raw[i] for i in val_idx]
        preds_s550 = data.rank_and_gate(p_550_val, val_idx, q_dist)
        sc_s550 = calculate_map3(y_true_fold, preds_s550)
        fold_scores_s550.append(sc_s550)

        if not has_cached_conv:
            _, p_conv_val, _, _, sc_conv = train_and_eval_fold(
                data,
                fold_idx=fold_num,
                device=device,
                epochs=15,
                batch_size=32,
                lr=1e-3
            )
            oof_p_conv[val_idx] = p_conv_val
            fold_scores_conv.append(sc_conv)
        else:
            p_conv_val = oof_p_conv[val_idx]
            preds_conv = [[data.classes[c] for c in np.argsort(-p_conv_val[r])[:3]] for r in range(len(val_idx))]
            sc_conv = calculate_map3(y_true_fold, preds_conv)
            fold_scores_conv.append(sc_conv)

        print(f"  Fold {fold_num}/5 ({time.time() - t0:.1f}s) | S550 Base: {sc_s550:.5f} | ConvNeXt Fusion: {sc_conv:.5f}", flush=True)

    if not has_cached_conv:
        np.save(cached_conv_oof_path, oof_p_conv)
        print(f"\n[Saved] {cached_conv_oof_path.name} (shape: {oof_p_conv.shape})")

    # Evaluate standalone and fusion variants across all 1500 traces
    weights_to_test = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    results_summary: List[Dict[str, Any]] = []

    y_true_all = data.y_raw

    print("\n" + "=" * 115)
    print("OUT-OF-FOLD PERFORMANCE: S550 BASELINE vs CONVNEXT-TINY 448PX FUSION VARIANTS")
    print("=" * 115)
    print(f"{'Variant / Model Description':55s} | {'Seen MAP@3':12s} | {'Unseen MAP@3':12s} | {'Comp MAP@3':11s} | {'Delta vs S550':13s}")
    print("-" * 115)

    base_comp = 0.0

    for w in weights_to_test:
        preds_all = [[] for _ in range(n_samples)]
        for fold_num in range(1, 6):
            fold_info = data.folds[fold_num - 1]
            val_idx = np.array(fold_info["val_indices"], dtype=np.int64)
            q_dist = q_dists_per_fold[fold_num]

            if w == 0.0:
                p_fused = oof_p_s550[val_idx]
                preds_val = data.rank_and_gate(p_fused, val_idx, q_dist)
            elif w == 1.0:
                p_fused = oof_p_conv[val_idx]
                preds_val = [[data.classes[c] for c in np.argsort(-p_fused[r])[:3]] for r in range(len(val_idx))]
            else:
                p_fused = (1.0 - w) * oof_p_s550[val_idx] + w * oof_p_conv[val_idx]
                preds_val = data.rank_and_gate(p_fused, val_idx, q_dist)

            for i, idx in enumerate(val_idx):
                preds_all[idx] = preds_val[i]

        y_seen = [y_true_all[i] for i in range(n_samples) if seen_mask[i]]
        p_seen = [preds_all[i] for i in range(n_samples) if seen_mask[i]]
        map_seen = calculate_map3(y_seen, p_seen)

        y_unseen = [y_true_all[i] for i in range(n_samples) if not seen_mask[i]]
        p_unseen = [preds_all[i] for i in range(n_samples) if not seen_mask[i]]
        map_unseen = calculate_map3(y_unseen, p_unseen)

        comp = 0.957 * map_seen + 0.043 * map_unseen

        if w == 0.0:
            base_comp = comp
            desc = "0. S550 Baseline Reference (w_conv = 0.0)"
        elif w == 1.0:
            desc = "Standalone ConvNeXt-Tiny Fusion Model (w_conv = 1.0)"
        else:
            desc = f"Ensemble: (1-{w})*S550 + {w}*ConvNeXt-Fusion"

        delta = comp - base_comp
        results_summary.append({
            "desc": desc,
            "weight_conv": w,
            "seen_map3": round(map_seen, 5),
            "unseen_map3": round(map_unseen, 5),
            "composite_map3": round(comp, 5),
            "delta_vs_s550": round(delta, 5)
        })

        print(f"{desc:55s} | {map_seen:.5f}      | {map_unseen:.5f}        | {comp:.5f}     | {delta:+.5f}")

    # Decision Gate Check
    best_entry = max(results_summary[1:], key=lambda x: float(x["composite_map3"]))
    best_delta: float = float(best_entry["delta_vs_s550"])
    promotable: bool = best_delta > 0.00030

    print("\n" + "=" * 95)
    print("DECISION GATE EVALUATION:")
    if promotable:
        print(f"  [PASS] ConvNeXt-Tiny fusion model achieved consistent held-out gain: {best_delta:+.5f}")
        print(f"  Optimal configuration: {best_entry['desc']}")
    else:
        print(f"  [REJECT] ConvNeXt-Tiny fusion model did not clear the promotion gate over S550 baseline.")
        print(f"  Best Delta vs S550: {best_delta:+.5f} ({best_entry['desc']})")
        print("  Active champion S550 remains untouched. No submission created.")
    print("=" * 95)

    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": {
            "vision_backbone": "ConvNeXt-Tiny 448px Letterbox (768-d features)",
            "text_representation": "Dense Context Embeddings (384-d)",
            "fusion": "FiLM Modulation + Adaptive Sigmoid Gating + Hardswish MLP",
            "epochs": 15,
            "lr": 1e-3,
            "batch_size": 32
        },
        "per_fold_scores": {
            f"fold_{i+1}": {
                "s550": round(fold_scores_s550[i], 5),
                "convnext_fusion": round(fold_scores_conv[i], 5)
            }
            for i in range(5)
        },
        "evaluation_protocol": {
            "splits": "data/splits.json (canonical 5 folds, 1200 train, 300 val)",
            "seen_traces": int(seen_mask.sum()),
            "unseen_traces": int((~seen_mask).sum()),
            "composite_metric": "0.957 * Seen + 0.043 * Unseen"
        },
        "results": results_summary,
        "decision": {
            "promoted": promotable,
            "best_delta": round(best_delta, 5),
            "best_configuration": str(best_entry["desc"]),
            "runtime_seconds": round(time.time() - start_time, 2)
        }
    }

    report_path = ART_DIR / "convnext_448_fusion_pilot_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Report] {report_path.name}")

if __name__ == "__main__":
    main()

