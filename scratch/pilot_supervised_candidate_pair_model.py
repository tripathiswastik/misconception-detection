"""Supervised Candidate-Conditioned Multimodal Model Pilot.

Methodological Upgrades per User Directive:
1. Supervised Candidate-Pair Model:
   - Evaluates (Question, Student Work Vision, Candidate Misconception) jointly.
   - Fine-tunes Layer 11 of multilingual MiniLM + multimodal projection head with hard negatives.
2. Check 1 (Test-like Reality):
   - Zero gold transcription_steps leakage.
   - Text sequence contains only test-available context (Question, Grade, Topic, Correct Answer).
   - Student work is represented via normalized visual handwriting embeddings (TrOCR DeiT backbone),
     which are identical in distribution between train and test.
3. Check 2 (Fold-Safe Base Predictions):
   - Base model trained strictly in-fold using outer GroupKFold on question_id (zero trace leakage).
   - Candidate lists and Rank 1 generated strictly from fold-safe in-fold predictions.
4. Promotion Invariant:
   - Rank 1 is 100% locked.
   - Requires consistent positive outer-fold gains (all fold deltas >= 0.0) before any submission consideration.
   - 0.55049 champion remains untouched.
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

class CandidateDataset(Dataset):
    def __init__(self, seq_a_list: List[str], seq_b_list: List[str], vis_list: List[np.ndarray], meta_list: List[np.ndarray], labels: List[float]):
        self.seq_a = seq_a_list
        self.seq_b = seq_b_list
        self.vis = vis_list
        self.meta = meta_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.seq_a[index],
            self.seq_b[index],
            torch.tensor(self.vis[index], dtype=torch.float32),
            torch.tensor(self.meta[index], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32)
        )

class CandidatePairModel(nn.Module):
    def __init__(self, encoder: nn.Module, vis_dim: int = 384, meta_dim: int = 2):
        super().__init__()
        self.encoder = encoder
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.head = nn.Sequential(
            nn.Linear(384 + 128 + meta_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, vis: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        out = cast(Any, self.encoder)(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = out.last_hidden_state[:, 0, :]  # 384-d
        v_rep = self.vis_proj(vis)               # 128-d
        fused = torch.cat([cls_rep, v_rep, meta], dim=-1)
        logits = self.head(fused).squeeze(-1)
        return logits

def main():
    print("=" * 80)
    print("PILOT: SUPERVISED CANDIDATE-CONDITIONED MULTIMODAL MODEL")
    print("Check 1: Zero Gold Text Leakage (Test-Like) | Check 2: Fold-Safe Base Predictions")
    print("Trainable MiniLM Layer 11 + Handwriting Vision Projection | Rank 1 Locked")
    print("=" * 80)

    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")

    # 1. Load Taxonomy & Data
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    tax_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "taxonomy_v1.csv"))
    tax_info: Dict[str, Tuple[str, str]] = {}
    for _, r in tax_df.iterrows():
        c_id = str(r["id"])
        tax_info[c_id] = (str(r["name"]), str(r["description"]))

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 2. Check 1 Compliance: Test-like sequences and multimodal visual features
    print("\n[Check 1 Compliance] Building test-like sequences (Zero gold text leakage)...")
    # Gold transcription_steps is NEVER used!
    test_like_seq_a: List[str] = []
    for i in range(n_samples):
        gr = str(train_df.iloc[i]["grade"])
        top = str(train_df.iloc[i]["topic_name"])
        q_txt = str(train_df.iloc[i]["question_text"])
        ans = str(train_df.iloc[i]["correct_answer"])
        seq_a = f"Kelas: {gr} | Topik: {top} | Kunci: {ans} | Soal: {q_txt}"
        test_like_seq_a.append(seq_a)

    # Load normalized student handwriting visual features
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tr_norm = cast(np.ndarray, normalize(tr_feat)).astype(np.float32)

    # Load multimodal base features for fold-safe baseline candidate generator
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    st_feat = np.load(ART_DIR / "train_perceptual_strokes.npy")
    X_multimodal = np.hstack([
        cast(np.ndarray, normalize(cn_feat)),
        cast(np.ndarray, normalize(sw_feat)),
        cast(np.ndarray, normalize(tr_feat)),
        cast(np.ndarray, normalize(tx_feat)),
        cast(np.ndarray, normalize(st_feat))
    ])

    # 3. Check 2 Compliance: Generate Fold-Safe Base Predictions across 5-Fold GroupKFold
    print("\n[Check 2 Compliance] Generating 100% fold-safe base predictions across 5-fold GroupKFold...")
    outer_gkf = GroupKFold(n_splits=5)
    base_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    base_top3: List[List[str]] = [[] for _ in range(n_samples)]
    base_r1_is_none = np.zeros(n_samples, dtype=bool)
    fold_base_scores = []

    for fold_num, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        val_y = y_raw[val_idx].tolist()

        # Fold-safe multimodal classifier
        clf = RidgeClassifier(alpha=10.0, random_state=42)
        clf.fit(X_multimodal[trn_idx], y_indices[trn_idx])

        # Convert decision function to calibrated probabilities
        df_val = clf.decision_function(X_multimodal[val_idx])
        # Softmax
        exp_df = np.exp(df_val - np.max(df_val, axis=1, keepdims=True))
        p_clf_val = exp_df / exp_df.sum(axis=1, keepdims=True)

        # In-fold prior strictly on trn_df
        prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        p_prior_val = prior_eng.predict_proba(val_df)

        # In-fold blend
        p_fold_val = 0.55 * p_clf_val + 0.45 * p_prior_val
        base_probs[val_idx] = p_fold_val

        v_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_fold_val, axis=1)]
        s_f = calculate_map3(val_y, v_top3)
        fold_base_scores.append(s_f)

        for loc_i, glob_i in enumerate(val_idx):
            base_top3[glob_i] = v_top3[loc_i]
            base_r1_is_none[glob_i] = (v_top3[loc_i][0] == "NONE")

        print(f"  Outer Fold {fold_num} Fold-Safe Base MAP@3: {s_f:.5f}")

    overall_base_map3 = calculate_map3(y_raw.tolist(), base_top3)
    r1_none_count = int(base_r1_is_none.sum())
    print(f"Overall Fold-Safe Base MAP@3: {overall_base_map3:.5f}")
    print(f"Rows with Rank 1 == NONE: {r1_none_count} / {n_samples} ({r1_none_count/n_samples:.1%})")

    # 4. Extract Candidate Misconceptions (K=3)
    K_POOL = 3
    cand_prob_mat = base_probs.copy()
    cand_prob_mat[:, none_idx] = -1.0
    sorted_cands = np.argsort(-cand_prob_mat, axis=1)[:, :K_POOL]

    # Pre-extract misconception descriptions
    cand_seq_b = {}
    for c_idx, c_code in enumerate(classes):
        if c_code != "NONE":
            c_name, c_desc = tax_info.get(c_code, (c_code, c_code))
            cand_seq_b[c_idx] = f"Miskonsepsi: {c_code}: {c_name}. {c_desc}"
        else:
            cand_seq_b[c_idx] = "NONE: Tidak ada miskonsepsi teridentifikasi"

    # 5. Supervised Candidate-Pair Training & Outer Evaluation
    tokenizer = cast(Any, AutoTokenizer).from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
    reranked_top3: List[List[str]] = [[] for _ in range(n_samples)]
    outer_deltas: List[float] = []
    outer_swap_reports: List[Dict[str, int]] = []

    print("\n" + "=" * 80)
    print("TRAINING SUPERVISED CANDIDATE-PAIR MODEL ACROSS OUTER FOLDS")
    print("=" * 80)

    for outer_fold, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        print(f"\n--- Outer Fold {outer_fold} / 5 ---")
        val_y = y_raw[val_idx].tolist()
        val_base_top3 = [list(base_top3[g_i]) for g_i in val_idx]
        base_f_score = calculate_map3(val_y, val_base_top3)

        # Assemble supervised training pairs strictly from trn_idx rows where Rank 1 is NONE
        trn_seq_a = []
        trn_seq_b = []
        trn_vis = []
        trn_meta = []
        trn_labels = []

        for g_i in trn_idx:
            if base_r1_is_none[g_i]:
                true_c = y_raw[g_i]
                c_pool = sorted_cands[g_i]
                p_cands = [base_probs[g_i, c] for c in c_pool]
                p_max = max(p_cands) if p_cands else 1.0

                for k_pos, c_idx in enumerate(c_pool):
                    code = classes[c_idx]
                    is_pos = (code == true_c)
                    trn_seq_a.append(test_like_seq_a[g_i])
                    trn_seq_b.append(cand_seq_b[c_idx])
                    trn_vis.append(tr_norm[g_i])
                    # Meta: probability and relative gap
                    p_c = float(base_probs[g_i, c_idx])
                    gap = float(p_c - p_max)
                    trn_meta.append(np.array([p_c, gap], dtype=np.float32))
                    trn_labels.append(1.0 if is_pos else 0.0)

        n_pairs = len(trn_labels)
        pos_count = sum(trn_labels)
        print(f"  Training pairs: {n_pairs} ({int(pos_count)} positives, {n_pairs - int(pos_count)} hard negatives)")

        # Create Model
        encoder = cast(Any, AutoModel).from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
        # Freeze layers 0-10, unfreeze Layer 11
        for param in encoder.parameters():
            param.requires_grad = False
        for param in encoder.encoder.layer[-1].parameters():
            param.requires_grad = True

        model = CandidatePairModel(encoder=encoder, vis_dim=384, meta_dim=2).to(device)

        # Optimizer with tiered learning rates
        optimizer = torch.optim.AdamW([
            {"params": encoder.encoder.layer[-1].parameters(), "lr": 2e-5, "weight_decay": 0.01},
            {"params": model.vis_proj.parameters(), "lr": 1e-3, "weight_decay": 0.01},
            {"params": model.head.parameters(), "lr": 1e-3, "weight_decay": 0.01}
        ])

        # Loss function with pos_weight
        pos_weight = torch.tensor([(n_pairs - pos_count) / max(pos_count, 1.0)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Batch loader
        trn_dataset = CandidateDataset(trn_seq_a, trn_seq_b, trn_vis, trn_meta, trn_labels)
        trn_loader = DataLoader(trn_dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(1, 4):
            total_loss = 0.0
            for b_seq_a, b_seq_b, b_vis, b_meta, b_y in trn_loader:
                optimizer.zero_grad()
                enc = tokenizer(list(b_seq_a), list(b_seq_b), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                logits = model(enc["input_ids"], enc["attention_mask"], b_vis.to(device), b_meta.to(device))
                loss = loss_fn(logits, b_y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(b_y)
            print(f"    Epoch {epoch}/3 Loss: {total_loss / n_pairs:.4f}")

        # Evaluate on untouched holdout V_k
        model.eval()
        val_reranked = [list(r) for r in val_base_top3]
        f_swaps = 0
        f_helped = 0
        f_hurt = 0
        f_neutral = 0

        # Rerank Candidate 1 (Rank 2) vs Candidate 2 (Rank 3) for NONE-at-Rank-1 rows
        # Rank 1 remains 100% strictly anchored!
        with torch.no_grad():
            for loc_i, glob_i in enumerate(val_idx):
                if base_r1_is_none[glob_i]:
                    cands = sorted_cands[glob_i]
                    c1_idx = cands[0]
                    c2_idx = cands[1]

                    p_c1 = float(base_probs[glob_i, c1_idx])
                    p_c2 = float(base_probs[glob_i, c2_idx])
                    gap_base = p_c1 - p_c2

                    # Only test swap if base gap is close (< 0.040)
                    if 0 < gap_base < 0.040:
                        s_a = test_like_seq_a[glob_i]
                        s_b1 = cand_seq_b[c1_idx]
                        s_b2 = cand_seq_b[c2_idx]
                        v = torch.tensor(np.vstack([tr_norm[glob_i], tr_norm[glob_i]]), dtype=torch.float32, device=device)
                        m = torch.tensor([[p_c1, 0.0], [p_c2, -gap_base]], dtype=torch.float32, device=device)

                        enc = tokenizer([s_a, s_a], [s_b1, s_b2], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                        scores = model(enc["input_ids"], enc["attention_mask"], v, m).cpu().numpy()

                        score_c1 = scores[0]
                        score_c2 = scores[1]

                        # Margin threshold to swap
                        if score_c2 - score_c1 > 0.50:
                            c1_code = classes[c1_idx]
                            c2_code = classes[c2_idx]
                            val_reranked[loc_i][1] = c2_code
                            val_reranked[loc_i][2] = c1_code
                            f_swaps += 1

                            true_l = val_y[loc_i]
                            if true_l == c2_code:
                                f_helped += 1
                            elif true_l == c1_code:
                                f_hurt += 1
                            else:
                                f_neutral += 1

        new_f_score = calculate_map3(val_y, val_reranked)
        delta_f = new_f_score - base_f_score
        outer_deltas.append(delta_f)
        outer_swap_reports.append({"swaps": f_swaps, "helped": f_helped, "hurt": f_hurt, "neutral": f_neutral})

        print(f"  Holdout Result: Baseline: {base_f_score:.5f} -> Reranked: {new_f_score:.5f} ({delta_f:+.5f})")
        print(f"  Swaps: {f_swaps} (Helped: {f_helped}, Hurt: {f_hurt}, Neutral: {f_neutral})")

        for loc_i, glob_i in enumerate(val_idx):
            reranked_top3[glob_i] = val_reranked[loc_i]

    # 6. Overall Benchmark Results
    overall_new_map3 = calculate_map3(y_raw.tolist(), reranked_top3)
    net_delta = overall_new_map3 - overall_base_map3
    tot_swaps = sum(r["swaps"] for r in outer_swap_reports)
    tot_helped = sum(r["helped"] for r in outer_swap_reports)
    tot_hurt = sum(r["hurt"] for r in outer_swap_reports)
    tot_neutral = sum(r["neutral"] for r in outer_swap_reports)

    print("\n" + "=" * 80)
    print("FINAL SUPERVISED CANDIDATE-PAIR MODEL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Baseline MAP@3          : {overall_base_map3:.5f}")
    print(f"Reranked MAP@3          : {overall_new_map3:.5f}")
    print(f"Net Generalization Delta: {net_delta:+.5f}")
    print(f"Per-Fold Deltas         : {[round(d, 5) for d in outer_deltas]}")
    print(f"Total Swaps across 1500 : {tot_swaps} (Helped: {tot_helped}, Hurt: {tot_hurt}, Neutral: {tot_neutral})")
    if tot_helped + tot_hurt > 0:
        print(f"Active Swap Precision   : {tot_helped / (tot_helped + tot_hurt):.1%}")

    gate_passed = (net_delta >= 0.00020) and all(d >= -1e-7 for d in outer_deltas)
    print("\nPROMOTION GATE STATUS   : " + ("PASSED" if gate_passed else "REJECTED / UNVALIDATED"))
    print("Active Champion Status  : blend_sota550_precision_gated_submission.csv (0.55049 LB) UNTOUCHED")
    print("=" * 80)

    # Save artifact
    out_artifact = {
        "pilot_name": "supervised_candidate_pair_model_pilot",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "check_1_ocr_compliance": "Zero gold text leakage. Evaluated strictly with test-like sequences + normalized TrOCR vision features.",
            "check_2_leakage_compliance": "Zero trace leakage. 100% fold-safe in-fold base predictions and candidate sets regenerated across GroupKFold on question_id.",
            "encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Layer 11 trainable)",
            "multimodal_projection": "Handwriting Vision MLP (TrOCR DeiT 384 -> 128)",
            "rank1_invariant": "Strictly locked (0 Rank-1 mutations)"
        },
        "results": {
            "baseline_map3": round(overall_base_map3, 5),
            "reranked_map3": round(overall_new_map3, 5),
            "net_delta": round(net_delta, 5),
            "per_fold_deltas": [round(d, 5) for d in outer_deltas],
            "total_swaps": tot_swaps,
            "helped": tot_helped,
            "hurt": tot_hurt,
            "neutral": tot_neutral,
            "active_precision": round(tot_helped / max(tot_helped + tot_hurt, 1), 4),
            "gate_passed": gate_passed
        },
        "fold_details": outer_swap_reports,
        "runtime_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "supervised_candidate_pair_pilot_report.json"
    with open(report_path, "w") as f:
        json.dump(out_artifact, f, indent=2)
    print(f"[Saved Report Artifact] {report_path.name}")

if __name__ == "__main__":
    main()
