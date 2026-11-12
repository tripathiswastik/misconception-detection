"""Generate and Submit Candidate S559 (ConvNeXt-Tiny 448px Letterbox + Question Fusion at w=0.05).

Verification Protocol:
- Baseline S550 Composite CV: 0.54214 (Seen: 0.54295, Unseen: 0.52424)
- Candidate S559 Composite CV: 0.54383 (Seen: 0.54457, Unseen: 0.52727) -> Delta: +0.00168 (Passes Gate)
- Reconstructs leak-free S550 champion continuous test probabilities
- Trains 5-fold ensemble of ConvNeXtQuestionFusionModel and averages test probabilities
- Blends at optimal w=0.05, applies beta=0.82 discounting and precision gating (tau=0.78, eps=0.04)
- Submits to Kaggle API and monitors live Leaderboard scoring
"""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.fusion.validator import validate_submission_file

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"
DOWNLOADS_DIR = Path("C:/Users/hp/Downloads")
COMPETITION = "misconception-detection-in-k-12-student-math-workings"
SUB_NAME = "blend_sota559_convnext448_fusion_submission.csv"
SUB_FILE = SUB_DIR / SUB_NAME
MESSAGE = "blend_sota559_convnext448_fusion (5-fold ConvNeXt-Tiny 448px letterbox + FiLM/gate fusion w=0.05, CV delta +0.00168)"

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
        v_h = self.vis_proj(vis_feat)
        t_h = self.text_proj(text_emb)

        gamma = self.film_gamma(text_emb)
        beta = self.film_beta(text_emb)
        v_film = (1.0 + gamma) * v_h + beta

        g = self.gate(torch.cat([v_film, t_h], dim=-1))
        fused = self.ln_fused(g * v_film + (1.0 - g) * t_h)

        logits = self.classifier(fused)
        return logits

def train_and_predict_test_fold(
    fold_idx: int,
    trn_idx: np.ndarray,
    train_conv: torch.Tensor,
    train_text: torch.Tensor,
    train_targets: np.ndarray,
    test_conv: torch.Tensor,
    test_text: torch.Tensor,
    n_classes: int,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3
) -> np.ndarray:
    """Trains fusion head on fold training data and generates test probabilities."""
    trn_idx_t = torch.as_tensor(trn_idx, dtype=torch.long)
    X_vis_trn = train_conv[trn_idx_t]
    X_txt_trn = train_text[trn_idx_t]
    y_trn = torch.as_tensor(train_targets[trn_idx], dtype=torch.int64)

    train_loader = DataLoader(
        TensorDataset(X_vis_trn, X_txt_trn, y_trn),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_conv, test_text),
        batch_size=64,
        shuffle=False
    )

    model = ConvNeXtQuestionFusionModel(num_classes=n_classes).to(device)
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

    model.eval()
    test_logits_list = []
    with torch.no_grad():
        for b_vis, b_txt in test_loader:
            b_vis = b_vis.to(device)
            b_txt = b_txt.to(device)
            logits = model(b_vis, b_txt)
            test_logits_list.append(logits.cpu())

    test_logits = torch.cat(test_logits_list, dim=0)
    p_test_fold = F.softmax(test_logits, dim=-1).numpy()
    return p_test_fold

def generate_s550_test_probs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: List[str],
    none_idx: int
) -> Tuple[np.ndarray, Dict[str, Set[str]]]:
    """Reconstructs continuous test probability matrix of active champion S550."""
    n_test = len(test_df)
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") + 0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
    top3_tr = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_test_cal = p_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    q_allowed: Dict[str, Set[str]] = {
        str(q): {str(x) for x in grp["root_code"].dropna().unique()}
        for q, grp in train_df.groupby("question_id")
    }
    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_allowed:
            for ci, c in enumerate(classes):
                if c != "NONE" and c in q_allowed[qid]:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
    p_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc3500_cal

    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

    p_clean = (
        0.6448 * test_prior_probs +
        0.0839 * sharp(np.load(ART_DIR / "test_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "test_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "test_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "test_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "test_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "test_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "test_clean_arcface_2328.npy"), 0.8)
    )
    p_s550 = 0.96 * p_champ + 0.04 * p_clean
    return p_s550, q_allowed

def main():
    print("=" * 85)
    print("GENERATING CANDIDATE S559 (CONVNEXT-TINY 448PX FUSION + S550 BLEND)")
    print("=" * 85)
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Data & Taxonomy
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_train = len(train_df)
    n_test = len(test_df)
    train_targets = np.array([class_to_idx.get(str(y), none_idx) for y in train_df["root_code"].fillna("NONE")], dtype=np.int64)

    splits_json = json.loads((DATA_DIR / "splits.json").read_text())
    folds = splits_json["folds"]

    # 2. Load Visual & Text Features
    print("\nLoading feature matrices...")
    train_conv = torch.from_numpy(np.load(ART_DIR / "train_convnext_448_letterbox.npy")).float()
    train_text = torch.from_numpy(np.load(ART_DIR / "train_text_embeddings_384.npy")).float()
    test_conv = torch.from_numpy(np.load(ART_DIR / "test_convnext_448_letterbox.npy")).float()
    test_text = torch.from_numpy(np.load(ART_DIR / "test_text_embeddings_384.npy")).float()
    print(f"  Train ConvNeXt: {train_conv.shape}, Train Text: {train_text.shape}")
    print(f"  Test ConvNeXt: {test_conv.shape}, Test Text: {test_text.shape}")

    # 3. Train 5 Folds & Average Test Probabilities
    cached_conv_test_path = ART_DIR / "test_convnext_448_question_fusion.npy"
    if cached_conv_test_path.exists():
        print(f"\nFound existing {cached_conv_test_path.name}. Loading cached test predictions...")
        p_conv_test = np.load(cached_conv_test_path)
    else:
        print("\nTraining 5-Fold ConvNeXt Question Fusion Ensemble on Canonical Splits...")
        test_probs_folds = np.zeros((5, n_test, n_classes), dtype=np.float32)

        for fold_num in range(1, 6):
            t0 = time.time()
            fold_info = folds[fold_num - 1]
            trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)

            p_test_f = train_and_predict_test_fold(
                fold_idx=fold_num,
                trn_idx=trn_idx,
                train_conv=train_conv,
                train_text=train_text,
                train_targets=train_targets,
                test_conv=test_conv,
                test_text=test_text,
                n_classes=n_classes,
                device=device,
                epochs=15,
                batch_size=32,
                lr=1e-3
            )
            test_probs_folds[fold_num - 1] = p_test_f
            print(f"  Fold {fold_num}/5 complete ({time.time() - t0:.1f}s)")

        p_conv_test = test_probs_folds.mean(axis=0)
        np.save(cached_conv_test_path, p_conv_test)
        print(f"\n[Saved] test_convnext_448_question_fusion.npy (shape: {p_conv_test.shape})")

    # 4. Reconstruct S550 Champion Test Probabilities
    print("\nReconstructing S550 champion test probability matrix...")
    p_s550_test, q_allowed = generate_s550_test_probs(train_df, test_df, classes, none_idx)

    # 5. Continuous Probability Fusion (w=0.05)
    w_conv = 0.05
    p_fused = (1.0 - w_conv) * p_s550_test + w_conv * p_conv_test
    print(f"Fused probabilities at w_conv={w_conv:.2f} (Shape: {p_fused.shape})")

    # 6. Ranking with beta=0.82 discounting
    beta = 0.82
    cand = p_fused.copy()
    row_idx = np.arange(n_test)
    r1 = np.argmax(cand, axis=1)
    cand[row_idx, r1] = -1.0
    cand[r1 != none_idx, none_idx] *= beta
    top2 = np.argsort(-cand, axis=1)[:, :2]

    fused_preds: List[str] = [
        f"{classes[r1[i]]} {classes[top2[i, 0]]} {classes[top2[i, 1]]}"
        for i in range(n_test)
    ]

    # 7. Precision Gating (tau=0.78, eps=0.04)
    p_bin_error = np.load(ART_DIR / "test_compact_binary_prob.npy")
    tau = 0.78
    eps = 0.04
    gated_count = 0

    for i in range(n_test):
        pred_tokens = fused_preds[i].split()
        if pred_tokens[0] == "NONE" and p_bin_error[i] >= tau:
            row_p = p_fused[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            p_misc_val = row_p[top_misc_idx]
            gap = p_none_val - p_misc_val

            if 0 < gap < eps:
                qid = test_df.iloc[i]["question_id"]
                if top_misc_code in q_allowed.get(qid, set()):
                    row_p[top_misc_idx] = -1.0
                    second_misc_idx = int(np.argmax(row_p))
                    second_misc_code = classes[second_misc_idx]
                    fused_preds[i] = f"{top_misc_code} NONE {second_misc_code}"
                    gated_count += 1

    print(f"Precision Gating: Flipped {gated_count} rows with P(error) >= {tau:.2f} and gap < {eps:.2f}")

    # 8. Divergence Audit vs Active Champion S550
    s550_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
    s550_preds = s550_sub["root_code"].tolist()

    r1_s550 = [p.split()[0] for p in s550_preds]
    r1_s559 = [p.split()[0] for p in fused_preds]
    r1_diffs = sum(1 for a, b in zip(r1_s550, r1_s559) if a != b)

    total_diffs = sum(1 for a, b in zip(s550_preds, fused_preds) if a != b)
    print("\n" + "=" * 80)
    print("DIVERGENCE AUDIT: S559 vs ACTIVE CHAMPION S550")
    print("=" * 80)
    print(f"Total Test Traces: {n_test}")
    print(f"Rank-1 Divergences: {r1_diffs} ({r1_diffs / n_test:.2%})")
    print(f"Total Ranking Divergences (Rank 1-3): {total_diffs} ({total_diffs / n_test:.2%})")

    diff_indices = [i for i, (a, b) in enumerate(zip(s550_preds, fused_preds)) if a != b]
    for idx in diff_indices[:8]:
        print(f"  Trace {test_df.iloc[idx]['trace_id']} (Q: {test_df.iloc[idx]['question_id']}):")
        print(f"    S550: {s550_preds[idx]}")
        print(f"    S559: {fused_preds[idx]}")

    # 9. Save and Validate Submission File
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": fused_preds})
    sub_df.to_csv(SUB_FILE, index=False)
    print(f"\n[Saved Submission] {SUB_FILE.name}")

    val_res = validate_submission_file(SUB_FILE)
    if not val_res["valid"]:
        raise ValueError(f"Submission validation failed: {val_res['errors']}")
    print(f"[Validation Passed] 3,498 rows, taxonomy compliant, non-empty rankings.")

    try:
        shutil.copy(SUB_FILE, DOWNLOADS_DIR / SUB_NAME)
        print(f"[Copied to Downloads] {DOWNLOADS_DIR / SUB_NAME}")
    except Exception as e:
        print(f"[Notice] Copy to downloads skipped: {e}")

    # 10. Kaggle API Authentication & Submission
    print("\n" + "=" * 80)
    print("AUTHENTICATING KAGGLE API & SUBMITTING S559")
    print("=" * 80)
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_API_TOKEN"] = token
    os.environ["KAGGLE_USERNAME"] = "Swastik.T"

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated successfully.")

    print(f"Submitting {SUB_NAME} to {COMPETITION}...")
    submit_result = api.competition_submit(
        file_name=str(SUB_FILE),
        message=MESSAGE,
        competition=COMPETITION
    )
    print(f"Submission Response: {submit_result}")

    # 11. Monitor Submission Scoring
    print("\nMonitoring Kaggle submission status...")
    score_recorded = None
    for attempt in range(1, 35):
        time.sleep(10)
        submissions = api.competition_submissions(COMPETITION)
        if submissions:
            latest = submissions[0]
            fname = getattr(latest, "fileName", "")
            status = getattr(latest, "status", "")
            score = getattr(latest, "publicScore", None)
            status_str = str(status).upper()

            print(f"[{attempt:02d} | {time.strftime('%H:%M:%S')}] File: {fname} | Status: {status} | Score: {score}")

            if "COMPLETE" in status_str:
                score_recorded = score
                print("\n" + "=" * 80)
                print("SUBMISSION COMPLETE!")
                print(f"Candidate: {fname}")
                print(f"Public Score: {score}")
                print(f"Active Champion Reference: 0.55049")
                if score is not None:
                    try:
                        sc_f = float(score)
                        delta = sc_f - 0.55049
                        if delta > 0:
                            print(f"Result: NEW ALL-TIME BEST LEADERBOARD PEAK! (+{delta:.5f})")
                        elif delta == 0:
                            print(f"Result: TIED LEADERBOARD PEAK (0.55049)")
                        else:
                            print(f"Result: Below Champion ({delta:+.5f})")
                    except (ValueError, TypeError):
                        pass
                print("=" * 80)
                break
            elif "ERROR" in status_str or "FAIL" in status_str:
                print(f"[Error] Submission failed: {latest}")
                break

if __name__ == "__main__":
    main()

