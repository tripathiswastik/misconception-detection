"""Multi-Model Cross-Validation and Benchmarking Suite.

Provides leakage-free Cross-Validation (Stratified and Question-Grouped)
with exact Top-3 Mean Average Precision (MAP@3) computation and rich diagnostics:
- Overall MAP@3 and fold-by-fold scores
- Rank-1, Rank-2, Rank-3, and Missed Top-3 counts
- Non-NONE error breakdown and NONE-saturation analysis (Rank 1 == NONE on error traces)
- Decision guidance: Stage A (Base Representation) vs Stage B (Reranking & Calibration)
- Slices by question seen/unseen, topic_name, and grade
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.config import PATHS
from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.metrics.map3 import calculate_map3 as compute_map3
from src.utils.device import clear_gpu_memory, get_device


@dataclass
class DiagnosticBreakdown:
    """Detailed diagnostic breakdown of ranking quality and error modes."""
    total_samples: int
    rank1_count: int
    rank1_pct: float
    rank2_count: int
    rank2_pct: float
    rank3_count: int
    rank3_pct: float
    missed_count: int
    missed_pct: float

    # Error rows specific breakdown (where true label != NONE)
    error_total: int
    error_rank1: int
    error_rank2: int
    error_rank3: int
    error_missed: int
    error_rank1_is_none: int

    # Slices
    none_map3: float
    error_map3: float
    seen_q_map3: Optional[float] = None
    unseen_q_map3: Optional[float] = None
    topic_scores: Dict[str, float] = field(default_factory=dict)
    grade_scores: Dict[int, float] = field(default_factory=dict)

    def decision_summary(self) -> str:
        """Clear guidance on Stage A vs Stage B optimization."""
        stage_b_targets = self.error_rank2 + self.error_rank3
        stage_a_targets = self.error_missed
        lines = [
            f"  [Optimization Decision Guidance]",
            f"  - Stage B (Reranking/Calibration) Target: {stage_b_targets} error rows already in Top 3 (Rank 2 or 3).",
            f"    * Of these, {self.error_rank1_is_none} rows have NONE sitting at Rank 1.",
            f"    * Re-ranking these to Rank 1 yields up to +{(self.error_rank2 * 0.5 + self.error_rank3 * 0.6667) / max(1, self.total_samples):.4f} MAP@3 headroom.",
            f"  - Stage A (Base Feature/Model) Target: {stage_a_targets} error rows completely outside Top 3.",
            f"    * Requires improved visual stroke encoders, better handwriting OCR, or finer semantic embeddings."
        ]
        return "\n".join(lines)


@dataclass
class CrossValidationResult:
    """Out-of-fold evaluation metrics and predictions across cross-validation splits."""
    model_name: str
    fold_scores: List[float]
    mean_map3: float
    std_map3: float
    overall_oof_map3: float
    oof_predictions: List[List[str]]
    oof_probabilities: np.ndarray
    classes: List[str]
    diagnostics: Optional[DiagnosticBreakdown] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        folds_str = ", ".join([f"F{i+1}: {score:.4f}" for i, score in enumerate(self.fold_scores)])
        return (
            f"[{self.model_name}] OOF MAP@3: {self.overall_oof_map3:.4f} "
            f"(Mean: {self.mean_map3:.4f} +/- {self.std_map3:.4f}) | {folds_str}"
        )

    def diagnostic_report(self) -> str:
        """Format full diagnostic breakdown report."""
        if self.diagnostics is None:
            return self.summary()
        d = self.diagnostics
        lines = [
            "=" * 75,
            f"DIAGNOSTIC REPORT: {self.model_name}",
            "=" * 75,
            f"  Overall OOF MAP@3: {self.overall_oof_map3:.5f} (Mean: {self.mean_map3:.5f} +/- {self.std_map3:.5f})",
            f"  Rank Distribution (All {d.total_samples} samples):",
            f"    - Rank 1 hits (1.00):  {d.rank1_count:4d} ({d.rank1_pct:5.1f}%)",
            f"    - Rank 2 hits (0.50):  {d.rank2_count:4d} ({d.rank2_pct:5.1f}%)",
            f"    - Rank 3 hits (0.33):  {d.rank3_count:4d} ({d.rank3_pct:5.1f}%)",
            f"    - Missed Top-3 (0.00): {d.missed_count:4d} ({d.missed_pct:5.1f}%)",
            "",
            f"  Error Trace Analysis ({d.error_total} true non-NONE misconception rows):",
            f"    - True label at Rank 1: {d.error_rank1:4d} ({d.error_rank1 / max(1, d.error_total) * 100:.1f}%)",
            f"    - True label at Rank 2: {d.error_rank2:4d} ({d.error_rank2 / max(1, d.error_total) * 100:.1f}%)",
            f"    - True label at Rank 3: {d.error_rank3:4d} ({d.error_rank3 / max(1, d.error_total) * 100:.1f}%)",
            f"    - True label Missed:    {d.error_missed:4d} ({d.error_missed / max(1, d.error_total) * 100:.1f}%)",
            f"    - Incorrect Rank-1 NONE: {d.error_rank1_is_none:4d} ({d.error_rank1_is_none / max(1, d.error_total) * 100:.1f}%) [NONE Saturation]",
            "",
            f"  Category Slices:",
            f"    - NONE Rows MAP@3:      {d.none_map3:.5f}",
            f"    - Error Rows MAP@3:     {d.error_map3:.5f}",
        ]
        if d.seen_q_map3 is not None and d.unseen_q_map3 is not None:
            lines.extend([
                f"    - Seen Questions MAP@3:   {d.seen_q_map3:.5f}",
                f"    - Unseen Questions MAP@3: {d.unseen_q_map3:.5f}"
            ])
        lines.append("")
        lines.append(d.decision_summary())
        lines.append("=" * 75)
        return "\n".join(lines)


class ModelCrossValidator:
    """Leakage-free Cross-Validator supporting Stratified and Question-Grouped CV."""

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        taxonomy: Optional[TaxonomyManager] = None,
        use_canonical_splits: bool = True,
        grouped_by: Optional[str] = None
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.taxonomy = taxonomy or TaxonomyManager()
        self.use_canonical_splits = use_canonical_splits
        self.grouped_by = grouped_by

    def create_folds(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate fold partitions (canonical, group-disjoint, or stratified)."""
        # 1. Question-Grouped Cross-Validation (Guardrail for transductive leakage)
        if self.grouped_by and self.grouped_by in df.columns:
            gkf = GroupKFold(n_splits=self.n_splits)
            groups = df[self.grouped_by].values
            y_dummy = df["root_code"].astype(str).values if "root_code" in df.columns else np.zeros(len(df))
            return list(gkf.split(df, y_dummy, groups=groups))

        # 2. Canonical split loading from splits cache (data/splits.json)
        if self.use_canonical_splits and len(df) == 1500 and PATHS.splits_cache.exists():
            from src.data.splits import FoldSplitter
            splitter = FoldSplitter(n_splits=self.n_splits, random_state=self.random_state)
            return splitter.load_splits(PATHS.splits_cache)

        # 3. Dynamic stratified splits with rare class pooling
        return self.create_stratified_folds(df)

    def create_stratified_folds(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified indices pooling rare classes (< n_splits samples)."""
        y = df["root_code"].astype(str).values
        class_counts = pd.Series(y).value_counts()

        strat_key = y.copy()
        for label, count in class_counts.items():
            if count < self.n_splits:
                strat_key[strat_key == label] = "RARE_CLASS"

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return list(skf.split(df, strat_key))

    def compute_diagnostics(
        self,
        y_true: List[str],
        oof_predictions: List[List[str]],
        df: pd.DataFrame,
        folds: List[Tuple[np.ndarray, np.ndarray]]
    ) -> DiagnosticBreakdown:
        """Compute exhaustive rank distribution and sliced metrics."""
        n_samples = len(y_true)
        assert len(oof_predictions) == n_samples

        r1 = sum(1 for y, p in zip(y_true, oof_predictions) if y == p[0])
        r2 = sum(1 for y, p in zip(y_true, oof_predictions) if y == p[1])
        r3 = sum(1 for y, p in zip(y_true, oof_predictions) if y == p[2])
        miss = sum(1 for y, p in zip(y_true, oof_predictions) if y not in p[:3])

        # Error rows (y != NONE)
        err_indices = [i for i, y in enumerate(y_true) if y != "NONE"]
        err_total = len(err_indices)
        err_r1 = sum(1 for i in err_indices if y_true[i] == oof_predictions[i][0])
        err_r2 = sum(1 for i in err_indices if y_true[i] == oof_predictions[i][1])
        err_r3 = sum(1 for i in err_indices if y_true[i] == oof_predictions[i][2])
        err_miss = sum(1 for i in err_indices if y_true[i] not in oof_predictions[i][:3])
        err_r1_none = sum(1 for i in err_indices if oof_predictions[i][0] == "NONE")

        # Slices: NONE vs Error
        none_indices = [i for i, y in enumerate(y_true) if y == "NONE"]
        none_map3 = compute_map3(
            [y_true[i] for i in none_indices],
            [oof_predictions[i] for i in none_indices]
        ) if none_indices else 0.0

        error_map3 = compute_map3(
            [y_true[i] for i in err_indices],
            [oof_predictions[i] for i in err_indices]
        ) if err_indices else 0.0

        # Seen vs Unseen questions in fold training
        seen_q_map3 = None
        unseen_q_map3 = None
        if "question_id" in df.columns:
            seen_mask = np.zeros(n_samples, dtype=bool)
            for tr_idx, va_idx in folds:
                tr_questions = set(df.iloc[tr_idx]["question_id"].values)
                for idx in va_idx:
                    if df.iloc[idx]["question_id"] in tr_questions:
                        seen_mask[idx] = True

            seen_indices = [i for i in range(n_samples) if seen_mask[i]]
            unseen_indices = [i for i in range(n_samples) if not seen_mask[i]]

            if seen_indices:
                seen_q_map3 = compute_map3([y_true[i] for i in seen_indices], [oof_predictions[i] for i in seen_indices])
            if unseen_indices:
                unseen_q_map3 = compute_map3([y_true[i] for i in unseen_indices], [oof_predictions[i] for i in unseen_indices])

        # Slices by topic
        topic_scores = {}
        if "topic_name" in df.columns:
            for topic, group in df.groupby("topic_name"):
                idx_list = group.index.tolist()
                topic_scores[str(topic)] = compute_map3([y_true[i] for i in idx_list], [oof_predictions[i] for i in idx_list])

        # Slices by grade
        grade_scores = {}
        if "grade" in df.columns:
            for grade, group in df.groupby("grade"):
                idx_list = group.index.tolist()
                try:
                    g_int = int(float(grade))
                    grade_scores[g_int] = compute_map3([y_true[i] for i in idx_list], [oof_predictions[i] for i in idx_list])
                except (ValueError, TypeError):
                    pass

        return DiagnosticBreakdown(
            total_samples=n_samples,
            rank1_count=r1,
            rank1_pct=r1 / max(1, n_samples) * 100.0,
            rank2_count=r2,
            rank2_pct=r2 / max(1, n_samples) * 100.0,
            rank3_count=r3,
            rank3_pct=r3 / max(1, n_samples) * 100.0,
            missed_count=miss,
            missed_pct=miss / max(1, n_samples) * 100.0,
            error_total=err_total,
            error_rank1=err_r1,
            error_rank2=err_r2,
            error_rank3=err_r3,
            error_missed=err_miss,
            error_rank1_is_none=err_r1_none,
            none_map3=none_map3,
            error_map3=error_map3,
            seen_q_map3=seen_q_map3,
            unseen_q_map3=unseen_q_map3,
            topic_scores=topic_scores,
            grade_scores=grade_scores
        )

    def evaluate_probabilities(
        self,
        probs: np.ndarray,
        train_df: pd.DataFrame,
        model_name: str = "ProbabilityMatrix",
        classes: Optional[List[str]] = None
    ) -> CrossValidationResult:
        """Directly benchmark a precomputed out-of-fold probability matrix across the locked splits."""
        all_classes = classes or self.taxonomy.all_codes(include_none=True)
        folds = self.create_folds(train_df)
        y_true = [str(y) for y in train_df["root_code"].fillna("NONE").values]
        n_samples = len(train_df)

        # Fast argpartition top 3
        k = min(3, probs.shape[1])
        row_idx = np.arange(n_samples)[:, None]
        part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-probs[row_idx, part], axis=1, kind="mergesort")
        top3_indices = part[row_idx, order]

        oof_predictions = [[all_classes[idx] for idx in row] for row in top3_indices]

        # Fold scores
        fold_scores = []
        for fold_idx, (_, val_idx) in enumerate(folds, start=1):
            f_true = [y_true[i] for i in val_idx]
            f_preds = [oof_predictions[i] for i in val_idx]
            fold_scores.append(compute_map3(f_true, f_preds))

        overall_map3 = compute_map3(y_true, oof_predictions)
        mean_map3 = float(np.mean(fold_scores))
        std_map3 = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

        diagnostics = self.compute_diagnostics(y_true, oof_predictions, train_df, folds)

        return CrossValidationResult(
            model_name=model_name,
            fold_scores=fold_scores,
            mean_map3=mean_map3,
            std_map3=std_map3,
            overall_oof_map3=overall_map3,
            oof_predictions=oof_predictions,
            oof_probabilities=probs,
            classes=all_classes,
            diagnostics=diagnostics
        )

    def evaluate_model(
        self,
        model_factory: Callable[[], Any],
        train_df: pd.DataFrame,
        model_name: str = "Model",
        post_process_fn: Optional[Callable[[np.ndarray, List[str], pd.DataFrame], np.ndarray]] = None
    ) -> CrossValidationResult:
        """Run Cross-Validation on a candidate model architecture with full diagnostics."""
        folds = self.create_folds(train_df)
        n_samples = len(train_df)
        y_true = [str(y) for y in train_df["root_code"].fillna("NONE").values]

        all_classes = self.taxonomy.all_codes(include_none=True)
        oof_probabilities = np.zeros((n_samples, len(all_classes)), dtype=np.float32)
        oof_predictions: List[Optional[List[str]]] = [None] * n_samples
        fold_scores: List[float] = []

        print(f"\n[CrossValidator] Starting {self.n_splits}-Fold CV for '{model_name}' on device: {get_device(prefer_gpu=True)}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_train = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val = train_df.iloc[val_idx].reset_index(drop=True)

            model = model_factory()
            model.fit(fold_train)

            raw_val_probs = model.predict_proba(fold_val)
            model_classes = list(model.classes_)

            aligned_probs = np.zeros((len(val_idx), len(all_classes)), dtype=np.float32)
            for m_idx, class_name in enumerate(model_classes):
                if class_name in all_classes:
                    master_idx = all_classes.index(class_name)
                    aligned_probs[:, master_idx] = raw_val_probs[:, m_idx]

            row_sums = aligned_probs.sum(axis=1, keepdims=True)
            aligned_probs = np.divide(aligned_probs, row_sums, where=row_sums > 0)

            if post_process_fn is not None:
                aligned_probs = post_process_fn(aligned_probs, all_classes, fold_val)

            k = min(3, aligned_probs.shape[1])
            part = np.argpartition(-aligned_probs, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(len(val_idx))[:, None]
            order = np.argsort(-aligned_probs[row_idx, part], axis=1, kind="mergesort")
            top_indices = part[row_idx, order]

            fold_preds_top3 = []
            for i, global_idx in enumerate(val_idx):
                top_3 = [all_classes[idx] for idx in top_indices[i]]
                oof_predictions[global_idx] = top_3
                oof_probabilities[global_idx] = aligned_probs[i]
                fold_preds_top3.append(top_3)

            fold_y_true = [y_true[g_idx] for g_idx in val_idx]
            fold_map3 = compute_map3(fold_y_true, fold_preds_top3)
            fold_scores.append(fold_map3)
            print(f"  Fold {fold_idx}/{self.n_splits} MAP@3: {fold_map3:.4f}")

            clear_gpu_memory()

        valid_preds = [p for p in oof_predictions if p is not None]
        overall_oof_map3 = compute_map3(y_true, valid_preds)
        mean_map3 = float(np.mean(fold_scores))
        std_map3 = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

        diagnostics = self.compute_diagnostics(y_true, valid_preds, train_df, folds)

        result = CrossValidationResult(
            model_name=model_name,
            fold_scores=fold_scores,
            mean_map3=mean_map3,
            std_map3=std_map3,
            overall_oof_map3=overall_oof_map3,
            oof_predictions=valid_preds,
            oof_probabilities=oof_probabilities,
            classes=all_classes,
            diagnostics=diagnostics
        )
        print(f"[CrossValidator] {result.summary()}\n")
        return result

    @staticmethod
    def evaluate_ensemble(
        results: Sequence[CrossValidationResult],
        weights: Optional[Sequence[float]] = None,
        y_true: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[str]] = None
    ) -> Tuple[float, List[List[str]], np.ndarray]:
        """Compute ensemble OOF MAP@3 by weighted probability averaging."""
        n_models = len(results)
        if weights is None:
            w = np.ones(n_models) / n_models
        else:
            w = np.array(weights) / np.sum(weights)

        master_classes = classes or results[0].classes
        blended_probs = np.zeros_like(results[0].oof_probabilities)
        for i, res in enumerate(results):
            blended_probs += w[i] * res.oof_probabilities

        k = min(3, blended_probs.shape[1])
        row_idx = np.arange(len(blended_probs))[:, None]
        part = np.argpartition(-blended_probs, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-blended_probs[row_idx, part], axis=1, kind="mergesort")
        top3_indices = part[row_idx, order]

        blended_preds = [[master_classes[idx] for idx in row] for row in top3_indices]

        if y_true is not None:
            score = compute_map3(list(y_true), blended_preds)
        else:
            score = 0.0

        return score, blended_preds, blended_probs
