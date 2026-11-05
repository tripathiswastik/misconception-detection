"""Baseline Text Diagnoser: Predicts root misconception codes from question context & student transcription.

Establishes the text-reasoning upper bound (Oracle ceiling) using gold transcription steps.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.diagnoser.math_parser import build_enhanced_feature_text
from src.metrics.map3 import calculate_map3


def build_feature_text(row: pd.Series) -> str:
    """Compose structured text payload from question, math pattern tokens, student steps, and key."""
    return build_enhanced_feature_text(row)


class TextMisconceptionDiagnoser:
    """TF-IDF and Regularized Linear Classifier for Math Misconception Diagnosis."""

    def __init__(
        self,
        taxonomy_manager: Optional[TaxonomyManager] = None,
        max_features_word: int = 8000,
        max_features_char: int = 12000,
        alpha: float = 1e-4,
        random_state: int = 42
    ):
        self.taxonomy = taxonomy_manager
        self.random_state = random_state
        self.alpha = alpha
        self.classes_: np.ndarray = np.array([])
        
        # Build text feature extractor (word n-grams + char n-grams for math syntax)
        self.pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word_tfidf", TfidfVectorizer(
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    max_features=max_features_word,
                    token_pattern=r"(?u)\b\w+\b|[+\-*/=^><\\]"
                )),
                ("char_tfidf", TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char",
                    sublinear_tf=True,
                    max_features=max_features_char
                ))
            ])),
            ("clf", SGDClassifier(
                loss="log",
                alpha=self.alpha,
                penalty="l2",
                max_iter=500,
                class_weight="balanced",
                random_state=self.random_state
            ))
        ])

    def fit(self, train_df: pd.DataFrame) -> "TextMisconceptionDiagnoser":
        """Train model on dataframe containing question and transcription columns."""
        texts = [build_feature_text(row) for _, row in train_df.iterrows()]
        y = train_df["root_code"].astype(str).values

        self.pipeline.fit(texts, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class probability distribution."""
        texts = [build_feature_text(row) for _, row in df.iterrows()]
        return self.pipeline.predict_proba(texts)

    def predict_top_k(self, df: pd.DataFrame, k: int = 3) -> List[List[str]]:
        """Predict top-k candidate taxonomy codes for each sample."""
        probs = self.predict_proba(df)
        preds = []
        for prob_row in probs:
            top_indices = np.argsort(prob_row)[::-1][:k]
            preds.append([str(self.classes_[idx]) for idx in top_indices])
        return preds

    def evaluate_cv(
        self,
        train_df: pd.DataFrame,
        n_splits: int = 5
    ) -> Tuple[float, List[List[str]], np.ndarray]:
        """Perform Out-Of-Fold Cross Validation and calculate MAP@3 score.
        
        Returns:
            Tuple of (overall_map3, oof_predictions, oof_probabilities)
        """
        y_true = train_df["root_code"].astype(str).values
        n_samples = len(train_df)

        # Handle class rarity for stratification
        class_counts = pd.Series(y_true).value_counts()
        strat_key = y_true.copy()
        for c, count in class_counts.items():
            if count < n_splits:
                strat_key[strat_key == c] = "RARE_CLASS"

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_preds: List[Optional[List[str]]] = [None] * n_samples

        print(f"[BaselineDiagnoser] Running {n_splits}-Fold Stratified Cross Validation...")

        all_classes = np.array(sorted(list(set(y_true))))
        oof_prob_matrix = np.zeros((n_samples, len(all_classes)))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, strat_key), start=1):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]

            fold_model = TextMisconceptionDiagnoser(
                taxonomy_manager=self.taxonomy,
                alpha=self.alpha,
                random_state=self.random_state + fold
            )
            fold_model.fit(fold_train)
            val_probs = fold_model.predict_proba(fold_val)

            val_preds_top3 = []
            for i, local_val_idx in enumerate(val_idx):
                prob_row = val_probs[i]
                top_indices = np.argsort(prob_row)[::-1][:3]
                preds_3 = [str(fold_model.classes_[idx]) for idx in top_indices]
                oof_preds[local_val_idx] = preds_3
                val_preds_top3.append(preds_3)

                for local_cls_idx, prob in enumerate(prob_row):
                    cls_name = fold_model.classes_[local_cls_idx]
                    if cls_name in class_to_idx:
                        oof_prob_matrix[local_val_idx, class_to_idx[cls_name]] = prob

            fold_map3 = calculate_map3(y_true[val_idx], val_preds_top3, k=3)
            fold_scores.append(fold_map3)
            print(f"  Fold {fold} MAP@3: {fold_map3:.4f}")

        overall_map3 = calculate_map3(y_true, [p for p in oof_preds if p is not None], k=3)
        print(f"[BaselineDiagnoser] Mean Fold MAP@3: {np.mean(fold_scores):.4f} | Overall OOF MAP@3: {overall_map3:.4f}")
        return overall_map3, [p for p in oof_preds if p is not None], oof_prob_matrix


def run_baseline_training(
    data_dir: Path,
    output_submission_path: Optional[Path] = None
) -> float:
    """Train baseline text diagnoser on real dataset, evaluate CV, and generate predictions."""
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.csv at: {train_path}")

    train_df = cast(pd.DataFrame, pd.read_csv(train_path))
    print(f"[BaselineDiagnoser] Loaded {len(train_df)} training samples.")

    diagnoser = TextMisconceptionDiagnoser()
    cv_score, oof_preds, _ = diagnoser.evaluate_cv(train_df, n_splits=5)

    # Train on full dataset
    print("[BaselineDiagnoser] Training final model on 100% of training data...")
    diagnoser.fit(train_df)

    # If test.csv exists and submission path requested, generate submission
    test_path = data_dir / "test.csv"
    if output_submission_path and test_path.exists():
        test_df = cast(pd.DataFrame, pd.read_csv(test_path))
        print(f"[BaselineDiagnoser] Generating predictions for {len(test_df)} test samples...")
        
        # Note: at test time, transcription_steps may be empty if OCR has not run yet.
        # This produces the question-prior / topic-prior prediction.
        test_preds = diagnoser.predict_top_k(test_df, k=3)
        
        # Save submission
        sub_df = pd.DataFrame({
            "trace_id": test_df["trace_id"],
            "root_code": [" ".join(p) for p in test_preds]
        })
        output_submission_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(output_submission_path, index=False)
        print(f"[BaselineDiagnoser] Saved test predictions to: {output_submission_path}")

    return cv_score


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = project_dir / "data"
    sub_path = project_dir / "submissions" / "baseline_text_submission.csv"
    run_baseline_training(data_dir, output_submission_path=sub_path)
