"""Question-Level Empirical Prior and Smoothed Hierarchical Target Encoder.

Provides strictly in-fold, leak-free empirical distribution of root misconceptions:
1. Question-level prior P(c | question_id) smoothed with topic-grade prior.
2. Smooth hierarchical back-off:
   - If question has training samples: P(c | q) = (count(c, q) + m2 * P(c | topic, grade)) / (total(q) + m2)
   - If question is unseen in fold: backs off to P(c | topic, grade)
   - If (topic, grade) unseen: backs off to global class prior P(c)
3. Strictly fit on fold training split only (zero leakage to validation or test).
"""
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast, overload
from typing_extensions import Literal
import numpy as np
import pandas as pd


class QuestionPriorEngine:
    """Strictly in-fold Hierarchical Target Encoder for Math Questions."""

    def __init__(
        self,
        classes: Sequence[str],
        smoothing_question: float = 10.0,
        smoothing_topic_grade: float = 5.0
    ):
        self.classes = list(classes)
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        self.m_q = smoothing_question
        self.m_tg = smoothing_topic_grade

        # Fitted in-fold statistics
        self.global_prior: np.ndarray = np.ones(self.n_classes, dtype=np.float32) / self.n_classes
        self.topic_grade_priors: Dict[Tuple[str, int], np.ndarray] = {}
        self.question_priors: Dict[str, np.ndarray] = {}
        self.is_fitted: bool = False

    def fit(self, fold_train_df: pd.DataFrame, target_col: str = "root_code") -> "QuestionPriorEngine":
        """Fit empirical distributions strictly on training split."""
        # 1. Global prior
        class_counts = np.zeros(self.n_classes, dtype=np.float32)
        y_train = fold_train_df[target_col].astype(str).values
        for code in y_train:
            idx = self.class_to_idx.get(code, -1)
            if idx >= 0:
                class_counts[idx] += 1.0

        total_samples = len(y_train)
        self.global_prior = (class_counts + 1.0) / (total_samples + self.n_classes)

        # 2. Topic + Grade priors
        self.topic_grade_priors.clear()
        tg_grouped = fold_train_df.groupby(["topic_name", "grade"])
        for tg_key_raw, group in tg_grouped:
            tg_tuple = cast(Tuple[Any, Any], tg_key_raw)
            topic, grade = tg_tuple[0], tg_tuple[1]
            try:
                grade_int = int(float(grade))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to parse training grade '{grade}' as int: {e}") from e
            tg_key = (str(topic), grade_int)
            tg_counts = np.zeros(self.n_classes, dtype=np.float32)
            for code in group[target_col].astype(str):
                idx = self.class_to_idx.get(code, -1)
                if idx >= 0:
                    tg_counts[idx] += 1.0
            n_tg = len(group)
            smoothed_tg = (tg_counts + self.m_tg * self.global_prior) / (n_tg + self.m_tg)
            self.topic_grade_priors[tg_key] = smoothed_tg

        # 3. Question-level priors
        self.question_priors.clear()
        q_grouped = fold_train_df.groupby("question_id")
        for q_id, group in q_grouped:
            q_id_str = str(q_id)
            # Find representative topic & grade for this question
            topic = str(group["topic_name"].iloc[0])
            raw_grade = group["grade"].iloc[0]
            try:
                grade_int = int(float(raw_grade))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to parse question grade '{raw_grade}' as int: {e}") from e
            tg_prior = self.topic_grade_priors.get((topic, grade_int), self.global_prior)

            q_counts = np.zeros(self.n_classes, dtype=np.float32)
            for code in group[target_col].astype(str):
                idx = self.class_to_idx.get(code, -1)
                if idx >= 0:
                    q_counts[idx] += 1.0
            n_q = len(group)
            smoothed_q = (q_counts + self.m_q * tg_prior) / (n_q + self.m_q)
            self.question_priors[q_id_str] = smoothed_q

        self.is_fitted = True
        return self

    @overload
    def predict_proba(
        self, df: pd.DataFrame, return_levels: Literal[True] = True
    ) -> Tuple[np.ndarray, List[str]]:
        ...

    @overload
    def predict_proba(
        self, df: pd.DataFrame, return_levels: Literal[False]
    ) -> np.ndarray:
        ...

    def predict_proba(
        self, df: pd.DataFrame, return_levels: bool = True
    ) -> Union[Tuple[np.ndarray, List[str]], np.ndarray]:
        """Predict prior probability matrix for rows in df.

        Iterates efficiently over zip(df.question_id, df.topic_name, df.grade).
        Parses grade via int(float(grade_raw)) and raises ValueError on failure.
        Returns (probs, levels) where level is 'question', 'topic+grade', or 'global'.
        """
        if not self.is_fitted:
            raise RuntimeError("QuestionPriorEngine must be fitted before predict_proba.")

        n_rows = len(df)
        probs = np.zeros((n_rows, self.n_classes), dtype=np.float32)
        levels: List[str] = []

        for i, (q_id_raw, topic_raw, grade_raw) in enumerate(
            zip(df.question_id, df.topic_name, df.grade)
        ):
            q_id = str(q_id_raw)
            topic = str(topic_raw)
            try:
                grade = int(float(grade_raw))
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid grade value '{grade_raw}' at row index {i}: cannot parse as int(float())."
                ) from e

            if q_id in self.question_priors:
                # Level 1: Seen question in train fold
                probs[i] = self.question_priors[q_id]
                levels.append("question")
            elif (topic, grade) in self.topic_grade_priors:
                # Level 2: Unseen question, smooth backoff to (topic, grade)
                probs[i] = self.topic_grade_priors[(topic, grade)]
                levels.append("topic+grade")
            else:
                # Level 3: Unseen topic/grade, backoff to global prior
                probs[i] = self.global_prior
                levels.append("global")

        # Row normalize
        r_sums = probs.sum(axis=1, keepdims=True)
        norm_probs = np.divide(probs, r_sums, where=r_sums > 0)

        if return_levels:
            return norm_probs, levels
        return norm_probs
