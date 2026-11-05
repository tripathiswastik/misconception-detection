"""Conditioned Question-Prior Engine (Soft Bayesian Mixture).

Blends question-level misconceptions conditioned on student answer correctness:
P(c | q, p) = p * P(c | q, correct=True) + (1 - p) * P(c | q, correct=False)

Key Properties:
1. When p = 1.0 (verified correct): Heavily concentrates on 'NONE' (94% empirical density).
2. When p = 0.0 (verified incorrect): Redistributes probability mass across specific error codes (98% empirical density).
3. When p = None or 0.5 (ABSTAIN / uncertain): Smoothly falls back to unconditioned prior P(c | q).
4. Strictly out-of-fold and leak-free: Fits separate empirical Dirichlet smoothing per fold.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
import pandas as pd

from src.diagnoser.question_prior import QuestionPriorEngine


class ConditionedQuestionPriorEngine:
    """Soft Bayesian Mixture Prior conditioned on student final-answer correctness."""

    def __init__(
        self,
        classes: Sequence[str],
        smoothing_question: float = 10.0,
        smoothing_topic_grade: float = 5.0
    ) -> None:
        self.classes = list(classes)
        self.n_classes = len(self.classes)
        self.m_q = smoothing_question
        self.m_tg = smoothing_topic_grade

        # Separate in-fold priors
        self.prior_unconditioned = QuestionPriorEngine(
            classes=self.classes, smoothing_question=self.m_q, smoothing_topic_grade=self.m_tg
        )
        self.prior_correct = QuestionPriorEngine(
            classes=self.classes, smoothing_question=self.m_q, smoothing_topic_grade=self.m_tg
        )
        self.prior_incorrect = QuestionPriorEngine(
            classes=self.classes, smoothing_question=self.m_q, smoothing_topic_grade=self.m_tg
        )
        self.is_fitted: bool = False

    def fit(self, fold_train_df: pd.DataFrame, target_col: str = "root_code") -> "ConditionedQuestionPriorEngine":
        """Fit empirical priors separately on correct and incorrect training subsets."""
        # 1. Unconditioned baseline prior
        self.prior_unconditioned.fit(fold_train_df, target_col=target_col)

        # 2. Partition by true final_answer_correct
        mask_corr = fold_train_df["final_answer_correct"].astype(bool) == True
        df_corr = fold_train_df[mask_corr]
        df_incorr = fold_train_df[~mask_corr]

        self.prior_correct.fit(df_corr, target_col=target_col)
        self.prior_incorrect.fit(df_incorr, target_col=target_col)

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        df: pd.DataFrame,
        verifier_probs: Optional[Sequence[Optional[float]]] = None,
        default_prob: float = 0.50
    ) -> np.ndarray:
        """Predict soft mixture probabilities.
        
        Args:
            df: Validation or test DataFrame containing question_id, topic_name, grade.
            verifier_probs: Sequence of float probabilities in [0.0, 1.0], or None for abstention.
            default_prob: Fallback probability for abstentions (default: 0.50).
            
        Returns:
            np.ndarray of shape (len(df), n_classes) with normalized probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("ConditionedQuestionPriorEngine must be fitted before predict_proba.")

        n_rows = len(df)
        p_corr = self.prior_correct.predict_proba(df, return_levels=False)
        p_incorr = self.prior_incorrect.predict_proba(df, return_levels=False)

        # If no verifier probabilities provided, return unconditioned prior
        if verifier_probs is None:
            return self.prior_unconditioned.predict_proba(df, return_levels=False)

        mixed_probs = np.zeros((n_rows, self.n_classes), dtype=np.float32)

        for i in range(n_rows):
            p = verifier_probs[i]
            # Safety fallback on ABSTAIN or None
            if p is None or (isinstance(p, float) and np.isnan(p)):
                p = default_prob

            # Clip probability between 0.0 and 1.0
            p_val = float(np.clip(p, 0.0, 1.0))

            # Convex mixture
            mixed_probs[i] = p_val * p_corr[i] + (1.0 - p_val) * p_incorr[i]

            # Normalize row
            row_sum = np.sum(mixed_probs[i])
            if row_sum > 0:
                mixed_probs[i] /= row_sum

        return mixed_probs
