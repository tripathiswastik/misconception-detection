"""Adaptive Question-Level Empirical Prior and Hierarchical Target Encoder.

Implements Prior Smoothing Experiments P1 - P4:
- P1: Global shrinkage sweep (m_q in {4, 6, 8, 10, 12, 14, 16})
- P2: Adaptive sample-size shrinkage m_q(n_q) based on training examples per question
- P3: Semantic question-similarity backoff for unseen questions using 384-d MiniLM vectors
- P4: Hierarchical topic-grade smoothing sweep (m_tg in {1, 3, 5, 8, 10})

Strictly leak-free: fit exclusively on each fold's training partition.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
import pandas as pd


class AdaptiveQuestionPriorEngine:
    """Strictly in-fold Hierarchical Target Encoder with Adaptive and Semantic Smoothing."""

    def __init__(
        self,
        classes: Sequence[str],
        m_q_base: float = 10.0,
        m_tg: float = 5.0,
        adaptive_nq: bool = False,
        adaptive_power: float = 0.5,
        use_semantic_backoff: bool = False,
        semantic_embeddings: Optional[np.ndarray] = None,
        semantic_top_k: int = 3,
        semantic_threshold: float = 0.70,
        semantic_weight: float = 0.50
    ):
        self.classes = list(classes)
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        self.m_q_base = m_q_base
        self.m_tg = m_tg
        self.adaptive_nq = adaptive_nq
        self.adaptive_power = adaptive_power
        self.use_semantic_backoff = use_semantic_backoff
        self.semantic_embeddings = semantic_embeddings
        self.semantic_top_k = semantic_top_k
        self.semantic_threshold = semantic_threshold
        self.semantic_weight = semantic_weight

        # Fitted stats
        self.global_prior: np.ndarray = np.ones(self.n_classes, dtype=np.float32) / self.n_classes
        self.topic_grade_priors: Dict[Tuple[str, int], np.ndarray] = {}
        self.question_priors: Dict[str, np.ndarray] = {}
        self.train_qids: List[str] = []
        self.train_q_embeddings: Optional[np.ndarray] = None
        self.train_q_priors: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def compute_mq(self, n_q: int) -> float:
        """Compute effective shrinkage m_q for a question with n_q training traces."""
        if not self.adaptive_nq:
            return self.m_q_base
        # Adaptive: scale down shrinkage for high-sample questions (more data = trust counts more)
        # For n_q=1: m_q = m_0. For n_q=9: m_q = m_0 / 3 (if power=0.5).
        return float(self.m_q_base / (max(1.0, float(n_q)) ** self.adaptive_power))

    def fit(
        self,
        fold_train_df: pd.DataFrame,
        fold_train_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        target_col: str = "root_code"
    ) -> "AdaptiveQuestionPriorEngine":
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
        self.train_qids.clear()
        q_grouped = fold_train_df.groupby("question_id")
        
        q_priors_list = []
        q_embeds_list = []

        for q_id, group in q_grouped:
            q_id_str = str(q_id)
            self.train_qids.append(q_id_str)

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
            effective_mq = self.compute_mq(n_q)
            smoothed_q = (q_counts + effective_mq * tg_prior) / (n_q + effective_mq)
            self.question_priors[q_id_str] = smoothed_q
            q_priors_list.append(smoothed_q)

            # Record representative embedding for question
            if self.use_semantic_backoff and self.semantic_embeddings is not None and fold_train_indices is not None:
                first_local_idx = int(cast(Any, group.index[0]))
                global_idx = int(fold_train_indices[first_local_idx])
                q_embeds_list.append(self.semantic_embeddings[global_idx])

        if self.use_semantic_backoff and q_embeds_list:
            self.train_q_embeddings = np.vstack(q_embeds_list)
            self.train_q_priors = np.vstack(q_priors_list)

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        df: pd.DataFrame,
        eval_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        eval_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict prior probability matrix for rows in df."""
        if not self.is_fitted:
            raise RuntimeError("AdaptiveQuestionPriorEngine must be fitted before predict_proba.")

        n_samples = len(df)
        probs = np.zeros((n_samples, self.n_classes), dtype=np.float32)

        for i, (q_id, topic, raw_grade) in enumerate(zip(df["question_id"], df["topic_name"], df["grade"])):
            q_id_str = str(q_id)
            try:
                grade_int = int(float(raw_grade))
            except (ValueError, TypeError):
                grade_int = -1

            tg_key = (str(topic), grade_int)

            if q_id_str in self.question_priors:
                # Level 1: Question known in training split
                probs[i] = self.question_priors[q_id_str]
            else:
                # Unseen question: check for semantic nearest-neighbor backoff (P3)
                base_tg = self.topic_grade_priors.get(tg_key, self.global_prior)
                
                if (
                    self.use_semantic_backoff
                    and self.train_q_embeddings is not None
                    and self.train_q_priors is not None
                    and eval_embeddings is not None
                ):
                    q_emb = eval_embeddings[i]
                    # Cosine similarities against all training questions
                    sims = np.dot(self.train_q_embeddings, q_emb)
                    top_k_indices = np.argsort(-sims)[:self.semantic_top_k]
                    best_sim = float(sims[top_k_indices[0]])

                    if best_sim >= self.semantic_threshold:
                        top_sims = sims[top_k_indices]
                        exp_weights = np.exp((top_sims - best_sim) / 0.1)
                        norm_weights = exp_weights / np.sum(exp_weights)
                        sem_prior = np.dot(norm_weights, self.train_q_priors[top_k_indices])
                        
                        # Blend semantic prior with topic-grade backoff
                        probs[i] = self.semantic_weight * sem_prior + (1.0 - self.semantic_weight) * base_tg
                    else:
                        probs[i] = base_tg
                else:
                    # Level 2: Topic + Grade backoff
                    probs[i] = base_tg

        # Row normalize
        r_sums = probs.sum(axis=1, keepdims=True)
        return np.divide(probs, r_sums, where=r_sums > 0)
