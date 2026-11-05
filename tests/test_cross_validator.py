"""Unit tests for ModelCrossValidator and CrossValidationResult."""
import numpy as np
import pandas as pd
from src.evaluation.cross_validator import CrossValidationResult, ModelCrossValidator


class MockDiagnoser:
    """Mock diagnoser returning dummy probabilities."""
    def __init__(self, classes=None):
        self.classes_ = classes or ["FRM-01", "INT-04", "NONE"]

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        probs = np.zeros((n, len(self.classes_)))
        # Simple deterministic pattern based on grade
        for i, row in df.reset_index(drop=True).iterrows():
            raw_grade = row.get("grade")
            grade_val = int(raw_grade) if raw_grade is not None else 6
            if grade_val <= 6:
                probs[i, 0] = 0.7  # FRM-01
                probs[i, 1] = 0.2
                probs[i, 2] = 0.1
            else:
                probs[i, 0] = 0.1
                probs[i, 1] = 0.7  # INT-04
                probs[i, 2] = 0.2
        return probs


def test_stratified_folds_creation():
    df = pd.DataFrame({
        "trace_id": [f"t_{i}" for i in range(25)],
        "grade": [6] * 25,
        "topic_name": ["fractions"] * 25,
        "question_text": ["q"] * 25,
        "correct_answer": ["a"] * 25,
        "root_code": ["FRM-01"] * 10 + ["INT-04"] * 10 + ["RARE_01"] * 3 + ["RARE_02"] * 2
    })

    validator = ModelCrossValidator(n_splits=5, random_state=42)
    folds = validator.create_stratified_folds(df)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert len(train_idx) + len(val_idx) == 25
        assert len(set(train_idx).intersection(set(val_idx))) == 0


def test_evaluate_model_and_ensemble():
    df = pd.DataFrame([
        {"trace_id": f"t_{i}", "grade": 6 if i % 2 == 0 else 7, "topic_name": "math",
         "question_text": "q", "correct_answer": "ans", "root_code": "FRM-01" if i % 2 == 0 else "INT-04"}
        for i in range(30)
    ])

    validator = ModelCrossValidator(n_splits=5, random_state=42)
    res1 = validator.evaluate_model(lambda: MockDiagnoser(), df, model_name="Mock1")
    assert isinstance(res1, CrossValidationResult)
    assert len(res1.fold_scores) == 5
    assert 0.0 <= res1.overall_oof_map3 <= 1.0
    assert len(res1.oof_predictions) == 30

    res2 = validator.evaluate_model(lambda: MockDiagnoser(), df, model_name="Mock2")
    ensemble_score, preds, blended_probs = ModelCrossValidator.evaluate_ensemble(
        [res1, res2], weights=[0.5, 0.5], y_true=df["root_code"].values, classes=res1.classes
    )
    assert 0.0 <= ensemble_score <= 1.0
    assert len(preds) == 30
    assert blended_probs.shape == res1.oof_probabilities.shape
