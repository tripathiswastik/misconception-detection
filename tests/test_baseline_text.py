"""Unit tests for Baseline Text Diagnoser."""
import pandas as pd
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser, build_feature_text


def test_build_feature_text():
    row = pd.Series({
        "grade": 7,
        "topic_name": "fractions",
        "question_text": "Compute 1/2 + 1/3",
        "correct_answer": "5/6",
        "transcription_steps": "L1 | 1/2 + 1/3 = 2/5"
    })
    feature_str = build_feature_text(row)
    assert "GRADE_7" in feature_str
    assert "TOPIC_fractions" in feature_str
    assert "Compute 1/2 + 1/3" in feature_str
    assert "L1 | 1/2 + 1/3 = 2/5" in feature_str


def test_diagnoser_fit_and_predict():
    train_data = pd.DataFrame([
        {
            "grade": 6,
            "topic_name": "fractions",
            "question_text": "1/3 + 1/4",
            "correct_answer": "7/12",
            "transcription_steps": "L1 | 1/3 + 1/4 = 2/7",
            "root_code": "FRM-01"
        },
        {
            "grade": 6,
            "topic_name": "fractions",
            "question_text": "1/3 + 1/4",
            "correct_answer": "7/12",
            "transcription_steps": "L1 | 1/3 + 1/4 = 7/12",
            "root_code": "NONE"
        },
        {
            "grade": 7,
            "topic_name": "integers",
            "question_text": "-3 - 5",
            "correct_answer": "-8",
            "transcription_steps": "L1 | -3 - 5 = -2",
            "root_code": "INT-04"
        },
        {
            "grade": 7,
            "topic_name": "integers",
            "question_text": "-3 - 5",
            "correct_answer": "-8",
            "transcription_steps": "L1 | -3 - 5 = -8",
            "root_code": "NONE"
        }
    ] * 3)  # Duplicate rows so there are enough samples

    model = TextMisconceptionDiagnoser(max_features_word=100, max_features_char=100)
    model.fit(train_data)

    test_data = pd.DataFrame([
        {
            "grade": 6,
            "topic_name": "fractions",
            "question_text": "1/5 + 1/3",
            "correct_answer": "8/15",
            "transcription_steps": "L1 | 1/5 + 1/3 = 2/8"
        }
    ])
    preds = model.predict_top_k(test_data, k=3)
    assert len(preds) == 1
    assert len(preds[0]) == 3
    assert all(isinstance(p, str) for p in preds[0])


def test_diagnoser_cv():
    train_data = pd.DataFrame([
        {"grade": 6, "topic_name": "fractions", "question_text": "1/3 + 1/4", "correct_answer": "7/12", "transcription_steps": "L1 | 2/7", "root_code": "FRM-01"},
        {"grade": 6, "topic_name": "fractions", "question_text": "1/3 + 1/4", "correct_answer": "7/12", "transcription_steps": "L1 | 7/12", "root_code": "NONE"},
        {"grade": 7, "topic_name": "integers", "question_text": "-3 - 5", "correct_answer": "-8", "transcription_steps": "L1 | -2", "root_code": "INT-04"},
        {"grade": 7, "topic_name": "integers", "question_text": "-3 - 5", "correct_answer": "-8", "transcription_steps": "L1 | -8", "root_code": "NONE"},
        {"grade": 8, "topic_name": "powers", "question_text": "2^3", "correct_answer": "8", "transcription_steps": "L1 | 6", "root_code": "EXP-01"},
        {"grade": 8, "topic_name": "powers", "question_text": "2^3", "correct_answer": "8", "transcription_steps": "L1 | 8", "root_code": "NONE"}
    ] * 4)

    model = TextMisconceptionDiagnoser(max_features_word=100, max_features_char=100)
    cv_score, oof_preds, _ = model.evaluate_cv(train_data, n_splits=3)
    assert 0.0 <= cv_score <= 1.0
    assert len(oof_preds) == len(train_data)
