"""Unit tests for QuestionPriorEngine."""
import numpy as np
import pandas as pd
import pytest

from src.diagnoser.question_prior import QuestionPriorEngine


def test_question_prior_fit_and_levels():
    classes = ["C0", "C1", "C2", "NONE"]
    engine = QuestionPriorEngine(classes=classes, smoothing_question=1.0, smoothing_topic_grade=1.0)

    train_df = pd.DataFrame({
        "question_id": ["q1", "q1", "q2"],
        "topic_name": ["Algebra", "Algebra", "Geometry"],
        "grade": [8.0, "8.0", 7],
        "root_code": ["C0", "C0", "C1"]
    })

    engine.fit(train_df)
    assert engine.is_fitted
    assert ("Algebra", 8) in engine.topic_grade_priors
    assert ("Geometry", 7) in engine.topic_grade_priors
    assert "q1" in engine.question_priors
    assert "q2" in engine.question_priors

    # Test DataFrame with seen question, unseen question with seen topic+grade, and unseen topic
    eval_df = pd.DataFrame({
        "question_id": ["q1", "q3", "q4"],
        "topic_name": ["Algebra", "Algebra", "Calculus"],
        "grade": ["8.0", 8, "12.0"],
    })

    probs, levels = engine.predict_proba(eval_df)
    assert probs.shape == (3, 4)
    # Check row sums equal 1
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Check backoff levels
    assert levels == ["question", "topic+grade", "global"]

    # Level 1 (q1) should heavily favor C0
    assert np.argmax(probs[0]) == 0


def test_grade_parsing_failure_raises_value_error():
    classes = ["C0", "C1"]
    engine = QuestionPriorEngine(classes=classes)
    train_df = pd.DataFrame({
        "question_id": ["q1"],
        "topic_name": ["Algebra"],
        "grade": [8],
        "root_code": ["C0"]
    })
    engine.fit(train_df)

    bad_df = pd.DataFrame({
        "question_id": ["q1"],
        "topic_name": ["Algebra"],
        "grade": ["invalid_grade"]
    })

    with pytest.raises(ValueError, match="Invalid grade value"):
        engine.predict_proba(bad_df)


def test_return_levels_flag():
    classes = ["C0", "C1"]
    engine = QuestionPriorEngine(classes=classes)
    train_df = pd.DataFrame({
        "question_id": ["q1"],
        "topic_name": ["Algebra"],
        "grade": [8],
        "root_code": ["C0"]
    })
    engine.fit(train_df)

    eval_df = pd.DataFrame({
        "question_id": ["q1"],
        "topic_name": ["Algebra"],
        "grade": [8]
    })

    # return_levels=False returns only ndarray
    probs_only = engine.predict_proba(eval_df, return_levels=False)
    assert isinstance(probs_only, np.ndarray)
    assert probs_only.shape == (1, 2)
