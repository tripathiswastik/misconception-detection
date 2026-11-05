"""Unit tests for ConditionedQuestionPriorEngine."""
import numpy as np
import pandas as pd
import pytest

from src.diagnoser.conditioned_prior import ConditionedQuestionPriorEngine


def test_conditioned_prior_fit_and_soft_mixture():
    classes = ["NONE", "ERR_A", "ERR_B"]
    
    # Synthetic training data
    df_train = pd.DataFrame({
        "question_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
        "topic_name": ["algebra", "algebra", "algebra", "geometry", "geometry", "geometry"],
        "grade": [8, 8, 8, 8, 8, 8],
        "final_answer_correct": [True, True, False, False, False, True],
        "root_code": ["NONE", "NONE", "ERR_A", "ERR_B", "ERR_A", "NONE"]
    })

    engine = ConditionedQuestionPriorEngine(classes=classes, smoothing_question=2.0)
    engine.fit(df_train)

    df_test = pd.DataFrame({
        "question_id": ["q1", "q2"],
        "topic_name": ["algebra", "geometry"],
        "grade": [8, 8]
    })

    # Test with p=1.0 (verified correct)
    probs_correct = engine.predict_proba(df_test, verifier_probs=[1.0, 1.0])
    none_idx = classes.index("NONE")
    # For verified correct students, NONE should be dominant
    assert probs_correct[0, none_idx] > probs_correct[0, classes.index("ERR_A")]

    # Test with p=0.0 (verified incorrect)
    probs_incorrect = engine.predict_proba(df_test, verifier_probs=[0.0, 0.0])
    # For verified incorrect students, error code should have higher mass than NONE
    assert probs_incorrect[0, none_idx] < probs_correct[0, none_idx]

    # Test safety fallback: ABSTAIN / None should smoothly blend
    probs_abstain = engine.predict_proba(df_test, verifier_probs=[None, None])
    assert np.allclose(np.sum(probs_abstain, axis=1), 1.0)
