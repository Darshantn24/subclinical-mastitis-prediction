import numpy as np
from src.evaluation.evaluate import evaluate_model


class DummyModel:
    def predict(self, X):
        return np.array([0, 0, 1, 1])

    def predict_proba(self, X):
        return np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])


def test_evaluate_model():
    X_test = np.array([[1], [2], [3], [4]])
    y_test = np.array([0, 0, 1, 1])

    results = evaluate_model(DummyModel(), X_test, y_test)

    assert results["accuracy"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["roc_auc"] == 1.0
    assert "confusion_matrix" in results
    assert "classification_report" in results