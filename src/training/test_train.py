import numpy as np
from sklearn.linear_model import LogisticRegression
from src.training.train import save_model

from src.training.train import train_model


def test_train_model():
    X_train = np.array([[1], [2], [3], [4], [5], [6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    trained_model = train_model(model, X_train, y_train)

    assert hasattr(trained_model, "predict")
    assert len(trained_model.predict(X_train)) == 6

def test_save_model(tmp_path, monkeypatch):
    from src.training import train

    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)

    model = LogisticRegression()
    model_path = train.save_model(model)

    assert model_path.exists()
    assert model_path.name == "baseline_random_forest_1.0.0.pkl"