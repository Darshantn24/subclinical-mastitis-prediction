import numpy as np
from sklearn.linear_model import LogisticRegression
from src.training.train import save_model

from src.training.train import train_model

from src.training.train import load_dataset

from src.training.train import prepare_features
from src.training.train import split_dataset
from src.training.train import validate_dataset
from src.training.train import create_preprocessor

from src.training.train import create_model

def test_prepare_features():
    df = load_dataset()
    X, y = prepare_features(df)

    assert "mastitis_label" not in X.columns
    assert "cow_id" not in X.columns
    assert len(X) == len(y)
    assert y.name == "mastitis_label"


def test_load_dataset():
    df = load_dataset()

    assert not df.empty
    assert "mastitis_label" in df.columns

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



def test_split_dataset():
    df = load_dataset()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) == 2000
    assert len(X_test) == 500
    assert len(X_train) + len(X_test) == len(df)
    assert set(y_train.unique()) == set(y.unique())
    assert set(y_test.unique()) == set(y.unique())

def test_validate_dataset():
    df = load_dataset()

    assert validate_dataset(df) is True


def test_create_preprocessor():
    df = load_dataset()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    preprocessor = create_preprocessor(X_train)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 1

def test_create_model():
    df = load_dataset()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    preprocessor = create_preprocessor(X_train)
    model = create_model(preprocessor)

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps    


def test_run_training(tmp_path, monkeypatch):
    from src.training import train

    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)

    model, results, model_path = train.run_training()

    assert model_path.exists()
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "roc_auc" in results
    assert hasattr(model, "predict")