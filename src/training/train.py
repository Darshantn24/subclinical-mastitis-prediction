from pathlib import Path
import joblib

from src.config import MODEL_DIR, MODEL_NAME, MODEL_VERSION


def save_model(model):
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.pkl"

    joblib.dump(model, model_path)

    return model_path

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model