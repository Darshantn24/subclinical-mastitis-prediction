from pathlib import Path
import joblib
import pandas as pd
from src.config import RAW_DATA_DIR
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE
from src.config import MODEL_DIR, MODEL_NAME, MODEL_VERSION

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from src.evaluation.evaluate import evaluate_model


def load_dataset():
    data_path = RAW_DATA_DIR / "raw_mastitis_dataset.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    validate_dataset(df)

    return df


def prepare_features(df):
    target_column = "mastitis_label"

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X = df.drop(columns=[target_column, "cow_id"])
    y = df[target_column]

    return X, y

def save_model(model):
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.pkl"

    joblib.dump(model, model_path)

    return model_path

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

def validate_dataset(df):
    required_columns = [
        "cow_id", "parity", "days_in_milk",
        "somatic_cell_count", "milk_conductivity",
        "milk_yield", "milk_pH", "udder_skin_temp",
        "rectal_temp", "activity_steps", "heart_rate",
        "ambient_temp", "ambient_humidity", "mastitis_label"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().sum().any():
        raise ValueError("Dataset contains missing values.")

    if not set(df["mastitis_label"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    return True



def create_preprocessor(X_train):
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def create_model(preprocessor):
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return model_pipeline

def run_training():
    df = load_dataset()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    preprocessor = create_preprocessor(X_train)

    model = create_model(preprocessor)

    trained_model = train_model(model, X_train, y_train)

    results = evaluate_model(trained_model, X_test, y_test)

    model_path = save_model(trained_model)

    return trained_model, results, model_path