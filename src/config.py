from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Model configuration
MODEL_NAME = "baseline_random_forest"
MODEL_VERSION = "1.0.0"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20