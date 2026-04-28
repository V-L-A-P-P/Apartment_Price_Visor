from __future__ import annotations

from pathlib import Path

# ---------- Paths ----------
DEFAULT_CLEAN_PATH = Path("data/processed/move_ru/listings_clean.parquet")
DEFAULT_TRAIN_DIR = Path("data/processed/train")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

CATBOOST_TRAIN_DATASET_NAME = "train_tabular_catboost.parquet"
BASELINE_TRAIN_DATASET_NAME = "train_tabular_baseline.parquet"
DATASET_META_NAME = "dataset_meta.json"

DEFAULT_CATBOOST_MODEL_PATH = Path("artifacts/models/catboost_model.cbm")
DEFAULT_PREDICTIONS_PATH = Path("artifacts/predictions/catboost_predictions.parquet")

# ---------- ML configs ----------
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# CATBOOST_TRAIN_PARAMS = {
#     "loss_function": "MAE",
#     "eval_metric": "MAE",
#     "depth": 3,
#     "learning_rate": 0.05,
#     "n_estimators": 500,
#     "random_seed": 42,
#     "verbose": 200,
# }

CATBOOST_TRAIN_PARAMS = {'iterations': 900, 'learning_rate': 0.04497205567975632, 'depth': 4, 'l2_leaf_reg': 5.218859878694623e-08, 'min_data_in_leaf': 32}

# ---------- Shared features ----------
DESCRIPTION_COL = "description"
DESCRIPTION_EMBEDDING_COL = "description_embedding"
CATBOOST_EMBEDDING_FEATURES = [DESCRIPTION_EMBEDDING_COL]


BASE_NUM_FEATURES = [
    "listing_id",
    "price",
    "area_total",
    "floor",
    "floors_total",
    "rooms_count",
    "nearest_metro_duration_min",
    "views",
]

# ---------- Runtime configs ----------
DEFAULT_INFERENCE_API_URL = "http://localhost:8000/v1/predict"
DEFAULT_BOT_HTTP_TIMEOUT_SECONDS = 180
