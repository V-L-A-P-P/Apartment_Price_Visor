from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer

from apartment_price_visor.config import (
    CAT_FEATURES_SELLER,
    CAT_FEATURES,
    DEFAULT_CATBOOST_MODEL_PATH,
    DEFAULT_CATBOOST_SELLER_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DESCRIPTION_COL,
    DESCRIPTION_EMBEDDING_COL,
    EMBEDDING_MODEL_NAME,
    TABLE_COLS_CB_SELLER,
    TABLE_COLS_CB,
)


DEFAULT_MODEL_PATH = DEFAULT_CATBOOST_MODEL_PATH
DEFAULT_OUT_PATH = DEFAULT_PREDICTIONS_PATH

_EMBEDDING_MODEL: SentenceTransformer | None = None
_CATBOOST_MODELS: dict[str, CatBoostRegressor] = {}


def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def _get_catboost_model(model_path: Path) -> CatBoostRegressor:
    key = str(model_path.resolve())
    model = _CATBOOST_MODELS.get(key)
    if model is not None:
        return model

    loaded_model = CatBoostRegressor()
    loaded_model.load_model(str(model_path))
    _CATBOOST_MODELS[key] = loaded_model
    return loaded_model


def preload_inference_models(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model not found: {model_path}")
    _get_embedding_model()
    _get_catboost_model(model_path)


def _encode_description_embeddings(description: pd.Series) -> list[list[float]]:
    model = _get_embedding_model()
    texts = description.fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32").tolist()


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_feature_frame_for_inference(raw_df: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    data = raw_df.copy()

    if DESCRIPTION_EMBEDDING_COL not in data.columns:
        if DESCRIPTION_COL not in data.columns:
            raise ValueError(
                "Either 'description' or 'description_embedding' column is required for inference"
            )
        data[DESCRIPTION_EMBEDDING_COL] = _encode_description_embeddings(data[DESCRIPTION_COL])

    if DESCRIPTION_COL in data.columns:
        data = data.drop(columns=[DESCRIPTION_COL])

    if mode == "seller":
        table_cols = TABLE_COLS_CB_SELLER
        cat_features = CAT_FEATURES_SELLER
    else:
        table_cols = TABLE_COLS_CB
        cat_features = CAT_FEATURES

    if "date_added" not in data.columns:
        data["date_added"] = datetime.utcnow().date().isoformat()

    required_cols = [c for c in table_cols if c not in {"price", DESCRIPTION_COL}]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {missing}")

    df = data[required_cols + [DESCRIPTION_EMBEDDING_COL]].copy()

    # Date features (same logic as training pipeline).
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year"] = df["date_added"].dt.year
    df["month"] = df["date_added"].dt.month
    df["day_of_week"] = df["date_added"].dt.dayofweek
    df["season"] = (df["date_added"].dt.month % 12 // 3 + 1).astype("float")
    df["days_since_first"] = (df["date_added"] - df["date_added"].min()).dt.days
    df = df.drop(columns=["date_added"])

    numeric_cols = [
        "rooms_count",
        "area_total",
        "living_area",
        "kitchen_area",
        "floor",
        "floors_total",
        "ceiling_height_m",
        "nearest_metro_duration_min",
        "nearest_metro_distance_km",
        "views",
        "complex_floors_total",
        "year",
        "month",
        "day_of_week",
        "season",
        "days_since_first",
    ]
    df = _to_numeric(df, numeric_cols)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    return df


def predict_catboost(
    input_path: Path,
    model_path: Path,
    mode: str = "full",
    output_path: Path | None = None,
) -> pd.DataFrame:
    if input_path.suffix.lower() == ".parquet":
        raw_df = pd.read_parquet(input_path)
    else:
        raw_df = pd.read_csv(input_path)

    model = _get_catboost_model(model_path)

    features_df = _prepare_feature_frame_for_inference(raw_df, mode=mode)
    model_features = list(model.feature_names_)
    missing_model_features = [c for c in model_features if c not in features_df.columns]
    if missing_model_features:
        raise ValueError(
            f"Prepared dataframe misses model features: {missing_model_features}"
        )

    preds = model.predict(features_df[model_features])
    result = raw_df.copy()
    result["predicted_price"] = preds

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)

    return result


def predict_price_from_features(
    features: dict,
    model_path: Path = DEFAULT_MODEL_PATH,
    mode: str = "full",
) -> float:
    model = _get_catboost_model(model_path)

    raw_df = pd.DataFrame([features])
    prepared_df = _prepare_feature_frame_for_inference(raw_df, mode=mode)

    model_features = list(model.feature_names_)
    missing_model_features = [c for c in model_features if c not in prepared_df.columns]
    if missing_model_features:
        raise ValueError(
            f"Prepared dataframe misses model features: {missing_model_features}"
        )

    prediction = model.predict(prepared_df[model_features])[0]
    return float(prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with CatBoost apartment model")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to parquet/csv with raw rows for prediction",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained catboost model (*.cbm)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "seller"],
        default="full",
        help="Inference mode: full (listing-like) or seller (pre-listing friendly)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Where to save predictions parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if args.model_path == DEFAULT_MODEL_PATH and args.mode == "seller":
        model_path = DEFAULT_CATBOOST_SELLER_MODEL_PATH
    result = predict_catboost(
        input_path=args.input_path,
        model_path=model_path,
        mode=args.mode,
        output_path=args.output_path,
    )
    payload = {
        "rows": int(len(result)),
        "mode": args.mode,
        "model_path": str(model_path),
        "output_path": str(args.output_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
