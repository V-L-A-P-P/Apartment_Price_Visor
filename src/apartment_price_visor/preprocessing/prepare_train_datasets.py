from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from apartment_price_visor.config import (
    BASELINE_TRAIN_DATASET_NAME,
    BASE_NUM_FEATURES,
    CATBOOST_SELLER_TRAIN_DATASET_NAME,
    CATBOOST_EMBEDDING_FEATURES,
    CATBOOST_TRAIN_DATASET_NAME,
    CAT_FEATURES_SELLER,
    CAT_FEATURES,
    DATASET_META_NAME,
    DEFAULT_CLEAN_PATH,
    DEFAULT_TRAIN_DIR,
    DESCRIPTION_COL,
    DESCRIPTION_EMBEDDING_COL,
    EMBEDDING_MODEL_NAME,
    TABLE_COLS_CB_SELLER,
    TABLE_COLS_CB,
)

_EMBEDDING_MODEL: SentenceTransformer | None = None


@dataclass
class PreparedDataset:
    df: pd.DataFrame
    target_col: str
    feature_cols: list[str]
    cat_features: list[str] | None = None
    embedding_features: list[str] | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


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


def _ensure_columns(df: pd.DataFrame, cols: list[str], dataset_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] Missing required columns: {missing}")


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prepare_catboost_df(data: pd.DataFrame, mode: str = "full") -> PreparedDataset:
    if mode == "seller":
        required_cols = TABLE_COLS_CB_SELLER
        cat_features = CAT_FEATURES_SELLER
        dataset_name = "catboost_seller"
    else:
        required_cols = TABLE_COLS_CB
        cat_features = CAT_FEATURES
        dataset_name = "catboost_full"

    _ensure_columns(data, required_cols, dataset_name)

    df = data[required_cols].copy()
    df[DESCRIPTION_EMBEDDING_COL] = _encode_description_embeddings(df[DESCRIPTION_COL])
    df = df.drop(columns=[DESCRIPTION_COL])

    # Date features
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year"] = df["date_added"].dt.year
    df["month"] = df["date_added"].dt.month
    df["day_of_week"] = df["date_added"].dt.dayofweek
    df["season"] = (df["date_added"].dt.month % 12 // 3 + 1).astype("float")
    df["days_since_first"] = (df["date_added"] - df["date_added"].min()).dt.days
    df = df.drop(columns=["date_added"])

    # Numeric cols imputation
    numeric_cols = [
        "price",
        "views",
        "rooms_count",
        "area_total",
        "living_area",
        "kitchen_area",
        "floor",
        "floors_total",
        "ceiling_height_m",
        "nearest_metro_duration_min",
        "nearest_metro_distance_km",
        "complex_floors_total",
        "year",
        "month",
        "day_of_week",
        "season",
        "days_since_first",
    ]
    df = _to_numeric(df, numeric_cols)

    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Cat cols imputation
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)

    # Drop rows without target
    df = df[df["price"].notna()].copy()

    feature_cols = [c for c in df.columns if c not in {"price"}]
    return PreparedDataset(
        df=df,
        target_col="price",
        feature_cols=feature_cols,
        cat_features=cat_features,
        embedding_features=CATBOOST_EMBEDDING_FEATURES,
    )


def prepare_baseline_df(data: pd.DataFrame) -> PreparedDataset:
    _ensure_columns(data, BASE_NUM_FEATURES, "baseline")

    df = data[BASE_NUM_FEATURES].copy()
    num_cols = [c for c in df.columns if c not in {"listing_id"}]
    df = _to_numeric(df, num_cols)

    # Median fill
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Drop rows without target
    df = df[df["price"].notna()].copy()

    feature_cols = [c for c in df.columns if c not in {"price"}]
    return PreparedDataset(
        df=df,
        target_col="price",
        feature_cols=feature_cols,
        cat_features=None,
    )


def save_prepared_datasets(
    catboost_ds: PreparedDataset,
    catboost_seller_ds: PreparedDataset,
    baseline_ds: PreparedDataset,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    catboost_path = out_dir / CATBOOST_TRAIN_DATASET_NAME
    catboost_seller_path = out_dir / CATBOOST_SELLER_TRAIN_DATASET_NAME
    baseline_path = out_dir / BASELINE_TRAIN_DATASET_NAME
    meta_path = out_dir / DATASET_META_NAME

    catboost_ds.df.to_parquet(catboost_path, index=False)
    catboost_seller_ds.df.to_parquet(catboost_seller_path, index=False)
    baseline_ds.df.to_parquet(baseline_path, index=False)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "catboost": {
            "rows": int(len(catboost_ds.df)),
            "target_col": catboost_ds.target_col,
            "feature_cols": catboost_ds.feature_cols,
            "cat_features": catboost_ds.cat_features,
            "embedding_features": catboost_ds.embedding_features,
            "path": str(catboost_path),
        },
        "catboost_seller": {
            "rows": int(len(catboost_seller_ds.df)),
            "target_col": catboost_seller_ds.target_col,
            "feature_cols": catboost_seller_ds.feature_cols,
            "cat_features": catboost_seller_ds.cat_features,
            "embedding_features": catboost_seller_ds.embedding_features,
            "path": str(catboost_seller_path),
        },
        "baseline": {
            "rows": int(len(baseline_ds.df)),
            "target_col": baseline_ds.target_col,
            "feature_cols": baseline_ds.feature_cols,
            "path": str(baseline_path),
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train-ready tabular datasets from listings_clean.parquet"
    )
    parser.add_argument(
        "--clean-path",
        type=Path,
        default=DEFAULT_CLEAN_PATH,
        help="Path to listings_clean.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Output directory for prepared train datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = pd.read_parquet(args.clean_path)

    catboost_ds = prepare_catboost_df(data, mode="full")
    catboost_seller_ds = prepare_catboost_df(data, mode="seller")
    baseline_ds = prepare_baseline_df(data)

    meta = save_prepared_datasets(catboost_ds, catboost_seller_ds, baseline_ds, args.out_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()