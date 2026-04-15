from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


DEFAULT_TRAIN_DIR = Path("data/processed/train")
DEFAULT_OUT_DIR = Path("artifacts")


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Простой deterministic holdout:
    - если есть days_since_first -> сортируем по времени
    - иначе split по индексу
    """
    data = df.copy()

    if "days_since_first" in data.columns:
        data = data.sort_values("days_since_first").reset_index(drop=True)

    split_idx = int(len(data) * (1 - test_size))
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    return train_df, test_df


def train_catboost(
    train_path: Path,
    out_dir: Path,
    test_size: float = 0.2,
) -> dict:
    df = pd.read_parquet(train_path)

    # Обязательные поля
    if "price" not in df.columns:
        raise ValueError("Column 'price' is required in CatBoost dataset")

    # cat features ожидаем как в prepare_train_datasets
    cat_features = [
        "housing_type",
        "object_type",
        "renovation",
        "address_city",
        "address_street",
        "nearest_metro_name",
        "housing_class",
        "building_stage",
        "delivery_quarter",
    ]
    cat_features = [c for c in cat_features if c in df.columns]

    train_df, test_df = split_train_test(df, test_size=test_size)

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        depth=3,
        learning_rate=0.05,
        n_estimators=500,
        random_seed=42,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)

    model_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "catboost_model.cbm"
    metrics_path = metrics_dir / "catboost_metrics.json"

    model.save_model(str(model_path))

    payload = {
        "model": "catboost",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "train_path": str(train_path),
        "model_path": str(model_path),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def train_baseline(
    train_path: Path,
    out_dir: Path,
    test_size: float = 0.2,
) -> dict:
    df = pd.read_parquet(train_path)

    if "price" not in df.columns:
        raise ValueError("Column 'price' is required in baseline dataset")

    train_df, test_df = split_train_test(df, test_size=test_size)

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)

    model_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline_rf.joblib"
    metrics_path = metrics_dir / "baseline_metrics.json"

    joblib.dump(model, model_path)

    payload = {
        "model": "baseline_random_forest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "train_path": str(train_path),
        "model_path": str(model_path),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular models")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory with prepared train datasets",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for model artifacts and metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    catboost_train_path = args.train_dir / "train_tabular_catboost.parquet"
    baseline_train_path = args.train_dir / "train_tabular_baseline.parquet"

    catboost_result = train_catboost(
        train_path=catboost_train_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
    )
    baseline_result = train_baseline(
        train_path=baseline_train_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
    )

    print(json.dumps(
        {
            "catboost": catboost_result["metrics"],
            "baseline": baseline_result["metrics"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()