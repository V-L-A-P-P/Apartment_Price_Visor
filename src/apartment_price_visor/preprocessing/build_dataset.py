from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pandera as pa

from apartment_price_visor.preprocessing.schema import validate_listings_df


RAW_LISTINGS_PATH = Path("data/raw/move_ru/listings.jsonl")
PROCESSED_DIR = Path("data/processed/move_ru")
CLEAN_PARQUET_PATH = PROCESSED_DIR / "listings_clean.parquet"
REJECTED_PARQUET_PATH = PROCESSED_DIR / "listings_rejected.parquet"


NUMERIC_COLUMNS = [
    "price",
    "price_per_m2",
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
    "previous_price",
    "last_price_drop_abs",
    "elite_flag",
    "complex_lift_flag",
    "parking_flag",
    "security_flag",
]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # На этом этапе просто пропускаем битые строки
                continue

    return pd.DataFrame(rows)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if "date_added" in result.columns:
        result["date_added"] = pd.to_datetime(result["date_added"], errors="coerce").dt.date

    if "scraped_at" in result.columns:
        result["scraped_at"] = pd.to_datetime(result["scraped_at"], errors="coerce")

    # listing_id храним строкой, чтобы не терять ведущие нули (если вдруг)
    if "listing_id" in result.columns:
        result["listing_id"] = result["listing_id"].astype("string")

    return result


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "listing_id" not in df.columns:
        return df

    # Оставляем последнюю запись по listing_id
    return df.drop_duplicates(subset=["listing_id"], keep="last")


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает (valid_df, invalid_df).
    invalid_df строим через индексы из pandera.failure_cases.
    """
    try:
        valid_df = validate_listings_df(df)
        invalid_df = df.iloc[0:0].copy()
        return valid_df, invalid_df
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases

        # Индексы строк, которые не прошли валидацию
        bad_indices = set(
            idx
            for idx in failure_cases["index"].tolist()
            if isinstance(idx, (int, float)) and pd.notna(idx)
        )

        bad_indices = {int(i) for i in bad_indices}
        invalid_df = df.loc[df.index.isin(bad_indices)].copy()
        valid_df = df.loc[~df.index.isin(bad_indices)].copy()

        # На всякий случай прогоняем валидные строки ещё раз
        valid_df = validate_listings_df(valid_df)

        return valid_df, invalid_df


def build_dataset(
    raw_path: Path = RAW_LISTINGS_PATH,
    clean_path: Path = CLEAN_PARQUET_PATH,
    rejected_path: Path = REJECTED_PARQUET_PATH,
) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = read_jsonl(raw_path)
    typed_df = coerce_types(raw_df)
    dedup_df = drop_duplicates(typed_df)

    valid_df, invalid_df = split_valid_invalid(dedup_df)

    valid_df.to_parquet(clean_path, index=False)
    invalid_df.to_parquet(rejected_path, index=False)

    stats = {
        "raw_rows": len(raw_df),
        "after_dedup": len(dedup_df),
        "valid_rows": len(valid_df),
        "invalid_rows": len(invalid_df),
        "clean_path": str(clean_path),
        "rejected_path": str(rejected_path),
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean parquet dataset from raw JSONL")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_LISTINGS_PATH,
        help="Path to raw listings jsonl",
    )
    parser.add_argument(
        "--clean-path",
        type=Path,
        default=CLEAN_PARQUET_PATH,
        help="Path to output clean parquet",
    )
    parser.add_argument(
        "--rejected-path",
        type=Path,
        default=REJECTED_PARQUET_PATH,
        help="Path to output rejected parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        raw_path=args.raw_path,
        clean_path=args.clean_path,
        rejected_path=args.rejected_path,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()