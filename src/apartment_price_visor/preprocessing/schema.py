from __future__ import annotations

import pandera as pa
from pandera import Check
from pandera.typing import DataFrame


# Базовая schema для уже распарсенных raw записей
LISTINGS_SCHEMA = pa.DataFrameSchema(
    {
        # идентификация
        "source": pa.Column(str, nullable=False),
        "listing_id": pa.Column(str, nullable=True),  # иногда может не вытащиться
        "url": pa.Column(str, nullable=False),

        # цена / площадь
        "price": pa.Column(
            float,  # в pandas nullable int часто приходит как float, потом приведём
            nullable=True,
            checks=[Check.ge(300_000), Check.le(10_000_000_000)],
        ),
        "price_per_m2": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(20_000), Check.le(15_000_000)],
        ),
        "area_total": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(10), Check.le(600)],
        ),
        "living_area": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(0), Check.le(550)],
        ),
        "kitchen_area": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(0), Check.le(320)],
        ),

        # этажность / комнатность
        "rooms_count": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(0), Check.le(20)],
        ),
        "floor": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(1), Check.le(150)],
        ),
        "floors_total": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(1), Check.le(150)],
        ),

        # гео/транспорт
        "nearest_metro_duration_min": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(0), Check.le(240)],
        ),
        "nearest_metro_distance_km": pa.Column(
            float,
            nullable=True,
            checks=[Check.ge(0), Check.le(60)],
        ),

        # флаги (0/1)
        "elite_flag": pa.Column(
            float, nullable=True, checks=[Check.isin([0, 1])]
        ),
        "complex_lift_flag": pa.Column(
            float, nullable=True, checks=[Check.isin([0, 1])]
        ),
        "parking_flag": pa.Column(
            float, nullable=True, checks=[Check.isin([0, 1])]
        ),
        "security_flag": pa.Column(
            float, nullable=True, checks=[Check.isin([0, 1])]
        ),
    },
    checks=[
        # floor <= floors_total (если оба не null)
        Check(
            lambda df: (
                df["floor"].isna()
                | df["floors_total"].isna()
                | (df["floor"] <= df["floors_total"])
            ),
            error="floor must be <= floors_total",
        ),
        # living_area <= area_total
        Check(
            lambda df: (
                df["living_area"].isna()
                | df["area_total"].isna()
                | (df["living_area"] <= df["area_total"])
            ),
            error="living_area must be <= area_total",
        ),
        # kitchen_area <= area_total
        Check(
            lambda df: (
                df["kitchen_area"].isna()
                | df["area_total"].isna()
                | (df["kitchen_area"] <= df["area_total"])
            ),
            error="kitchen_area must be <= area_total",
        ),
    ],
    strict=False,  # разрешаем лишние поля, чтобы не ломать raw слой
    coerce=True,   # пытаемся привести типы автоматически
)


def validate_listings_df(df):
    """Валидирует DataFrame и возвращает validated df."""
    return LISTINGS_SCHEMA.validate(df, lazy=True)