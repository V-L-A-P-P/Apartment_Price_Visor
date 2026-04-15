from __future__ import annotations

import pandera as pa
from pandera import Check


ALLOWED_COLUMNS = [
    "source",
    "scraped_at",
    "url",
    "listing_id",
    "price",
    "views",
    "date_added",
    "rooms_count",
    "area_total",
    "living_area",
    "kitchen_area",
    "floor",
    "floors_total",
    "housing_type",
    "object_type",
    "renovation",
    "ceiling_height_m",
    "address_city",
    "address_street",
    "nearest_metro_name",
    "nearest_metro_duration_min",
    "nearest_metro_distance_km",
    "housing_class",
    "building_type",
    "building_stage",
    "complex_floors_total",
    "complex_lift_flag",
    "parking_flag",
    "delivery_quarter",
    "previous_price",
    "feature_list",
    "title",
    "description",
    "image_urls",
    "search_page_num",
    "search_url",
    "image_s3_uris",
]


LISTINGS_SCHEMA = pa.DataFrameSchema(
    {
        "source": pa.Column(str, nullable=False),
        "scraped_at": pa.Column(pa.DateTime, nullable=True),
        "url": pa.Column(str, nullable=False),
        "listing_id": pa.Column(str, nullable=True),

        "price": pa.Column(float, nullable=True, checks=[Check.ge(300_000), Check.le(5_000_000_000)]),
        "views": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(10_000_000)]),
        "date_added": pa.Column(pa.Date, nullable=True),

        "rooms_count": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(15)]),
        "area_total": pa.Column(float, nullable=True, checks=[Check.ge(10), Check.le(800)]),
        "living_area": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(800)]),
        "kitchen_area": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(400)]),

        "floor": pa.Column(float, nullable=True, checks=[Check.ge(1), Check.le(150)]),
        "floors_total": pa.Column(float, nullable=True, checks=[Check.ge(1), Check.le(150)]),

        "housing_type": pa.Column(str, nullable=True),
        "object_type": pa.Column(str, nullable=True),
        "renovation": pa.Column(str, nullable=True),
        "ceiling_height_m": pa.Column(float, nullable=True, checks=[Check.ge(1.8), Check.le(8)]),

        "address_city": pa.Column(str, nullable=True),
        "address_street": pa.Column(str, nullable=True),

        "nearest_metro_name": pa.Column(str, nullable=True),
        "nearest_metro_duration_min": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(500)]),
        "nearest_metro_distance_km": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(300)]),

        "housing_class": pa.Column(str, nullable=True),
        "building_type": pa.Column(str, nullable=True),
        "building_stage": pa.Column(str, nullable=True),
        "complex_floors_total": pa.Column(float, nullable=True, checks=[Check.ge(1), Check.le(200)]),
        "complex_lift_flag": pa.Column(float, nullable=True, checks=[Check.isin([0, 1])]),
        "parking_flag": pa.Column(float, nullable=True, checks=[Check.isin([0, 1])]),
        "delivery_quarter": pa.Column(str, nullable=True),
        "previous_price": pa.Column(float, nullable=True, checks=[Check.ge(0), Check.le(5_000_000_000)]),

        "feature_list": pa.Column(object, nullable=True),   # list[str]
        "title": pa.Column(str, nullable=True),
        "description": pa.Column(str, nullable=True),
        "image_urls": pa.Column(object, nullable=True),     # list[str]
        "search_page_num": pa.Column(float, nullable=True, checks=[Check.ge(1), Check.le(500000)]),
        "search_url": pa.Column(str, nullable=True),
        "image_s3_uris": pa.Column(object, nullable=True),  # list[str]
    },
    checks=[
        Check(
            lambda df: (
                df["floor"].isna()
                | df["floors_total"].isna()
                | (df["floor"] <= df["floors_total"])
            ),
            error="floor must be <= floors_total",
        ),
        Check(
            lambda df: (
                df["living_area"].isna()
                | df["area_total"].isna()
                | (df["living_area"] <= df["area_total"])
            ),
            error="living_area must be <= area_total",
        ),
        Check(
            lambda df: (
                df["kitchen_area"].isna()
                | df["area_total"].isna()
                | (df["kitchen_area"] <= df["area_total"])
            ),
            error="kitchen_area must be <= area_total",
        ),
    ],
    strict="filter",  # <- ключевой момент: лишние колонки удаляются
    coerce=True,
)


def validate_listings_df(df):
    # Если твоя версия pandera не поддерживает strict="filter",
    # раскомментируй строку ниже:
    # df = df.reindex(columns=ALLOWED_COLUMNS)
    return LISTINGS_SCHEMA.validate(df, lazy=True)