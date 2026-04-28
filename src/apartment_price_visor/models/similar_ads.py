from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from apartment_price_visor.config import (
    DEFAULT_CLEAN_PATH,
    EMBEDDING_MODEL_NAME,
)

_EMBEDDING_MODEL: SentenceTransformer | None = None
_SIMILAR_ADS_DF: pd.DataFrame | None = None
_SIMILAR_ADS_EMBEDDINGS: np.ndarray | None = None
_LOCK = Lock()

SIMILAR_ADS_COLUMNS = [
    "listing_id",
    "price",
    "area_total",
    "rooms_count",
    "address_city",
    "address_street",
    "nearest_metro_name",
    "description",
]


def _resolve_dataset_path() -> Path:
    return Path(os.getenv("SIMILAR_ADS_DATA_PATH", str(DEFAULT_CLEAN_PATH)))


def _get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def _build_query_text(features: dict[str, Any]) -> str:
    raw_description = str(features.get("description", "") or "").strip()
    if raw_description:
        return raw_description
    city = features.get("address_city", "")
    street = features.get("address_street", "")
    rooms = features.get("rooms_count", "")
    area_total = features.get("area_total", "")
    renovation = features.get("renovation", "")
    metro = features.get("nearest_metro_name", "")
    return (
        f"{rooms}-комнатная квартира {area_total} м2, "
        f"{city}, {street}, метро {metro}, ремонт {renovation}"
    )


def _norm_or_ones(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _prepare_similar_ads_index() -> None:
    global _SIMILAR_ADS_DF, _SIMILAR_ADS_EMBEDDINGS
    if _SIMILAR_ADS_DF is not None and _SIMILAR_ADS_EMBEDDINGS is not None:
        return

    with _LOCK:
        if _SIMILAR_ADS_DF is not None and _SIMILAR_ADS_EMBEDDINGS is not None:
            return

        path = _resolve_dataset_path()
        if not path.exists():
            raise FileNotFoundError(f"Similar ads dataset not found: {path}")

        df = pd.read_parquet(path)
        missing = [c for c in SIMILAR_ADS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Similar ads dataset misses columns: {missing}")

        df = df[SIMILAR_ADS_COLUMNS].copy()
        df = df[df["description"].fillna("").astype(str).str.strip() != ""].copy()
        if df.empty:
            raise ValueError("Similar ads dataset has no non-empty descriptions")

        texts = df["description"].fillna("").astype(str).tolist()
        embeddings = _get_embedding_model().encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        embeddings = _norm_or_ones(embeddings)

        _SIMILAR_ADS_DF = df.reset_index(drop=True)
        _SIMILAR_ADS_EMBEDDINGS = embeddings


def find_similar_ads(features: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    _prepare_similar_ads_index()
    assert _SIMILAR_ADS_DF is not None
    assert _SIMILAR_ADS_EMBEDDINGS is not None

    query_text = _build_query_text(features)
    query_emb = _get_embedding_model().encode(
        [query_text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    query_emb = _norm_or_ones(query_emb)

    sims = (_SIMILAR_ADS_EMBEDDINGS @ query_emb[0]).astype("float32")
    top_n = min(top_k + 10, len(sims))
    candidate_idx = np.argpartition(-sims, top_n - 1)[:top_n]
    sorted_idx = candidate_idx[np.argsort(-sims[candidate_idx])]

    requested_listing_id = features.get("listing_id")
    out: list[dict[str, Any]] = []
    for idx in sorted_idx:
        row = _SIMILAR_ADS_DF.iloc[int(idx)]
        if requested_listing_id not in (None, "", 0):
            if str(row.get("listing_id")) == str(requested_listing_id):
                continue
        price = float(row["price"]) if pd.notna(row["price"]) else None
        area = float(row["area_total"]) if pd.notna(row["area_total"]) else None
        out.append(
            {
                "listing_id": row.get("listing_id"),
                "price": price,
                "price_per_m2": (price / area) if price and area else None,
                "rooms_count": row.get("rooms_count"),
                "city": row.get("address_city"),
                "street": row.get("address_street"),
                "metro": row.get("nearest_metro_name"),
                "similarity": float(sims[int(idx)]),
            }
        )
        if len(out) >= top_k:
            break

    return out
