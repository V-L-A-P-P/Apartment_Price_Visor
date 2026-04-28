from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apartment_price_visor.config import (
    DEFAULT_CATBOOST_MODEL_PATH,
    DEFAULT_CATBOOST_SELLER_MODEL_PATH,
)
from apartment_price_visor.models.infer_tabular_model import (
    preload_inference_models,
    predict_price_from_features,
)


class ApartmentFeatures(BaseModel):
    listing_id: int = 0
    views: int | None = None
    date_added: str | None = None
    rooms_count: float
    area_total: float
    living_area: float | None = None
    kitchen_area: float | None = None
    floor: float
    floors_total: float
    housing_type: str
    object_type: str
    renovation: str
    ceiling_height_m: float | None = None
    address_city: str
    address_street: str
    nearest_metro_name: str
    nearest_metro_duration_min: float
    nearest_metro_distance_km: float | None = None
    housing_class: str | None = None
    building_stage: str | None = None
    complex_floors_total: float | None = None
    delivery_quarter: str | None = None
    description: str = Field(default="", description="Free text apartment description")


class PredictRequest(BaseModel):
    mode: str = Field(default="seller", pattern="^(seller|full)$")
    features: ApartmentFeatures


class PredictResponse(BaseModel):
    predicted_price: float
    currency: str = "RUB"
    price_per_m2: float
    mode: str
    model_path: str


app = FastAPI(title="Apartment Price Visor Inference API", version="0.1.0")


def _resolve_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_CATBOOST_MODEL_PATH)))


def _resolve_model_path_for_mode(mode: str) -> Path:
    if mode == "seller":
        return Path(os.getenv("SELLER_MODEL_PATH", str(DEFAULT_CATBOOST_SELLER_MODEL_PATH)))
    return _resolve_model_path()


@app.on_event("startup")
def startup_preload() -> None:
    full_model_path = _resolve_model_path()
    if full_model_path.exists():
        preload_inference_models(model_path=full_model_path)
    seller_model_path = _resolve_model_path_for_mode("seller")
    if seller_model_path.exists() and seller_model_path != full_model_path:
        preload_inference_models(model_path=seller_model_path)


@app.get("/v1/health")
def health() -> dict:
    full_model_path = _resolve_model_path()
    seller_model_path = _resolve_model_path_for_mode("seller")
    return {
        "status": "ok",
        "model_path_full": str(full_model_path),
        "model_exists_full": full_model_path.exists(),
        "model_path_seller": str(seller_model_path),
        "model_exists_seller": seller_model_path.exists(),
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    mode = payload.mode
    model_path = _resolve_model_path_for_mode(mode)
    if not model_path.exists():
        raise HTTPException(status_code=500, detail=f"Model not found: {model_path}")

    features = payload.features.model_dump()
    try:
        predicted_price = predict_price_from_features(
            features=features,
            model_path=model_path,
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    area_total = features["area_total"]
    price_per_m2 = predicted_price / area_total if area_total else 0.0

    return PredictResponse(
        predicted_price=predicted_price,
        price_per_m2=price_per_m2,
        mode=mode,
        model_path=str(model_path),
    )
