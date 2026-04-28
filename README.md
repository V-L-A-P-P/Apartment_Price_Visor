# Apartment Price Visor

Проект для автоматизации сбора, валидации и анализа объявлений о квартирах на `move.ru` с последующей подготовкой табличных данных и обучением моделей оценки цены.

## Что делает проект

- собирает объявления о продаже/аренде квартир со страницы Move.ru
- парсит карточки объявлений и формирует структурированный JSONL
- загружает изображения в S3-совместимое хранилище и сохраняет URI
- приводит данные к единой схеме и разделяет на `clean` и `rejected`
- готовит табличные датасеты для обучения моделей
- обучает CatBoost и baseline RandomForest для предсказания цены

## Ключевые компоненты

- `src/apartment_price_visor/scrapers/` — скрейпинг объявлений и парсинг структурированных свойств
- `src/apartment_price_visor/storage/` — загрузка изображений в S3 API (Yandex Object Storage)
- `src/apartment_price_visor/preprocessing/` — чтение сырья, приведение типов, дедупликация, валидация схемы и экспорт parquet
- `src/apartment_price_visor/models/` — обучение табличных моделей и сохранение результатов

## Что уже реализовано

- работа с DVC для хранения `raw` и `processed` данных
- подробная Pandera-схема с проверками диапазонов и кросс-полями
- хранение изображений в S3 и формирование поля `image_s3_uris`
- отделение отклонённых строк в `listings_rejected.parquet`
- обучение CatBoost-модели и baseline RandomForest

## Структура проекта

```text
.
├── README.md
├── pyproject.toml
├── main.py
├── data/
│   ├── raw/
│   │   └── move_ru/
│   │       ├── listings.jsonl
│   │       └── errors.jsonl
│   └── processed/
│       └── move_ru/
│           ├── listings_clean.parquet
│           └── listings_rejected.parquet
├── src/apartment_price_visor/
│   ├── preprocessing/
│   ├── scrapers/
│   ├── storage/
│   └── models/
└── artifacts/
    ├── models/
    └── metrics/
```

## Требования

- Python 3.11+
- `uv` для управления зависимостями
- `dvc` и `dvc-s3` для работы с артефактами
- доступ в интернет для скрейпинга
- S3-совместимый бакет для хранения изображений

## Установка

1. Установить зависимости:

```bash
uv sync
```

2. Проверить, что пакеты установлены корректно:

```bash
uv run python -c "import pandas, pandera, requests, boto3"
```

## Переменные окружения

Для загрузки изображений через `S3Uploader` задать:

```bash
export APARTMENT_PRICE_VISOR_S3_BUCKET="your-bucket-name"
export APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL="https://storage.yandexcloud.net"
export APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID="your-access-key"
export APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY="your-secret-key"
export APARTMENT_PRICE_VISOR_S3_REGION="ru-central1"
export APARTMENT_PRICE_VISOR_S3_KEY_PREFIX="move_ru/images"
```

Можно сохранить в файл `.env`, чтобы `dotenv` загрузил значения автоматически.

## Использование

### 1. Построение clean/rejected датасета

```bash
uv run python -m apartment_price_visor.preprocessing.build_dataset
```

Если `uv` не настроен, можно запустить напрямую из корня:

```bash
PYTHONPATH=src uv run python -m apartment_price_visor.preprocessing.build_dataset
```

### 2. Обучение табличных моделей

```bash
uv run python -m apartment_price_visor.models.train_tabular_model --train-dir data/processed/train --out-dir artifacts --test-size 0.2
```

По умолчанию ожидаются файлы:

- `data/processed/train/train_tabular_catboost.parquet`
- `data/processed/train/train_tabular_baseline.parquet`

### 3. Тест S3-загрузки

```bash
uv run python src/apartment_price_visor/storage/s3.py
```

Этот модуль проверяет переменные окружения и загружает тестовый объект в указанный бакет.

## Данные

- `data/raw/move_ru/listings.jsonl` — сырые записи объявлений
- `data/raw/move_ru/errors.jsonl` — ошибки парсинга
- `data/processed/move_ru/listings_clean.parquet` — валидные объявления
- `data/processed/move_ru/listings_rejected.parquet` — объявления, не прошедшие валидацию

## Схема данных

Схема определена в `src/apartment_price_visor/preprocessing/schema.py` и включает поля:

- `source`, `scraped_at`, `url`, `listing_id`
- `price`, `views`, `date_added`, `rooms_count`, `area_total`, `living_area`, `kitchen_area`
- `floor`, `floors_total`, `housing_type`, `object_type`, `renovation`
- `ceiling_height_m`, `address_city`, `address_street`, `nearest_metro_name`
- `nearest_metro_duration_min`, `nearest_metro_distance_km`, `housing_class`
- `building_type`, `building_stage`, `complex_floors_total`, `complex_lift_flag`
- `parking_flag`, `delivery_quarter`, `previous_price`, `feature_list`
- `title`, `description`, `image_urls`, `search_page_num`, `search_url`, `image_s3_uris`

Валидация проверяет диапазоны, типы и логические зависимости (`floor <= floors_total`, `living_area <= area_total`, `kitchen_area <= area_total`).

## Тестирование

```bash
uv run pytest
```

## Цели развития

- расширить обработку ещё большего числа сегментов объявлений
- добавить inference API для быстрой оценки цены по URL
- улучшить обучение моделей, выполнив feature engineering и кросс-валидацию
- добавить отчёты по переоценке/недооценке объявлений

## Контакты

Если нужно дополнить README или добавить новые сценарии, просто открой issue или отправь PR.
