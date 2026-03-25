# Apartment Price Visor

Сервис для автоматического сбора и анализа объявлений о продаже/аренде квартир с целью оценки справедливости цены.

## Цель проекта

Построить систему, которая получает ссылку на объявление, собирает данные и формирует основу для анализа:
- насколько цена объявления соответствует ожидаемой рыночной;
- какие факторы влияют на цену;
- какие объявления выглядят переоценёнными/недооценёнными.

---

## Что уже реализовано

### 1) Инициализация проекта
- Python-проект на `uv`
- структура репозитория в стиле data/ML-проекта:
  - `src/`
  - `data/raw`, `data/processed`
  - `tests/`
- Git + `.gitignore`

### 2) DVC
- DVC подключён для отслеживания тяжёлых/производных артефактов.

### 3) Скрейпинг Move.ru
Реализованы модули:
- поиск URL объявлений в выдаче;
- парсинг карточки объявления (цена, площадь, комнаты, этажи, описание, метро, и т.д.);
- сбор `image_urls`.

### 4) Загрузка изображений в S3 (Yandex Object Storage)
- изображения по `image_urls` скачиваются и загружаются в S3-совместимое хранилище;
- в raw-датасете хранится `image_s3_uris` (а не локальные пути).

### 5) Raw pipeline
Pipeline делает:
1. сбор новых URL;
2. парсинг карточек;
3. запись сырого JSONL;
4. загрузку изображений в S3;
5. сохранение `image_s3_uris`.

Выход:
- `data/raw/move_ru/listings.jsonl`
- `data/raw/move_ru/errors.jsonl`

### 6) Валидация схемы (Pandera)
- описана схема датасета;
- добавлены правила диапазонов и кросс-полевые проверки
  (например `floor <= floors_total`, `living_area <= area_total`).

### 7) Сборка clean-датасета
Скрипт preprocessing:
- читает raw JSONL;
- приводит типы;
- удаляет дубликаты;
- валидирует через Pandera;
- сохраняет:
  - clean parquet,
  - rejected parquet (строки с нарушениями правил).

### 8) DVC для processed parquet
- clean/rejected parquet добавляются под DVC;
- артефакты пушатся в DVC remote.

---

## Текущая архитектура данных

### Raw layer
- `data/raw/move_ru/listings.jsonl`
- `data/raw/move_ru/errors.jsonl`
- изображения в Yandex Object Storage, ссылки в поле `image_s3_uris`.

### Processed layer
- `data/processed/move_ru/listings_clean.parquet`
- `data/processed/move_ru/listings_rejected.parquet`

---

## Требования

- Python 3.11+
- `uv`
- доступ к интернету для скрейпинга
- бакет в Yandex Object Storage (S3 API)

---

## Установка

```bash
uv sync
