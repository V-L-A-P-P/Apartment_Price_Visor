from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from apartment_price_visor.scrapers.movero_listings_search import (
    MoveRuListingsSearch,
    SearchResultItem,
)
from apartment_price_visor.scrapers.moveru_image_download import MoveRuImagesDownloader
from apartment_price_visor.scrapers.moveru_listing_scrape import MoveRuListingScraper
from apartment_price_visor.storage.s3 import S3Settings, S3Uploader


RAW_DIR = Path("data/raw/move_ru")
LISTINGS_PATH = RAW_DIR / "listings.jsonl"
ERRORS_PATH = RAW_DIR / "errors.jsonl"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_known_listing_ids(path: Path) -> set[str]:
    known_ids: set[str] = set()

    if not path.exists():
        return known_ids

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            listing_id = record.get("listing_id")
            if listing_id:
                known_ids.add(str(listing_id))

    return known_ids


def save_error(
    *,
    item: SearchResultItem,
    error: Exception,
) -> None:
    append_jsonl(
        ERRORS_PATH,
        {
            "source": "move_ru",
            "search_url": item.search_url,
            "listing_url": item.listing_url,
            "listing_id": item.listing_id,
            "page_num": item.page_num,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def update_moveru_raw(
    *,
    start_page: int = 1,
    max_pages: int = 5,
    search_sleep_sec: float = 1.0,
    listing_sleep_sec: float = 1.0,
    stop_on_known_streak: bool = False,
    upload_images_to_s3: bool = True,
) -> dict:
    ensure_directories()

    known_listing_ids = load_known_listing_ids(LISTINGS_PATH)
    print('Загружены listings')
    searcher = MoveRuListingsSearch(sleep_sec=search_sleep_sec)
    scraper = MoveRuListingScraper()
    image_downloader = MoveRuImagesDownloader()

    s3_uploader = None
    if upload_images_to_s3:
        s3_settings = S3Settings.from_env()
        s3_uploader = S3Uploader(s3_settings)
    print('Подготовлен s3 uploader')

    search_items = searcher.collect_listing_urls(
        start_page=start_page,
        max_pages=max_pages,
        known_listing_ids=known_listing_ids,
        stop_on_known_streak=stop_on_known_streak,
    )
    print('Collected items')

    new_items = searcher.filter_only_new_items(
        items=search_items,
        known_listing_ids=known_listing_ids,
    )
    print('Filtered items')

    stats = {
        "known_before_run": len(known_listing_ids),
        "found_in_search": len(search_items),
        "new_urls_found": len(new_items),
        "parsed_successfully": 0,
        "parse_errors": 0,
        "images_uploaded_to_s3_for": 0,
    }

    for idx, item in enumerate(new_items, start=1):
        try:
            record = scraper.parse_listing(item.listing_url)

            record["search_page_num"] = item.page_num
            record["search_url"] = item.search_url

            image_s3_uris: list[str] = []

            if upload_images_to_s3 and s3_uploader is not None:
                listing_id = record.get("listing_id")
                image_urls = record.get("image_urls") or []

                if listing_id and image_urls:
                    image_s3_uris = image_downloader.upload_listing_images_to_s3(
                        listing_id=str(listing_id),
                        image_urls=image_urls,
                        s3_uploader=s3_uploader,
                    )
                    stats["images_uploaded_to_s3_for"] += 1

            record["image_s3_uris"] = image_s3_uris

            append_jsonl(LISTINGS_PATH, record)
            stats["parsed_successfully"] += 1

            if record.get("listing_id"):
                known_listing_ids.add(str(record["listing_id"]))

            print(
                f"[{idx}/{len(new_items)}] OK  "
                f"listing_id={record.get('listing_id')} "
                f"url={item.listing_url}"
            )

        except Exception as exc:
            stats["parse_errors"] += 1
            save_error(item=item, error=exc)

            print(
                f"[{idx}/{len(new_items)}] ERR "
                f"listing_id={item.listing_id} "
                f"url={item.listing_url} "
                f"error={exc}"
            )

        time.sleep(listing_sleep_sec)

    return stats


def main() -> None:
    load_dotenv()
    print("Starting Move.ru raw data update...")
    stats = update_moveru_raw(
        start_page=1200,
        max_pages=1500,
        search_sleep_sec=0.1,
        listing_sleep_sec=0.1,
        stop_on_known_streak=False,
        upload_images_to_s3=True,
    )

    print("\nRun finished:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()