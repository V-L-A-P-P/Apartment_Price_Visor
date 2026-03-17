from __future__ import annotations

import json
import time
from pathlib import Path

from apartment_price_visor.scrapers.moveru_image_download import MoveRuImagesDownloader  # файл moveru_image_download.py
from apartment_price_visor.scrapers.movero_listings_search import MoveRuListingsSearch, SearchResultItem  # файл movero_listings_search.py
from apartment_price_visor.scrapers.moveru_listing_scrape import MoveRuListingScraper  # файл moveru_listing_scrape.py
from apartment_price_visor.utils.dvc import dvc_add, dvc_push



RAW_DIR = Path("data/raw/move_ru")
IMAGES_DIR = RAW_DIR / "images"
LISTINGS_PATH = RAW_DIR / "listings.jsonl"
ERRORS_PATH = RAW_DIR / "errors.jsonl"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


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
    download_images: bool = True,
    push_images_to_dvc: bool = False,
) -> dict:
    ensure_directories()

    known_listing_ids = load_known_listing_ids(LISTINGS_PATH)

    searcher = MoveRuListingsSearch(sleep_sec=search_sleep_sec)
    scraper = MoveRuListingScraper()
    image_downloader = MoveRuImagesDownloader()

    search_items = searcher.collect_listing_urls(
        start_page=start_page,
        max_pages=max_pages,
        known_listing_ids=known_listing_ids,
        stop_on_known_streak=stop_on_known_streak,
    )

    new_items = searcher.filter_only_new_items(
        items=search_items,
        known_listing_ids=known_listing_ids,
    )

    stats = {
        "known_before_run": len(known_listing_ids),
        "found_in_search": len(search_items),
        "new_urls_found": len(new_items),
        "parsed_successfully": 0,
        "parse_errors": 0,
        "images_downloaded_for": 0,
    }

    for idx, item in enumerate(new_items, start=1):
        try:
            record = scraper.parse_listing(item.listing_url)

            record["search_page_num"] = item.page_num
            record["search_url"] = item.search_url

            local_image_paths: list[str] = []

            if download_images:
                listing_id = record.get("listing_id")
                image_urls = record.get("image_urls") or []

                if listing_id and image_urls:
                    local_image_paths = image_downloader.download_listing_images(
                        listing_id=str(listing_id),
                        image_urls=image_urls,
                        images_root=IMAGES_DIR,
                        overwrite=False,
                    )
                    stats["images_downloaded_for"] += 1

            record["local_image_paths"] = local_image_paths

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

    if push_images_to_dvc:
        dvc_add(IMAGES_DIR)
        dvc_push()

    return stats


def main() -> None:
    stats = update_moveru_raw(
        start_page=1,
        max_pages=3,
        search_sleep_sec=1.0,
        listing_sleep_sec=1.0,
        stop_on_known_streak=False,
        download_images=True,
        push_images_to_dvc=False,
    )

    print("\nRun finished:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 