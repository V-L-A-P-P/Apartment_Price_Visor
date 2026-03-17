from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(slots=True)
class SearchResultItem:
    source: str
    search_url: str
    listing_url: str
    listing_id: str | None
    page_num: int


class MoveRuListingsSearch:
    BASE_URL = "https://move.ru"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

    def __init__(
        self,
        sleep_sec: float = 1.0,
        timeout: int = 30,
    ) -> None:
        self.sleep_sec = sleep_sec
        self.timeout = timeout
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.HEADERS)

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _clean_url(url: str | None) -> str | None:
        if not url:
            return None
        return url.strip().split("#")[0]

    @staticmethod
    def _extract_listing_id(url: str) -> str | None:
        match = re.search(r"_(\d+)/?$", url)
        return match.group(1) if match else None

    @staticmethod
    def _is_listing_url(url: str) -> bool:
        return bool(re.search(r"^https?://move\.ru/objects/.+_\d+/?$", url))

    def _request(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def build_search_url(page: int) -> str:
        if page < 1:
            raise ValueError("page must be >= 1")
        return f"https://move.ru/kvartiry/?page={page}"

    def _extract_listing_urls_from_html(self, html: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        def add(url: str | None) -> None:
            if not url:
                return

            cleaned = self._clean_url(url)
            if not cleaned:
                return

            if cleaned.startswith("/"):
                cleaned = urljoin(self.BASE_URL, cleaned)

            if self._is_listing_url(cleaned) and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)

        pattern = r'https?://move\.ru/objects/[^"\'\s]+?_\d+/?'
        for url in re.findall(pattern, html):
            add(url)

        if not result:
            soup = BeautifulSoup(html, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                add(a_tag.get("href"))

        return result

    def parse_search_page(self, page_num: int) -> list[SearchResultItem]:
        search_url = self.build_search_url(page=page_num)
        html = self._request(search_url)
        listing_urls = self._extract_listing_urls_from_html(html)

        items: list[SearchResultItem] = []
        for listing_url in listing_urls:
            items.append(
                SearchResultItem(
                    source="move_ru",
                    search_url=search_url,
                    listing_url=listing_url,
                    listing_id=self._extract_listing_id(listing_url),
                    page_num=page_num,
                )
            )
        return items

    def collect_listing_urls(
        self,
        start_page: int = 1,
        max_pages: int = 5,
        known_listing_ids: set[str] | None = None,
        stop_on_known_streak: bool = False,
    ) -> list[SearchResultItem]:
        if start_page < 1:
            raise ValueError("start_page must be >= 1")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")

        known_listing_ids = known_listing_ids or set()

        all_items: list[SearchResultItem] = []
        seen_urls: set[str] = set()

        for page_num in range(start_page, start_page + max_pages):
            page_items = self.parse_search_page(page_num=page_num)

            if not page_items:
                break

            new_page_items: list[SearchResultItem] = []
            known_count_on_page = 0

            for item in page_items:
                if item.listing_url in seen_urls:
                    continue

                seen_urls.add(item.listing_url)

                if item.listing_id and item.listing_id in known_listing_ids:
                    known_count_on_page += 1

                new_page_items.append(item)

            if not new_page_items:
                break

            all_items.extend(new_page_items)

            if stop_on_known_streak and known_count_on_page == len(new_page_items):
                break

            time.sleep(self.sleep_sec)

        return all_items

    @staticmethod
    def filter_only_new_items(
        items: Iterable[SearchResultItem],
        known_listing_ids: set[str],
    ) -> list[SearchResultItem]:
        result: list[SearchResultItem] = []
        for item in items:
            if item.listing_id is None:
                result.append(item)
            elif item.listing_id not in known_listing_ids:
                result.append(item)
        return result


if __name__ == "__main__":
    searcher = MoveRuListingsSearch(sleep_sec=1.0)

    items = searcher.collect_listing_urls(
        start_page=1,
        max_pages=2,
    )

    for item in items[:10]:
        print(item)