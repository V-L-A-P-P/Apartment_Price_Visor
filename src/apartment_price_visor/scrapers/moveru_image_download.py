from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MoveRuImagesDownloader:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

    def __init__(self, timeout: int = 30) -> None:
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
    def _guess_extension_from_url(url: str) -> str:
        path = urlparse(url).path.lower()

        if path.endswith(".jpeg"):
            return ".jpeg"
        if path.endswith(".jpg"):
            return ".jpg"
        if path.endswith(".png"):
            return ".png"
        if path.endswith(".webp"):
            return ".webp"

        return ".jpg"

    def download_listing_images(
        self,
        *,
        listing_id: str,
        image_urls: list[str],
        images_root: Path,
        overwrite: bool = False,
    ) -> list[str]:
        """
        Скачивает картинки объявления в папку:
        images_root / listing_id / 0001.jpg ...

        Возвращает список локальных путей.
        """
        listing_dir = images_root / str(listing_id)
        listing_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []

        for idx, image_url in enumerate(image_urls, start=1):
            ext = self._guess_extension_from_url(image_url)
            file_path = listing_dir / f"{idx:04d}{ext}"

            if file_path.exists() and not overwrite:
                saved_paths.append(str(file_path))
                continue

            response = self.session.get(image_url, timeout=self.timeout)
            response.raise_for_status()

            file_path.write_bytes(response.content)
            saved_paths.append(str(file_path))

        return saved_paths