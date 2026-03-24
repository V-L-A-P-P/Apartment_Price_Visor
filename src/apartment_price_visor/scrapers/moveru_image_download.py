from __future__ import annotations

from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from apartment_price_visor.storage.s3 import S3Uploader


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

    def upload_listing_images_to_s3(
        self,
        *,
        listing_id: str,
        image_urls: list[str],
        s3_uploader: S3Uploader,
    ) -> list[str]:
        """
        Скачивает картинки объявления и загружает их в S3-compatible storage.

        Возвращает список S3 URI.
        """
        uploaded_uris: list[str] = []

        for idx, image_url in enumerate(image_urls, start=1):
            ext = self._guess_extension_from_url(image_url)
            filename = f"{idx:04d}{ext}"

            response = self.session.get(image_url, timeout=self.timeout)
            response.raise_for_status()

            object_key = s3_uploader.build_object_key(
                listing_id=listing_id,
                filename=filename,
            )

            uploaded_uri = s3_uploader.upload_bytes(
                content=response.content,
                object_key=object_key,
                content_type=s3_uploader.guess_content_type(filename),
            )
            uploaded_uris.append(uploaded_uri)

        return uploaded_uris