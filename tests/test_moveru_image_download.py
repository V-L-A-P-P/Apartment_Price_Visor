from apartment_price_visor.scrapers.moveru_image_download import MoveRuImagesDownloader


class DummyResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class DummySession:
    def __init__(self):
        self.requested_urls = []

    def get(self, url, timeout):
        self.requested_urls.append((url, timeout))
        return DummyResponse(content=f"img-bytes-{url}".encode("utf-8"))


class DummyS3Uploader:
    def __init__(self):
        self.upload_calls = []

    def build_object_key(self, *, listing_id: str, filename: str) -> str:
        return f"move_ru/images/{listing_id}/{filename}"

    def guess_content_type(self, filename: str):
        if filename.endswith(".jpg"):
            return "image/jpeg"
        return None

    def upload_bytes(self, *, content: bytes, object_key: str, content_type: str | None = None) -> str:
        self.upload_calls.append(
            {
                "content": content,
                "object_key": object_key,
                "content_type": content_type,
            }
        )
        return f"s3://test-bucket/{object_key}"


def test_upload_listing_images_to_s3(monkeypatch):
    downloader = MoveRuImagesDownloader(timeout=5)
    dummy_session = DummySession()
    downloader.session = dummy_session  # подменяем requests.Session

    uploader = DummyS3Uploader()

    image_urls = [
        "https://example.com/1.jpg",
        "https://example.com/2.jpg",
    ]

    uris = downloader.upload_listing_images_to_s3(
        listing_id="9288118981",
        image_urls=image_urls,
        s3_uploader=uploader,
    )

    assert uris == [
        "s3://test-bucket/move_ru/images/9288118981/0001.jpg",
        "s3://test-bucket/move_ru/images/9288118981/0002.jpg",
    ]

    assert len(dummy_session.requested_urls) == 2
    assert dummy_session.requested_urls[0][0] == "https://example.com/1.jpg"
    assert dummy_session.requested_urls[1][0] == "https://example.com/2.jpg"

    assert len(uploader.upload_calls) == 2
    assert uploader.upload_calls[0]["object_key"] == "move_ru/images/9288118981/0001.jpg"
    assert uploader.upload_calls[1]["object_key"] == "move_ru/images/9288118981/0002.jpg"
    assert uploader.upload_calls[0]["content_type"] == "image/jpeg"