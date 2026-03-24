import pytest

from apartment_price_visor.storage.s3 import S3Settings, S3Uploader


def test_s3_settings_from_env_success(monkeypatch):
    monkeypatch.setenv("APARTMENT_PRICE_VISOR_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    monkeypatch.setenv("APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setenv("APARTMENT_PRICE_VISOR_S3_REGION", "ru-central1")
    monkeypatch.setenv("APARTMENT_PRICE_VISOR_S3_KEY_PREFIX", "move_ru/images")

    settings = S3Settings.from_env()

    assert settings.bucket_name == "test-bucket"
    assert settings.endpoint_url == "https://storage.yandexcloud.net"
    assert settings.access_key_id == "AKIA_TEST"
    assert settings.secret_access_key == "SECRET_TEST"
    assert settings.region_name == "ru-central1"
    assert settings.key_prefix == "move_ru/images"


def test_s3_settings_from_env_missing(monkeypatch):
    monkeypatch.delenv("APARTMENT_PRICE_VISOR_S3_BUCKET", raising=False)
    monkeypatch.delenv("APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc:
        S3Settings.from_env()

    msg = str(exc.value)
    assert "APARTMENT_PRICE_VISOR_S3_BUCKET" in msg
    assert "APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL" in msg
    assert "APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID" in msg
    assert "APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY" in msg


def test_s3_uploader_upload_bytes_calls_put_object(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = []

        def put_object(self, **kwargs):
            self.calls.append(kwargs)

    dummy_client = DummyClient()

    def fake_build_client(self):
        return dummy_client

    monkeypatch.setattr(S3Uploader, "_build_client", fake_build_client)

    settings = S3Settings(
        bucket_name="test-bucket",
        endpoint_url="https://storage.yandexcloud.net",
        access_key_id="AKIA_TEST",
        secret_access_key="SECRET_TEST",
        region_name="ru-central1",
        key_prefix="move_ru/images",
    )
    uploader = S3Uploader(settings)

    uri = uploader.upload_bytes(
        content=b"hello",
        object_key="move_ru/images/123/0001.jpg",
        content_type="image/jpeg",
    )

    assert uri == "s3://test-bucket/move_ru/images/123/0001.jpg"
    assert len(dummy_client.calls) == 1
    assert dummy_client.calls[0]["Bucket"] == "test-bucket"
    assert dummy_client.calls[0]["Key"] == "move_ru/images/123/0001.jpg"
    assert dummy_client.calls[0]["Body"] == b"hello"
    assert dummy_client.calls[0]["ContentType"] == "image/jpeg"