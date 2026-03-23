from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass

import boto3
from botocore.client import BaseClient


@dataclass(slots=True)
class S3Settings:
    bucket_name: str
    region_name: str | None = None
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    key_prefix: str = "move_ru/images"

    @classmethod
    def from_env(cls) -> "S3Settings":
        bucket_name = os.getenv("APARTMENT_PRICE_VISOR_S3_BUCKET")
        if not bucket_name:
            raise RuntimeError(
                "Environment variable APARTMENT_PRICE_VISOR_S3_BUCKET is required for S3 uploads"
            )

        return cls(
            bucket_name=bucket_name,
            region_name=os.getenv("APARTMENT_PRICE_VISOR_S3_REGION"),
            endpoint_url=os.getenv("APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL"),
            access_key_id=os.getenv("APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY"),
            key_prefix=os.getenv("APARTMENT_PRICE_VISOR_S3_KEY_PREFIX", "move_ru/images"),
        )


class S3Uploader:
    def __init__(self, settings: S3Settings) -> None:
        self.settings = settings
        self.client = self._build_client()

    def _build_client(self) -> BaseClient:
        session = boto3.session.Session()
        return session.client(
            "s3",
            region_name=self.settings.region_name,
            endpoint_url=self.settings.endpoint_url,
            aws_access_key_id=self.settings.access_key_id,
            aws_secret_access_key=self.settings.secret_access_key,
        )

    def upload_bytes(
        self,
        *,
        content: bytes,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        extra_args: dict[str, str] = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.put_object(
            Bucket=self.settings.bucket_name,
            Key=object_key,
            Body=content,
            **extra_args,
        )
        return self.build_s3_uri(object_key)

    def build_object_key(self, *, listing_id: str, filename: str) -> str:
        prefix = self.settings.key_prefix.strip("/")
        return f"{prefix}/{listing_id}/{filename}" if prefix else f"{listing_id}/{filename}"

    def build_s3_uri(self, object_key: str) -> str:
        return f"s3://{self.settings.bucket_name}/{object_key}"

    @staticmethod
    def guess_content_type(filename: str) -> str | None:
        content_type, _ = mimetypes.guess_type(filename)
        return content_type
