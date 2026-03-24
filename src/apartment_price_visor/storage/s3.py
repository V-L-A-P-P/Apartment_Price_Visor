from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass

import boto3
from botocore.client import BaseClient
from dotenv import load_dotenv



@dataclass(slots=True)
class S3Settings:
    bucket_name: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region_name: str = "ru-central1"
    key_prefix: str = "move_ru/images"

    @classmethod
    def from_env(cls) -> "S3Settings":
        bucket_name = os.getenv("APARTMENT_PRICE_VISOR_S3_BUCKET")
        endpoint_url = os.getenv("APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL")
        access_key_id = os.getenv("APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID")
        secret_access_key = os.getenv("APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY")
        region_name = os.getenv("APARTMENT_PRICE_VISOR_S3_REGION", "ru-central1")
        key_prefix = os.getenv("APARTMENT_PRICE_VISOR_S3_KEY_PREFIX", "move_ru/images")

        missing = []
        if not bucket_name:
            missing.append("APARTMENT_PRICE_VISOR_S3_BUCKET")
        if not endpoint_url:
            missing.append("APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL")
        if not access_key_id:
            missing.append("APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID")
        if not secret_access_key:
            missing.append("APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY")

        if missing:
            raise RuntimeError(
                "Missing required S3 environment variables: "
                + ", ".join(missing)
            )

        return cls(
            bucket_name=bucket_name,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name=region_name,
            key_prefix=key_prefix,
        )


class S3Uploader:
    def __init__(self, settings: S3Settings) -> None:
        self.settings = settings
        self.client = self._build_client()

    def _build_client(self) -> BaseClient:
        session = boto3.session.Session()
        return session.client(
            service_name="s3",
            endpoint_url=self.settings.endpoint_url,
            region_name=self.settings.region_name,
            aws_access_key_id=self.settings.access_key_id,
            aws_secret_access_key=self.settings.secret_access_key,
        )

    def build_object_key(
        self,
        *,
        listing_id: str,
        filename: str,
    ) -> str:
        prefix = self.settings.key_prefix.strip("/")
        return f"{prefix}/{listing_id}/{filename}"

    def build_s3_uri(self, object_key: str) -> str:
        return f"s3://{self.settings.bucket_name}/{object_key}"

    @staticmethod
    def guess_content_type(filename: str) -> str | None:
        content_type, _ = mimetypes.guess_type(filename)
        return content_type

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

if __name__ == "__main__":
    load_dotenv()
    import sys
    from pathlib import Path
    
    print("=" * 60)
    print("S3 Uploader - Тест загрузки файлов")
    print("=" * 60)
    
    # 1. Проверяем переменные окружения
    print("\n🔍 Проверка переменных окружения...")
    
    required_vars = [
        "APARTMENT_PRICE_VISOR_S3_BUCKET",
        "APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL",
        "APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID",
        "APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY",
    ]
    
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print("❌ Отсутствуют переменные окружения:")
        for v in missing:
            print(f"   - {v}")
        print("\n💡 Установите их перед запуском:")
        print("   export APARTMENT_PRICE_VISOR_S3_BUCKET='your-bucket-name'")
        print("   export APARTMENT_PRICE_VISOR_S3_ENDPOINT_URL='https://storage.yandexcloud.net'")
        print("   export APARTMENT_PRICE_VISOR_S3_ACCESS_KEY_ID='your-key-id'")
        print("   export APARTMENT_PRICE_VISOR_S3_SECRET_ACCESS_KEY='your-secret'")
        sys.exit(1)
    
    print("✅ Все переменные установлены")
    
    # 2. Инициализация
    print("\n🚀 Инициализация S3Uploader...")
    try:
        settings = S3Settings.from_env()
        uploader = S3Uploader(settings)
        print(f"   📦 Бакет: {settings.bucket_name}")
        print(f"   🌐 Эндпоинт: {settings.endpoint_url}")
        print(f"   📁 Префикс: {settings.key_prefix}")
        print("✅ Успешно")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
    
    # 3. Тестовая загрузка текстового файла
    print("\n📝 Тест 1: Загрузка текстового файла")
    
    test_content = b"Hello, Yandex Cloud! Test upload at " + str(os.times()).encode()
    listing_id = "test_listing_001"
    test_filename = "test.txt"
    
    object_key = uploader.build_object_key(
        listing_id=listing_id,
        filename=test_filename,
    )
    content_type = uploader.guess_content_type(test_filename)
    
    try:
        s3_uri = uploader.upload_bytes(
            content=test_content,
            object_key=object_key,
            content_type=content_type,
        )
        print(f"   ✅ Загружено: {object_key}")
        print(f"   🔗 URI: {s3_uri}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # 4. Проверка, что файл существует
    print("\n🔍 Проверка загруженного файла...")
    try:
        response = uploader.client.head_object(
            Bucket=settings.bucket_name,
            Key=object_key,
        )
        print(f"   ✅ Файл найден")
        print(f"   📏 Размер: {response['ContentLength']} байт")
        print(f"   📄 Content-Type: {response.get('ContentType', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Файл не найден: {e}")
    
    # 5. Тест загрузки изображения (если есть)
    print("\n🖼️ Тест 2: Загрузка изображения")
    
    # Проверяем наличие тестовых изображений
    test_images = ["test.jpg", "test.png", "sample.jpg", "image.png"]
    found_image = None
    
    for img in test_images:
        if Path(img).exists():
            found_image = img
            break
    
    if found_image:
        try:
            with open(found_image, "rb") as f:
                image_content = f.read()
            
            image_key = uploader.build_object_key(
                listing_id=listing_id,
                filename=found_image,
            )
            image_content_type = uploader.guess_content_type(found_image)
            
            uploader.upload_bytes(
                content=image_content,
                object_key=image_key,
                content_type=image_content_type,
            )
            print(f"   ✅ Загружено: {image_key}")
            print(f"   📏 Размер: {len(image_content)} байт")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print("   ⏭️ Нет тестовых изображений, пропускаем")
        print("   💡 Создайте test.jpg в текущей папке для теста")
    
    # 6. Список загруженных файлов
    print("\n📋 Список файлов в бакете:")
    try:
        response = uploader.client.list_objects_v2(
            Bucket=settings.bucket_name,
            Prefix=settings.key_prefix,
            MaxKeys=10,
        )
        
        if "Contents" in response:
            for obj in response["Contents"]:
                size_kb = obj["Size"] / 1024
                print(f"   📄 {obj['Key']} ({size_kb:.2f} KB)")
            if len(response["Contents"]) >= 10:
                print("   ... (показаны первые 10)")
        else:
            print("   🗂️ Бакет пуст")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Тест завершен")
    print("=" * 60)