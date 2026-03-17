import json
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MoveRuListingScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

    MONTHS_RU = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }

    def __init__(self) -> None:
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
    def _clean_text(text: str | None) -> str | None:
        if text is None:
            return None
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    @staticmethod
    def _normalize_units(text: str | None) -> str | None:
        if text is None:
            return None
        return (
            text.replace("м²", "м2")
            .replace("м 2", "м2")
            .replace("за м²", "за м2")
            .replace("за м 2", "за м2")
        )

    @staticmethod
    def _parse_int_price(text: str | None) -> int | None:
        if not text:
            return None
        match = re.search(r"([\d\s]+)\s*₽", text)
        if not match:
            return None
        digits = re.sub(r"\D", "", match.group(1))
        return int(digits) if digits else None

    @staticmethod
    def _parse_float_area(text: str | None) -> float | None:
        if not text:
            return None
        text = text.replace(",", ".")
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_int_from_text(text: str | None) -> int | None:
        if not text:
            return None
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else None

    @staticmethod
    def _extract_listing_id(url: str) -> str | None:
        match = re.search(r"_(\d+)/?$", url)
        return match.group(1) if match else None

    @staticmethod
    def _parse_floor_info(text: str | None) -> tuple[int | None, int | None]:
        if not text:
            return None, None
        match = re.match(r"(\d+)\s*/\s*(\d+)", text)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))

    def _parse_russian_date(self, text: str | None) -> str | None:
        if not text:
            return None

        text = text.lower()
        match = re.search(r"(\d{1,2})\s+([а-яё]+),\s*(\d{4})", text)
        if not match:
            return None

        day = int(match.group(1))
        month = self.MONTHS_RU.get(match.group(2))
        year = int(match.group(3))

        if not month:
            return None

        return f"{year:04d}-{month:02d}-{day:02d}"

    @staticmethod
    def _parse_duration_to_minutes(text: str | None) -> int | None:
        if not text:
            return None

        hours = 0
        minutes = 0

        h_match = re.search(r"(\d+)\s*ч", text)
        m_match = re.search(r"(\d+)\s*мин", text)

        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))

        total = hours * 60 + minutes
        return total if total > 0 else None

    @staticmethod
    def _parse_distance_km(text: str | None) -> float | None:
        if not text:
            return None
        text = text.replace(",", ".")
        match = re.search(r"(\d+(?:\.\d+)?)\s*км", text.lower())
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_yes_no_flag(text: str | None) -> int | None:
        if not text:
            return None
        text = text.strip().lower()
        if text == "да":
            return 1
        if text == "нет":
            return 0
        return None

    @staticmethod
    def _parse_height_meters(text: str | None) -> float | None:
        if not text:
            return None
        text = text.replace(",", ".")
        match = re.search(r"(\d+(?:\.\d+)?)\s*м", text.lower())
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_quarter_year(text: str | None) -> str | None:
        if not text:
            return None
        match = re.search(r"(\d)\s*кв\.\s*(\d{4})", text.lower())
        if not match:
            return None
        quarter = int(match.group(1))
        year = int(match.group(2))
        return f"{year:04d}-Q{quarter}"

    def _get_section_by_title(self, soup: BeautifulSoup, title: str):
        for section in soup.select("div.card-objects-sections__section"):
            title_el = section.select_one(".card-objects-sections__title")
            if title_el and self._clean_text(title_el.get_text(" ", strip=True)) == title:
                return section
        return None

    def _validate_listing_page(self, soup: BeautifulSoup) -> None:
        title_el = soup.select_one("h1.card-objects__title")
        detailed_section = self._get_section_by_title(soup, "Подробные характеристики")

        if title_el is None:
            raise ValueError("Не найден заголовок карточки объявления")
        if detailed_section is None:
            raise ValueError("Не найден блок 'Подробные характеристики'")

    def _extract_meta(self, soup: BeautifulSoup) -> dict[str, str | None]:
        updated = None
        created = None
        views = None

        for el in soup.select("span.card-meta__item"):
            txt = self._clean_text(el.get_text(" ", strip=True))
            if not txt:
                continue
            if "Обновлено" in txt:
                updated = txt
            elif "просмотр" in txt:
                views = txt

        created_el = soup.select_one("span.card-meta__additional")
        if created_el:
            created = self._clean_text(created_el.get_text(" ", strip=True))

        return {
            "updated": updated,
            "created": created,
            "views": views,
        }

    def _extract_pairs_from_section(self, section) -> dict[str, str]:
        data: dict[str, str] = {}
        if section is None:
            return data

        for item in section.select("div.card-specifications-table__item"):
            key_el = item.select_one("span.card-specifications-table__description")
            value_el = item.select_one("span.card-specifications-table__title")

            key = self._clean_text(key_el.get_text(" ", strip=True)) if key_el else None
            value = self._clean_text(value_el.get_text(" ", strip=True)) if value_el else None

            if key and value:
                data[self._normalize_units(key)] = self._normalize_units(value)

        return data

    def _extract_description(self, soup: BeautifulSoup) -> str | None:
        section = self._get_section_by_title(soup, "Описание")
        if section is None:
            return None

        desc_el = section.select_one("div.card-objects-description__description-text")
        if desc_el:
            return self._clean_text(desc_el.get_text(" ", strip=True))

        seo_el = section.select_one("div.card-objects-description__seo-text")
        if seo_el:
            return self._clean_text(seo_el.get_text(" ", strip=True))

        return None

    def _extract_address_parts(self, soup: BeautifulSoup) -> list[str]:
        section = self._get_section_by_title(soup, "Расположение")
        if section is None:
            return []

        parts = []
        for el in section.select(
            ".card-objects-location__address-link, .card-objects-location__address-text"
        ):
            txt = self._clean_text(el.get_text(" ", strip=True))
            if txt:
                parts.append(txt)
        return parts

    def _extract_city_and_street(
        self,
        address_parts: list[str],
    ) -> tuple[str | None, str | None]:
        city = None
        street = None

        if address_parts:
            city = address_parts[0]

        for part in address_parts:
            lower = part.lower()
            if (
                "ул " in lower
                or "ул." in lower
                or "улица" in lower
                or "проспект" in lower
                or "пр-т" in lower
            ):
                street = part
                break

        return city, street

    def _extract_nearest_metro(self, soup: BeautifulSoup) -> dict[str, str | int | float | None]:
        for li in soup.select("li.card-objects-near-stations__station"):
            name_el = li.select_one(".card-objects-near-stations__station-link")
            duration_el = li.select_one(".card-objects-near-stations__station-duration")
            distance_el = li.select_one(".card-objects-near-stations__station-distance")

            name = self._clean_text(name_el.get_text(" ", strip=True)) if name_el else None
            duration = self._clean_text(duration_el.get_text(" ", strip=True)) if duration_el else None
            distance = self._clean_text(distance_el.get_text(" ", strip=True)) if distance_el else None

            if name:
                return {
                    "nearest_metro_name": name,
                    "nearest_metro_duration_min": self._parse_duration_to_minutes(duration),
                    "nearest_metro_distance_km": self._parse_distance_km(distance),
                }

        return {
            "nearest_metro_name": None,
            "nearest_metro_duration_min": None,
            "nearest_metro_distance_km": None,
        }

    def _extract_images(self, html: str, soup: BeautifulSoup, listing_id: str | None) -> list[str]:
        """
        Извлекает все URL изображений, содержащих указанный listing_id.
        """
        result = []
        seen = set()

        def add(url: str | None):
            if not url:
                return
            # Очищаем URL от экранирования и хвостовых параметров
            url = url.replace("\\/", "/").strip().split("?")[0]
            if url not in seen:
                seen.add(url)
                result.append(url)

        if not listing_id:
            return result

        # Более гибкое регулярное выражение: ищем ID в любом месте пути
        # (?:i\d+|static-i\d+) - разные домены, [^"'\s]* - любые символы кроме кавычек и пробелов
        pattern = rf'https?://(?:i\d+|static-i\d+)\.move\.ru/[^"\'\\\s]*{listing_id}[^"\'\\\s]*\.(?:jpe?g|png|webp)'
        for url in re.findall(pattern, html, re.IGNORECASE):
            add(url)

        # Если ничего не нашли, пробуем искать в data-атрибутах (часто там лежат URL)
        if not result:
            for tag in soup.find_all(attrs={"data-src": True, "data-original": True, "data-lazy": True}):
                for attr in ["data-src", "data-original", "data-lazy"]:
                    if tag.get(attr) and listing_id in tag[attr]:
                        add(tag[attr])

        # Последний шанс: Open Graph изображение, если оно содержит ID
        if not result:
            og = soup.find("meta", attrs={"property": "og:image"})
            if og and og.get("content") and listing_id in og["content"]:
                add(og["content"])

        # === ФИЛЬТР: оставляем только каждый второй элемент ===
        # Если картинки идут парами (оригинал и уменьшенная копия),
        # берём только первые из каждой пары
        filtered_result = []
        for i, url in enumerate(result):
            if i % 2 == 0:
                filtered_result.append(url)

        return filtered_result if filtered_result else result

    def _extract_object_features(self, soup: BeautifulSoup) -> list[str]:
        section = self._get_section_by_title(soup, "Особенности объекта")
        if section is None:
            return []

        features = []
        for line in section.get_text("\n", strip=True).split("\n"):
            txt = self._clean_text(line)
            if txt and txt != "Особенности объекта":
                features.append(txt)

        return features

    def _extract_complex_info(self, soup: BeautifulSoup) -> dict[str, str]:
        section = self._get_section_by_title(soup, "Подробно о ЖК")
        return self._extract_pairs_from_section(section)

    def _extract_price_history_features(self, soup: BeautifulSoup) -> dict[str, int | None]:
        text = soup.get_text("\n", strip=True)

        prices = re.findall(r"(\d[\d\s]*₽)", text)
        parsed_prices = []
        for price_text in prices:
            value = self._parse_int_price(price_text)
            if value is not None:
                parsed_prices.append(value)

        previous_price = parsed_prices[1] if len(parsed_prices) > 1 else None

        delta_abs = None
        match = re.search(r"-\s*([\d\s]+)\s*₽", text)
        if match:
            digits = re.sub(r"\D", "", match.group(1))
            delta_abs = int(digits) if digits else None

        return {
            "previous_price": previous_price,
            "last_price_drop_abs": delta_abs,
        }

    @staticmethod
    def _get_scraped_at() -> str:
        return datetime.now(timezone.utc).isoformat()

    def parse_listing(self, url: str) -> dict:
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        self._validate_listing_page(soup)

        listing_id = self._extract_listing_id(url)

        title_el = soup.select_one("h1.card-objects__title")
        title = self._clean_text(title_el.get_text(" ", strip=True)) if title_el else None

        meta = self._extract_meta(soup)
        detailed = self._extract_pairs_from_section(
            self._get_section_by_title(soup, "Подробные характеристики")
        )
        complex_info = self._extract_complex_info(soup)
        object_features = self._extract_object_features(soup)

        address_parts = self._extract_address_parts(soup)
        address_city, address_street = self._extract_city_and_street(address_parts)

        nearest_metro = self._extract_nearest_metro(soup)
        image_urls = self._extract_images(html, soup, listing_id)

        price = detailed.get("Цена")
        price_per_m2 = detailed.get("Цена за м2")
        views = self._parse_int_from_text(meta.get("views"))
        date_added = self._parse_russian_date(detailed.get("Дата добавления"))
        rooms_count = self._parse_int_from_text(detailed.get("Количество комнат"))
        area_total = self._parse_float_area(detailed.get("Общая площадь"))
        floor, floors_total = self._parse_floor_info(detailed.get("Этаж"))
        housing_type = detailed.get("Тип жилья")
        object_type = detailed.get("Тип объекта")
        description = self._extract_description(soup)

        price_history = self._extract_price_history_features(soup)

        return {
            "source": "move_ru",
            "scraped_at": self._get_scraped_at(),
            "url": url,
            "listing_id": listing_id,
            "price": self._parse_int_price(price),
            "price_per_m2": self._parse_int_price(price_per_m2),
            "views": views,
            "date_added": date_added,
            "rooms_count": rooms_count,
            "area_total": area_total,
            "living_area": self._parse_float_area(detailed.get("Жилая площадь")),
            "kitchen_area": self._parse_float_area(detailed.get("Площадь кухни")),
            "floor": floor,
            "floors_total": floors_total,
            "housing_type": housing_type,
            "object_type": object_type,
            "renovation": detailed.get("Ремонт"),
            "bathroom_type": detailed.get("Тип санузла"),
            "window_view": detailed.get("Вид из окна"),
            "floor_covering": detailed.get("Покрытие пола"),
            "ceiling_height_m": self._parse_height_meters(detailed.get("Высота потолков")),
            "elite_flag": self._parse_yes_no_flag(detailed.get("Элитная недвижимость")),
            "address_city": address_city,
            "address_street": address_street,
            "nearest_metro_name": nearest_metro["nearest_metro_name"],
            "nearest_metro_duration_min": nearest_metro["nearest_metro_duration_min"],
            "nearest_metro_distance_km": nearest_metro["nearest_metro_distance_km"],
            "residential_complex_name": complex_info.get("Название ЖК"),
            "developer_name": complex_info.get("Застройщик"),
            "housing_class": complex_info.get("Класс жилья"),
            "building_type": complex_info.get("Тип здания"),
            "building_stage": complex_info.get("Этап строительства"),
            "complex_floors_total": self._parse_int_from_text(complex_info.get("Этажность")),
            "complex_lift_flag": self._parse_yes_no_flag(complex_info.get("Лифт")),
            "parking_flag": self._parse_yes_no_flag(complex_info.get("Парковка")),
            "security_flag": self._parse_yes_no_flag(complex_info.get("Охрана")),
            "delivery_quarter": self._parse_quarter_year(complex_info.get("Срок сдачи")),
            "previous_price": price_history["previous_price"],
            "last_price_drop_abs": price_history["last_price_drop_abs"],
            "feature_list": object_features,
            "title": title,
            "description": description,
            "image_urls": image_urls,
        }


if __name__ == "__main__":
    parser = MoveRuListingScraper()

    data = parser.parse_listing(
        "https://move.ru/objects/moskva_proezd_elektrolitnyy_d_16k7_9288118981/"
    )

    print(json.dumps(data, ensure_ascii=False, indent=2))