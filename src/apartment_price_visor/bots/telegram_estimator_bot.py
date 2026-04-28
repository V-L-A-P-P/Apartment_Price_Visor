from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

import httpx
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from apartment_price_visor.config import (
    DEFAULT_BOT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_INFERENCE_API_URL,
)
from apartment_price_visor.models.similar_ads import find_similar_ads

API_URL = os.getenv("INFERENCE_API_URL", DEFAULT_INFERENCE_API_URL)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

BTN_ESTIMATE = "Оценить квартиру"
BTN_CHECK_PRICE = "Проверить мою цену"
BTN_IMPROVE_DESCRIPTION = "Улучшить описание"
BTN_SELLER_TIPS = "Советы по цене"
BTN_SIMILAR_ADS = "Похожие объявления"
BTN_CANCEL = "Отмена"


class EstimateFlow(StatesGroup):
    collecting = State()


class CheckPriceFlow(StatesGroup):
    waiting_for_price = State()


class ImproveDescriptionFlow(StatesGroup):
    waiting_for_text = State()


FieldParser = Callable[[str], Any]
FIELD_SPECS: list[tuple[str, str, FieldParser]] = [
    ("rooms_count", "Количество комнат:", float),
    ("area_total", "Общая площадь (м2):", float),
    ("living_area", "Жилая площадь (м2, можно 0):", float),
    ("kitchen_area", "Площадь кухни (м2, можно 0):", float),
    ("floor", "Этаж:", float),
    ("floors_total", "Этажность дома:", float),
    ("housing_type", "Тип жилья (new/secondary и т.п.):", str),
    ("object_type", "Тип объекта (flat/studio и т.п.):", str),
    ("renovation", "Ремонт (none/cosmetic/euro и т.п.):", str),
    ("ceiling_height_m", "Высота потолка (м, можно 0):", float),
    ("address_city", "Город:", str),
    ("address_street", "Улица:", str),
    ("nearest_metro_name", "Ближайшее метро:", str),
    ("nearest_metro_duration_min", "До метро (мин):", float),
    ("nearest_metro_distance_km", "До метро (км, можно 0):", float),
    ("description", "Описание квартиры (свободный текст):", str),
]

router = Router()


def _main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_ESTIMATE), KeyboardButton(text=BTN_CHECK_PRICE)],
            [KeyboardButton(text=BTN_IMPROVE_DESCRIPTION)],
            [KeyboardButton(text=BTN_SELLER_TIPS), KeyboardButton(text=BTN_SIMILAR_ADS)],
            [KeyboardButton(text=BTN_CANCEL)],
        ],
        resize_keyboard=True,
    )


def _parse_value(value: str, parser: FieldParser) -> Any:
    parsed = parser(value.strip())
    if parser is float and parsed == 0:
        return None
    return parsed


def _format_http_error(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ReadTimeout):
        return (
            "Сервис модели не успел ответить (таймаут). "
            "Если модель только запускается, подожди 1-2 минуты и повтори /estimate."
        )

    if isinstance(exc, httpx.ConnectError):
        return (
            "Не удается подключиться к сервису модели. "
            "Проверь, что API запущен и доступен по INFERENCE_API_URL."
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail = ""
        try:
            body = exc.response.json()
            detail = str(body.get("detail", ""))
        except ValueError:
            detail = exc.response.text.strip()
        detail = detail or "без деталей"
        return f"Сервис модели вернул ошибку {status}: {detail}"

    fallback = str(exc).strip() or exc.__class__.__name__
    return f"Неизвестная ошибка запроса: {fallback}"


async def _ask_next_question(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    idx = data.get("field_idx", 0)
    if idx >= len(FIELD_SPECS):
        await _submit_estimate(message, state)
        return
    _, prompt, _ = FIELD_SPECS[idx]
    await message.answer(prompt)


async def _submit_estimate(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    features = data.get("features", {})
    features.setdefault("listing_id", 0)

    payload = {"mode": "seller", "features": features}
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_BOT_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        await message.answer(f"Ошибка запроса к модели: {_format_http_error(exc)}")
        await state.clear()
        return

    body = response.json()
    predicted_price = body.get("predicted_price")
    price_per_m2 = body.get("price_per_m2")
    await state.update_data(last_predicted_price=predicted_price, last_features=features)
    await message.answer(
        "Оценка готова:\n"
        f"Цена: {predicted_price:,.0f} RUB\n"
        f"Цена за м2: {price_per_m2:,.0f} RUB\n\n"
        "Команды:\n"
        "/estimate - новая оценка\n"
        "/check_price - проверить твою цену\n"
        "/improve_description - улучшить описание\n"
        "/seller_tips - советы по цене\n"
        "/similar_ads - найти похожие объявления\n"
        "/cancel - отменить текущий ввод"
    )
    await state.set_state(None)


def _price_verdict(diff_ratio: float) -> str:
    if diff_ratio >= 0.15:
        return "Сильно выше модели: высокая вероятность долгой экспозиции."
    if diff_ratio >= 0.07:
        return "Немного выше модели: можно тестировать, но следить за откликом."
    if diff_ratio <= -0.12:
        return "Сильно ниже модели: можно попробовать поднять цену."
    if diff_ratio <= -0.05:
        return "Немного ниже модели: это может ускорить продажу."
    return "Близко к рыночной оценке: выглядит сбалансированно."


def _seller_price_tips(predicted_price: float, area_total: Any) -> str:
    conservative = predicted_price * 0.98
    balanced = predicted_price * 1.03
    ambitious = predicted_price * 1.08
    per_m2_hint = ""
    if isinstance(area_total, (int, float)) and area_total > 0:
        per_m2 = predicted_price / float(area_total)
        per_m2_hint = f"\nОриентир за м2: {per_m2:,.0f} RUB."
    return (
        "Рекомендации по стартовой цене:\n"
        f"- Быстрая сделка: {conservative:,.0f} RUB\n"
        f"- Сбалансированный старт: {balanced:,.0f} RUB\n"
        f"- Амбициозный старт: {ambitious:,.0f} RUB\n"
        "Если за 2-3 недели мало звонков, снижай цену на 2-3%."
        f"{per_m2_hint}"
    )


def _build_seller_description(features: dict, raw_description: str) -> str:
    city = features.get("address_city", "город")
    street = features.get("address_street", "хорошая локация")
    metro = features.get("nearest_metro_name", "метро")
    metro_min = features.get("nearest_metro_duration_min")
    rooms = features.get("rooms_count", "")
    area_total = features.get("area_total", "")
    floor = features.get("floor", "")
    floors_total = features.get("floors_total", "")
    renovation = features.get("renovation", "состояние уточняется")

    intro = (
        f"Продается {rooms}-комнатная квартира {area_total} м2 в {city}, {street}. "
        f"Этаж {floor} из {floors_total}."
    )
    location = f"До метро {metro} около {metro_min} минут."
    condition = f"Состояние: {renovation}. Документы готовы к сделке."
    selling_points = (
        "Преимущества: удобная локация, функциональная планировка, "
        "подходит для проживания и инвестиции."
    )
    call_to_action = (
        "Звоните или пишите, оперативно покажем квартиру и ответим на вопросы."
    )

    if raw_description.strip():
        note = f"Дополнительно от собственника: {raw_description.strip()}"
        return "\n".join([intro, location, condition, selling_points, note, call_to_action])
    return "\n".join([intro, location, condition, selling_points, call_to_action])


def _format_similar_ads_message(similar_ads: list[dict[str, Any]], predicted_price: float) -> str:
    if not similar_ads:
        return "Не нашел похожие объявления в текущем датасете."
    lines = ["Похожие объявления по описанию:"]
    for idx, ad in enumerate(similar_ads, start=1):
        price = ad.get("price")
        price_per_m2 = ad.get("price_per_m2")
        similarity = ad.get("similarity", 0.0)
        city = ad.get("city") or "город не указан"
        street = ad.get("street") or "улица не указана"
        metro = ad.get("metro") or "метро не указано"
        rooms = ad.get("rooms_count") or "?"
        listing_id = ad.get("listing_id")
        delta = (price - predicted_price) if price else None
        lines.append(
            f"{idx}) id={listing_id} | sim={similarity:.3f}\n"
            f"   {rooms}к, {city}, {street}, метро: {metro}\n"
            f"   Цена: {(f'{price:,.0f} RUB' if price else 'n/a')}, "
            f"м2: {(f'{price_per_m2:,.0f} RUB' if price_per_m2 else 'n/a')}, "
            f"Δ к модели: {(f'{delta:+,.0f} RUB' if delta is not None else 'n/a')}"
        )
    return "\n".join(lines)


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я бот-оценщик квартир.\n"
        "Команды для продавца:\n"
        "/estimate - получить оценку цены\n"
        "/check_price - проверить свою цену\n"
        "/improve_description - улучшить текст объявления\n"
        "/seller_tips - получить советы по стратегии цены\n"
        "/similar_ads - похожие объявления по описанию\n\n"
        "Можно пользоваться кнопками ниже.",
        reply_markup=_main_keyboard(),
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Текущий сценарий отменен. Для новой оценки нажми кнопку 'Оценить квартиру'.",
        reply_markup=_main_keyboard(),
    )


@router.message(Command("estimate"))
async def cmd_estimate(message: Message, state: FSMContext) -> None:
    await state.set_state(EstimateFlow.collecting)
    await state.set_data({"field_idx": 0, "features": {}})
    await _ask_next_question(message, state)


@router.message(Command("check_price"))
async def cmd_check_price(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if "last_predicted_price" not in data:
        await message.answer("Сначала сделай оценку через /estimate.")
        return
    await state.set_state(CheckPriceFlow.waiting_for_price)
    await message.answer("Введи цену, по которой хочешь выставить квартиру (в RUB):")


@router.message(CheckPriceFlow.waiting_for_price)
async def process_check_price(message: Message, state: FSMContext) -> None:
    raw_text = (message.text or "").replace(" ", "").replace(",", ".")
    try:
        asked_price = float(raw_text)
    except ValueError:
        await message.answer("Не удалось распознать число. Введи цену в формате 15500000.")
        return

    data = await state.get_data()
    predicted_price = float(data["last_predicted_price"])
    diff = asked_price - predicted_price
    diff_ratio = diff / predicted_price if predicted_price else 0.0
    verdict = _price_verdict(diff_ratio)
    await message.answer(
        f"Модель: {predicted_price:,.0f} RUB\n"
        f"Твоя цена: {asked_price:,.0f} RUB\n"
        f"Отклонение: {diff:+,.0f} RUB ({diff_ratio:+.1%})\n\n"
        f"Вывод: {verdict}"
    )
    await state.set_state(None)


@router.message(Command("improve_description"))
async def cmd_improve_description(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if "last_features" not in data:
        await message.answer("Сначала сделай оценку через /estimate, чтобы собрать параметры.")
        return
    await state.set_state(ImproveDescriptionFlow.waiting_for_text)
    await message.answer("Отправь текущее описание объявления. Я предложу улучшенную версию.")


@router.message(ImproveDescriptionFlow.waiting_for_text)
async def process_improve_description(message: Message, state: FSMContext) -> None:
    raw_description = message.text or ""
    data = await state.get_data()
    features = data.get("last_features", {})
    improved = _build_seller_description(features=features, raw_description=raw_description)
    await message.answer("Вариант улучшенного описания:\n\n" + improved)
    await state.set_state(None)


@router.message(Command("seller_tips"))
async def cmd_seller_tips(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if "last_predicted_price" not in data:
        await message.answer("Сначала сделай оценку через /estimate.")
        return
    predicted_price = float(data["last_predicted_price"])
    features = data.get("last_features", {})
    tips = _seller_price_tips(predicted_price, features.get("area_total"))
    await message.answer(tips)


@router.message(Command("similar_ads"))
async def cmd_similar_ads(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if "last_features" not in data or "last_predicted_price" not in data:
        await message.answer("Сначала сделай оценку через /estimate.")
        return

    features = data["last_features"]
    predicted_price = float(data["last_predicted_price"])
    await message.answer("Ищу похожие объявления по эмбеддингам описания...")
    try:
        similar_ads = await asyncio.to_thread(find_similar_ads, features, 5)
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"Не удалось найти похожие объявления: {exc}")
        return

    await message.answer(_format_similar_ads_message(similar_ads, predicted_price))


@router.message(lambda m: (m.text or "").strip() == BTN_ESTIMATE)
async def btn_estimate(message: Message, state: FSMContext) -> None:
    await cmd_estimate(message, state)


@router.message(lambda m: (m.text or "").strip() == BTN_CHECK_PRICE)
async def btn_check_price(message: Message, state: FSMContext) -> None:
    await cmd_check_price(message, state)


@router.message(lambda m: (m.text or "").strip() == BTN_IMPROVE_DESCRIPTION)
async def btn_improve_description(message: Message, state: FSMContext) -> None:
    await cmd_improve_description(message, state)


@router.message(lambda m: (m.text or "").strip() == BTN_SELLER_TIPS)
async def btn_seller_tips(message: Message, state: FSMContext) -> None:
    await cmd_seller_tips(message, state)


@router.message(lambda m: (m.text or "").strip() == BTN_SIMILAR_ADS)
async def btn_similar_ads(message: Message, state: FSMContext) -> None:
    await cmd_similar_ads(message, state)


@router.message(lambda m: (m.text or "").strip() == BTN_CANCEL)
async def btn_cancel(message: Message, state: FSMContext) -> None:
    await cmd_cancel(message, state)


@router.message(EstimateFlow.collecting)
async def collect_features(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    idx = data.get("field_idx", 0)
    if idx >= len(FIELD_SPECS):
        await _submit_estimate(message, state)
        return

    field_name, _, parser = FIELD_SPECS[idx]
    try:
        parsed = _parse_value(message.text or "", parser)
    except ValueError:
        await message.answer("Не удалось распознать значение. Попробуй еще раз.")
        return

    features = data.get("features", {})
    features[field_name] = parsed
    await state.update_data(features=features, field_idx=idx + 1)
    await _ask_next_question(message, state)


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN environment variable")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
