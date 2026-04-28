from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apartment Price Visor entrypoint")
    parser.add_argument(
        "--help-services",
        action="store_true",
        help="Show commands for API and Telegram bot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.help_services:
        print("Run inference API:")
        print("  uvicorn apartment_price_visor.api.inference_api:app --host 0.0.0.0 --port 8000")
        print()
        print("Run Telegram bot:")
        print("  TELEGRAM_BOT_TOKEN=... INFERENCE_API_URL=http://localhost:8000/v1/predict "
              "python -m apartment_price_visor.bots.telegram_estimator_bot")
        return
    print("Use --help-services to see service startup commands.")


if __name__ == "__main__":
    main()
