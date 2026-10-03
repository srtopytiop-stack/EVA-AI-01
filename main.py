"""
EVA-AI-01
=========
Application entry point.

Startup flow
------------
1. Validate static configuration.
2. Optionally run one complete research cycle.
3. Optionally send the resulting Phase-15-B report to Telegram.

Research runtime is disabled by default so that deployment/configuration
changes cannot accidentally trigger external data processing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os

from config.settings import (
    ConfigurationError,
    configuration_summary,
    load_settings,
)
from core.application_runtime import (
    ApplicationRuntimeConfig,
    ApplicationRuntimeError,
    run_research_cycle,
)
from core.notification_layer import (
    NotificationError,
    TelegramNotifier,
    format_phase15b_message,
)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        "EVA_RESEARCH_ENABLED must be boolean "
        "(true/false, yes/no, 1/0)"
    )


def main() -> int:
    """Validate configuration and optionally execute one research cycle."""

    print("=" * 72)
    print("EVA-AI-01")
    print("System initialization")
    print("=" * 72)

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"UTC time: {timestamp}")

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print("Status: CONFIG_ERROR")
        print(f"Reason: {exc}")
        return 1

    print()
    print("Configuration:")
    for key, value in configuration_summary(settings).items():
        print(f"  {key}: {value}")

    print()
    print("Architecture:")
    print("  Market Data")
    print("    -> Feature Engineering")
    print("    -> Backtest Engine")
    print("       -> Signal / Regime / Volatility / Risk")
    print("       -> Portfolio / Execution Simulator / Paper Trading")
    print("    -> Statistical Validation")
    print("    -> Moving-Block Bootstrap")
    print("    -> Production Gate")
    print("    -> Monitoring")
    print("    -> Phase-15-B Runtime")
    print("    -> Optional Telegram Notification")

    try:
        research_enabled = _parse_bool(
            os.getenv("EVA_RESEARCH_ENABLED"),
            default=False,
        )
    except ValueError as exc:
        print("Status: CONFIG_ERROR")
        print(f"Reason: {exc}")
        return 1

    if not research_enabled:
        print()
        print("Research runtime: DISABLED")
        print("Status: CONFIG_OK")
        print("Safety: no market-data cycle executed")
        print("=" * 72)
        return 0

    try:
        runtime_config = ApplicationRuntimeConfig.from_environment()
        run_result = run_research_cycle(runtime_config)

        print()
        print("Research runtime: COMPLETED")
        print(f"Symbol: {run_result.symbol}")
        print(f"Interval: {run_result.interval}")
        print(f"Completed candles: {run_result.market_data.count}")
        print(f"Phase-15-B status: {run_result.phase15b.status.value}")
        print(f"Release ready: {run_result.release_ready}")

        print()
        print(format_phase15b_message(run_result.phase15b))

        notifier = TelegramNotifier()
        notification = notifier.send(run_result.phase15b)
        print()
        print(f"Telegram: {notification.message}")

    except (ApplicationRuntimeError, NotificationError, ValueError) as exc:
        print()
        print("Status: RUNTIME_ERROR")
        print(f"Reason: {exc}")
        return 1

    print()
    print("Status: RUNTIME_OK")
    print("Safety: no live trading path is enabled")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
