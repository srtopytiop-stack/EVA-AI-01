
"""
EVA-AI-01
=========
Application entry point.

Startup flow
------------
1. Validate static configuration.
2. Keep research disabled by default.
3. When enabled, run a research cycle for each configured Spot symbol.
4. Print an independent report for each symbol.
5. Send an independent Telegram diagnostic report for each symbol.

Safety
------
- Binance public market data only.
- Spot research / paper analysis only.
- No live orders are placed.
- Telegram does not bypass the Production Gate.
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
    run_research_cycles,
)
from core.notification_layer import (
    NotificationError,
    TelegramNotifier,
    format_phase15b_message,
)


def _parse_bool(
    value: str | None,
    default: bool = False,
) -> bool:
    """Parse a boolean environment variable safely."""

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
    """Validate configuration and optionally run multi-asset research."""

    print("=" * 72)
    print("EVA-AI-01")
    print("System initialization")
    print("=" * 72)

    print(
        "UTC time: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )

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
    print("  Binance Spot Public Market Data")
    print("    -> Multi-Asset Research Runtime")
    print("    -> Feature Engineering")
    print("    -> Backtest / Paper Research")
    print("    -> Statistical Validation")
    print("    -> Moving-Block Bootstrap")
    print("    -> Production Gate")
    print("    -> Monitoring")
    print("    -> Per-Asset Research Report")
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
        runtime_config = (
            ApplicationRuntimeConfig.from_environment()
        )

        symbols = runtime_config.effective_symbols

        print()
        print("Research runtime: ENABLED")
        print(f"Configured symbols: {len(symbols)}")

        for symbol in symbols:
            print(f"  - {symbol}")

        # All configured symbols are processed independently.
        # The runtime raises an error if a symbol fails; a partial batch
        # is never silently presented as a complete research run.
        results = run_research_cycles(runtime_config)

        if not results:
            raise ApplicationRuntimeError(
                "The multi-asset runtime returned no results"
            )

        print()
        print(f"Research cycles completed: {len(results)}")

        ready_count = sum(
            1
            for result in results
            if result.release_ready
        )

        print(f"Release-ready research results: {ready_count}")
        print(f"Blocked research results: {len(results) - ready_count}")

        # Construct one notifier and use it for each asset report.
        notifier = TelegramNotifier()
        notification_failures = 0

        for index, result in enumerate(results, start=1):
            print()
            print("=" * 72)
            print(f"Research result {index}/{len(results)}")
            print(f"Symbol: {result.symbol}")
            print(f"Interval: {result.interval}")
            print(
                "Completed candles: "
                f"{result.market_data.count}"
            )
            print(
                "Phase-15-B status: "
                f"{result.phase15b.status.value}"
            )
            print(f"Release ready: {result.release_ready}")
            print("-" * 72)

            # The existing formatter includes the symbol, interval,
            # validation results, and Production Gate diagnostics.
            report = format_phase15b_message(result.phase15b)
            print(report)

            # A notification failure for one symbol must not prevent
            # attempts for the remaining symbols.
            try:
                notification = notifier.send(result.phase15b)
                print()
                print(f"Telegram [{result.symbol}]: {notification.message}")

            except NotificationError as exc:
                notification_failures += 1
                print()
                print(f"Telegram [{result.symbol}]: FAILED")
                print(f"Reason: {exc}")

        print()
        print("=" * 72)

        if notification_failures:
            print("Status: RUNTIME_COMPLETED_WITH_NOTIFICATION_ERRORS")
            print(
                "Notification failures: "
                f"{notification_failures}"
            )
            print("Safety: no live trading path was invoked")
            print("=" * 72)
            return 1

    except (
        ApplicationRuntimeError,
        NotificationError,
        ValueError,
    ) as exc:
        print()
        print("Status: RUNTIME_ERROR")
        print(f"Reason: {exc}")
        print("Safety: no live order was placed")
        return 1

    print()
    print("Status: RUNTIME_OK")
    print("Safety: research/paper analysis only")
    print("Safety: no live orders were placed")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
