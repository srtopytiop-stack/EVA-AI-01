"""
EVA-AI-01
=========
Main application entry point.

Phase 2:
Configuration & Security
"""

from __future__ import annotations

from datetime import datetime, timezone

from config.settings import (
    ConfigurationError,
    configuration_summary,
    load_settings,
)


def main() -> int:
    """Start EVA-AI-01 and validate runtime configuration."""

    print("=" * 60)
    print("EVA-AI-01")
    print("System initialization")
    print("=" * 60)

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
    print("Status: CONFIG_OK")
    print("Architecture: CLEAN_REBUILD")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
