"""
EVA-AI-01
=========
Central configuration and security boundary.

Design principles:
- Secrets come only from environment variables.
- No API keys or tokens are stored in source code.
- Configuration is validated at startup.
- Trading mode is explicit.
- Live trading is disabled by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigurationError(RuntimeError):
    """Raised when EVA-AI-01 configuration is invalid."""


def _get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable without exposing its value."""
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return value.strip()


def _require_env(name: str) -> str:
    """Require a non-empty environment variable."""
    value = _get_env(name)

    if value is None:
        raise ConfigurationError(
            f"Required environment variable is missing: {name}"
        )

    return value


def _parse_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable safely."""
    value = _get_env(name)

    if value is None:
        return default

    normalized = value.lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ConfigurationError(
        f"{name} must be a boolean value "
        "(true/false, yes/no, 1/0)."
    )


def _validate_mode(mode: str) -> str:
    """Validate the system operating mode."""
    allowed_modes = {
        "development",
        "paper",
        "testnet",
        "live",
    }

    normalized = mode.lower()

    if normalized not in allowed_modes:
        raise ConfigurationError(
            f"Invalid EVA_ENV={mode!r}. "
            f"Allowed values: {sorted(allowed_modes)}"
        )

    return normalized


@dataclass(frozen=True)
class Settings:
    """
    Immutable runtime configuration.

    The object intentionally contains no secret values by default.
    Secrets will be introduced only by the modules that actually need them.
    """

    environment: str
    log_level: str

    trading_enabled: bool
    live_trading_enabled: bool

    exchange: str
    trading_type: str

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build and validate settings from environment variables."""

        environment = _validate_mode(
            _get_env("EVA_ENV", "development")
        )

        log_level = _get_env("LOG_LEVEL", "INFO").upper()

        allowed_log_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if log_level not in allowed_log_levels:
            raise ConfigurationError(
                f"Invalid LOG_LEVEL={log_level!r}. "
                f"Allowed values: {sorted(allowed_log_levels)}"
            )

        exchange = _get_env("EXCHANGE", "binance").lower()

        if exchange != "binance":
            raise ConfigurationError(
                "Unsupported exchange. "
                "EVA-AI-01 currently supports Binance only."
            )

        trading_type = _get_env("TRADING_TYPE", "spot").lower()

        if trading_type != "spot":
            raise ConfigurationError(
                "Only Spot trading is permitted by EVA-AI-01."
            )

        trading_enabled = _parse_bool(
            "TRADING_ENABLED",
            default=False,
        )

        live_trading_enabled = _parse_bool(
            "LIVE_TRADING_ENABLED",
            default=False,
        )

        # Safety invariant:
        # Live trading cannot be enabled unless the system is explicitly
        # running in live mode.
        if live_trading_enabled and environment != "live":
            raise ConfigurationError(
                "LIVE_TRADING_ENABLED=true requires EVA_ENV=live."
            )

        # Another safety invariant:
        # The current architecture does not permit live execution yet.
        if live_trading_enabled:
            raise ConfigurationError(
                "Live trading is intentionally disabled during "
                "the EVA-AI-01 development phase."
            )

        return cls(
            environment=environment,
            log_level=log_level,
            trading_enabled=trading_enabled,
            live_trading_enabled=live_trading_enabled,
            exchange=exchange,
            trading_type=trading_type,
        )


def load_settings() -> Settings:
    """Public configuration loader."""
    return Settings.from_environment()


def configuration_summary(settings: Settings) -> dict[str, object]:
    """
    Return a safe configuration summary.

    Never include API keys, tokens, secrets, or credentials.
    """

    return {
        "environment": settings.environment,
        "log_level": settings.log_level,
        "trading_enabled": settings.trading_enabled,
        "live_trading_enabled": settings.live_trading_enabled,
        "exchange": settings.exchange,
        "trading_type": settings.trading_type,
    }


if __name__ == "__main__":
    settings = load_settings()

    print("EVA-AI-01 configuration check")
    print("=" * 40)

    for key, value in configuration_summary(settings).items():
        print(f"{key}: {value}")

    print("=" * 40)
    print("CONFIG_OK")
