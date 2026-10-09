"""
EVA-AI-01 - Application Runtime

Research-only runtime supporting one or multiple explicitly configured
Binance Spot symbols. This module never places exchange orders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import os
from typing import Any

from core.phase15b_runtime import Phase15BConfig, Phase15BResult, run_phase15b
from core.system_integration import IntegrationConfig
from data.market_data import BinanceMarketDataClient, MarketDataResult


class ApplicationRuntimeError(RuntimeError):
    """Raised when the application runtime cannot execute safely."""


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _parse_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ApplicationRuntimeError(
            f"{name} must be an integer; received {raw!r}"
        ) from exc

    if value < minimum:
        raise ApplicationRuntimeError(
            f"{name} must be >= {minimum}; received {value}"
        )
    return value


def _parse_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ApplicationRuntimeError(
            f"{name} must be numeric; received {raw!r}"
        ) from exc


def _parse_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "true" if default else "false").lower()

    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False

    raise ApplicationRuntimeError(
        f"{name} must be boolean (true/false, yes/no, 1/0)"
    )


def _normalize_symbols(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize symbols and reject empty values or duplicates."""

    if isinstance(values, (str, bytes)):
        raise ApplicationRuntimeError(
            "symbols must be a sequence of symbols, not a single string"
        )

    try:
        normalized = tuple(value.strip().upper() for value in values)
    except (AttributeError, TypeError) as exc:
        raise ApplicationRuntimeError(
            "symbols must contain strings"
        ) from exc

    if not normalized:
        raise ApplicationRuntimeError("symbols cannot be empty")

    for symbol in normalized:
        if not symbol or not symbol.isalnum():
            raise ApplicationRuntimeError(
                f"invalid symbol: {symbol!r}"
            )

    if len(set(normalized)) != len(normalized):
        raise ApplicationRuntimeError(
            "symbols must not contain duplicates"
        )

    return normalized


@dataclass(frozen=True)
class ApplicationRuntimeConfig:
    """Immutable configuration for the research runtime."""

    symbol: str = "BTCUSDT"
    interval: str = "1m"
    market_data_limit: int = 500
    minimum_candles: int = 240

    number_of_trials: int = 2
    benchmark_sharpe: float = 0.0

    bootstrap_replications: int = 200
    bootstrap_block_length: int = 5
    bootstrap_seed: int | None = 42
    risk_free_per_period: float = 0.0

    strict: bool = False

    # Optional explicit multi-symbol list. An empty tuple preserves the
    # legacy single-symbol behavior through the "symbol" field.
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str):
            raise ApplicationRuntimeError("symbol must be a string")

        symbol = self.symbol.strip().upper()
        interval = self.interval.strip()

        if not symbol or not symbol.isalnum():
            raise ApplicationRuntimeError(
                "symbol must be a non-empty alphanumeric string"
            )
        if not interval:
            raise ApplicationRuntimeError("interval must be non-empty")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "interval", interval)

        if self.symbols:
            normalized_symbols = _normalize_symbols(self.symbols)
            object.__setattr__(self, "symbols", normalized_symbols)
        else:
            object.__setattr__(self, "symbols", ())

        if self.market_data_limit < self.minimum_candles:
            raise ApplicationRuntimeError(
                "market_data_limit must be >= minimum_candles"
            )
        if self.minimum_candles < 120:
            raise ApplicationRuntimeError(
                "minimum_candles must be >= 120 for the current backtest warm-up"
            )
        if self.number_of_trials < 2:
            raise ApplicationRuntimeError("number_of_trials must be >= 2")
        if self.bootstrap_replications < 200:
            raise ApplicationRuntimeError(
                "bootstrap_replications must be >= 200"
            )
        if self.bootstrap_block_length < 2:
            raise ApplicationRuntimeError(
                "bootstrap_block_length must be >= 2"
            )

    @property
    def effective_symbols(self) -> tuple[str, ...]:
        """Return configured symbols, preserving legacy single-symbol use."""
        return self.symbols or (self.symbol,)

    @classmethod
    def from_environment(cls) -> "ApplicationRuntimeConfig":
        """Load configuration from environment variables.

        EVA_SYMBOLS takes precedence when supplied, for example:
        EVA_SYMBOLS=BTCUSDT,ETHUSDT
        Otherwise EVA_SYMBOL retains its original behavior.
        """

        seed_raw = _env("EVA_BOOTSTRAP_SEED", "42")
        if seed_raw.lower() in {"none", "null"}:
            bootstrap_seed = None
        else:
            try:
                bootstrap_seed = int(seed_raw)
            except ValueError as exc:
                raise ApplicationRuntimeError(
                    "EVA_BOOTSTRAP_SEED must be an integer or 'none'"
                ) from exc

        symbols_raw = os.getenv("EVA_SYMBOLS", "").strip()
        if symbols_raw:
            symbols = _normalize_symbols(
                tuple(part.strip() for part in symbols_raw.split(","))
            )
            primary_symbol = symbols[0]
        else:
            symbols = ()
            primary_symbol = _env("EVA_SYMBOL", "BTCUSDT")

        return cls(
            symbol=primary_symbol,
            symbols=symbols,
            interval=_env("EVA_INTERVAL", "1m"),
            market_data_limit=_parse_int(
                "EVA_MARKET_DATA_LIMIT", 500, minimum=120
            ),
            minimum_candles=_parse_int(
                "EVA_MINIMUM_CANDLES", 240, minimum=120
            ),
            number_of_trials=_parse_int(
                "EVA_NUMBER_OF_TRIALS", 2, minimum=2
            ),
            benchmark_sharpe=_parse_float(
                "EVA_BENCHMARK_SHARPE", 0.0
            ),
            bootstrap_replications=_parse_int(
                "EVA_BOOTSTRAP_REPLICATIONS", 200, minimum=200
            ),
            bootstrap_block_length=_parse_int(
                "EVA_BOOTSTRAP_BLOCK_LENGTH", 5, minimum=2
            ),
            bootstrap_seed=bootstrap_seed,
            risk_free_per_period=_parse_float(
                "EVA_RISK_FREE_PER_PERIOD", 0.0
            ),
            strict=_parse_bool("EVA_RUNTIME_STRICT", False),
        )


@dataclass(frozen=True)
class ApplicationRuntimeResult:
    """Auditable output of one complete research cycle."""

    started_at_utc: str
    finished_at_utc: str
    market_data: MarketDataResult
    phase15b: Phase15BResult

    @property
    def symbol(self) -> str:
        return self.market_data.symbol

    @property
    def interval(self) -> str:
        return self.market_data.interval

    @property
    def release_ready(self) -> bool:
        return self.phase15b.release_ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "symbol": self.symbol,
            "interval": self.interval,
            "market_data_count": self.market_data.count,
            "last_completed_candle": (
                self.market_data.last_candle.open_time
                if self.market_data.last_candle is not None
                else None
            ),
            "phase15b": self.phase15b.to_dict(),
        }


def build_phase15b_config(
    config: ApplicationRuntimeConfig,
) -> Phase15BConfig:
    """Translate application settings into the Phase-15-B contract."""

    integration_config = IntegrationConfig(
        number_of_trials=config.number_of_trials,
        benchmark_sharpe=config.benchmark_sharpe,
        bootstrap_replications=config.bootstrap_replications,
        bootstrap_block_length=config.bootstrap_block_length,
        bootstrap_seed=config.bootstrap_seed,
        risk_free_per_period=config.risk_free_per_period,
    )

    return Phase15BConfig(
        integration_config=integration_config,
        strict=config.strict,
    )


def run_research_cycle(
    config: ApplicationRuntimeConfig | None = None,
    *,
    market_client: BinanceMarketDataClient | None = None,
) -> ApplicationRuntimeResult:
    """Run one research cycle for config.symbol using public Spot data."""

    cfg = config or ApplicationRuntimeConfig.from_environment()
    client = market_client or BinanceMarketDataClient()
    started = datetime.now(timezone.utc).isoformat()

    try:
        market_data = client.get_klines(
            symbol=cfg.symbol,
            interval=cfg.interval,
            limit=cfg.market_data_limit,
        )
    except Exception as exc:
        raise ApplicationRuntimeError(
            f"market-data acquisition failed for {cfg.symbol}: {exc}"
        ) from exc

    if market_data.count < cfg.minimum_candles:
        raise ApplicationRuntimeError(
            f"{cfg.symbol}: only {market_data.count} completed candles "
            f"were available; minimum required is {cfg.minimum_candles}"
        )

    phase15b_config = build_phase15b_config(cfg)

    try:
        phase15b = run_phase15b(
            market_data,
            config=phase15b_config,
        )
    except Exception as exc:
        raise ApplicationRuntimeError(
            f"research pipeline execution failed for {cfg.symbol}: {exc}"
        ) from exc

    finished = datetime.now(timezone.utc).isoformat()

    return ApplicationRuntimeResult(
        started_at_utc=started,
        finished_at_utc=finished,
        market_data=market_data,
        phase15b=phase15b,
    )


def run_research_cycles(
    config: ApplicationRuntimeConfig | None = None,
    *,
    market_client: BinanceMarketDataClient | None = None,
) -> tuple[ApplicationRuntimeResult, ...]:
    """Run one independent research cycle per configured symbol.

    Results are returned in the same order as the configured symbols.
    If a symbol fails, the function raises an error rather than silently
    presenting a partial batch as complete.
    """

    cfg = config or ApplicationRuntimeConfig.from_environment()
    results: list[ApplicationRuntimeResult] = []

    for symbol in cfg.effective_symbols:
        single_symbol_config = replace(
            cfg,
            symbol=symbol,
            symbols=(),
        )
        result = run_research_cycle(
            single_symbol_config,
            market_client=market_client,
        )
        results.append(result)

    return tuple(results)


__all__ = [
    "ApplicationRuntimeConfig",
    "ApplicationRuntimeError",
    "ApplicationRuntimeResult",
    "build_phase15b_config",
    "run_research_cycle",
    "run_research_cycles",
]
