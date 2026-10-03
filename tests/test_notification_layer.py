"""Tests for EVA-AI-01 notification boundary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.notification_layer import (
    NotificationError,
    TelegramConfig,
    TelegramNotifier,
    format_phase15b_message,
)
from core.phase15b_runtime import Phase15BStatus


@dataclass(frozen=True)
class Metrics:
    initial_equity: float = 1000.0
    final_equity: float = 1010.0
    total_return_pct: float = 1.0
    max_drawdown_pct: float = -2.0
    trade_count: int = 3
    rejected_actions: int = 1


@dataclass(frozen=True)
class Backtest:
    metrics: Metrics = Metrics()


@dataclass(frozen=True)
class Statistical:
    periodic_sharpe: float = 0.5
    probabilistic_sharpe: float = 0.6
    deflated_sharpe: float = 0.4


@dataclass(frozen=True)
class Bootstrap:
    observations: int = 39
    bootstrap_replications: int = 200
    block_length: int = 5
    sharpe_ci_lower: float = -0.1
    sharpe_ci_upper: float = 0.8
    bootstrap_positive_sharpe_fraction: float = 0.6


@dataclass(frozen=True)
class Research:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    candle_count: int = 40
    backtest: Backtest = Backtest()
    statistical_validation: Statistical = Statistical()
    bootstrap_validation: Bootstrap = Bootstrap()


@dataclass(frozen=True)
class Issue:
    severity: object
    code: str
    message: str


@dataclass(frozen=True)
class Gate:
    approved: bool = True
    critical_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Monitoring:
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class Result:
    research: Research = Research()
    production_gate: Gate = Gate()
    monitoring: Monitoring = Monitoring()
    status: Phase15BStatus = Phase15BStatus.READY

    @property
    def release_ready(self) -> bool:
        return True


def test_formatter_contains_core_metrics() -> None:
    result = Result()
    message = format_phase15b_message(result)

    assert "BTCUSDT" in message
    assert "Periodic Sharpe: 0.500000" in message
    assert "Moving-block bootstrap" in message
    assert "No live order was placed" in message


def test_disabled_telegram_is_skipped() -> None:
    notifier = TelegramNotifier(TelegramConfig(enabled=False))
    outcome = notifier.send(Result())

    assert outcome.skipped is True
    assert outcome.sent is False


def test_enabled_telegram_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(NotificationError):
        TelegramConfig.from_environment()
