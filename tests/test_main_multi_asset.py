
from __future__ import annotations

from types import SimpleNamespace

import pytest

import main as app_main
from core.notification_layer import NotificationError


def _runtime_result(symbol: str):
    phase15b = SimpleNamespace(
        symbol=symbol,
        status=SimpleNamespace(value="GATE_BLOCKED"),
        release_ready=False,
    )

    return SimpleNamespace(
        symbol=symbol,
        interval="1m",
        market_data=SimpleNamespace(count=240),
        phase15b=phase15b,
        release_ready=False,
    )


def _enable_research(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "EVA_RESEARCH_ENABLED",
        "true",
    )

    monkeypatch.setattr(
        app_main,
        "load_settings",
        lambda: object(),
    )

    monkeypatch.setattr(
        app_main,
        "configuration_summary",
        lambda settings: {
            "environment": "test",
            "trading_enabled": False,
            "live_trading_enabled": False,
            "exchange": "binance",
            "trading_type": "spot",
        },
    )

    monkeypatch.setattr(
        app_main.ApplicationRuntimeConfig,
        "from_environment",
        classmethod(
            lambda cls: cls(
                symbols=("ETHUSDT", "SOLUSDT"),
                symbol="ETHUSDT",
                market_data_limit=240,
                minimum_candles=240,
            )
        ),
    )


def test_main_processes_and_reports_all_symbols(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _enable_research(monkeypatch)

    results = (
        _runtime_result("ETHUSDT"),
        _runtime_result("SOLUSDT"),
    )

    runtime_config = (
        app_main.ApplicationRuntimeConfig.from_environment()
    )

    calls = []

    def fake_cycles(config):
        calls.append(config)
        return results

    monkeypatch.setattr(
        app_main,
        "run_research_cycles",
        fake_cycles,
    )

    monkeypatch.setattr(
        app_main,
        "format_phase15b_message",
        lambda result: f"REPORT::{result.symbol}",
    )

    sent_symbols = []

    class FakeNotifier:
        def send(self, result):
            sent_symbols.append(result.symbol)

            return SimpleNamespace(
                message=f"sent {result.symbol}"
            )

    monkeypatch.setattr(
        app_main,
        "TelegramNotifier",
        FakeNotifier,
    )

    exit_code = app_main.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert calls == [runtime_config]
    assert sent_symbols == ["ETHUSDT", "SOLUSDT"]

    assert "Symbol: ETHUSDT" in output
    assert "Symbol: SOLUSDT" in output
    assert "REPORT::ETHUSDT" in output
    assert "REPORT::SOLUSDT" in output
    assert "Research cycles completed: 2" in output


def test_main_keeps_research_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(
        "EVA_RESEARCH_ENABLED",
        raising=False,
    )

    monkeypatch.setattr(
        app_main,
        "load_settings",
        lambda: object(),
    )

    monkeypatch.setattr(
        app_main,
        "configuration_summary",
        lambda settings: {},
    )

    monkeypatch.setattr(
        app_main,
        "run_research_cycles",
        lambda config: pytest.fail(
            "research must remain disabled by default"
        ),
    )

    exit_code = app_main.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Research runtime: DISABLED" in output


def test_main_continues_notification_attempts_after_one_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _enable_research(monkeypatch)

    results = (
        _runtime_result("ETHUSDT"),
        _runtime_result("SOLUSDT"),
    )

    monkeypatch.setattr(
        app_main,
        "run_research_cycles",
        lambda config: results,
    )

    monkeypatch.setattr(
        app_main,
        "format_phase15b_message",
        lambda result: f"REPORT::{result.symbol}",
    )

    attempted = []

    class FakeNotifier:
        def send(self, result):
            attempted.append(result.symbol)

            if result.symbol == "ETHUSDT":
                raise NotificationError(
                    "simulated Telegram failure"
                )

            return SimpleNamespace(
                message="sent SOLUSDT"
            )

    monkeypatch.setattr(
        app_main,
        "TelegramNotifier",
        FakeNotifier,
    )

    exit_code = app_main.main()
    output = capsys.readouterr().out

    assert attempted == ["ETHUSDT", "SOLUSDT"]
    assert exit_code == 1
    assert "Telegram [ETHUSDT]: FAILED" in output
    assert "Telegram [SOLUSDT]: sent SOLUSDT" in output
