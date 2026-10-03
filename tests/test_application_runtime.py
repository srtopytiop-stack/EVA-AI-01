"""Tests for the final research application runtime boundary."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from core.application_runtime import (
    ApplicationRuntimeConfig,
    ApplicationRuntimeError,
    build_phase15b_config,
    run_research_cycle,
)
from core.phase15b_runtime import Phase15BStatus
from data.market_data import Candle, MarketDataResult


def _market_data(count: int = 240) -> MarketDataResult:
    candles = []
    price = 30_000.0

    for index in range(count):
        change = 0.001 if index % 2 == 0 else -0.0005
        open_price = price
        close_price = open_price * (1.0 + change)
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        open_time = 1_700_000_000_000 + index * 60_000

        candles.append(
            Candle(
                open_time=open_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=1_000_000.0,
                close_time=open_time + 59_999,
                quote_volume=close_price * 1_000_000.0,
                number_of_trades=1_000,
            )
        )
        price = close_price

    return MarketDataResult(
        symbol="BTCUSDT",
        interval="1m",
        candles=tuple(candles),
    )


def test_runtime_config_rejects_insufficient_market_data_limit() -> None:
    with pytest.raises(ApplicationRuntimeError):
        ApplicationRuntimeConfig(
            market_data_limit=120,
            minimum_candles=240,
        )


def test_runtime_config_maps_to_phase15b() -> None:
    config = ApplicationRuntimeConfig(
        bootstrap_replications=200,
        bootstrap_block_length=5,
    )
    phase_config = build_phase15b_config(config)

    assert phase_config.strict is False
    assert phase_config.integration_config is not None
    assert phase_config.integration_config.bootstrap_replications == 200
    assert phase_config.integration_config.bootstrap_block_length == 5


def test_runtime_rejects_short_market_data() -> None:
    client = Mock()
    client.get_klines.return_value = _market_data(120)

    config = ApplicationRuntimeConfig(
        market_data_limit=240,
        minimum_candles=240,
    )

    with pytest.raises(ApplicationRuntimeError, match="completed candles"):
        run_research_cycle(config, market_client=client)


def test_runtime_passes_valid_data_to_phase15b(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _market_data(240)
    client = Mock()
    client.get_klines.return_value = data

    from core import application_runtime as runtime

    class DummyPhaseResult:
        status = Phase15BStatus.READY
        release_ready = True

        def to_dict(self):
            return {"status": "READY", "release_ready": True}

    config = ApplicationRuntimeConfig(
        market_data_limit=240,
        minimum_candles=240,
    )

    monkeypatch.setattr(
        runtime,
        "run_phase15b",
        lambda market_data, config=None: (
            DummyPhaseResult()
            if market_data is data and config is not None
            else (_ for _ in ()).throw(
                AssertionError("wrong runtime arguments")
            )
        ),
    )

    result = run_research_cycle(config, market_client=client)

    assert result.market_data is data
    assert result.release_ready is True
    assert result.phase15b.status is Phase15BStatus.READY


def test_runtime_wraps_market_data_errors() -> None:
    client = Mock()
    client.get_klines.side_effect = RuntimeError("network")

    with pytest.raises(
        ApplicationRuntimeError,
        match="market-data acquisition failed",
    ):
        run_research_cycle(
            ApplicationRuntimeConfig(
                market_data_limit=240,
                minimum_candles=240,
            ),
            market_client=client,
        )
