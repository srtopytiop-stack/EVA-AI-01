"""Phase 11 historical-backtest engine tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.backtest_engine import BacktestConfig, BacktestEngine, BacktestError
from core.execution_simulator import ExecutionConfig
from core.paper_trading_engine import PaperTradingConfig
from core.portfolio_engine import PortfolioConfig
from core.risk_engine import RiskConfig
from data.market_data import Candle

BASE_TS = 1_800_000_000_000
BAR_MS = 60_000


def make_candles(count: int = 180, *, drift: float = 0.002, quote_volume: float = 10_000_000.0) -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(count):
        previous = price
        price = previous * (1.0 + drift)
        open_price = previous
        close_price = price
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        open_time = BASE_TS + index * BAR_MS
        candles.append(
            Candle(
                open_time=open_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=100_000.0,
                close_time=open_time + BAR_MS - 1,
                quote_volume=quote_volume,
                number_of_trades=1_000,
            )
        )
    return candles


def config() -> BacktestConfig:
    return BacktestConfig(
        portfolio_config=PortfolioConfig(
            initial_cash=1_000.0,
            max_total_exposure_pct=0.80,
            max_position_exposure_pct=0.25,
            min_cash_reserve_pct=0.10,
            max_positions=1,
        ),
        execution_config=ExecutionConfig(
            half_spread_bps=1.0,
            base_slippage_bps=0.5,
            volatility_slippage_factor=0.0,
            impact_coefficient_bps=0.0,
            impact_exponent=0.5,
            max_participation_rate=0.10,
            reject_when_liquidity_missing=False,
        ),
        paper_config=PaperTradingConfig(
            fee_rate=0.001,
            min_signal_confidence=0.0,
        ),
        risk_config=RiskConfig(
            risk_per_trade_pct=0.01,
            max_exposure_pct=0.25,
            target_volatility_ratio=1.0,
            minimum_volatility_multiplier=0.25,
            maximum_volatility_multiplier=1.0,
            maximum_stop_distance_pct=0.20,
        ),
        warmup_bars=70,
        history_window_bars=120,
        signal_entry_threshold=0.20,
        signal_neutral_threshold=0.05,
    )


def test_engine_produces_results_and_forces_final_liquidation() -> None:
    result = BacktestEngine(config()).run(make_candles(), symbol="BTCUSDT")
    assert result.bars_processed == 180
    assert result.equity_curve
    assert result.metrics.bars_processed == 180
    assert result.final_position_open is False


def test_execution_happens_on_next_bar_open() -> None:
    candles = make_candles()
    result = BacktestEngine(config()).run(candles, symbol="BTCUSDT")
    opens = [event for event in result.events if event.action.value == "OPEN"]
    assert opens, "synthetic trend should create at least one entry"
    first_open = opens[0]
    signal_bar = next(
        index for index, candle in enumerate(candles)
        if candle.open_time < first_open.timestamp and candle.close_time >= first_open.timestamp - 1
    )
    assert first_open.timestamp == candles[signal_bar + 1].open_time


def test_next_bar_full_volume_is_not_used_as_entry_liquidity() -> None:
    original = make_candles()
    altered = make_candles()
    altered[70] = replace(altered[70], quote_volume=1.0)
    result_original = BacktestEngine(config()).run(original, symbol="BTCUSDT")
    result_altered = BacktestEngine(config()).run(altered, symbol="BTCUSDT")
    assert result_original.events
    assert result_altered.events
    assert result_original.events[0].timestamp == result_altered.events[0].timestamp
    assert result_original.events[0].execution_price == pytest.approx(result_altered.events[0].execution_price)


def test_future_data_cannot_change_prior_events() -> None:
    original = make_candles()
    altered = make_candles()
    change_index = 130
    for index in range(change_index, len(altered)):
        candle = altered[index]
        altered[index] = replace(
            candle,
            open=candle.open * 1.50,
            high=candle.high * 1.50,
            low=candle.low * 1.50,
            close=candle.close * 1.50,
            quote_volume=1.0,
        )
    baseline = BacktestEngine(config()).run(original, symbol="BTCUSDT")
    changed = BacktestEngine(config()).run(altered, symbol="BTCUSDT")
    cutoff = altered[change_index].open_time
    baseline_prior = [event.to_dict() for event in baseline.events if event.timestamp < cutoff]
    changed_prior = [event.to_dict() for event in changed.events if event.timestamp < cutoff]
    assert baseline_prior == changed_prior


def test_irregular_candles_are_rejected() -> None:
    candles = make_candles()
    candles[100] = replace(
        candles[100],
        open_time=candles[100].open_time + 1_000,
        close_time=candles[100].close_time + 1_000,
    )
    with pytest.raises(BacktestError):
        BacktestEngine(config()).run(candles, symbol="BTCUSDT")


def test_history_window_is_never_smaller_than_warmup() -> None:
    with pytest.raises(BacktestError):
        BacktestConfig(warmup_bars=100, history_window_bars=90)
