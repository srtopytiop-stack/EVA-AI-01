"""Phase 14-A integration tests for EVA-AI-01."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.backtest_engine import BacktestConfig, BacktestEngine
from core.backtest_metrics import EquityPoint
from core.risk_engine import RiskEngine
from core.signal_engine import SignalResult
from core.system_integration import (
    IntegrationConfig,
    IntegrationError,
    run_integrated_research,
)
from core.volatility_engine import VolatilityEngine
from data.market_data import Candle, MarketDataResult


def _market_data(count: int = 180) -> MarketDataResult:
    candles: list[Candle] = []
    price = 30_000.0
    start = 1_700_000_000_000
    step = 60_000

    for i in range(count):
        # Alternating deterministic regimes make the fixture non-constant.
        change = 0.0025 if (i % 20) < 10 else -0.0020
        open_price = price
        close_price = open_price * (1.0 + change)
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        open_time = start + i * step

        candles.append(
            Candle(
                open_time=open_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=100_000.0 + (i % 17) * 1_000.0,
                close_time=open_time + step - 1,
                quote_volume=close_price * 100_000.0,
                number_of_trades=100 + i,
            )
        )
        price = close_price

    return MarketDataResult(
        symbol="BTCUSDT",
        interval="1m",
        candles=tuple(candles),
    )


def _config() -> IntegrationConfig:
    # Keep the production default architecture, but use 200 bootstrap
    # replications so this unit/integration test remains fast.
    bt = BacktestConfig(
        warmup_bars=120,
        history_window_bars=240,
        require_regular_intervals=True,
        expected_interval_ms=60_000,
    )
    return IntegrationConfig(
        backtest_config=bt,
        number_of_trials=2,
        benchmark_sharpe=0.0,
        bootstrap_replications=200,
        bootstrap_block_length=5,
        bootstrap_seed=42,
        risk_free_per_period=0.0,
    )


def test_invalid_market_data_is_rejected() -> None:
    with pytest.raises(IntegrationError):
        run_integrated_research(object(), config=_config())


def test_invalid_trial_count_is_rejected() -> None:
    with pytest.raises(IntegrationError):
        IntegrationConfig(number_of_trials=1)


def test_pipeline_reaches_statistical_and_bootstrap_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market_data()
    original = BacktestEngine._decision_at_close
    calls = {"n": 0}

    def deterministic_decision(self, *, candle, closes, volumes, equity):
        # Use the real volatility/risk engines. Only the directional signal is
        # deterministic here so the test does not depend on market-signal
        # heuristics to create a trade.
        volatility = self.volatility_engine.classify(closes)
        risk = self.risk_engine.evaluate(
            equity=equity,
            entry_price=float(candle.close),
            stop_distance=float(candle.close) * 0.02,
            volatility_ratio=max(0.1, volatility.volatility_ratio),
            config=self.config.risk_config,
        )

        calls["n"] += 1
        phase = calls["n"] % 16
        direction = 1 if phase in (1, 2) else (-1 if phase in (9, 10) else 0)
        score = 0.90 if direction == 1 else (-0.90 if direction == -1 else 0.0)

        signal = SignalResult(
            symbol=self._symbol,
            timestamp=int(candle.close_time),
            direction=direction,
            score=score,
            confidence=0.95 if direction else 0.80,
            classical_z=0.0,
            robust_z=0.0,
            volatility=volatility.ewma_volatility,
            trend_score=0.0,
            mean_reversion_score=0.0,
            volume_score=0.0,
            regime_factor=1.0,
            volatility_factor=1.0,
            reasons=("integration-test",),
        )
        return signal, risk, volatility

    monkeypatch.setattr(BacktestEngine, "_decision_at_close", deterministic_decision)

    result = run_integrated_research(market, config=_config())

    assert result.symbol == "BTCUSDT"
    assert result.interval == "1m"
    assert result.candle_count == 180
    assert result.feature_rows == 180
    assert "return" in result.feature_columns
    assert "volatility" in result.feature_columns
    assert "momentum" in result.feature_columns

    assert result.backtest.symbol == "BTCUSDT"
    assert len(result.backtest.equity_curve) == 180
    assert result.backtest.metrics.final_equity > 0.0

    assert result.statistical_validation.observations == 179
    assert result.bootstrap_validation.observations == 179
    assert result.bootstrap_validation.bootstrap_replications == 200
    assert result.bootstrap_validation.block_length == 5

    payload = result.to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["final_equity"] == result.final_equity

    with pytest.raises(AttributeError):
        result.candle_count = 999  # type: ignore[misc]
