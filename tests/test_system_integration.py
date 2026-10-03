"""Phase 14-A integration tests for EVA-AI-01."""

from __future__ import annotations

import pytest

from core.backtest_engine import BacktestConfig, BacktestEngine
from core.signal_engine import SignalResult
from core.system_integration import (
    IntegrationConfig,
    IntegrationError,
    run_integrated_research,
)
from data.market_data import Candle, MarketDataResult


WARMUP_BARS = 120
CANDLE_INTERVAL_MS = 60_000


def _market_data(count: int = 180) -> MarketDataResult:
    """Build deterministic OHLCV data with genuine non-zero return variance.

    The post-warmup section alternates between six +1% bars and six -1% bars.
    The integration test opens at each rising-leg start and closes at each
    falling-leg start, producing multiple actual paper trades and a changing
    mark-to-market equity curve.
    """
    candles: list[Candle] = []
    price = 30_000.0
    start = 1_700_000_000_000

    for i in range(count):
        if i < WARMUP_BARS:
            # Keep the historical window non-flat so the real volatility and
            # signal-related calculations receive a valid varying series.
            change = 0.002 if (i % 8) < 4 else -0.0015
        else:
            phase = (i - WARMUP_BARS) % 12
            change = 0.01 if phase < 6 else -0.01

        open_price = price
        close_price = open_price * (1.0 + change)
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        open_time = start + i * CANDLE_INTERVAL_MS

        candles.append(
            Candle(
                open_time=open_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=100_000.0 + (i % 17) * 1_000.0,
                close_time=open_time + CANDLE_INTERVAL_MS - 1,
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
    """Return a deterministic, fast integration-test configuration."""
    bt = BacktestConfig(
        warmup_bars=WARMUP_BARS,
        history_window_bars=240,
        require_regular_intervals=True,
        expected_interval_ms=CANDLE_INTERVAL_MS,
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

    def deterministic_decision(self, *, candle, closes, volumes, equity):
        """Keep real risk/volatility engines; control only direction.

        The direction schedule is tied to candle time rather than a mutable
        call counter, so it remains deterministic regardless of how the
        backtest requests decisions.
        """
        signal, risk, volatility = original(
            self,
            candle=candle,
            closes=closes,
            volumes=volumes,
            equity=equity,
        )

        candle_index = (
            int(candle.open_time) - 1_700_000_000_000
        ) // CANDLE_INTERVAL_MS

        if candle_index < WARMUP_BARS:
            direction = 0
            score = 0.0
            confidence = 0.80
        else:
            phase = (candle_index - WARMUP_BARS) % 12

            # The backtest executes the decision on the following bar's open.
            # Therefore the entry is prepared at the start of a rising leg and
            # the exit at the start of the following falling leg.
            if phase == 0:
                direction = 1
                score = 0.90
                confidence = 0.95
            elif phase == 6:
                direction = -1
                score = -0.90
                confidence = 0.95
            else:
                direction = 0
                score = 0.0
                confidence = 0.80

        controlled_signal = SignalResult(
            symbol=self._symbol,
            timestamp=int(candle.close_time),
            direction=direction,
            score=score,
            confidence=confidence,
            classical_z=signal.classical_z,
            robust_z=signal.robust_z,
            volatility=signal.volatility,
            trend_score=signal.trend_score,
            mean_reversion_score=signal.mean_reversion_score,
            volume_score=signal.volume_score,
            regime_factor=signal.regime_factor,
            volatility_factor=signal.volatility_factor,
            reasons=("integration-test-deterministic-direction",),
        )

        return controlled_signal, risk, volatility

    monkeypatch.setattr(
        BacktestEngine,
        "_decision_at_close",
        deterministic_decision,
    )

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

    # The test must execute actual paper trades; it must not manufacture an
    # equity curve directly for the statistical layers.
    assert len(result.backtest.trades) >= 2

    equity_values = [
        float(point.equity)
        for point in result.backtest.equity_curve
    ]
    returns = [
        current / previous - 1.0
        for previous, current in zip(
            equity_values[:-1],
            equity_values[1:],
        )
    ]

    # Statistical Validation requires positive return variance because its
    # Sharpe/PSR/DSR calculations divide by the return standard deviation.
    mean_return = sum(returns) / len(returns)
    variance = sum(
        (value - mean_return) ** 2
        for value in returns
    ) / len(returns)
    assert variance > 1e-16

    assert result.statistical_validation.observations == 179
    assert result.bootstrap_validation.observations == 179
    assert result.bootstrap_validation.bootstrap_replications == 200
    assert result.bootstrap_validation.block_length == 5

    payload = result.to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["final_equity"] == result.final_equity

    with pytest.raises(AttributeError):
        result.candle_count = 999  # type: ignore[misc]
