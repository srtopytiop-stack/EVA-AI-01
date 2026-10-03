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
START_TIMESTAMP_MS = 1_700_000_000_000


def _market_data(count: int = 180) -> MarketDataResult:
    """Create deterministic OHLCV data whose post-warmup returns vary every bar.

    The integration test deliberately keeps a Spot position open through the
    post-warmup section. Because the close price alternates between a positive
    and a negative return on every completed bar, the resulting marked-to-
    market equity curve also varies on every bar. Therefore every moving block
    of length 5 contains more than one return value and cannot have zero
    variance merely because a sampled block consists of flat observations.
    """
    candles: list[Candle] = []
    price = 30_000.0

    for i in range(count):
        if i < WARMUP_BARS:
            # Non-flat historical data for the real volatility/signal engines.
            change = 0.0015 if (i % 6) < 3 else -0.0010
        else:
            # Strictly alternating returns after warm-up. Every 5-bar moving
            # block contains both signs (+1% and -0.8%).
            phase = (i - WARMUP_BARS) % 2
            change = 0.01 if phase == 0 else -0.008

        open_price = price
        close_price = open_price * (1.0 + change)
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        open_time = START_TIMESTAMP_MS + i * CANDLE_INTERVAL_MS

        candles.append(
            Candle(
                open_time=open_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=1_000_000.0 + (i % 17) * 10_000.0,
                close_time=open_time + CANDLE_INTERVAL_MS - 1,
                quote_volume=close_price * 1_000_000.0,
                number_of_trades=1_000 + i,
            )
        )
        price = close_price

    return MarketDataResult(
        symbol="BTCUSDT",
        interval="1m",
        candles=tuple(candles),
    )


def _config() -> IntegrationConfig:
    """Fast, deterministic integration-test configuration."""
    backtest = BacktestConfig(
        warmup_bars=WARMUP_BARS,
        history_window_bars=240,
        require_regular_intervals=True,
        expected_interval_ms=CANDLE_INTERVAL_MS,
    )

    return IntegrationConfig(
        backtest_config=backtest,
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
        """Use the real volatility/risk calculations and control only direction.

        The first post-warmup decision opens a long Spot position. Neutral
        decisions then keep that position open, allowing mark-to-market equity
        to move with every bar. The real BacktestEngine closes the position on
        the final bar, so this remains a genuine end-to-end backtest path.
        """
        signal, risk, volatility = original(
            self,
            candle=candle,
            closes=closes,
            volumes=volumes,
            equity=equity,
        )

        candle_index = (
            int(candle.open_time) - START_TIMESTAMP_MS
        ) // CANDLE_INTERVAL_MS

        if candle_index == WARMUP_BARS:
            direction = 1
            score = 0.90
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
    assert len(result.backtest.trades) == 1

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

    # Direct invariant required by Statistical Validation.
    mean_return = sum(returns) / len(returns)
    variance = sum(
        (value - mean_return) ** 2
        for value in returns
    ) / len(returns)
    assert variance > 1e-16

    # The bootstrap uses moving blocks of length 5. The synthetic post-warmup
    # path alternates sign on every bar, so each possible 5-return block has
    # non-zero variance while the position is open.
    post_warmup_returns = returns[WARMUP_BARS:]
    assert len(post_warmup_returns) >= 20
    for start in range(0, len(post_warmup_returns) - 4):
        block = post_warmup_returns[start : start + 5]
        block_mean = sum(block) / len(block)
        block_variance = sum(
            (value - block_mean) ** 2
            for value in block
        ) / len(block)
        assert block_variance > 1e-16

    assert result.statistical_validation.observations == 179
    assert result.bootstrap_validation.observations == 179
    assert result.bootstrap_validation.bootstrap_replications == 200
    assert result.bootstrap_validation.block_length == 5

    payload = result.to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["final_equity"] == result.final_equity

    with pytest.raises(AttributeError):
        result.candle_count = 999  # type: ignore[misc]
