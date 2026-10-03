"""Phase 14-A integration tests for EVA-AI-01.

The integration boundary is tested in two parts:
- MarketDataResult -> FeatureEngine is exercised for real.
- BacktestResult -> Statistical Validation -> Bootstrap Validation is exercised
  with a deterministic BacktestResult fixture.

The BacktestEngine itself has its own dedicated test suite. Mocking its output
here is intentional: this test is specifically designed to prove that the
system-integration layer can consume a valid, non-degenerate backtest result
and reach both statistical-validation layers without making production code
more permissive around undefined Sharpe inference.
"""

from __future__ import annotations

from math import isfinite

import pytest

from core.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from core.backtest_metrics import EquityPoint, calculate_metrics
from core.system_integration import (
    IntegrationConfig,
    IntegrationError,
    run_integrated_research,
)
from data.market_data import Candle, MarketDataResult


CANDLE_INTERVAL_MS = 60_000
START_TIMESTAMP_MS = 1_700_000_000_000
BACKTEST_OBSERVATIONS = 180
INITIAL_EQUITY = 1_000.0


def _market_data(count: int = BACKTEST_OBSERVATIONS) -> MarketDataResult:
    """Create deterministic OHLCV data for the FeatureEngine boundary."""
    candles: list[Candle] = []
    price = 30_000.0

    for i in range(count):
        # The market fixture is deliberately non-flat, but the statistical
        # test below does not depend on the strategy discovering trades from it.
        cycle = i % 10
        change = 0.0015 if cycle < 5 else -0.0010

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
    """Fast deterministic integration configuration."""
    backtest = BacktestConfig(
        warmup_bars=120,
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


def _synthetic_backtest_result(
    *,
    symbol: str,
    observations: int = BACKTEST_OBSERVATIONS,
) -> BacktestResult:
    """Build a valid BacktestResult whose returns have guaranteed variance.

    The five-period return motif is intentionally non-constant. Every possible
    moving block of length five therefore contains multiple return values, so
    the bootstrap layer cannot receive a constant block and hit its
    ``return variance is zero`` guard.
    """
    return_pattern = (
        0.0040,
        -0.0020,
        0.0030,
        -0.0010,
        0.0020,
    )

    equity_points: list[EquityPoint] = []
    equity = INITIAL_EQUITY

    for i in range(observations):
        if i > 0:
            period_return = return_pattern[(i - 1) % len(return_pattern)]
            equity *= 1.0 + period_return

        timestamp = START_TIMESTAMP_MS + i * CANDLE_INTERVAL_MS
        equity_points.append(
            EquityPoint(
                timestamp=timestamp,
                equity=equity,
                cash=equity,
                gross_exposure=0.0,
                exposure_pct=0.0,
                drawdown_pct=0.0,
            )
        )

    equity_curve = tuple(equity_points)
    metrics = calculate_metrics(
        equity_curve=equity_curve,
        trades=(),
        initial_equity=INITIAL_EQUITY,
        benchmark_return=0.0,
        rejected_actions=0,
        risk_free_rate_annual=0.0,
    )

    return BacktestResult(
        symbol=symbol,
        bars_processed=observations,
        warmup_bars=120,
        rejected_actions=0,
        equity_curve=equity_curve,
        trades=(),
        events=(),
        metrics=metrics,
        final_position_open=False,
    )


def _assert_nonzero_variance(values: list[float]) -> None:
    mean_value = sum(values) / len(values)
    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / len(values)
    assert isfinite(variance)
    assert variance > 1e-16


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
    calls: dict[str, object] = {}

    def fake_backtest_run(
        self: BacktestEngine,
        candles: object,
        *,
        symbol: str,
    ) -> BacktestResult:
        """Inject a valid BacktestResult at the integration boundary."""
        del self
        calls["symbol"] = symbol
        calls["candle_count"] = len(candles)  # type: ignore[arg-type]
        return _synthetic_backtest_result(symbol=symbol)

    monkeypatch.setattr(
        BacktestEngine,
        "run",
        fake_backtest_run,
    )

    result = run_integrated_research(market, config=_config())

    # Prove the integration layer invoked the BacktestEngine contract.
    assert calls == {
        "symbol": "BTCUSDT",
        "candle_count": BACKTEST_OBSERVATIONS,
    }

    assert result.symbol == "BTCUSDT"
    assert result.interval == "1m"
    assert result.candle_count == BACKTEST_OBSERVATIONS
    assert result.feature_rows == BACKTEST_OBSERVATIONS
    assert "return" in result.feature_columns
    assert "volatility" in result.feature_columns
    assert "momentum" in result.feature_columns

    assert result.backtest.symbol == "BTCUSDT"
    assert len(result.backtest.equity_curve) == BACKTEST_OBSERVATIONS
    assert result.backtest.metrics.final_equity > 0.0

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

    # This is the exact invariant required by statistical Sharpe inference.
    _assert_nonzero_variance(returns)

    # The moving-block bootstrap uses block_length=5. The fixture repeats a
    # five-return motif containing multiple values, so every possible moving
    # block has non-zero variance before blocks are sampled with replacement.
    block_length = _config().bootstrap_block_length
    for start in range(0, len(returns) - block_length + 1):
        block = returns[start : start + block_length]
        _assert_nonzero_variance(block)

    assert result.statistical_validation.observations == BACKTEST_OBSERVATIONS - 1
    assert result.bootstrap_validation.observations == BACKTEST_OBSERVATIONS - 1
    assert result.bootstrap_validation.bootstrap_replications == 200
    assert result.bootstrap_validation.block_length == 5

    assert isfinite(result.statistical_validation.periodic_sharpe)
    assert isfinite(result.statistical_validation.probabilistic_sharpe)
    assert isfinite(result.statistical_validation.deflated_sharpe)
    assert isfinite(result.bootstrap_validation.observed_periodic_sharpe)

    payload = result.to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["final_equity"] == result.final_equity

    with pytest.raises(AttributeError):
        result.candle_count = 999  # type: ignore[misc]
