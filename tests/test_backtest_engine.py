"""
Phase 11 - Historical Backtest Engine Tests
============================================

Test scope
----------
These tests validate the event-driven backtest orchestration layer.

Important testing principle
---------------------------
The production BacktestEngine combines several independent layers:

    Signal
        -> Regime
        -> Volatility
        -> Risk
        -> Paper Trading
        -> Execution
        -> Portfolio

Tests that specifically validate execution timing, liquidity causality,
or look-ahead protection must not depend on whether the real signal
strategy happens to generate a trade from a particular synthetic dataset.

Therefore:

1. Production/integration behavior is tested with the real signal engine.
2. Execution-timing/liquidity tests inject a deterministic valid decision.
3. The injected decision is generated only from the current candle/equity.
4. No future candle is used to construct the decision.

This keeps the tests scientifically isolated and prevents a strategy
threshold change from breaking unrelated execution-layer tests.

Scope:
- Spot long only
- No leverage
- No margin
- No shorting
- Next-bar-open execution
- Decision-bar liquidity proxy
- No future-data leakage
- Final-position liquidation
- Candle chronology validation
- Warmup/history configuration validation
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestError,
)
from core.execution_simulator import ExecutionConfig
from core.paper_trading_engine import PaperTradingConfig
from core.portfolio_engine import PortfolioConfig
from core.risk_engine import RiskConfig
from core.signal_engine import SignalResult
from data.market_data import Candle


# ---------------------------------------------------------------------------
# Deterministic synthetic market
# ---------------------------------------------------------------------------

BASE_TS = 1_800_000_000_000
BAR_MS = 60_000


def make_candles(
    count: int = 180,
    *,
    drift: float = 0.002,
    quote_volume: float = 10_000_000.0,
) -> list[Candle]:
    """
    Build a deterministic, strictly chronological OHLCV series.

    The default series is a gentle positive drift. It is deliberately simple:
    these candles are test data, not an attempt to model a real market.
    """

    if count < 2:
        raise ValueError("count must be >= 2")

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


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

def config() -> BacktestConfig:
    """
    Deterministic configuration used by Phase 11 tests.

    The configuration deliberately keeps transaction costs simple so that
    execution-timing assertions are not obscured by unnecessary model noise.
    """

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


# ---------------------------------------------------------------------------
# Deterministic decision injection
# ---------------------------------------------------------------------------

def install_deterministic_entry_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replace only BacktestEngine._decision_at_close with a deterministic,
    valid decision for execution-layer tests.

    Why this is necessary
    ----------------------
    Tests such as "next-bar execution" and "future liquidity must not be
    used" are not signal-engine tests.

    If they rely on the real signal engine, a harmless change in:
        - signal threshold
        - regime classification
        - volatility filter
        - trend calculation
        - confidence calculation

    could cause the synthetic market to produce no entry at all.

    That would make the test fail before it ever tests execution timing.

    The injected decision is intentionally simple:

        direction = +1
        confidence = 1.0
        valid risk result
        deterministic volatility

    The actual BacktestEngine, PaperTradingEngine, ExecutionSimulator,
    PortfolioEngine, event creation, liquidity check, and final liquidation
    remain real production code.

    pytest's monkeypatch fixture safely restores the original method after
    the test. See pytest's monkeypatch documentation.
    """

    def deterministic_decision(
        self: BacktestEngine,
        *,
        candle: Candle,
        closes: list[float] | tuple[float, ...],
        volumes: list[float] | tuple[float, ...],
        equity: float,
    ):
        """
        Produce a valid long decision using only the current decision context.

        `closes` and `volumes` are deliberately accepted because they are part
        of the production method contract, but they are not used here.
        """

        del closes
        del volumes

        entry_price = float(candle.close)

        # A 5% theoretical stop is comfortably inside the configured
        # maximum stop-distance constraint.
        stop_distance = entry_price * 0.05

        risk = self.risk_engine.evaluate(
            equity=float(equity),
            entry_price=entry_price,
            stop_distance=stop_distance,
            volatility_ratio=1.0,
            config=self.config.risk_config,
        )

        signal = SignalResult(
            symbol=self._symbol,
            timestamp=int(candle.close_time),
            direction=1,
            score=1.0,
            confidence=1.0,
            classical_z=0.0,
            robust_z=0.0,
            volatility=0.001,
            trend_score=1.0,
            mean_reversion_score=0.0,
            volume_score=1.0,
            regime_factor=1.0,
            volatility_factor=1.0,
            reasons=("deterministic-test-entry",),
        )

        # BacktestEngine only requires the volatility object to expose
        # ewma_volatility when constructing the pending decision.
        volatility = SimpleNamespace(
            ewma_volatility=0.001,
        )

        return signal, risk, volatility

    monkeypatch.setattr(
        BacktestEngine,
        "_decision_at_close",
        deterministic_decision,
    )


# ---------------------------------------------------------------------------
# Integration-level result test
# ---------------------------------------------------------------------------

def test_engine_produces_results_and_forces_final_liquidation() -> None:
    """
    Verify that the real production signal/risk/regime stack can execute
    through the complete BacktestEngine without requiring a trade.

    A historical strategy is allowed to legitimately produce zero trades.
    Therefore this test checks result integrity rather than assuming an entry.
    """

    result = BacktestEngine(config()).run(
        make_candles(),
        symbol="BTCUSDT",
    )

    assert result.symbol == "BTCUSDT"
    assert result.bars_processed == 180
    assert result.warmup_bars == 70

    assert result.equity_curve
    assert len(result.equity_curve) == 180

    assert result.metrics.bars_processed == 180

    # The BacktestEngine must never leave an open position after the final
    # historical bar.
    assert result.final_position_open is False


# ---------------------------------------------------------------------------
# Next-bar execution
# ---------------------------------------------------------------------------

def test_execution_happens_on_next_bar_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A decision formed at bar t close must execute at bar t+1 open.

    The first eligible decision occurs at index:

        warmup_bars - 1 = 69

    and therefore its earliest execution is:

        index 70 open.
    """

    install_deterministic_entry_decision(monkeypatch)

    candles = make_candles()

    result = BacktestEngine(config()).run(
        candles,
        symbol="BTCUSDT",
    )

    opens = [
        event
        for event in result.events
        if event.action.value == "OPEN"
    ]

    assert opens, (
        "deterministic test decision should create at least one entry"
    )

    first_open = opens[0]

    expected_execution_timestamp = candles[70].open_time

    assert first_open.timestamp == expected_execution_timestamp

    # Explicitly prove that execution occurred at the NEXT candle's open,
    # not at the decision candle's close.
    assert first_open.timestamp > candles[69].close_time

    assert first_open.timestamp == candles[70].open_time


# ---------------------------------------------------------------------------
# Liquidity causality
# ---------------------------------------------------------------------------

def test_next_bar_full_volume_is_not_used_as_entry_liquidity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The execution at bar t+1 open must use liquidity known at decision time.

    For the first entry:

        decision bar  = 69
        execution bar = 70

    The BacktestEngine should use candle[69].quote_volume as the liquidity
    proxy, not candle[70].quote_volume.

    We therefore make candle[70] extremely illiquid in one dataset.

    If the implementation incorrectly reads future execution-bar volume,
    the two runs can produce different execution behavior.

    If the implementation is causal, both first entries remain identical.
    """

    install_deterministic_entry_decision(monkeypatch)

    original = make_candles()

    altered = make_candles()

    # This is the execution candle, not the decision candle.
    #
    # It must NOT influence the already-created pending order.
    altered[70] = replace(
        altered[70],
        quote_volume=1.0,
    )

    result_original = BacktestEngine(config()).run(
        original,
        symbol="BTCUSDT",
    )

    result_altered = BacktestEngine(config()).run(
        altered,
        symbol="BTCUSDT",
    )

    original_opens = [
        event
        for event in result_original.events
        if event.action.value == "OPEN"
    ]

    altered_opens = [
        event
        for event in result_altered.events
        if event.action.value == "OPEN"
    ]

    assert original_opens
    assert altered_opens

    first_original = original_opens[0]
    first_altered = altered_opens[0]

    # The entry must still happen at exactly the same next-bar open.
    assert first_original.timestamp == candles_timestamp(
        original,
        70,
    )

    assert first_altered.timestamp == candles_timestamp(
        altered,
        70,
    )

    assert first_original.timestamp == first_altered.timestamp

    # Execution price must be identical because the decision-bar liquidity
    # and all execution inputs available at the time of decision are equal.
    assert first_original.execution_price == pytest.approx(
        first_altered.execution_price,
    )

    # The altered future volume must not cause rejection of the already
    # generated entry.
    assert first_original.filled_quantity > 0.0
    assert first_altered.filled_quantity > 0.0


def candles_timestamp(
    candles: list[Candle],
    index: int,
) -> int:
    """
    Small helper used only to make timestamp assertions explicit.
    """

    return int(candles[index].open_time)


# ---------------------------------------------------------------------------
# Future-information / look-ahead test
# ---------------------------------------------------------------------------

def test_future_data_cannot_change_prior_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Changing candles from index 130 onward must not change events that occur
    before index 130.

    The deterministic decision is important here because the purpose of this
    test is to validate BacktestEngine causality, not signal sensitivity.
    """

    install_deterministic_entry_decision(monkeypatch)

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

    baseline = BacktestEngine(config()).run(
        original,
        symbol="BTCUSDT",
    )

    changed = BacktestEngine(config()).run(
        altered,
        symbol="BTCUSDT",
    )

    cutoff = altered[change_index].open_time

    baseline_prior = [
        event.to_dict()
        for event in baseline.events
        if event.timestamp < cutoff
    ]

    changed_prior = [
        event.to_dict()
        for event in changed.events
        if event.timestamp < cutoff
    ]

    assert baseline_prior == changed_prior


# ---------------------------------------------------------------------------
# Final liquidation behavior
# ---------------------------------------------------------------------------

def test_final_open_position_is_liquidated_on_last_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A deterministic entry must not remain open after the final historical bar.

    This verifies the explicit final-bar liquidation path in BacktestEngine.
    """

    install_deterministic_entry_decision(monkeypatch)

    candles = make_candles()

    result = BacktestEngine(config()).run(
        candles,
        symbol="BTCUSDT",
    )

    opens = [
        event
        for event in result.events
        if event.action.value == "OPEN"
    ]

    closes = [
        event
        for event in result.events
        if event.action.value == "CLOSE"
    ]

    assert opens, "deterministic entry should produce an OPEN event"
    assert closes, "final-bar liquidation should produce a CLOSE event"

    assert result.final_position_open is False

    last_close = closes[-1]

    assert last_close.timestamp == candles[-1].close_time


# ---------------------------------------------------------------------------
# Candle validation
# ---------------------------------------------------------------------------

def test_irregular_candles_are_rejected() -> None:
    """
    Missing/irregular intervals must be rejected when regular intervals are
    required by BacktestConfig.
    """

    candles = make_candles()

    candles[100] = replace(
        candles[100],
        open_time=candles[100].open_time + 1_000,
        close_time=candles[100].close_time + 1_000,
    )

    with pytest.raises(BacktestError):
        BacktestEngine(config()).run(
            candles,
            symbol="BTCUSDT",
        )


# ---------------------------------------------------------------------------
# Configuration invariants
# ---------------------------------------------------------------------------

def test_history_window_is_never_smaller_than_warmup() -> None:
    """
    history_window_bars must be at least as large as warmup_bars.
    """

    with pytest.raises(BacktestError):
        BacktestConfig(
            warmup_bars=100,
            history_window_bars=90,
        )


def test_insufficient_candles_are_rejected() -> None:
    """
    BacktestEngine requires at least warmup_bars + 1 candles.
    """

    test_config = config()

    candles = make_candles(
        count=test_config.warmup_bars,
    )

    with pytest.raises(BacktestError):
        BacktestEngine(test_config).run(
            candles,
            symbol="BTCUSDT",
        )


# ---------------------------------------------------------------------------
# Symbol validation
# ---------------------------------------------------------------------------

def test_symbol_is_normalized_to_uppercase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Symbol normalization is performed by BacktestEngine.
    """

    install_deterministic_entry_decision(monkeypatch)

    result = BacktestEngine(config()).run(
        make_candles(),
        symbol="btcusdt",
    )

    assert result.symbol == "BTCUSDT"


def test_invalid_symbol_is_rejected() -> None:
    """
    Symbols containing invalid characters must be rejected.
    """

    with pytest.raises(BacktestError):
        BacktestEngine(config()).run(
            make_candles(),
            symbol="BTC/USDT",
        )
