"""Phase 10 integration tests for the Paper Trading Engine."""

from __future__ import annotations

import pytest

from core.execution_simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)
from core.paper_trading_engine import (
    PaperAction,
    PaperTradingConfig,
    PaperTradingEngine,
    PaperTradingError,
)
from core.portfolio_engine import (
    PortfolioConfig,
    PortfolioEngine,
    PortfolioError,
)
from core.risk_engine import (
    RiskDecision,
    RiskResult,
)
from core.signal_engine import SignalResult


def make_signal(
    *,
    symbol: str = "BTCUSDT",
    direction: int = 1,
    confidence: float = 0.90,
    score: float = 0.80,
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        timestamp=1_800_000_000,
        direction=direction,
        score=score,
        confidence=confidence,
        classical_z=1.0,
        robust_z=1.0,
        volatility=0.01,
        trend_score=0.80,
        mean_reversion_score=0.10,
        volume_score=0.20,
        regime_factor=1.0,
        volatility_factor=0.90,
        reasons=("test-signal",),
    )


def make_risk(
    *,
    quantity: float = 1.0,
    entry_price: float = 100.0,
    decision: RiskDecision = RiskDecision.APPROVED,
) -> RiskResult:
    notional = quantity * entry_price

    return RiskResult(
        decision=decision,
        confidence=0.90,
        equity=1_000.0,
        entry_price=entry_price,
        stop_distance=5.0,
        stop_distance_pct=0.05,
        risk_budget=10.0,
        raw_quantity=quantity,
        volatility_ratio=1.0,
        volatility_multiplier=1.0,
        adjusted_quantity=quantity,
        adjusted_notional=notional,
        maximum_notional=250.0,
        risk_at_stop=quantity * 5.0,
        actual_risk_pct=0.01,
        reasons=("test-risk",),
    )


def make_engine() -> PaperTradingEngine:
    portfolio = PortfolioEngine(
        PortfolioConfig(
            initial_cash=1_000.0,
            max_total_exposure_pct=0.80,
            max_position_exposure_pct=0.25,
            min_cash_reserve_pct=0.10,
            max_positions=10,
        )
    )

    execution = ExecutionSimulator(
        ExecutionConfig(
            half_spread_bps=1.0,
            base_slippage_bps=0.5,
            volatility_slippage_factor=0.05,
            impact_coefficient_bps=2.0,
        )
    )

    config = PaperTradingConfig(
        fee_rate=0.001,
        min_signal_confidence=0.55,
        min_trade_notional=0.0,
        close_on_bearish_signal=True,
    )

    return PaperTradingEngine(
        portfolio=portfolio,
        execution=execution,
        config=config,
    )


def test_long_signal_opens_one_spot_position() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(quantity=1.0, entry_price=100.0),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert result.action == PaperAction.OPEN
    assert result.event is not None
    assert result.event.action == PaperAction.OPEN
    assert result.event.side.value == "BUY"

    position = engine.portfolio.position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(1.0)
    assert position.entry_price > 100.0

    assert engine.snapshot().position_count == 1
    assert len(engine.events) == 1


def test_second_bullish_signal_holds_existing_position() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    result = engine.step(
        timestamp=1_800_000_002,
        symbol="BTCUSDT",
        reference_price=105.0,
        signal=make_signal(direction=1),
        risk=make_risk(entry_price=105.0),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert result.action == PaperAction.HOLD
    assert result.event is None
    assert len(engine.events) == 1

    position = engine.portfolio.position("BTCUSDT")

    assert position is not None
    assert position.current_price == pytest.approx(105.0)


def test_neutral_signal_marks_existing_position() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    result = engine.step(
        timestamp=1_800_000_002,
        symbol="BTCUSDT",
        reference_price=110.0,
        signal=make_signal(direction=0, score=0.0),
        risk=make_risk(entry_price=110.0),
    )

    assert result.action == PaperAction.HOLD

    position = engine.portfolio.position("BTCUSDT")

    assert position is not None
    assert position.current_price == pytest.approx(110.0)


def test_neutral_signal_without_position_waits() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(
            direction=0,
            score=0.0,
        ),
        risk=make_risk(),
    )

    assert result.action == PaperAction.WAIT
    assert engine.snapshot().position_count == 0


def test_bearish_signal_closes_long_spot_position() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    result = engine.step(
        timestamp=1_800_000_002,
        symbol="BTCUSDT",
        reference_price=110.0,
        signal=make_signal(
            direction=-1,
            confidence=0.90,
            score=-0.80,
        ),
        risk=make_risk(
            entry_price=110.0,
        ),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert result.action == PaperAction.CLOSE
    assert result.event is not None
    assert result.event.action == PaperAction.CLOSE
    assert result.event.side.value == "SELL"

    assert engine.portfolio.position("BTCUSDT") is None
    assert engine.snapshot().position_count == 0
    assert len(engine.events) == 2


def test_bearish_signal_without_position_waits() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(
            direction=-1,
            score=-0.80,
        ),
        risk=make_risk(),
    )

    assert result.action == PaperAction.WAIT
    assert result.reason == (
        "bearish-signal-without-open-position"
    )


def test_rejected_risk_blocks_new_entry() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(
            decision=RiskDecision.REJECTED,
        ),
    )

    assert result.action == PaperAction.REJECTED
    assert result.event is None
    assert engine.snapshot().position_count == 0


def test_rejected_risk_does_not_block_bearish_exit() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    result = engine.step(
        timestamp=1_800_000_002,
        symbol="BTCUSDT",
        reference_price=105.0,
        signal=make_signal(
            direction=-1,
            score=-0.80,
        ),
        risk=make_risk(
            decision=RiskDecision.REJECTED,
            entry_price=105.0,
        ),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert result.action == PaperAction.CLOSE
    assert engine.snapshot().position_count == 0


def test_low_signal_confidence_blocks_entry() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(
            direction=1,
            confidence=0.40,
        ),
        risk=make_risk(),
    )

    assert result.action == PaperAction.WAIT
    assert result.reason == (
        "signal-confidence-below-paper-trading-threshold"
    )


def test_low_exit_confidence_holds_position() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    result = engine.step(
        timestamp=1_800_000_002,
        symbol="BTCUSDT",
        reference_price=105.0,
        signal=make_signal(
            direction=-1,
            confidence=0.40,
            score=-0.80,
        ),
        risk=make_risk(entry_price=105.0),
    )

    assert result.action == PaperAction.HOLD
    assert engine.snapshot().position_count == 1


def test_minimum_trade_notional_is_enforced() -> None:
    engine = PaperTradingEngine(
        portfolio=PortfolioEngine(),
        config=PaperTradingConfig(
            min_trade_notional=500.0,
        ),
    )

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(quantity=1.0),
    )

    assert result.action == PaperAction.REJECTED
    assert result.reason == "trade-notional-below-minimum"


def test_execution_liquidity_gate_can_reject_trade() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(quantity=20.0),
        liquidity_notional=1_000.0,
    )

    assert result.action == PaperAction.REJECTED
    assert engine.snapshot().position_count == 0


def test_portfolio_constraints_can_reject_execution() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(
            initial_cash=1_000.0,
            max_total_exposure_pct=0.20,
            max_position_exposure_pct=0.20,
            min_cash_reserve_pct=0.10,
        )
    )

    engine = PaperTradingEngine(
        portfolio=portfolio,
    )

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(
            quantity=3.0,
            entry_price=100.0,
        ),
    )

    assert result.action == PaperAction.REJECTED
    assert result.reason.startswith(
        "portfolio-rejected-entry:"
    )


def test_mark_price_does_not_create_trade() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        liquidity_notional=10_000.0,
    )

    before_events = len(engine.events)

    snapshot = engine.mark_price(
        "BTCUSDT",
        110.0,
    )

    position = engine.portfolio.position("BTCUSDT")

    assert position is not None
    assert snapshot.position_count == 1
    assert snapshot.gross_exposure == pytest.approx(
        position.quantity * 110.0
    )
    assert len(engine.events) == before_events


def test_trade_event_ids_are_monotonic() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        liquidity_notional=10_000.0,
    )

    engine.step(
        timestamp=1_800_000_002,
        symbol="BTCUSDT",
        reference_price=105.0,
        signal=make_signal(direction=-1, score=-0.80),
        risk=make_risk(entry_price=105.0),
        liquidity_notional=10_000.0,
    )

    assert engine.events[0].event_id == "PAPER-00000001"
    assert engine.events[1].event_id == "PAPER-00000002"


def test_event_and_step_are_json_friendly() -> None:
    engine = make_engine()

    result = engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(direction=1),
        risk=make_risk(),
        liquidity_notional=10_000.0,
    )

    payload = result.to_dict()

    assert payload["action"] == "OPEN"
    assert payload["event"]["side"] == "BUY"
    assert "impact_cost" in payload["event"]


def test_symbol_mismatch_is_rejected() -> None:
    engine = make_engine()

    with pytest.raises(PaperTradingError):
        engine.step(
            timestamp=1_800_000_001,
            symbol="ETHUSDT",
            reference_price=100.0,
            signal=make_signal(symbol="BTCUSDT"),
            risk=make_risk(),
        )


def test_invalid_timestamp_is_rejected() -> None:
    engine = make_engine()

    with pytest.raises(PaperTradingError):
        engine.step(
            timestamp=0,
            symbol="BTCUSDT",
            reference_price=100.0,
            signal=make_signal(),
            risk=make_risk(),
        )


def test_reset_clears_portfolio_and_event_history() -> None:
    engine = make_engine()

    engine.step(
        timestamp=1_800_000_001,
        symbol="BTCUSDT",
        reference_price=100.0,
        signal=make_signal(),
        risk=make_risk(),
        liquidity_notional=10_000.0,
    )

    assert engine.snapshot().position_count == 1
    assert len(engine.events) == 1

    engine.reset()

    assert engine.snapshot().cash == pytest.approx(1_000.0)
    assert engine.snapshot().position_count == 0
    assert len(engine.events) == 0
