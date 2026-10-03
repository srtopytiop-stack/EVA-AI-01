"""
EVA-AI-01 - Paper Trading Engine
================================

Phase 10: Paper Trading.

This module is the controlled integration layer between:
- Signal Engine
- Risk Engine
- Execution Simulator
- Portfolio Engine

The engine is Spot-only and fully simulated.

It does NOT:
- send orders to Binance
- use API keys
- use leverage or margin
- open short positions
- generate signals
- calculate risk itself
- fetch market data

The caller supplies the already-computed signal and risk decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.execution_simulator import (
    ExecutionError,
    ExecutionFill,
    ExecutionSide,
    ExecutionSimulator,
)
from core.portfolio_engine import (
    PortfolioEngine,
    PortfolioError,
    PortfolioSnapshot,
)
from core.risk_engine import (
    RiskDecision,
    RiskResult,
)
from core.signal_engine import SignalResult


class PaperTradingError(RuntimeError):
    """Raised for invalid paper-trading operations."""


class PaperAction(str, Enum):
    """High-level paper-trading action."""

    WAIT = "WAIT"
    HOLD = "HOLD"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class PaperTradingConfig:
    """
    Paper-trading policy.

    fee_rate is a model assumption. It must be replaced by the actual
    fee schedule used for the research environment when available.
    """

    fee_rate: float = 0.001

    min_signal_confidence: float = 0.55

    min_trade_notional: float = 0.0

    close_on_bearish_signal: bool = True

    def __post_init__(self) -> None:
        values = (
            ("fee_rate", self.fee_rate),
            (
                "min_signal_confidence",
                self.min_signal_confidence,
            ),
            (
                "min_trade_notional",
                self.min_trade_notional,
            ),
        )

        for name, value in values:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise PaperTradingError(
                    f"{name} must be a real number"
                )

            if not value == value:
                raise PaperTradingError(
                    f"{name} must be finite"
                )

            if value in (
                float("inf"),
                float("-inf"),
            ):
                raise PaperTradingError(
                    f"{name} must be finite"
                )

        if self.fee_rate < 0.0:
            raise PaperTradingError(
                "fee_rate must be >= 0"
            )

        if not 0.0 <= self.min_signal_confidence <= 1.0:
            raise PaperTradingError(
                "min_signal_confidence must be in [0, 1]"
            )

        if self.min_trade_notional < 0.0:
            raise PaperTradingError(
                "min_trade_notional must be >= 0"
            )

        if not isinstance(
            self.close_on_bearish_signal,
            bool,
        ):
            raise PaperTradingError(
                "close_on_bearish_signal must be boolean"
            )


@dataclass(frozen=True)
class PaperTradeEvent:
    """Immutable audit event emitted after a paper trade."""

    event_id: str
    timestamp: int
    symbol: str
    action: PaperAction

    side: ExecutionSide

    requested_quantity: float
    filled_quantity: float

    reference_price: float
    execution_price: float

    gross_notional: float

    fee: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float
    impact_bps: float
    effective_cost_bps: float

    realized_pnl: float

    signal_direction: int
    signal_score: float
    signal_confidence: float

    risk_decision: str
    risk_quantity: float

    equity_after: float

    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly event record."""

        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "action": self.action.value,
            "side": self.side.value,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "reference_price": self.reference_price,
            "execution_price": self.execution_price,
            "gross_notional": self.gross_notional,
            "fee": self.fee,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "impact_cost": self.impact_cost,
            "impact_bps": self.impact_bps,
            "effective_cost_bps": self.effective_cost_bps,
            "realized_pnl": self.realized_pnl,
            "signal_direction": self.signal_direction,
            "signal_score": self.signal_score,
            "signal_confidence": self.signal_confidence,
            "risk_decision": self.risk_decision,
            "risk_quantity": self.risk_quantity,
            "equity_after": self.equity_after,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PaperStepResult:
    """Immutable result of one paper-trading decision cycle."""

    action: PaperAction
    event: PaperTradeEvent | None
    snapshot: PortfolioSnapshot
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "event": (
                self.event.to_dict()
                if self.event is not None
                else None
            ),
            "snapshot": {
                "cash": self.snapshot.cash,
                "equity": self.snapshot.equity,
                "gross_exposure": self.snapshot.gross_exposure,
                "available_cash": self.snapshot.available_cash,
                "unrealized_pnl": self.snapshot.unrealized_pnl,
                "realized_pnl": self.snapshot.realized_pnl,
                "total_fees": self.snapshot.total_fees,
                "position_count": self.snapshot.position_count,
                "exposure_pct": self.snapshot.exposure_pct,
            },
            "reason": self.reason,
        }


class PaperTradingEngine:
    """
    Deterministic paper-trading orchestrator.

    Direction semantics:
        +1 -> open/hold long Spot
         0 -> wait/hold
        -1 -> close an existing long Spot

    A -1 direction never creates a short position.
    """

    def __init__(
        self,
        *,
        portfolio: PortfolioEngine | None = None,
        execution: ExecutionSimulator | None = None,
        config: PaperTradingConfig | None = None,
    ) -> None:
        self.portfolio = portfolio or PortfolioEngine()
        self.execution = execution or ExecutionSimulator()
        self.config = config or PaperTradingConfig()

        self._event_sequence = 0
        self._events: list[PaperTradeEvent] = []

    @property
    def events(self) -> tuple[PaperTradeEvent, ...]:
        """Return an immutable view of the event ledger."""

        return tuple(self._events)

    def snapshot(self) -> PortfolioSnapshot:
        """Return current portfolio valuation."""

        return self.portfolio.snapshot()

    def mark_price(
        self,
        symbol: str,
        price: float,
    ) -> PortfolioSnapshot:
        """Update mark-to-market price without executing a trade."""

        self.portfolio.mark_price(
            symbol,
            price,
        )

        return self.portfolio.snapshot()

    def step(
        self,
        *,
        timestamp: int,
        symbol: str,
        reference_price: float,
        signal: SignalResult,
        risk: RiskResult,
        volatility: float = 0.0,
        liquidity_notional: float | None = None,
    ) -> PaperStepResult:
        """
        Process one completed market observation.

        The signal and risk objects are treated as immutable evidence
        generated before this method is called.

        No future information is accessed.
        """

        self._validate_timestamp(timestamp)

        normalized_symbol = self._normalize_symbol(symbol)

        price = self._positive(
            reference_price,
            "reference_price",
        )

        self._validate_signal(
            signal,
            normalized_symbol,
        )

        self._validate_risk(
            risk,
        )

        position = self.portfolio.position(
            normalized_symbol
        )

        if signal.direction > 0:
            if risk.decision == RiskDecision.REJECTED:
                return self._result(
                    PaperAction.REJECTED,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "risk-engine-rejected-position",
                )

            if signal.confidence < self.config.min_signal_confidence:
                return self._result(
                    PaperAction.WAIT,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "signal-confidence-below-paper-trading-threshold",
                )

            if position is not None:
                self.portfolio.mark_price(
                    normalized_symbol,
                    price,
                )

                return self._result(
                    PaperAction.HOLD,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "existing-long-position-held",
                )

            quantity = risk.quantity

            if quantity <= 0.0:
                return self._result(
                    PaperAction.REJECTED,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "risk-engine-returned-nonpositive-quantity",
                )

            requested_notional = (
                quantity * price
            )

            if (
                requested_notional
                < self.config.min_trade_notional
            ):
                return self._result(
                    PaperAction.REJECTED,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "trade-notional-below-minimum",
                )

            return self._open_long(
                timestamp=timestamp,
                symbol=normalized_symbol,
                reference_price=price,
                signal=signal,
                risk=risk,
                volatility=volatility,
                liquidity_notional=liquidity_notional,
            )

        if signal.direction < 0:
            if position is None:
                return self._result(
                    PaperAction.WAIT,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "bearish-signal-without-open-position",
                )

            if signal.confidence < self.config.min_signal_confidence:
                return self._result(
                    PaperAction.HOLD,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "exit-signal-confidence-below-threshold",
                )

            if not self.config.close_on_bearish_signal:
                self.portfolio.mark_price(
                    normalized_symbol,
                    price,
                )

                return self._result(
                    PaperAction.HOLD,
                    timestamp,
                    normalized_symbol,
                    signal,
                    risk,
                    "automatic-bearish-exit-disabled",
                )

            return self._close_long(
                timestamp=timestamp,
                symbol=normalized_symbol,
                reference_price=price,
                signal=signal,
                risk=risk,
                quantity=position.quantity,
                volatility=volatility,
                liquidity_notional=liquidity_notional,
            )

        if position is not None:
            self.portfolio.mark_price(
                normalized_symbol,
                price,
            )

            return self._result(
                PaperAction.HOLD,
                timestamp,
                normalized_symbol,
                signal,
                risk,
                "neutral-signal-existing-position",
            )

        return self._result(
            PaperAction.WAIT,
            timestamp,
            normalized_symbol,
            signal,
            risk,
            "neutral-signal-no-position",
        )

    def close(
        self,
        *,
        timestamp: int,
        symbol: str,
        reference_price: float,
        signal: SignalResult,
        risk: RiskResult,
        volatility: float = 0.0,
        liquidity_notional: float | None = None,
    ) -> PaperStepResult:
        """Explicitly close an existing long position."""

        self._validate_timestamp(timestamp)

        normalized_symbol = self._normalize_symbol(symbol)
        self._validate_signal(signal, normalized_symbol)
        self._validate_risk(risk)

        position = self.portfolio.position(
            normalized_symbol
        )

        if position is None:
            return self._result(
                PaperAction.WAIT,
                timestamp,
                normalized_symbol,
                signal,
                risk,
                "close-request-without-open-position",
            )

        return self._close_long(
            timestamp=timestamp,
            symbol=normalized_symbol,
            reference_price=self._positive(
                reference_price,
                "reference_price",
            ),
            signal=signal,
            risk=risk,
            quantity=position.quantity,
            volatility=volatility,
            liquidity_notional=liquidity_notional,
        )

    def reset(self) -> None:
        """Reset portfolio and audit history."""

        self.portfolio.reset()
        self._event_sequence = 0
        self._events.clear()

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _open_long(
        self,
        *,
        timestamp: int,
        symbol: str,
        reference_price: float,
        signal: SignalResult,
        risk: RiskResult,
        volatility: float,
        liquidity_notional: float | None,
    ) -> PaperStepResult:
        try:
            fill = self.execution.market_order(
                ExecutionSide.BUY,
                risk.quantity,
                reference_price,
                volatility=volatility,
                liquidity_notional=liquidity_notional,
            )
        except ExecutionError as exc:
            return self._result(
                PaperAction.REJECTED,
                timestamp,
                symbol,
                signal,
                risk,
                f"execution-error: {exc}",
            )

        if not fill.fully_filled:
            return self._result(
                PaperAction.REJECTED,
                timestamp,
                symbol,
                signal,
                risk,
                fill.reason or "execution-rejected",
            )

        try:
            position = self.portfolio.open_position(
                symbol,
                quantity=fill.filled_quantity,
                entry_price=fill.execution_price,
                fee_rate=self.config.fee_rate,
            )
        except PortfolioError as exc:
            return self._result(
                PaperAction.REJECTED,
                timestamp,
                symbol,
                signal,
                risk,
                f"portfolio-rejected-entry: {exc}",
            )

        fee = (
            fill.gross_notional
            * self.config.fee_rate
        )

        snapshot = self.portfolio.snapshot()

        event = self._event(
            timestamp=timestamp,
            symbol=symbol,
            action=PaperAction.OPEN,
            fill=fill,
            fee=fee,
            realized_pnl=0.0,
            signal=signal,
            risk=risk,
            equity_after=snapshot.equity,
            reason=(
                "paper-long-opened:"
                + position.symbol
            ),
        )

        self._events.append(event)

        return PaperStepResult(
            action=PaperAction.OPEN,
            event=event,
            snapshot=snapshot,
            reason="paper-long-opened",
        )

    def _close_long(
        self,
        *,
        timestamp: int,
        symbol: str,
        reference_price: float,
        signal: SignalResult,
        risk: RiskResult,
        quantity: float,
        volatility: float,
        liquidity_notional: float | None,
    ) -> PaperStepResult:
        try:
            fill = self.execution.market_order(
                ExecutionSide.SELL,
                quantity,
                reference_price,
                volatility=volatility,
                liquidity_notional=liquidity_notional,
            )
        except ExecutionError as exc:
            return self._result(
                PaperAction.REJECTED,
                timestamp,
                symbol,
                signal,
                risk,
                f"execution-error: {exc}",
            )

        if not fill.fully_filled:
            return self._result(
                PaperAction.REJECTED,
                timestamp,
                symbol,
                signal,
                risk,
                fill.reason or "execution-rejected",
            )

        try:
            realized_pnl = self.portfolio.close_position(
                symbol,
                exit_price=fill.execution_price,
                quantity=fill.filled_quantity,
                fee_rate=self.config.fee_rate,
            )
        except PortfolioError as exc:
            return self._result(
                PaperAction.REJECTED,
                timestamp,
                symbol,
                signal,
                risk,
                f"portfolio-rejected-exit: {exc}",
            )

        fee = (
            fill.gross_notional
            * self.config.fee_rate
        )

        snapshot = self.portfolio.snapshot()

        event = self._event(
            timestamp=timestamp,
            symbol=symbol,
            action=PaperAction.CLOSE,
            fill=fill,
            fee=fee,
            realized_pnl=realized_pnl,
            signal=signal,
            risk=risk,
            equity_after=snapshot.equity,
            reason="paper-long-closed",
        )

        self._events.append(event)

        return PaperStepResult(
            action=PaperAction.CLOSE,
            event=event,
            snapshot=snapshot,
            reason="paper-long-closed",
        )

    # ------------------------------------------------------------------
    # Validation and result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_timestamp(timestamp: int) -> None:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise PaperTradingError(
                "timestamp must be an integer"
            )

        if timestamp <= 0:
            raise PaperTradingError(
                "timestamp must be positive"
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise PaperTradingError(
                "symbol must be a non-empty string"
            )

        return symbol.strip().upper()

    @staticmethod
    def _positive(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise PaperTradingError(
                f"{name} must be a real number"
            )

        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise PaperTradingError(
                f"{name} must be numeric"
            ) from exc

        if not normalized == normalized:
            raise PaperTradingError(
                f"{name} must be finite"
            )

        if normalized in (
            float("inf"),
            float("-inf"),
        ):
            raise PaperTradingError(
                f"{name} must be finite"
            )

        if normalized <= 0.0:
            raise PaperTradingError(
                f"{name} must be > 0"
            )

        return normalized

    @classmethod
    def _validate_signal(
        cls,
        signal: SignalResult,
        symbol: str,
    ) -> None:
        if not isinstance(signal, SignalResult):
            raise PaperTradingError(
                "signal must be a SignalResult"
            )

        if not isinstance(signal.symbol, str):
            raise PaperTradingError(
                "signal symbol must be a string"
            )

        if signal.symbol.strip().upper() != symbol:
            raise PaperTradingError(
                "signal symbol does not match trading symbol"
            )

        if signal.direction not in (-1, 0, 1):
            raise PaperTradingError(
                "signal direction must be -1, 0, or 1"
            )

        if not 0.0 <= signal.confidence <= 1.0:
            raise PaperTradingError(
                "signal confidence must be in [0, 1]"
            )

    @staticmethod
    def _validate_risk(
        risk: RiskResult,
    ) -> None:
        if not isinstance(risk, RiskResult):
            raise PaperTradingError(
                "risk must be a RiskResult"
            )

        if risk.quantity < 0.0:
            raise PaperTradingError(
                "risk quantity cannot be negative"
            )

        if risk.notional < 0.0:
            raise PaperTradingError(
                "risk notional cannot be negative"
            )

    def _event(
        self,
        *,
        timestamp: int,
        symbol: str,
        action: PaperAction,
        fill: ExecutionFill,
        fee: float,
        realized_pnl: float,
        signal: SignalResult,
        risk: RiskResult,
        equity_after: float,
        reason: str,
    ) -> PaperTradeEvent:
        self._event_sequence += 1

        event_id = (
            f"PAPER-{self._event_sequence:08d}"
        )

        return PaperTradeEvent(
            event_id=event_id,
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            side=fill.side,
            requested_quantity=fill.requested_quantity,
            filled_quantity=fill.filled_quantity,
            reference_price=fill.reference_price,
            execution_price=fill.execution_price,
            gross_notional=fill.gross_notional,
            fee=fee,
            spread_cost=fill.spread_cost,
            slippage_cost=fill.slippage_cost,
            impact_cost=fill.impact_cost,
            impact_bps=fill.impact_bps,
            effective_cost_bps=fill.effective_cost_bps,
            realized_pnl=realized_pnl,
            signal_direction=signal.direction,
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            risk_decision=risk.decision.value,
            risk_quantity=risk.quantity,
            equity_after=equity_after,
            reason=reason,
        )

    def _result(
        self,
        action: PaperAction,
        timestamp: int,
        symbol: str,
        signal: SignalResult,
        risk: RiskResult,
        reason: str,
    ) -> PaperStepResult:
        del timestamp, symbol, signal, risk

        return PaperStepResult(
            action=action,
            event=None,
            snapshot=self.portfolio.snapshot(),
            reason=reason,
        )


if __name__ == "__main__":
    print("PAPER_TRADING_ENGINE_READY")
