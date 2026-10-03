"""EVA-AI-01 - Risk Engine

Phase 8: Risk Engine.

Responsibilities
----------------
Convert a hypothetical Spot position into an auditable risk decision.

The engine calculates:
- risk budget from account equity and configured risk percentage
- raw position quantity from entry/stop distance
- volatility-adjusted quantity
- maximum Spot notional exposure
- final quantity and notional after all limits
- risk decision and reasons

Important boundaries
--------------------
- No exchange/API access.
- No order placement.
- No leverage or margin.
- No BUY/SELL signal generation.
- No portfolio mutation.
- Position size is a theoretical risk calculation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


EPS = 1e-12


class RiskEngineError(ValueError):
    """Raised when risk-engine inputs or configuration are invalid."""


class RiskDecision(str, Enum):
    """Risk-policy outcome for a hypothetical Spot position."""

    APPROVED = "APPROVED"
    REDUCED = "REDUCED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class RiskConfig:
    """Immutable risk-policy configuration."""

    risk_per_trade_pct: float = 0.01
    max_exposure_pct: float = 0.25
    target_volatility_ratio: float = 1.0
    minimum_volatility_multiplier: float = 0.25
    maximum_volatility_multiplier: float = 1.0
    maximum_stop_distance_pct: float = 0.20

    def __post_init__(self) -> None:
        fields = (
            ("risk_per_trade_pct", self.risk_per_trade_pct),
            ("max_exposure_pct", self.max_exposure_pct),
            ("target_volatility_ratio", self.target_volatility_ratio),
            (
                "minimum_volatility_multiplier",
                self.minimum_volatility_multiplier,
            ),
            (
                "maximum_volatility_multiplier",
                self.maximum_volatility_multiplier,
            ),
            ("maximum_stop_distance_pct", self.maximum_stop_distance_pct),
        )

        for name, value in fields:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RiskEngineError(
                    f"{name} must be a real number"
                )
            if not isfinite(float(value)):
                raise RiskEngineError(
                    f"{name} must be finite"
                )

        if not 0.0 < self.risk_per_trade_pct <= 1.0:
            raise RiskEngineError(
                "risk_per_trade_pct must be in (0, 1]"
            )

        if not 0.0 < self.max_exposure_pct <= 1.0:
            raise RiskEngineError(
                "max_exposure_pct must be in (0, 1]"
            )

        if self.target_volatility_ratio <= 0.0:
            raise RiskEngineError(
                "target_volatility_ratio must be positive"
            )

        if not 0.0 < self.minimum_volatility_multiplier <= 1.0:
            raise RiskEngineError(
                "minimum_volatility_multiplier must be in (0, 1]"
            )

        if not 0.0 < self.maximum_volatility_multiplier <= 1.0:
            raise RiskEngineError(
                "maximum_volatility_multiplier must be in (0, 1]"
            )

        if not 0.0 < self.maximum_stop_distance_pct < 1.0:
            raise RiskEngineError(
                "maximum_stop_distance_pct must be in (0, 1)"
            )

        if (
            self.minimum_volatility_multiplier
            > self.maximum_volatility_multiplier
        ):
            raise RiskEngineError(
                "minimum_volatility_multiplier cannot exceed maximum"
            )


@dataclass(frozen=True)
class RiskResult:
    """Immutable and auditable output of the Risk Engine."""

    decision: RiskDecision
    confidence: float
    equity: float
    entry_price: float
    stop_distance: float
    stop_distance_pct: float
    risk_budget: float
    raw_quantity: float
    volatility_ratio: float
    volatility_multiplier: float
    adjusted_quantity: float
    adjusted_notional: float
    maximum_notional: float
    risk_at_stop: float
    actual_risk_pct: float
    reasons: tuple[str, ...]

    @property
    def quantity(self) -> float:
        """Final theoretical Spot quantity."""

        return self.adjusted_quantity

    @property
    def notional(self) -> float:
        """Final theoretical position notional."""

        return self.adjusted_notional

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "equity": self.equity,
            "entry_price": self.entry_price,
            "stop_distance": self.stop_distance,
            "stop_distance_pct": self.stop_distance_pct,
            "risk_budget": self.risk_budget,
            "raw_quantity": self.raw_quantity,
            "volatility_ratio": self.volatility_ratio,
            "volatility_multiplier": self.volatility_multiplier,
            "adjusted_quantity": self.adjusted_quantity,
            "adjusted_notional": self.adjusted_notional,
            "maximum_notional": self.maximum_notional,
            "risk_at_stop": self.risk_at_stop,
            "actual_risk_pct": self.actual_risk_pct,
            "quantity": self.quantity,
            "notional": self.notional,
            "reasons": list(self.reasons),
        }


def _finite_positive(value: float, name: str) -> float:
    """Validate a strictly positive finite scalar."""

    if isinstance(value, bool):
        raise RiskEngineError(f"{name} must be a real number")

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskEngineError(f"{name} must be numeric") from exc

    if not isfinite(normalized):
        raise RiskEngineError(f"{name} must be finite")

    if normalized <= 0.0:
        raise RiskEngineError(f"{name} must be positive")

    return normalized


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to a closed interval."""

    return max(low, min(high, value))


def _volatility_multiplier(
    volatility_ratio: float,
    config: RiskConfig,
) -> float:
    """Reduce theoretical size as current volatility rises."""

    if volatility_ratio <= 0.0:
        raise RiskEngineError(
            "volatility_ratio must be positive"
        )

    raw_multiplier = (
        config.target_volatility_ratio
        / volatility_ratio
    )

    return _clamp(
        raw_multiplier,
        config.minimum_volatility_multiplier,
        config.maximum_volatility_multiplier,
    )


def evaluate(
    *,
    equity: float,
    entry_price: float,
    stop_distance: float,
    volatility_ratio: float = 1.0,
    config: RiskConfig | None = None,
) -> RiskResult:
    """Evaluate risk for one hypothetical Spot position.

    Parameters
    ----------
    equity:
        Current account equity available to the risk policy.

    entry_price:
        Hypothetical entry price.

    stop_distance:
        Absolute distance from entry price to the proposed stop.
        This is not an order and is not submitted anywhere.

    volatility_ratio:
        Current volatility divided by a reference volatility.
        Values above 1 reduce theoretical position size.

    config:
        Immutable risk-policy settings.

    Notes
    -----
    The engine never places an order and never changes account state.
    """

    if config is None:
        config = RiskConfig()

    normalized_equity = _finite_positive(
        equity,
        "equity",
    )
    normalized_entry = _finite_positive(
        entry_price,
        "entry_price",
    )
    normalized_stop = _finite_positive(
        stop_distance,
        "stop_distance",
    )

    if stop_distance >= entry_price:
        raise RiskEngineError(
            "stop_distance must be smaller than entry_price"
        )

    normalized_volatility_ratio = _finite_positive(
        volatility_ratio,
        "volatility_ratio",
    )

    stop_distance_pct = (
        normalized_stop / normalized_entry
    )

    if stop_distance_pct > config.maximum_stop_distance_pct:
        return RiskResult(
            decision=RiskDecision.REJECTED,
            confidence=0.0,
            equity=normalized_equity,
            entry_price=normalized_entry,
            stop_distance=normalized_stop,
            stop_distance_pct=stop_distance_pct,
            risk_budget=normalized_equity * config.risk_per_trade_pct,
            raw_quantity=0.0,
            volatility_ratio=normalized_volatility_ratio,
            volatility_multiplier=0.0,
            adjusted_quantity=0.0,
            adjusted_notional=0.0,
            maximum_notional=normalized_equity * config.max_exposure_pct,
            risk_at_stop=0.0,
            actual_risk_pct=0.0,
            reasons=(
                "stop-distance-exceeds-configured-risk-limit",
            ),
        )

    risk_budget = (
        normalized_equity
        * config.risk_per_trade_pct
    )

    raw_quantity = (
        risk_budget / normalized_stop
    )

    multiplier = _volatility_multiplier(
        normalized_volatility_ratio,
        config,
    )

    adjusted_quantity = (
        raw_quantity * multiplier
    )

    maximum_notional = (
        normalized_equity
        * config.max_exposure_pct
    )

    adjusted_notional = (
        adjusted_quantity
        * normalized_entry
    )

    reasons: list[str] = []
    decision = RiskDecision.APPROVED

    if normalized_volatility_ratio > config.target_volatility_ratio:
        reasons.append(
            "position-size-reduced-for-elevated-volatility"
        )
        decision = RiskDecision.REDUCED

    if adjusted_notional > maximum_notional:
        adjusted_quantity = (
            maximum_notional / normalized_entry
        )
        adjusted_notional = maximum_notional
        reasons.append(
            "position-size-capped-by-maximum-exposure"
        )
        decision = RiskDecision.REDUCED

    risk_at_stop = (
        adjusted_quantity * normalized_stop
    )

    actual_risk_pct = (
        risk_at_stop / normalized_equity
    )

    # A non-zero stop distance is required; the previous validation ensures
    # the theoretical position can be measured safely.
    if adjusted_quantity <= EPS or adjusted_notional <= EPS:
        decision = RiskDecision.REJECTED
        reasons.append(
            "theoretical-position-size-is-zero"
        )

    # This can only occur because of a future implementation/configuration
    # inconsistency. It is kept as a final safety invariant.
    if actual_risk_pct > config.risk_per_trade_pct + EPS:
        adjusted_quantity = (
            risk_budget / normalized_stop
        )
        adjusted_quantity = min(
            adjusted_quantity,
            maximum_notional / normalized_entry,
        )
        adjusted_notional = (
            adjusted_quantity * normalized_entry
        )
        risk_at_stop = (
            adjusted_quantity * normalized_stop
        )
        actual_risk_pct = (
            risk_at_stop / normalized_equity
        )
        reasons.append(
            "risk-budget-final-safety-cap"
        )
        decision = RiskDecision.REDUCED

    # Confidence represents how completely the theoretical position is inside
    # the configured risk envelope; it is not a probability of profit.
    risk_utilization = (
        actual_risk_pct
        / max(config.risk_per_trade_pct, EPS)
    )

    exposure_utilization = (
        adjusted_notional
        / max(maximum_notional, EPS)
    )

    confidence = _clamp(
        1.0
        - 0.5 * min(risk_utilization, 1.0)
        - 0.2 * min(exposure_utilization, 1.0)
        + 0.2 * multiplier,
        0.0,
        1.0,
    )

    return RiskResult(
        decision=decision,
        confidence=confidence,
        equity=normalized_equity,
        entry_price=normalized_entry,
        stop_distance=normalized_stop,
        stop_distance_pct=stop_distance_pct,
        risk_budget=risk_budget,
        raw_quantity=raw_quantity,
        volatility_ratio=normalized_volatility_ratio,
        volatility_multiplier=multiplier,
        adjusted_quantity=adjusted_quantity,
        adjusted_notional=adjusted_notional,
        maximum_notional=maximum_notional,
        risk_at_stop=risk_at_stop,
        actual_risk_pct=actual_risk_pct,
        reasons=tuple(dict.fromkeys(reasons)),
    )


class RiskEngine:
    """Stateless facade around the module-level Risk Engine API."""

    def evaluate(
        self,
        *,
        equity: float,
        entry_price: float,
        stop_distance: float,
        volatility_ratio: float = 1.0,
        config: RiskConfig | None = None,
    ) -> RiskResult:
        """Evaluate hypothetical Spot-position risk."""

        return evaluate(
            equity=equity,
            entry_price=entry_price,
            stop_distance=stop_distance,
            volatility_ratio=volatility_ratio,
            config=config,
        )

    def _self_test(self) -> bool:
        """Run deterministic internal smoke tests."""

        _self_test()
        return True


def _self_test() -> bool:
    """Run deterministic smoke tests without external services."""

    result = evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=5.0,
        volatility_ratio=1.0,
    )

    assert result.decision == RiskDecision.APPROVED
    assert result.risk_budget == 10.0
    assert result.raw_quantity == 2.0
    assert result.adjusted_quantity == 2.0
    assert result.risk_at_stop == 10.0
    assert result.actual_risk_pct == 0.01

    reduced = evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=5.0,
        volatility_ratio=2.0,
    )

    assert reduced.decision == RiskDecision.REDUCED
    assert reduced.adjusted_quantity < result.adjusted_quantity
    assert reduced.risk_at_stop <= reduced.risk_budget

    return True


if __name__ == "__main__":
    _self_test()
    print("Risk Engine self-test: PASS")
