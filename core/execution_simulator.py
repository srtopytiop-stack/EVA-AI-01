"""
EVA-AI-01 - Paper Execution Simulator
======================================

Phase 10.

Deterministic Spot market-execution simulator.

The simulator models:
- half bid/ask spread
- baseline slippage
- volatility-dependent slippage
- nonlinear liquidity impact
- optional participation-cap rejection

It does NOT:
- connect to an exchange
- submit orders
- use leverage/margin
- predict future prices
- calculate strategy signals
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


EPS = 1e-12
BPS = 10_000.0


class ExecutionError(ValueError):
    """Raised when execution-simulation inputs are invalid."""


class ExecutionSide(str, Enum):
    """Supported Spot execution directions."""

    BUY = "BUY"
    SELL = "SELL"


class ExecutionStatus(str, Enum):
    """Execution-simulation outcome."""

    FILLED = "FILLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Configuration for the paper execution model.

    All *_bps values are basis points.

    The model deliberately keeps trading costs separate:
    spread is a market-friction component, while slippage/impact
    model the execution price moving away from the reference price.

    These parameters are assumptions and must be calibrated from
    observed market data before being used for research conclusions.
    """

    half_spread_bps: float = 1.0
    base_slippage_bps: float = 0.5
    volatility_slippage_factor: float = 0.05

    impact_coefficient_bps: float = 2.0
    impact_exponent: float = 0.5

    max_participation_rate: float = 0.10
    reject_when_liquidity_missing: bool = False

    def __post_init__(self) -> None:
        numeric = (
            ("half_spread_bps", self.half_spread_bps),
            ("base_slippage_bps", self.base_slippage_bps),
            (
                "volatility_slippage_factor",
                self.volatility_slippage_factor,
            ),
            (
                "impact_coefficient_bps",
                self.impact_coefficient_bps,
            ),
            ("impact_exponent", self.impact_exponent),
            (
                "max_participation_rate",
                self.max_participation_rate,
            ),
        )

        for name, value in numeric:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ExecutionError(
                    f"{name} must be a real number"
                )

            if not isfinite(float(value)):
                raise ExecutionError(
                    f"{name} must be finite"
                )

        if self.half_spread_bps < 0.0:
            raise ExecutionError(
                "half_spread_bps must be >= 0"
            )

        if self.base_slippage_bps < 0.0:
            raise ExecutionError(
                "base_slippage_bps must be >= 0"
            )

        if self.volatility_slippage_factor < 0.0:
            raise ExecutionError(
                "volatility_slippage_factor must be >= 0"
            )

        if self.impact_coefficient_bps < 0.0:
            raise ExecutionError(
                "impact_coefficient_bps must be >= 0"
            )

        if not 0.0 < self.impact_exponent <= 1.0:
            raise ExecutionError(
                "impact_exponent must be in (0, 1]"
            )

        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ExecutionError(
                "max_participation_rate must be in (0, 1]"
            )

        if not isinstance(
            self.reject_when_liquidity_missing,
            bool,
        ):
            raise ExecutionError(
                "reject_when_liquidity_missing must be boolean"
            )


@dataclass(frozen=True)
class ExecutionFill:
    """Immutable result of a simulated market execution."""

    side: ExecutionSide
    status: ExecutionStatus

    requested_quantity: float
    filled_quantity: float

    reference_price: float
    execution_price: float

    gross_notional: float

    spread_cost: float
    slippage_cost: float
    impact_cost: float
    impact_bps: float
    effective_cost_bps: float

    reason: str = ""

    @property
    def fully_filled(self) -> bool:
        """Whether the requested quantity was completely filled."""

        return (
            self.status == ExecutionStatus.FILLED
            and abs(
                self.filled_quantity
                - self.requested_quantity
            )
            <= EPS
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "side": self.side.value,
            "status": self.status.value,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "reference_price": self.reference_price,
            "execution_price": self.execution_price,
            "gross_notional": self.gross_notional,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "impact_cost": self.impact_cost,
            "impact_bps": self.impact_bps,
            "effective_cost_bps": self.effective_cost_bps,
            "reason": self.reason,
        }


class ExecutionSimulator:
    """
    Deterministic market-order execution model.

    Execution price is derived from a supplied reference price.

    BUY:
        reference
        + half spread
        + base slippage
        + volatility slippage
        + liquidity impact

    SELL:
        reference
        - half spread
        - base slippage
        - volatility slippage
        - liquidity impact

    No future price is inferred by this class.
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()

    @staticmethod
    def _positive(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise ExecutionError(
                f"{name} must be a real number"
            )

        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                f"{name} must be numeric"
            ) from exc

        if not isfinite(normalized):
            raise ExecutionError(
                f"{name} must be finite"
            )

        if normalized <= 0.0:
            raise ExecutionError(
                f"{name} must be > 0"
            )

        return normalized

    @staticmethod
    def _nonnegative(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise ExecutionError(
                f"{name} must be a real number"
            )

        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                f"{name} must be numeric"
            ) from exc

        if not isfinite(normalized):
            raise ExecutionError(
                f"{name} must be finite"
            )

        if normalized < 0.0:
            raise ExecutionError(
                f"{name} must be >= 0"
            )

        return normalized

    def market_order(
        self,
        side: ExecutionSide | str,
        quantity: float,
        reference_price: float,
        *,
        volatility: float = 0.0,
        liquidity_notional: float | None = None,
    ) -> ExecutionFill:
        """
        Simulate a fully-filled Spot market order.

        When liquidity_notional is supplied, the order must not exceed
        max_participation_rate * liquidity_notional. This is a conservative
        execution gate rather than a claim about actual order-book depth.
        """

        try:
            normalized_side = (
                side
                if isinstance(side, ExecutionSide)
                else ExecutionSide(str(side).upper())
            )
        except ValueError as exc:
            raise ExecutionError(
                "side must be BUY or SELL"
            ) from exc

        normalized_quantity = self._positive(
            quantity,
            "quantity",
        )

        normalized_reference = self._positive(
            reference_price,
            "reference_price",
        )

        normalized_volatility = self._nonnegative(
            volatility,
            "volatility",
        )

        if liquidity_notional is not None:
            normalized_liquidity = self._positive(
                liquidity_notional,
                "liquidity_notional",
            )
        else:
            normalized_liquidity = None

        requested_notional = (
            normalized_quantity
            * normalized_reference
        )

        if (
            normalized_liquidity is None
            and self.config.reject_when_liquidity_missing
        ):
            return self._rejected(
                normalized_side,
                normalized_quantity,
                normalized_reference,
                "liquidity-notional-is-required",
            )

        if normalized_liquidity is not None:
            participation = (
                requested_notional
                / normalized_liquidity
            )

            if (
                participation
                > self.config.max_participation_rate
                + EPS
            ):
                return self._rejected(
                    normalized_side,
                    normalized_quantity,
                    normalized_reference,
                    "order-exceeds-participation-limit",
                )
        else:
            participation = 0.0

        volatility_bps = (
            normalized_volatility
            * BPS
            * self.config.volatility_slippage_factor
        )

        impact_bps = (
            self.config.impact_coefficient_bps
            * (
                participation
                ** self.config.impact_exponent
            )
        )

        variable_slippage_bps = (
            self.config.base_slippage_bps
            + volatility_bps
            + impact_bps
        )

        half_spread_bps = (
            self.config.half_spread_bps
        )

        total_adjustment_bps = (
            half_spread_bps
            + variable_slippage_bps
        )

        adjustment = (
            total_adjustment_bps
            / BPS
        )

        if normalized_side == ExecutionSide.BUY:
            execution_price = (
                normalized_reference
                * (1.0 + adjustment)
            )
        else:
            if adjustment >= 1.0:
                return self._rejected(
                    normalized_side,
                    normalized_quantity,
                    normalized_reference,
                    "execution-adjustment-would-make-sell-price-nonpositive",
                )

            execution_price = (
                normalized_reference
                * (1.0 - adjustment)
            )

        gross_notional = (
            normalized_quantity
            * execution_price
        )

        spread_cost = (
            normalized_quantity
            * normalized_reference
            * half_spread_bps
            / BPS
        )

        pure_slippage_bps = (
            self.config.base_slippage_bps
            + volatility_bps
        )

        slippage_cost = (
            normalized_quantity
            * normalized_reference
            * pure_slippage_bps
            / BPS
        )

        impact_cost = (
            normalized_quantity
            * normalized_reference
            * impact_bps
            / BPS
        )

        reference_notional = requested_notional

        if reference_notional <= EPS:
            effective_cost_bps = 0.0
        else:
            if normalized_side == ExecutionSide.BUY:
                price_cost = (
                    gross_notional
                    - reference_notional
                )
            else:
                price_cost = (
                    reference_notional
                    - gross_notional
                )

            effective_cost_bps = (
                price_cost
                / reference_notional
                * BPS
            )

        return ExecutionFill(
            side=normalized_side,
            status=ExecutionStatus.FILLED,
            requested_quantity=normalized_quantity,
            filled_quantity=normalized_quantity,
            reference_price=normalized_reference,
            execution_price=execution_price,
            gross_notional=gross_notional,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            impact_cost=impact_cost,
            impact_bps=impact_bps,
            effective_cost_bps=effective_cost_bps,
        )

    def _rejected(
        self,
        side: ExecutionSide,
        quantity: float,
        reference_price: float,
        reason: str,
    ) -> ExecutionFill:
        return ExecutionFill(
            side=side,
            status=ExecutionStatus.REJECTED,
            requested_quantity=quantity,
            filled_quantity=0.0,
            reference_price=reference_price,
            execution_price=0.0,
            gross_notional=0.0,
            spread_cost=0.0,
            slippage_cost=0.0,
            impact_cost=0.0,
            impact_bps=0.0,
            effective_cost_bps=0.0,
            reason=reason,
        )


if __name__ == "__main__":
    simulator = ExecutionSimulator()

    result = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert result.status == ExecutionStatus.FILLED
    print("EXECUTION_SIMULATOR_SELF_TEST_OK")
