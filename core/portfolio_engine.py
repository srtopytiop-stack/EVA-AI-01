"""
EVA-AI-01 - Portfolio & Position Management
============================================

Phase 9: Portfolio & Position Management.

Spot-only portfolio state and accounting layer.

This module:
- manages cash
- manages open Spot positions
- calculates equity
- calculates exposure
- tracks realized/unrealized PnL
- tracks trading fees
- enforces portfolio exposure limits

This module does NOT:
- fetch market data
- generate signals
- place exchange orders
- use leverage
- use margin
- perform live trading

The next phase (Phase 10) will use this layer for Paper Trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


EPS = 1e-12


class PortfolioError(ValueError):
    """Raised when a portfolio operation violates an invariant."""


@dataclass(frozen=True)
class PortfolioConfig:
    """
    Immutable portfolio-level configuration.

    All exposure limits are expressed as fractions.

    Example:
        0.25 = 25%
    """

    initial_cash: float = 1000.0

    max_total_exposure_pct: float = 0.80

    max_position_exposure_pct: float = 0.25

    min_cash_reserve_pct: float = 0.10

    max_positions: int = 10

    def __post_init__(self) -> None:
        numeric_fields = (
            ("initial_cash", self.initial_cash),
            (
                "max_total_exposure_pct",
                self.max_total_exposure_pct,
            ),
            (
                "max_position_exposure_pct",
                self.max_position_exposure_pct,
            ),
            (
                "min_cash_reserve_pct",
                self.min_cash_reserve_pct,
            ),
        )

        for name, value in numeric_fields:
            if isinstance(value, bool):
                raise PortfolioError(
                    f"{name} must be a real number"
                )

            if not isinstance(value, (int, float)):
                raise PortfolioError(
                    f"{name} must be a real number"
                )

        if self.initial_cash <= 0:
            raise PortfolioError(
                "initial_cash must be > 0"
            )

        if not 0.0 < self.max_total_exposure_pct <= 1.0:
            raise PortfolioError(
                "max_total_exposure_pct must be in (0, 1]"
            )

        if not 0.0 < self.max_position_exposure_pct <= 1.0:
            raise PortfolioError(
                "max_position_exposure_pct must be in (0, 1]"
            )

        if not 0.0 <= self.min_cash_reserve_pct < 1.0:
            raise PortfolioError(
                "min_cash_reserve_pct must be in [0, 1)"
            )

        if (
            not isinstance(self.max_positions, int)
            or isinstance(self.max_positions, bool)
        ):
            raise PortfolioError(
                "max_positions must be an integer"
            )

        if self.max_positions < 1:
            raise PortfolioError(
                "max_positions must be >= 1"
            )

        if (
            self.max_position_exposure_pct
            > self.max_total_exposure_pct
        ):
            raise PortfolioError(
                "max_position_exposure_pct cannot exceed "
                "max_total_exposure_pct"
            )


@dataclass
class Position:
    """
    Represents one open long Spot position.

    Short positions are intentionally unsupported.
    """

    symbol: str

    quantity: float

    entry_price: float

    current_price: float

    cost_basis: float

    entry_fee: float = 0.0

    @property
    def notional(self) -> float:
        """Current market value of the position."""

        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized profit/loss."""

        return self.notional - self.cost_basis

    @property
    def return_pct(self) -> float:
        """Return relative to the fee-adjusted cost basis."""

        if self.cost_basis <= EPS:
            return 0.0

        return self.unrealized_pnl / self.cost_basis


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Immutable point-in-time portfolio valuation.
    """

    cash: float

    equity: float

    gross_exposure: float

    available_cash: float

    unrealized_pnl: float

    realized_pnl: float

    total_fees: float

    position_count: int

    @property
    def exposure_pct(self) -> float:
        """Gross portfolio exposure as a fraction of equity."""

        if self.equity <= EPS:
            return 0.0

        return self.gross_exposure / self.equity


class PortfolioEngine:
    """
    Deterministic in-memory Spot portfolio ledger.

    Portfolio invariants:

    - no short positions
    - no negative cash
    - no negative quantities
    - one open position per symbol
    - maximum position exposure enforced
    - maximum total exposure enforced
    - minimum cash reserve enforced
    - maximum number of positions enforced

    The engine does not communicate with Binance or any other exchange.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
    ) -> None:

        self.config = config or PortfolioConfig()

        self.cash = float(
            self.config.initial_cash
        )

        self.realized_pnl = 0.0

        self.total_fees = 0.0

        self.positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize and validate a trading symbol."""

        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise PortfolioError(
                "symbol must be a non-empty string"
            )

        return symbol.strip().upper()

    @staticmethod
    def _positive(
        value: float,
        name: str,
    ) -> float:
        """Validate a strictly positive numeric value."""

        if isinstance(value, bool):
            raise PortfolioError(
                f"{name} must be a real number"
            )

        try:
            value = float(value)

        except (TypeError, ValueError) as exc:
            raise PortfolioError(
                f"{name} must be numeric"
            ) from exc

        if value <= 0:
            raise PortfolioError(
                f"{name} must be > 0"
            )

        return value

    @staticmethod
    def _nonnegative(
        value: float,
        name: str,
    ) -> float:
        """Validate a non-negative numeric value."""

        if isinstance(value, bool):
            raise PortfolioError(
                f"{name} must be a real number"
            )

        try:
            value = float(value)

        except (TypeError, ValueError) as exc:
            raise PortfolioError(
                f"{name} must be numeric"
            ) from exc

        if value < 0:
            raise PortfolioError(
                f"{name} must be >= 0"
            )

        return value

    @classmethod
    def _fee(
        cls,
        notional: float,
        fee_rate: float,
    ) -> float:
        """Calculate trading fee."""

        normalized_fee_rate = cls._nonnegative(
            fee_rate,
            "fee_rate",
        )

        return notional * normalized_fee_rate

    # ------------------------------------------------------------------
    # Internal portfolio calculations
    # ------------------------------------------------------------------

    def _gross_exposure(self) -> float:
        """Return total current market value of open positions."""

        return sum(
            position.notional
            for position in self.positions.values()
        )

    def _equity(self) -> float:
        """Return current mark-to-market portfolio equity."""

        return (
            self.cash
            + self._gross_exposure()
        )

    # ------------------------------------------------------------------
    # Public portfolio state
    # ------------------------------------------------------------------

    def snapshot(self) -> PortfolioSnapshot:
        """
        Return the current portfolio state.

        Equity is:

            cash + current market value of positions
        """

        unrealized = sum(
            position.unrealized_pnl
            for position in self.positions.values()
        )

        gross_exposure = self._gross_exposure()

        equity = (
            self.cash
            + gross_exposure
        )

        return PortfolioSnapshot(
            cash=self.cash,
            equity=equity,
            gross_exposure=gross_exposure,
            available_cash=self.cash,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            total_fees=self.total_fees,
            position_count=len(self.positions),
        )

    def position(
        self,
        symbol: str,
    ) -> Optional[Position]:
        """Return an open position or None."""

        normalized_symbol = self._normalize_symbol(
            symbol
        )

        return self.positions.get(
            normalized_symbol
        )

    # ------------------------------------------------------------------
    # Market marking
    # ------------------------------------------------------------------

    def mark_price(
        self,
        symbol: str,
        price: float,
    ) -> None:
        """
        Update the current market price of one open position.
        """

        normalized_symbol = self._normalize_symbol(
            symbol
        )

        normalized_price = self._positive(
            price,
            "price",
        )

        if normalized_symbol not in self.positions:
            raise PortfolioError(
                f"no open position for {normalized_symbol}"
            )

        self.positions[
            normalized_symbol
        ].current_price = normalized_price

    def mark_prices(
        self,
        prices: Mapping[str, float],
    ) -> None:
        """
        Update multiple position prices.

        Validation occurs before mutation so an invalid symbol
        does not result in a partially updated batch.
        """

        normalized = {
            self._normalize_symbol(symbol):
            self._positive(price, "price")
            for symbol, price in prices.items()
        }

        missing = sorted(
            symbol
            for symbol in normalized
            if symbol not in self.positions
        )

        if missing:
            raise PortfolioError(
                "no open position for "
                + ", ".join(missing)
            )

        for symbol, price in normalized.items():
            self.positions[
                symbol
            ].current_price = price

    # ------------------------------------------------------------------
    # Opening positions
    # ------------------------------------------------------------------

    def can_open(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        fee_rate: float = 0.0,
    ) -> bool:
        """
        Check whether a position can be opened.

        This method never mutates portfolio state.
        """

        normalized_symbol = self._normalize_symbol(
            symbol
        )

        normalized_quantity = self._positive(
            quantity,
            "quantity",
        )

        normalized_entry_price = self._positive(
            entry_price,
            "entry_price",
        )

        normalized_fee_rate = self._nonnegative(
            fee_rate,
            "fee_rate",
        )

        if normalized_symbol in self.positions:
            return False

        if (
            len(self.positions)
            >= self.config.max_positions
        ):
            return False

        notional = (
            normalized_quantity
            * normalized_entry_price
        )

        fee = self._fee(
            notional,
            normalized_fee_rate,
        )

        equity = self._equity()

        if equity <= EPS:
            return False

        maximum_position_notional = (
            equity
            * self.config.max_position_exposure_pct
        )

        maximum_total_notional = (
            equity
            * self.config.max_total_exposure_pct
        )

        minimum_cash_reserve = (
            equity
            * self.config.min_cash_reserve_pct
        )

        if (
            notional
            > maximum_position_notional + EPS
        ):
            return False

        if (
            self._gross_exposure()
            + notional
            > maximum_total_notional + EPS
        ):
            return False

        if (
            self.cash
            - notional
            - fee
            < minimum_cash_reserve - EPS
        ):
            return False

        return True

    def open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        fee_rate: float = 0.0,
    ) -> Position:
        """
        Open a long Spot position.

        The quantity must already be determined by the
        upstream risk layer.

        This method performs portfolio accounting only.
        """

        normalized_symbol = self._normalize_symbol(
            symbol
        )

        normalized_quantity = self._positive(
            quantity,
            "quantity",
        )

        normalized_entry_price = self._positive(
            entry_price,
            "entry_price",
        )

        notional = (
            normalized_quantity
            * normalized_entry_price
        )

        fee = self._fee(
            notional,
            fee_rate,
        )

        if not self.can_open(
            normalized_symbol,
            normalized_quantity,
            normalized_entry_price,
            fee_rate,
        ):
            raise PortfolioError(
                "portfolio constraints reject "
                f"opening {normalized_symbol}"
            )

        if (
            self.cash
            < notional + fee - EPS
        ):
            raise PortfolioError(
                "insufficient cash"
            )

        self.cash -= (
            notional + fee
        )

        self.total_fees += fee

        position = Position(
            symbol=normalized_symbol,
            quantity=normalized_quantity,
            entry_price=normalized_entry_price,
            current_price=normalized_entry_price,
            cost_basis=notional + fee,
            entry_fee=fee,
        )

        self.positions[
            normalized_symbol
        ] = position

        return position

    # ------------------------------------------------------------------
    # Closing positions
    # ------------------------------------------------------------------

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        quantity: Optional[float] = None,
        fee_rate: float = 0.0,
    ) -> float:
        """
        Close all or part of a position.

        Returns:
            realized PnL for the closed quantity.

        For partial closes, cost basis is reduced
        proportionally.
        """

        normalized_symbol = self._normalize_symbol(
            symbol
        )

        normalized_exit_price = self._positive(
            exit_price,
            "exit_price",
        )

        position = self.positions.get(
            normalized_symbol
        )

        if position is None:
            raise PortfolioError(
                f"no open position for {normalized_symbol}"
            )

        if quantity is None:
            close_quantity = position.quantity

        else:
            close_quantity = self._positive(
                quantity,
                "quantity",
            )

        if (
            close_quantity
            > position.quantity + EPS
        ):
            raise PortfolioError(
                "close quantity exceeds open quantity"
            )

        exit_notional = (
            close_quantity
            * normalized_exit_price
        )

        exit_fee = self._fee(
            exit_notional,
            fee_rate,
        )

        basis_portion = (
            position.cost_basis
            * (
                close_quantity
                / position.quantity
            )
        )

        realized_pnl = (
            exit_notional
            - exit_fee
            - basis_portion
        )

        self.cash += (
            exit_notional
            - exit_fee
        )

        self.realized_pnl += realized_pnl

        self.total_fees += exit_fee

        remaining_quantity = (
            position.quantity
            - close_quantity
        )

        if remaining_quantity <= EPS:
            del self.positions[
                normalized_symbol
            ]

        else:
            position.quantity = (
                remaining_quantity
            )

            position.cost_basis -= (
                basis_portion
            )

            position.current_price = (
                normalized_exit_price
            )

        return realized_pnl

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.

        Useful for deterministic testing and future backtesting.
        """

        self.cash = float(
            self.config.initial_cash
        )

        self.realized_pnl = 0.0

        self.total_fees = 0.0

        self.positions.clear()
