"""
EVA-AI-01 - Portfolio Engine Tests
===================================

Phase 9: Portfolio & Position Management.

Tests:
- portfolio initialization
- position opening
- exposure limits
- cash reserve
- maximum positions
- mark-to-market
- unrealized PnL
- realized PnL
- partial closes
- full closes
- fees
- reset behavior
"""

import math

import pytest

from core.portfolio_engine import (
    PortfolioConfig,
    PortfolioEngine,
    PortfolioError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def portfolio() -> PortfolioEngine:
    """Return a deterministic default portfolio."""

    return PortfolioEngine(
        PortfolioConfig(
            initial_cash=1000.0,
            max_total_exposure_pct=0.80,
            max_position_exposure_pct=0.25,
            min_cash_reserve_pct=0.10,
            max_positions=10,
        )
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initial_portfolio_state() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(initial_cash=1000.0)
    )

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(1000.0)
    assert snapshot.equity == pytest.approx(1000.0)
    assert snapshot.gross_exposure == pytest.approx(0.0)
    assert snapshot.available_cash == pytest.approx(1000.0)
    assert snapshot.unrealized_pnl == pytest.approx(0.0)
    assert snapshot.realized_pnl == pytest.approx(0.0)
    assert snapshot.total_fees == pytest.approx(0.0)
    assert snapshot.position_count == 0
    assert snapshot.exposure_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_cash": 0},
        {"initial_cash": -1},
        {"max_total_exposure_pct": 0},
        {"max_total_exposure_pct": 1.1},
        {"max_position_exposure_pct": 0},
        {"max_position_exposure_pct": 1.1},
        {"min_cash_reserve_pct": -0.1},
        {"min_cash_reserve_pct": 1.0},
        {"max_positions": 0},
    ],
)
def test_invalid_configuration(kwargs) -> None:
    with pytest.raises(PortfolioError):
        PortfolioConfig(**kwargs)


def test_position_exposure_cannot_exceed_total_exposure() -> None:
    with pytest.raises(PortfolioError):
        PortfolioConfig(
            max_total_exposure_pct=0.20,
            max_position_exposure_pct=0.30,
        )


# ---------------------------------------------------------------------------
# Symbol / numeric validation
# ---------------------------------------------------------------------------


def test_empty_symbol_is_rejected(
    portfolio: PortfolioEngine,
) -> None:
    with pytest.raises(PortfolioError):
        portfolio.open_position(
            "",
            quantity=1.0,
            entry_price=100.0,
        )


@pytest.mark.parametrize(
    "quantity",
    [0, -1],
)
def test_invalid_quantity_is_rejected(
    portfolio: PortfolioEngine,
    quantity: float,
) -> None:
    with pytest.raises(PortfolioError):
        portfolio.open_position(
            "BTCUSDT",
            quantity=quantity,
            entry_price=100.0,
        )


@pytest.mark.parametrize(
    "price",
    [0, -1],
)
def test_invalid_entry_price_is_rejected(
    portfolio: PortfolioEngine,
    price: float,
) -> None:
    with pytest.raises(PortfolioError):
        portfolio.open_position(
            "BTCUSDT",
            quantity=1.0,
            entry_price=price,
        )


# ---------------------------------------------------------------------------
# Opening positions
# ---------------------------------------------------------------------------


def test_open_position_updates_cash(
    portfolio: PortfolioEngine,
) -> None:
    position = portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    assert position.symbol == "BTCUSDT"
    assert position.quantity == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(100.0)
    assert position.current_price == pytest.approx(100.0)

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(900.0)
    assert snapshot.gross_exposure == pytest.approx(100.0)
    assert snapshot.equity == pytest.approx(1000.0)
    assert snapshot.position_count == 1


def test_symbol_is_normalized(
    portfolio: PortfolioEngine,
) -> None:
    position = portfolio.open_position(
        " btcusdt ",
        quantity=1.0,
        entry_price=100.0,
    )

    assert position.symbol == "BTCUSDT"
    assert portfolio.position("btcusdt") is position


def test_duplicate_position_is_rejected(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    with pytest.raises(PortfolioError):
        portfolio.open_position(
            "BTCUSDT",
            quantity=1.0,
            entry_price=100.0,
        )


# ---------------------------------------------------------------------------
# Exposure constraints
# ---------------------------------------------------------------------------


def test_position_exposure_limit_is_enforced() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(
            initial_cash=1000.0,
            max_position_exposure_pct=0.25,
        )
    )

    assert portfolio.can_open(
        "BTCUSDT",
        quantity=2.5,
        entry_price=100.0,
    )

    assert not portfolio.can_open(
        "ETHUSDT",
        quantity=2.51,
        entry_price=100.0,
    )


def test_total_exposure_limit_is_enforced() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(
            initial_cash=1000.0,
            max_total_exposure_pct=0.50,
            max_position_exposure_pct=0.30,
        )
    )

    portfolio.open_position(
        "BTCUSDT",
        quantity=3.0,
        entry_price=100.0,
    )

    assert portfolio.can_open(
        "ETHUSDT",
        quantity=2.0,
        entry_price=100.0,
    )

    assert not portfolio.can_open(
        "SOLUSDT",
        quantity=2.01,
        entry_price=100.0,
    )


def test_max_positions_is_enforced() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(
            initial_cash=10000.0,
            max_total_exposure_pct=1.0,
            max_position_exposure_pct=0.20,
            max_positions=2,
        )
    )

    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    portfolio.open_position(
        "ETHUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    assert not portfolio.can_open(
        "SOLUSDT",
        quantity=1.0,
        entry_price=100.0,
    )


def test_minimum_cash_reserve_is_enforced() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(
            initial_cash=1000.0,
            max_total_exposure_pct=1.0,
            max_position_exposure_pct=1.0,
            min_cash_reserve_pct=0.20,
        )
    )

    assert portfolio.can_open(
        "BTCUSDT",
        quantity=8.0,
        entry_price=100.0,
    )

    assert not portfolio.can_open(
        "ETHUSDT",
        quantity=8.1,
        entry_price=100.0,
    )


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------


def test_entry_fee_is_accounted_for(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        fee_rate=0.001,
    )

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(899.9)
    assert snapshot.total_fees == pytest.approx(0.1)

    position = portfolio.position("BTCUSDT")

    assert position is not None
    assert position.cost_basis == pytest.approx(100.1)


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------


def test_mark_price_updates_unrealized_pnl(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    portfolio.mark_price(
        "BTCUSDT",
        120.0,
    )

    snapshot = portfolio.snapshot()

    assert snapshot.gross_exposure == pytest.approx(120.0)
    assert snapshot.equity == pytest.approx(1020.0)
    assert snapshot.unrealized_pnl == pytest.approx(20.0)


def test_mark_price_loss(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    portfolio.mark_price(
        "BTCUSDT",
        80.0,
    )

    snapshot = portfolio.snapshot()

    assert snapshot.unrealized_pnl == pytest.approx(-20.0)
    assert snapshot.equity == pytest.approx(980.0)


def test_mark_unknown_position_is_rejected(
    portfolio: PortfolioEngine,
) -> None:
    with pytest.raises(PortfolioError):
        portfolio.mark_price(
            "BTCUSDT",
            100.0,
        )


def test_batch_mark_prices(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    portfolio.open_position(
        "ETHUSDT",
        quantity=2.0,
        entry_price=100.0,
    )

    portfolio.mark_prices(
        {
            "BTCUSDT": 110.0,
            "ETHUSDT": 90.0,
        }
    )

    snapshot = portfolio.snapshot()

    assert snapshot.unrealized_pnl == pytest.approx(-10.0)
    assert snapshot.gross_exposure == pytest.approx(290.0)


# ---------------------------------------------------------------------------
# Full close
# ---------------------------------------------------------------------------


def test_full_close_realizes_profit(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    realized = portfolio.close_position(
        "BTCUSDT",
        exit_price=120.0,
    )

    assert realized == pytest.approx(20.0)

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(1020.0)
    assert snapshot.equity == pytest.approx(1020.0)
    assert snapshot.gross_exposure == pytest.approx(0.0)
    assert snapshot.realized_pnl == pytest.approx(20.0)
    assert snapshot.position_count == 0


def test_full_close_realizes_loss(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    realized = portfolio.close_position(
        "BTCUSDT",
        exit_price=80.0,
    )

    assert realized == pytest.approx(-20.0)

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(980.0)
    assert snapshot.realized_pnl == pytest.approx(-20.0)
    assert snapshot.position_count == 0


# ---------------------------------------------------------------------------
# Exit fees
# ---------------------------------------------------------------------------


def test_exit_fee_is_accounted_for(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    realized = portfolio.close_position(
        "BTCUSDT",
        exit_price=120.0,
        fee_rate=0.001,
    )

    assert realized == pytest.approx(19.88)

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(1019.88)
    assert snapshot.realized_pnl == pytest.approx(19.88)
    assert snapshot.total_fees == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Partial close
# ---------------------------------------------------------------------------


def test_partial_close_reduces_position() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(initial_cash=1000.0)
    )

    portfolio.open_position(
        "BTCUSDT",
        quantity=2.0,
        entry_price=100.0,
    )

    realized = portfolio.close_position(
        "BTCUSDT",
        exit_price=120.0,
        quantity=1.0,
    )

    assert realized == pytest.approx(20.0)

    position = portfolio.position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(1.0)
    assert position.cost_basis == pytest.approx(100.0)

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(920.0)
    assert snapshot.realized_pnl == pytest.approx(20.0)


def test_partial_close_with_fee() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(initial_cash=1000.0)
    )

    portfolio.open_position(
        "BTCUSDT",
        quantity=2.0,
        entry_price=100.0,
        fee_rate=0.001,
    )

    realized = portfolio.close_position(
        "BTCUSDT",
        exit_price=120.0,
        quantity=1.0,
        fee_rate=0.001,
    )

    assert realized == pytest.approx(19.86)

    position = portfolio.position("BTCUSDT")

    assert position is not None
    assert position.quantity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_close_unknown_position_is_rejected(
    portfolio: PortfolioEngine,
) -> None:
    with pytest.raises(PortfolioError):
        portfolio.close_position(
            "BTCUSDT",
            exit_price=100.0,
        )


def test_close_more_than_owned_is_rejected(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    with pytest.raises(PortfolioError):
        portfolio.close_position(
            "BTCUSDT",
            exit_price=100.0,
            quantity=1.1,
        )


def test_close_zero_quantity_is_rejected(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    with pytest.raises(PortfolioError):
        portfolio.close_position(
            "BTCUSDT",
            exit_price=100.0,
            quantity=0,
        )


# ---------------------------------------------------------------------------
# Position properties
# ---------------------------------------------------------------------------


def test_position_return_percentage(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
    )

    portfolio.mark_price(
        "BTCUSDT",
        110.0,
    )

    position = portfolio.position("BTCUSDT")

    assert position is not None
    assert position.unrealized_pnl == pytest.approx(10.0)
    assert position.return_pct == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_restores_initial_state(
    portfolio: PortfolioEngine,
) -> None:
    portfolio.open_position(
        "BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        fee_rate=0.001,
    )

    portfolio.mark_price(
        "BTCUSDT",
        120.0,
    )

    portfolio.close_position(
        "BTCUSDT",
        exit_price=120.0,
        fee_rate=0.001,
    )

    portfolio.reset()

    snapshot = portfolio.snapshot()

    assert snapshot.cash == pytest.approx(1000.0)
    assert snapshot.equity == pytest.approx(1000.0)
    assert snapshot.gross_exposure == pytest.approx(0.0)
    assert snapshot.realized_pnl == pytest.approx(0.0)
    assert snapshot.unrealized_pnl == pytest.approx(0.0)
    assert snapshot.total_fees == pytest.approx(0.0)
    assert snapshot.position_count == 0


# ---------------------------------------------------------------------------
# Accounting invariant
# ---------------------------------------------------------------------------


def test_equity_consistency_after_profit() -> None:
    portfolio = PortfolioEngine(
        PortfolioConfig(initial_cash=1000.0)
    )

    portfolio.open_position(
        "BTCUSDT",
        quantity=2.0,
        entry_price=100.0,
    )

    portfolio.mark_price(
        "BTCUSDT",
        125.0,
    )

    snapshot = portfolio.snapshot()

    assert math.isclose(
        snapshot.equity,
        snapshot.cash + snapshot.gross_exposure,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert snapshot.unrealized_pnl == pytest.approx(50.0)
