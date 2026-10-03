"""Phase 11 performance-metric tests."""

from __future__ import annotations

import pytest

from core.backtest_metrics import BacktestMetricsError, BacktestTrade, EquityPoint, calculate_metrics


def point(timestamp: int, equity: float, exposure: float = 0.0) -> EquityPoint:
    return EquityPoint(
        timestamp=timestamp,
        equity=equity,
        cash=equity * (1.0 - exposure),
        gross_exposure=equity * exposure,
        exposure_pct=exposure,
        drawdown_pct=0.0,
    )


def test_basic_return_and_drawdown_metrics() -> None:
    curve = [
        point(1_800_000_000_000, 1_000.0),
        point(1_800_000_360_000, 1_100.0),
        point(1_800_000_720_000, 990.0),
        point(1_800_001_080_000, 1_200.0),
    ]
    metrics = calculate_metrics(equity_curve=curve, trades=[], initial_equity=1_000.0)
    assert metrics.total_return_pct == pytest.approx(20.0)
    assert metrics.max_drawdown_pct == pytest.approx(-10.0)
    assert metrics.trade_count == 0
    assert metrics.total_turnover == pytest.approx(0.0)


def test_trade_statistics_and_costs() -> None:
    trade = BacktestTrade(
        trade_id=1,
        symbol="BTCUSDT",
        entry_event_id="PAPER-00000001",
        exit_event_id="PAPER-00000002",
        entry_timestamp=1_800_000_000_000,
        exit_timestamp=1_800_003_600_000,
        entry_price=100.0,
        exit_price=105.0,
        quantity=1.0,
        entry_notional=100.0,
        exit_notional=105.0,
        total_fees=0.205,
        total_spread_cost=0.02,
        total_slippage_cost=0.03,
        total_impact_cost=0.01,
        net_pnl=4.795,
        return_pct=4.795,
        holding_seconds=3600.0,
        entry_signal_score=0.8,
        exit_signal_score=-0.7,
    )
    curve = [
        point(1_800_000_000_000, 1_000.0, 0.10),
        point(1_800_001_800_000, 1_004.795, 0.0),
    ]
    metrics = calculate_metrics(equity_curve=curve, trades=[trade], initial_equity=1_000.0)
    assert metrics.trade_count == 1
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 0
    assert metrics.win_rate_pct == pytest.approx(100.0)
    assert metrics.profit_factor == float("inf")
    assert metrics.total_realized_pnl == pytest.approx(4.795)
    assert metrics.total_fees == pytest.approx(0.205)
    assert metrics.total_spread_cost == pytest.approx(0.02)
    assert metrics.total_slippage_cost == pytest.approx(0.03)
    assert metrics.total_impact_cost == pytest.approx(0.01)
    assert metrics.total_turnover == pytest.approx(205.0)
    assert metrics.average_holding_hours == pytest.approx(1.0)


def test_invalid_equity_curve_is_rejected() -> None:
    with pytest.raises(BacktestMetricsError):
        calculate_metrics(equity_curve=[point(1_800_000_000_000, 1_000.0)], trades=[], initial_equity=1_000.0)

    with pytest.raises(BacktestMetricsError):
        calculate_metrics(
            equity_curve=[
                point(1_800_000_000_000, 1_000.0),
                point(1_800_000_000_000, 1_001.0),
            ],
            trades=[],
            initial_equity=1_000.0,
        )


def test_benchmark_is_reported() -> None:
    metrics = calculate_metrics(
        equity_curve=[
            point(1_800_000_000_000, 1_000.0),
            point(1_800_003_600_000, 1_010.0),
        ],
        trades=[],
        initial_equity=1_000.0,
        benchmark_return=0.20,
    )
    assert metrics.benchmark_return_pct == pytest.approx(20.0)


def test_to_dict_does_not_emit_infinite_profit_factor() -> None:
    trade = BacktestTrade(
        trade_id=1,
        symbol="BTCUSDT",
        entry_event_id="PAPER-1",
        exit_event_id="PAPER-2",
        entry_timestamp=1_800_000_000_000,
        exit_timestamp=1_800_000_060_000,
        entry_price=100.0,
        exit_price=101.0,
        quantity=1.0,
        entry_notional=100.0,
        exit_notional=101.0,
        total_fees=0.0,
        total_spread_cost=0.0,
        total_slippage_cost=0.0,
        total_impact_cost=0.0,
        net_pnl=1.0,
        return_pct=1.0,
        holding_seconds=60.0,
        entry_signal_score=0.8,
        exit_signal_score=-0.8,
    )
    metrics = calculate_metrics(
        equity_curve=[
            point(1_800_000_000_000, 1_000.0),
            point(1_800_000_060_000, 1_001.0),
        ],
        trades=[trade],
        initial_equity=1_000.0,
    )
    assert metrics.to_dict()["profit_factor"] is None
