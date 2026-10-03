"""
EVA-AI-01 - Backtest Performance Metrics
========================================
Phase 11: Historical backtest performance measurement.

Post-trade analytics only. No signal generation, execution, or portfolio
mutation occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from statistics import mean, median
from typing import Sequence

SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
EPS = 1e-12


class BacktestMetricsError(ValueError):
    """Raised when backtest performance inputs are invalid."""


@dataclass(frozen=True)
class EquityPoint:
    """Immutable point on the marked-to-market equity curve."""

    timestamp: int
    equity: float
    cash: float
    gross_exposure: float
    exposure_pct: float
    drawdown_pct: float

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise BacktestMetricsError("timestamp must be an integer")
        if self.timestamp <= 0:
            raise BacktestMetricsError("timestamp must be positive")
        for name, value in (
            ("equity", self.equity),
            ("cash", self.cash),
            ("gross_exposure", self.gross_exposure),
            ("exposure_pct", self.exposure_pct),
            ("drawdown_pct", self.drawdown_pct),
        ):
            if not isfinite(float(value)):
                raise BacktestMetricsError(f"{name} must be finite")
        if self.equity <= 0.0:
            raise BacktestMetricsError("equity must be positive")
        if self.cash < -EPS:
            raise BacktestMetricsError("cash cannot be negative")
        if self.gross_exposure < -EPS:
            raise BacktestMetricsError("gross_exposure cannot be negative")
        if not 0.0 <= self.exposure_pct <= 1.0 + EPS:
            raise BacktestMetricsError("exposure_pct must be in [0, 1]")
        if self.drawdown_pct > EPS or self.drawdown_pct < -1.0 - EPS:
            raise BacktestMetricsError("drawdown_pct must be in [-1, 0]")

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "equity": self.equity,
            "cash": self.cash,
            "gross_exposure": self.gross_exposure,
            "exposure_pct": self.exposure_pct,
            "drawdown_pct": self.drawdown_pct,
        }


@dataclass(frozen=True)
class BacktestTrade:
    """Immutable completed long-Spot trade reconstructed from paper events."""

    trade_id: int
    symbol: str
    entry_event_id: str
    exit_event_id: str
    entry_timestamp: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    quantity: float
    entry_notional: float
    exit_notional: float
    total_fees: float
    total_spread_cost: float
    total_slippage_cost: float
    total_impact_cost: float
    net_pnl: float
    return_pct: float
    holding_seconds: float
    entry_signal_score: float
    exit_signal_score: float

    def __post_init__(self) -> None:
        if self.trade_id < 1:
            raise BacktestMetricsError("trade_id must be >= 1")
        if not self.symbol.strip():
            raise BacktestMetricsError("symbol must not be empty")
        if self.entry_timestamp <= 0 or self.exit_timestamp <= 0:
            raise BacktestMetricsError("trade timestamps must be positive")
        if self.exit_timestamp < self.entry_timestamp:
            raise BacktestMetricsError("exit must not precede entry")
        for name, value in (
            ("entry_price", self.entry_price),
            ("exit_price", self.exit_price),
            ("quantity", self.quantity),
            ("entry_notional", self.entry_notional),
            ("exit_notional", self.exit_notional),
            ("total_fees", self.total_fees),
            ("total_spread_cost", self.total_spread_cost),
            ("total_slippage_cost", self.total_slippage_cost),
            ("total_impact_cost", self.total_impact_cost),
            ("net_pnl", self.net_pnl),
            ("return_pct", self.return_pct),
            ("holding_seconds", self.holding_seconds),
            ("entry_signal_score", self.entry_signal_score),
            ("exit_signal_score", self.exit_signal_score),
        ):
            if not isfinite(float(value)):
                raise BacktestMetricsError(f"{name} must be finite")
        if self.entry_price <= 0.0 or self.exit_price <= 0.0:
            raise BacktestMetricsError("trade prices must be positive")
        if self.quantity <= 0.0:
            raise BacktestMetricsError("trade quantity must be positive")
        if min(
            self.total_fees,
            self.total_spread_cost,
            self.total_slippage_cost,
            self.total_impact_cost,
        ) < -EPS:
            raise BacktestMetricsError("trade costs cannot be negative")
        if self.holding_seconds < -EPS:
            raise BacktestMetricsError("holding_seconds cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_event_id": self.entry_event_id,
            "exit_event_id": self.exit_event_id,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_notional": self.entry_notional,
            "exit_notional": self.exit_notional,
            "total_fees": self.total_fees,
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "total_impact_cost": self.total_impact_cost,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "holding_seconds": self.holding_seconds,
            "entry_signal_score": self.entry_signal_score,
            "exit_signal_score": self.exit_signal_score,
        }


@dataclass(frozen=True)
class PerformanceMetrics:
    """Complete backtest performance summary."""

    initial_equity: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_hours: float
    calmar_ratio: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    average_trade_pnl: float
    median_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    average_holding_hours: float
    total_realized_pnl: float
    total_fees: float
    total_spread_cost: float
    total_slippage_cost: float
    total_impact_cost: float
    total_turnover: float
    turnover_to_initial_equity: float
    average_exposure_pct: float
    max_exposure_pct: float
    benchmark_return_pct: float | None
    bars_processed: int
    rejected_actions: int

    def to_dict(self) -> dict[str, object]:
        profit_factor = _json_number(self.profit_factor)
        calmar_ratio = _json_number(self.calmar_ratio)
        annualized_return_pct = _json_number(self.annualized_return_pct)
        annualized_volatility_pct = _json_number(self.annualized_volatility_pct)
        sharpe_ratio = _json_number(self.sharpe_ratio)
        sortino_ratio = _json_number(self.sortino_ratio)
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "annualized_volatility_pct": annualized_volatility_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_hours": self.max_drawdown_duration_hours,
            "calmar_ratio": calmar_ratio,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": profit_factor,
            "average_trade_pnl": self.average_trade_pnl,
            "median_trade_pnl": self.median_trade_pnl,
            "best_trade_pnl": self.best_trade_pnl,
            "worst_trade_pnl": self.worst_trade_pnl,
            "average_holding_hours": self.average_holding_hours,
            "total_realized_pnl": self.total_realized_pnl,
            "total_fees": self.total_fees,
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "total_impact_cost": self.total_impact_cost,
            "total_turnover": self.total_turnover,
            "turnover_to_initial_equity": self.turnover_to_initial_equity,
            "average_exposure_pct": self.average_exposure_pct,
            "max_exposure_pct": self.max_exposure_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "bars_processed": self.bars_processed,
            "rejected_actions": self.rejected_actions,
        }


def _validate_equity_curve(equity_curve: Sequence[EquityPoint]) -> None:
    if len(equity_curve) < 2:
        raise BacktestMetricsError("equity_curve requires at least two points")
    previous_timestamp: int | None = None
    for point in equity_curve:
        if previous_timestamp is not None and point.timestamp <= previous_timestamp:
            raise BacktestMetricsError("equity timestamps must be strictly increasing")
        previous_timestamp = point.timestamp


def _annualization_factor(equity_curve: Sequence[EquityPoint]) -> float:
    intervals = [
        (current.timestamp - previous.timestamp) / 1000.0
        for previous, current in zip(equity_curve[:-1], equity_curve[1:])
    ]
    positive = [interval for interval in intervals if interval > 0.0]
    if not positive:
        return 1.0
    return SECONDS_PER_YEAR / median(positive)


def _max_drawdown(equity_curve: Sequence[EquityPoint]) -> tuple[float, float]:
    peak_equity = equity_curve[0].equity
    peak_timestamp = equity_curve[0].timestamp
    max_drawdown = 0.0
    max_duration_seconds = 0.0
    for point in equity_curve:
        if point.equity > peak_equity:
            peak_equity = point.equity
            peak_timestamp = point.timestamp
        drawdown = point.equity / peak_equity - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
        if drawdown < -EPS:
            max_duration_seconds = max(
                max_duration_seconds,
                (point.timestamp - peak_timestamp) / 1000.0,
            )
    return max_drawdown, max_duration_seconds


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= EPS:
        if abs(numerator) <= EPS:
            return 0.0
        return float("inf") if numerator > 0.0 else float("-inf")
    return numerator / denominator


def _json_number(value: float) -> float | None:
    """Return a standard-JSON-safe number."""
    return float(value) if isfinite(float(value)) else None


def calculate_metrics(
    *,
    equity_curve: Sequence[EquityPoint],
    trades: Sequence[BacktestTrade],
    initial_equity: float,
    benchmark_return: float | None = None,
    rejected_actions: int = 0,
    risk_free_rate_annual: float = 0.0,
) -> PerformanceMetrics:
    """Calculate deterministic performance statistics."""
    _validate_equity_curve(equity_curve)
    if not isfinite(float(initial_equity)) or initial_equity <= 0.0:
        raise BacktestMetricsError("initial_equity must be positive and finite")
    if rejected_actions < 0:
        raise BacktestMetricsError("rejected_actions cannot be negative")
    if not isfinite(float(risk_free_rate_annual)) or risk_free_rate_annual <= -1.0:
        raise BacktestMetricsError("risk_free_rate_annual must be finite and > -1")

    final = equity_curve[-1].equity
    total_return = final / initial_equity - 1.0
    elapsed_seconds = (equity_curve[-1].timestamp - equity_curve[0].timestamp) / 1000.0
    if elapsed_seconds > 0.0 and final > 0.0:
        log_growth = log(final / initial_equity)
        annualized_log_growth = log_growth * (SECONDS_PER_YEAR / elapsed_seconds)
        if annualized_log_growth > 709.0:
            annualized_return = float("inf")
        elif annualized_log_growth < -745.0:
            annualized_return = -1.0
        else:
            annualized_return = exp(annualized_log_growth) - 1.0
    else:
        annualized_return = 0.0

    bar_returns = [
        current.equity / previous.equity - 1.0
        for previous, current in zip(equity_curve[:-1], equity_curve[1:])
    ]
    periods_per_year = _annualization_factor(equity_curve)
    risk_free_bar = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess_returns = [value - risk_free_bar for value in bar_returns]

    if len(excess_returns) >= 2:
        excess_mean = mean(excess_returns)
        variance = sum((value - excess_mean) ** 2 for value in excess_returns) / (len(excess_returns) - 1)
        excess_std = sqrt(max(variance, 0.0))
    else:
        excess_mean = 0.0
        excess_std = 0.0

    sharpe = excess_mean / excess_std * sqrt(periods_per_year) if excess_std > EPS else 0.0

    downside_squared = [min(value, 0.0) ** 2 for value in excess_returns]
    downside_deviation = sqrt(mean(downside_squared)) if downside_squared else 0.0
    sortino = excess_mean / downside_deviation * sqrt(periods_per_year) if downside_deviation > EPS else 0.0

    if len(bar_returns) >= 2:
        raw_mean = mean(bar_returns)
        raw_variance = sum((value - raw_mean) ** 2 for value in bar_returns) / (len(bar_returns) - 1)
        annualized_volatility = sqrt(max(raw_variance, 0.0)) * sqrt(periods_per_year)
    else:
        annualized_volatility = 0.0

    max_drawdown, max_duration_seconds = _max_drawdown(equity_curve)
    calmar = _safe_ratio(annualized_return, abs(max_drawdown))

    pnl_values = [trade.net_pnl for trade in trades]
    positive_pnl = [pnl for pnl in pnl_values if pnl > EPS]
    negative_pnl = [pnl for pnl in pnl_values if pnl < -EPS]
    gross_profit = sum(positive_pnl)
    gross_loss = abs(sum(negative_pnl))
    if gross_loss > EPS:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > EPS:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    if pnl_values:
        average_trade_pnl = mean(pnl_values)
        median_trade_pnl = median(pnl_values)
        best_trade_pnl = max(pnl_values)
        worst_trade_pnl = min(pnl_values)
        average_holding_hours = mean(trade.holding_seconds for trade in trades) / 3600.0
    else:
        average_trade_pnl = median_trade_pnl = best_trade_pnl = worst_trade_pnl = 0.0
        average_holding_hours = 0.0

    trade_count = len(trades)
    winning_trades = len(positive_pnl)
    losing_trades = len(negative_pnl)
    win_rate = winning_trades / trade_count if trade_count else 0.0

    total_realized_pnl = sum(pnl_values)
    total_fees = sum(trade.total_fees for trade in trades)
    total_spread_cost = sum(trade.total_spread_cost for trade in trades)
    total_slippage_cost = sum(trade.total_slippage_cost for trade in trades)
    total_impact_cost = sum(trade.total_impact_cost for trade in trades)
    total_turnover = sum(trade.entry_notional + trade.exit_notional for trade in trades)
    turnover_to_initial_equity = total_turnover / initial_equity

    exposure_values = [point.exposure_pct for point in equity_curve]
    average_exposure_pct = mean(exposure_values) if exposure_values else 0.0
    max_exposure_pct = max(exposure_values) if exposure_values else 0.0

    benchmark_return_pct = None
    if benchmark_return is not None:
        if not isfinite(float(benchmark_return)):
            raise BacktestMetricsError("benchmark_return must be finite")
        benchmark_return_pct = benchmark_return * 100.0

    return PerformanceMetrics(
        initial_equity=float(initial_equity),
        final_equity=float(final),
        total_return_pct=total_return * 100.0,
        annualized_return_pct=annualized_return * 100.0,
        annualized_volatility_pct=annualized_volatility * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_drawdown * 100.0,
        max_drawdown_duration_hours=max_duration_seconds / 3600.0,
        calmar_ratio=calmar,
        trade_count=trade_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate * 100.0,
        profit_factor=profit_factor,
        average_trade_pnl=average_trade_pnl,
        median_trade_pnl=median_trade_pnl,
        best_trade_pnl=best_trade_pnl,
        worst_trade_pnl=worst_trade_pnl,
        average_holding_hours=average_holding_hours,
        total_realized_pnl=total_realized_pnl,
        total_fees=total_fees,
        total_spread_cost=total_spread_cost,
        total_slippage_cost=total_slippage_cost,
        total_impact_cost=total_impact_cost,
        total_turnover=total_turnover,
        turnover_to_initial_equity=turnover_to_initial_equity,
        average_exposure_pct=average_exposure_pct * 100.0,
        max_exposure_pct=max_exposure_pct * 100.0,
        benchmark_return_pct=benchmark_return_pct,
        bars_processed=len(equity_curve),
        rejected_actions=rejected_actions,
    )
