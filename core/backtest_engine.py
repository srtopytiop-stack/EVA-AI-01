"""
EVA-AI-01 - Historical Backtesting Engine
==========================================
Phase 11: event-driven historical backtest.

Critical anti-leakage rule:
A decision made at bar t close can execute no earlier than bar t+1 open.
The execution model receives only information known before that open.

Scope: long Spot only; no leverage, margin, shorting, exchange orders, or
future-information access.

Diagnostics
-----------
The engine now exposes an immutable, diagnostics-only evidence bundle with
signal, regime, risk, paper-trading, and equity-variation counters.

These diagnostics do NOT alter signal thresholds, risk policy, execution,
portfolio accounting, or trade decisions. Their purpose is to identify where
a zero-return-variance backtest is originating before strategy parameters are
changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Sequence

from core.backtest_metrics import (
    BacktestMetricsError,
    BacktestTrade,
    EquityPoint,
    PerformanceMetrics,
    calculate_metrics,
)
from core.execution_simulator import ExecutionConfig, ExecutionSimulator
from core.paper_trading_engine import (
    PaperAction,
    PaperTradeEvent,
    PaperTradingConfig,
    PaperTradingEngine,
    PaperStepResult,
)
from core.portfolio_engine import PortfolioConfig, PortfolioEngine
from core.regime_engine import MarketRegime, RegimeEngine
from core.risk_engine import RiskConfig, RiskDecision, RiskEngine, RiskResult
from core.signal_engine import SignalResult, evaluate as evaluate_signal
from core.volatility_engine import VolatilityEngine, VolatilityResult
from data.market_data import Candle

EPS = 1e-12


class BacktestError(ValueError):
    """Raised when backtest inputs or invariants are invalid."""


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable research configuration for Phase 11."""

    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    paper_config: PaperTradingConfig = field(default_factory=PaperTradingConfig)

    warmup_bars: int = 120
    history_window_bars: int = 240
    require_regular_intervals: bool = True
    expected_interval_ms: int | None = None

    signal_z_window: int = 40
    signal_trend_window: int = 24
    signal_volume_window: int = 20
    signal_ewma_decay: float = 0.94
    signal_entry_threshold: float = 0.55
    signal_neutral_threshold: float = 0.25

    stop_volatility_multiplier: float = 2.5
    minimum_stop_pct: float = 0.01
    maximum_stop_pct: float = 0.20
    risk_volatility_ratio_cap: float = 10.0
    liquidity_proxy_quote_volume_multiplier: float = 1.0

    regime_factor_trend_up: float = 1.00
    regime_factor_trend_down: float = 0.25
    regime_factor_range: float = 0.70
    regime_factor_high_volatility: float = 0.40
    regime_factor_low_volatility: float = 0.85
    regime_factor_transition: float = 0.60

    risk_free_rate_annual: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("warmup_bars", self.warmup_bars),
            ("history_window_bars", self.history_window_bars),
            ("signal_z_window", self.signal_z_window),
            ("signal_trend_window", self.signal_trend_window),
            ("signal_volume_window", self.signal_volume_window),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise BacktestError(f"{name} must be an integer >= 2")

        if self.history_window_bars < self.warmup_bars:
            raise BacktestError("history_window_bars must be >= warmup_bars")

        minimum_required = max(
            self.signal_z_window,
            self.signal_trend_window,
            self.signal_volume_window,
            41,
            61,
            8,
        )
        if self.warmup_bars < minimum_required:
            raise BacktestError("warmup_bars is too small for the current engines")

        if self.expected_interval_ms is not None:
            if (
                isinstance(self.expected_interval_ms, bool)
                or not isinstance(self.expected_interval_ms, int)
                or self.expected_interval_ms <= 0
            ):
                raise BacktestError(
                    "expected_interval_ms must be a positive integer"
                )

        if not 0.0 < self.signal_entry_threshold <= 1.0:
            raise BacktestError("signal_entry_threshold must be in (0, 1]")

        if not 0.0 <= self.signal_neutral_threshold < self.signal_entry_threshold:
            raise BacktestError(
                "signal_neutral_threshold must be in [0, entry_threshold)"
            )

        if not 0.0 < self.signal_ewma_decay < 1.0:
            raise BacktestError("signal_ewma_decay must be in (0, 1)")

        finite_fields = (
            ("stop_volatility_multiplier", self.stop_volatility_multiplier),
            ("minimum_stop_pct", self.minimum_stop_pct),
            ("maximum_stop_pct", self.maximum_stop_pct),
            ("risk_volatility_ratio_cap", self.risk_volatility_ratio_cap),
            (
                "liquidity_proxy_quote_volume_multiplier",
                self.liquidity_proxy_quote_volume_multiplier,
            ),
            ("risk_free_rate_annual", self.risk_free_rate_annual),
            ("regime_factor_trend_up", self.regime_factor_trend_up),
            ("regime_factor_trend_down", self.regime_factor_trend_down),
            ("regime_factor_range", self.regime_factor_range),
            ("regime_factor_high_volatility", self.regime_factor_high_volatility),
            ("regime_factor_low_volatility", self.regime_factor_low_volatility),
            ("regime_factor_transition", self.regime_factor_transition),
        )

        for name, value in finite_fields:
            if isinstance(value, bool) or not isfinite(float(value)):
                raise BacktestError(f"{name} must be finite")

        if self.stop_volatility_multiplier <= 0.0:
            raise BacktestError(
                "stop_volatility_multiplier must be positive"
            )

        if not 0.0 < self.minimum_stop_pct <= self.maximum_stop_pct:
            raise BacktestError(
                "stop percentages must satisfy 0 < minimum <= maximum"
            )

        if (
            self.maximum_stop_pct
            > self.risk_config.maximum_stop_distance_pct + EPS
        ):
            raise BacktestError(
                "maximum_stop_pct cannot exceed "
                "RiskConfig.maximum_stop_distance_pct"
            )

        if self.risk_volatility_ratio_cap <= 0.0:
            raise BacktestError(
                "risk_volatility_ratio_cap must be positive"
            )

        if self.liquidity_proxy_quote_volume_multiplier <= 0.0:
            raise BacktestError(
                "liquidity_proxy_quote_volume_multiplier must be positive"
            )

        if self.risk_free_rate_annual <= -1.0:
            raise BacktestError(
                "risk_free_rate_annual must be > -1"
            )

        for name, value in (
            ("regime_factor_trend_up", self.regime_factor_trend_up),
            ("regime_factor_trend_down", self.regime_factor_trend_down),
            ("regime_factor_range", self.regime_factor_range),
            (
                "regime_factor_high_volatility",
                self.regime_factor_high_volatility,
            ),
            ("regime_factor_low_volatility", self.regime_factor_low_volatility),
            ("regime_factor_transition", self.regime_factor_transition),
        ):
            if not 0.0 <= value <= 1.0:
                raise BacktestError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class BacktestDiagnostics:
    """Immutable diagnostics for locating the source of a flat backtest."""

    candles_processed: int
    decision_evaluations: int

    signal_buy_count: int
    signal_neutral_count: int
    signal_sell_count: int

    signal_score_min: float
    signal_score_mean: float
    signal_score_max: float

    signal_confidence_min: float
    signal_confidence_mean: float
    signal_confidence_max: float

    regime_counts: tuple[tuple[str, int], ...]

    risk_approved_count: int
    risk_reduced_count: int
    risk_rejected_count: int

    paper_wait_count: int
    paper_hold_count: int
    paper_open_count: int
    paper_close_count: int
    paper_rejected_count: int

    paper_reason_counts: tuple[tuple[str, int], ...]

    completed_trades: int
    initial_equity: float
    final_equity: float
    equity_return_variance: float
    nonzero_equity_returns: int

    @classmethod
    def empty(cls) -> "BacktestDiagnostics":
        """Return a valid zero-valued diagnostics object."""
        return cls(
            candles_processed=0,
            decision_evaluations=0,
            signal_buy_count=0,
            signal_neutral_count=0,
            signal_sell_count=0,
            signal_score_min=0.0,
            signal_score_mean=0.0,
            signal_score_max=0.0,
            signal_confidence_min=0.0,
            signal_confidence_mean=0.0,
            signal_confidence_max=0.0,
            regime_counts=(),
            risk_approved_count=0,
            risk_reduced_count=0,
            risk_rejected_count=0,
            paper_wait_count=0,
            paper_hold_count=0,
            paper_open_count=0,
            paper_close_count=0,
            paper_rejected_count=0,
            paper_reason_counts=(),
            completed_trades=0,
            initial_equity=0.0,
            final_equity=0.0,
            equity_return_variance=0.0,
            nonzero_equity_returns=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candles_processed": self.candles_processed,
            "decision_evaluations": self.decision_evaluations,
            "signal_buy_count": self.signal_buy_count,
            "signal_neutral_count": self.signal_neutral_count,
            "signal_sell_count": self.signal_sell_count,
            "signal_score_min": self.signal_score_min,
            "signal_score_mean": self.signal_score_mean,
            "signal_score_max": self.signal_score_max,
            "signal_confidence_min": self.signal_confidence_min,
            "signal_confidence_mean": self.signal_confidence_mean,
            "signal_confidence_max": self.signal_confidence_max,
            "regime_counts": {
                key: value
                for key, value in self.regime_counts
            },
            "risk_approved_count": self.risk_approved_count,
            "risk_reduced_count": self.risk_reduced_count,
            "risk_rejected_count": self.risk_rejected_count,
            "paper_wait_count": self.paper_wait_count,
            "paper_hold_count": self.paper_hold_count,
            "paper_open_count": self.paper_open_count,
            "paper_close_count": self.paper_close_count,
            "paper_rejected_count": self.paper_rejected_count,
            "paper_reason_counts": {
                key: value
                for key, value in self.paper_reason_counts
            },
            "completed_trades": self.completed_trades,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "equity_return_variance": self.equity_return_variance,
            "nonzero_equity_returns": self.nonzero_equity_returns,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Immutable complete output of a historical backtest."""

    symbol: str
    bars_processed: int
    warmup_bars: int
    rejected_actions: int
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[BacktestTrade, ...]
    events: tuple[PaperTradeEvent, ...]
    metrics: PerformanceMetrics
    final_position_open: bool

    diagnostics: BacktestDiagnostics = field(
        default_factory=BacktestDiagnostics.empty,
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "bars_processed": self.bars_processed,
            "warmup_bars": self.warmup_bars,
            "rejected_actions": self.rejected_actions,
            "equity_curve": [
                point.to_dict()
                for point in self.equity_curve
            ],
            "trades": [
                trade.to_dict()
                for trade in self.trades
            ],
            "events": [
                event.to_dict()
                for event in self.events
            ],
            "metrics": self.metrics.to_dict(),
            "final_position_open": self.final_position_open,
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class _PendingDecision:
    signal: SignalResult
    risk: RiskResult
    volatility: float
    liquidity_notional: float | None


class BacktestEngine:
    """Event-driven single-symbol Spot backtesting orchestrator."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.regime_engine = RegimeEngine()
        self.volatility_engine = VolatilityEngine()
        self.risk_engine = RiskEngine()
        self._symbol = ""

    def _new_paper_engine(self) -> PaperTradingEngine:
        portfolio = PortfolioEngine(self.config.portfolio_config)
        execution = ExecutionSimulator(self.config.execution_config)

        return PaperTradingEngine(
            portfolio=portfolio,
            execution=execution,
            config=self.config.paper_config,
        )

    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        if not isinstance(symbol, str) or not symbol.strip():
            raise BacktestError(
                "symbol must be a non-empty string"
            )

        normalized = symbol.strip().upper()

        if not normalized.isalnum():
            raise BacktestError(
                "symbol contains invalid characters"
            )

        return normalized

    @staticmethod
    def _validate_candles(
        candles: Sequence[Candle],
        *,
        config: BacktestConfig,
    ) -> None:
        if not candles:
            raise BacktestError(
                "candles cannot be empty"
            )

        previous_open: int | None = None
        inferred_interval: int | None = None

        for index, candle in enumerate(candles):
            if not candle.is_valid_ohlc:
                raise BacktestError(
                    f"invalid OHLC at candle index {index}"
                )

            if candle.close_time <= candle.open_time:
                raise BacktestError(
                    f"invalid candle timestamps at index {index}"
                )

            if previous_open is not None:
                delta = candle.open_time - previous_open

                if delta <= 0:
                    raise BacktestError(
                        "candles must be strictly chronological"
                    )

                if inferred_interval is None:
                    inferred_interval = delta

                expected = (
                    config.expected_interval_ms
                    or inferred_interval
                )

                if (
                    config.require_regular_intervals
                    and delta != expected
                ):
                    raise BacktestError(
                        "irregular or missing candle interval detected "
                        f"at index {index}: expected {expected}, got {delta}"
                    )

            previous_open = candle.open_time

        if len(candles) < config.warmup_bars + 1:
            raise BacktestError(
                "not enough candles: warmup_bars + 1 required"
            )

    def _history_window(
        self,
        candles: Sequence[Candle],
        index: int,
    ) -> tuple[list[float], list[float]]:
        start = max(
            0,
            index + 1 - self.config.history_window_bars,
        )

        return (
            [
                float(candle.close)
                for candle in candles[start:index + 1]
            ],
            [
                float(candle.volume)
                for candle in candles[start:index + 1]
            ],
        )

    def _regime_factor(
        self,
        regime: MarketRegime,
    ) -> float:
        return {
            MarketRegime.TREND_UP:
                self.config.regime_factor_trend_up,
            MarketRegime.TREND_DOWN:
                self.config.regime_factor_trend_down,
            MarketRegime.RANGE:
                self.config.regime_factor_range,
            MarketRegime.HIGH_VOLATILITY:
                self.config.regime_factor_high_volatility,
            MarketRegime.LOW_VOLATILITY:
                self.config.regime_factor_low_volatility,
            MarketRegime.TRANSITION:
                self.config.regime_factor_transition,
        }[regime]

    def _safe_volatility_ratio(
        self,
        volatility: VolatilityResult,
    ) -> float:
        ratio = float(
            volatility.volatility_ratio
        )

        if not isfinite(ratio):
            return self.config.risk_volatility_ratio_cap

        return max(
            EPS,
            min(
                ratio,
                self.config.risk_volatility_ratio_cap,
            ),
        )

    def _stop_distance(
        self,
        price: float,
        volatility: VolatilityResult,
    ) -> float:
        stop_pct = (
            self.config.stop_volatility_multiplier
            * max(
                0.0,
                volatility.ewma_volatility,
            )
        )

        stop_pct = max(
            self.config.minimum_stop_pct,
            min(
                stop_pct,
                self.config.maximum_stop_pct,
            ),
        )

        return price * stop_pct

    def _liquidity_proxy(
        self,
        candle: Candle,
    ) -> float | None:
        quote_volume = float(
            candle.quote_volume
        )

        if (
            not isfinite(quote_volume)
            or quote_volume <= 0.0
        ):
            return None

        return (
            quote_volume
            * self.config.liquidity_proxy_quote_volume_multiplier
        )

    def _decision_at_close(
        self,
        *,
        candle: Candle,
        closes: Sequence[float],
        volumes: Sequence[float],
        equity: float,
    ) -> tuple[
        SignalResult,
        RiskResult,
        VolatilityResult,
    ]:
        # Keep this public-to-tests contract unchanged.
        # Existing Phase 11 causality tests monkeypatch
        # this method directly.

        regime = self.regime_engine.classify(
            closes
        )

        volatility = self.volatility_engine.classify(
            closes
        )

        signal = evaluate_signal(
            symbol=self._symbol,
            timestamp=int(
                candle.close_time
            ),
            closes=closes,
            volumes=volumes,
            z_window=self.config.signal_z_window,
            trend_window=self.config.signal_trend_window,
            volume_window=self.config.signal_volume_window,
            ewma_decay=self.config.signal_ewma_decay,
            entry_threshold=self.config.signal_entry_threshold,
            neutral_threshold=self.config.signal_neutral_threshold,
            external_regime_factor=self._regime_factor(
                regime.regime
            ),
        )

        risk = self.risk_engine.evaluate(
            equity=equity,
            entry_price=float(
                candle.close
            ),
            stop_distance=self._stop_distance(
                float(candle.close),
                volatility,
            ),
            volatility_ratio=self._safe_volatility_ratio(
                volatility
            ),
            config=self.config.risk_config,
        )

        return signal, risk, volatility

    @staticmethod
    def _drawdown(
        equity: float,
        peak_equity: float,
    ) -> float:
        return (
            0.0
            if peak_equity <= EPS
            else equity / peak_equity - 1.0
        )

    @staticmethod
    def _reconstruct_trades(
        events: Sequence[PaperTradeEvent],
    ) -> tuple[BacktestTrade, ...]:
        completed: list[BacktestTrade] = []
        open_event: PaperTradeEvent | None = None

        for event in events:
            if event.action == PaperAction.OPEN:
                if open_event is not None:
                    raise BacktestError(
                        "paper event invariant violated: "
                        "multiple open trades"
                    )

                open_event = event

            elif event.action == PaperAction.CLOSE:
                if open_event is None:
                    raise BacktestError(
                        "paper event invariant violated: "
                        "close without open"
                    )

                if open_event.gross_notional <= EPS:
                    raise BacktestError(
                        "entry notional is non-positive"
                    )

                net_pnl = float(
                    event.realized_pnl
                )

                completed.append(
                    BacktestTrade(
                        trade_id=len(completed) + 1,
                        symbol=open_event.symbol,
                        entry_event_id=open_event.event_id,
                        exit_event_id=event.event_id,
                        entry_timestamp=open_event.timestamp,
                        exit_timestamp=event.timestamp,
                        entry_price=open_event.execution_price,
                        exit_price=event.execution_price,
                        quantity=event.filled_quantity,
                        entry_notional=open_event.gross_notional,
                        exit_notional=event.gross_notional,
                        total_fees=(
                            open_event.fee
                            + event.fee
                        ),
                        total_spread_cost=(
                            open_event.spread_cost
                            + event.spread_cost
                        ),
                        total_slippage_cost=(
                            open_event.slippage_cost
                            + event.slippage_cost
                        ),
                        total_impact_cost=(
                            open_event.impact_cost
                            + event.impact_cost
                        ),
                        net_pnl=net_pnl,
                        return_pct=(
                            net_pnl
                            / open_event.gross_notional
                            * 100.0
                        ),
                        holding_seconds=(
                            event.timestamp
                            - open_event.timestamp
                        ) / 1000.0,
                        entry_signal_score=(
                            open_event.signal_score
                        ),
                        exit_signal_score=(
                            event.signal_score
                        ),
                    )
                )

                open_event = None

        return tuple(completed)

    @staticmethod
    def _summary(
        values: Sequence[float],
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """Return min/mean/max for a finite numeric series."""

        if not values:
            return 0.0, 0.0, 0.0

        normalized = [
            float(value)
            for value in values
        ]

        if any(
            not isfinite(value)
            for value in normalized
        ):
            raise BacktestError(
                "diagnostic series contains a non-finite value"
            )

        return (
            min(normalized),
            sum(normalized) / len(normalized),
            max(normalized),
        )

    @staticmethod
    def _sample_variance(
        values: Sequence[float],
    ) -> float:
        """Return sample variance; zero for fewer than two observations."""

        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)

        return sum(
            (value - mean) ** 2
            for value in values
        ) / (len(values) - 1)

    @classmethod
    def _build_diagnostics(
        cls,
        *,
        candles_processed: int,
        signal_scores: Sequence[float],
        signal_confidences: Sequence[float],
        signal_counts: dict[int, int],
        regime_counts: dict[str, int],
        risk_counts: dict[str, int],
        paper_action_counts: dict[str, int],
        paper_reason_counts: dict[str, int],
        trades: Sequence[BacktestTrade],
        equity_curve: Sequence[EquityPoint],
    ) -> BacktestDiagnostics:
        (
            score_min,
            score_mean,
            score_max,
        ) = cls._summary(
            signal_scores
        )

        (
            confidence_min,
            confidence_mean,
            confidence_max,
        ) = cls._summary(
            signal_confidences
        )

        equity_returns = [
            current.equity / previous.equity - 1.0
            for previous, current in zip(
                equity_curve[:-1],
                equity_curve[1:],
            )
        ]

        if any(
            not isfinite(value)
            for value in equity_returns
        ):
            raise BacktestError(
                "equity return series contains a non-finite value"
            )

        return BacktestDiagnostics(
            candles_processed=candles_processed,
            decision_evaluations=len(
                signal_scores
            ),
            signal_buy_count=signal_counts.get(
                1,
                0,
            ),
            signal_neutral_count=signal_counts.get(
                0,
                0,
            ),
            signal_sell_count=signal_counts.get(
                -1,
                0,
            ),
            signal_score_min=score_min,
            signal_score_mean=score_mean,
            signal_score_max=score_max,
            signal_confidence_min=confidence_min,
            signal_confidence_mean=confidence_mean,
            signal_confidence_max=confidence_max,
            regime_counts=tuple(
                sorted(
                    regime_counts.items()
                )
            ),
            risk_approved_count=risk_counts.get(
                RiskDecision.APPROVED.value,
                0,
            ),
            risk_reduced_count=risk_counts.get(
                RiskDecision.REDUCED.value,
                0,
            ),
            risk_rejected_count=risk_counts.get(
                RiskDecision.REJECTED.value,
                0,
            ),
            paper_wait_count=paper_action_counts.get(
                PaperAction.WAIT.value,
                0,
            ),
            paper_hold_count=paper_action_counts.get(
                PaperAction.HOLD.value,
                0,
            ),
            paper_open_count=paper_action_counts.get(
                PaperAction.OPEN.value,
                0,
            ),
            paper_close_count=paper_action_counts.get(
                PaperAction.CLOSE.value,
                0,
            ),
            paper_rejected_count=paper_action_counts.get(
                PaperAction.REJECTED.value,
                0,
            ),
            paper_reason_counts=tuple(
                sorted(
                    paper_reason_counts.items()
                )
            ),
            completed_trades=len(
                trades
            ),
            initial_equity=(
                float(
                    equity_curve[0].equity
                )
                if equity_curve
                else 0.0
            ),
            final_equity=(
                float(
                    equity_curve[-1].equity
                )
                if equity_curve
                else 0.0
            ),
            equity_return_variance=cls._sample_variance(
                equity_returns
            ),
            nonzero_equity_returns=sum(
                abs(value) > EPS
                for value in equity_returns
            ),
        )

    @staticmethod
    def _record_paper_step(
        step: PaperStepResult,
        *,
        action_counts: dict[str, int],
        reason_counts: dict[str, int],
    ) -> None:
        action_key = step.action.value

        action_counts[action_key] = (
            action_counts.get(
                action_key,
                0,
            )
            + 1
        )

        reason = step.reason.strip()

        if reason:
            reason_counts[reason] = (
                reason_counts.get(
                    reason,
                    0,
                )
                + 1
            )

    def run(
        self,
        candles: Sequence[Candle],
        *,
        symbol: str,
    ) -> BacktestResult:
        """Run the historical backtest deterministically."""

        self._symbol = self._validate_symbol(
            symbol
        )

        self._validate_candles(
            candles,
            config=self.config,
        )

        paper = self._new_paper_engine()

        equity_curve: list[EquityPoint] = []
        pending: _PendingDecision | None = None

        peak_equity = (
            self.config
            .portfolio_config
            .initial_cash
        )

        signal_scores: list[float] = []
        signal_confidences: list[float] = []

        signal_counts: dict[int, int] = {
            1: 0,
            0: 0,
            -1: 0,
        }

        regime_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        paper_action_counts: dict[str, int] = {}
        paper_reason_counts: dict[str, int] = {}

        for index, candle in enumerate(candles):

            if pending is not None:
                step = paper.step(
                    timestamp=int(
                        candle.open_time
                    ),
                    symbol=self._symbol,
                    reference_price=float(
                        candle.open
                    ),
                    signal=pending.signal,
                    risk=pending.risk,
                    volatility=pending.volatility,
                    liquidity_notional=(
                        pending.liquidity_notional
                    ),
                )

                self._record_paper_step(
                    step,
                    action_counts=paper_action_counts,
                    reason_counts=paper_reason_counts,
                )

                pending = None

            if (
                paper.portfolio.position(
                    self._symbol
                )
                is not None
            ):
                snapshot = paper.mark_price(
                    self._symbol,
                    float(candle.close),
                )
            else:
                snapshot = paper.snapshot()

            if (
                index == len(candles) - 1
                and index + 1
                >= self.config.warmup_bars
            ):
                closes, volumes = (
                    self._history_window(
                        candles,
                        index,
                    )
                )

                (
                    signal,
                    risk,
                    volatility,
                ) = self._decision_at_close(
                    candle=candle,
                    closes=closes,
                    volumes=volumes,
                    equity=snapshot.equity,
                )

                final_regime = (
                    self.regime_engine
                    .classify(closes)
                    .regime
                )

                signal_scores.append(
                    float(signal.score)
                )

                signal_confidences.append(
                    float(signal.confidence)
                )

                signal_counts[
                    int(signal.direction)
                ] = (
                    signal_counts.get(
                        int(signal.direction),
                        0,
                    )
                    + 1
                )

                regime_key = final_regime.value

                regime_counts[
                    regime_key
                ] = (
                    regime_counts.get(
                        regime_key,
                        0,
                    )
                    + 1
                )

                risk_key = risk.decision.value

                risk_counts[
                    risk_key
                ] = (
                    risk_counts.get(
                        risk_key,
                        0,
                    )
                    + 1
                )

                if (
                    paper.portfolio.position(
                        self._symbol
                    )
                    is not None
                ):
                    step = paper.close(
                        timestamp=int(
                            candle.close_time
                        ),
                        symbol=self._symbol,
                        reference_price=float(
                            candle.close
                        ),
                        signal=signal,
                        risk=risk,
                        volatility=(
                            volatility.ewma_volatility
                        ),
                        liquidity_notional=(
                            self._liquidity_proxy(
                                candle
                            )
                        ),
                    )

                    self._record_paper_step(
                        step,
                        action_counts=paper_action_counts,
                        reason_counts=paper_reason_counts,
                    )

                snapshot = paper.snapshot()

            peak_equity = max(
                peak_equity,
                snapshot.equity,
            )

            equity_curve.append(
                EquityPoint(
                    timestamp=int(
                        candle.close_time
                    ),
                    equity=float(
                        snapshot.equity
                    ),
                    cash=float(
                        snapshot.cash
                    ),
                    gross_exposure=float(
                        snapshot.gross_exposure
                    ),
                    exposure_pct=float(
                        snapshot.exposure_pct
                    ),
                    drawdown_pct=self._drawdown(
                        snapshot.equity,
                        peak_equity,
                    ),
                )
            )

            if (
                index < len(candles) - 1
                and index + 1
                >= self.config.warmup_bars
            ):
                closes, volumes = (
                    self._history_window(
                        candles,
                        index,
                    )
                )

                (
                    signal,
                    risk,
                    volatility,
                ) = self._decision_at_close(
                    candle=candle,
                    closes=closes,
                    volumes=volumes,
                    equity=snapshot.equity,
                )

                regime = (
                    self.regime_engine
                    .classify(closes)
                    .regime
                )

                signal_scores.append(
                    float(signal.score)
                )

                signal_confidences.append(
                    float(signal.confidence)
                )

                signal_counts[
                    int(signal.direction)
                ] = (
                    signal_counts.get(
                        int(signal.direction),
                        0,
                    )
                    + 1
                )

                regime_key = regime.value

                regime_counts[
                    regime_key
                ] = (
                    regime_counts.get(
                        regime_key,
                        0,
                    )
                    + 1
                )

                risk_key = risk.decision.value

                risk_counts[
                    risk_key
                ] = (
                    risk_counts.get(
                        risk_key,
                        0,
                    )
                    + 1
                )

                pending = _PendingDecision(
                    signal=signal,
                    risk=risk,
                    volatility=(
                        volatility.ewma_volatility
                    ),
                    liquidity_notional=(
                        self._liquidity_proxy(
                            candle
                        )
                    ),
                )

        events = tuple(
            paper.events
        )

        trades = self._reconstruct_trades(
            events
        )

        if len(equity_curve) < 2:
            raise BacktestError(
                "backtest produced fewer than two equity observations"
            )

        rejected_actions = paper_action_counts.get(
            PaperAction.REJECTED.value,
            0,
        )

        benchmark_return = (
            float(candles[-1].close)
            / float(candles[0].close)
            - 1.0
        )

        try:
            metrics = calculate_metrics(
                equity_curve=equity_curve,
                trades=trades,
                initial_equity=(
                    self.config
                    .portfolio_config
                    .initial_cash
                ),
                benchmark_return=benchmark_return,
                rejected_actions=rejected_actions,
                risk_free_rate_annual=(
                    self.config
                    .risk_free_rate_annual
                ),
            )

        except BacktestMetricsError as exc:
            raise BacktestError(
                "performance-metric calculation failed: "
                f"{exc}"
            ) from exc

        diagnostics = self._build_diagnostics(
            candles_processed=len(candles),
            signal_scores=signal_scores,
            signal_confidences=signal_confidences,
            signal_counts=signal_counts,
            regime_counts=regime_counts,
            risk_counts=risk_counts,
            paper_action_counts=paper_action_counts,
            paper_reason_counts=paper_reason_counts,
            trades=trades,
            equity_curve=equity_curve,
        )

        return BacktestResult(
            symbol=self._symbol,
            bars_processed=len(candles),
            warmup_bars=self.config.warmup_bars,
            rejected_actions=rejected_actions,
            equity_curve=tuple(
                equity_curve
            ),
            trades=trades,
            events=events,
            metrics=metrics,
            final_position_open=(
                paper.portfolio.position(
                    self._symbol
                )
                is not None
            ),
            diagnostics=diagnostics,
        )


if __name__ == "__main__":
    print(
        "BACKTEST_ENGINE_READY"
    )
