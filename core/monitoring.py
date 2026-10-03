"""
EVA-AI-01 - Monitoring & Diagnostics
====================================
Phase 15: deterministic observability around the Phase-14 research result.

This module is observational only. It never places exchange orders, loads
Binance credentials, sends Telegram messages, or changes a trading decision.

The monitor checks:
- integration metadata;
- equity-curve integrity and chronological order;
- final-equity consistency;
- backtest performance invariants;
- statistical-validation outputs;
- moving-block bootstrap metadata and probability bounds;
- trade/event chronology when the required fields exist;
- NaN/Inf contamination in auditable numeric outputs.

The implementation is intentionally duck-typed to avoid coupling the stable
engines to this observability layer and to prevent import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence


class MonitoringStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class MonitoringError(RuntimeError):
    """Raised when strict monitoring rejects an observed state."""


@dataclass(frozen=True)
class MonitoringConfig:
    """Immutable policy for Phase-15 monitoring."""

    min_equity_points: int = 21
    min_bootstrap_observations: int = 20
    min_bootstrap_replications: int = 200
    min_block_length: int = 2
    require_bootstrap_observation_consistency: bool = True
    audit_trade_chronology: bool = True
    audit_event_chronology: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("min_equity_points", self.min_equity_points),
            ("min_bootstrap_observations", self.min_bootstrap_observations),
            ("min_bootstrap_replications", self.min_bootstrap_replications),
            ("min_block_length", self.min_block_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.min_equity_points < 2:
            raise ValueError("min_equity_points must be >= 2")
        if self.min_bootstrap_observations < 2:
            raise ValueError("min_bootstrap_observations must be >= 2")
        if self.min_bootstrap_replications < 1:
            raise ValueError("min_bootstrap_replications must be >= 1")
        if self.min_block_length < 2:
            raise ValueError("min_block_length must be >= 2")


@dataclass(frozen=True)
class MonitoringIssue:
    code: str
    severity: MonitoringStatus
    message: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class MonitoringReport:
    status: MonitoringStatus
    checks_run: int
    checks_passed: int
    issues: tuple[MonitoringIssue, ...]
    metrics: Mapping[str, Any]

    @property
    def healthy(self) -> bool:
        return self.status is MonitoringStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }

    def raise_if_unhealthy(self) -> None:
        if not self.healthy:
            detail = "; ".join(
                f"{issue.code}: {issue.message}" for issue in self.issues
            ) or "monitoring state is not healthy"
            raise MonitoringError(detail)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(float(value))
    )


def _walk_values(obj: Any, path: str = "root", seen: set[int] | None = None):
    """Yield ``(path, value)`` pairs for scalar numeric values recursively."""
    if seen is None:
        seen = set()

    if obj is None or isinstance(obj, (str, bytes, bool)):
        return
    if isinstance(obj, Real):
        yield path, obj
        return

    object_id = id(obj)
    if object_id in seen:
        return
    seen.add(object_id)

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _walk_values(value, f"{path}.{key}", seen)
        return

    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _walk_values(value, f"{path}[{index}]", seen)
        return

    if is_dataclass(obj):
        for field in fields(obj):
            yield from _walk_values(
                getattr(obj, field.name),
                f"{path}.{field.name}",
                seen,
            )
        return

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            yield from _walk_values(to_dict(), path, seen)
            return
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        yield from _walk_values(vars(obj), path, seen)


def _issue(code: str, severity: MonitoringStatus, message: str, **evidence: Any) -> MonitoringIssue:
    return MonitoringIssue(code, severity, message, evidence)


def _audit_metadata(result: Any) -> tuple[list[MonitoringIssue], dict[str, Any]]:
    issues: list[MonitoringIssue] = []
    metrics: dict[str, Any] = {}

    symbol = _get(result, "symbol")
    interval = _get(result, "interval")
    candle_count = _get(result, "candle_count")
    feature_rows = _get(result, "feature_rows")
    feature_columns = _get(result, "feature_columns")

    if not isinstance(symbol, str) or not symbol.strip():
        issues.append(_issue("METADATA_SYMBOL_INVALID", MonitoringStatus.CRITICAL, "symbol is missing or empty"))
    else:
        metrics["symbol"] = symbol.strip().upper()

    if not isinstance(interval, str) or not interval.strip():
        issues.append(_issue("METADATA_INTERVAL_INVALID", MonitoringStatus.CRITICAL, "interval is missing or empty"))
    else:
        metrics["interval"] = interval.strip()

    if isinstance(candle_count, bool) or not isinstance(candle_count, int) or candle_count < 2:
        issues.append(_issue(
            "METADATA_CANDLE_COUNT_INVALID",
            MonitoringStatus.CRITICAL,
            "candle_count must be an integer >= 2",
            candle_count=candle_count,
        ))
    else:
        metrics["candle_count"] = candle_count

    if isinstance(feature_rows, bool) or not isinstance(feature_rows, int) or feature_rows < 1:
        issues.append(_issue(
            "METADATA_FEATURE_ROWS_INVALID",
            MonitoringStatus.CRITICAL,
            "feature_rows must be an integer >= 1",
            feature_rows=feature_rows,
        ))
    else:
        metrics["feature_rows"] = feature_rows

    if not isinstance(feature_columns, Sequence) or isinstance(feature_columns, (str, bytes)):
        issues.append(_issue(
            "METADATA_FEATURE_COLUMNS_INVALID",
            MonitoringStatus.CRITICAL,
            "feature_columns must be a non-string sequence",
        ))
    elif not feature_columns:
        issues.append(_issue(
            "METADATA_FEATURE_COLUMNS_EMPTY",
            MonitoringStatus.CRITICAL,
            "feature_columns is empty",
        ))
    else:
        metrics["feature_column_count"] = len(feature_columns)

    return issues, metrics


def _audit_equity_curve(result: Any, cfg: MonitoringConfig) -> tuple[list[MonitoringIssue], dict[str, Any]]:
    issues: list[MonitoringIssue] = []
    metrics: dict[str, Any] = {}
    backtest = _get(result, "backtest")
    curve = _get(backtest, "equity_curve")

    if curve is None:
        return [
            _issue("EQUITY_CURVE_MISSING", MonitoringStatus.CRITICAL, "backtest.equity_curve is missing")
        ], metrics

    points = tuple(curve)
    metrics["equity_points"] = len(points)

    if len(points) < cfg.min_equity_points:
        issues.append(_issue(
            "EQUITY_CURVE_TOO_SHORT",
            MonitoringStatus.CRITICAL,
            "equity curve is shorter than the monitoring minimum",
            observations=len(points),
            minimum=cfg.min_equity_points,
        ))

    timestamps: list[int] = []
    equity_values: list[float] = []

    for index, point in enumerate(points):
        timestamp = _get(point, "timestamp")
        equity = _get(point, "equity")

        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            issues.append(_issue(
                "EQUITY_TIMESTAMP_INVALID",
                MonitoringStatus.CRITICAL,
                "equity timestamp must be a positive integer",
                index=index,
                timestamp=timestamp,
            ))
        else:
            timestamps.append(timestamp)

        if not _is_finite_real(equity):
            issues.append(_issue(
                "EQUITY_VALUE_NONFINITE",
                MonitoringStatus.CRITICAL,
                "equity contains NaN, Inf, or a non-numeric value",
                index=index,
                equity=equity,
            ))
        elif float(equity) <= 0.0:
            issues.append(_issue(
                "EQUITY_VALUE_NONPOSITIVE",
                MonitoringStatus.CRITICAL,
                "equity must remain strictly positive",
                index=index,
                equity=float(equity),
            ))
        else:
            equity_values.append(float(equity))

    if len(timestamps) >= 2 and any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        issues.append(_issue(
            "EQUITY_TIMESTAMPS_NOT_STRICT",
            MonitoringStatus.CRITICAL,
            "equity timestamps must be strictly increasing",
        ))

    reported_final = _get(result, "final_equity")
    if reported_final is None:
        reported_final = _get(_get(backtest, "metrics"), "final_equity")

    if equity_values:
        curve_final = equity_values[-1]
        metrics["curve_final_equity"] = curve_final
        if not _is_finite_real(reported_final):
            issues.append(_issue(
                "FINAL_EQUITY_NONFINITE",
                MonitoringStatus.CRITICAL,
                "final equity is missing or non-finite",
                final_equity=reported_final,
            ))
        elif abs(float(reported_final) - curve_final) > max(1e-9, abs(curve_final) * 1e-12):
            issues.append(_issue(
                "FINAL_EQUITY_MISMATCH",
                MonitoringStatus.CRITICAL,
                "reported final equity differs from the last equity point",
                reported=float(reported_final),
                curve_final=curve_final,
            ))
        else:
            metrics["final_equity"] = float(reported_final)

    rejected_actions = _get(backtest, "rejected_actions")
    if isinstance(rejected_actions, bool) or not isinstance(rejected_actions, int) or rejected_actions < 0:
        issues.append(_issue(
            "REJECTED_ACTION_COUNT_INVALID",
            MonitoringStatus.CRITICAL,
            "rejected_actions must be a non-negative integer",
            rejected_actions=rejected_actions,
        ))
    else:
        metrics["rejected_actions"] = rejected_actions

    return issues, metrics


def _audit_performance(result: Any) -> tuple[list[MonitoringIssue], dict[str, Any]]:
    issues: list[MonitoringIssue] = []
    metrics: dict[str, Any] = {}
    backtest = _get(result, "backtest")
    performance = _get(backtest, "metrics")

    if performance is None:
        return [
            _issue("PERFORMANCE_METRICS_MISSING", MonitoringStatus.CRITICAL, "backtest.metrics is missing")
        ], metrics

    required_finite = (
        "initial_equity",
        "final_equity",
        "total_return_pct",
        "max_drawdown_pct",
        "bars_processed",
        "rejected_actions",
    )

    for name in required_finite:
        value = _get(performance, name)
        if name in {"bars_processed", "rejected_actions"}:
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else:
            valid = _is_finite_real(value)
        if not valid:
            issues.append(_issue(
                f"PERFORMANCE_{name.upper()}_INVALID",
                MonitoringStatus.CRITICAL,
                f"performance field {name!r} is invalid",
                value=value,
            ))
        else:
            metrics[name] = float(value) if isinstance(value, Real) and not isinstance(value, bool) else value

    trade_count = _get(performance, "trade_count")
    winning = _get(performance, "winning_trades")
    losing = _get(performance, "losing_trades")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (trade_count, winning, losing)):
        issues.append(_issue(
            "PERFORMANCE_TRADE_COUNTS_INVALID",
            MonitoringStatus.CRITICAL,
            "trade counts must be non-negative integers",
            trade_count=trade_count,
            winning_trades=winning,
            losing_trades=losing,
        ))
    else:
        metrics.update({
            "trade_count": trade_count,
            "winning_trades": winning,
            "losing_trades": losing,
        })
        if winning + losing > trade_count:
            issues.append(_issue(
                "PERFORMANCE_TRADE_COUNT_INCONSISTENT",
                MonitoringStatus.CRITICAL,
                "winning + losing trades exceeds trade_count",
                trade_count=trade_count,
                winning_trades=winning,
                losing_trades=losing,
            ))

    return issues, metrics


def _audit_numeric_finiteness(report: Any, prefix: str, ignored_fields: set[str] | None = None) -> list[MonitoringIssue]:
    """Find non-finite numeric values without treating booleans as numbers."""
    ignored_fields = ignored_fields or set()
    issues: list[MonitoringIssue] = []
    for path, value in _walk_values(report, prefix):
        last = path.rsplit(".", 1)[-1]
        if last in ignored_fields:
            continue
        if isinstance(value, Real) and not isinstance(value, bool) and not isfinite(float(value)):
            issues.append(_issue(
                "NONFINITE_NUMERIC",
                MonitoringStatus.CRITICAL,
                "an auditable numeric value is NaN or Inf",
                path=path,
                value=value,
            ))
    return issues


def _audit_statistical(result: Any) -> list[MonitoringIssue]:
    report = _get(result, "statistical_validation")
    if report is None:
        return [_issue("STATISTICAL_REPORT_MISSING", MonitoringStatus.CRITICAL, "statistical validation report is missing")]
    return _audit_numeric_finiteness(report, "statistical_validation")


def _audit_bootstrap(result: Any, cfg: MonitoringConfig) -> tuple[list[MonitoringIssue], dict[str, Any]]:
    issues: list[MonitoringIssue] = []
    metrics: dict[str, Any] = {}
    report = _get(result, "bootstrap_validation")
    if report is None:
        return [
            _issue("BOOTSTRAP_REPORT_MISSING", MonitoringStatus.CRITICAL, "bootstrap validation report is missing")
        ], metrics

    observations = _get(report, "observations")
    replications = _get(report, "bootstrap_replications")
    block_length = _get(report, "block_length")

    metrics.update({
        "bootstrap_observations": observations,
        "bootstrap_replications": replications,
        "bootstrap_block_length": block_length,
    })

    if isinstance(observations, bool) or not isinstance(observations, int) or observations < cfg.min_bootstrap_observations:
        issues.append(_issue(
            "BOOTSTRAP_OBSERVATIONS_INVALID",
            MonitoringStatus.CRITICAL,
            "bootstrap observations are below the monitoring minimum",
            observations=observations,
            minimum=cfg.min_bootstrap_observations,
        ))

    if isinstance(replications, bool) or not isinstance(replications, int) or replications < cfg.min_bootstrap_replications:
        issues.append(_issue(
            "BOOTSTRAP_REPLICATIONS_INVALID",
            MonitoringStatus.CRITICAL,
            "bootstrap replication count is below the monitoring minimum",
            replications=replications,
            minimum=cfg.min_bootstrap_replications,
        ))

    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < cfg.min_block_length:
        issues.append(_issue(
            "BOOTSTRAP_BLOCK_LENGTH_INVALID",
            MonitoringStatus.CRITICAL,
            "bootstrap block length is invalid",
            block_length=block_length,
            minimum=cfg.min_block_length,
        ))

    curve = _get(_get(result, "backtest"), "equity_curve")
    curve_length = len(tuple(curve)) if curve is not None else 0
    expected = max(0, curve_length - 1)
    if cfg.require_bootstrap_observation_consistency and isinstance(observations, int) and observations != expected:
        issues.append(_issue(
            "BOOTSTRAP_OBSERVATION_MISMATCH",
            MonitoringStatus.CRITICAL,
            "bootstrap observations must equal the number of equity-curve returns",
            reported=observations,
            expected=expected,
        ))

    probability_fields = (
        "bootstrap_positive_sharpe_fraction",
        "probability",
        "p_value",
        "pvalue",
    )
    for field_name in probability_fields:
        value = _get(report, field_name)
        if value is None:
            continue
        if not _is_finite_real(value) or not 0.0 <= float(value) <= 1.0:
            issues.append(_issue(
                "BOOTSTRAP_PROBABILITY_INVALID",
                MonitoringStatus.CRITICAL,
                "probability-like bootstrap field must lie in [0, 1]",
                field=field_name,
                value=value,
            ))

    issues.extend(_audit_numeric_finiteness(
        report,
        "bootstrap_validation",
    ))
    return issues, metrics


def _audit_trades_and_events(result: Any, cfg: MonitoringConfig) -> list[MonitoringIssue]:
    issues: list[MonitoringIssue] = []
    backtest = _get(result, "backtest")

    if cfg.audit_trade_chronology:
        trades = _get(backtest, "trades")
        if trades is not None:
            for index, trade in enumerate(tuple(trades)):
                entry = _get(trade, "entry_timestamp")
                exit_ = _get(trade, "exit_timestamp")
                if isinstance(entry, int) and isinstance(exit_, int):
                    if exit_ < entry:
                        issues.append(_issue(
                            "TRADE_TIME_ORDER_INVALID",
                            MonitoringStatus.CRITICAL,
                            "trade exit timestamp precedes entry timestamp",
                            index=index,
                            entry_timestamp=entry,
                            exit_timestamp=exit_,
                        ))
                else:
                    issues.append(_issue(
                        "TRADE_TIMESTAMP_INVALID",
                        MonitoringStatus.CRITICAL,
                        "trade timestamps are not integers",
                        index=index,
                        entry_timestamp=entry,
                        exit_timestamp=exit_,
                    ))

    if cfg.audit_event_chronology:
        events = _get(backtest, "events")
        if events is not None:
            timestamps = [_get(event, "timestamp") for event in tuple(events)]
            valid = all(isinstance(ts, int) and not isinstance(ts, bool) for ts in timestamps)
            if timestamps and not valid:
                issues.append(_issue(
                    "EVENT_TIMESTAMP_INVALID",
                    MonitoringStatus.CRITICAL,
                    "event timestamps must be integers when events are exposed",
                ))
            elif len(timestamps) >= 2 and any(
                later < earlier for earlier, later in zip(timestamps, timestamps[1:])
            ):
                issues.append(_issue(
                    "EVENT_TIME_ORDER_INVALID",
                    MonitoringStatus.CRITICAL,
                    "backtest events are not chronologically ordered",
                ))

    return issues


def _status(issues: Sequence[MonitoringIssue]) -> MonitoringStatus:
    if any(issue.severity is MonitoringStatus.CRITICAL for issue in issues):
        return MonitoringStatus.CRITICAL
    if any(issue.severity is MonitoringStatus.DEGRADED for issue in issues):
        return MonitoringStatus.DEGRADED
    return MonitoringStatus.HEALTHY


def monitor_integration_result(
    result: Any,
    *,
    config: MonitoringConfig | None = None,
) -> MonitoringReport:
    """Run Phase-15 diagnostics on an IntegrationResult-like object."""
    if result is None:
        raise MonitoringError("result cannot be None")

    cfg = config or MonitoringConfig()
    issues: list[MonitoringIssue] = []
    metrics: dict[str, Any] = {}

    checks = (
        lambda value: _audit_metadata(value),
        lambda value: _audit_equity_curve(value, cfg),
        lambda value: _audit_performance(value),
        lambda value: (_audit_statistical(value), {}),
        lambda value: _audit_bootstrap(value, cfg),
        lambda value: (_audit_trades_and_events(value, cfg), {}),
    )

    for check in checks:
        check_issues, check_metrics = check(result)
        issues.extend(check_issues)
        metrics.update(check_metrics)

    # De-duplicate identical findings while preserving first occurrence.
    unique: list[MonitoringIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (issue.code, issue.message)
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    issues = unique

    # Six independent diagnostic domains are executed above.
    checks_run = 6
    failing_codes = {issue.code for issue in issues}
    checks_passed = max(0, checks_run - len(failing_codes))

    return MonitoringReport(
        status=_status(issues),
        checks_run=checks_run,
        checks_passed=checks_passed,
        issues=tuple(issues),
        metrics=metrics,
    )


__all__ = [
    "MonitoringConfig",
    "MonitoringError",
    "MonitoringIssue",
    "MonitoringReport",
    "MonitoringStatus",
    "monitor_integration_result",
]
