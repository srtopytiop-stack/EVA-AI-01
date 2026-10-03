"""
production_gate.py
==================
Phase 14: Final Research-Integration / Production Gate.

Purpose
-------
This module is a defensive validation layer around the already-integrated
research pipeline.  It does NOT decide whether a strategy is profitable and
it does NOT authorize live trading.  It answers a narrower question:

    "Is the research result internally coherent, numerically valid, and
     statistically validated enough to be released as a research result?"

Design principles
-----------------
1. No look-ahead assumption is introduced here.
2. No live orders, API keys, Telegram, leverage, margin, or shorting.
3. Negative performance is not a gate failure; invalid inference is.
4. Time-series dependence is respected through the existing bootstrap layer.
5. Validation metadata must agree with the actual equity-curve sample size.
6. Every numeric output exposed by the validation reports must be finite.
7. The gate is deterministic for deterministic inputs.

Scientific basis
----------------
The checks are deliberately conservative and are informed by the logic of:
- White (2000), Reality Check for data snooping.
- Hansen (2005), Superior Predictive Ability test.
- Politis & Romano (1994), Stationary Bootstrap.
- Bailey & Lopez de Prado (2014), Deflated Sharpe Ratio.
- Bailey et al. (2017), Probability of Backtest Overfitting.

The project already has dedicated statistical and moving-block bootstrap
engines.  This module verifies their integration instead of reimplementing
those estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


_EPS = 1e-12


class ProductionGateError(RuntimeError):
    """Base exception for production-gate failures."""


class ProductionGateBlocked(ProductionGateError):
    """Raised when a strict caller requests a gate pass but it is blocked."""


class GateStatus(str, Enum):
    """Outcome of one gate check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ProductionGateConfig:
    """Deterministic policy for the Phase-14 research gate."""

    min_equity_points: int = 21
    min_bootstrap_observations: int = 20
    min_bootstrap_replications: int = 200
    min_block_length: int = 2
    require_statistical_validation: bool = True
    require_bootstrap_validation: bool = True
    warn_on_unavailable_temporal_audit: bool = True
    require_temporal_audit: bool = False

    def __post_init__(self) -> None:
        if self.min_equity_points < 2:
            raise ValueError("min_equity_points must be >= 2")
        if self.min_bootstrap_observations < 2:
            raise ValueError("min_bootstrap_observations must be >= 2")
        if self.min_bootstrap_replications < 1:
            raise ValueError("min_bootstrap_replications must be >= 1")
        if self.min_block_length < 2:
            raise ValueError("min_block_length must be >= 2")


@dataclass(frozen=True)
class GateCheck:
    """Immutable evidence for one production-gate criterion."""

    name: str
    status: GateStatus
    message: str
    evidence: Mapping[str, Any] | None = None

    @property
    def blocking(self) -> bool:
        return self.status is GateStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "evidence": dict(self.evidence or {}),
        }


@dataclass(frozen=True)
class ProductionGateReport:
    """Complete Phase-14 gate decision."""

    approved: bool
    checks: tuple[GateCheck, ...]
    critical_failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "checks": [check.to_dict() for check in self.checks],
            "critical_failures": list(self.critical_failures),
            "warnings": list(self.warnings),
        }

    def raise_if_blocked(self) -> None:
        if not self.approved:
            detail = "; ".join(self.critical_failures) or "unknown gate failure"
            raise ProductionGateBlocked(detail)


@dataclass(frozen=True)
class TemporalAuditReport:
    """Result of an explicit signal-to-execution temporal-order audit."""

    audited_pairs: int
    passed: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audited_pairs": self.audited_pairs,
            "passed": self.passed,
            "violations": list(self.violations),
        }


def audit_signal_execution_order(
    timestamp_pairs: Iterable[tuple[Any, Any]],
) -> TemporalAuditReport:
    """Verify that every execution occurs strictly after its signal.

    Parameters
    ----------
    timestamp_pairs:
        Iterable of ``(signal_timestamp, execution_timestamp)`` pairs.

    Notes
    -----
    The strict inequality is intentional for a completed-bar signal engine:
    execution at exactly the same timestamp would be ambiguous and can hide
    look-ahead or same-bar execution.
    """

    violations: list[str] = []
    audited = 0
    for index, (signal_ts, execution_ts) in enumerate(timestamp_pairs):
        audited += 1
        try:
            valid = execution_ts > signal_ts
        except TypeError:
            valid = False
        if not valid:
            violations.append(
                f"pair {index}: execution timestamp is not strictly after signal timestamp"
            )

    return TemporalAuditReport(
        audited_pairs=audited,
        passed=not violations,
        violations=tuple(violations),
    )



def run_gated_research(
    market_data: Any,
    *,
    integration_config: Any = None,
    gate_config: ProductionGateConfig | None = None,
) -> tuple[Any, ProductionGateReport]:
    """Run the existing integrated research pipeline and immediately gate it.

    The import is lazy to avoid creating a circular import during module
    discovery. The existing ``run_integrated_research`` function remains the
    source of truth for the research pipeline itself.
    """

    from core.system_integration import run_integrated_research

    if integration_config is None:
        result = run_integrated_research(market_data)
    else:
        result = run_integrated_research(market_data, config=integration_config)

    report = evaluate_integration_result(result, config=gate_config)
    return result, report

def evaluate_integration_result(
    result: Any,
    *,
    config: ProductionGateConfig | None = None,
) -> ProductionGateReport:
    """Run the complete Phase-14 gate against an integration result.

    ``result`` is intentionally duck-typed so this gate remains compatible
    with the frozen dataclasses already used by the project without creating a
    circular dependency back into ``system_integration.py``.
    """

    cfg = config or ProductionGateConfig()
    checks: list[GateCheck] = []

    checks.append(_check_metadata(result))
    checks.append(_check_equity_curve(result, cfg))
    checks.append(_check_backtest_numeric_integrity(result))
    checks.append(_check_statistical_validation(result, cfg))
    checks.append(_check_bootstrap_validation(result, cfg))
    checks.append(_check_temporal_audit(result, cfg))

    failures = tuple(
        f"{check.name}: {check.message}"
        for check in checks
        if check.status is GateStatus.FAIL
    )
    warnings = tuple(
        f"{check.name}: {check.message}"
        for check in checks
        if check.status is GateStatus.WARN
    )

    return ProductionGateReport(
        approved=not failures,
        checks=tuple(checks),
        critical_failures=failures,
        warnings=warnings,
    )


def _check_metadata(result: Any) -> GateCheck:
    required = ("symbol", "interval", "candle_count", "feature_rows", "feature_columns")
    missing = [name for name in required if _get(result, name, None) is None]
    if missing:
        return GateCheck(
            name="integration_metadata",
            status=GateStatus.FAIL,
            message=f"missing integration metadata: {', '.join(missing)}",
        )

    candle_count = _get(result, "candle_count")
    feature_rows = _get(result, "feature_rows")
    feature_columns = _get(result, "feature_columns")

    if not isinstance(candle_count, int) or candle_count < 2:
        return GateCheck(
            name="integration_metadata",
            status=GateStatus.FAIL,
            message="candle_count must be an integer >= 2",
            evidence={"candle_count": candle_count},
        )
    if not isinstance(feature_rows, int) or feature_rows < 1:
        return GateCheck(
            name="integration_metadata",
            status=GateStatus.FAIL,
            message="feature_rows must be an integer >= 1",
            evidence={"feature_rows": feature_rows},
        )
    if not isinstance(feature_columns, Sequence) or isinstance(feature_columns, (str, bytes)):
        return GateCheck(
            name="integration_metadata",
            status=GateStatus.FAIL,
            message="feature_columns must be a sequence",
        )
    if len(feature_columns) == 0:
        return GateCheck(
            name="integration_metadata",
            status=GateStatus.FAIL,
            message="feature_columns is empty",
        )

    return GateCheck(
        name="integration_metadata",
        status=GateStatus.PASS,
        message="required integration metadata is present",
        evidence={
            "symbol": _get(result, "symbol"),
            "interval": _get(result, "interval"),
            "candle_count": candle_count,
            "feature_rows": feature_rows,
            "feature_column_count": len(feature_columns),
        },
    )


def _check_equity_curve(result: Any, cfg: ProductionGateConfig) -> GateCheck:
    backtest = _get(result, "backtest")
    equity_curve = _get(backtest, "equity_curve", None)
    if equity_curve is None:
        return GateCheck(
            name="equity_curve_integrity",
            status=GateStatus.FAIL,
            message="backtest equity_curve is missing",
        )

    points = list(equity_curve)
    if len(points) < cfg.min_equity_points:
        return GateCheck(
            name="equity_curve_integrity",
            status=GateStatus.FAIL,
            message=(
                f"equity curve has {len(points)} points; "
                f"minimum is {cfg.min_equity_points}"
            ),
            evidence={"observations": len(points)},
        )

    equities: list[float] = []
    timestamps: list[Any] = []
    for index, point in enumerate(points):
        equity = _get(point, "equity", None)
        timestamp = _get(point, "timestamp", None)
        if not _finite_positive(equity):
            return GateCheck(
                name="equity_curve_integrity",
                status=GateStatus.FAIL,
                message=f"equity at index {index} is not finite and positive",
                evidence={"index": index, "equity": equity},
            )
        if timestamp is None:
            return GateCheck(
                name="equity_curve_integrity",
                status=GateStatus.FAIL,
                message=f"timestamp at index {index} is missing",
            )
        equities.append(float(equity))
        timestamps.append(timestamp)

    for index in range(1, len(timestamps)):
        try:
            ordered = timestamps[index] > timestamps[index - 1]
        except TypeError:
            ordered = False
        if not ordered:
            return GateCheck(
                name="equity_curve_integrity",
                status=GateStatus.FAIL,
                message="equity-curve timestamps are not strictly increasing",
                evidence={"index": index},
            )

    returns = [(equities[i] / equities[i - 1]) - 1.0 for i in range(1, len(equities))]
    if any(not math.isfinite(value) for value in returns):
        return GateCheck(
            name="equity_curve_integrity",
            status=GateStatus.FAIL,
            message="derived periodic returns contain non-finite values",
        )

    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    if variance <= _EPS:
        return GateCheck(
            name="equity_curve_integrity",
            status=GateStatus.FAIL,
            message="periodic return variance is zero; Sharpe inference is undefined",
            evidence={"variance": variance, "return_observations": len(returns)},
        )

    final_equity = _get(result, "final_equity", None)
    if final_equity is not None:
        try:
            mismatch = abs(float(final_equity) - equities[-1])
        except (TypeError, ValueError):
            return GateCheck(
                name="equity_curve_integrity",
                status=GateStatus.FAIL,
                message="final_equity is not numeric",
            )
        tolerance = 1e-9 * max(1.0, abs(equities[-1]))
        if mismatch > tolerance:
            return GateCheck(
                name="equity_curve_integrity",
                status=GateStatus.FAIL,
                message="final_equity does not match the last equity-curve point",
                evidence={"reported": final_equity, "curve_last": equities[-1]},
            )

    return GateCheck(
        name="equity_curve_integrity",
        status=GateStatus.PASS,
        message="equity curve is finite, positive, ordered, and statistically non-degenerate",
        evidence={
            "equity_points": len(points),
            "return_observations": len(returns),
            "return_variance": variance,
            "final_equity": equities[-1],
        },
    )


def _check_backtest_numeric_integrity(result: Any) -> GateCheck:
    backtest = _get(result, "backtest")
    if backtest is None:
        return GateCheck(
            name="backtest_integrity",
            status=GateStatus.FAIL,
            message="backtest result is missing",
        )

    bars_processed = _get(backtest, "bars_processed", None)
    if not isinstance(bars_processed, int) or bars_processed < 1:
        return GateCheck(
            name="backtest_integrity",
            status=GateStatus.FAIL,
            message="bars_processed must be an integer >= 1",
            evidence={"bars_processed": bars_processed},
        )

    metrics = _get(backtest, "metrics", None)
    if metrics is not None:
        violations = _find_nonfinite_numbers(metrics)
        if violations:
            return GateCheck(
                name="backtest_integrity",
                status=GateStatus.FAIL,
                message="backtest metrics contain non-finite numeric values",
                evidence={"paths": violations[:10]},
            )

    return GateCheck(
        name="backtest_integrity",
        status=GateStatus.PASS,
        message="backtest structure and exposed numeric metrics are valid",
        evidence={"bars_processed": bars_processed},
    )


def _check_statistical_validation(result: Any, cfg: ProductionGateConfig) -> GateCheck:
    report = _get(result, "statistical_validation", None)
    if report is None:
        status = GateStatus.FAIL if cfg.require_statistical_validation else GateStatus.WARN
        return GateCheck(
            name="statistical_validation",
            status=status,
            message="statistical validation report is missing",
        )

    violations = _find_nonfinite_numbers(report)
    if violations:
        return GateCheck(
            name="statistical_validation",
            status=GateStatus.FAIL,
            message="statistical validation contains non-finite numeric values",
            evidence={"paths": violations[:10]},
        )

    return GateCheck(
        name="statistical_validation",
        status=GateStatus.PASS,
        message="statistical validation report is present and numerically finite",
    )


def _check_bootstrap_validation(result: Any, cfg: ProductionGateConfig) -> GateCheck:
    report = _get(result, "bootstrap_validation", None)
    if report is None:
        status = GateStatus.FAIL if cfg.require_bootstrap_validation else GateStatus.WARN
        return GateCheck(
            name="bootstrap_validation",
            status=status,
            message="bootstrap validation report is missing",
        )

    equity_curve = _get(_get(result, "backtest"), "equity_curve", ())
    return_observations = max(0, len(list(equity_curve)) - 1)

    observations = _get(report, "observations", None)
    replications = _get(report, "bootstrap_replications", None)
    block_length = _get(report, "block_length", None)

    required = {
        "observations": observations,
        "bootstrap_replications": replications,
        "block_length": block_length,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message=f"bootstrap metadata is incomplete: {', '.join(missing)}",
        )

    if observations != return_observations:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message="bootstrap observations do not match equity-curve return observations",
            evidence={
                "reported_observations": observations,
                "derived_observations": return_observations,
            },
        )

    if observations < cfg.min_bootstrap_observations:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message=(
                f"bootstrap has {observations} observations; "
                f"minimum is {cfg.min_bootstrap_observations}"
            ),
        )

    if not isinstance(replications, int) or replications < cfg.min_bootstrap_replications:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message=(
                f"bootstrap_replications must be >= {cfg.min_bootstrap_replications}"
            ),
            evidence={"bootstrap_replications": replications},
        )

    if not isinstance(block_length, int):
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message="block_length must be an integer",
        )
    if block_length < cfg.min_block_length or block_length >= observations:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message="block_length is outside the valid bootstrap range",
            evidence={"block_length": block_length, "observations": observations},
        )

    violations = _find_nonfinite_numbers(report)
    if violations:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message="bootstrap validation contains non-finite numeric values",
            evidence={"paths": violations[:10]},
        )

    probability_violations = _find_invalid_probabilities(report)
    if probability_violations:
        return GateCheck(
            name="bootstrap_validation",
            status=GateStatus.FAIL,
            message="bootstrap validation contains probabilities outside [0, 1]",
            evidence={"paths": probability_violations[:10]},
        )

    return GateCheck(
        name="bootstrap_validation",
        status=GateStatus.PASS,
        message="moving-block bootstrap metadata is coherent and sufficiently replicated",
        evidence={
            "observations": observations,
            "bootstrap_replications": replications,
            "block_length": block_length,
        },
    )


def _check_temporal_audit(result: Any, cfg: ProductionGateConfig) -> GateCheck:
    backtest = _get(result, "backtest")
    events = _get(backtest, "events", None)
    if events is None:
        if cfg.require_temporal_audit:
            return GateCheck(
                name="temporal_order_audit",
                status=GateStatus.FAIL,
                message="backtest events are unavailable for temporal-order auditing",
            )
        return GateCheck(
            name="temporal_order_audit",
            status=GateStatus.WARN if cfg.warn_on_unavailable_temporal_audit else GateStatus.PASS,
            message="current integration API does not expose standardized signal/execution timestamp pairs",
        )

    pairs = _extract_temporal_pairs(events)
    if not pairs:
        if cfg.require_temporal_audit:
            return GateCheck(
                name="temporal_order_audit",
                status=GateStatus.FAIL,
                message="events exist, but no recognized signal/execution timestamp pairs were found",
            )
        return GateCheck(
            name="temporal_order_audit",
            status=GateStatus.WARN if cfg.warn_on_unavailable_temporal_audit else GateStatus.PASS,
            message="events exist but their schema is not standardized for automatic temporal auditing",
        )

    audit = audit_signal_execution_order(pairs)
    if not audit.passed:
        return GateCheck(
            name="temporal_order_audit",
            status=GateStatus.FAIL,
            message="at least one execution occurs at or before its signal timestamp",
            evidence=audit.to_dict(),
        )

    return GateCheck(
        name="temporal_order_audit",
        status=GateStatus.PASS,
        message="all recognized signal/execution pairs respect strict temporal order",
        evidence=audit.to_dict(),
    )


def _extract_temporal_pairs(events: Iterable[Any]) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for event in events:
        signal_ts = _first_present(
            event,
            "signal_timestamp",
            "signal_ts",
            "decision_timestamp",
            "decision_ts",
        )
        execution_ts = _first_present(
            event,
            "execution_timestamp",
            "execution_ts",
            "fill_timestamp",
            "fill_ts",
        )
        if signal_ts is not None and execution_ts is not None:
            pairs.append((signal_ts, execution_ts))
    return pairs


def _first_present(obj: Any, *names: str) -> Any:
    for name in names:
        value = _get(obj, name, None)
        if value is not None:
            return value
    return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


def _as_data(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bytes, bool, int, float)):
        return obj
    if isinstance(obj, Mapping):
        return obj
    if is_dataclass(obj):
        try:
            return {field.name: getattr(obj, field.name) for field in fields(obj)}
        except Exception:
            return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return obj


def _find_nonfinite_numbers(obj: Any, path: str = "root") -> list[str]:
    data = _as_data(obj)
    violations: list[str] = []

    if isinstance(data, Real) and not isinstance(data, bool):
        try:
            if not math.isfinite(float(data)):
                violations.append(path)
        except (TypeError, ValueError, OverflowError):
            violations.append(path)
        return violations

    if isinstance(data, Mapping):
        for key, value in data.items():
            violations.extend(_find_nonfinite_numbers(value, f"{path}.{key}"))
        return violations

    if isinstance(data, (list, tuple, set)):
        for index, value in enumerate(data):
            violations.extend(_find_nonfinite_numbers(value, f"{path}[{index}]"))
        return violations

    return violations


def _find_invalid_probabilities(obj: Any, path: str = "root") -> list[str]:
    data = _as_data(obj)
    violations: list[str] = []

    if isinstance(data, Mapping):
        for key, value in data.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}"
            if (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and ("probabil" in key_text or "p_value" in key_text or key_text in {"p", "pvalue"})
            ):
                if not 0.0 <= float(value) <= 1.0:
                    violations.append(child_path)
            else:
                violations.extend(_find_invalid_probabilities(value, child_path))
        return violations

    if isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            violations.extend(_find_invalid_probabilities(value, f"{path}[{index}]"))
        return violations

    return violations


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number > 0.0
