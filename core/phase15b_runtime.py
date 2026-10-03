"""
EVA-AI-01 - Phase 15-B Runtime Boundary
========================================
Compose the existing research integration, Production Gate, and Monitoring
layers into one deterministic, auditable boundary.

This module intentionally does NOT:
- place exchange orders;
- load Binance credentials;
- send Telegram messages;
- use leverage, margin, or shorting;
- alter strategy decisions.

The purpose is to give the next Notification Layer one stable object to
consume without bypassing validation or monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.monitoring import (
    MonitoringConfig,
    MonitoringReport,
    MonitoringStatus,
    monitor_integration_result,
)
from core.production_gate import (
    ProductionGateConfig,
    ProductionGateReport,
    evaluate_integration_result,
)
from core.system_integration import (
    IntegrationConfig,
    IntegrationResult,
    run_integrated_research,
)


class Phase15BRuntimeError(RuntimeError):
    """Raised when the Phase-15-B runtime boundary is not ready."""


class Phase15BStatus(str, Enum):
    """Operational state of the combined research diagnostics boundary."""

    READY = "READY"
    GATE_BLOCKED = "GATE_BLOCKED"
    MONITORING_DEGRADED = "MONITORING_DEGRADED"
    MONITORING_CRITICAL = "MONITORING_CRITICAL"
    GATE_BLOCKED_AND_MONITORING_DEGRADED = "GATE_BLOCKED_AND_MONITORING_DEGRADED"
    GATE_BLOCKED_AND_MONITORING_CRITICAL = "GATE_BLOCKED_AND_MONITORING_CRITICAL"


@dataclass(frozen=True)
class Phase15BConfig:
    """Immutable configuration for the Phase-15-B composition layer."""

    integration_config: IntegrationConfig | None = None
    gate_config: ProductionGateConfig | None = None
    monitoring_config: MonitoringConfig | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strict, bool):
            raise Phase15BRuntimeError("strict must be boolean")


@dataclass(frozen=True)
class Phase15BResult:
    """Complete Phase-15-B evidence bundle."""

    research: IntegrationResult
    production_gate: ProductionGateReport
    monitoring: MonitoringReport
    status: Phase15BStatus

    @property
    def release_ready(self) -> bool:
        """True only when the gate passes and monitoring is HEALTHY."""
        return (
            self.production_gate.approved
            and self.monitoring.status is MonitoringStatus.HEALTHY
            and self.status is Phase15BStatus.READY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "research": self.research.to_dict(),
            "production_gate": self.production_gate.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "status": self.status.value,
            "release_ready": self.release_ready,
        }

    def raise_if_not_ready(self) -> None:
        """Raise with all blocking evidence preserved in the exception text."""
        if self.release_ready:
            return

        reasons: list[str] = []

        if not self.production_gate.approved:
            reasons.extend(self.production_gate.critical_failures)

        if not self.monitoring.healthy:
            reasons.extend(
                f"{issue.code}: {issue.message}"
                for issue in self.monitoring.issues
            )

        detail = "; ".join(reasons) or "Phase-15-B result is not release-ready"
        raise Phase15BRuntimeError(detail)


def _status_from_reports(
    gate: ProductionGateReport,
    monitoring: MonitoringReport,
) -> Phase15BStatus:
    gate_blocked = not gate.approved
    monitoring_critical = monitoring.status is MonitoringStatus.CRITICAL
    monitoring_degraded = monitoring.status is MonitoringStatus.DEGRADED

    if not gate_blocked and not monitoring_critical and not monitoring_degraded:
        return Phase15BStatus.READY

    if gate_blocked and monitoring_critical:
        return Phase15BStatus.GATE_BLOCKED_AND_MONITORING_CRITICAL

    if gate_blocked and monitoring_degraded:
        return Phase15BStatus.GATE_BLOCKED_AND_MONITORING_DEGRADED

    if gate_blocked:
        return Phase15BStatus.GATE_BLOCKED

    if monitoring_critical:
        return Phase15BStatus.MONITORING_CRITICAL

    return Phase15BStatus.MONITORING_DEGRADED


def diagnose_integrated_result(
    result: IntegrationResult,
    *,
    gate_config: ProductionGateConfig | None = None,
    monitoring_config: MonitoringConfig | None = None,
) -> Phase15BResult:
    """Evaluate an already-computed IntegrationResult with both safeguards."""
    if result is None:
        raise Phase15BRuntimeError("result cannot be None")

    gate = evaluate_integration_result(
        result,
        config=gate_config,
    )
    monitoring = monitor_integration_result(
        result,
        config=monitoring_config,
    )

    return Phase15BResult(
        research=result,
        production_gate=gate,
        monitoring=monitoring,
        status=_status_from_reports(gate, monitoring),
    )


def run_phase15b(
    market_data: Any,
    *,
    config: Phase15BConfig | None = None,
) -> Phase15BResult:
    """Run research integration and immediately apply Gate + Monitoring."""
    cfg = config or Phase15BConfig()

    research = run_integrated_research(
        market_data,
        config=cfg.integration_config,
    )

    result = diagnose_integrated_result(
        research,
        gate_config=cfg.gate_config,
        monitoring_config=cfg.monitoring_config,
    )

    if cfg.strict:
        result.raise_if_not_ready()

    return result


__all__ = [
    "Phase15BConfig",
    "Phase15BResult",
    "Phase15BStatus",
    "Phase15BRuntimeError",
    "diagnose_integrated_result",
    "run_phase15b",
]
