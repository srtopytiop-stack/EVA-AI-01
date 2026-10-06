"""
EVA-AI-01 - Notification Layer
===============================
Phase-15-B notification boundary.

The formatter is always available and has no side effects.
Telegram delivery is explicitly opt-in through TELEGRAM_ENABLED=true.

Important
---------
Telegram is an analysis/diagnostic notification channel only.

A blocked Production Gate does NOT prevent EVA from reporting its
research result. The Gate status is included honestly in the message.

Safety
------
- No trading decisions are made here.
- No live orders are placed.
- No Binance trading credentials are accepted.
- Telegram is disabled by default.
- Telegram never changes release_ready.
- Telegram never bypasses Production Gate.
- Telegram only reports already-computed research results.
- Failed statistical validation is reported honestly.
- No metric is invented or fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import requests

from core.phase15b_runtime import Phase15BResult


class NotificationError(RuntimeError):
    """Raised when a notification cannot be delivered safely."""


@dataclass(frozen=True)
class TelegramConfig:
    """Immutable Telegram notification configuration."""

    enabled: bool = False
    bot_token: str | None = None
    chat_id: str | None = None
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> "TelegramConfig":
        enabled = _parse_bool("TELEGRAM_ENABLED", False)

        token = (
            _env_optional("TELEGRAM_BOT_TOKEN")
            or _env_optional("BOT_TOKEN")
        )

        chat_id = (
            _env_optional("TELEGRAM_CHAT_ID")
            or _env_optional("ADMIN_ID")
        )

        if enabled and (not token or not chat_id):
            raise NotificationError(
                "TELEGRAM_ENABLED=true requires "
                "TELEGRAM_BOT_TOKEN or BOT_TOKEN, and "
                "TELEGRAM_CHAT_ID or ADMIN_ID"
            )

        return cls(
            enabled=enabled,
            bot_token=token,
            chat_id=chat_id,
        )


@dataclass(frozen=True)
class NotificationResult:
    """Result of a notification attempt."""

    enabled: bool
    sent: bool
    skipped: bool
    message: str


def _env_optional(name: str) -> str | None:
    """Return a stripped environment variable or None."""
    value = os.getenv(name)

    if value is None or not value.strip():
        return None

    return value.strip()


def _parse_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable safely."""
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    normalized = raw.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise NotificationError(
        f"{name} must be boolean (true/false, yes/no, 1/0)"
    )


def _safe_attr(
    obj: object,
    name: str,
    default: object = None,
) -> object:
    """Read an optional field from real or test-double objects."""
    return getattr(obj, name, default)


def format_phase15b_message(result: Phase15BResult) -> str:
    """
    Create a concise and auditable Telegram research report.

    The message reports the Production Gate state but does not allow
    that state to suppress the research notification.
    """

    research = result.research
    backtest = research.backtest
    metrics = backtest.metrics
    stats = research.statistical_validation
    bootstrap = research.bootstrap_validation

    lines = [
        "EVA-AI-01 — Research Report",
        "============================",
        f"Symbol: {research.symbol}",
        f"Interval: {research.interval}",
        f"Candles: {research.candle_count}",
        f"Runtime status: {result.status.value}",
        f"Release ready: {result.release_ready}",
        "",
        "Backtest",
        f"Initial equity: {metrics.initial_equity:.4f}",
        f"Final equity: {metrics.final_equity:.4f}",
        f"Total return: {metrics.total_return_pct:.4f}%",
        f"Max drawdown: {metrics.max_drawdown_pct:.4f}%",
        f"Trades: {metrics.trade_count}",
        f"Rejected actions: {metrics.rejected_actions}",
        "",
        "Statistical validation",
    ]

    if stats is None:
        lines.append("Status: INSUFFICIENT_DATA")

        reason = _safe_attr(
            research,
            "statistical_validation_error",
            None,
        )

        if reason:
            lines.append(f"Reason: {reason}")

    else:
        lines.extend(
            [
                f"Periodic Sharpe: {stats.periodic_sharpe:.6f}",
                (
                    "Probabilistic Sharpe: "
                    f"{stats.probabilistic_sharpe:.6f}"
                ),
                (
                    "Deflated Sharpe: "
                    f"{stats.deflated_sharpe:.6f}"
                ),
            ]
        )

    lines.append("")
    lines.append("Moving-block bootstrap")

    if bootstrap is None:
        lines.append("Status: INSUFFICIENT_DATA")

        reason = _safe_attr(
            research,
            "bootstrap_validation_error",
            None,
        )

        if reason:
            lines.append(f"Reason: {reason}")

    else:
        lines.extend(
            [
                f"Observations: {bootstrap.observations}",
                (
                    "Replications: "
                    f"{bootstrap.bootstrap_replications}"
                ),
                f"Block length: {bootstrap.block_length}",
                (
                    "Sharpe CI: "
                    f"[{bootstrap.sharpe_ci_lower:.6f}, "
                    f"{bootstrap.sharpe_ci_upper:.6f}]"
                ),
                (
                    "Positive Sharpe fraction: "
                    f"{bootstrap.bootstrap_positive_sharpe_fraction:.4f}"
                ),
            ]
        )

    if result.production_gate.critical_failures:
        lines.extend(
            [
                "",
                "Production Gate — CRITICAL FAILURES:",
                *[
                    f"- {failure}"
                    for failure
                    in result.production_gate.critical_failures
                ],
            ]
        )

    if result.production_gate.warnings:
        lines.extend(
            [
                "",
                "Production Gate — WARNINGS:",
                *[
                    f"- {warning}"
                    for warning
                    in result.production_gate.warnings
                ],
            ]
        )

    if result.monitoring.issues:
        lines.extend(
            [
                "",
                "Monitoring issues:",
                *[
                    (
                        f"- [{issue.severity.value}] "
                        f"{issue.code}: {issue.message}"
                    )
                    for issue in result.monitoring.issues
                ],
            ]
        )

    # Explicitly distinguish analysis from trading readiness.
    if result.release_ready:
        lines.extend(
            [
                "",
                "Gate status: RELEASE-READY",
                "This report passed the current release gate.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Gate status: BLOCKED",
                (
                    "Research was reported for diagnostics. "
                    "The blocked gate was NOT bypassed."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Safety: analysis/research-only.",
            "No live order was placed.",
            "No trading action was executed by EVA-AI-01.",
        ]
    )

    return "\n".join(lines)


class TelegramNotifier:
    """
    Opt-in Telegram transport for already-computed research reports.

    This class does NOT make trading decisions and does NOT bypass
    Production Gate or Monitoring.
    """

    TELEGRAM_API = "https://api.telegram.org"
    MAX_MESSAGE_LENGTH = 4096

    def __init__(
        self,
        config: TelegramConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or TelegramConfig.from_environment()
        self.session = session or requests.Session()

    def send(self, result: Phase15BResult) -> NotificationResult:
        """
        Send the Phase-15-B research report.

        A blocked release gate is reported rather than silently suppressing
        the analysis. Telegram is informational only.
        """

        message = format_phase15b_message(result)

        if not self.config.enabled:
            return NotificationResult(
                enabled=False,
                sent=False,
                skipped=True,
                message="Telegram notifications are disabled",
            )

        if not self.config.bot_token or not self.config.chat_id:
            raise NotificationError(
                "Telegram configuration is incomplete"
            )

        url = (
            f"{self.TELEGRAM_API}/bot"
            f"{self.config.bot_token}/sendMessage"
        )

        payload = {
            "chat_id": self.config.chat_id,
            "text": message[: self.MAX_MESSAGE_LENGTH],
            "disable_web_page_preview": True,
        }

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )

        except requests.RequestException as exc:
            raise NotificationError(
                f"Telegram request failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise NotificationError(
                "Telegram returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            body = response.json()

        except ValueError as exc:
            raise NotificationError(
                "Telegram returned invalid JSON"
            ) from exc

        if body.get("ok") is not True:
            raise NotificationError(
                f"Telegram rejected the message: {body!r}"
            )

        return NotificationResult(
            enabled=True,
            sent=True,
            skipped=False,
            message="Telegram notification sent",
        )


__all__ = [
    "NotificationError",
    "NotificationResult",
    "TelegramConfig",
    "TelegramNotifier",
    "format_phase15b_message",
]
