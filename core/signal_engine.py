"""
EVA-AI-01 — Regime-Aware Robust Signal Engine (R-RSE v1)
===========================================================

Purpose
-------
A deterministic, research-oriented signal layer for spot-crypto research.
The engine converts a historical/current OHLCV-like close/volume stream into
an auditable directional hypothesis without using future observations.

Design principles
-----------------
1. No look-ahead: every statistic at index t depends only on observations
   available at or before t.
2. Robustness: classical z-score and median/MAD z-score are combined so that
   a single extreme observation does not dominate the signal.
3. Volatility awareness: EWMA volatility scales the trend/momentum component.
4. Regime readiness: an optional external regime prior and volatility forecast
   can be injected later by HMM/GARCH modules without coupling this engine to
   those implementations.
5. Hysteresis: entry/hold thresholds reduce signal chattering near zero.
6. Auditability: every returned score has component scores and reasons.
7. Spot-safe default: the engine expresses a direction hypothesis only;
   execution and risk limits live in separate layers.

Scientific lineage
------------------
The implementation is intentionally modular so later EVA modules can connect
literature-backed components such as:
- conditional volatility models (ARCH/GARCH),
- Markov-switching / HMM regime probabilities,
- order-flow imbalance and market-depth features,
- triple-barrier labels and meta-labeling,
- purged/embargoed time-series validation,
- deflated Sharpe / probability-of-backtest-overfitting controls.

This file does not claim that any component is a profitable strategy. It is a
signal hypothesis generator that must be validated with out-of-sample tests,
transaction-cost assumptions and paper trading before any financial use.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt, tanh
from statistics import median
from typing import Sequence

EPS = 1e-12


@dataclass(frozen=True)
class SignalResult:
    """Immutable, auditable output of the signal engine."""

    symbol: str
    timestamp: int
    direction: int
    score: float
    confidence: float
    classical_z: float
    robust_z: float
    volatility: float
    trend_score: float
    mean_reversion_score: float
    volume_score: float
    regime_factor: float
    volatility_factor: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "score": self.score,
            "confidence": self.confidence,
            "classical_z": self.classical_z,
            "robust_z": self.robust_z,
            "volatility": self.volatility,
            "trend_score": self.trend_score,
            "mean_reversion_score": self.mean_reversion_score,
            "volume_score": self.volume_score,
            "regime_factor": self.regime_factor,
            "volatility_factor": self.volatility_factor,
            "reasons": list(self.reasons),
        }


class SignalEngineError(ValueError):
    """Raised when input data cannot be evaluated safely."""


def _validate_series(values: Sequence[float], name: str, minimum: int) -> None:
    if len(values) < minimum:
        raise SignalEngineError(f"{name} requires at least {minimum} observations")
    if any(not isfinite(float(v)) for v in values):
        raise SignalEngineError(f"{name} contains non-finite values")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _z_score(value: float, window: Sequence[float]) -> float:
    sigma = _std(window)
    if sigma <= EPS:
        return 0.0
    return (value - _mean(window)) / sigma


def _robust_z_score(value: float, window: Sequence[float]) -> float:
    med = median(window)
    mad = median([abs(x - med) for x in window])
    robust_sigma = 1.4826 * mad
    if robust_sigma <= EPS:
        return 0.0
    return (value - med) / robust_sigma


def _ewma_volatility(prices: Sequence[float], decay: float) -> float:
    """Dimensionless EWMA volatility of log returns."""
    if not 0.0 < decay < 1.0:
        raise SignalEngineError("decay must be in (0, 1)")
    if len(prices) < 3:
        return 0.0

    variance = 0.0
    weight_sum = 0.0
    weight = 1.0
    for i in range(len(prices) - 1, 0, -1):
        p_now = float(prices[i])
        p_prev = float(prices[i - 1])
        if p_now <= 0 or p_prev <= 0:
            raise SignalEngineError("prices must be strictly positive")
        r = log(p_now / p_prev)
        variance += weight * r * r
        weight_sum += weight
        weight *= decay
    return sqrt(variance / max(weight_sum, EPS))


def _normalized_trend(prices: Sequence[float], lookback: int, vol: float) -> float:
    """Volatility-normalized least-squares slope of log price."""
    if lookback < 2 or len(prices) < lookback:
        return 0.0
    segment = [log(float(x)) for x in prices[-lookback:]]
    x_bar = (lookback - 1) / 2.0
    y_bar = _mean(segment)
    numerator = sum((i - x_bar) * (y - y_bar) for i, y in enumerate(segment))
    denominator = sum((i - x_bar) ** 2 for i in range(lookback))
    slope = numerator / max(denominator, EPS)
    scale = max(vol, 0.002)
    return tanh(slope / scale * 8.0)


def _volume_z_score(volumes: Sequence[float], lookback: int) -> float:
    if len(volumes) < lookback:
        return 0.0
    window = list(map(float, volumes[-lookback:]))
    return _clamp(_z_score(window[-1], window), -4.0, 4.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate(
    *,
    symbol: str,
    timestamp: int,
    closes: Sequence[float],
    volumes: Sequence[float],
    z_window: int = 40,
    trend_window: int = 24,
    volume_window: int = 20,
    ewma_decay: float = 0.94,
    entry_threshold: float = 0.55,
    neutral_threshold: float = 0.25,
    external_regime_factor: float = 1.0,
    forecast_volatility: float | None = None,
) -> SignalResult:
    """Evaluate the current bar using only observations supplied by the caller.

    No future price or volume is accessed internally. External regime and
    volatility inputs are treated as conviction modifiers, not as hidden data.
    """
    if not symbol.strip():
        raise SignalEngineError("symbol must not be empty")
    if timestamp <= 0:
        raise SignalEngineError("timestamp must be a positive Unix timestamp")
    if not 0.0 < entry_threshold <= 1.0:
        raise SignalEngineError("entry_threshold must be in (0, 1]")
    if not 0.0 <= neutral_threshold < entry_threshold:
        raise SignalEngineError("neutral_threshold must be in [0, entry_threshold)")
    if len(closes) != len(volumes):
        raise SignalEngineError("closes and volumes must have equal length")

    minimum = max(z_window, trend_window, volume_window, 8)
    _validate_series(closes, "closes", minimum)
    _validate_series(volumes, "volumes", minimum)
    if any(float(p) <= 0 for p in closes):
        raise SignalEngineError("all close prices must be strictly positive")
    if any(float(v) < 0 for v in volumes):
        raise SignalEngineError("volumes cannot be negative")

    regime_factor = _clamp(float(external_regime_factor), 0.0, 1.0)
    vol = _ewma_volatility(closes, ewma_decay)

    price_window = list(map(float, closes[-z_window:]))
    current = price_window[-1]
    classical_z = _clamp(_z_score(current, price_window), -8.0, 8.0)
    robust_z = _clamp(_robust_z_score(current, price_window), -8.0, 8.0)

    # Robust/classical consensus. Robust statistics get slightly higher weight
    # because crypto returns and prices can be heavy-tailed and outlier-prone.
    blended_z = 0.45 * classical_z + 0.55 * robust_z
    mean_reversion_score = _clamp(-tanh(blended_z / 2.2), -1.0, 1.0)
    trend_score = _normalized_trend(closes, trend_window, vol)

    raw_volume_z = _volume_z_score(volumes, volume_window)
    volume_confirmation = tanh(raw_volume_z / 1.5)
    direction_hint = trend_score if abs(trend_score) >= 0.15 else (-1.0 if blended_z > 0 else 1.0)
    volume_score = _clamp(volume_confirmation * direction_hint, -1.0, 1.0)

    # Adaptive mixture: stronger trends receive more weight; weak trends allow
    # the mean-reversion component to dominate rather than blindly fade trends.
    trend_strength = abs(trend_score)
    w_trend = _clamp(0.25 + 0.55 * trend_strength, 0.25, 0.80)
    w_mean_rev = 1.0 - w_trend
    w_volume = 0.15
    core = (1.0 - w_volume) * (
        w_trend * trend_score + w_mean_rev * mean_reversion_score
    ) + w_volume * volume_score

    volatility_factor = 1.0
    if forecast_volatility is not None:
        fvol = max(float(forecast_volatility), EPS)
        ratio = fvol / max(vol, EPS)
        volatility_factor = _clamp(1.0 / (1.0 + max(0.0, ratio - 1.0)), 0.35, 1.0)

    score = _clamp(core * regime_factor * volatility_factor, -1.0, 1.0)

    components = [trend_score, mean_reversion_score, volume_score]
    same_direction = [
        ((x >= 0) == (score >= 0)) and abs(x) > 0.05 for x in components
    ]
    agreement = sum(same_direction) / len(components)
    data_quality = min(1.0, len(closes) / float(max(120, minimum * 2)))
    confidence = _clamp(
        abs(score) * (0.55 + 0.45 * agreement) * (0.65 + 0.35 * data_quality),
        0.0,
        1.0,
    )

    if abs(score) < neutral_threshold:
        direction = 0
    elif abs(score) >= entry_threshold:
        direction = 1 if score > 0 else -1
    else:
        direction = 0

    reasons: list[str] = []
    if abs(robust_z) >= 2.0:
        reasons.append("robust-price-deviation")
    if abs(trend_score) >= 0.5:
        reasons.append("volatility-normalized-trend")
    if abs(volume_score) >= 0.35:
        reasons.append("volume-confirmation")
    if regime_factor < 0.75:
        reasons.append("regime-conviction-reduced")
    if volatility_factor < 0.80:
        reasons.append("forecast-volatility-damping")
    if direction == 0:
        reasons.append("hysteresis-neutral-zone")

    return SignalResult(
        symbol=symbol,
        timestamp=int(timestamp),
        direction=direction,
        score=score,
        confidence=confidence,
        classical_z=classical_z,
        robust_z=robust_z,
        volatility=vol,
        trend_score=trend_score,
        mean_reversion_score=mean_reversion_score,
        volume_score=volume_score,
        regime_factor=regime_factor,
        volatility_factor=volatility_factor,
        reasons=tuple(reasons),
    )


def _self_test() -> None:
    """Small deterministic smoke test; no external packages required."""
    closes = [100.0 + 0.05 * i for i in range(80)]
    volumes = [1000.0 + (i % 5) * 25.0 for i in range(80)]
    result = evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
    )
    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.direction in (-1, 0, 1)


if __name__ == "__main__":
    _self_test()
    print("R-RSE v1 self-test: PASS")
