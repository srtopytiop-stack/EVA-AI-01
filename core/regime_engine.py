"""
EVA-AI-01 - Market Regime Engine
================================

Phase 6: Market Regime Detection.

This module provides a deterministic, auditable baseline classifier for the
latest market state. It uses only observations supplied by the caller and
never performs trading or execution.

Supported regimes:
- TREND_UP
- TREND_DOWN
- RANGE
- HIGH_VOLATILITY
- LOW_VOLATILITY
- TRANSITION

The module exposes both a functional API (``classify``) and a thin
``RegimeEngine`` facade for compatibility with object-oriented callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log, sqrt, tanh
from statistics import mean
from typing import Sequence


EPS = 1e-12


class RegimeEngineError(ValueError):
    """Raised when Market Regime Engine inputs are invalid."""


class MarketRegime(str, Enum):
    """Supported market-regime states."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class RegimeResult:
    """Immutable and auditable market-regime result."""

    regime: MarketRegime
    confidence: float
    trend_score: float
    volatility: float
    volatility_ratio: float
    directional_efficiency: float
    directional_persistence: float
    reasons: tuple[str, ...]

    @property
    def persistence(self) -> float:
        """Backward-compatible alias for directional persistence."""

        return self.directional_persistence

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly result."""

        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "trend_score": self.trend_score,
            "volatility": self.volatility,
            "volatility_ratio": self.volatility_ratio,
            "directional_efficiency": self.directional_efficiency,
            "directional_persistence": self.directional_persistence,
            "persistence": self.directional_persistence,
            "reasons": list(self.reasons),
        }


def _validate_integer_parameter(
    value: int,
    name: str,
    minimum: int,
) -> None:
    """Validate a positive integer parameter."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise RegimeEngineError(f"{name} must be an integer")

    if value < minimum:
        raise RegimeEngineError(
            f"{name} must be at least {minimum}"
        )


def _validate_prices(
    prices: Sequence[float],
    minimum: int,
) -> list[float]:
    """Validate and normalize positive finite price observations."""

    if not isinstance(prices, Sequence):
        raise RegimeEngineError("prices must be a sequence")

    if len(prices) < minimum:
        raise RegimeEngineError(
            f"prices requires at least {minimum} observations"
        )

    normalized: list[float] = []

    for index, raw_value in enumerate(prices):
        if isinstance(raw_value, bool):
            raise RegimeEngineError(
                f"prices[{index}] must be a real number"
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RegimeEngineError(
                f"prices[{index}] must be numeric"
            ) from exc

        if not isfinite(value):
            raise RegimeEngineError(
                f"prices[{index}] must be finite"
            )

        if value <= 0.0:
            raise RegimeEngineError(
                f"prices[{index}] must be strictly positive"
            )

        normalized.append(value)

    return normalized


def _log_returns(prices: Sequence[float]) -> list[float]:
    """Calculate one-period log returns."""

    returns: list[float] = []

    for previous, current in zip(prices[:-1], prices[1:]):
        returns.append(log(float(current) / float(previous)))

    return returns


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean with an empty-sequence guard."""

    if not values:
        return 0.0

    return mean(float(value) for value in values)


def _std(values: Sequence[float]) -> float:
    """Sample standard deviation."""

    if len(values) < 2:
        return 0.0

    mu = _mean(values)
    variance = sum(
        (float(value) - mu) ** 2
        for value in values
    ) / (len(values) - 1)

    return sqrt(max(variance, 0.0))


def _ewma_volatility(
    returns: Sequence[float],
    decay: float,
) -> float:
    """
    Compute EWMA volatility for a finite return window.

    More recent squared returns receive larger weights. The estimator is
    intentionally zero-mean because this is a volatility state variable,
    not a return forecast.
    """

    if not 0.0 < decay < 1.0:
        raise RegimeEngineError("ewma_decay must be in (0, 1)")

    if not returns:
        return 0.0

    weighted_variance = 0.0
    weight_sum = 0.0
    weight = 1.0

    for value in reversed(returns):
        squared = float(value) ** 2
        weighted_variance += weight * squared
        weight_sum += weight
        weight *= decay

    return sqrt(weighted_variance / max(weight_sum, EPS))


def _normalized_trend(
    prices: Sequence[float],
    lookback: int,
    volatility: float,
) -> float:
    """Return a volatility-normalized least-squares log-price slope in [-1, 1]."""

    if len(prices) < lookback:
        return 0.0

    segment = [log(float(value)) for value in prices[-lookback:]]
    x_mean = (lookback - 1) / 2.0
    y_mean = _mean(segment)

    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(segment)
    )
    denominator = sum(
        (index - x_mean) ** 2
        for index in range(lookback)
    )

    slope = numerator / max(denominator, EPS)
    scale = max(volatility, 0.002)

    return tanh((slope / scale) * 8.0)


def _directional_efficiency(
    prices: Sequence[float],
    lookback: int,
) -> float:
    """Return Kaufman-style directional efficiency in [0, 1]."""

    if len(prices) < lookback + 1:
        return 0.0

    segment = [
        float(value)
        for value in prices[-(lookback + 1):]
    ]

    net_change = abs(segment[-1] - segment[0])
    path_length = sum(
        abs(current - previous)
        for previous, current in zip(segment[:-1], segment[1:])
    )

    if path_length <= EPS:
        return 0.0

    return max(0.0, min(1.0, net_change / path_length))


def _directional_persistence(
    prices: Sequence[float],
    lookback: int,
) -> float:
    """Return signed persistence of recent return directions in [-1, 1]."""

    returns = _log_returns(prices)

    if len(returns) < lookback:
        return 0.0

    window = returns[-lookback:]
    signs = [
        1.0 if value > 0.0
        else -1.0 if value < 0.0
        else 0.0
        for value in window
    ]

    return _mean(signs)


def _volatility_ratio(
    current_volatility: float,
    historical_volatility: float,
) -> float:
    """Return current volatility divided by prior-window volatility."""

    if historical_volatility <= EPS:
        if current_volatility <= EPS:
            return 1.0
        return float("inf")

    ratio = current_volatility / historical_volatility

    if not isfinite(ratio):
        return float("inf")

    return ratio


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to [low, high]."""

    return max(low, min(high, value))


def classify(
    prices: Sequence[float],
    *,
    trend_window: int = 24,
    volatility_window: int = 20,
    efficiency_window: int = 20,
    persistence_window: int = 12,
    ewma_decay: float = 0.94,
    trend_threshold: float = 0.35,
    high_volatility_ratio: float = 1.50,
    low_volatility_ratio: float = 0.67,
    range_efficiency_threshold: float = 0.25,
) -> RegimeResult:
    """
    Classify the latest supplied market state.

    Current volatility is an EWMA estimate over the latest
    ``volatility_window`` returns. The reference volatility is the EWMA
    estimate over the immediately preceding return window of equal length.
    Thus current and historical volatility are non-overlapping.
    """

    _validate_integer_parameter(trend_window, "trend_window", 2)
    _validate_integer_parameter(volatility_window, "volatility_window", 2)
    _validate_integer_parameter(efficiency_window, "efficiency_window", 2)
    _validate_integer_parameter(persistence_window, "persistence_window", 1)

    if not 0.0 < ewma_decay < 1.0:
        raise RegimeEngineError("ewma_decay must be in (0, 1)")

    if not 0.0 < trend_threshold <= 1.0:
        raise RegimeEngineError("trend_threshold must be in (0, 1]")

    if high_volatility_ratio <= 1.0 or not isfinite(high_volatility_ratio):
        raise RegimeEngineError(
            "high_volatility_ratio must be finite and greater than 1"
        )

    if not 0.0 < low_volatility_ratio < 1.0:
        raise RegimeEngineError(
            "low_volatility_ratio must be in (0, 1)"
        )

    if not 0.0 < range_efficiency_threshold < 1.0:
        raise RegimeEngineError(
            "range_efficiency_threshold must be in (0, 1)"
        )

    minimum = max(
        trend_window,
        volatility_window * 2 + 1,
        efficiency_window + 1,
        persistence_window + 1,
        8,
    )

    normalized_prices = _validate_prices(prices, minimum)
    returns = _log_returns(normalized_prices)

    current_window = returns[-volatility_window:]
    historical_window = returns[-(2 * volatility_window):-volatility_window]

    current_volatility = _ewma_volatility(
        current_window,
        ewma_decay,
    )
    historical_volatility = _ewma_volatility(
        historical_window,
        ewma_decay,
    )
    volatility_ratio = _volatility_ratio(
        current_volatility,
        historical_volatility,
    )

    trend_score = _normalized_trend(
        normalized_prices,
        trend_window,
        current_volatility,
    )
    efficiency = _directional_efficiency(
        normalized_prices,
        efficiency_window,
    )
    persistence = _directional_persistence(
        normalized_prices,
        persistence_window,
    )

    high_volatility = volatility_ratio >= high_volatility_ratio
    low_volatility = volatility_ratio <= low_volatility_ratio

    strong_uptrend = (
        trend_score >= trend_threshold
        and persistence >= 0.20
        and efficiency >= range_efficiency_threshold
    )
    strong_downtrend = (
        trend_score <= -trend_threshold
        and persistence <= -0.20
        and efficiency >= range_efficiency_threshold
    )
    range_state = (
        abs(trend_score) < trend_threshold
        and efficiency < range_efficiency_threshold
    )

    reasons: list[str] = []

    if high_volatility:
        regime = MarketRegime.HIGH_VOLATILITY
        reasons.append("current-volatility-above-historical-baseline")
    elif low_volatility:
        regime = MarketRegime.LOW_VOLATILITY
        reasons.append("current-volatility-below-historical-baseline")
    elif strong_uptrend:
        regime = MarketRegime.TREND_UP
        reasons.extend([
            "persistent-upward-direction",
            "directional-efficiency-confirmed",
        ])
    elif strong_downtrend:
        regime = MarketRegime.TREND_DOWN
        reasons.extend([
            "persistent-downward-direction",
            "directional-efficiency-confirmed",
        ])
    elif range_state:
        regime = MarketRegime.RANGE
        reasons.append("low-directional-efficiency")
    else:
        regime = MarketRegime.TRANSITION
        reasons.append("trend-and-range-evidence-not-dominant")

    trend_strength = abs(trend_score)
    persistence_strength = abs(persistence)

    if isfinite(volatility_ratio):
        ratio_distance = abs(log(max(volatility_ratio, EPS)))
        volatility_stability = 1.0 / (1.0 + ratio_distance)
    else:
        ratio_distance = float("inf")
        volatility_stability = 0.0

    if regime in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN):
        confidence = (
            0.40 * trend_strength
            + 0.35 * efficiency
            + 0.25 * persistence_strength
        )
    elif regime == MarketRegime.RANGE:
        confidence = (
            0.55 * (
                1.0
                - min(
                    1.0,
                    efficiency / max(range_efficiency_threshold, EPS),
                )
            )
            + 0.45 * (1.0 - trend_strength)
        )
    elif regime in (MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY):
        if isfinite(ratio_distance):
            confidence = (
                0.50
                + 0.35 * tanh(ratio_distance)
                + 0.15 * volatility_stability
            )
        else:
            confidence = 0.85
    else:
        confidence = (
            0.45 * (1.0 - trend_strength)
            + 0.30 * (1.0 - efficiency)
            + 0.25 * volatility_stability
        )

    confidence = _clamp(confidence, 0.0, 1.0)

    if trend_strength >= 0.50:
        reasons.append("strong-normalized-trend")
    if efficiency >= 0.60:
        reasons.append("high-directional-efficiency")
    if efficiency <= 0.20:
        reasons.append("low-directional-efficiency")
    if persistence >= 0.50:
        reasons.append("positive-return-persistence")
    if persistence <= -0.50:
        reasons.append("negative-return-persistence")

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        trend_score=_clamp(trend_score, -1.0, 1.0),
        volatility=max(0.0, current_volatility),
        volatility_ratio=volatility_ratio,
        directional_efficiency=_clamp(efficiency, 0.0, 1.0),
        directional_persistence=_clamp(persistence, -1.0, 1.0),
        reasons=tuple(dict.fromkeys(reasons)),
    )


class RegimeEngine:
    """Thin stateless facade around the module-level Market Regime API."""

    def classify(self, prices: Sequence[float], **kwargs: object) -> RegimeResult:
        """Classify the latest supplied market state."""

        return classify(prices, **kwargs)

    def _self_test(self) -> bool:
        """Run the deterministic internal self-test."""

        _self_test()
        return True


def _self_test() -> bool:
    """Run deterministic smoke tests without external dependencies."""

    prices = [100.0 + 0.5 * index for index in range(80)]
    result = classify(prices)

    assert isinstance(result, RegimeResult)
    assert isinstance(result.regime, MarketRegime)
    assert 0.0 <= result.confidence <= 1.0
    assert -1.0 <= result.trend_score <= 1.0
    assert result.volatility >= 0.0
    assert 0.0 <= result.directional_efficiency <= 1.0
    assert -1.0 <= result.directional_persistence <= 1.0

    return True


if __name__ == "__main__":
    _self_test()
    print("Market Regime Engine self-test: PASS")
