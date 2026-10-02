"""
EVA-AI-01 — Market Regime Engine
=================================

Phase 6: Market Regime Detection

Purpose
-------
Classify the current market environment using only information available
up to the current observation.

The engine is deliberately deterministic and interpretable. It provides
a baseline regime classifier before introducing probabilistic models such
as HMM/Markov-switching models.

Regimes
-------
TREND_UP
TREND_DOWN
RANGE
HIGH_VOLATILITY
LOW_VOLATILITY
TRANSITION

Design principles
-----------------
1. No look-ahead.
2. Deterministic output.
3. Explicit mathematical inputs.
4. No trading execution.
5. No BUY/SELL decision.
6. No leverage or margin assumptions.
7. Every result is auditable.
8. Inputs are validated strictly.
9. Regime classification is separate from signal generation.
10. Future probabilistic models can replace or augment this baseline.

Scientific basis
----------------
The implementation uses ideas related to:

- volatility clustering and conditional volatility;
- EWMA volatility estimation;
- normalized trend/momentum;
- realized volatility relative to a historical baseline;
- persistence and directional consistency.

This module is a research baseline, not a claim of predictive profitability.

A future HMM/Markov-switching implementation can consume the same observable
features and produce probabilistic regime estimates. Its performance must be
validated out-of-sample against this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log, sqrt, tanh
from statistics import mean, median
from typing import Sequence


EPS = 1e-12


class RegimeEngineError(ValueError):
    """Raised when regime-engine inputs are invalid."""


class MarketRegime(str, Enum):
    """Supported market regimes."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class RegimeResult:
    """
    Immutable and auditable market-regime result.
    """

    regime: MarketRegime
    confidence: float
    trend_score: float
    volatility: float
    volatility_ratio: float
    directional_efficiency: float
    persistence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "trend_score": self.trend_score,
            "volatility": self.volatility,
            "volatility_ratio": self.volatility_ratio,
            "directional_efficiency": self.directional_efficiency,
            "persistence": self.persistence,
            "reasons": list(self.reasons),
        }


def _validate_prices(
    prices: Sequence[float],
    minimum: int,
) -> None:
    """Validate positive finite price observations."""

    if len(prices) < minimum:
        raise RegimeEngineError(
            f"prices requires at least {minimum} observations"
        )

    for value in prices:
        value = float(value)

        if not isfinite(value):
            raise RegimeEngineError(
                "prices contains non-finite values"
            )

        if value <= 0:
            raise RegimeEngineError(
                "prices must be strictly positive"
            )


def _log_returns(
    prices: Sequence[float],
) -> list[float]:
    """Calculate log returns without using future observations."""

    returns: list[float] = []

    for previous, current in zip(prices[:-1], prices[1:]):
        previous = float(previous)
        current = float(current)

        returns.append(
            log(current / previous)
        )

    return returns


def _mean(
    values: Sequence[float],
) -> float:
    """Arithmetic mean."""

    if not values:
        return 0.0

    return mean(
        float(value)
        for value in values
    )


def _std(
    values: Sequence[float],
) -> float:
    """Sample standard deviation."""

    if len(values) < 2:
        return 0.0

    mu = _mean(values)

    variance = sum(
        (float(value) - mu) ** 2
        for value in values
    ) / (len(values) - 1)

    return sqrt(
        max(variance, 0.0)
    )


def _ewma_volatility(
    returns: Sequence[float],
    decay: float,
) -> float:
    """
    Calculate EWMA volatility.

    Recent observations receive greater weight.

    decay ∈ (0, 1)
    """

    if not 0.0 < decay < 1.0:
        raise RegimeEngineError(
            "ewma_decay must be in (0, 1)"
        )

    if not returns:
        return 0.0

    variance = 0.0
    weight_sum = 0.0
    weight = 1.0

    for value in reversed(returns):
        value = float(value)

        variance += (
            weight * value * value
        )

        weight_sum += weight
        weight *= decay

    return sqrt(
        variance / max(weight_sum, EPS)
    )


def _normalized_trend(
    prices: Sequence[float],
    lookback: int,
    volatility: float,
) -> float:
    """
    Estimate directional trend using a normalized log-price slope.

    Output:
        [-1, +1]

    Positive values indicate upward directional pressure.
    Negative values indicate downward directional pressure.
    """

    if lookback < 2:
        raise RegimeEngineError(
            "trend_window must be at least 2"
        )

    if len(prices) < lookback:
        return 0.0

    segment = [
        log(float(value))
        for value in prices[-lookback:]
    ]

    x_mean = (
        lookback - 1
    ) / 2.0

    y_mean = _mean(segment)

    numerator = sum(
        (
            index - x_mean
        )
        * (
            value - y_mean
        )
        for index, value in enumerate(segment)
    )

    denominator = sum(
        (
            index - x_mean
        ) ** 2
        for index in range(lookback)
    )

    slope = (
        numerator
        / max(denominator, EPS)
    )

    scale = max(
        volatility,
        0.002,
    )

    return tanh(
        slope / scale * 8.0
    )


def _directional_efficiency(
    prices: Sequence[float],
    lookback: int,
) -> float:
    """
    Calculate Kaufman-style directional efficiency.

    Efficiency Ratio:

        |net movement| / total path movement

    Output:
        [0, 1]

    A value close to 1 indicates persistent directional movement.
    A value close to 0 indicates a noisy/ranging path.
    """

    if lookback < 2:
        raise RegimeEngineError(
            "efficiency_window must be at least 2"
        )

    if len(prices) < lookback + 1:
        return 0.0

    segment = [
        float(value)
        for value in prices[-(lookback + 1):]
    ]

    net_change = abs(
        segment[-1] - segment[0]
    )

    path_length = sum(
        abs(current - previous)
        for previous, current in zip(
            segment[:-1],
            segment[1:],
        )
    )

    if path_length <= EPS:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            net_change / path_length,
        ),
    )


def _directional_persistence(
    prices: Sequence[float],
    lookback: int,
) -> float:
    """
    Estimate directional persistence.

    Returns:
        [-1, +1]

    Positive values mean upward returns dominate.
    Negative values mean downward returns dominate.
    """

    returns = _log_returns(prices)

    if len(returns) < lookback:
        return 0.0

    window = returns[-lookback:]

    signs = [
        1.0 if value > 0
        else -1.0 if value < 0
        else 0.0
        for value in window
    ]

    return _mean(signs)


def _rolling_volatility(
    returns: Sequence[float],
    window: int,
) -> float:
    """Calculate rolling standard deviation of returns."""

    if window < 2:
        raise RegimeEngineError(
            "volatility_window must be at least 2"
        )

    if len(returns) < window:
        return 0.0

    return _std(
        returns[-window:]
    )


def _historical_volatility(
    returns: Sequence[float],
    window: int,
) -> float:
    """
    Calculate a historical volatility baseline.

    The comparison window ends immediately before the current volatility
    window. This prevents the current volatility observation from defining
    its own reference baseline.
    """

    if len(returns) < window * 2:
        return 0.0

    reference = returns[
        -(window * 2):-window
    ]

    return _std(reference)


def _volatility_ratio(
    current_volatility: float,
    historical_volatility: float,
) -> float:
    """
    Current volatility divided by prior volatility baseline.
    """

    if historical_volatility <= EPS:
        if current_volatility <= EPS:
            return 1.0

        return float("inf")

    return (
        current_volatility
        / historical_volatility
    )


def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    """Clamp a numeric value."""

    return max(
        low,
        min(high, value),
    )


def classify(
    *,
    prices: Sequence[float],
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
    Classify the latest market environment.

    Important:
    This function uses only the supplied historical sequence and evaluates
    the last observation. It never accesses observations after the last
    supplied observation.

    Parameters
    ----------
    prices:
        Positive chronological price observations.

    trend_window:
        Number of observations used for normalized trend estimation.

    volatility_window:
        Number of returns used for current realized volatility.

    efficiency_window:
        Window used for directional efficiency.

    persistence_window:
        Window used for directional persistence.

    ewma_decay:
        EWMA decay parameter.

    trend_threshold:
        Minimum absolute trend score for a directional trend regime.

    high_volatility_ratio:
        Ratio above which volatility is considered elevated.

    low_volatility_ratio:
        Ratio below which volatility is considered compressed.

    range_efficiency_threshold:
        Efficiency below which price action is considered range-like.
    """

    if trend_window < 2:
        raise RegimeEngineError(
            "trend_window must be at least 2"
        )

    if volatility_window < 2:
        raise RegimeEngineError(
            "volatility_window must be at least 2"
        )

    if efficiency_window < 2:
        raise RegimeEngineError(
            "efficiency_window must be at least 2"
        )

    if persistence_window < 1:
        raise RegimeEngineError(
            "persistence_window must be at least 1"
        )

    if not 0.0 < ewma_decay < 1.0:
        raise RegimeEngineError(
            "ewma_decay must be in (0, 1)"
        )

    if not 0.0 < trend_threshold <= 1.0:
        raise RegimeEngineError(
            "trend_threshold must be in (0, 1]"
        )

    if high_volatility_ratio <= 1.0:
        raise RegimeEngineError(
            "high_volatility_ratio must be greater than 1"
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

    _validate_prices(
        prices,
        minimum,
    )

    returns = _log_returns(prices)

    current_volatility = _rolling_volatility(
        returns,
        volatility_window,
    )

    historical_volatility = _historical_volatility(
        returns,
        volatility_window,
    )

    volatility_ratio = _volatility_ratio(
        current_volatility,
        historical_volatility,
    )

    trend_score = _normalized_trend(
        prices,
        trend_window,
        current_volatility,
    )

    efficiency = _directional_efficiency(
        prices,
        efficiency_window,
    )

    persistence = _directional_persistence(
        prices,
        persistence_window,
    )

    reasons: list[str] = []

    # ---------------------------------------------------------------
    # 1. Volatility state
    # ---------------------------------------------------------------

    high_volatility = (
        volatility_ratio
        >= high_volatility_ratio
    )

    low_volatility = (
        volatility_ratio
        <= low_volatility_ratio
    )

    # ---------------------------------------------------------------
    # 2. Directional state
    # ---------------------------------------------------------------

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
        abs(trend_score)
        < trend_threshold
        and efficiency
        < range_efficiency_threshold
    )

    # ---------------------------------------------------------------
    # 3. Classification hierarchy
    #
    # Extreme volatility is separated first because it changes the
    # interpretation of ordinary trend/range statistics.
    # ---------------------------------------------------------------

    if high_volatility:
        regime = MarketRegime.HIGH_VOLATILITY
        reasons.append(
            "current-volatility-above-historical-baseline"
        )

    elif low_volatility:
        regime = MarketRegime.LOW_VOLATILITY
        reasons.append(
            "current-volatility-below-historical-baseline"
        )

    elif strong_uptrend:
        regime = MarketRegime.TREND_UP
        reasons.append(
            "persistent-upward-direction"
        )
        reasons.append(
            "directional-efficiency-confirmed"
        )

    elif strong_downtrend:
        regime = MarketRegime.TREND_DOWN
        reasons.append(
            "persistent-downward-direction"
        )
        reasons.append(
            "directional-efficiency-confirmed"
        )

    elif range_state:
        regime = MarketRegime.RANGE
        reasons.append(
            "low-directional-efficiency"
        )

    else:
        regime = MarketRegime.TRANSITION
        reasons.append(
            "trend-and-range-evidence-not-dominant"
        )

    # ---------------------------------------------------------------
    # 4. Confidence
    # ---------------------------------------------------------------

    trend_strength = abs(
        trend_score
    )

    persistence_strength = abs(
        persistence
    )

    volatility_stability = _clamp(
        1.0 / (
            1.0
            + abs(
                log(
                    max(
                        volatility_ratio,
                        EPS,
                    )
                )
            )
        ),
        0.0,
        1.0,
    )

    if regime in (
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
    ):
        confidence = (
            0.40 * trend_strength
            + 0.35 * efficiency
            + 0.25 * persistence_strength
        )

    elif regime == MarketRegime.RANGE:
        confidence = (
            0.55
            * (
                1.0
                - min(
                    1.0,
                    efficiency
                    / max(
                        range_efficiency_threshold,
                        EPS,
                    ),
                )
            )
            + 0.45
            * (
                1.0
                - trend_strength
            )
        )

    elif regime in (
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY,
    ):
        ratio_distance = (
            abs(
                log(
                    max(
                        volatility_ratio,
                        EPS,
                    )
                )
            )
        )

        confidence = _clamp(
            0.50
            + 0.35
            * tanh(
                ratio_distance
            )
            + 0.15
            * volatility_stability,
            0.0,
            1.0,
        )

    else:
        confidence = (
            0.45
            * (1.0 - trend_strength)
            + 0.30
            * (1.0 - efficiency)
            + 0.25
            * volatility_stability
        )

    confidence = _clamp(
        confidence,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------------
    # 5. Additional audit reasons
    # ---------------------------------------------------------------

    if trend_strength >= 0.50:
        reasons.append(
            "strong-normalized-trend"
        )

    if efficiency >= 0.60:
        reasons.append(
            "high-directional-efficiency"
        )

    if efficiency <= 0.20:
        reasons.append(
            "low-directional-efficiency"
        )

    if persistence >= 0.50:
        reasons.append(
            "positive-return-persistence"
        )

    if persistence <= -0.50:
        reasons.append(
            "negative-return-persistence"
        )

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        trend_score=_clamp(
            trend_score,
            -1.0,
            1.0,
        ),
        volatility=max(
            0.0,
            current_volatility,
        ),
        volatility_ratio=(
            volatility_ratio
            if isfinite(volatility_ratio)
            else float("inf")
        ),
        directional_efficiency=_clamp(
            efficiency,
            0.0,
            1.0,
        ),
        persistence=_clamp(
            persistence,
            -1.0,
            1.0,
        ),
        reasons=tuple(
            dict.fromkeys(reasons)
        ),
    )


def _self_test() -> None:
    """Deterministic internal smoke test."""

    prices = [
        100.0 + 0.5 * index
        for index in range(80)
    ]

    result = classify(
        prices=prices,
    )

    assert isinstance(
        result,
        RegimeResult,
    )

    assert isinstance(
        result.regime,
        MarketRegime,
    )

    assert 0.0 <= result.confidence <= 1.0

    assert -1.0 <= result.trend_score <= 1.0

    assert result.volatility >= 0.0

    assert 0.0 <= result.directional_efficiency <= 1.0

    assert -1.0 <= result.persistence <= 1.0


if __name__ == "__main__":
    _self_test()
    print("Market Regime Engine self-test: PASS")
