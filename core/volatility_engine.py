"""EVA-AI-01 - Volatility Engine

Phase 7: Volatility Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log, sqrt
from statistics import mean
from typing import Sequence


EPS = 1e-12


class VolatilityEngineError(ValueError):
    """Raised when Volatility Engine inputs are invalid."""


class VolatilityState(str, Enum):
    """Descriptive volatility states."""

    EXTREME = "EXTREME"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


@dataclass(frozen=True)
class VolatilityResult:
    """Immutable and auditable volatility measurement."""

    state: VolatilityState
    confidence: float
    ewma_volatility: float
    realized_volatility: float
    reference_volatility: float
    volatility_ratio: float
    volatility_of_volatility: float
    percentile_proxy: float
    reasons: tuple[str, ...]

    @property
    def volatility(self) -> float:
        """Primary volatility estimate."""

        return self.ewma_volatility

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return {
            "state": self.state.value,
            "confidence": self.confidence,
            "ewma_volatility": self.ewma_volatility,
            "realized_volatility": self.realized_volatility,
            "reference_volatility": self.reference_volatility,
            "volatility_ratio": self.volatility_ratio,
            "volatility_of_volatility": self.volatility_of_volatility,
            "percentile_proxy": self.percentile_proxy,
            "volatility": self.volatility,
            "reasons": list(self.reasons),
        }


def _validate_integer(
    value: int,
    name: str,
    minimum: int,
) -> None:
    """Validate an integer parameter."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise VolatilityEngineError(
            f"{name} must be an integer"
        )

    if value < minimum:
        raise VolatilityEngineError(
            f"{name} must be at least {minimum}"
        )


def _validate_prices(
    prices: Sequence[float],
    minimum: int,
) -> list[float]:
    """Validate positive finite prices."""

    if not isinstance(prices, Sequence):
        raise VolatilityEngineError(
            "prices must be a sequence"
        )

    if len(prices) < minimum:
        raise VolatilityEngineError(
            f"prices requires at least {minimum} observations"
        )

    normalized: list[float] = []

    for index, raw in enumerate(prices):
        if isinstance(raw, bool):
            raise VolatilityEngineError(
                f"prices[{index}] must be a real number"
            )

        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise VolatilityEngineError(
                f"prices[{index}] must be numeric"
            ) from exc

        if not isfinite(value):
            raise VolatilityEngineError(
                f"prices[{index}] must be finite"
            )

        if value <= 0.0:
            raise VolatilityEngineError(
                f"prices[{index}] must be strictly positive"
            )

        normalized.append(value)

    return normalized


def _log_returns(
    prices: Sequence[float],
) -> list[float]:
    """Calculate one-period logarithmic returns."""

    return [
        log(current / previous)
        for previous, current in zip(
            prices[:-1],
            prices[1:],
        )
    ]


def _mean(values: Sequence[float]) -> float:
    """Calculate an arithmetic mean."""

    if not values:
        return 0.0

    return mean(float(value) for value in values)


def _sample_std(
    values: Sequence[float],
) -> float:
    """Calculate sample standard deviation."""

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
    """Calculate EWMA volatility with greater weight on recent returns."""

    if not 0.0 < decay < 1.0:
        raise VolatilityEngineError(
            "ewma_decay must be in (0, 1)"
        )

    if not returns:
        return 0.0

    weighted_variance = 0.0
    weight_sum = 0.0
    weight = 1.0

    for value in reversed(returns):
        weighted_variance += (
            weight * float(value) ** 2
        )
        weight_sum += weight
        weight *= decay

    return sqrt(
        weighted_variance / max(weight_sum, EPS)
    )


def _volatility_ratio(
    current: float,
    reference: float,
) -> float:
    """Compare current volatility with the reference baseline."""

    if reference <= EPS:
        if current <= EPS:
            return 1.0

        return float("inf")

    ratio = current / reference

    if not isfinite(ratio):
        return float("inf")

    return ratio


def _rolling_volatility_series(
    returns: Sequence[float],
    window: int,
) -> list[float]:
    """Create a rolling volatility series."""

    if len(returns) < window:
        return []

    return [
        _sample_std(
            returns[index - window:index]
        )
        for index in range(
            window,
            len(returns) + 1,
        )
    ]


def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    """Clamp a value to a closed interval."""

    return max(low, min(high, value))


def _classify_ratio(
    ratio: float,
    *,
    extreme_threshold: float,
    high_threshold: float,
    low_threshold: float,
) -> VolatilityState:
    """Convert a volatility ratio into a state."""

    if (
        ratio == float("inf")
        or ratio >= extreme_threshold
    ):
        return VolatilityState.EXTREME

    if ratio >= high_threshold:
        return VolatilityState.HIGH

    if ratio <= low_threshold:
        return VolatilityState.LOW

    return VolatilityState.NORMAL


def classify(
    prices: Sequence[float],
    *,
    volatility_window: int = 20,
    reference_window: int = 40,
    ewma_decay: float = 0.94,
    high_ratio: float = 1.50,
    extreme_ratio: float = 2.50,
    low_ratio: float = 0.67,
) -> VolatilityResult:
    """
    Estimate the latest volatility state.

    The current and reference return windows are explicitly
    non-overlapping. This prevents the current volatility shock
    from contaminating its own historical baseline.
    """

    _validate_integer(
        volatility_window,
        "volatility_window",
        2,
    )

    _validate_integer(
        reference_window,
        "reference_window",
        2,
    )

    if not 0.0 < ewma_decay < 1.0:
        raise VolatilityEngineError(
            "ewma_decay must be in (0, 1)"
        )

    if (
        not isfinite(high_ratio)
        or high_ratio <= 1.0
    ):
        raise VolatilityEngineError(
            "high_ratio must be finite and greater than 1"
        )

    if (
        not isfinite(extreme_ratio)
        or extreme_ratio <= high_ratio
    ):
        raise VolatilityEngineError(
            "extreme_ratio must be finite and greater than high_ratio"
        )

    if (
        not isfinite(low_ratio)
        or not 0.0 < low_ratio < 1.0
    ):
        raise VolatilityEngineError(
            "low_ratio must be finite and in (0, 1)"
        )

    minimum = (
        volatility_window
        + reference_window
        + 1
    )

    normalized = _validate_prices(
        prices,
        minimum,
    )

    returns = _log_returns(normalized)

    current_returns = returns[
        -volatility_window:
    ]

    reference_returns = returns[
        -(volatility_window + reference_window):
        -volatility_window
    ]

    current_ewma = _ewma_volatility(
        current_returns,
        ewma_decay,
    )

    reference_ewma = _ewma_volatility(
        reference_returns,
        ewma_decay,
    )

    realized_current = _sample_std(
        current_returns
    )

    ratio = _volatility_ratio(
        current_ewma,
        reference_ewma,
    )

    rolling_series = _rolling_volatility_series(
        returns,
        volatility_window,
    )

    if len(rolling_series) >= 2:
        volatility_of_volatility = _sample_std(
            rolling_series
        )
    else:
        volatility_of_volatility = 0.0

    if rolling_series:
        minimum_volatility = min(
            rolling_series
        )

        maximum_volatility = max(
            rolling_series
        )

        span = (
            maximum_volatility
            - minimum_volatility
        )

        if span <= EPS:
            percentile_proxy = 0.5
        else:
            percentile_proxy = (
                rolling_series[-1]
                - minimum_volatility
            ) / span
    else:
        percentile_proxy = 0.5

    state = _classify_ratio(
        ratio,
        extreme_threshold=extreme_ratio,
        high_threshold=high_ratio,
        low_threshold=low_ratio,
    )

    reasons: list[str] = []

    if state == VolatilityState.EXTREME:
        reasons.append(
            "current-volatility-is-extreme-vs-reference"
        )
    elif state == VolatilityState.HIGH:
        reasons.append(
            "current-volatility-is-high-vs-reference"
        )
    elif state == VolatilityState.LOW:
        reasons.append(
            "current-volatility-is-low-vs-reference"
        )
    else:
        reasons.append(
            "current-volatility-near-reference"
        )

    if percentile_proxy >= 0.80:
        reasons.append(
            "recent-volatility-near-upper-historical-range"
        )
    elif percentile_proxy <= 0.20:
        reasons.append(
            "recent-volatility-near-lower-historical-range"
        )

    if ratio == float("inf"):
        confidence = 1.0
    else:
        separation = abs(
            log(max(ratio, EPS))
        )

        confidence = _clamp(
            0.50
            + 0.50
            * (
                1.0
                - 1.0 / (1.0 + separation)
            ),
            0.0,
            1.0,
        )

    if (
        volatility_of_volatility
        > max(current_ewma, EPS)
    ):
        reasons.append(
            "volatility-itself-is-unstable"
        )
        confidence *= 0.90

    confidence = _clamp(
        confidence,
        0.0,
        1.0,
    )

    return VolatilityResult(
        state=state,
        confidence=confidence,
        ewma_volatility=max(
            0.0,
            current_ewma,
        ),
        realized_volatility=max(
            0.0,
            realized_current,
        ),
        reference_volatility=max(
            0.0,
            reference_ewma,
        ),
        volatility_ratio=ratio,
        volatility_of_volatility=max(
            0.0,
            volatility_of_volatility,
        ),
        percentile_proxy=_clamp(
            percentile_proxy,
            0.0,
            1.0,
        ),
        reasons=tuple(
            dict.fromkeys(reasons)
        ),
    )


class VolatilityEngine:
    """Stateless facade for the volatility API."""

    def classify(
        self,
        prices: Sequence[float],
        **kwargs: object,
    ) -> VolatilityResult:
        """Classify the latest volatility state."""

        return classify(
            prices,
            **kwargs,
        )

    def _self_test(self) -> bool:
        """Run the deterministic internal test."""

        _self_test()
        return True


def _self_test() -> bool:
    """Run deterministic smoke tests."""

    prices = [100.0]

    for index in range(100):
        step = (
            0.01
            if index < 50
            else 0.001
        )

        direction = (
            1.0
            if index % 2 == 0
            else -1.0
        )

        prices.append(
            prices[-1]
            * (1.0 + direction * step)
        )

    result = classify(
        prices,
        volatility_window=20,
        reference_window=40,
    )

    assert isinstance(
        result,
        VolatilityResult,
    )

    assert isinstance(
        result.state,
        VolatilityState,
    )

    assert 0.0 <= result.confidence <= 1.0
    assert result.ewma_volatility >= 0.0
    assert result.realized_volatility >= 0.0
    assert result.reference_volatility >= 0.0
    assert result.volatility_ratio > 0.0
    assert result.volatility_of_volatility >= 0.0
    assert 0.0 <= result.percentile_proxy <= 1.0

    return True


if __name__ == "__main__":
    _self_test()
    print("Volatility Engine self-test: PASS")
