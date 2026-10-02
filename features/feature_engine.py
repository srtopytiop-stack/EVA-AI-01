"""
EVA-AI-01
=========
Phase 4: Feature Engineering

Purpose
-------
Convert validated OHLCV market data into deterministic numerical features.

Design principles
-----------------
1. No look-ahead bias.
2. Features at time t use only data available at or before t.
3. No trading decisions are made here.
4. No API calls.
5. No exchange credentials.
6. No order execution.
7. Explicit validation of numerical inputs.
8. NaN values are preserved where a rolling feature does not yet
   have enough historical observations.

Initial feature groups
----------------------
- Simple return
- Log return
- Candle range
- Body size
- Upper/lower wick
- True range
- Rolling volatility
- Rolling volume statistics
- Price position inside candle
- Rolling momentum
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class FeatureEngineeringError(ValueError):
    """Raised when feature-engineering input is invalid."""


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for deterministic feature generation.

    Parameters
    ----------
    volatility_window:
        Number of observations used for rolling return volatility.

    volume_window:
        Number of observations used for rolling volume statistics.

    momentum_window:
        Number of observations used for price momentum.
    """

    volatility_window: int = 20
    volume_window: int = 20
    momentum_window: int = 10

    def __post_init__(self) -> None:
        for name, value in (
            ("volatility_window", self.volatility_window),
            ("volume_window", self.volume_window),
            ("momentum_window", self.momentum_window),
        ):
            if not isinstance(value, int):
                raise FeatureEngineeringError(
                    f"{name} must be an integer."
                )

            if value < 2:
                raise FeatureEngineeringError(
                    f"{name} must be >= 2."
                )


REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _validate_input_frame(df: pd.DataFrame) -> None:
    """Validate the minimum OHLCV structure."""

    if not isinstance(df, pd.DataFrame):
        raise FeatureEngineeringError(
            "Input must be a pandas DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise FeatureEngineeringError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise FeatureEngineeringError(
            "Input DataFrame must not be empty."
        )

    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")

        if values.isna().any():
            raise FeatureEngineeringError(
                f"Column '{column}' contains non-numeric or NaN values."
            )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise FeatureEngineeringError(
                f"Column '{column}' contains infinite values."
            )

    if (df["open"] <= 0).any():
        raise FeatureEngineeringError(
            "Open prices must be positive."
        )

    if (df["high"] <= 0).any():
        raise FeatureEngineeringError(
            "High prices must be positive."
        )

    if (df["low"] <= 0).any():
        raise FeatureEngineeringError(
            "Low prices must be positive."
        )

    if (df["close"] <= 0).any():
        raise FeatureEngineeringError(
            "Close prices must be positive."
        )

    if (df["volume"] < 0).any():
        raise FeatureEngineeringError(
            "Volume cannot be negative."
        )

    if (df["high"] < df["low"]).any():
        raise FeatureEngineeringError(
            "High price cannot be lower than low price."
        )

    if (df["open"] < df["low"]).any() or (
        df["open"] > df["high"]
    ).any():
        raise FeatureEngineeringError(
            "Open price must lie inside the candle range."
        )

    if (df["close"] < df["low"]).any() or (
        df["close"] > df["high"]
    ).any():
        raise FeatureEngineeringError(
            "Close price must lie inside the candle range."
        )


def _validate_timestamps(df: pd.DataFrame) -> None:
    """Validate chronological ordering."""

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise FeatureEngineeringError(
            "Timestamp column contains invalid values."
        )

    if not timestamps.is_monotonic_increasing:
        raise FeatureEngineeringError(
            "Timestamps must be monotonically increasing."
        )

    if timestamps.duplicated().any():
        raise FeatureEngineeringError(
            "Duplicate timestamps are not allowed."
        )


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Wilder-style True Range.

    TR_t =
        max(
            High_t - Low_t,
            |High_t - Close_(t-1)|,
            |Low_t - Close_(t-1)|
        )

    For the first observation, previous close is unavailable,
    therefore High - Low is used.
    """

    previous_close = df["close"].shift(1)

    current_range = df["high"] - df["low"]

    high_gap = (
        df["high"] - previous_close
    ).abs()

    low_gap = (
        df["low"] - previous_close
    ).abs()

    return pd.concat(
        [current_range, high_gap, low_gap],
        axis=1,
    ).max(axis=1)


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build deterministic OHLCV-derived features.

    Parameters
    ----------
    df:
        DataFrame containing:
        timestamp, open, high, low, close, volume.

    config:
        Feature-generation configuration.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame plus engineered features.

    Notes
    -----
    Rolling calculations use only current and historical observations.
    No future rows are accessed.
    """

    if config is None:
        config = FeatureConfig()

    _validate_input_frame(df)
    _validate_timestamps(df)

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    # ------------------------------------------------------------
    # 1. Simple return
    # ------------------------------------------------------------
    #
    # R_t = Close_t / Close_(t-1) - 1
    #
    result["return"] = (
        result["close"]
        .div(result["close"].shift(1))
        .sub(1.0)
    )

    # ------------------------------------------------------------
    # 2. Log return
    # ------------------------------------------------------------
    #
    # r_t = ln(C_t / C_(t-1))
    #
    result["log_return"] = np.log(
        result["close"]
        .div(result["close"].shift(1))
    )

    # ------------------------------------------------------------
    # 3. Candle range
    # ------------------------------------------------------------
    #
    # Range_t = High_t - Low_t
    #
    result["range"] = (
        result["high"] - result["low"]
    )

    # ------------------------------------------------------------
    # 4. Normalized candle range
    # ------------------------------------------------------------
    #
    # Normalized range makes the feature more comparable
    # across different price levels.
    #
    # NR_t = (High_t - Low_t) / Close_t
    #
    result["range_pct"] = (
        result["range"]
        .div(result["close"])
    )

    # ------------------------------------------------------------
    # 5. Candle body
    # ------------------------------------------------------------
    #
    # Body_t = |Close_t - Open_t|
    #
    result["body"] = (
        result["close"] - result["open"]
    ).abs()

    result["body_pct"] = (
        result["body"]
        .div(result["close"])
    )

    # ------------------------------------------------------------
    # 6. Upper wick
    # ------------------------------------------------------------
    #
    # UpperWick =
    #     High - max(Open, Close)
    #
    result["upper_wick"] = (
        result["high"]
        - result[["open", "close"]].max(axis=1)
    )

    # ------------------------------------------------------------
    # 7. Lower wick
    # ------------------------------------------------------------
    #
    # LowerWick =
    #     min(Open, Close) - Low
    #
    result["lower_wick"] = (
        result[["open", "close"]].min(axis=1)
        - result["low"]
    )

    # ------------------------------------------------------------
    # 8. Candle direction
    # ------------------------------------------------------------
    #
    # +1 bullish
    #  0 neutral
    # -1 bearish
    #
    result["candle_direction"] = np.sign(
        result["close"] - result["open"]
    ).astype(int)

    # ------------------------------------------------------------
    # 9. Close position inside candle
    # ------------------------------------------------------------
    #
    # Position =
    #     (Close - Low) / (High - Low)
    #
    # A zero-range candle is handled explicitly.
    #
    candle_range = result["high"] - result["low"]

    result["close_position"] = np.where(
        candle_range > 0,
        (result["close"] - result["low"])
        / candle_range,
        0.5,
    )

    # ------------------------------------------------------------
    # 10. True Range
    # ------------------------------------------------------------
    result["true_range"] = _true_range(result)

    result["true_range_pct"] = (
        result["true_range"]
        .div(result["close"])
    )

    # ------------------------------------------------------------
    # 11. Rolling volatility
    # ------------------------------------------------------------
    #
    # Standard deviation of log returns.
    #
    result["volatility"] = (
        result["log_return"]
        .rolling(
            window=config.volatility_window,
            min_periods=config.volatility_window,
        )
        .std(ddof=1)
    )

    # ------------------------------------------------------------
    # 12. Rolling mean volume
    # ------------------------------------------------------------
    result["volume_mean"] = (
        result["volume"]
        .rolling(
            window=config.volume_window,
            min_periods=config.volume_window,
        )
        .mean()
    )

    # ------------------------------------------------------------
    # 13. Volume ratio
    # ------------------------------------------------------------
    #
    # VR_t = Volume_t / MeanVolume_t
    #
    result["volume_ratio"] = (
        result["volume"]
        .div(result["volume_mean"])
    )

    # ------------------------------------------------------------
    # 14. Rolling volume standard deviation
    # ------------------------------------------------------------
    result["volume_std"] = (
        result["volume"]
        .rolling(
            window=config.volume_window,
            min_periods=config.volume_window,
        )
        .std(ddof=1)
    )

    # ------------------------------------------------------------
    # 15. Momentum
    # ------------------------------------------------------------
    #
    # M_t = Close_t / Close_(t-n) - 1
    #
    result["momentum"] = (
        result["close"]
        .div(
            result["close"].shift(
                config.momentum_window
            )
        )
        .sub(1.0)
    )

    # ------------------------------------------------------------
    # 16. Rolling price mean
    # ------------------------------------------------------------
    result["price_mean"] = (
        result["close"]
        .rolling(
            window=config.momentum_window,
            min_periods=config.momentum_window,
        )
        .mean()
    )

    # ------------------------------------------------------------
    # 17. Price distance from rolling mean
    # ------------------------------------------------------------
    result["price_distance_from_mean"] = (
        result["close"]
        .div(result["price_mean"])
        .sub(1.0)
    )

    # ------------------------------------------------------------
    # Final numerical sanity check
    # ------------------------------------------------------------
    feature_columns = [
        column
        for column in result.columns
        if column not in REQUIRED_COLUMNS
    ]

    for column in feature_columns:
        values = result[column]

        numeric_values = pd.to_numeric(
            values,
            errors="coerce",
        )

        finite_mask = numeric_values.notna()

        if not np.isfinite(
            numeric_values[finite_mask]
            .to_numpy(dtype=float)
        ).all():
            raise FeatureEngineeringError(
                f"Feature '{column}' contains infinite values."
            )

    return result


def feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return the names of engineered features in a DataFrame.
    """

    return [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]


def validate_feature_output(
    df: pd.DataFrame,
) -> bool:
    """
    Validate structural invariants of a generated feature DataFrame.

    Returns
    -------
    bool
        True when all invariants hold.

    Raises
    ------
    FeatureEngineeringError
        If a feature invariant is violated.
    """

    if not isinstance(df, pd.DataFrame):
        raise FeatureEngineeringError(
            "Feature output must be a pandas DataFrame."
        )

    required_output_features = {
        "return",
        "log_return",
        "range",
        "range_pct",
        "body",
        "body_pct",
        "upper_wick",
        "lower_wick",
        "candle_direction",
        "close_position",
        "true_range",
        "true_range_pct",
        "volatility",
        "volume_mean",
        "volume_ratio",
        "volume_std",
        "momentum",
        "price_mean",
        "price_distance_from_mean",
    }

    missing = required_output_features.difference(df.columns)

    if missing:
        raise FeatureEngineeringError(
            f"Missing generated features: {sorted(missing)}"
        )

    if (df["range"] < 0).any():
        raise FeatureEngineeringError(
            "Candle range cannot be negative."
        )

    if (df["body"] < 0).any():
        raise FeatureEngineeringError(
            "Candle body cannot be negative."
        )

    if (df["upper_wick"] < 0).any():
        raise FeatureEngineeringError(
            "Upper wick cannot be negative."
        )

    if (df["lower_wick"] < 0).any():
        raise FeatureEngineeringError(
            "Lower wick cannot be negative."
        )

    if not df["candle_direction"].isin(
        [-1, 0, 1]
    ).all():
        raise FeatureEngineeringError(
            "Candle direction must be -1, 0, or 1."
        )

    valid_positions = df["close_position"].dropna()

    if not (
        (valid_positions >= 0)
        & (valid_positions <= 1)
    ).all():
        raise FeatureEngineeringError(
            "Close position must be between 0 and 1."
        )

    valid_true_range = df["true_range"].dropna()

    if (valid_true_range < 0).any():
        raise FeatureEngineeringError(
            "True range cannot be negative."
        )

    return True
