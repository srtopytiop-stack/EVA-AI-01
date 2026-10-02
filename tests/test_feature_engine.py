"""
Tests for EVA-AI-01 Phase 4 Feature Engineering.
"""

import math

import numpy as np
import pandas as pd
import pytest

from features.feature_engine import (
    FeatureConfig,
    FeatureEngineeringError,
    build_features,
    feature_columns,
    validate_feature_output,
)


def make_ohlcv(rows: int = 30) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV data."""

    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="1min",
        tz="UTC",
    )

    close = np.arange(100.0, 100.0 + rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1000.0),
        }
    )


def test_build_features_returns_dataframe():
    df = make_ohlcv()

    result = build_features(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_input_dataframe_is_not_mutated():
    df = make_ohlcv()
    original = df.copy(deep=True)

    build_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_required_features_exist():
    df = make_ohlcv()

    result = build_features(df)

    expected = {
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

    assert expected.issubset(result.columns)


def test_first_return_is_nan():
    df = make_ohlcv()

    result = build_features(df)

    assert pd.isna(result.loc[0, "return"])
    assert pd.isna(result.loc[0, "log_return"])


def test_simple_return_is_correct():
    df = make_ohlcv()

    result = build_features(df)

    expected = (101.0 / 100.0) - 1.0

    assert math.isclose(
        result.loc[1, "return"],
        expected,
        rel_tol=1e-12,
    )


def test_log_return_is_correct():
    df = make_ohlcv()

    result = build_features(df)

    expected = math.log(101.0 / 100.0)

    assert math.isclose(
        result.loc[1, "log_return"],
        expected,
        rel_tol=1e-12,
    )


def test_candle_range():
    df = make_ohlcv()

    result = build_features(df)

    assert (result["range"] == 2.0).all()


def test_body_is_correct():
    df = make_ohlcv()

    result = build_features(df)

    assert (result["body"] == 0.5).all()


def test_wicks_are_non_negative():
    df = make_ohlcv()

    result = build_features(df)

    assert (result["upper_wick"] >= 0).all()
    assert (result["lower_wick"] >= 0).all()


def test_candle_direction():
    df = make_ohlcv()

    result = build_features(df)

    assert set(result["candle_direction"].unique()) <= {
        -1,
        0,
        1,
    }


def test_close_position_is_between_zero_and_one():
    df = make_ohlcv()

    result = build_features(df)

    assert (
        (result["close_position"] >= 0)
        & (result["close_position"] <= 1)
    ).all()


def test_true_range_first_row_equals_candle_range():
    df = make_ohlcv()

    result = build_features(df)

    assert math.isclose(
        result.loc[0, "true_range"],
        result.loc[0, "range"],
        rel_tol=1e-12,
    )


def test_rolling_features_respect_window():
    df = make_ohlcv()

    config = FeatureConfig(
        volatility_window=5,
        volume_window=5,
        momentum_window=5,
    )

    result = build_features(df, config)

    # Volatility is calculated from log returns.
    # The first log return is NaN because there is no
    # previous closing price.
    #
    # Therefore, a 5-observation volatility window
    # becomes valid for the first time at index 5.
    assert result["volatility"].iloc[:5].isna().all()
    assert result["volatility"].iloc[5:].notna().all()

    # Volume has no initial NaN values.
    # Therefore, a 5-observation rolling mean becomes
    # valid for the first time at index 4.
    assert result["volume_mean"].iloc[:4].isna().all()
    assert result["volume_mean"].iloc[4:].notna().all()

    # Momentum with a 5-period shift requires five
    # previous observations, so the first valid value
    # is at index 5.
    assert result["momentum"].iloc[:5].isna().all()
    assert result["momentum"].iloc[5:].notna().all()


def test_constant_volume_ratio_is_one():
    df = make_ohlcv()

    result = build_features(df)

    valid = result["volume_ratio"].dropna()

    assert np.allclose(valid, 1.0)


def test_feature_columns_excludes_ohlcv_columns():
    df = make_ohlcv()

    result = build_features(df)

    columns = feature_columns(result)

    for column in (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        assert column not in columns

    assert "return" in columns
    assert "volatility" in columns


def test_validate_feature_output():
    df = make_ohlcv()

    result = build_features(df)

    assert validate_feature_output(result) is True


def test_missing_column_is_rejected():
    df = make_ohlcv().drop(columns=["volume"])

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_negative_volume_is_rejected():
    df = make_ohlcv()
    df.loc[5, "volume"] = -1.0

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_invalid_ohlc_relationship_is_rejected():
    df = make_ohlcv()

    df.loc[5, "high"] = df.loc[5, "low"] - 1.0

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_invalid_timestamp_order_is_rejected():
    df = make_ohlcv()

    df.loc[5, "timestamp"] = df.loc[4, "timestamp"]

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_duplicate_timestamp_is_rejected():
    df = make_ohlcv()

    df.loc[6, "timestamp"] = df.loc[5, "timestamp"]

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_nan_price_is_rejected():
    df = make_ohlcv()

    df.loc[5, "close"] = np.nan

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_infinite_price_is_rejected():
    df = make_ohlcv()

    df.loc[5, "close"] = np.inf

    with pytest.raises(FeatureEngineeringError):
        build_features(df)


def test_invalid_window_is_rejected():
    with pytest.raises(FeatureEngineeringError):
        FeatureConfig(volatility_window=1)


def test_custom_windows_are_supported():
    df = make_ohlcv(50)

    config = FeatureConfig(
        volatility_window=10,
        volume_window=7,
        momentum_window=6,
    )

    result = build_features(df, config)

    # Volatility uses log returns.
    # The first log return is NaN, so a 10-observation
    # volatility window first becomes valid at index 10.
    assert result["volatility"].iloc[:10].isna().all()
    assert result["volatility"].iloc[10:].notna().all()

    # Volume has no initial NaN values, so a 7-observation
    # rolling mean first becomes valid at index 6.
    assert result["volume_mean"].iloc[:6].isna().all()
    assert result["volume_mean"].iloc[6:].notna().all()

    # Momentum with a 6-period shift first becomes valid
    # at index 6.
    assert result["momentum"].iloc[:6].isna().all()
    assert result["momentum"].iloc[6:].notna().all()


def test_no_infinite_generated_features():
    df = make_ohlcv()

    result = build_features(df)

    numeric_features = result[
        feature_columns(result)
    ].select_dtypes(include=[np.number])

    finite_values = np.isfinite(
        numeric_features.dropna().to_numpy()
    )

    assert finite_values.all()
