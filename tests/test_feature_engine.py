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
       
