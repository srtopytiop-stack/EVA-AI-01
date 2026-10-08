import pytest

from core.asset_metadata import AssetMarketSnapshot, SpotAssetMetadata
from core.universe_selector import UniverseCriteria, UniverseSelectionError, select_universe


def metadata(symbol: str, status: str = "TRADING") -> SpotAssetMetadata:
    return SpotAssetMetadata(
        symbol=symbol,
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        status=status,
    )


def snapshot(symbol: str, volume: float, spread: float, quality: float = 1.0) -> AssetMarketSnapshot:
    return AssetMarketSnapshot(
        symbol=symbol,
        quote_volume=volume,
        median_spread_bps=spread,
        realized_volatility=0.04,
        observation_count=10_000,
        data_quality_score=quality,
    )


def test_selector_is_not_hard_coded_to_bitcoin():
    result = select_universe(
        [metadata("SOLUSDT"), metadata("ETHUSDT")],
        [snapshot("SOLUSDT", 8_000_000, 3), snapshot("ETHUSDT", 5_000_000, 4)],
        UniverseCriteria(max_assets=2),
    )

    assert [item.symbol for item in result] == ["SOLUSDT", "ETHUSDT"]


def test_low_quality_asset_is_filtered():
    result = select_universe(
        [metadata("BTCUSDT"), metadata("ADAUSDT")],
        [snapshot("BTCUSDT", 10_000_000, 2, 1.0), snapshot("ADAUSDT", 20_000_000, 1, 0.4)],
        UniverseCriteria(min_data_quality_score=0.8, max_assets=10),
    )

    assert [item.symbol for item in result] == ["BTCUSDT"]


def test_non_trading_symbol_is_filtered():
    result = select_universe(
        [metadata("BTCUSDT", status="BREAK"), metadata("ETHUSDT")],
        [snapshot("BTCUSDT", 50_000_000, 1), snapshot("ETHUSDT", 5_000_000, 3)],
    )

    assert [item.symbol for item in result] == ["ETHUSDT"]


def test_max_assets_limits_selected_universe():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    result = select_universe(
        [metadata(symbol) for symbol in symbols],
        [snapshot(symbol, 10_000_000 - index * 1_000_000, 2) for index, symbol in enumerate(symbols)],
        UniverseCriteria(max_assets=2),
    )

    assert len(result) == 2
    assert [item.rank for item in result] == [1, 2]


def test_duplicate_snapshot_is_rejected():
    with pytest.raises(UniverseSelectionError):
        select_universe(
            [metadata("ETHUSDT")],
            [snapshot("ETHUSDT", 10, 1), snapshot("ETHUSDT", 11, 1)],
        )
