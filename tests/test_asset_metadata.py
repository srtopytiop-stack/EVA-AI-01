import pytest

from core.asset_metadata import AssetMarketSnapshot, AssetMetadataError, SpotAssetMetadata


def test_spot_metadata_accepts_any_binance_symbol():
    item = SpotAssetMetadata(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        quantity_step=0.001,
        price_step=0.01,
    )

    assert item.is_trading is True
    assert item.exchange == "BINANCE"
    assert item.market_type == "SPOT"


def test_non_spot_market_is_rejected():
    with pytest.raises(AssetMetadataError):
        SpotAssetMetadata(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            market_type="FUTURES",
        )


def test_metadata_rejects_same_base_and_quote():
    with pytest.raises(AssetMetadataError):
        SpotAssetMetadata(symbol="USDTUSDT", base_asset="USDT", quote_asset="USDT")


def test_snapshot_quality_must_be_between_zero_and_one():
    with pytest.raises(AssetMetadataError):
        AssetMarketSnapshot(
            symbol="ETHUSDT",
            quote_volume=1_000_000,
            median_spread_bps=2,
            realized_volatility=0.5,
            observation_count=100,
            data_quality_score=1.1,
        )
