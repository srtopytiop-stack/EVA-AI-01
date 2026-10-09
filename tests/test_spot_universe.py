
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from data.spot_universe import (
    BinanceSpotUniverseClient,
    SpotUniverseData,
    SpotUniverseDataError,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = "fake response"

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))

        for path, payload in self.responses.items():
            if url.endswith(path):
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)

        raise AssertionError(f"Unexpected endpoint: {url}")


def _exchange_info():
    return {
        "symbols": [
            {
                "symbol": "ETHUSDT",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
            },
            {
                "symbol": "SOLUSDT",
                "baseAsset": "SOL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
            },
            {
                "symbol": "OLDUSDT",
                "baseAsset": "OLD",
                "quoteAsset": "USDT",
                "status": "BREAK",
                "isSpotTradingAllowed": True,
            },
            {
                "symbol": "ETHUSDC",
                "baseAsset": "ETH",
                "quoteAsset": "USDC",
                "status": "TRADING",
                "isSpotTradingAllowed": True,
            },
        ]
    }


def _tickers():
    return [
        {
            "symbol": "ETHUSDT",
            "quoteVolume": "9000000",
        },
        {
            "symbol": "SOLUSDT",
            "quoteVolume": "5000000",
        },
        {
            "symbol": "OLDUSDT",
            "quoteVolume": "100000000",
        },
        {
            "symbol": "ETHUSDC",
            "quoteVolume": "20000000",
        },
    ]


def _book_tickers():
    return [
        {
            "symbol": "ETHUSDT",
            "bidPrice": "2000",
            "askPrice": "2000.2",
        },
        {
            "symbol": "SOLUSDT",
            "bidPrice": "100",
            "askPrice": "100.2",
        },
        {
            "symbol": "OLDUSDT",
            "bidPrice": "1",
            "askPrice": "1.01",
        },
        {
            "symbol": "ETHUSDC",
            "bidPrice": "2000",
            "askPrice": "2000.2",
        },
    ]


def _client(
    *,
    exchange_info=None,
    tickers=None,
    book_tickers=None,
    market_client=None,
):
    session = FakeSession(
        {
            "/api/v3/exchangeInfo": (
                _exchange_info()
                if exchange_info is None
                else exchange_info
            ),
            "/api/v3/ticker/24hr": (
                _tickers() if tickers is None else tickers
            ),
            "/api/v3/ticker/bookTicker": (
                _book_tickers()
                if book_tickers is None
                else book_tickers
            ),
        }
    )

    client = BinanceSpotUniverseClient(
        session=session,
        market_data_client=market_client or Mock(),
    )

    return client, session


def test_discovery_uses_only_active_spot_quote_pairs():
    client, _ = _client()

    result = client.discover(
        candidate_limit=10,
        candles_per_symbol=30,
    )

    symbols = {item.symbol for item in result.metadata}

    assert symbols == {"ETHUSDT", "SOLUSDT"}
    assert all(item.quote_asset == "USDT" for item in result.metadata)
    assert all(item.market_type == "SPOT" for item in result.metadata)


def test_candidates_are_limited_by_quote_volume():
    client, _ = _client()

    result = client.discover(
        candidate_limit=1,
        candles_per_symbol=30,
    )

    assert result.candidates_examined == 1
    assert [item.symbol for item in result.metadata] == ["ETHUSDT"]


def test_invalid_bid_ask_pair_is_excluded():
    books = [
        {
            "symbol": "ETHUSDT",
            "bidPrice": "2001",
            "askPrice": "2000",
        },
        {
            "symbol": "SOLUSDT",
            "bidPrice": "100",
            "askPrice": "100.2",
        },
    ]

    client, _ = _client(book_tickers=books)

    result = client.discover(
        candidate_limit=10,
        candles_per_symbol=30,
    )

    assert [item.symbol for item in result.metadata] == ["SOLUSDT"]


def test_missing_ticker_volume_is_excluded():
    client, _ = _client(
        tickers=[
            {
                "symbol": "ETHUSDT",
                "quoteVolume": "NaN",
            },
            {
                "symbol": "SOLUSDT",
                "quoteVolume": "5000000",
            },
        ]
    )

    result = client.discover(
        candidate_limit=10,
        candles_per_symbol=30,
    )

    assert [item.symbol for item in result.metadata] == ["SOLUSDT"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"symbols": "invalid"},
    ],
)
def test_invalid_exchange_info_is_rejected(payload):
    client, _ = _client(exchange_info=payload)

    with pytest.raises(SpotUniverseDataError):
        client.discover()


def test_http_failure_is_reported():
    session = FakeSession(
        {
            "/api/v3/exchangeInfo": requests.ConnectionError(
                "network unavailable"
            ),
        }
    )

    client = BinanceSpotUniverseClient(session=session)

    with pytest.raises(
        SpotUniverseDataError,
        match="request failed",
    ):
        client.discover()


def test_spot_universe_rejects_duplicate_metadata():
    from core.asset_metadata import SpotAssetMetadata

    metadata = SpotAssetMetadata(
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
    )

    with pytest.raises(
        SpotUniverseDataError,
        match="duplicate",
    ):
        SpotUniverseData(
            metadata=(metadata, metadata),
            snapshots=(),
            candidates_examined=0,
        )


@pytest.mark.parametrize(
    "candidate_limit",
    [0, -1, 101, True],
)
def test_candidate_limit_is_validated(candidate_limit):
    client, _ = _client()

    with pytest.raises(SpotUniverseDataError):
        client.discover(candidate_limit=candidate_limit)


@pytest.mark.parametrize(
    "candles_per_symbol",
    [29, 1001, True],
)
def test_candle_count_is_validated(candles_per_symbol):
    client, _ = _client()

    with pytest.raises(SpotUniverseDataError):
        client.discover(candles_per_symbol=candles_per_symbol)
