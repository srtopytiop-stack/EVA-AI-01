from datetime import datetime, timezone

import pytest

from core.research_provenance import DatasetProvenance, ProvenanceValidationError, content_hash


UTC = timezone.utc


def make_record(**overrides):
    values = dict(
        exchange="BINANCE",
        market_type="SPOT",
        symbol="ETHUSDT",
        interval="1h",
        source="binance_public_klines",
        dataset_version="v1",
        retrieved_at=datetime(2026, 10, 8, 12, tzinfo=UTC),
        start_at=datetime(2026, 10, 1, tzinfo=UTC),
        end_at=datetime(2026, 10, 8, tzinfo=UTC),
        row_count=168,
        data_hash="a" * 64,
    )
    values.update(overrides)
    return DatasetProvenance(**values)


def test_provenance_is_asset_agnostic_and_has_stable_asset_key():
    record = make_record(symbol="SOLUSDT", interval="15m")

    assert record.symbol == "SOLUSDT"
    assert record.asset_key == "BINANCE:SPOT:SOLUSDT:15m"


def test_provenance_is_immutable():
    record = make_record()

    with pytest.raises(AttributeError):
        record.symbol = "BTCUSDT"


def test_invalid_time_range_is_rejected():
    with pytest.raises(ProvenanceValidationError):
        make_record(
            start_at=datetime(2026, 10, 8, tzinfo=UTC),
            end_at=datetime(2026, 10, 7, tzinfo=UTC),
        )


def test_naive_datetime_is_rejected():
    with pytest.raises(ProvenanceValidationError):
        make_record(retrieved_at=datetime(2026, 10, 8, 12))


def test_hash_is_deterministic():
    rows = [{"time": 1, "close": 100.0}, {"time": 2, "close": 101.0}]

    assert content_hash(rows) == content_hash(list(rows))
    assert len(content_hash(rows)) == 64


def test_hash_rejects_non_sha256_length():
    with pytest.raises(ProvenanceValidationError):
        make_record(data_hash="abc")
