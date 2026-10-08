"""Research data provenance contracts for EVA.

This module records where research data came from and which exact dataset
snapshot was used. It is asset-agnostic and intended for Binance Spot or
other research sources without coupling the contract to one symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Mapping, Any


class ProvenanceValidationError(ValueError):
    """Raised when a provenance record violates its invariants."""


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_utc_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ProvenanceValidationError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProvenanceValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def content_hash(rows: list[Mapping[str, Any]]) -> str:
    """Create a deterministic SHA-256 digest for a research row snapshot."""
    normalized = []
    for row in rows:
        normalized.append(sorted((str(k), repr(v)) for k, v in row.items()))
    payload = repr(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Immutable metadata describing one research dataset snapshot."""

    exchange: str
    market_type: str
    symbol: str
    interval: str
    source: str
    dataset_version: str
    retrieved_at: datetime
    start_at: datetime
    end_at: datetime
    row_count: int
    data_hash: str
    validation_status: str = "validated"
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.exchange, "exchange")
        _require_non_empty(self.market_type, "market_type")
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.interval, "interval")
        _require_non_empty(self.source, "source")
        _require_non_empty(self.dataset_version, "dataset_version")
        _require_non_empty(self.validation_status, "validation_status")
        _require_non_empty(self.data_hash, "data_hash")

        for name in ("retrieved_at", "start_at", "end_at"):
            _require_utc_datetime(getattr(self, name), name)

        if self.start_at >= self.end_at:
            raise ProvenanceValidationError("start_at must be earlier than end_at")
        if self.row_count < 0:
            raise ProvenanceValidationError("row_count must be non-negative")
        if len(self.data_hash) != 64 or any(c not in "0123456789abcdef" for c in self.data_hash.lower()):
            raise ProvenanceValidationError("data_hash must be a SHA-256 hexadecimal digest")

        if not isinstance(self.metadata, tuple):
            raise ProvenanceValidationError("metadata must be an immutable tuple")
        for item in self.metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ProvenanceValidationError("metadata entries must be (key, value) tuples")
            _require_non_empty(item[0], "metadata key")
            _require_non_empty(item[1], "metadata value")

    @property
    def asset_key(self) -> str:
        """Stable human-readable key for the research asset."""
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.interval}"

    def with_validation_status(self, status: str) -> "DatasetProvenance":
        """Return a new record with an updated validation status."""
        return DatasetProvenance(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            interval=self.interval,
            source=self.source,
            dataset_version=self.dataset_version,
            retrieved_at=self.retrieved_at,
            start_at=self.start_at,
            end_at=self.end_at,
            row_count=self.row_count,
            data_hash=self.data_hash,
            validation_status=status,
            metadata=self.metadata,
        )
