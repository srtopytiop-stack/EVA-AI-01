
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.application_runtime as runtime
from core.application_runtime import (
    ApplicationRuntimeConfig,
    ApplicationRuntimeError,
    run_research_cycles,
)


def _config(
    symbols: tuple[str, ...] = ("ETHUSDT", "SOLUSDT"),
) -> ApplicationRuntimeConfig:
    return ApplicationRuntimeConfig(
        symbol=symbols[0],
        symbols=symbols,
        interval="1m",
        market_data_limit=240,
        minimum_candles=240,
        bootstrap_replications=200,
        bootstrap_block_length=5,
    )


def test_runtime_executes_each_configured_symbol_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    client = Mock()
    seen: list[str] = []

    def fake_cycle(
        single_config: ApplicationRuntimeConfig,
        *,
        market_client=None,
    ):
        seen.append(single_config.symbol)

        assert single_config.symbols == ()
        assert market_client is client

        return SimpleNamespace(
            symbol=single_config.symbol,
            client=market_client,
        )

    monkeypatch.setattr(
        runtime,
        "run_research_cycle",
        fake_cycle,
    )

    results = run_research_cycles(
        config,
        market_client=client,
    )

    assert seen == ["ETHUSDT", "SOLUSDT"]
    assert [result.symbol for result in results] == [
        "ETHUSDT",
        "SOLUSDT",
    ]


def test_runtime_preserves_single_symbol_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ApplicationRuntimeConfig(
        symbol="ETHUSDT",
        interval="1m",
        market_data_limit=240,
        minimum_candles=240,
    )

    monkeypatch.setattr(
        runtime,
        "run_research_cycle",
        lambda single_config, market_client=None: SimpleNamespace(
            symbol=single_config.symbol
        ),
    )

    results = run_research_cycles(config)

    assert len(results) == 1
    assert results[0].symbol == "ETHUSDT"


def test_environment_symbol_list_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "EVA_SYMBOLS",
        " ethusdt, solusdt ",
    )
    monkeypatch.delenv("EVA_SYMBOL", raising=False)

    config = ApplicationRuntimeConfig.from_environment()

    assert config.symbol == "ETHUSDT"
    assert config.effective_symbols == (
        "ETHUSDT",
        "SOLUSDT",
    )


def test_duplicate_symbols_are_rejected() -> None:
    with pytest.raises(
        ApplicationRuntimeError,
        match="duplicates",
    ):
        _config(("ETHUSDT", "ETHUSDT"))


def test_invalid_symbol_list_is_rejected() -> None:
    with pytest.raises(ApplicationRuntimeError):
        _config(("ETHUSDT", "SOL/USDT"))


def test_runtime_does_not_claim_full_batch_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    seen: list[str] = []

    def fake_cycle(
        single_config: ApplicationRuntimeConfig,
        *,
        market_client=None,
    ):
        seen.append(single_config.symbol)

        if single_config.symbol == "SOLUSDT":
            raise ApplicationRuntimeError(
                "simulated market-data failure"
            )

        return SimpleNamespace(
            symbol=single_config.symbol
        )

    monkeypatch.setattr(
        runtime,
        "run_research_cycle",
        fake_cycle,
    )

    with pytest.raises(
        ApplicationRuntimeError,
        match="simulated market-data failure",
    ):
        run_research_cycles(config)

    assert seen == ["ETHUSDT", "SOLUSDT"]
