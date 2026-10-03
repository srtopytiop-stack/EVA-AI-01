Phase 15-B — Integrated Research Diagnostics
Exact role
Phase 15-B composes the existing research stack without modifying the stable engines:
MarketDataResult -> System Integration -> Production Gate -> Monitoring
The new entry point is:
from core.phase15b_runtime import run_phase15b

result = run_phase15b(market_data)
The returned Phase15BResult contains:
research
production_gate
monitoring
status
release_ready
Safety boundary
This phase still does not:
send Telegram messages;
use Binance API credentials;
place exchange orders;
use leverage, margin, or shorting.
The future Notification Layer should consume Phase15BResult rather than bypassing the gate or monitoring layers.
Strict mode
from core.phase15b_runtime import Phase15BConfig, run_phase15b

result = run_phase15b(
    market_data,
    config=Phase15BConfig(strict=True),
)
Strict mode raises Phase15BRuntimeError when the combined result is not release-ready.
