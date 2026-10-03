EVA-AI-01 — Phase 15: Monitoring & Diagnostics
Files
core/monitoring.py
tests/test_monitoring.py
Purpose
Phase 15 is an observational safety/diagnostics layer around the Phase-14 research result. It does not alter signal generation, portfolio sizing, execution simulation, Binance connectivity, or Telegram.
Checks
metadata validity;
equity-curve length and timestamps;
positive finite equity;
reported final_equity consistency;
performance-metric integrity;
statistical-report numeric integrity;
bootstrap observations/replications/block length;
bootstrap probability bounds;
trade chronology;
event chronology;
strict detection of NaN/Inf in auditable numeric fields.
Integration rule
Do not replace core/system_integration.py or core/production_gate.py in this phase. First add these two files and run the complete CI suite. After the suite passes, connect monitor_integration_result() to the gated runtime in a separate integration change.
Safety
This module contains no Binance credentials and cannot place live orders.
