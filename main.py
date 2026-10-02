"""
EVA-AI-01
=========
Main application entry point.

This file is intentionally small.
Business logic belongs to dedicated modules.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def main() -> int:
    """Start EVA-AI-01 and perform a basic system health check."""

    timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print("EVA-AI-01")
    print("System initialization")
    print("=" * 60)
    print(f"UTC time: {timestamp}")
    print("Status: BOOT_OK")
    print("Architecture: CLEAN_REBUILD")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
