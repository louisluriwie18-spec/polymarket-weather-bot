"""
validate.py — Standalone Diagnostic & Validation Utility.

This script performs quick end-to-end sanity checks of the bot's core pipeline:
1. Validates temperature format parsing across all supported Celsius and Fahrenheit patterns.
2. Queries the live Polymarket Gamma API to verify market discovery and event filtering.
3. Tests tomorrow-target date matching logic.

This is a developer verification tool and does not place trades or mutate database state.
Run via:
    python validate.py
"""
import asyncio
import logging
import re
import sys
from datetime import date

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

async def main():
    from config import MAX_TRADES_PER_RUN
    from data import get_active_temperature_events
    from model import parse_group_item_title
    from main import parse_event_date, get_tomorrow

    tomorrow = get_tomorrow()
    today = date.today()
    
    print(f"{'='*70}")
    print(f"  FULL PIPELINE VALIDATION")
    print(f"  Today: {today} | Tomorrow (target): {tomorrow}")
    print(f"  Max trades per run: {MAX_TRADES_PER_RUN}")
    print(f"{'='*70}")

    # 1. Test parsing all temperature formats
    print("\n--- 1. TEMPERATURE FORMAT PARSING ---")
    formats = [
        "25°C", "24°C or below", "34°C or higher",
        "62-63°F", "78-79°F", "61°F or below", "80°F or higher",
        "30°C", "18°C or below", "58-59°F", "76-77°F",
        # Non-degree variants (also supported)
        "25C", "24C or below", "34C or higher",
    ]
    all_passed = True
    for fmt in formats:
        result = parse_group_item_title(fmt)
        if result:
            lo, hi, stype, unit = result
            lo_s = f"{lo:.1f}" if lo is not None else "-inf"
            hi_s = f"{hi:.1f}" if hi is not None else "+inf"
            print(f"  [OK] '{fmt:20s}' -> {stype:12s} [{lo_s}, {hi_s}]C  (orig: {unit})")
        else:
            print(f"  [FAIL] '{fmt}' -> PARSE FAILED")
            all_passed = False

    # 2. Discover events and filter for tomorrow
    print("\n--- 2. MARKET DISCOVERY ---")
    events = await get_active_temperature_events()
    print(f"  Total events found: {len(events)}")
    
    tomorrow_events = []
    for e in events:
        title = e.get("title", "")
        m = re.search(r"on\s+([A-Z][a-z]+\s+\d+)", title)
        date_str = m.group(1) if m else ""
        event_date = parse_event_date(date_str)
        if event_date == tomorrow:
            tomorrow_events.append(e)
    
    print(f"  Events for tomorrow ({tomorrow}): {len(tomorrow_events)}")
    for e in tomorrow_events:
        print(f"    -> {e.get('title', 'Unknown')}")

    # 3. Summary
    print(f"\n--- 3. VALIDATION SUMMARY ---")
    print(f"  Temperature format parsing: {'PASS' if all_passed else 'FAIL'}")
    print(f"  Total events found: {len(events)}")
    print(f"  Tomorrow's events: {len(tomorrow_events)}")
    print("  Validation complete.")

if __name__ == "__main__":
    asyncio.run(main())