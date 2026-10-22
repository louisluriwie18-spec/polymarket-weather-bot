"""
trade_logger.py — Orderly Human-Readable Text Logger for Past Trades.

Maintains a permanent, cleanly-formatted text log (`past_trades.txt`)
of every trade recommended and resolved by the bot, separate from the SQLite database.
"""

import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "past_trades.txt")


def _init_log_file():
    """Ensure the text log exists with an informative header banner."""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("                POLYMARKET WEATHER BOT — PAST TRADES AUDIT LOG\n")
            f.write("  Independent human-readable ledger of all trade recommendations & resolutions.\n")
            f.write("=" * 80 + "\n\n")


def log_trade_text(candidate, stake: float, order_id: str, status: str = "OPEN") -> None:
    """
    Append a formatted trade entry to past_trades.txt.
    """
    _init_log_file()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    entry = (
        f"┌{'─' * 78}┐\n"
        f"│ [TRADE RECOMMENDATION]  {now_str:<53}│\n"
        f"├{'─' * 78}┤\n"
        f"│ Order ID:       {order_id:<61}│\n"
        f"│ Market ID:      {str(candidate.market_id):<61}│\n"
        f"│ Condition ID:   {str(getattr(candidate, 'condition_id', 'N/A')):<61}│\n"
        f"│ Event:          {candidate.event_title[:61]:<61}│\n"
        f"│ Question:       {candidate.question[:61]:<61}│\n"
        f"│ Action:         BUY {candidate.side:<57}│\n"
        f"│ Model Prob:     {candidate.our_prob * 100:>5.1f}%{'':<55}│\n"
        f"│ Market Price:   ${candidate.mkt_price:>5.3f} ({candidate.mkt_price * 100:.1f}%){'':<46}│\n"
        f"│ Edge Detected:  {candidate.edge * 100:>+5.1f}%{'':<55}│\n"
        f"│ Stake:          ${stake:>5.2f} pUSD{'':<51}│\n"
        f"│ Status:         {status:<61}│\n"
        f"└{'─' * 78}┘\n\n"
    )

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def log_resolution_text(market_id: str, status: str, pnl: float, winner: str | None = None) -> None:
    """
    Append a formatted resolution entry to past_trades.txt.
    """
    _init_log_file()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    pnl_sign = "+" if pnl > 0 else ""
    pnl_str = f"{pnl_sign}${pnl:.2f} pUSD"
    winner_str = winner if winner else "N/A"

    entry = (
        f"  ├{'┄' * 76}┤\n"
        f"  │ >>> [RESOLUTION EVENT]  {now_str:<47}│\n"
        f"  │ Market ID:       {str(market_id):<57}│\n"
        f"  │ Outcome:         {status:<57}│\n"
        f"  │ Official Winner: {winner_str:<57}│\n"
        f"  │ Realized PnL:    {pnl_str:<57}│\n"
        f"  └{'┄' * 76}┘\n\n"
    )

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
