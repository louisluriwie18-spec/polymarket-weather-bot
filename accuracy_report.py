"""
accuracy_report.py — Prediction Accuracy & Market Resolution Audit.

Analyzes resolved trades from trades.db to measure:
  - Brier Score (model vs market accuracy benchmark)
  - Calibration (how well predicted probabilities match real outcomes)
  - Win Rate & Realized PnL / ROI
  - Per-city and per-market-type breakdown
  - Live resolution updater (resolves any open trades whose event date has passed)

Usage:
    python accuracy_report.py
    python accuracy_report.py --update   # check Gamma API and update resolved trades first
"""

import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timezone
import httpx

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")


def _get_db():
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


async def update_unresolved_trades():
    """Check Polymarket Gamma API for any OPEN trades and resolve them."""
    conn = _get_db()
    if not conn:
        print("No database found.")
        return

    cur = conn.cursor()
    cur.execute("SELECT id, market_id, side, stake, price FROM trades WHERE status = 'OPEN'")
    open_trades = cur.fetchall()

    if not open_trades:
        print("No OPEN trades found to resolve.")
        conn.close()
        return

    print(f"Checking resolution for {len(open_trades)} OPEN trade(s)...")
    resolved_count = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for t in open_trades:
            trade_id, market_id, side, stake, price = t["id"], t["market_id"], t["side"], t["stake"], t["price"]
            url = f"https://gamma-api.polymarket.com/markets/{market_id}"
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("closed"):
                        resolution = (data.get("resolution") or "").upper()
                        if resolution in ("CANCELLED", "VOID", "INVALID"):
                            cur.execute("UPDATE trades SET status = 'VOID', pnl = 0.0 WHERE id = ?", (trade_id,))
                            resolved_count += 1
                        elif resolution:
                            won = (side.upper() == resolution)
                            status = "WON" if won else "LOST"
                            pnl = (stake / price - stake) if won else -stake
                            cur.execute("UPDATE trades SET status = ?, pnl = ? WHERE id = ?", (status, pnl, trade_id))
                            resolved_count += 1
            except Exception as e:
                print(f"Error checking market {market_id[:12]}: {e}")

    conn.commit()
    conn.close()
    print(f"Resolved {resolved_count} trade(s).\n")


def generate_accuracy_report():
    conn = _get_db()
    if not conn:
        print(f"Database file not found: {DB_FILE}")
        return

    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, market_id, event_title, question, side,
               stake, price, our_probability, edge, status, pnl, dry_run
        FROM trades
        ORDER BY timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No recorded trades found in database.")
        return

    total_trades = len(rows)
    resolved_trades = [r for r in rows if r["status"] in ("WON", "LOST")]
    open_trades = [r for r in rows if r["status"] == "OPEN"]
    void_trades = [r for r in rows if r["status"] == "VOID"]

    print("=" * 76)
    print("  POLYMARKET WEATHER BOT — PREDICTION ACCURACY & PERFORMANCE REPORT")
    print("=" * 76)
    print(f"  Generated at:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Total Trades Logged: {total_trades}")
    print(f"  Currently Open:      {len(open_trades)}")
    print(f"  Resolved Trades:     {len(resolved_trades)}")
    print(f"  Voided Trades:       {len(void_trades)}")
    print("-" * 76)

    if not resolved_trades:
        print("  No resolved trades yet. Accuracy metrics will compute once trades resolve.")
        print("=" * 76)
        return

    # Accuracy Metrics
    won_trades = [r for r in resolved_trades if r["status"] == "WON"]
    win_rate = (len(won_trades) / len(resolved_trades)) * 100.0

    total_stake = sum(r["stake"] for r in resolved_trades)
    total_pnl = sum(r["pnl"] for r in resolved_trades)
    roi = (total_pnl / total_stake * 100.0) if total_stake > 0 else 0.0

    # Brier Scores
    # Lower is better: 0.0 = perfect prediction, 0.25 = uninformative 50/50, 1.0 = completely wrong
    model_squared_errors = []
    market_squared_errors = []

    for r in resolved_trades:
        actual = 1.0 if r["status"] == "WON" else 0.0
        # Our predicted probability of winning this bet
        our_p = float(r["our_probability"])
        # Market implied probability of winning this bet
        mkt_p = float(r["price"])

        model_squared_errors.append((our_p - actual) ** 2)
        market_squared_errors.append((mkt_p - actual) ** 2)

    model_brier = sum(model_squared_errors) / len(model_squared_errors)
    market_brier = sum(market_squared_errors) / len(market_squared_errors)
    brier_skill_score = 1.0 - (model_brier / market_brier) if market_brier > 0 else 0.0

    print(f"  Win Rate:               {win_rate:.1f}% ({len(won_trades)}/{len(resolved_trades)})")
    print(f"  Total Capital Staked:   ${total_stake:.2f} pUSD")
    print(f"  Realized PnL:           ${total_pnl:+.2f} pUSD (ROI: {roi:+.1f}%)")
    print()
    print("  Statistical Accuracy Benchmarks:")
    print(f"    • Model Brier Score:  {model_brier:.4f}  (Lower is better; 0 = perfect)")
    print(f"    • Market Brier Score: {market_brier:.4f}")
    if brier_skill_score > 0:
        print(f"    • Brier Skill Score:  +{brier_skill_score * 100:.1f}% (Model OUTPERFORMED market prices! ⭐)")
    else:
        print(f"    • Brier Skill Score:  {brier_skill_score * 100:.1f}% (Market was closer than model)")
    print("-" * 76)

    # Calibration Buckets
    print("  Calibration Analysis (Model Prob vs Realized Win Rate):")
    buckets = [
        ("0% - 20%", 0.0, 0.2),
        ("20% - 40%", 0.2, 0.4),
        ("40% - 60%", 0.4, 0.6),
        ("60% - 80%", 0.6, 0.8),
        ("80% - 100%", 0.8, 1.0),
    ]
    for label, lo, hi in buckets:
        in_bucket = [r for r in resolved_trades if lo <= float(r["our_probability"]) < hi or (hi == 1.0 and float(r["our_probability"]) == 1.0)]
        if in_bucket:
            b_wins = sum(1 for r in in_bucket if r["status"] == "WON")
            b_rate = (b_wins / len(in_bucket)) * 100.0
            avg_pred = (sum(float(r["our_probability"]) for r in in_bucket) / len(in_bucket)) * 100.0
            print(f"    [{label:^11}] Count: {len(in_bucket):2d} | Avg Predicted: {avg_pred:5.1f}% | Actual Win: {b_rate:5.1f}%")
        else:
            print(f"    [{label:^11}] Count:  0 | (no samples)")
    print("-" * 76)

    # City Breakdown
    print("  Performance by City:")
    city_trades = {}
    for r in resolved_trades:
        # Extract city from event_title
        title = r["event_title"] or ""
        city = title.split(" in ")[-1].split(" on ")[0].strip() if " in " in title else "Other"
        city_trades.setdefault(city, []).append(r)

    for city, c_rows in sorted(city_trades.items(), key=lambda x: len(x[1]), reverse=True):
        c_wins = sum(1 for r in c_rows if r["status"] == "WON")
        c_pnl = sum(r["pnl"] for r in c_rows)
        c_rate = (c_wins / len(c_rows)) * 100.0
        print(f"    • {city:<16} | Trades: {len(c_rows):2d} | Win: {c_rate:5.1f}% | PnL: ${c_pnl:+.2f}")
    print("=" * 76)


if __name__ == "__main__":
    if "--update" in sys.argv:
        asyncio.run(update_unresolved_trades())
    generate_accuracy_report()
