"""
view_db.py — Quick CLI viewer for the trades database.

Usage:
    python view_db.py

No extra dependencies — uses stdlib sqlite3 only.
"""
import sqlite3
import os

# Resolve DB path relative to this file
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")


def _print_table(cursor, title: str):
    """Print query results as a formatted text table."""
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description] if cursor.description else []
    if not rows:
        print("  (no records)\n")
        return
    if not cols:
        return  # Guard: no column descriptors — avoid IndexError in width loop
    # Compute column widths
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell) if cell is not None else ""))
    # Print header
    header = "  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep    = "  " + "-+-".join("-" * w for w in widths)
    print(f"\n  {title}")
    print("  " + "=" * (len(header) - 2))
    print(header)
    print(sep)
    for row in rows:
        line = "  " + " | ".join(
            str(cell if cell is not None else "").ljust(widths[i])
            for i, cell in enumerate(row)
        )
        print(line)
    print()


def main():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row

            cur = conn.execute(
                "SELECT timestamp, event_title, side, stake, price, "
                "our_probability, edge, status, pnl, dry_run "
                "FROM trades ORDER BY timestamp DESC"
            )
            _print_table(cur, "WEATHER BOT TRADES")

            # Summary
            cur2 = conn.execute(
                "SELECT status, COUNT(*) as count, "
                "ROUND(SUM(pnl), 2) as total_pnl "
                "FROM trades GROUP BY status"
            )
            _print_table(cur2, "SUMMARY BY STATUS")

    except sqlite3.OperationalError:
        print(f"\nDatabase file not found: {DB_FILE}")
        print("It will be created automatically when the bot runs.\n")
    except Exception as e:
        print(f"Error reading database: {e}")


if __name__ == "__main__":
    main()