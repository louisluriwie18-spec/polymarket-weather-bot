"""
database.py — SQLite database for tracking trades and PnL.

Design principles:
  - WRITE functions (log_trade_db) RE-RAISE exceptions.
    A write failure after a simulated trade is placed is a CRITICAL event —
    the recommendation exists but has no local record, so the next cycle's
    cooldown check will miss it and produce a duplicate recommendation.
    Callers MUST catch and handle these exceptions.

  - READ functions (get_open_trades, etc.) SWALLOW exceptions and return
    safe empty defaults. A read failure just means the bot skips a cycle
    or misses a cooldown — bad, but not catastrophic.

  - UPDATE functions (update_trade_resolution) RE-RAISE because a failed
    update leaves trades permanently stuck as OPEN, blocking future
    recommendations on that market forever.
"""
import logging
import os
import aiosqlite
from datetime import datetime, timezone

from config import MAX_TRADES_PER_RUN

logger = logging.getLogger(__name__)

# Resolve DB path relative to this file so it works regardless of CWD
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")


async def init_db():
    """
    Create database tables if they don't exist.

    NOTE: No automatic schema migration. If you add a column, run:
      ALTER TABLE trades ADD COLUMN new_col TYPE DEFAULT value;
    on the existing trades.db before restarting the bot.
    """
    async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
        # Weather bot trade recommendations
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                event_title TEXT NOT NULL,
                question TEXT NOT NULL,
                side TEXT NOT NULL,
                stake REAL NOT NULL,
                price REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge REAL NOT NULL,
                order_id TEXT,
                status TEXT NOT NULL,
                pnl REAL DEFAULT 0.0,
                dry_run INTEGER NOT NULL
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        # Runtime settings (key-value store)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await conn.commit()
    logger.info("Database initialised: %s", DB_FILE)


async def get_setting(key: str) -> str | None:
    """
    Read a runtime setting from bot_settings.
    Returns None if the key doesn't exist (caller uses config default).
    """
    try:
        async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
            async with conn.execute(
                "SELECT value FROM bot_settings WHERE key = ?", (key,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.error("DB error reading setting '%s': %s", key, e)
        return None


async def set_setting(key: str, value: str) -> None:
    """
    Write a runtime setting to bot_settings (upsert).
    RAISES on error — caller must handle.
    """
    ts = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
        await conn.execute(
            """INSERT INTO bot_settings (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE
               SET value = excluded.value,
                   updated_at = excluded.updated_at""",
            (key, value, ts),
        )
        await conn.commit()


async def log_trade_db(
    market_id, event_title, question, side, stake,
    price, our_prob, edge, order_id, status="OPEN", dry_run=False
):
    """
    Insert a trade record into the `trades` table.

    RAISES on any DB error — callers must handle this.
    A write failure means the recommendation has no local record,
    causing duplicate recommendations on the next cycle.
    """
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
        await conn.execute("""
            INSERT INTO trades (
                timestamp, market_id, event_title, question, side,
                stake, price, our_probability, edge, order_id, status, dry_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, market_id, event_title, question, side,
            stake, price, our_prob, edge, order_id, status, int(dry_run)
        ))
        await conn.commit()
    logger.debug("Trade logged: market=%s side=%s stake=%.2f", market_id, side, stake)


async def get_open_trades() -> list[dict]:
    """
    Return all trades with status='OPEN'.

    Returns an empty list on DB error (consistent with the module-level design:
    READ functions swallow exceptions and return safe empty defaults).
    Callers that need to distinguish "no open trades" from "DB error" should
    use get_total_open_count() which returns MAX_TRADES_PER_RUN on error.
    """
    try:
        async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM trades WHERE status='OPEN'") as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.error("DB error reading open trades: %s", e)
        return []


async def get_total_open_count() -> int:
    """
    Return the count of open positions in the trades table.
    Used to enforce the hard position cap.

    Returns MAX_TRADES_PER_RUN on error (safe-fail: blocks new trade
    recommendations when open count cannot be verified, preventing
    over-positioning on a DB hiccup).
    """
    try:
        async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='OPEN'"
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
    except Exception as e:
        logger.error("DB Error getting total open count: %s", e)
        # Safe default: block all new recommendations when the count cannot be verified.
        # Returning 0 here would incorrectly signal "no open positions" and allow
        # the bot to recommend positions even when it cannot confirm the position cap.
        return MAX_TRADES_PER_RUN


async def get_realized_pnl() -> float:
    """
    Return the total PnL of all resolved trades (WON or LOST), regardless of
    dry_run status.  Filtering on dry_run=1 caused live-mode PnL to always
    report $0.00, making the bankroll appear at its starting value after wins/losses.
    """
    try:
        async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
            async with conn.execute(
                "SELECT SUM(pnl) FROM trades WHERE status IN ('WON', 'LOST')"
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row and row[0] else 0.0
    except Exception as e:
        logger.error("DB Error getting realized PnL: %s", e)
        return 0.0


async def update_trade_resolution(market_id: str, status: str, pnl: float):
    """
    Update all OPEN trades for a given market_id to the resolved status and PnL.

    RAISES on DB error — a failed update leaves trades stuck as OPEN permanently,
    blocking future recommendations on that market until manually corrected.
    """
    async with aiosqlite.connect(DB_FILE, timeout=10.0) as conn:
        await conn.execute(
            "UPDATE trades SET status=?, pnl=? WHERE market_id=? AND status='OPEN'",
            (status, pnl, market_id)
        )
        await conn.commit()
    logger.debug("Resolution updated: market=%s status=%s pnl=%.4f", market_id, status, pnl)