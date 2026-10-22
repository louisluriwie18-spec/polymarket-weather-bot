"""
trader.py — Position sizing, trade simulation, and trade logging.

Live CLOB execution has been removed. This module now:
  - Sizes the stake based on the current bankroll and STAKING_PLAN
  - Simulates trades with a UUID order_id (dry_run prefix)
  - Records all recommendations to trades.db for cooldown tracking

To manually place a trade on Polymarket, use the recommendations printed
to the terminal by main.py. Record the result using view_db.py.
"""
import logging
import uuid
from config import (
    DRY_RUN, DRY_RUN_STARTING_BANKROLL, STAKING_PLAN, MAX_STAKE_FRACTION,
    KELLY_FRACTION, MIN_KELLY_STAKE,
)
from database import log_trade_db, get_open_trades, get_realized_pnl

logger = logging.getLogger(__name__)


def get_kelly_stake(
    bankroll: float,
    edge: float,
    mkt_price: float,
    confidence: float,
) -> float:
    """
    Compute a half-Kelly position size scaled by model confidence.

    Kelly fraction: f = edge / (1 - mkt_price)
      where edge = our_prob - mkt_price (the probability advantage).

    Multiplied by KELLY_FRACTION (0.5 = half-Kelly) and by confidence
    to reduce size when the model is uncertain.

    Result is capped at MAX_STAKE_FRACTION of bankroll and floored at
    MIN_KELLY_STAKE to avoid sub-dollar micro-bets.

    Returns 0.0 if bankroll is below the lowest STAKING_PLAN tier.
    """
    # Minimum bankroll guard — use STAKING_PLAN floor
    below_floor = not any(bankroll >= min_br for min_br, _ in STAKING_PLAN)
    if below_floor:
        return 0.0

    # Kelly fraction
    denom = max(0.001, 1.0 - mkt_price)
    kelly_f = (edge / denom) * KELLY_FRACTION * confidence
    kelly_f = max(0.0, kelly_f)  # can't be negative

    # Apply bankroll fraction and hard cap
    raw_stake = bankroll * kelly_f
    cap = bankroll * MAX_STAKE_FRACTION
    final_stake = min(raw_stake, cap)

    # Apply minimum floor (avoid $0.01 bets)
    if final_stake < MIN_KELLY_STAKE:
        final_stake = 0.0  # skip rather than place micro-bet

    logger.debug(
        "Kelly stake: bankroll=$%.2f  edge=%.1f%%  mkt=%.2f  conf=%.2f  "
        "kelly_f=%.3f  raw=$%.2f  cap=$%.2f  -> final=$%.2f",
        bankroll, edge * 100, mkt_price, confidence,
        kelly_f, raw_stake, cap, final_stake,
    )
    return final_stake


def get_current_stake(bankroll: float) -> float:
    """
    Flat-stake fallback: look up the STAKING_PLAN tier for the given bankroll.
    Used only when edge/confidence data are unavailable.
    """
    base_stake = 0.0
    for min_br, stake in STAKING_PLAN:
        if bankroll >= min_br:
            base_stake = stake

    cap = bankroll * MAX_STAKE_FRACTION
    return min(base_stake, cap)


async def get_bankroll() -> float:
    """
    Return the current available bankroll in pUSD.

    Simulates a realistic bankroll by starting at DRY_RUN_STARTING_BANKROLL
    and deducting the stake of every currently-OPEN trade. This makes all
    downstream staking and position-limit logic behave exactly as it would
    against a real wallet balance.

    Resolved trades (WON/LOST) are already freed capital and are not deducted.
    """
    try:
        open_trades = await get_open_trades()
    except Exception as e:
        logger.error("DB error getting open trades: %s", e)
        open_trades = []

    # Deduct ALL open stakes regardless of dry_run status — any OPEN position
    # represents tied-up capital whether it was recorded as dry or live.
    tied_up = sum(t["stake"] for t in open_trades)
    realized_pnl = await get_realized_pnl()

    available = max(0.0, DRY_RUN_STARTING_BANKROLL + realized_pnl - tied_up)
    logger.info(
        "Bankroll: $%.2f starting + $%.2f realized - $%.2f in %d open position(s) = $%.2f available",
        DRY_RUN_STARTING_BANKROLL, realized_pnl, tied_up, len(open_trades), available,
    )
    return available


async def check_trade_cooldown(market_id: str) -> bool:
    """
    Return True if it is safe to recommend this market.
    Returns False (block) if any trade for this market is currently OPEN in the DB.
    """
    try:
        open_trades = await get_open_trades()
    except Exception as e:
        logger.error("DB error checking cooldown: %s", e)
        return False
        
    for t in open_trades:
        if t["market_id"] == market_id:
            return False  # Duplicate — cooldown active
    return True


async def execute_trade(candidate, stake: float) -> dict | None:
    """
    Simulate a trade recommendation and record it to the local DB.

    No live order is placed. A UUID order_id (prefixed 'dry_') is assigned
    to uniquely identify the recommendation and prevent duplicates on the
    next cycle.

    Returns a result dict on success, None on duplicate or invalid stake.
    """
    if not await check_trade_cooldown(candidate.market_id):
        logger.info(
            "Market %s already OPEN in DB — skipping duplicate recommendation.",
            candidate.market_id,
        )
        return None

    if stake <= 0:
        logger.warning("Stake is zero or negative ($%.4f) — skipping.", stake)
        return None

    order_id = f"dry_{uuid.uuid4().hex}"

    # Persist to DB BEFORE logging the recommendation. If the write fails, no
    # recommendation banner is printed, so the operator is not misled into
    # placing a trade that has no cooldown record.
    try:
        await log_trade(candidate, stake, order_id, "OPEN")
    except Exception as e:
        logger.critical(
            "CRITICAL: Recommendation generated (ID=%s, market=%s) but DB write FAILED: %s. "
            "This market will not be cooldown-blocked next cycle — duplicate risk!",
            order_id, candidate.market_id, e,
        )
        return None

    logger.info(
        "RECOMMENDATION | %s %s | stake=$%.2f pUSD | market=%s",
        candidate.side, candidate.group_title, stake, candidate.market_id,
    )
    return {"status": "success", "stake": stake, "order_id": order_id}


async def log_trade(candidate, stake: float, order_id: str, status: str) -> None:
    """
    Write a trade record to the database.

    Uses candidate.question (the full market question text) — NOT group_title
    (the short temperature label like "25°C"). This ensures the DB shows
    the complete, human-readable market description.

    Re-raises database exceptions so callers can detect write failures.
    """
    await log_trade_db(
        market_id=candidate.market_id,
        condition_id=candidate.condition_id,
        event_title=candidate.event_title,
        question=candidate.question,
        side=candidate.side,
        stake=stake,
        price=candidate.mkt_price,
        our_prob=candidate.our_prob,
        edge=candidate.edge,
        order_id=order_id,
        status=status,
        dry_run=DRY_RUN,  # Use config flag
    )

    try:
        from trade_logger import log_trade_text
        log_trade_text(candidate, stake, order_id, status)
    except Exception as log_err:
        logger.warning("Could not append trade to past_trades.txt: %s", log_err)