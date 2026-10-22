import os
import tempfile
import pytest
from unittest.mock import MagicMock
import trade_logger


def test_trade_logger_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "test_past_trades.txt")
        trade_logger.LOG_FILE = tmp_file

        mock_candidate = MagicMock()
        mock_candidate.market_id = "999888"
        mock_candidate.condition_id = "0xcondition123"
        mock_candidate.event_title = "Will the highest temperature in Miami be 80°F or higher on Oct 23?"
        mock_candidate.question = "Will the highest temperature in Miami be 80°F or higher?"
        mock_candidate.side = "YES"
        mock_candidate.our_prob = 0.754
        mock_candidate.mkt_price = 0.420
        mock_candidate.edge = 0.334

        trade_logger.log_trade_text(mock_candidate, stake=2.50, order_id="dry_test_123", status="OPEN")

        assert os.path.exists(tmp_file)
        with open(tmp_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "PAST TRADES AUDIT LOG" in content
        assert "999888" in content
        assert "0xcondition123" in content
        assert "dry_test_123" in content
        assert "75.4%" in content
        assert "$ 2.50 pUSD" in content

        # Now test resolution logging
        trade_logger.log_resolution_text("999888", "WON", pnl=3.45, winner="YES")
        with open(tmp_file, "r", encoding="utf-8") as f:
            res_content = f.read()

        assert "RESOLUTION EVENT" in res_content
        assert "WON" in res_content
        assert "+$3.45 pUSD" in res_content
        assert "YES" in res_content
