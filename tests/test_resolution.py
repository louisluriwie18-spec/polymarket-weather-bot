import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from data import get_market_resolution


@pytest.mark.asyncio
async def test_resolution_tier1_explicit():
    # Tier 1: direct resolution field
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "closed": True,
        "resolution": "YES",
    }
    with patch("data._get_with_retry", new=AsyncMock(return_value=mock_resp)):
        res = await get_market_resolution("12345")
        assert res == {"closed": True, "winner": "YES"}


@pytest.mark.asyncio
async def test_resolution_tier1_voided():
    # Tier 1: voided / cancelled market
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "closed": True,
        "resolution": "CANCELLED",
    }
    with patch("data._get_with_retry", new=AsyncMock(return_value=mock_resp)):
        res = await get_market_resolution("12345")
        assert res == {"closed": True, "winner": None, "voided": True}


@pytest.mark.asyncio
async def test_resolution_tier2_outcome_prices():
    # Tier 2: resolution is None, but outcomePrices show winner
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "closed": True,
        "resolution": None,
        "outcomePrices": '["0.99", "0.01"]',
        "outcomes": '["Yes", "No"]',
    }
    with patch("data._get_with_retry", new=AsyncMock(return_value=mock_resp)):
        res = await get_market_resolution("12345")
        assert res == {"closed": True, "winner": "YES"}


@pytest.mark.asyncio
async def test_resolution_tier3_clob_winner():
    # Tier 3: resolution is None, outcomePrices not converged, but CLOB returns winner
    gamma_resp = MagicMock()
    gamma_resp.status_code = 200
    gamma_resp.json.return_value = {
        "closed": True,
        "resolution": None,
        "outcomePrices": '["0.50", "0.50"]',
        "outcomes": '["Yes", "No"]',
        "conditionId": "0xabc123",
    }

    clob_resp = MagicMock()
    clob_resp.status_code = 200
    clob_resp.json.return_value = {
        "tokens": [
            {"outcome": "Yes", "winner": False},
            {"outcome": "No", "winner": True},
        ]
    }

    async def mock_get(client, url):
        if "clob.polymarket.com" in url:
            return clob_resp
        return gamma_resp

    with patch("data._get_with_retry", side_effect=mock_get):
        res = await get_market_resolution("12345")
        assert res == {"closed": True, "winner": "NO"}


@pytest.mark.asyncio
async def test_resolution_unresolved():
    # Market still open
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "closed": False,
    }
    with patch("data._get_with_retry", new=AsyncMock(return_value=mock_resp)):
        res = await get_market_resolution("12345")
        assert res is None
