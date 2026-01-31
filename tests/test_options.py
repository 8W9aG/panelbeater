import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from panelbeater.options import find_mispriced_options_comprehensive

@pytest.fixture
def mock_sim_df():
    """Create a dummy simulation matrix for QQQ."""
    dates = pd.date_range("2026-02-10", periods=1)
    # 1000 paths ending at 625.0
    data = np.full((1, 1000), 625.0)
    return pd.DataFrame(data, index=dates)

@patch("yfinance.Ticker")
def test_liquidity_filter_removes_wide_spreads(mock_ticker_class, mock_sim_df):
    # 1. Setup Mock Ticker
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    
    # Mock current spot price
    mock_ticker.history.return_value = pd.DataFrame({"Close": [620.0]})
    mock_ticker.options = ("2026-02-10",)

    # 2. Create "Fishy" Option Data
    # Row 0: Healthy spread ($0.10)
    # Row 1: Fishy spread ($2.93 - matching your index 1720)
    fishy_data = pd.DataFrame({
        "strike": [610.0, 626.0],
        "bid": [6.00, 8.64],
        "ask": [6.10, 11.57],
        "openInterest": [1000, 1000],
        "volume": [100, 100],
        "impliedVolatility": [0.2, 0.2],
        "contractSymbol": ["QQQ_GOOD", "QQQ_FISHY"]
    })
    
    mock_chain = MagicMock()
    mock_chain.calls = pd.DataFrame() # No calls for this test
    mock_chain.puts = fishy_data
    mock_ticker.option_chain.return_value = mock_chain

    # 3. ACT - Run the function
    # Note: We expect to see 0 rows if our spread filter works, 
    # or 1 row (the good one) if the filter is correctly implemented.
    result_df = find_mispriced_options_comprehensive("QQQ", mock_sim_df)

    # 4. ASSERT
    # This will fail now because your current code only filters OI > 50 and Volume > 5
    assert result_df is not None, "Should have found at least the liquid option"
    assert "QQQ_FISHY" not in result_df["option_symbol"].values, \
        "The wide-spread 'fishy' option should have been filtered out."
    assert "QQQ_GOOD" in result_df["option_symbol"].values
