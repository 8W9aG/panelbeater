import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from panelbeater.options import find_mispriced_options_comprehensive, apply_volatility_sanity_filter

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

def test_volatility_sanity_filter_rejects_unrealistic_vol():
    # 1. Mock a row with 20% Market IV
    mock_row = pd.Series({"impliedVolatility": 0.20, "contractSymbol": "QQQ_TEST_FILTER"})
    
    # 2. Case A: Model Vol is too high vs Market IV
    is_rational, reason = apply_volatility_sanity_filter(mock_row, 0.40, 0.18)
    assert is_rational is False
    # Use 'IV' instead of 'Market IV' to match your new f-string output
    assert "IV" in reason 
    
    # 3. Case B: Ratio vs SPY is too high
    is_rational, reason = apply_volatility_sanity_filter(mock_row, 0.25, 0.10)
    assert is_rational is False
    # Use 'RATIO' to match 'RATIO_TOO_HIGH'
    assert "RATIO" in reason

@patch("yfinance.Ticker")
def test_smart_liquidity_tiered_logic(mock_ticker_class, mock_sim_df):
    """
    Verifies the 'Smart Liquidity' logic:
    1. ALLOWS wide % spreads if the dollar width is tiny (<= $0.10).
    2. REJECTS wide % spreads if the dollar width is large.
    3. ALLOWS 0 volume if Open Interest is high (morning trading).
    """
    # 1. Setup Mock
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    
    # Mock spot and dates
    mock_ticker.history.return_value = pd.DataFrame({"Close": [100.0]})
    mock_ticker.options = ("2026-02-10",)

    # 2. Create Test Data covering 4 scenarios
    test_data = pd.DataFrame({
        "contractSymbol": [
            "CASE_A_CHEAP_KEEP",   # Bid 0.05/Ask 0.10 -> 50% spread (Fail %), but $0.05 width (Pass $)
            "CASE_B_PRICEY_DROP",  # Bid 10.0/Ask 20.0 -> 50% spread (Fail %), $10.0 width (Fail $)
            "CASE_C_PRICEY_KEEP",  # Bid 10.0/Ask 10.2 -> 2% spread (Pass %), $0.2 width (Fail $)
            "CASE_D_NO_VOL_KEEP"   # Good spread, Volume 0, High OI -> Should Keep
        ],
        "strike": [100, 100, 100, 100],
        # Prices
        "bid": [0.05, 10.00, 10.00, 10.00],
        "ask": [0.10, 20.00, 10.20, 10.20],
        # Liquidity
        "openInterest": [100, 100, 100, 500],
        "volume":       [0,   0,   0,   0], # All have 0 volume to test the "Morning Lull" rule
        "impliedVolatility": [0.2, 0.2, 0.2, 0.2]
    })

    mock_chain = MagicMock()
    mock_chain.calls = test_data
    mock_chain.puts = pd.DataFrame()
    mock_ticker.option_chain.return_value = mock_chain

    # 3. Run
    # (We assume mock_sim_df is provided by your existing fixture)
    result_df = find_mispriced_options_comprehensive("QQQ", mock_sim_df)

    # 4. Assertions
    assert result_df is not None
    symbols = result_df["option_symbol"].values.tolist()

    # Case A: Should be KEPT because width ($0.05) is <= $0.10, despite 50% spread
    assert "CASE_A_CHEAP_KEEP" in symbols, "Failed to keep cheap option with wide % but tight $"

    # Case B: Should be DROPPED because width is large AND spread is 50%
    assert "CASE_B_PRICEY_DROP" not in symbols, "Failed to drop expensive option with wide spread"

    # Case C: Should be KEPT because spread % is low (2%), despite width > $0.10
    assert "CASE_C_PRICEY_KEEP" in symbols, "Failed to keep expensive option with tight spread"

    # Case D: Should be KEPT because OI > 10, even though Volume is 0
    assert "CASE_D_NO_VOL_KEEP" in symbols, "Failed to keep 0-volume option with high OI"
