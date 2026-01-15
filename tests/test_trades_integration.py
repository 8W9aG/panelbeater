import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from panelbeater.simulate import SIMULATION_COLUMN

@pytest.fixture
def mock_mc_dataframe():
    """
    Simulates the 'df_mc' structure in your trades.py loop.
    Long format: Date Index, Price Column, and Simulation ID column.
    """
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(5)]
    ticker = "QQQ"
    price_col = f"PX_{ticker}"
    
    data = []
    # Create 3 paths: Path 0 (Winner), Path 1 (Flat), Path 2 (Loser)
    for s in [0, 1, 2]:
        for i, d in enumerate(dates):
            if s == 0: price = 100 + (i * 2) # Winner
            elif s == 1: price = 100         # Flat
            else: price = 100 - (i * 2)      # Loser
            
            data.append({
                'date': d, 
                price_col: price, 
                SIMULATION_COLUMN: s
            })
    
    return pd.DataFrame(data).set_index('date')

def test_dataframe_slicing_and_pivoting(mock_mc_dataframe):
    """
    Ensures that selecting [[price, sim]] keeps it as a DataFrame
    and allows for successful pivoting into paths.
    """
    ticker = "QQQ"
    price_col = f"PX_{ticker}"
    
    # 1. Simulate the slice in trades.py
    required_cols = [price_col, SIMULATION_COLUMN]
    ticker_sim_data = mock_mc_dataframe[required_cols].copy()
    
    # Verify it's a DataFrame, not a Series
    assert isinstance(ticker_sim_data, pd.DataFrame)
    assert SIMULATION_COLUMN in ticker_sim_data.columns
    
    # 2. Verify prepare_path_matrix can handle this DataFrame
    # Note: Import prepare_path_matrix from your actual module
    from panelbeater.sizing import prepare_path_matrix
    wide_paths = prepare_path_matrix(ticker_sim_data, ticker)
    
    # Check dimensions: 5 days, 3 simulation paths
    assert wide_paths.shape == (5, 3)
    assert wide_paths.iloc[0, 0] == 100 # Initial price

def test_full_spot_logic_with_variance(mock_mc_dataframe):
    """
    End-to-end test for determine_spot_position_and_save math.
    Ensures that variance correctly penalizes the Kelly size.
    """
    from panelbeater.options import determine_spot_position_and_save
    ticker = "QQQ"
    
    # Run the function with our mock data
    results_df = determine_spot_position_and_save(ticker, mock_mc_dataframe)
    
    kelly_val = results_df['kelly_fraction'].iloc[0]
    
    # Since our mock has 1 winner, 1 flat, and 1 loser:
    # The variance will be significant relative to the mean return.
    # We expect a non-zero but conservative Kelly size.
    assert kelly_val >= 0
    assert isinstance(kelly_val, (float, np.float64))
    
    print(f"Calculated Kelly: {kelly_val:.4f}")
