import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from panelbeater.sizing import prepare_path_matrix, calculate_path_aware_mean_variance

@pytest.fixture
def mock_sim_df():
    """Creates a mock long-format DataFrame like Panelbeater output."""
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(10)]
    # Two paths: Path 0 is a winner, Path 1 is a loser
    data = []
    for s in [0, 1]:
        for i, d in enumerate(dates):
            # Simulation 0: Steady gain (+1 per day)
            # Simulation 1: Steady loss (-1 per day)
            price = 100 + (i if s == 0 else -i)
            data.append({'date': d, 'PX_QQQ': price, 'simulation': s})
    
    df = pd.DataFrame(data).set_index('date')
    return df

def test_prepare_path_matrix_dimensions(mock_sim_df):
    """Verify that long data pivots to (Time, Paths) wide data."""
    wide = prepare_path_matrix(mock_sim_df, "QQQ")
    # 10 days, 2 simulation paths
    assert wide.shape == (10, 2)
    assert list(wide.columns) == [0, 1]

def test_kelly_variance_penalty():
    """
    Verify that higher variance results in a smaller Kelly size
    even if the mean return is the same.
    """
    spot = 100
    tp, sl = 110, 90
    
    # Scenario A: Low Variance
    # Path 0 hits 110, Path 1 hits 111. 
    # Returns: [0.10, 0.11] -> Small variance, High Confidence
    path_matrix_low_var = np.array([
        [100, 100],
        [110, 111]
    ])
    kelly_low, mean_low, var_low, _ = calculate_path_aware_mean_variance(
        path_matrix_low_var, spot, True, tp, sl
    )
    
    # Scenario B: High Variance
    # Path 0 hits 110, Path 1 hits 101.
    # Returns: [0.10, 0.01] -> Large variance, Low Confidence
    path_matrix_high_var = np.array([
        [100, 100],
        [110, 101]
    ])
    kelly_high, mean_high, var_high, _ = calculate_path_aware_mean_variance(
        path_matrix_high_var, spot, True, tp, sl
    )
    
    print(f"\nLow Var Scenario: Mean={mean_low:.4f}, Var={var_low:.4f}, Kelly={kelly_low:.2f}")
    print(f"High Var Scenario: Mean={mean_high:.4f}, Var={var_high:.4f}, Kelly={kelly_high:.2f}")

    # Kelly should correctly penalize the higher variance
    assert kelly_low > kelly_high
    assert kelly_low > 0

def test_missing_simulation_column_raises_error():
    """Ensure we catch the KeyError before it crashes the loop."""
    df = pd.DataFrame({'PX_QQQ': [100, 101]})
    with pytest.raises(KeyError, match="simulation"):
        prepare_path_matrix(df, "QQQ")
