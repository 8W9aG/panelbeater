import pandas as pd
import numpy as np
from wavetrainer.model.model import PROBABILITY_COLUMN_PREFIX

from panelbeater.normalizer import normalize, denormalize


def test_normalize_logic():
    print("Running test_normalize_logic...")
    
    # 1. Setup Data
    # We need >365 rows to satisfy the rolling window, 
    # though the 'null' logic technically works without it.
    N_ROWS = 400
    dates = pd.date_range(start="2023-01-01", periods=N_ROWS, freq="D")
    
    # Create synthetic price data
    # Start with a simple range so prices change every day
    prices = np.arange(100, 100 + N_ROWS, dtype=float)
    
    # MANUALLY INJECT SCENARIOS at the end of the data
    # Case 1: Price changes (normal behavior)
    prices[390] = 200.0
    prices[391] = 205.0 # Change happening here
    
    # Case 2: Price stays exactly the same (Zero pct_change)
    prices[392] = 210.0
    prices[393] = 210.0 # No change happening here (pct_change == 0.0)
    
    df = pd.DataFrame({'close': prices}, index=dates)

    # 2. Run Function
    result = normalize(df)

    # 3. Verifications
    
    # Check A: Does the column exist?
    expected_col = 'close_null'
    assert expected_col in result.columns, \
        f"Test Failed: '{expected_col}' column was not found in output."

    # Check B: Verify 'True' condition (No change)
    # At index 393, price was 210, previous was 210. Change is 0.
    is_null_detected = result.iloc[393][expected_col]
    assert is_null_detected == True, \
        f"Test Failed: Row 393 should be True (No change), got {is_null_detected}"

    # Check C: Verify 'False' condition (Normal change)
    # At index 391, price was 205, previous was 200. Change occurred.
    is_change_detected = result.iloc[391][expected_col]
    assert is_change_detected == False, \
        f"Test Failed: Row 391 should be False (Price changed), got {is_change_detected}"

    print("SUCCESS: '{col}_null' column logic verified.")
    print(f"Sample Output at target rows:\n{result[expected_col].iloc[390:].to_string()}")

def test_denormalize_null_logic():
    print("Running test_denormalize_null_logic...")

    # 1. Setup Historical Data (y)
    # We need enough history for rolling(365) to produce a non-NaN value, 
    # or we rely on the .fillna(0.0) in the function.
    dates = pd.date_range(start="2020-01-01", periods=400, freq="D")
    price_data = np.linspace(100, 200, 400) # Simple trend
    y = pd.DataFrame({'BTC_USD': price_data}, index=dates)

    # Calculate Last Price for verification
    last_price = y['BTC_USD'].iloc[-1]

    # 2. Setup Prediction Data (df)
    # This df usually contains the probabilities output by the model.
    # We need to construct columns that match the naming convention:
    # {col}_{bucket}_{PROBABILITY_COLUMN_PREFIX}
    
    # We create a single row DataFrame for the "next day" prediction
    df_idx = dates
    df = pd.DataFrame(index=df_idx)
    
    # --- SCENARIO A: Force "No Change" (Null) ---
    # Bucket 0.0 has 100% probability -> highest_std will be 0.0
    # Null column has 100% probability -> value should be forced to 0.0
    
    df[f'BTC_USD_0.0_{PROBABILITY_COLUMN_PREFIX}'] = 1.0
    df[f'BTC_USD_1.0_{PROBABILITY_COLUMN_PREFIX}'] = 0.0 # Dummy other bucket
    df[f'BTC_USD_null_{PROBABILITY_COLUMN_PREFIX}'] = 1.0 # FORCE NULL
    
    # Run Denormalize
    # u_sample=None allows random choice, but our probs are 1.0/0.0 so it's deterministic
    result_null = denormalize(df.copy(), y, u_sample=None)
    
    # Check A: Did we get a new row?
    new_date = dates[-1] + pd.Timedelta(days=1)
    assert new_date in result_null.index
    
    # Check B: Is price EXACTLY the same?
    new_price_null = result_null.loc[new_date, 'BTC_USD']
    assert new_price_null == last_price, \
        f"Scenario A Failed: Price should be equal ({last_price}), got {new_price_null}"
    print("Scenario A (Force Null) Passed: Price remained constant.")

    # --- SCENARIO B: Force "Jitter" (Bucket 0.0 but Null Prob is 0) ---
    # Bucket 0.0 has 100% probability -> highest_std will be 0.0
    # Null column has 0% probability -> value should jitter
    
    df_jitter = pd.DataFrame(index=df_idx)
    df_jitter[f'BTC_USD_0.0_{PROBABILITY_COLUMN_PREFIX}'] = 1.0
    df_jitter[f'BTC_USD_1.0_{PROBABILITY_COLUMN_PREFIX}'] = 0.0
    df_jitter[f'BTC_USD_null_{PROBABILITY_COLUMN_PREFIX}'] = 0.0 # FORCE JITTER
    
    # Run Denormalize
    result_jitter = denormalize(df_jitter.copy(), y, u_sample=None)
    new_price_jitter = result_jitter.loc[new_date, 'BTC_USD']
    
    # Check C: Is price DIFFERENT?
    # Because std/mu are calculated from 'y' which has movement, sigma should be > 0
    # Therefore jitter should result in a different price.
    assert new_price_jitter != last_price, \
        f"Scenario B Failed: Price should have jittered, but remained {last_price}"
        
    print(f"Scenario B (Jitter) Passed: Price changed from {last_price} to {new_price_jitter}")
