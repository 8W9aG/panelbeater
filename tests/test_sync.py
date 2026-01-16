import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from alpaca.trading.enums import OrderSide, TimeInForce
from panelbeater.sync import sync_positions  # Adjust import based on your file structure

@pytest.fixture
def mock_alpaca_client(mocker):
    """Fixture to mock the Alpaca TradingClient and its methods."""
    # Patch the TradingClient class where it is instantiated in sync.py
    mock_client_cls = mocker.patch("panelbeater.sync.TradingClient")
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Mock Account
    mock_account = MagicMock()
    mock_account.buying_power = "100000.00"
    mock_client.get_account.return_value = mock_account
    
    # Mock OS Environment variables
    mocker.patch.dict("os.environ", {
        "ALPACA_API_KEY": "fake_key",
        "ALPACA_SECRET_KEY": "fake_secret"
    })
    
    return mock_client

def test_kelly_scaling_logic(mock_alpaca_client):
    """Verifies that buying power is distributed proportionally to Kelly fractions."""
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'kelly_fraction': [1.0, 1.0],  # Total = 2.0
        'type': ['spot_long', 'spot_long'],
        'ask': [150.0, 300.0],
        'tp_target': [160.0, 310.0],
        'sl_target': [140.0, 290.0]
    })
    
    mock_alpaca_client.get_all_positions.return_value = []
    
    sync_positions(df)
    
    # args is a tuple of positional arguments: (MarketOrderRequest, )
    # kwargs is a dict of keyword arguments: {}
    args, kwargs = mock_alpaca_client.submit_order.call_args_list[0]
    
    # Correct the assertion to use the positional argument
    entry_order = args[0]
    assert entry_order.symbol == 'AAPL'
    
    # Check the quantity: (47500 / 150) = 316.66... rounded to 317
    assert entry_order.qty == 317.0

def test_crypto_normalization_and_notional(mock_alpaca_client):
    """Verifies BTC-USD is normalized to BTC/USD and trades using Notional."""
    df = pd.DataFrame({
        'ticker': ['BTC-USD'],
        'kelly_fraction': [1.0],
        'type': ['spot_long'],
        'ask': [50000.0],
        'tp_target': [60000.0],
        'sl_target': [40000.0]
    })
    
    # Simulate already owning some BTC
    mock_pos = MagicMock()
    mock_pos.symbol = "BTCUSD"
    mock_pos.qty = "0.5"
    mock_pos.current_price = "50000.0"
    mock_pos.asset_class = "crypto"
    mock_alpaca_client.get_all_positions.return_value = [mock_pos]
    
    sync_positions(df)
    
    # FIX: Access the FIRST call (index 0) which is the Market Entry
    # The Stop Loss was index 2, which is why .notional was None
    entry_order_req = mock_alpaca_client.submit_order.call_args_list[0].args[0]
    
    # 1. Verify the symbol was normalized
    assert entry_order_req.symbol == 'BTC/USD'
    
    # 2. Verify Notional was used for the entry
    # Total BP (100k) * Safety (0.95) = 95k. 
    # Current (0.5 * 50k) = 25k. Delta = 70k.
    assert entry_order_req.notional == 70000.0
    
    # 3. Optional: Verify the subsequent orders (Exits)
    tp_order_req = mock_alpaca_client.submit_order.call_args_list[1].args[0]
    sl_order_req = mock_alpaca_client.submit_order.call_args_list[2].args[0]
    
    assert tp_order_req.limit_price == 60000.0
    assert sl_order_req.stop_price == 40000.0

def test_crypto_short_liquidation(mock_alpaca_client):
    """Verifies that a 'short' signal for crypto results in a sell-to-zero."""
    df = pd.DataFrame({
        'ticker': ['ETH-USD'],
        'kelly_fraction': [1.0],
        'type': ['spot_short'], # Signal to short
        'ask': [3000.0],
        'tp_target': [2000.0],
        'sl_target': [4000.0]
    })
    
    mock_pos = MagicMock()
    mock_pos.symbol = "ETHUSD"
    mock_pos.qty = "10.0"
    mock_pos.current_price = "3000.0"
    mock_pos.asset_class = "crypto"
    mock_alpaca_client.get_all_positions.return_value = [mock_pos]
    
    sync_positions(df)

    # FIX: Access the call history list instead of just the last call
    # [0] is the Market Order, [1] is Limit, [2] is Stop
    args, kwargs = mock_alpaca_client.submit_order.call_args_list[0]
    order_req = args[0]

    # 3. Assert on the request object properties
    assert order_req.side == OrderSide.SELL
    # Now notional will correctly be 30000.0
    assert order_req.notional == 30000.0

def test_option_symbol_and_multiplier(mock_alpaca_client):
    """Verifies that option_symbol is used and the 100x multiplier is applied."""
    # We set a target of $10,652 for this specific call option
    # At an ask of $53.26, this should result in exactly 2 contracts
    # Calculation: 10652 / (53.26 * 100) = 2.0
    df = pd.DataFrame({
        'ticker': ['SPY'],
        'option_symbol': ['SPY260115C00640000'],
        'kelly_fraction': [1.0],
        'type': ['call'],
        'ask': [53.26],
        'tp_target': [60.0],
        'sl_target': [40.0]
    })

    # Mock buying power so that 100% Kelly = $10,652 (after 0.95 safety)
    # 10652 / 0.95 = 11212.63
    mock_account = MagicMock()
    mock_account.buying_power = "11212.63"
    mock_alpaca_client.get_account.return_value = mock_account
    
    mock_alpaca_client.get_all_positions.return_value = []

    sync_positions(df)

    # Capture the call to submit_order
    # Based on the sync logic, options use execute_option_strategy (MarketOrder)
    args, _ = mock_alpaca_client.submit_order.call_args_list[0]
    order_req = args[0]

    # ASSERTIONS
    assert order_req.symbol == 'SPY260115C00640000'
    assert order_req.qty == 2.0  # (Target $10,652 / $5,326 per contract)
    assert order_req.side == OrderSide.BUY

def test_option_uses_day_tif(mock_alpaca_client):
    df = pd.DataFrame({
        'ticker': ['SPY'],
        'option_symbol': ['SPY260115C00640000'],
        'kelly_fraction': [1.0],
        'type': ['call'],
        'ask': [53.26],
        'tp_target': [60.0], 'sl_target': [40.0]
    })
    
    sync_positions(df)
    
    # Check that the first order (entry) used DAY
    entry_order = mock_alpaca_client.submit_order.call_args_list[0].args[0]
    assert entry_order.time_in_force == TimeInForce.DAY
