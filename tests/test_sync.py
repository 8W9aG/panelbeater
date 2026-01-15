import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from alpaca.trading.enums import OrderSide
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

    # 1. Capture the call arguments
    # call_args returns a tuple (args, kwargs)
    args, kwargs = mock_alpaca_client.submit_order.call_args
    
    # 2. Extract the request object from positional arguments
    # Since it was called as submit_order(MarketOrderRequest(...)), it's in args[0]
    order_req = args[0]
    
    # 3. Assert on the request object properties
    assert order_req.side == OrderSide.SELL
    # Since it's crypto in your latest script, you likely use notional
    assert order_req.notional == 30000.0
