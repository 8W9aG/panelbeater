"""Synchronise the account to the latest information."""

# pylint: disable=too-many-locals,broad-exception-caught,too-many-arguments,too-many-positional-arguments,superfluous-parens
import os
import time

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (OrderClass, OrderSide, QueryOrderStatus,
                                  TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, ReplaceOrderRequest,
                                     StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)

# Minimum change in position (in USD) required to trigger a trade
MIN_TRADE_USD = 50.0


def sync_positions(df: pd.DataFrame):
    """
    Main entry point to sync a model DataFrame with Alpaca.
    Handles scaling, equity vs crypto logic, and bracket management.
    """
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    # 1. Determine Capital Allocation
    account = trading_client.get_account()
    # Using buying_power ensures we stay within Alpaca's limits
    available_funds = float(account.buying_power)  # type: ignore

    total_conviction = df["kelly_fraction"].sum()
    # Proportionally distribute available funds based on Kelly weights
    df["target_usd"] = (df["kelly_fraction"] / total_conviction) * available_funds

    # 2. Get Current State
    positions = {p.symbol: p for p in trading_client.get_all_positions()}  # type: ignore

    for _, row in df.iterrows():
        # Identify asset type (e.g., ETH-USD is crypto, AAPL is equity)
        ticker_raw = row["ticker"]
        is_crypto = "-" in ticker_raw or "/" in ticker_raw
        symbol = ticker_raw.replace("-", "")  # pyright: ignore

        # Get Current Price and Quantity
        if symbol in positions:
            price = float(positions[symbol].current_price)  # type: ignore
            current_qty = float(positions[symbol].qty)  # type: ignore
        else:
            # Fallback to model's price if not currently held
            price = float(row["ask"])
            current_qty = 0.0

        # Calculate Delta
        target_qty = row["target_usd"] / price
        if row["type"] == "spot_short":
            target_qty = -target_qty

        diff_qty = target_qty - current_qty
        diff_usd = abs(diff_qty * price)

        # 3. Conviction Threshold Check
        if diff_usd < MIN_TRADE_USD:
            print(
                f"[{symbol}] Change (${diff_usd:.2f}) below threshold. Updating TP/SL only."
            )
            update_exits(symbol, row["tp_target"], row["sl_target"], trading_client)
            continue

        # 4. Clear Old Orders
        # We must cancel existing exit orders before changing position size
        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])  # pyright: ignore
        )
        for order in open_orders:
            trading_client.cancel_order_by_id(order.id)  # type: ignore

        # 5. Execute Strategy
        side = OrderSide.BUY if diff_qty > 0 else OrderSide.SELL
        trade_qty = abs(round(diff_qty, 4 if is_crypto else 0))  # pyright: ignore

        if is_crypto:
            execute_crypto_strategy(
                symbol,
                trade_qty,
                target_qty,
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )
        else:
            execute_equity_strategy(
                symbol,
                trade_qty,
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )


def execute_equity_strategy(symbol, qty, side, tp, sl, trading_client):
    """Uses advanced Bracket Orders (OTOCO) with validation fix."""

    # Validation Check: Ensure TP and SL are on the correct side of each other
    # for the given order side.
    if side == OrderSide.BUY:
        if not (tp > sl):
            print(f"Error: For LONG {symbol}, TP ({tp}) must be > SL ({sl}). Skipping.")
            return
    elif side == OrderSide.SELL:
        if not (tp < sl):
            print(
                f"Error: For SHORT {symbol}, TP ({tp}) must be < SL ({sl}). Skipping."
            )
            return

    order_req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
        stop_loss=StopLossRequest(stop_price=round(sl, 2)),
    )

    try:
        trading_client.submit_order(order_req)
    except Exception as e:
        print(f"Bracket order failed for {symbol}: {e}")


def execute_crypto_strategy(
    symbol, trade_qty, total_target_qty, side, tp, sl, trading_client
):
    """Executes sequential orders because Crypto doesn't support Bracket Orders."""
    print(f"[{symbol}] Executing Crypto Sequential Orders...")
    try:
        # Step 1: Market Order
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol, qty=trade_qty, side=side, time_in_force=TimeInForce.GTC
            )
        )

        # Step 2: Brief pause to allow for execution
        time.sleep(1.5)

        # Step 3: Set independent TP/SL based on the NEW total position
        exit_side = OrderSide.SELL if total_target_qty > 0 else OrderSide.BUY
        abs_target_qty = abs(round(total_target_qty, 4))

        # Take Profit
        trading_client.submit_order(
            LimitOrderRequest(
                symbol=symbol,
                qty=abs_target_qty,
                side=exit_side,
                limit_price=round(tp, 2),
                time_in_force=TimeInForce.GTC,
            )
        )
        # Stop Loss
        trading_client.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=abs_target_qty,
                side=exit_side,
                stop_price=round(sl, 2),
                time_in_force=TimeInForce.GTC,
            )
        )
    except Exception as e:
        print(f"Crypto execution error: {e}")


def update_exits(symbol, model_tp, model_sl, trading_client):
    """Replaces open exit orders with updated model targets."""
    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )
    for order in open_orders:
        try:
            if order.type == "limit" and abs(float(order.limit_price) - model_tp) > 0.5:
                trading_client.replace_order_by_id(
                    order.id, ReplaceOrderRequest(limit_price=round(model_tp, 2))
                )
            elif order.type == "stop" and abs(float(order.stop_price) - model_sl) > 0.5:
                trading_client.replace_order_by_id(
                    order.id, ReplaceOrderRequest(stop_price=round(model_sl, 2))
                )
        except Exception as e:
            print(f"Update failed for {symbol}: {e}")
