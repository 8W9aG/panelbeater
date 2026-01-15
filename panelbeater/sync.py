"""Synchronise the account to the latest information."""

# pylint: disable=too-many-locals,broad-exception-caught
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (OrderClass, OrderSide, QueryOrderStatus,
                                  TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, MarketOrderRequest,
                                     ReplaceOrderRequest, StopLossRequest,
                                     TakeProfitRequest)

# Minimum change in position (in USD) required to trigger a trade
MIN_TRADE_USD = 50.0


def sync_positions(df):
    """Synchronise positions on alpaca."""
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )

    account = trading_client.get_account()
    total_capital = float(account.buying_power)  # pyright: ignore

    # Scale via Model Conviction
    total_kelly_sum = df["kelly_fraction"].sum()
    df["target_usd"] = (df["kelly_fraction"] / total_kelly_sum) * total_capital

    positions = {p.symbol: p for p in trading_client.get_all_positions()}  # pyright: ignore

    for _, row in df.iterrows():
        symbol = row["ticker"].replace("-", "")
        target_usd = row["target_usd"]

        if symbol in positions:
            current_price = float(positions[symbol].current_price)  # pyright: ignore
            current_qty = float(positions[symbol].qty)  # pyright: ignore
        else:
            current_price = row["ask"]
            current_qty = 0.0

        target_qty = target_usd / current_price
        if row["type"] == "spot_short":
            target_qty = -target_qty

        diff_qty = target_qty - current_qty
        diff_usd = abs(diff_qty * current_price)

        # 1. Conviction Threshold & Bracket Updates
        if diff_usd < MIN_TRADE_USD:
            update_bracket_orders(
                symbol, row["tp_target"], row["sl_target"], trading_client
            )
            continue

        # 2. Cancel Existing Orders for this Symbol
        # Instead of CancelOrdersRequest, we fetch and cancel individually
        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for order in open_orders:
            trading_client.cancel_order_by_id(order.id)  # pyright: ignore

        # 3. Execute Rebalance with Bracket
        side = OrderSide.BUY if diff_qty > 0 else OrderSide.SELL

        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=abs(round(diff_qty, 4)),
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(row["tp_target"], 2)),
            stop_loss=StopLossRequest(stop_price=round(row["sl_target"], 2)),
        )

        try:
            trading_client.submit_order(order_req)
        except Exception as e:
            print(f"Error for {symbol}: {e}")


def update_bracket_orders(symbol, model_tp, model_sl, trading_client):
    """
    Updates existing Take Profit and Stop Loss orders.
    """
    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )

    for order in open_orders:
        try:
            # Update Take Profit (Limit)
            if order.type == "limit" and abs(float(order.limit_price) - model_tp) > 1.0:
                trading_client.replace_order_by_id(
                    order.id, ReplaceOrderRequest(limit_price=round(model_tp, 2))
                )

            # Update Stop Loss (Stop)
            elif order.type == "stop" and abs(float(order.stop_price) - model_sl) > 1.0:
                trading_client.replace_order_by_id(
                    order.id, ReplaceOrderRequest(stop_price=round(model_sl, 2))
                )
        except Exception as e:
            print(f"Failed to replace {order.id}: {e}")
