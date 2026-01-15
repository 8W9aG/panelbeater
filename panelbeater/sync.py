"""Synchronise the account to the latest information."""

# pylint: disable=too-many-locals,broad-exception-caught,too-many-arguments,too-many-positional-arguments,superfluous-parens,line-too-long
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
    """Sync the portfolio to alpaca."""
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    account = trading_client.get_account()
    available_funds = float(account.buying_power)  # type: ignore

    total_conviction = df["kelly_fraction"].sum()
    df["target_usd"] = (df["kelly_fraction"] / total_conviction) * available_funds

    positions = {p.symbol: p for p in trading_client.get_all_positions()}  # type: ignore

    for _, row in df.iterrows():
        ticker_raw = row["ticker"]
        is_crypto = "-" in ticker_raw or "/" in ticker_raw
        symbol = ticker_raw.replace("-", "").replace("/", "")  # pyright: ignore

        # 1. Determine Current State
        price = (
            float(positions[symbol].current_price)  # type: ignore
            if symbol in positions
            else float(row["ask"])
        )
        current_qty = float(positions[symbol].qty) if symbol in positions else 0.0  # type: ignore

        # 2. Calculate Target Quantity with Shorting Overrides
        target_qty = row["target_usd"] / price

        if row["type"] == "spot_short":
            if is_crypto:
                # Alpaca Crypto is Long-Only; Short signals = Liquidate
                target_qty = 0.0
            else:
                # Equities can be shorted (requires Margin account)
                target_qty = -target_qty

        # 3. Precision Rounding & Zero-Quantity Guard
        diff_qty = target_qty - current_qty
        trade_qty = abs(round(diff_qty, 4 if is_crypto else 0))  # pyright: ignore

        # FIX: If trade_qty rounds to 0, or we are trying to 'short' from 0 balance
        if trade_qty <= 0:
            print(
                f"[{symbol}] No significant change needed or shorting from zero. Updating exits only."
            )
            update_exits(symbol, row["tp_target"], row["sl_target"], trading_client)
            continue

        diff_usd = trade_qty * price
        if diff_usd < MIN_TRADE_USD:
            print(
                f"[{symbol}] Change of ${diff_usd:.2f} too small. Updating exits only."
            )
            update_exits(symbol, row["tp_target"], row["sl_target"], trading_client)
            continue

        # 4. Clear Old Orders & Execute
        clear_orders(symbol, trading_client)
        side = OrderSide.BUY if diff_qty > 0 else OrderSide.SELL

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


def clear_orders(symbol, trading_client):
    """Cancels all open orders for a symbol to avoid 'insufficient balance' conflicts."""
    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )
    for order in open_orders:
        trading_client.cancel_order_by_id(order.id)


def execute_crypto_strategy(
    symbol, trade_qty, total_target_qty, side, tp, sl, trading_client
):
    """Handles crypto as sequential orders (Entry -> TP/SL)."""
    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol, qty=trade_qty, side=side, time_in_force=TimeInForce.GTC
            )
        )

        # Skip TP/SL if we just liquidated the position
        if total_target_qty == 0:
            print(f"[{symbol}] Position closed. No exit orders set.")
            return

        time.sleep(1.5)  # Wait for fill
        exit_side = OrderSide.SELL if total_target_qty > 0 else OrderSide.BUY
        abs_qty = abs(round(total_target_qty, 4))

        trading_client.submit_order(
            LimitOrderRequest(
                symbol=symbol,
                qty=abs_qty,
                side=exit_side,
                limit_price=round(tp, 2),
                time_in_force=TimeInForce.GTC,
            )
        )
        trading_client.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=abs_qty,
                side=exit_side,
                stop_price=round(sl, 2),
                time_in_force=TimeInForce.GTC,
            )
        )
    except Exception as e:
        print(f"Crypto Trade failed: {e}")


def execute_equity_strategy(symbol, qty, side, tp, sl, trading_client):
    """Uses Bracket Orders for Equities."""
    try:
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
            stop_loss=StopLossRequest(stop_price=round(sl, 2)),
        )
        trading_client.submit_order(order_req)
    except Exception as e:
        print(f"Equity Trade failed: {e}")


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
