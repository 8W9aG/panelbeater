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
# Safety factor to account for Alpaca's 2% price collar on market orders
SAFETY_FACTOR = 0.95


def sync_positions(df: pd.DataFrame):
    """Sync the portfolio to alpaca with robust symbol and balance handling."""
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    account = trading_client.get_account()

    # Use 95% of Buying Power to avoid 'Insufficient Balance' due to 2% price collars
    available_funds = float(account.buying_power) * SAFETY_FACTOR  # type: ignore

    total_conviction = df["kelly_fraction"].sum()
    df["target_usd"] = (df["kelly_fraction"] / total_conviction) * available_funds

    # NORMALIZE: Map 'BTCUSD' (position) to 'BTC/USD' (trading pair)
    raw_positions = trading_client.get_all_positions()
    positions = {}
    for p in raw_positions:
        sym = p.symbol  # type: ignore
        if p.asset_class == "crypto" and "/" not in sym:  # type: ignore
            # Convert 'BTCUSD' -> 'BTC/USD' for consistent matching
            sym = sym.replace("USD", "/USD")
        positions[sym] = p

    for _, row in df.iterrows():
        # Standardize ticker to match Alpaca's / format for crypto (e.g. BTC/USD)
        ticker_raw = row["ticker"].replace("-", "/")  # pyright: ignore
        is_crypto = "/" in ticker_raw
        symbol_for_position = ticker_raw.replace("/", "") if is_crypto else ticker_raw

        # 1. Determine Current State
        # Check normalized dict first, then fall back to raw symbol
        pos = positions.get(ticker_raw) or positions.get(symbol_for_position)

        price = float(pos.current_price) if pos else float(row["ask"])  # type: ignore
        current_qty = float(pos.qty) if pos else 0.0  # type: ignore
        current_usd_value = current_qty * price

        # 2. Calculate Target Quantity / Notional
        target_usd = row["target_usd"]

        if row["type"] == "spot_short":
            if is_crypto:
                # Alpaca Crypto is Long-Only; Short signals = Liquidate
                target_usd = 0.0
            else:
                # Equities can be shorted (requires Margin account)
                target_usd = -target_usd

        # 3. Decision Logic based on Dollar Delta
        # Using USD Delta is safer for balance checks than Qty Delta
        diff_usd = target_usd - current_usd_value

        if abs(diff_usd) < MIN_TRADE_USD:
            print(
                f"[{ticker_raw}] Delta ${diff_usd:.2f} too small. Updating exits only."
            )
            update_exits(ticker_raw, row["tp_target"], row["sl_target"], trading_client)
            continue

        # 4. Clear Old Orders & Execute
        clear_orders(ticker_raw, trading_client)
        side = OrderSide.BUY if diff_usd > 0 else OrderSide.SELL

        if is_crypto:
            execute_crypto_strategy(
                ticker_raw,
                abs(diff_usd),
                target_usd,
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )
        else:
            # Equities still use Quantity for non-fractional support
            trade_qty = abs(round(diff_usd / price, 0))  # pyright: ignore
            if trade_qty > 0:
                execute_equity_strategy(
                    ticker_raw,
                    trade_qty,
                    side,
                    row["tp_target"],
                    row["sl_target"],
                    trading_client,
                )


def clear_orders(symbol, trading_client):
    """Cancels all open orders for a symbol to avoid conflicts."""
    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )
    for order in open_orders:
        trading_client.cancel_order_by_id(order.id)


def execute_crypto_strategy(
    symbol, trade_notional, total_target_usd, side, tp, sl, trading_client
):
    """Handles crypto using Notional values to satisfy price collars."""
    try:
        # Use Notional for the entry to let Alpaca handle the collar/buffer
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                notional=round(trade_notional, 2),
                side=side,
                time_in_force=TimeInForce.GTC,
            )
        )

        if total_target_usd <= 0:
            print(f"[{symbol}] Position liquidated. No exits set.")
            return

        time.sleep(2.0)  # Brief pause for order to fill and position to update

        # Get new position to set accurate TP/SL quantities
        new_pos = trading_client.get_open_position(symbol)
        abs_qty = abs(float(new_pos.qty))
        exit_side = OrderSide.SELL if float(new_pos.qty) > 0 else OrderSide.BUY

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
        print(f"Crypto Strategy failed for {symbol}: {e}")


def execute_equity_strategy(symbol, qty, side, tp, sl, trading_client):
    """Uses Bracket Orders for Equities."""
    try:
        # Validation for Alpaca Bracket rules: Buy TP > SL, Sell TP < SL
        if (side == OrderSide.BUY and tp <= sl) or (
            side == OrderSide.SELL and tp >= sl
        ):
            print(f"[{symbol}] TP/SL validation failed. Skipping bracket.")
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
