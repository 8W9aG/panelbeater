"""Synchronise the account to the latest information."""

# pylint: disable=too-many-locals,broad-exception-caught,too-many-arguments,too-many-positional-arguments,superfluous-parens,line-too-long,too-many-branches,too-many-statements,unused-argument
import os
import time

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (OrderSide, OrderType, QueryOrderStatus,
                                  TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, ReplaceOrderRequest,
                                     StopOrderRequest)

# Minimum change in position (in USD) required to trigger a trade
MIN_TRADE_USD = 50.0
# Safety factor to account for Alpaca's 2% price collar on market orders
SAFETY_FACTOR = 0.95


def sync_positions(df: pd.DataFrame):
    """Sync the portfolio, now with explicit Options and Crypto/Equity handling."""
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    clock = trading_client.get_clock()
    is_market_open = clock.is_open  # type: ignore
    account = trading_client.get_account()
    available_funds = float(account.buying_power) * SAFETY_FACTOR  # type: ignore

    total_conviction = df["kelly_fraction"].sum()
    if total_conviction > 0:
        df["target_usd"] = (df["kelly_fraction"] / total_conviction) * available_funds
    else:
        df["target_usd"] = 0.0

    raw_positions = trading_client.get_all_positions()
    positions = {p.symbol: p for p in raw_positions}  # type: ignore

    for _, row in df.iterrows():
        # --- SYMBOL IDENTIFICATION ---
        is_option = pd.notna(row.get("option_symbol")) and row.get(
            "option_symbol"
        ) != row.get("ticker")
        if is_option and not is_market_open:  # pyright: ignore
            continue

        symbol = row["option_symbol"] if is_option else row["ticker"]  # pyright: ignore

        is_crypto = "-" in symbol or "/" in symbol
        trade_symbol = symbol.replace("-", "/") if is_crypto else symbol  # pyright: ignore

        # 1. Determine Current State
        pos = positions.get(symbol.replace("/", "").replace("-", ""))  # pyright: ignore

        # Use Ask price for calculations to be conservative on buying power
        price = float(pos.current_price) if pos else float(row["ask"])  # type: ignore
        current_qty = float(pos.qty) if pos else 0.0  # type: ignore

        # 2. Calculate Target Quantity (Total desired holding)
        multiplier = 100.0 if is_option else 1.0  # pyright: ignore
        target_qty = row["target_usd"] / (price * multiplier)

        if row["type"] in ["spot_short", "put_short", "call_short"]:
            if is_crypto:
                target_qty = 0.0
            else:
                target_qty = -target_qty

        # 3. Decision Logic (Calculate DELTA)
        current_usd_value = current_qty * price * multiplier

        # Determine the Dollar Difference (used for Min Trade Check)
        if row["type"] == "spot_short" and is_crypto:
            target_usd = 0.0
        else:
            target_usd = row["target_usd"]

        diff_usd = target_usd - current_usd_value

        # --- FIX START: Calculate Quantity Delta ---
        # We need to trade the DIFFERENCE, not the total target
        delta_qty = target_qty - current_qty

        # Rounding: Options must be whole numbers. Equities in this script seem to be treated as whole too.
        # If you are selling (negative delta), abs() handles the magnitude.
        qty_to_trade = abs(round(delta_qty, 0))  # pyright: ignore
        # --- FIX END ---

        # Check thresholds
        if abs(diff_usd) < MIN_TRADE_USD:
            update_exits(
                trade_symbol, row["tp_target"], row["sl_target"], trading_client
            )
            continue

        # 4. Execute
        # Prevent 0 quantity errors for Options/Equities
        if not is_crypto and qty_to_trade == 0:
            print(f"[{symbol}] Delta is too small to trade 1 unit. Skipping.")
            continue

        clear_orders(trade_symbol, trading_client)
        side = OrderSide.BUY if diff_usd > 0 else OrderSide.SELL

        if is_crypto:
            execute_crypto_strategy(
                symbol=trade_symbol,
                trade_notional=abs(
                    diff_usd
                ),  # Crypto uses Notional (USD), so diff_usd is correct here
                total_target_usd=row["target_usd"],
                side=side,
                tp=row["tp_target"],
                sl=row["sl_target"],
                trading_client=trading_client,
            )
        elif is_option:  # pyright: ignore
            # Pass qty_to_trade (the delta) instead of target_qty
            execute_option_strategy(
                trade_symbol,
                qty_to_trade,
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )
        else:
            # Pass qty_to_trade (the delta) instead of target_qty
            execute_equity_strategy(
                trade_symbol,
                qty_to_trade,
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
    """
    Crypto Strategy:
    1. Executes a Market order by USD Amount (Notional).
    2. Checks if position exists (handles full liquidation).
    3. Sets TP/SL based on the exact quantity we now own.
    """
    # 1. Execute the Immediate Trade (Entry or Exit)
    # We use 'notional' (USD) so we don't worry about decimal precision on the trade
    print(f"[{symbol}] Executing Market Order for ${trade_notional}...")
    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                notional=round(trade_notional, 2),
                side=side,
                time_in_force=TimeInForce.GTC,
            )
        )
    except Exception as e:
        print(f"[{symbol}] Execution Failed: {e}")
        return

    # 2. Wait for fill & update
    time.sleep(2.0)

    # 3. Check for Position Existence
    # If we sold everything, this returns 404/Error, and we stop.
    try:
        new_pos = trading_client.get_open_position(symbol)
    except APIError:
        print(f"[{symbol}] Position liquidated (or not found). No exits set.")
        return

    # 4. Set Safety Orders (TP / SL) on the *Total* new quantity
    # Crypto quantities can be very precise (e.g. 0.002341), so we read it directly from 'new_pos'
    abs_qty = abs(float(new_pos.qty))

    # In Crypto, we only hold Long positions usually.
    # If you are Shorting crypto, logic ensures side is BUY to cover.
    exit_side = OrderSide.SELL

    print(f"[{symbol}] Setting Exits for {abs_qty} units...")

    try:
        # Take Profit (Limit Sell)
        if tp > 0:
            trading_client.submit_order(
                LimitOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=exit_side,
                    limit_price=round(tp, 2),
                    time_in_force=TimeInForce.GTC,
                )
            )

        # Stop Loss (Stop Sell)
        if sl > 0:
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
        print(f"[{symbol}] Failed to set exit orders: {e}")


def execute_equity_strategy(symbol, qty, side, tp, sl, trading_client):
    """
    Equity Strategy (Standardized):
    1. Market Order for the delta (Change in shares).
    2. Re-reads total position.
    3. Sets fresh TP/SL for the entire holding.
    """
    action = "BUYING" if side == OrderSide.BUY else "SELLING"
    print(f"[{symbol}] {action} {qty} shares...")

    try:
        # 1. Execute the Trade
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,  # Or GTC
            )
        )

        # 2. Wait for fill
        time.sleep(2.0)

        # 3. Check Position
        try:
            new_pos = trading_client.get_open_position(symbol)
        except APIError:
            print(f"[{symbol}] Position closed. No exits needed.")
            return

        # 4. Set Stops on Total Position
        # We read the total quantity we now own (e.g. 150 shares)
        # and protect the *entire* stack, not just the new batch.
        total_qty = abs(float(new_pos.qty))

        # If we are Long, we Sell to exit.
        exit_side = OrderSide.SELL if float(new_pos.qty) > 0 else OrderSide.BUY

        print(f"[{symbol}] Resetting Exits for total {total_qty} shares...")

        if tp > 0:
            trading_client.submit_order(
                LimitOrderRequest(
                    symbol=symbol,
                    qty=total_qty,
                    side=exit_side,
                    limit_price=round(tp, 2),
                    time_in_force=TimeInForce.GTC,
                )
            )

        if sl > 0:
            trading_client.submit_order(
                StopOrderRequest(
                    symbol=symbol,
                    qty=total_qty,
                    side=exit_side,
                    stop_price=round(sl, 2),
                    time_in_force=TimeInForce.GTC,
                )
            )

    except Exception as e:
        print(f"[{symbol}] Equity Strategy Error: {e}")


def update_exits(symbol, model_tp, model_sl, trading_client):
    """Replaces open exit orders with refined logic for options and threshold sensitivity."""
    # Determine sensitivity: 0.01 for options/low-price, 0.5 for others
    is_option = len(symbol) > 12
    threshold = 0.01 if is_option else 0.25

    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )

    for order in open_orders:
        try:
            # 1. Update Take Profit (Limit Orders)
            if order.type == OrderType.LIMIT and model_tp > 0:
                if abs(float(order.limit_price) - model_tp) > threshold:
                    print(f"[{symbol}] Updating TP to {model_tp}")
                    trading_client.replace_order_by_id(
                        order.id, ReplaceOrderRequest(limit_price=round(model_tp, 2))
                    )

            # 2. Update Stop Loss (Stop or Stop-Limit Orders)
            elif order.type in [OrderType.STOP, OrderType.STOP_LIMIT] and model_sl > 0:
                # ReplaceOrderRequest uses 'stop_price' for both Stop and Stop-Limit types
                if abs(float(order.stop_price) - model_sl) > threshold:
                    print(f"[{symbol}] Updating SL to {model_sl}")
                    trading_client.replace_order_by_id(
                        order.id, ReplaceOrderRequest(stop_price=round(model_sl, 2))
                    )

            # 3. Handle 'Canceled' Signal
            elif model_tp == 0 or model_sl == 0:
                print(f"[{symbol}] Model target is 0. Canceling order {order.id}")
                trading_client.cancel_order_by_id(order.id)

        except Exception as e:
            # Common error: order is already 'pending_replace' or 'filled'
            print(f"Update failed for {symbol} ({order.type}): {e}")


def execute_option_strategy(symbol, qty, side, tp, sl, trading_client):
    """Executes orders for Options, handling both Entries and Exits."""
    action = "BUYING" if side == OrderSide.BUY else "SELLING"
    print(f"[{symbol}] {action} {qty} contracts (Market)...")

    try:
        # Step 1: Execute the Trade (Entry or Exit)
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )

        # Step 2: Wait for Fill
        time.sleep(2.0)

        # Step 3: Check if we still hold the position
        try:
            new_pos = trading_client.get_open_position(symbol)
        except APIError:
            # This happens if we sold everything (Position Not Found)
            print(f"[{symbol}] Position closed successfully. No new exits needed.")
            return

        # Step 4: If we still have shares (Partial Exit or Entry), reset TP/SL
        abs_qty = abs(float(new_pos.qty))

        # Determine exit side (If we are Long, exit is Sell)
        pos_side = OrderSide.SELL if float(new_pos.qty) > 0 else OrderSide.BUY

        print(f"[{symbol}] Resetting TP/SL for remaining {abs_qty} contracts...")

        # Take Profit
        if tp > 0:
            trading_client.submit_order(
                LimitOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=pos_side,
                    limit_price=round(tp, 2),
                    time_in_force=TimeInForce.DAY,
                )
            )

        # Stop Loss
        if sl > 0:
            trading_client.submit_order(
                StopOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=pos_side,
                    stop_price=round(sl, 2),
                    time_in_force=TimeInForce.DAY,
                )
            )

    except APIError as e:
        if e.code == 42210000 and "market hours" in str(e).lower():
            print(f"[{symbol}] Skipped: Options market closed.")
        else:
            print(f"[{symbol}] Alpaca API Error: {e}")
    except Exception as e:
        print(f"[{symbol}] Error: {e}")
