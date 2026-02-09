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

    # --- FIX 1: Normalize keys for reliable lookup ---
    # We strip slashes and dashes from the Alpaca symbol so it matches your logic below.
    # We store both the position object AND the original raw symbol for later use.
    positions = {
        p.symbol.replace("/", "").replace("-", ""): p  # type: ignore
        for p in raw_positions  # type: ignore
    }

    # Track which symbols we have processed to identify "Zombies" later
    processed_positions = set()

    for _, row in df.iterrows():
        # --- SYMBOL IDENTIFICATION ---
        is_option = pd.notna(row.get("option_symbol")) and row.get(
            "option_symbol"
        ) != row.get("ticker")
        if is_option and not is_market_open:  # pyright: ignore
            continue

        # --- FIX: Clean suffix and identify type ---
        raw_symbol = row["option_symbol"] if is_option else row["ticker"]  # pyright: ignore
        symbol = raw_symbol.replace("/SPOT", "").replace("-SPOT", "")  # pyright: ignore

        # Create the lookup key (stripped)
        lookup_key = symbol.replace("/", "").replace("-", "")

        # Mark this symbol as processed so we don't liquidate it later
        processed_positions.add(lookup_key)

        is_crypto = "-" in symbol or "/" in symbol
        trade_symbol = symbol.replace("-", "/") if is_crypto else symbol

        # 1. Determine Current State
        # Now this lookup works because both keys are stripped
        pos = positions.get(lookup_key)

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

        # 3. Decision Logic
        current_usd_value = current_qty * price * multiplier

        if row["type"] == "spot_short" and is_crypto:
            target_usd = 0.0
        else:
            target_usd = row["target_usd"]

        diff_usd = target_usd - current_usd_value
        delta_qty = target_qty - current_qty
        qty_to_trade = (
            abs(round(delta_qty, 0)) if is_option or not is_crypto else abs(delta_qty)  # pyright: ignore
        )  # Ensure float for crypto, int for others if needed

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
                trade_notional=abs(diff_usd),
                total_target_usd=row["target_usd"],
                side=side,
                tp=row["tp_target"],
                sl=row["sl_target"],
                trading_client=trading_client,
            )
        elif is_option:  # pyright: ignore
            execute_option_strategy(
                trade_symbol,
                qty_to_trade,
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )
        else:
            execute_equity_strategy(
                trade_symbol,
                qty_to_trade,
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )

    # --- FIX 2: LIQUIDATE LEFTOVERS (Zombies) ---
    for lookup_key, pos in positions.items():
        if lookup_key not in processed_positions:
            print(f"[{pos.symbol}] Not in target portfolio. Liquidating...")  # type: ignore
            try:
                # 1. Cancel open orders for this specific symbol
                # We reuse your existing helper function here for safety/consistency
                clear_orders(pos.symbol, trading_client)  # type: ignore

                # 2. Close the position (Market Order)
                trading_client.close_position(pos.symbol)  # type: ignore
                print(f"[{pos.symbol}] Liquidation order sent.")  # type: ignore
            except Exception as e:
                print(f"[{pos.symbol}] Failed to liquidate: {e}")  # type: ignore


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
    """Crypto Strategy: Market order by USD Amount."""
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

    time.sleep(2.0)

    try:
        new_pos = trading_client.get_open_position(symbol)
    except APIError:
        print(f"[{symbol}] Position liquidated (or not found). No exits set.")
        return

    abs_qty = abs(float(new_pos.qty))
    exit_side = OrderSide.SELL

    print(f"[{symbol}] Setting Exits for {abs_qty} units...")

    try:
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
    """Equity Strategy: Market Order for delta + Exit Reset."""
    action = "BUYING" if side == OrderSide.BUY else "SELLING"
    print(f"[{symbol}] {action} {qty} shares...")

    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )

        time.sleep(2.0)

        try:
            new_pos = trading_client.get_open_position(symbol)
        except APIError:
            print(f"[{symbol}] Position closed. No exits needed.")
            return

        total_qty = abs(float(new_pos.qty))
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
    """Replaces open exit orders."""
    is_option = len(symbol) > 12
    threshold = 0.01 if is_option else 0.25

    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )

    for order in open_orders:
        try:
            if order.type == OrderType.LIMIT and model_tp > 0:
                if abs(float(order.limit_price) - model_tp) > threshold:
                    print(f"[{symbol}] Updating TP to {model_tp}")
                    trading_client.replace_order_by_id(
                        order.id, ReplaceOrderRequest(limit_price=round(model_tp, 2))
                    )

            elif order.type in [OrderType.STOP, OrderType.STOP_LIMIT] and model_sl > 0:
                if abs(float(order.stop_price) - model_sl) > threshold:
                    print(f"[{symbol}] Updating SL to {model_sl}")
                    trading_client.replace_order_by_id(
                        order.id, ReplaceOrderRequest(stop_price=round(model_sl, 2))
                    )

            elif model_tp == 0 or model_sl == 0:
                print(f"[{symbol}] Model target is 0. Canceling order {order.id}")
                trading_client.cancel_order_by_id(order.id)

        except Exception as e:
            print(f"Update failed for {symbol} ({order.type}): {e}")


def execute_option_strategy(symbol, qty, side, tp, sl, trading_client):
    """Executes orders for Options."""
    action = "BUYING" if side == OrderSide.BUY else "SELLING"
    print(f"[{symbol}] {action} {qty} contracts (Market)...")

    try:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )

        time.sleep(2.0)

        try:
            new_pos = trading_client.get_open_position(symbol)
        except APIError:
            print(f"[{symbol}] Position closed successfully. No new exits needed.")
            return

        abs_qty = abs(float(new_pos.qty))
        pos_side = OrderSide.SELL if float(new_pos.qty) > 0 else OrderSide.BUY

        print(f"[{symbol}] Resetting TP/SL for remaining {abs_qty} contracts...")

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
